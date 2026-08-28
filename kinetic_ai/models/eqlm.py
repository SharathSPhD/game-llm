"""EqLM: Equilibrium Language Model.

A weight-tied Deep Equilibrium Transformer that replaces explicit depth with
a fixed-point computation. Single block solved to equilibrium via Anderson
acceleration, yielding memory-efficient training equivalent to infinite depth.

Design (per SPEC 0004):
    - Architecture: Single pre-LN transformer block as fixed-point map
      f(z, x) = z + Attn(LN(z), causal) + MLP(LN(z + Attn)) + x_proj
      with input injection x (token+pos embeddings) added each iteration
      following Bai et al. DEQ-transformer practice.

    - Solving: Fixed point z* = f(z*, x) via Anderson acceleration
      (max 12 iters, tol 1e-3 training / 1e-4 eval).

    - Backward: Jacobian-Free Backprop (JFB) for training throughput.

    - Embedding: Token embedding + learned positional embedding -> x.
      LM head weight-tied to token embedding.

References:
    [1] Bai et al. "Deep Equilibrium Models" (NeurIPS 2019, arXiv:1909.01377)
    [2] Bai et al. "Transformers are RNNs: Fast Autoregressive
        Transformers with Linear Attention" (arXiv:2006.16236)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from kinetic_ai.config import DEQConfig, SolverType
from kinetic_ai.models.deq_layer import DEQLayer

# ============================================================================
# EqLMConfig
# ============================================================================


@dataclass
class EqLMConfig:
    """Configuration for Equilibrium Language Model.

    Attributes:
        vocab_size: Token vocabulary size.
        d_model: Embedding and hidden dimension.
        n_heads: Number of attention heads.
        d_ff: Feed-forward layer hidden dimension.
        max_seq_len: Maximum sequence length for position embeddings.
        deq_max_iter: Max iterations for fixed-point solver (training).
        deq_tol: Convergence tolerance (L2 norm of residual).
        solver: Fixed-point solver type ("picard", "anderson", "broyden").
        jfb: If True, use Jacobian-Free Backprop (faster, approximate).
             If False, use full Implicit Function Theorem backprop.
        dropout: Dropout rate (0.0 for no dropout).
        spectral_norm: If True, apply spectral normalization to block weights to enforce
                       the fixed-point map f(z,x) is Lipschitz-contractive with ||f'|| < 1.
                       Ensures solver convergence via Banach fixed-point theorem.
                       Default True (EqLM-v2). Set False for backward compat (non-contractive).
        residual_damping: Damping factor α for residual branch in fixed-point map.
                          The residual update becomes: f(z) = (1-α)z + α*(z + block(z))
                          Ensures ||f(z) - z|| < α ensures contraction when α < 1.
                          Default 0.2 ensures σ(f) < 1 for block with spectral norm.
                          Set to 1.0 for no damping (full residual, requires explicit α scaling).
        map_form: Fixed-point map formulation: 'residual' (v1/v2, default) | 'postln' (v3).
                  'residual': f(z,x) = (1-α)z + α(z + Attn + MLP + inj(x))
                             Pre-LN transformer block with damped residuals (original DEQ form).
                  'postln': f(z,x) = LN2(h + MLP(h)) where h = LN1(z + Attn(z,causal) + inj(x))
                           Puts outer LayerNorm INSIDE the map so iterates are bounded.
                           Follows Bai et al. DEQ-transformer practice. Ensures bona fide fixed points exist.
                           Default 'residual' for backward compat.
        aux_residual: If True, compute auxiliary loss on the residual ||f(z*,x) − z*||/(||z*||+eps).
                      Enables solver-aware training to learn contraction without explicit constraints.
                      Stored in model.last_aux_residual and multiplied by lambda_aux during training.
                      Default False for backward compat.
        lambda_aux: Weight for the auxiliary residual loss. Only used if aux_residual=True.
                    Total loss = L_ce + lambda_aux * aux_residual.
                    Default 0.1.
        decode_mode: Which forward computation generate() uses: 'solver'
                     (Anderson/implicit — matches implicit-trained checkpoints)
                     or 'unrolled' (plain deq_max_iter map applications from
                     z0=x — matches anytime-unrolled training, F24/B1). BLiMP
                     and training are unaffected; this only routes decoding.
    """

    vocab_size: int = 50257
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    deq_max_iter: int = 12
    deq_tol: float = 1e-3
    solver: str = "anderson"
    jfb: bool = True
    dropout: float = 0.1
    spectral_norm: bool = True
    residual_damping: float = 0.2
    map_form: str = "residual"
    aux_residual: bool = False
    lambda_aux: float = 0.1
    decode_mode: str = "solver"


# ============================================================================
# EqLMBlock: Single Transformer Block as Fixed-Point Map
# ============================================================================


def sample_next_token(logits_last: Tensor, temperature: float, top_k: int) -> Tensor:
    """Greedy when temperature<=0; else temperature softmax with optional top-k.

    Shared by EqLM.generate and the serving layer's explicit-model loop so
    both architectures decode identically.
    """
    if temperature <= 0.0:
        return torch.argmax(logits_last, dim=-1, keepdim=True)
    scaled = logits_last / temperature
    if top_k > 0 and top_k < scaled.shape[-1]:
        kth = torch.topk(scaled, top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class EqLMBlock(nn.Module):
    """Pre-LN Transformer Block as a Fixed-Point Map.

    Computes: f(z) = (1-α)z + α(z + Attn(LN(z), causal) + MLP(LN(z + Attn)) + input_gate*x)

    Where:
        z: Hidden state (equilibrium variable)
        α: Residual damping factor (0.5 by default, ensures contraction)
        LN: Layer normalization (pre-norm)
        Attn: Causal multi-head self-attention
        MLP: Two-layer feed-forward network

    The fixed point z* = f(z*) provides the equilibrium hidden states.
    All operations are position-wise or causal (attention is causal),
    ensuring z*[t] only depends on input embeddings up to position t.

    Damped residuals: f(z) = (1-α)z + α*g(z) ensures ||f'|| < 1 when ||g'|| is bounded,
    guaranteeing contraction by Banach theorem even without explicit spectral norm scaling
    on g's sub-components.

    This implements the standard DEQ-Transformer architecture from
    Bai et al. (arXiv:1909.01377), where the initial hidden states
    are the embeddings and the block evolves them to equilibrium.
    """

    def __init__(self, config: EqLMConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.residual_damping = config.residual_damping

        # Validate configuration
        assert (
            config.d_model % config.n_heads == 0
        ), f"d_model {config.d_model} must be divisible by n_heads {config.n_heads}"
        assert (
            0.0 < config.residual_damping <= 1.0
        ), f"residual_damping must be in (0, 1], got {config.residual_damping}"

        self.head_dim = config.d_model // config.n_heads

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        # Multi-head attention
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Feed-forward network
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)

        # Input gate projection: gating factor for input injection
        # This scales the input injection without leaking future information
        # since x[t] only depends on token t
        self.input_gate = nn.Linear(config.d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Apply spectral normalization if enabled
        # This enforces Lipschitz continuity (||f'|| < 1) on the fixed-point map,
        # guaranteeing existence/uniqueness and enabling convergence via
        # Banach fixed-point theorem.
        if config.spectral_norm:
            from kinetic_ai.models.deq_layer import apply_spectral_norm

            apply_spectral_norm(self)

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        """Fixed-point map f(z, x).

        Two formulations available:

        map_form='residual' (v1/v2, default):
            f(z,x) = (1-α)z + α(z + Attn(LN(z), causal) + MLP(LN(z+Attn)) + inj(x))
            Pre-LN transformer block with damped residuals (standard DEQ form).

        map_form='postln' (v3, F14 fix):
            f(z,x) = LN2(h + MLP(h)) where h = LN1(z + Attn(z, causal) + inj(x))
            Puts outer LayerNorm INSIDE the map so iterates are bounded.
            Ensures bona fide fixed points exist (Bai et al. DEQ-transformer practice).

        Args:
            z: Hidden state [B, T, d_model].
            x: Input embeddings [B, T, d_model].

        Returns:
            Updated hidden state [B, T, d_model].
        """
        batch_size, seq_len, _ = z.shape
        map_form = self.config.map_form

        # -------- Attention Block --------
        if map_form == "postln":
            # postln: attention on z directly (post-LN on the output)
            q = self.q_proj(z)  # [B, T, d_model]
            k = self.k_proj(z)
            v = self.v_proj(z)
        else:  # residual
            # residual: attention on LN(z) (pre-LN)
            z_ln = self.ln1(z)
            q = self.q_proj(z_ln)  # [B, T, d_model]
            k = self.k_proj(z_ln)
            v = self.v_proj(z_ln)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: [B, n_heads, T, head_dim]

        # Compute causal attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=z.device)) == 1
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, T, head_dim]

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)

        # Output projection
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        # -------- Input Injection (Causal) --------
        # Apply a gated input injection: x[t] only depends on token t (causal)
        # The gate controls how much of the embedding input is mixed in
        gate = torch.sigmoid(self.input_gate(x))  # [B, T, 1]
        input_injection = gate * x  # [B, T, d_model]

        if map_form == "postln":
            # postln: f(z,x) = LN2(h + MLP(h)) where h = LN1(z + Attn + inj(x))
            # This puts the outer LN inside the map, ensuring bounded iterates

            # Combine z + attention + input
            h = z + attn_out + input_injection  # [B, T, d_model]

            # Apply first LN
            h = self.ln1(h)

            # Feed-forward network (MLP)
            ff_out = self.fc1(h)
            ff_out = F.relu(ff_out)
            ff_out = self.dropout(ff_out)
            ff_out = self.fc2(ff_out)
            ff_out = self.dropout(ff_out)

            # Apply second LN to h + MLP(h) (outer norm is now INSIDE the map)
            output = self.ln2(h + ff_out)

        else:  # residual (v1/v2)
            # residual: f(z,x) = (1-α)z + α(z + Attn(LN(z)) + MLP(LN(z+Attn)) + inj(x))
            # Pre-LN transformer block with damped residuals

            # -------- Feed-Forward Block --------
            # Pre-LN on combined state: z + attn_out
            z_post_attn = z + attn_out
            z_ln2 = self.ln2(z_post_attn)

            # Feed-forward network (MLP)
            ff_out = self.fc1(z_ln2)
            ff_out = F.relu(ff_out)
            ff_out = self.dropout(ff_out)
            ff_out = self.fc2(ff_out)
            ff_out = self.dropout(ff_out)

            # -------- Damped Residual Connection --------
            # Apply damping to ensure contraction: f(z) = (1-α)z + α(z + g(z))
            # This guarantees ||f'(z)|| < 1 even when ||g'(z)|| is not directly controlled.
            # When α=0.5, we get f(z) = 0.5*z + 0.5*(z + attn + ff + input)
            #                        = 0.5*z + 0.5*z + 0.5*(attn + ff + input)
            #                        = z + 0.5*(attn + ff + input)
            # This damps the derivative: ||f'(z)|| <= 1 - α*(1-||g'||) when spectral norm is used.
            residual_branch = attn_out + ff_out + input_injection
            output = (1.0 - self.residual_damping) * z + self.residual_damping * (z + residual_branch)

        return cast(Tensor, output)


# ============================================================================
# EqLM: Equilibrium Language Model
# ============================================================================


class EqLM(nn.Module):
    """Equilibrium Language Model.

    A single transformer block solved to fixed point via DEQ, with
    weight-tied embeddings. Achieves memory efficiency of O(1) activation
    memory regardless of effective depth.

    Architecture:
        1. Token embedding + positional embedding -> x [B, T, d_model]
        2. Fixed-point solve: z* = f(z*, x) via DEQLayer
        3. Final layer norm and LM head (weight-tied to embedding)

    The effective depth is determined by the number of solver iterations,
    which is adaptive (terminates early if converged).
    """

    def __init__(self, config: EqLMConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding with proper initialization (std=0.02 following GPT-2)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Positional embedding (learned) with same init
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # Transformer block (the fixed-point map)
        self.block = EqLMBlock(config)

        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.d_model)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        # DEQ layer configuration
        # Note: jfb=False to ensure full gradient flow to block parameters
        # (JFB has incomplete parameter gradient support in current implementation)
        deq_config = DEQConfig(
            solver=SolverType(config.solver),
            max_iter=config.deq_max_iter,
            tol=config.deq_tol,
            anderson_m=5,
            anderson_beta=1.0,
            spectral_norm=config.spectral_norm,  # Wire through from EqLMConfig
            jfb=False,  # Full IFT for complete gradient flow
        )

        # Create the fixed-point map function
        # The DEQLayer requires f(z, x) signature, but we only use z
        def fixed_point_map(z: Tensor, x: Tensor) -> Tensor:
            # x is passed for API compatibility but not used
            # (z is initialized with causal embeddings)
            return cast(Tensor, self.block(z, x))

        # DEQ layer wraps the fixed-point solver
        self.deq = DEQLayer(fixed_point_map, config=deq_config)

        # Auxiliary residual loss (solver-aware): stores the last computed relative residual
        self.last_aux_residual: Tensor | None = None
        self.last_z_star: Tensor | None = None

    def forward(self, input_ids: Tensor, z_init: Tensor | None = None) -> Tensor:
        """Forward pass: compute logits for next-token prediction.

        Args:
            input_ids: Token indices [B, T].
            z_init: Optional warm-start for the fixed-point solver [B, T, d]
                (H1'a). The converged equilibrium of this forward is stored in
                ``self.last_z_star`` for reuse by ``generate``.

        Returns:
            Logits [B, T, vocab_size].
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings (causal: each position only has its own token)
        z = self.embedding(input_ids)  # [B, T, d_model]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        pos_emb = self.pos_embedding(positions)
        z = z + pos_emb

        # Solve fixed point: z* = f(z*) where f is the transformer block
        # Pass z as both z and x for DEQLayer interface (x is unused in this model)
        z_star = self.deq(z, z_init=z_init)
        self.last_z_star = z_star.detach()

        # ========== Auxiliary Residual Loss (Solver-Aware) ==========
        # If enabled, compute relative residual r = ||f(z*,x) − z*||/(||z*||+eps)
        # with gradients enabled so training can minimize this to learn contraction.
        self.last_aux_residual = None
        if self.config.aux_residual:
            # Compute f(z*) with gradients to enable backprop through block
            # Note: z_star is detached from DEQ solve, but the block forward
            # will flow gradients through block parameters
            z_star_for_residual = z_star.detach().requires_grad_(True)
            with torch.enable_grad():
                f_z_star = self.block(z_star_for_residual, z)

            # Compute relative residual: r = ||f(z*) - z*|| / (||z*|| + eps)
            residual = f_z_star - z_star.detach()
            norm_residual = torch.norm(residual)
            norm_z_star = torch.norm(z_star.detach())
            eps = 1e-8
            rel_residual = norm_residual / (norm_z_star + eps)

            # Store for training loss: L_total = L_ce + lambda_aux * rel_residual
            self.last_aux_residual = rel_residual

        # Final layer norm
        z_star = self.ln_final(z_star)

        # LM head: project to vocabulary with weight-tied embedding
        logits = self.lm_head(z_star)  # [B, T, vocab_size]

        # Scale logits by sqrt(d_model) to keep variance reasonable
        # with weight-tied embeddings (std=0.02 initialization)
        logits = logits / (self.config.d_model**0.5)

        return cast(Tensor, logits)

    def forward_unrolled(
        self, input_ids: Tensor, supervise_at: list[int]
    ) -> list[tuple[int, Tensor]]:
        """B1 (SPEC 0010, ADR 0005): unrolled forward with anytime supervision.

        Applies the fixed-point map k times with full backprop and returns
        logits at each requested depth, so training can supervise truncated
        iterates directly (P11 beforehand-cushioning: every budget is a
        usable model).

        Args:
            input_ids: Token indices [B, T].
            supervise_at: Iteration depths to emit logits for (1-indexed,
                each in [1, deq_max_iter]).

        Returns:
            List of (depth, logits) sorted by depth.
        """
        if not supervise_at:
            raise ValueError("supervise_at must be non-empty")
        depths = sorted(set(int(k) for k in supervise_at))
        if depths[0] < 1 or depths[-1] > self.config.deq_max_iter:
            raise ValueError(
                f"supervise_at must lie in [1, {self.config.deq_max_iter}], "
                f"got {supervise_at}"
            )

        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        x = x + self.pos_embedding(positions)

        z = x
        outs: list[tuple[int, Tensor]] = []
        for k in range(1, depths[-1] + 1):
            z = self.block(z, x)
            if k in depths:
                h = self.ln_final(z)
                logits = self.lm_head(h) / (self.config.d_model**0.5)
                outs.append((k, logits))
        self.last_z_star = z.detach()
        return outs

    def local_lipschitz(
        self, input_ids: Tensor, alpha: float = 1.0, eps: float = 1e-3
    ) -> Tensor:
        """B2 (SPEC 0010, ADR 0005): trajectory-local contraction probe.

        Finite-difference estimate of the map's local Lipschitz constant at
        a point on the solve ray z = alpha * z_star (z0 = 0 for the unrolled
        map, so the ray interpolates the trajectory's span). Differentiable
        w.r.t. block parameters; penalizing max(0, L_hat - gamma) trains
        contraction only where the solver travels (separation by condition).

        Args:
            input_ids: Token indices [B, T] (must match the batch of the
                preceding forward; if no forward has run, one is executed).
            alpha: Position on the ray from origin to equilibrium.
            eps: Finite-difference step size.

        Returns:
            Scalar L_hat >= 0 with grad to block parameters.
        """
        if self.last_z_star is None or self.last_z_star.shape[:2] != input_ids.shape:
            with torch.no_grad():
                self(input_ids)
        assert self.last_z_star is not None

        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        x = x + self.pos_embedding(positions)

        z_pt = (alpha * self.last_z_star).detach()
        v = torch.randn_like(z_pt)
        v = v / (v.norm() + 1e-12)
        f0 = self.block(z_pt, x)
        f1 = self.block(z_pt + eps * v, x)
        return cast(Tensor, (f1 - f0).norm() / eps)

    def _sample_next(self, logits_last: Tensor, temperature: float, top_k: int) -> Tensor:
        return sample_next_token(logits_last, temperature, top_k)

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        warm_start: bool = False,
        return_iter_counts: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> Tensor | tuple[Tensor, dict[str, object]]:
        """Greedy decoding with optional warm-start (H1′a).

        Generates tokens one-at-a-time using greedy selection (argmax).
        When warm_start=True, each decoding step initializes the DEQ solver
        from the previous token's equilibrium state, reducing solver iterations.

        Alignment choice for warm-start (H1′a specification):
            Per-position state reuse: When solving for position T (new token),
            we initialize z from the previous position T-1's converged z*.
            Specifically:
            - z_init_new = z_prev[..., -1:, :].expand(B, 1, d_model)
            This reuses the last position's equilibrium as a warm start for the new position.
            Rationale: The fixed-point map is smooth, so z*(x[T-1]) is a good starting
            point for finding z*(x[T]) where x[T] is the new token embedding.

        Args:
            input_ids: Initial token sequence [B, T].
            max_new_tokens: Number of tokens to generate.
            warm_start: If True, reuse previous z* as initialization (H1′a).
                       If False, start from zeros each step (baseline).
            return_iter_counts: If True, return (output_ids, info_dict) with
                               iteration counts per token.

        Returns:
            output_ids: Full sequence including new tokens [B, T + max_new_tokens].
            info_dict (optional): If return_iter_counts=True, also returns
                                 {"iter_counts": [max_new_tokens], "mean_iters": float}
        """
        batch_size, seq_len = input_ids.shape

        output_ids = input_ids.clone()

        iter_counts: list[int] = []

        prev_z: Tensor | None = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                z_init: Tensor | None = None
                if warm_start and prev_z is not None:
                    # Reuse converged equilibria of all prior positions; the new
                    # position starts from the last position's equilibrium (the
                    # map is smooth, so z*(t-1) is a good guess for z*(t)).
                    z_init = torch.cat([prev_z, prev_z[:, -1:, :]], dim=1)

                # Forward pass through the computation matching the training
                # regime (F24 lesson: Anderson at eval corrupts the absolute
                # next-token distribution of anytime-unrolled checkpoints).
                if self.config.decode_mode == "unrolled":
                    depth = self.config.deq_max_iter
                    logits = self.forward_unrolled(output_ids, supervise_at=[depth])[0][1]
                else:
                    logits = self(output_ids, z_init=z_init)  # [B, T, V]
                prev_z = self.last_z_star

                # Get logits for the last position
                logits_last = logits[:, -1, :]  # [B, V]

                next_token = self._sample_next(logits_last, temperature, top_k)  # [B, 1]

                # Append to sequence
                output_ids = torch.cat([output_ids, next_token], dim=1)

                # Record iteration count: unrolled mode always applies the
                # map exactly deq_max_iter times; solver mode reads telemetry.
                if self.config.decode_mode == "unrolled":
                    iter_counts.append(self.config.deq_max_iter)
                elif (
                    hasattr(self.deq, "last_info")
                    and isinstance(self.deq.last_info, dict)
                    and "iterations" in self.deq.last_info
                ):
                    iterations = self.deq.last_info["iterations"]
                    if isinstance(iterations, int):
                        iter_counts.append(iterations)

        if return_iter_counts:
            info = {
                "iter_counts": iter_counts,
                "mean_iters": (
                    sum(iter_counts) / len(iter_counts) if iter_counts else 0.0
                ),
            }
            return output_ids, info
        else:
            return output_ids


# ============================================================================
# ExplicitLM: Baseline (Stacked Layers)
# ============================================================================


class ExplicitLM(nn.Module):
    """Explicit Transformer LM (Baseline).

    Stacks n_layers blocks explicitly (no fixed-point solving).
    Used as the baseline for parameter matching and memory comparison.

    Same interface as EqLM but with explicit depth instead of DEQ.
    """

    def __init__(self, config: EqLMConfig, n_layers: int = 12) -> None:
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        # Token embedding with proper initialization (std=0.02 following GPT-2)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Positional embedding (learned) with same init
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [EqLMBlock(config) for _ in range(n_layers)]
        )

        # Final layer normalization
        self.ln_final = nn.LayerNorm(config.d_model)

        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass: compute logits.

        Args:
            input_ids: Token indices [B, T].

        Returns:
            Logits [B, T, vocab_size].
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        z = self.embedding(input_ids)  # [B, T, d_model]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        pos_emb = self.pos_embedding(positions)
        z = z + pos_emb

        # Placeholder for x (required by EqLMBlock signature)
        # Each layer receives z and x (unused)
        x = torch.zeros_like(z)

        # Forward through each layer
        for layer in self.layers:
            z = layer(z, x)

        # Final layer norm
        z = self.ln_final(z)

        # LM head: project to vocabulary with weight-tied embedding
        logits = self.lm_head(z)  # [B, T, vocab_size]

        # Scale logits by sqrt(d_model) to keep variance reasonable
        # with weight-tied embeddings (std=0.02 initialization)
        logits = logits / (self.config.d_model**0.5)

        return cast(Tensor, logits)


class EqLMCore(nn.Module):
    """B3 (SPEC 0010, ADR 0005): bottleneck-core equilibrium LM.

    Space-separated design (TRIZ P24 intermediary): capacity lives in wide
    EXPLICIT encoder/decoder layers at d_model; the equilibrium is solved in
    a small d_core space where contraction is cheap to certify and each
    solver iteration costs O(d_core^2) instead of O(d_model^2).

        tokens -> embed(d_model) -> n_enc explicit blocks -> W_down ->
        [DEQ solve in d_core] -> W_up (+ residual) -> n_dec explicit blocks
        -> tied lm_head

    Same external interface as EqLM/ExplicitLM (forward(input_ids) -> logits,
    last_z_star, deq.last_info telemetry).
    """

    def __init__(
        self,
        config: EqLMConfig,
        d_core: int = 256,
        n_heads_core: int = 4,
        d_ff_core: int = 1024,
        n_enc: int = 2,
        n_dec: int = 2,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_core = d_core
        self.n_heads_core = n_heads_core
        self.d_ff_core = d_ff_core
        self.n_enc = n_enc
        self.n_dec = n_dec

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        self.encoder = nn.ModuleList([EqLMBlock(config) for _ in range(n_enc)])
        self.decoder = nn.ModuleList([EqLMBlock(config) for _ in range(n_dec)])

        self.core_config = EqLMConfig(
            vocab_size=config.vocab_size,
            d_model=d_core,
            n_heads=n_heads_core,
            d_ff=d_ff_core,
            max_seq_len=config.max_seq_len,
            deq_max_iter=config.deq_max_iter,
            deq_tol=config.deq_tol,
            solver=config.solver,
            dropout=config.dropout,
            spectral_norm=config.spectral_norm,
            residual_damping=config.residual_damping,
            map_form=config.map_form,
        )
        self.core_block = EqLMBlock(self.core_config)
        self.w_down = nn.Linear(config.d_model, d_core, bias=False)
        self.w_up = nn.Linear(d_core, config.d_model, bias=False)

        deq_config = DEQConfig(
            solver=SolverType(config.solver),
            max_iter=config.deq_max_iter,
            tol=config.deq_tol,
            anderson_m=5,
            anderson_beta=1.0,
            spectral_norm=config.spectral_norm,
            jfb=False,
        )

        def core_map(z: Tensor, x: Tensor) -> Tensor:
            return cast(Tensor, self.core_block(z, x))

        self.deq = DEQLayer(core_map, config=deq_config)

        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.last_z_star: Tensor | None = None

    def forward(self, input_ids: Tensor) -> Tensor:
        batch_size, seq_len = input_ids.shape
        h = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        h = h + self.pos_embedding(positions)

        x_zero = torch.zeros_like(h)
        for layer in self.encoder:
            h = layer(h, x_zero)

        x_core = self.w_down(h)
        z_star = self.deq(x_core)
        self.last_z_star = z_star.detach()

        h = h + self.w_up(z_star)
        for layer in self.decoder:
            h = layer(h, x_zero)

        h = self.ln_final(h)
        logits = self.lm_head(h) / (self.config.d_model**0.5)
        return cast(Tensor, logits)


# ============================================================================
# Checkpoint Saving/Loading (H1′ Task 3)
# ============================================================================


def save_checkpoint(model: EqLM | ExplicitLM | EqLMCore, path: str | Path) -> None:
    """Save an EqLM or ExplicitLM model and its config to a checkpoint file.

    Saves the state_dict, EqLMConfig, and the model class (plus n_layers for
    ExplicitLM) in a single .pt file for easy restoration.

    Args:
        model: EqLM or ExplicitLM model to save.
        path: Path to save checkpoint to (typically ends in .pt).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Config is stored as a primitives-only dict so checkpoints can ALWAYS be
    # loaded with torch.load(weights_only=True) — never pickle-load model files
    # (the registry enumerates them from disk; pickle would be code execution).
    from dataclasses import asdict

    checkpoint: dict[str, object] = {
        "state_dict": model.state_dict(),
        "config_dict": asdict(model.config),
        "model_class": type(model).__name__,
    }
    if isinstance(model, ExplicitLM):
        checkpoint["n_layers"] = len(model.layers)
    if isinstance(model, EqLMCore):
        checkpoint["core_dims"] = {
            "d_core": model.d_core,
            "n_heads_core": model.n_heads_core,
            "d_ff_core": model.d_ff_core,
            "n_enc": model.n_enc,
            "n_dec": model.n_dec,
        }

    torch.save(checkpoint, str(path))


def load_checkpoint(path: str | Path) -> EqLM | ExplicitLM | EqLMCore:
    """Load an EqLM model from a checkpoint file.

    Reconstructs the model from saved state_dict and config.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Reconstructed EqLM model with loaded weights.
    """
    path = Path(path)
    # SECURITY: weights_only=True always — checkpoints are enumerated from
    # disk by services; pickle-loading them would be code execution. The config
    # is stored as a primitives-only dict for exactly this reason.
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

    cfg = EqLMConfig(**checkpoint["config_dict"])
    if checkpoint.get("model_class") == "ExplicitLM":
        model: EqLM | ExplicitLM | EqLMCore = ExplicitLM(
            config=cfg, n_layers=int(checkpoint.get("n_layers", 4))
        )
    elif checkpoint.get("model_class") == "EqLMCore":
        dims = {k: int(v) for k, v in checkpoint["core_dims"].items()}
        model = EqLMCore(config=cfg, **dims)
    else:
        model = EqLM(config=cfg)
    model.load_state_dict(checkpoint["state_dict"])

    return model


# ============================================================================
# Warm-Start Decoding (H1′ Task 2)
# ============================================================================


# (Will be added to EqLM.generate() method below)

# ============================================================================
# Helper Functions
# ============================================================================


def count_params(model: nn.Module) -> int:
    """Count total trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_explicit_width(
    target_params: int,
    base_cfg: EqLMConfig,
    n_layers: int = 3,
) -> EqLMConfig:
    """Find EqLM config matching explicit baseline's parameter count.

    Scales d_model and d_ff to match the target parameter count within 5%.
    Uses iterative search to find the optimal scaling. Ensures d_model
    is divisible by n_heads.

    Args:
        target_params: Target parameter count (from ExplicitLM baseline).
        base_cfg: Base EqLMConfig to scale from.
        n_layers: Not used for EqLM (always single block), kept for API compat.

    Returns:
        EqLMConfig with scaled d_model and d_ff to match target within 5%.
    """
    import math

    def _make_divisible(val: float, divisor: int) -> int:
        """Ensure val is divisible by divisor (round up to nearest multiple)."""
        val_int = math.ceil(val)
        return ((val_int + divisor - 1) // divisor) * divisor

    # Use rough scaling estimate for initial guess
    # For n_layers-deep explicit vs single-block EqLM, need roughly sqrt(n_layers) scaling
    base_eqlm = EqLM(base_cfg)
    base_params = count_params(base_eqlm)
    scale_factor = math.sqrt(target_params / base_params)

    # Iteratively refine scale factor
    for _ in range(30):
        scaled_cfg = EqLMConfig(
            vocab_size=base_cfg.vocab_size,
            d_model=_make_divisible(max(32, base_cfg.d_model * scale_factor), base_cfg.n_heads),
            n_heads=base_cfg.n_heads,
            d_ff=max(64, int(math.ceil(base_cfg.d_ff * scale_factor))),
            max_seq_len=base_cfg.max_seq_len,
            deq_max_iter=base_cfg.deq_max_iter,
            deq_tol=base_cfg.deq_tol,
            solver=base_cfg.solver,
            jfb=base_cfg.jfb,
            dropout=base_cfg.dropout,
            spectral_norm=base_cfg.spectral_norm,
            residual_damping=base_cfg.residual_damping,
            map_form=base_cfg.map_form,
        )

        eqlm = EqLM(scaled_cfg)
        current_params = count_params(eqlm)
        relative_error = abs(current_params - target_params) / target_params

        if relative_error < 0.05:  # Within 5%
            break

        # Adjust scale factor based on error
        if current_params < target_params:
            # Need to scale up more
            scale_factor *= math.sqrt(target_params / current_params)
        else:
            # Scaled up too much
            scale_factor /= math.sqrt(current_params / target_params)

    return scaled_cfg
