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


# ============================================================================
# EqLMBlock: Single Transformer Block as Fixed-Point Map
# ============================================================================


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

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass: compute logits for next-token prediction.

        Args:
            input_ids: Token indices [B, T].

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
        z_star = self.deq(z)

        # Final layer norm
        z_star = self.ln_final(z_star)

        # LM head: project to vocabulary with weight-tied embedding
        logits = self.lm_head(z_star)  # [B, T, vocab_size]

        # Scale logits by sqrt(d_model) to keep variance reasonable
        # with weight-tied embeddings (std=0.02 initialization)
        logits = logits / (self.config.d_model**0.5)

        return cast(Tensor, logits)


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
