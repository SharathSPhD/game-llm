"""Equilibrium decoding (ADR 0008).

The next-token distribution is the tau-regularized quantal response equilibrium
of an influence game among model-players, rather than the output of a single
forward pass or a fixed blend of several.

At a decoding position, players hold logits $\\ell_i$. The consensus $y$ over the
vocabulary and the players' influence weights $w_i$ are jointly determined:

    w_i(y) = softmax_i( beta * <y, ell_i> )          influence follows payoff
    log y  <- (1-eta) log y + eta * ( sum_i w_i(y) ell_i + tau log p_ref )

normalised back onto the simplex each step. This is mirror descent under the
negative-entropy map — the geometry of the simplex — with the magnetic proximal
term supplied by tau, which is the form this project measured to converge
linearly in the last iterate where simultaneous play cycles. Its fixed point,

    log y* = sum_i w_i(y*) ell_i + tau log p_ref - log Z,

is the equilibrium the decoder samples from. The Euclidean form of the same
update was tried first and rejected: its fixed point is the arithmetic mean of
the players' distributions, whereas both ensembling practice and the simplex
geometry call for the geometric mean.

The construction is a strict generalisation of the rules it replaces. With
``beta = 0`` the weights stay uniform and the result is logit averaging; as
``beta`` grows the decisive player takes most of the influence, approaching
routing. Neither degenerate case is imposed — both fall out of the same solve,
which is why the equilibrium can beat either.

Cost is what makes it usable: after one forward pass per player, everything here
is softmax and dot products over the vocabulary, so a handful of iterations adds
negligible time against the forwards themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class EquilibriumConfig:
    """Parameters of the decode-time game.

    Attributes:
        tau: Magnetic strength pulling the consensus toward the reference
            distribution in log space. This is the fluency anchor; it plays the
            role the regularizer plays in the QRE, so larger values give a
            softer, more reference-like equilibrium. Zero recovers pure council
            aggregation.
        beta: Rationality of the influence assignment. Zero gives uniform
            weights (logit averaging); large values concentrate influence on
            whichever player the current consensus most rewards.
        eta: Mirror-descent step size.
        max_iter: Iteration budget. Truncation is safe — every iterate is a
            valid distribution, which is what makes the solve anytime.
        tol: Convergence tolerance on the L1 movement of the consensus.
        temperature: Applied to player logits before the game, so decoding
            temperature enters as the players' rationality rather than as a
            post-hoc reshaping of the result.
    """

    tau: float = 0.0
    beta: float = 2.0
    eta: float = 0.5
    max_iter: int = 32
    tol: float = 1e-5
    temperature: float = 1.0


def solve_equilibrium(
    logits: torch.Tensor,
    config: EquilibriumConfig | None = None,
    reference: torch.Tensor | None = None,
    y_init: torch.Tensor | None = None,
    return_info: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Solve for the equilibrium next-token distribution.

    Args:
        logits: Player logits, ``[N, V]`` for a single position or ``[B, N, V]``
            for a batch of positions.
        config: Game parameters.
        reference: Magnet distribution ``[V]`` or ``[B, V]``. Defaults to the
            mean of the players' distributions, which keeps the anchor inside
            the council when no general-purpose model is designated.
        y_init: Warm start, typically the previous position's equilibrium.
        return_info: Also return solver telemetry.

    Returns:
        The equilibrium distribution ``[V]`` or ``[B, V]``; with ``return_info``,
        a tuple of that and a dict carrying iterations, residual and converged.
    """
    cfg = config or EquilibriumConfig()
    batched = logits.dim() == 3
    ell = logits if batched else logits.unsqueeze(0)  # [B, N, V]
    ell = ell.float() / max(cfg.temperature, 1e-6)

    players = F.softmax(ell, dim=-1)  # [B, N, V]

    if reference is None:
        ref = players.mean(dim=1)
    else:
        ref = reference.float()
        ref = ref.unsqueeze(0) if ref.dim() == 1 else ref
        ref = ref / ref.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    y = ref.clone() if y_init is None else (
        y_init.float().unsqueeze(0) if y_init.dim() == 1 else y_init.float()
    ).clone()
    y = y / y.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if y.shape[0] != ell.shape[0]:
        y = y.expand(ell.shape[0], -1).contiguous()

    iterations = 0
    residual = float("inf")
    converged = False

    log_ref = torch.log(ref.clamp_min(1e-12))
    log_y = torch.log(y.clamp_min(1e-12))

    for step in range(cfg.max_iter):
        # Influence follows payoff: a player gains weight when the current
        # consensus scores well under its own logits. Truthful confidence bids
        # (F6) are what stop this from being gameable by a miscalibrated player.
        payoff = torch.einsum("bv,bnv->bn", y, ell)  # [B, N]
        w = (
            F.softmax(cfg.beta * payoff, dim=-1)
            if cfg.beta != 0
            else torch.full_like(payoff, 1.0 / payoff.shape[-1])
        )

        # Council direction in log space, plus the magnetic pull toward the
        # reference. Working in logs is what makes the degenerate case the
        # geometric mean, i.e. ordinary logit averaging.
        target = torch.einsum("bn,bnv->bv", w, ell) + cfg.tau * log_ref
        log_y_new = (1.0 - cfg.eta) * log_y + cfg.eta * target
        log_y_new = log_y_new - torch.logsumexp(log_y_new, dim=-1, keepdim=True)

        y_new = log_y_new.exp()
        residual = float((y_new - y).abs().sum(dim=-1).max().item())
        log_y, y = log_y_new, y_new
        iterations = step + 1
        if residual < cfg.tol:
            converged = True
            break

    out = y if batched else y.squeeze(0)
    if return_info:
        return out, {
            "iterations": iterations,
            "residual": residual,
            "converged": converged,
        }
    return out
