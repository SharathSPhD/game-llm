"""Quantal Response Equilibrium (QRE) computation and verification.

QRE generalizes Nash Equilibrium by modeling agents with bounded rationality.
Instead of playing exact best responses, agents play strategies with
probabilities proportional to their expected utilities.

The logit QRE is:
    σ_i(a) = exp(λ · u_i(a, σ_{-i})) / Σ_a' exp(λ · u_i(a', σ_{-i}))

where λ is the rationality parameter (inverse temperature).
    - λ → 0: Uniform random play (maximum entropy)
    - λ → ∞: Best response (converges to Nash Equilibrium)

Key Functions:
    - compute_qre: Find the QRE for a normal-form game via fixed-point iteration
    - verify_qre: Check if a strategy profile is an ε-QRE
    - nash_conv: Compute NashConv (exploitability) of a strategy profile
    - qre_path: Trace the QRE correspondence as λ varies (homotopy method)

References:
    [1] McKelvey & Palfrey "Quantal Response Equilibria for Normal Form
        Games" (Games and Economic Behavior, 1995)
    [2] Sokota et al. "A Unified Approach to RL, QRE, and Two-Player
        Zero-Sum Games" (NeurIPS 2023)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from kinetic_ai.games.payoff import NormalFormGame


@dataclass
class QREResult:
    """Result of QRE computation.

    Attributes:
        strategy_1: QRE strategy for player 1.
        strategy_2: QRE strategy for player 2.
        rationality: The λ parameter used.
        iterations: Number of iterations to converge.
        converged: Whether the computation converged.
        residual: Final residual (distance from fixed point).
        nash_conv: NashConv (exploitability) of the result.
    """

    strategy_1: Tensor
    strategy_2: Tensor
    rationality: float
    iterations: int
    converged: bool
    residual: float
    nash_conv: float


def logit_qre_response(
    payoff_matrix: Tensor,
    opponent_strategy: Tensor,
    rationality: float,
) -> Tensor:
    """Compute the logit QRE best response.

    σ(a) = softmax(λ · (A · σ_{-i}))

    Args:
        payoff_matrix: Payoff matrix. Shape: (num_actions, opponent_actions)
        opponent_strategy: Opponent's mixed strategy. Shape: (opponent_actions,)
        rationality: λ parameter.

    Returns:
        QRE response strategy. Shape: (num_actions,)
    """
    expected_utilities = payoff_matrix @ opponent_strategy
    return F.softmax(rationality * expected_utilities, dim=-1)


def compute_qre(
    game: NormalFormGame,
    rationality: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> QREResult:
    """Compute the Quantal Response Equilibrium via fixed-point iteration.

    Iterates:
        σ1 ← softmax(λ · A1 · σ2)
        σ2 ← softmax(λ · A2^T · σ1)

    until convergence.

    Args:
        game: The normal-form game.
        rationality: λ parameter (inverse temperature).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        QREResult with the equilibrium strategies and diagnostics.
    """
    # Initialize with uniform strategies
    s1 = torch.ones(game.num_actions_1) / game.num_actions_1
    s2 = torch.ones(game.num_actions_2) / game.num_actions_2

    for i in range(max_iter):
        s1_new = logit_qre_response(game.payoff_1, s2, rationality)
        s2_new = logit_qre_response(game.payoff_2.T, s1_new, rationality)

        residual = (torch.norm(s1_new - s1) + torch.norm(s2_new - s2)).item()

        s1, s2 = s1_new, s2_new

        if residual < tol:
            break

    nc = nash_conv(game, s1, s2)

    return QREResult(
        strategy_1=s1,
        strategy_2=s2,
        rationality=rationality,
        iterations=i + 1,
        converged=residual < tol,
        residual=residual,
        nash_conv=nc,
    )


def nash_conv(
    game: NormalFormGame,
    strategy_1: Tensor,
    strategy_2: Tensor,
) -> float:
    """Compute NashConv (exploitability) of a strategy profile.

    NashConv = Σ_i [max_a u_i(a, σ_{-i}) - u_i(σ_i, σ_{-i})]

    This measures how much each player could gain by deviating from
    their current strategy. At a Nash Equilibrium, NashConv = 0.

    Args:
        game: The normal-form game.
        strategy_1: Player 1's strategy.
        strategy_2: Player 2's strategy.

    Returns:
        Non-negative NashConv value.
    """
    # Current expected payoffs
    u1_current = (strategy_1 @ game.payoff_1 @ strategy_2).item()
    u2_current = (strategy_1 @ game.payoff_2 @ strategy_2).item()

    # Best response payoffs
    u1_br = (game.payoff_1 @ strategy_2).max().item()
    u2_br = (game.payoff_2.T @ strategy_1).max().item()

    return max(0.0, u1_br - u1_current) + max(0.0, u2_br - u2_current)


def verify_qre(
    game: NormalFormGame,
    strategy_1: Tensor,
    strategy_2: Tensor,
    rationality: float,
    tol: float = 1e-4,
) -> bool:
    """Verify that a strategy profile is an ε-QRE.

    Checks that each strategy is close to the logit QRE response
    to the opponent's strategy.

    Args:
        game: The normal-form game.
        strategy_1: Player 1's strategy to verify.
        strategy_2: Player 2's strategy to verify.
        rationality: λ parameter.
        tol: Tolerance for verification.

    Returns:
        True if the profile is an ε-QRE.
    """
    s1_expected = logit_qre_response(game.payoff_1, strategy_2, rationality)
    s2_expected = logit_qre_response(game.payoff_2.T, strategy_1, rationality)

    return (
        torch.norm(strategy_1 - s1_expected).item() < tol
        and torch.norm(strategy_2 - s2_expected).item() < tol
    )


def qre_path(
    game: NormalFormGame,
    rationality_values: list[float] | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> list[QREResult]:
    """Trace the QRE correspondence as λ varies.

    Computes the QRE for a sequence of λ values, using the previous
    solution as a warm start for the next. This traces the "principal
    branch" of the QRE correspondence from uniform play (λ=0) to
    Nash equilibrium (λ→∞).

    Args:
        game: The normal-form game.
        rationality_values: Sequence of λ values to compute.
            If None, uses a logarithmic sweep from 0.01 to 100.
        max_iter: Max iterations per QRE computation.
        tol: Convergence tolerance.

    Returns:
        List of QREResult, one per λ value.
    """
    if rationality_values is None:
        rationality_values = torch.logspace(-2, 2, 50).tolist()

    results: list[QREResult] = []
    for lam in rationality_values:
        result = compute_qre(game, rationality=lam, max_iter=max_iter, tol=tol)
        results.append(result)

    return results
