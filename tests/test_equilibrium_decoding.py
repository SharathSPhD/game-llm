"""TDD for equilibrium decoding (ADR 0008).

The next-token distribution is the tau-regularized QRE of an influence game
among model-players, computed by Magnetic Mirror Descent in policy space. The
properties below are what make it more than an ensemble: it must reduce to
known rules in the degenerate cases, converge, respect the magnet, and stay
cheap enough to run per token.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.decode.equilibrium import (
    EquilibriumConfig,
    solve_equilibrium,
)


def _logits(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


class TestDegenerateCases:
    """The equilibrium must contain the existing rules as special cases —
    otherwise it is a different method, not a generalisation."""

    def test_single_player_returns_that_player(self) -> None:
        ell = _logits([[2.0, 1.0, 0.0, -1.0]])
        out = solve_equilibrium(ell, EquilibriumConfig(tau=0.0, beta=0.0))
        torch.testing.assert_close(out, torch.softmax(ell[0], dim=-1), atol=2e-3, rtol=0)

    def test_zero_influence_recovers_uniform_averaging(self) -> None:
        """beta=0 removes the game: weights stay uniform, so the result is the
        standard logit-average ensemble."""
        ell = _logits([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        out = solve_equilibrium(ell, EquilibriumConfig(tau=0.0, beta=0.0))
        expected = torch.softmax(ell.mean(dim=0), dim=-1)
        torch.testing.assert_close(out, expected, atol=5e-3, rtol=0)

    def test_high_influence_concentrates_on_the_decisive_player(self) -> None:
        """Large beta makes the game winner-take-most, approaching routing."""
        confident = _logits([[8.0, 0.0, 0.0], [0.3, 0.2, 0.1]])
        out = solve_equilibrium(confident, EquilibriumConfig(tau=0.0, beta=6.0))
        avg = torch.softmax(confident.mean(dim=0), dim=-1)
        assert out[0] > avg[0], "decisive player should gain influence over averaging"


class TestEquilibriumProperties:
    def test_output_is_a_distribution(self) -> None:
        ell = _logits([[1.0, 2.0, 3.0], [3.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        out = solve_equilibrium(ell, EquilibriumConfig())
        assert torch.all(out >= 0)
        torch.testing.assert_close(out.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)

    def test_converges_and_reports_iterations(self) -> None:
        ell = _logits([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        out, info = solve_equilibrium(
            ell, EquilibriumConfig(max_iter=200, tol=1e-6), return_info=True
        )
        assert info["converged"], info
        assert info["iterations"] < 200
        assert info["residual"] < 1e-6

    def test_magnet_pulls_toward_reference(self) -> None:
        """tau is the magnetic strength: raising it moves the equilibrium toward
        the reference distribution, which is what keeps fluency."""
        ell = _logits([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        ref = torch.tensor([0.0, 1.0, 0.0])
        weak = solve_equilibrium(ell, EquilibriumConfig(tau=0.01), reference=ref)
        strong = solve_equilibrium(ell, EquilibriumConfig(tau=5.0), reference=ref)
        assert strong[1] > weak[1], "stronger magnet must move mass toward the reference"

    def test_warm_start_reduces_iterations(self) -> None:
        """F19's property: consecutive positions have nearby equilibria."""
        a = _logits([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        b = a + 0.05 * torch.randn_like(a)
        first, _ = solve_equilibrium(a, EquilibriumConfig(), return_info=True)
        _, cold = solve_equilibrium(b, EquilibriumConfig(), return_info=True)
        _, warm = solve_equilibrium(b, EquilibriumConfig(), y_init=first, return_info=True)
        assert warm["iterations"] <= cold["iterations"]

    def test_anytime_truncation_still_returns_a_valid_distribution(self) -> None:
        ell = _logits([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        out = solve_equilibrium(ell, EquilibriumConfig(max_iter=1))
        assert torch.all(out >= 0)
        torch.testing.assert_close(out.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)

    def test_batched_solve_matches_per_row_solve(self) -> None:
        """Decoding runs a batch of positions; batching must not change results."""
        batch = torch.stack([
            _logits([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]]),
            _logits([[0.5, 0.5, 3.0], [1.0, 0.0, 2.0]]),
        ])  # [B, N, V]
        got = solve_equilibrium(batch, EquilibriumConfig())
        for b in range(batch.shape[0]):
            one = solve_equilibrium(batch[b], EquilibriumConfig())
            torch.testing.assert_close(got[b], one, atol=1e-5, rtol=0)

    def test_disagreement_is_resolved_not_blurred(self) -> None:
        """Two confident players disagreeing: averaging splits mass evenly and
        leaves the argmax ambiguous, while the influence game should commit."""
        ell = _logits([[7.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.1, 0.0, 0.0]])
        avg = torch.softmax(ell.mean(dim=0), dim=-1)
        eq = solve_equilibrium(ell, EquilibriumConfig(beta=4.0))
        assert eq.max() > avg.max(), "equilibrium should be more decisive than the average"


class TestCost:
    def test_iterations_are_cheap_relative_to_vocabulary(self) -> None:
        """The claim that this runs at ensemble cost depends on the solve being
        vector operations over the vocabulary, not another forward pass."""
        import time

        ell = torch.randn(3, 32000)
        t0 = time.time()
        for _ in range(20):
            solve_equilibrium(ell, EquilibriumConfig(max_iter=8))
        elapsed = (time.time() - t0) / 20
        assert elapsed < 0.05, f"solve too slow for per-token use: {elapsed*1000:.1f}ms"
