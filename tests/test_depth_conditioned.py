"""TDD for depth conditioning (cycle 31, TRIZ cell 36/26).

The properties pinned here are the ones the comparison depends on: the modulated
model must start identical to the unmodulated one, the map must genuinely differ
between iterations once trained, the cost must stay negligible, and the anytime
property F24 established must survive running past the trained depth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.depth_conditioned import (
    DepthConditionedBlock,
    DepthFiLM,
    count_conditioning_parameters,
)


class _Dummy(nn.Module):
    """A stand-in block: deterministic, so any difference comes from modulation."""

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return z * 0.5 + x


class TestStartsAsIdentity:
    def test_untrained_modulation_leaves_the_block_unchanged(self) -> None:
        """Otherwise the comparison would confound depth conditioning with a
        different initialisation."""
        z = torch.randn(2, 4, 8)
        x = torch.randn(2, 4, 8)
        plain = _Dummy()
        cond = DepthConditionedBlock(_Dummy(), d_model=8)
        torch.testing.assert_close(cond(z, x), plain(z, x))

    def test_identity_holds_across_several_iterations(self) -> None:
        z, x = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        plain, cond = _Dummy(), DepthConditionedBlock(_Dummy(), d_model=8)
        zp, zc = z, z
        for _ in range(5):
            zp, zc = plain(zp, x), cond(zc, x)
        torch.testing.assert_close(zc, zp)


class TestTheMapActuallyVariesWithDepth:
    def test_trained_modulation_gives_different_maps_at_different_steps(self) -> None:
        cond = DepthConditionedBlock(_Dummy(), d_model=8)
        with torch.no_grad():
            cond.film.gamma[0].fill_(2.0)
            cond.film.gamma[1].fill_(0.5)
        z, x = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        first = cond(z, x)
        cond.reset()
        cond._step = 1
        second = cond(z, x)
        assert not torch.allclose(first, second)

    def test_reset_returns_to_the_first_map(self) -> None:
        cond = DepthConditionedBlock(_Dummy(), d_model=8)
        with torch.no_grad():
            cond.film.gamma[0].fill_(3.0)
        z, x = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        a = cond(z, x)
        cond(z, x)
        cond.reset()
        torch.testing.assert_close(cond(z, x), a)


class TestAnytimeSurvives:
    def test_iterating_past_the_trained_depth_reuses_the_last_map(self) -> None:
        """F24's budget dial must keep working: a model trained at twelve
        iterations has to run at twenty without failing."""
        film = DepthFiLM(d_model=8, max_depth=3)
        z = torch.randn(2, 4, 8)
        torch.testing.assert_close(film(z, 5), film(z, 2))

    def test_a_long_solve_does_not_raise(self) -> None:
        cond = DepthConditionedBlock(_Dummy(), d_model=8, max_depth=4)
        z, x = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
        for _ in range(20):
            z = cond(z, x)
        assert torch.isfinite(z).all()


class TestCost:
    def test_the_parameter_cost_is_negligible_against_untying(self) -> None:
        d, depth = 768, 12
        conditioning = count_conditioning_parameters(d, depth)
        # Untying would add eleven more blocks of 4d^2 + 2*d*d_ff each.
        untying = 11 * (4 * d * d + 2 * d * 4 * d)
        assert conditioning == 2 * d * depth
        assert untying / conditioning > 1000

    def test_reported_cost_matches_the_module(self) -> None:
        film = DepthFiLM(d_model=16, max_depth=5)
        assert sum(p.numel() for p in film.parameters()) == count_conditioning_parameters(16, 5)
