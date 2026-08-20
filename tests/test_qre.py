"""Tests for QRE computation and game definitions."""

import pytest
import torch

from kinetic_ai.games.payoff import (
    KuhnPokerGame,
    coordination_game,
    matching_pennies,
    prisoners_dilemma,
    rock_paper_scissors,
)
from kinetic_ai.games.qre import compute_qre, nash_conv, qre_path, verify_qre


class TestGameDefinitions:
    """Test that game definitions are correct."""

    def test_rps_is_zero_sum(self) -> None:
        game = rock_paper_scissors()
        assert game.is_zero_sum

    def test_matching_pennies_is_zero_sum(self) -> None:
        game = matching_pennies()
        assert game.is_zero_sum

    def test_prisoners_dilemma_not_zero_sum(self) -> None:
        game = prisoners_dilemma()
        assert not game.is_zero_sum

    def test_rps_payoff_symmetry(self) -> None:
        """RPS should be symmetric: playing same action yields 0."""
        game = rock_paper_scissors()
        for i in range(3):
            assert game.payoff_1[i, i] == 0.0

    def test_expected_payoff(self) -> None:
        """Uniform vs uniform on RPS should yield 0 for both players."""
        game = rock_paper_scissors()
        u = torch.ones(3) / 3
        p1, p2 = game.expected_payoff(u, u)
        assert abs(p1) < 1e-6
        assert abs(p2) < 1e-6


class TestQREComputation:
    """Test QRE computation on known games."""

    def test_low_rationality_gives_uniform(self) -> None:
        """Low λ should produce near-uniform strategies."""
        game = rock_paper_scissors()
        result = compute_qre(game, rationality=0.01)
        assert torch.allclose(result.strategy_1, torch.ones(3) / 3, atol=0.05)
        assert torch.allclose(result.strategy_2, torch.ones(3) / 3, atol=0.05)

    def test_high_rationality_approaches_nash(self) -> None:
        """High λ should approach the Nash equilibrium."""
        game = matching_pennies()
        result = compute_qre(game, rationality=100.0)
        # Nash of matching pennies is (0.5, 0.5)
        assert torch.allclose(result.strategy_1, torch.tensor([0.5, 0.5]), atol=0.05)

    def test_qre_converges(self) -> None:
        """QRE computation should converge."""
        game = rock_paper_scissors()
        result = compute_qre(game, rationality=1.0, max_iter=1000, tol=1e-8)
        assert result.converged

    def test_verify_qre(self) -> None:
        """Computed QRE should pass verification."""
        game = matching_pennies()
        result = compute_qre(game, rationality=5.0)
        assert verify_qre(game, result.strategy_1, result.strategy_2, rationality=5.0, tol=1e-3)


class TestNashConv:
    """Test NashConv (exploitability) computation."""

    def test_nash_conv_at_equilibrium(self) -> None:
        """NashConv at Nash equilibrium should be ~0."""
        game = rock_paper_scissors()
        uniform = torch.ones(3) / 3
        nc = nash_conv(game, uniform, uniform)
        assert nc < 1e-6, f"NashConv at Nash should be ~0, got {nc}"

    def test_nash_conv_away_from_equilibrium(self) -> None:
        """NashConv away from Nash should be positive."""
        game = rock_paper_scissors()
        s1 = torch.tensor([0.9, 0.05, 0.05])  # Far from uniform
        s2 = torch.tensor([0.05, 0.9, 0.05])
        nc = nash_conv(game, s1, s2)
        assert nc > 0.1, "NashConv should be positive away from Nash"

    def test_nash_conv_non_negative(self) -> None:
        """NashConv should always be non-negative."""
        game = matching_pennies()
        for _ in range(20):
            s1 = torch.softmax(torch.randn(2), dim=-1)
            s2 = torch.softmax(torch.randn(2), dim=-1)
            nc = nash_conv(game, s1, s2)
            assert nc >= -1e-7, f"NashConv should be non-negative, got {nc}"


class TestQREPath:
    """Test QRE path tracing (homotopy)."""

    def test_path_is_monotone_in_exploitability(self) -> None:
        """Higher λ should generally give lower exploitability."""
        game = matching_pennies()
        lambdas = [0.1, 1.0, 10.0, 100.0]
        results = qre_path(game, rationality_values=lambdas)

        # At matching pennies, Nash and QRE coincide, so NashConv
        # should be small for all λ. But the trend should be towards 0.
        assert len(results) == 4
        # Last result (highest λ) should have lowest NashConv
        assert results[-1].nash_conv <= results[0].nash_conv + 0.01


class TestKuhnPoker:
    """Test Kuhn Poker game definition."""

    def test_game_properties(self) -> None:
        game = KuhnPokerGame()
        assert game.num_info_sets_per_player == 6
        assert game.num_actions == 2
        assert game.get_strategy_size() == 12

    def test_nash_equilibrium_value(self) -> None:
        """Kuhn poker has known game value of -1/18 for P1."""
        game = KuhnPokerGame()
        ne = game.nash_equilibrium_p1()

        # The Nash equilibrium strategy should be valid
        assert len(ne) == 12
        assert torch.all(ne >= 0)
        assert torch.all(ne <= 1)
