"""Tests for Magnetic Mirror Descent optimizer.

Validates:
    1. Strategy-space MMD converges to QRE on known games
    2. Parameter-space MMD properly applies Bregman geometry
    3. Reference update mechanism (Regularized Nash Dynamics) works
    4. Functional mmd_strategy_update produces correct updates
"""

import pytest
import torch
import torch.nn as nn

from kinetic_ai.config import BregmanType, MMDConfig
from kinetic_ai.games.payoff import matching_pennies, rock_paper_scissors
from kinetic_ai.games.qre import nash_conv
from kinetic_ai.optim.bregman import Euclidean, NegativeEntropy
from kinetic_ai.optim.mmd import MagneticMirrorDescent, mmd_strategy_update


class TestMMDStrategyUpdate:
    """Tests for the functional strategy-space MMD update."""

    def test_stays_on_simplex(self) -> None:
        """After update, the strategy should remain a valid distribution."""
        bregman = NegativeEntropy()
        strategy = torch.softmax(torch.randn(5), dim=-1)
        reference = torch.softmax(torch.randn(5), dim=-1)
        gradient = torch.randn(5)

        updated = mmd_strategy_update(strategy, gradient, reference, bregman, lr=0.1, tau=0.1)

        assert torch.all(updated >= 0), "Negative probabilities after update"
        assert torch.isclose(updated.sum(), torch.tensor(1.0), atol=1e-5), "Not normalized"

    def test_zero_gradient_pulls_toward_reference(self) -> None:
        """With zero gradient, MMD should pull strategy toward the reference."""
        bregman = NegativeEntropy()
        strategy = torch.tensor([0.1, 0.9])
        reference = torch.tensor([0.5, 0.5])
        gradient = torch.zeros(2)

        updated = mmd_strategy_update(strategy, gradient, reference, bregman, lr=0.1, tau=1.0)

        # Should be closer to reference than before
        dist_before = torch.norm(strategy - reference)
        dist_after = torch.norm(updated - reference)
        assert dist_after < dist_before, "Should move toward reference"

    def test_high_tau_stays_near_reference(self) -> None:
        """High magnetic strength should keep strategy close to reference."""
        bregman = NegativeEntropy()
        strategy = torch.tensor([0.1, 0.9])
        reference = torch.tensor([0.5, 0.5])
        gradient = torch.tensor([10.0, -10.0])  # Strong gradient

        updated = mmd_strategy_update(strategy, gradient, reference, bregman, lr=0.01, tau=10.0)

        # Should still be close to reference despite strong gradient
        assert torch.norm(updated - reference) < 0.3, "High tau should resist gradient"


class TestMMDOptimizer:
    """Tests for the parameter-space MMD optimizer (nn.Module training)."""

    def test_basic_optimization(self) -> None:
        """MMD should optimize a simple loss function."""
        model = nn.Linear(3, 2)
        config = MMDConfig(lr=0.01, tau=0.0, bregman_type=BregmanType.EUCLIDEAN)
        optimizer = MagneticMirrorDescent(model.parameters(), config=config)

        x = torch.randn(5, 3)
        target = torch.randn(5, 2)

        initial_loss = nn.functional.mse_loss(model(x), target).item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()

        final_loss = nn.functional.mse_loss(model(x), target).item()
        assert final_loss < initial_loss, "Loss should decrease"

    def test_reference_pull(self) -> None:
        """With tau > 0, parameters should not deviate too far from reference."""
        model = nn.Linear(3, 2, bias=False)
        initial_params = model.weight.data.clone()

        # High tau = strong pull
        config = MMDConfig(lr=0.01, tau=5.0, bregman_type=BregmanType.EUCLIDEAN)
        optimizer = MagneticMirrorDescent(model.parameters(), config=config)

        x = torch.randn(5, 3)
        target = torch.randn(5, 2) * 100  # Large target to create strong gradients

        for _ in range(50):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()

        # Should not have moved too far from initial
        drift = torch.norm(model.weight.data - initial_params)
        assert drift < 5.0, f"High tau should limit drift, got {drift}"

    def test_reference_update(self) -> None:
        """Periodic reference updates should update the magnet."""
        model = nn.Linear(3, 2, bias=False)

        config = MMDConfig(
            lr=0.01, tau=0.1, bregman_type=BregmanType.EUCLIDEAN, reference_update_interval=5
        )
        optimizer = MagneticMirrorDescent(model.parameters(), config=config)

        initial_ref = optimizer._reference_state[0].clone()

        x = torch.randn(5, 3)
        target = torch.randn(5, 2)

        for _ in range(10):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()

        # After 10 steps with interval=5, reference should have been updated twice
        updated_ref = optimizer._reference_state[0]
        assert not torch.allclose(initial_ref, updated_ref), "Reference should have been updated"


@pytest.mark.slow
class TestMMDConvergence:
    """Integration tests: MMD convergence on known games."""

    def test_matching_pennies_convergence(self) -> None:
        """MMD should converge to near-Nash on matching pennies."""
        game = matching_pennies()
        bregman = NegativeEntropy()

        # Initialize strategies
        s1 = torch.tensor([0.8, 0.2])
        s2 = torch.tensor([0.3, 0.7])
        ref1 = torch.tensor([0.5, 0.5])
        ref2 = torch.tensor([0.5, 0.5])

        lr = 0.5
        tau = 0.1

        for _ in range(500):
            # Compute utility gradients
            g1 = game.utility_gradient(1, s1, s2)
            g2 = game.utility_gradient(2, s2, s1)

            # MMD updates
            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

        # Should be near uniform (Nash equilibrium)
        nc = nash_conv(game, s1, s2)
        assert nc < 0.1, f"NashConv should be small, got {nc}"

    def test_rps_convergence(self) -> None:
        """MMD should converge to near-Nash on Rock-Paper-Scissors."""
        game = rock_paper_scissors()
        bregman = NegativeEntropy()

        s1 = torch.softmax(torch.randn(3), dim=-1)
        s2 = torch.softmax(torch.randn(3), dim=-1)
        ref1 = torch.ones(3) / 3
        ref2 = torch.ones(3) / 3

        lr = 0.3
        tau = 0.05

        for _ in range(1000):
            g1 = game.utility_gradient(1, s1, s2)
            g2 = game.utility_gradient(2, s2, s1)

            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

        nc = nash_conv(game, s1, s2)
        assert nc < 0.15, f"NashConv should be small on RPS, got {nc}"
        # Should be near uniform
        assert torch.allclose(s1, torch.ones(3) / 3, atol=0.1)
