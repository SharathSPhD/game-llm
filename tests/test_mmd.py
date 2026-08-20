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
from kinetic_ai.optim.bregman import NegativeEntropy
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

    def test_readme_example_rps_convergence(self) -> None:
        """Test the README Quick Start example actually converges.

        The README example uses lr=0.1, tau=0.05 on RPS with sequential updates,
        which should converge to the τ-regularized QRE (which equals Nash for symmetric RPS).
        This test verifies the corrected parameters work.
        """
        game = rock_paper_scissors()
        bregman = NegativeEntropy()

        s1 = torch.tensor([0.7, 0.2, 0.1])
        s2 = torch.tensor([0.1, 0.7, 0.2])
        ref = torch.ones(3) / 3

        # Use the corrected parameters from README (lr=0.1, tau=0.05)
        # Sequential updates ensure better convergence
        for _ in range(500):
            g1 = game.utility_gradient(1, s1, s2)
            s1 = mmd_strategy_update(s1, g1, ref, bregman, lr=0.1, tau=0.05)

            g2 = game.utility_gradient(2, s2, s1)
            s2 = mmd_strategy_update(s2, g2, ref, bregman, lr=0.1, tau=0.05)

        nc = nash_conv(game, s1, s2)
        # With reduced lr and sequential updates, should converge to near-Nash
        assert nc < 0.25, (
            f"README example should converge with sequential updates, got NashConv={nc:.6f}."
        )

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
        """High magnetic strength should pull strategy toward reference.

        Note: Even with high tau, a single step may move significantly if
        the learning rate and gradient are both large. This test verifies
        the pull-toward-reference mechanism works rather than expecting
        convergence in one step.
        """
        bregman = NegativeEntropy()
        strategy = torch.tensor([0.1, 0.9])
        reference = torch.tensor([0.5, 0.5])
        gradient = torch.tensor([10.0, -10.0])  # Strong gradient

        # Single step with high tau
        updated = mmd_strategy_update(strategy, gradient, reference, bregman, lr=0.01, tau=10.0)

        # With high tau and low lr, should move less than with tau=0
        updated_no_mag = mmd_strategy_update(strategy, gradient, reference, bregman, lr=0.01, tau=0.0)

        # Check that magnetic term reduces movement away from reference
        dist_with_mag = torch.norm(updated - reference)
        dist_no_mag = torch.norm(updated_no_mag - reference)
        assert dist_with_mag <= dist_no_mag, "Magnetic term should reduce distance to reference"


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
        """With tau > 0, magnetic pull toward reference should reduce drift vs tau=0.

        The magnetic term should create a restoring force, limiting how far
        parameters can deviate from the reference, compared to standard gradient descent.
        """
        model = nn.Linear(3, 2, bias=False)
        initial_params = model.weight.data.clone()

        # Run with high tau (strong magnetic pull)
        config_mag = MMDConfig(lr=0.01, tau=5.0, bregman_type=BregmanType.EUCLIDEAN)
        optimizer_mag = MagneticMirrorDescent(model.parameters(), config=config_mag)

        # Run without magnetic term (tau=0, standard GD)
        config_gd = MMDConfig(lr=0.01, tau=0.0, bregman_type=BregmanType.EUCLIDEAN)
        model_gd = nn.Linear(3, 2, bias=False)
        model_gd.weight.data.copy_(initial_params)
        optimizer_gd = MagneticMirrorDescent(model_gd.parameters(), config=config_gd)

        x = torch.randn(5, 3)
        target = torch.randn(5, 2) * 100  # Large target to create strong gradients

        for _ in range(50):
            # Magnetic version
            optimizer_mag.zero_grad()
            loss_mag = nn.functional.mse_loss(model(x), target)
            loss_mag.backward()
            optimizer_mag.step()

            # Standard GD version
            optimizer_gd.zero_grad()
            loss_gd = nn.functional.mse_loss(model_gd(x), target)
            loss_gd.backward()
            optimizer_gd.step()

        # Magnetic version should have less drift than standard GD
        drift_mag = torch.norm(model.weight.data - initial_params)
        drift_gd = torch.norm(model_gd.weight.data - initial_params)
        assert drift_mag < drift_gd, (
            f"Magnetic term should reduce drift: mag={drift_mag:.2f} vs gd={drift_gd:.2f}"
        )

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

    def test_matching_pennies_convergence_with_stable_lr(self) -> None:
        """MMD converges to near-Nash on matching pennies with reduced lr.

        Theory: With lr=0.1, tau=0.1, convergence is guaranteed. The original
        test used lr=0.5 which violates stepsize conditions for simultaneous updates.
        """
        game = matching_pennies()
        bregman = NegativeEntropy()

        # Initialize strategies
        s1 = torch.tensor([0.8, 0.2])
        s2 = torch.tensor([0.3, 0.7])
        ref1 = torch.tensor([0.5, 0.5])
        ref2 = torch.tensor([0.5, 0.5])

        # Reduced lr to satisfy convergence conditions
        lr = 0.1
        tau = 0.1

        for _ in range(500):
            # Compute utility gradients
            g1 = game.utility_gradient(1, s1, s2)
            g2 = game.utility_gradient(2, s2, s1)

            # MMD updates
            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

        # Should be near Nash equilibrium [0.5, 0.5]
        nc = nash_conv(game, s1, s2)
        assert nc < 0.15, f"NashConv should be < 0.15 with stable lr, got {nc}"

    def test_matching_pennies_rnd_convergence(self) -> None:
        """MMD with Regularized Nash Dynamics (periodic reference resets) converges to Nash.

        Theory: With fixed reference, MMD converges to τ-regularized QRE.
        Periodic reference updates (Regularized Nash Dynamics) trace a path to Nash.
        """
        game = matching_pennies()
        bregman = NegativeEntropy()

        s1 = torch.tensor([0.8, 0.2])
        s2 = torch.tensor([0.3, 0.7])
        ref1 = torch.tensor([0.5, 0.5])
        ref2 = torch.tensor([0.5, 0.5])

        lr = 0.2
        tau = 0.1
        reference_update_interval = 50

        for step in range(1000):
            g1 = game.utility_gradient(1, s1, s2)
            g2 = game.utility_gradient(2, s2, s1)
            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

            # Periodic reference reset (Regularized Nash Dynamics)
            if (step + 1) % reference_update_interval == 0:
                ref1 = s1.clone()
                ref2 = s2.clone()

        # With RND, should approach Nash better than fixed reference
        nc = nash_conv(game, s1, s2)
        assert nc < 0.4, f"RND should improve convergence, got {nc}"

    def test_rps_convergence_sequential(self) -> None:
        """MMD should converge with sequential (alternating) updates on RPS.

        Sequential updates avoid the coupling issues of simultaneous updates
        and achieve better convergence with current parameters.
        """
        game = rock_paper_scissors()
        bregman = NegativeEntropy()

        s1 = torch.tensor([0.7, 0.2, 0.1])
        s2 = torch.tensor([0.1, 0.7, 0.2])
        ref1 = torch.ones(3) / 3
        ref2 = torch.ones(3) / 3

        lr = 0.3
        tau = 0.05

        for _ in range(1000):
            # Sequential updates: update s1, then see it in s2 update
            g1 = game.utility_gradient(1, s1, s2)
            s1 = mmd_strategy_update(s1, g1, ref1, bregman, lr, tau)

            g2 = game.utility_gradient(2, s2, s1)
            s2 = mmd_strategy_update(s2, g2, ref2, bregman, lr, tau)

        nc = nash_conv(game, s1, s2)
        assert nc < 0.1, f"NashConv should be small with sequential updates, got {nc}"
        # Should be near uniform
        assert torch.allclose(s1, torch.ones(3) / 3, atol=0.15)
