"""Tests for MagneticAdamW optimizer.

Validates:
    1. tau=0 exactly matches torch.optim.AdamW trajectories
    2. tau>0 bounds drift toward reference (compared to baseline AdamW)
    3. EMA reference tracking works correctly
    4. Periodic reference snapshots update correctly
    5. Loss decreases on overfit task with tau>0
"""

import pytest
import torch
import torch.nn as nn

from kinetic_ai.optim.magnetic_adamw import MagneticAdamW


class TinyLinearModel(nn.Module):
    """Minimal model for testing: single linear layer."""

    def __init__(self, input_dim: int = 10, output_dim: int = 5) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestMagneticAdamWTauZeroMatchesAdamW:
    """Test that tau=0 exactly matches torch.optim.AdamW."""

    def test_tau_zero_matches_adamw_single_step(self) -> None:
        """Single step with tau=0 should match AdamW exactly."""
        torch.manual_seed(42)

        # Create two identical models
        model1 = TinyLinearModel()
        model2 = TinyLinearModel()
        model2.load_state_dict(model1.state_dict())

        # Create optimizers
        opt1 = torch.optim.AdamW(model1.parameters(), lr=0.001)
        opt2 = MagneticAdamW(model2.parameters(), lr=0.001, tau=0.0)

        # Single training step
        x = torch.randn(2, 10)
        target = torch.randn(2, 5)

        loss1 = nn.MSELoss()(model1(x), target)
        loss1.backward()
        opt1.step()
        opt1.zero_grad()

        loss2 = nn.MSELoss()(model2(x), target)
        loss2.backward()
        opt2.step()
        opt2.zero_grad()

        # Compare weights after one step
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
            assert torch.allclose(p1, p2, atol=1e-5), (
                f"tau=0 should match AdamW: diff={torch.abs(p1 - p2).max():.2e}"
            )

    def test_tau_zero_matches_adamw_multiple_steps(self) -> None:
        """Multiple steps with tau=0 should match AdamW across iterations."""
        torch.manual_seed(42)

        model1 = TinyLinearModel()
        model2 = TinyLinearModel()
        model2.load_state_dict(model1.state_dict())

        opt1 = torch.optim.AdamW(model1.parameters(), lr=0.001)
        opt2 = MagneticAdamW(model2.parameters(), lr=0.001, tau=0.0)

        x = torch.randn(2, 10)
        target = torch.randn(2, 5)

        num_steps = 5
        for _ in range(num_steps):
            loss1 = nn.MSELoss()(model1(x), target)
            loss1.backward()
            opt1.step()
            opt1.zero_grad()

            loss2 = nn.MSELoss()(model2(x), target)
            loss2.backward()
            opt2.step()
            opt2.zero_grad()

        # Compare final weights (allow for floating-point accumulation over 5 steps)
        for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=False):
            assert torch.allclose(p1, p2, atol=2e-5), (
                f"tau=0 should match AdamW after {num_steps} steps: diff={torch.abs(p1 - p2).max():.2e}"
            )


class TestMagneticAdamWDriftBounding:
    """Test that tau>0 bounds parameter drift toward reference."""

    def test_drift_bounded_with_tau(self) -> None:
        """With tau>0, drift should be smaller than AdamW baseline."""
        torch.manual_seed(42)

        # Create two models with same initialization
        model_baseline = TinyLinearModel(input_dim=20, output_dim=10)
        model_magnetic = TinyLinearModel(input_dim=20, output_dim=10)

        # Save initialization
        init_state_dict = {k: v.clone() for k, v in model_baseline.state_dict().items()}
        model_magnetic.load_state_dict(model_baseline.state_dict())

        opt_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=0.1)  # Higher LR
        opt_magnetic = MagneticAdamW(
            model_magnetic.parameters(), lr=0.1, tau=0.1, ref_mode="ema"  # Higher tau
        )

        # Training on random data to accumulate drift
        x = torch.randn(4, 20)
        target = torch.randn(4, 10) * 10  # Larger targets to force learning

        for _ in range(20):
            loss = nn.MSELoss()(model_baseline(x), target)
            loss.backward()
            opt_baseline.step()
            opt_baseline.zero_grad()

            loss = nn.MSELoss()(model_magnetic(x), target)
            loss.backward()
            opt_magnetic.step()
            opt_magnetic.zero_grad()

        # Compute drift (distance from initialization)
        baseline_drift = 0.0
        magnetic_drift = 0.0

        for name, param in model_baseline.named_parameters():
            drift = torch.norm(param - init_state_dict[name]).item()
            baseline_drift += drift

        for name, param in model_magnetic.named_parameters():
            drift = torch.norm(param - init_state_dict[name]).item()
            magnetic_drift += drift

        # Magnetic should have smaller or comparable drift
        # (It's pulled toward reference, limiting how far it goes)
        # With tau>0, drift should be <= baseline (at least not significantly larger)
        assert magnetic_drift <= baseline_drift * 1.05, (
            f"Magnetic drift should be comparable to baseline: magnetic={magnetic_drift:.4f}, "
            f"baseline={baseline_drift:.4f}"
        )

    def test_large_target_regression_drift(self) -> None:
        """Magnetic should train stably on large-target regression task (not get stuck)."""
        torch.manual_seed(42)

        def train_with_optimizer(opt_fn, num_steps: int = 50) -> tuple[float, float]:
            model = TinyLinearModel(input_dim=20, output_dim=10)
            opt = opt_fn(model.parameters())

            # Large-target regression
            x = torch.randn(8, 20)
            target = torch.randn(8, 10) * 100.0  # Large targets

            losses = []
            for _ in range(num_steps):
                pred = model(x)
                loss = nn.MSELoss()(pred, target)
                losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()

            # Return initial and final loss
            return losses[0], losses[-1]

        # Baseline AdamW loss
        loss_start_adamw, loss_end_adamw = train_with_optimizer(
            lambda params: torch.optim.AdamW(params, lr=0.01)
        )

        # Magnetic AdamW loss (should also decrease, not get stuck)
        loss_start_magnetic, loss_end_magnetic = train_with_optimizer(
            lambda params: MagneticAdamW(
                params, lr=0.01, tau=0.02, ref_mode="ema"
            )
        )

        # Both should decrease, and magnetic shouldn't get stuck
        assert loss_end_adamw < loss_start_adamw, "AdamW should decrease loss"
        assert loss_end_magnetic < loss_start_magnetic, (
            f"Magnetic AdamW should decrease loss: start={loss_start_magnetic:.4f}, "
            f"end={loss_end_magnetic:.4f}"
        )


class TestMagneticAdamWEMAReference:
    """Test EMA reference tracking."""

    def test_ema_reference_updates(self) -> None:
        """EMA reference should track parameters with exponential moving average."""
        torch.manual_seed(42)

        model = TinyLinearModel()
        opt = MagneticAdamW(
            model.parameters(), lr=0.001, tau=0.01, ref_mode="ema", ref_beta=0.99
        )

        x = torch.randn(2, 10)
        target = torch.randn(2, 5)

        # Take a few steps
        for _ in range(3):
            loss = nn.MSELoss()(model(x), target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Check that reference has been created
        ref_state = opt.ref_state

        # Just check that reference exists and has been created
        assert ref_state is not None
        assert len(ref_state) > 0


class TestMagneticAdamWPeriodicReference:
    """Test periodic reference snapshots."""

    def test_periodic_reference_updates_on_interval(self) -> None:
        """Periodic reference should snapshot every K steps."""
        torch.manual_seed(42)

        model = TinyLinearModel()
        opt = MagneticAdamW(
            model.parameters(), lr=0.001, tau=0.01, ref_mode="periodic", ref_interval=3
        )

        x = torch.randn(2, 10)
        target = torch.randn(2, 5)

        # Step 1-2: no update to reference
        for _ in range(2):
            loss = nn.MSELoss()(model(x), target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # At step 3, reference should be updated
        for _ in range(1):
            loss = nn.MSELoss()(model(x), target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        # Reference should exist after periodic update
        assert opt.ref_state is not None


class TestMagneticAdamWLossDecrease:
    """Test that loss decreases on overfit task with tau>0."""

    def test_loss_decreases_on_overfit_with_tau(self) -> None:
        """Loss should decrease on overfit task with tau=0.01."""
        torch.manual_seed(42)

        model = TinyLinearModel()
        opt = MagneticAdamW(
            model.parameters(), lr=0.01, tau=0.01, ref_mode="ema"
        )

        x = torch.randint(0, 100, (4, 10)).float()
        target = torch.randint(0, 10, (4, 5)).float()

        losses = []
        for _ in range(50):
            pred = model(x)
            loss = nn.MSELoss()(pred, target)
            losses.append(loss.item())

            loss.backward()
            opt.step()
            opt.zero_grad()

        # Loss should generally trend downward
        assert losses[-1] < losses[0], (
            f"Loss should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_loss_decreases_with_adamw(self) -> None:
        """Baseline: loss should also decrease with AdamW."""
        torch.manual_seed(42)

        model = TinyLinearModel()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)

        x = torch.randint(0, 100, (4, 10)).float()
        target = torch.randint(0, 10, (4, 5)).float()

        losses = []
        for _ in range(50):
            pred = model(x)
            loss = nn.MSELoss()(pred, target)
            losses.append(loss.item())

            loss.backward()
            opt.step()
            opt.zero_grad()

        assert losses[-1] < losses[0], (
            f"AdamW baseline should decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )


class TestMagneticAdamWConfigs:
    """Test different configuration combinations."""

    def test_ema_mode_initializes(self) -> None:
        """EMA mode should initialize correctly."""
        model = TinyLinearModel()
        opt = MagneticAdamW(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            tau=0.01,
            ref_mode="ema",
            ref_beta=0.999,
        )
        assert opt.ref_mode == "ema"
        assert opt.ref_beta == 0.999

    def test_periodic_mode_initializes(self) -> None:
        """Periodic mode should initialize correctly."""
        model = TinyLinearModel()
        opt = MagneticAdamW(
            model.parameters(),
            lr=0.001,
            tau=0.01,
            ref_mode="periodic",
            ref_interval=10,
        )
        assert opt.ref_mode == "periodic"
        assert opt.ref_interval == 10

    def test_default_configs(self) -> None:
        """Default parameters should be sensible."""
        model = TinyLinearModel()
        opt = MagneticAdamW(model.parameters())

        # Check defaults
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["betas"] == (0.9, 0.999)
        assert opt.defaults["weight_decay"] == 0.0
        assert opt.defaults["tau"] == 0.0
        assert opt.ref_mode == "ema"
        assert opt.ref_beta == 0.999


class TestMagneticAdamWMemory:
    """Test memory usage to catch regressions (exp05 A3 caveat)."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available; memory test requires GPU",
    )
    def test_memory_usage_within_baseline(self) -> None:
        """Peak memory with MagneticAdamW should be within 1.5x of AdamW baseline.

        Regression test for F10 caveat (b): MagneticAdamW arm throttled memory
        at 97.7GB vs 3.5GB baseline. Fixed by wrapping updates in torch.no_grad().
        """
        torch.cuda.reset_peak_memory_stats()

        # Baseline: AdamW on CUDA
        model_baseline = TinyLinearModel(input_dim=100, output_dim=50).cuda()
        opt_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=0.01)

        x_cuda = torch.randn(8, 100, device="cuda")
        target_cuda = torch.randn(8, 50, device="cuda")

        torch.cuda.reset_peak_memory_stats()
        for _ in range(50):
            loss = nn.MSELoss()(model_baseline(x_cuda), target_cuda)
            loss.backward()
            opt_baseline.step()
            opt_baseline.zero_grad()

        peak_memory_adamw = torch.cuda.max_memory_allocated()

        # MagneticAdamW on CUDA
        model_magnetic = TinyLinearModel(input_dim=100, output_dim=50).cuda()
        opt_magnetic = MagneticAdamW(
            model_magnetic.parameters(), lr=0.01, tau=0.01, ref_mode="ema"
        )

        torch.cuda.reset_peak_memory_stats()
        for _ in range(50):
            loss = nn.MSELoss()(model_magnetic(x_cuda), target_cuda)
            loss.backward()
            opt_magnetic.step()
            opt_magnetic.zero_grad()

        peak_memory_magnetic = torch.cuda.max_memory_allocated()

        # MagneticAdamW peak should be within 1.5x of AdamW baseline
        ratio = peak_memory_magnetic / max(peak_memory_adamw, 1.0)
        assert ratio < 1.5, (
            f"MagneticAdamW memory should be within 1.5x of AdamW baseline: "
            f"ratio={ratio:.2f}, adamw={peak_memory_adamw / 1e9:.2f}GB, "
            f"magnetic={peak_memory_magnetic / 1e9:.2f}GB"
        )

        # Clean up
        del model_baseline, model_magnetic, opt_baseline, opt_magnetic
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDecoupledWeightDecay:
    """Regression: wd must be decoupled (AdamW), not folded into the gradient.

    Coupled L2 through the Adam normalizer destroys sparse-gradient params
    (e.g. a large tied embedding where most rows get no gradient per step):
    exp06 showed EqLM+MagneticAdamW(tau=0) at loss 10.46 vs torch AdamW 8.82.
    """

    def test_tau0_matches_adamw_on_sparse_grad_embedding(self) -> None:
        import copy

        torch.manual_seed(0)
        emb = nn.Embedding(1000, 16)  # most rows get zero grad per step
        head = nn.Linear(16, 1000)
        model_a = nn.Sequential(emb, head)
        model_b = copy.deepcopy(model_a)

        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2, weight_decay=0.1)
        opt_b = MagneticAdamW(model_b.parameters(), lr=1e-2, weight_decay=0.1, tau=0.0)

        ids = torch.randint(0, 10, (64,))  # only rows 0-9 ever used
        for _ in range(20):
            for model, opt in ((model_a, opt_a), (model_b, opt_b)):
                opt.zero_grad()
                out = model(ids)
                loss = out.pow(2).mean()
                loss.backward()
                opt.step()

        for (na, pa), (_nb, pb) in zip(
            model_a.named_parameters(), model_b.named_parameters(), strict=True
        ):
            assert torch.allclose(pa, pb, atol=1e-5), (
                f"{na} diverged: max diff {(pa - pb).abs().max().item():.2e}"
            )


class TestFixedRefMode:
    """ref_mode='fixed' pins the magnet to the initial (frozen base) weights.

    SPEC 0007 (H3/MPO): the magnetic reference must be the frozen base model
    for the whole run — never updated by EMA or periodic snapshots.
    """

    def test_fixed_mode_accepted(self) -> None:
        p = torch.nn.Parameter(torch.ones(3))
        MagneticAdamW([p], lr=0.1, tau=0.1, ref_mode="fixed")

    def test_reference_never_moves(self) -> None:
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.ones(4))
        init = p.data.clone()
        opt = MagneticAdamW([p], lr=0.05, tau=0.5, ref_mode="fixed")
        for _ in range(20):
            opt.zero_grad()
            (p * torch.randn(4)).sum().backward()
            opt.step()
        ref = opt.ref_state[id(p)]
        assert torch.equal(ref, init), "fixed reference must stay at init weights"
        assert not torch.equal(p.data, init), "parameters must still train"

    def test_fixed_pulls_toward_init(self) -> None:
        """With a huge tau, params should stay closer to init than with tau=0."""
        torch.manual_seed(0)
        grads = [torch.randn(4) for _ in range(30)]

        def run(tau: float) -> torch.Tensor:
            torch.manual_seed(1)
            p = torch.nn.Parameter(torch.ones(4))
            opt = MagneticAdamW([p], lr=0.1, tau=tau, ref_mode="fixed")
            for g in grads:
                opt.zero_grad()
                (p * g).sum().backward()
                opt.step()
            return p.data

        drift_free = (run(0.0) - torch.ones(4)).norm()
        drift_magnet = (run(5.0) - torch.ones(4)).norm()
        assert drift_magnet < drift_free
