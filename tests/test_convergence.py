"""Tests for convergence diagnostics and statistical testing.

Validates:
    1. ConvergenceTracker correctly logs and analyzes metrics
    2. Linear convergence detection works on known exponential decay
    3. Bootstrap CI has correct coverage
    4. Wilcoxon test detects significant differences
    5. Holm-Bonferroni correction controls FWER
"""

import numpy as np
import pytest
import torch

from kinetic_ai.eval.convergence import (
    ConvergenceTracker,
    verify_linear_convergence,
)
from kinetic_ai.eval.statistical import (
    bootstrap_ci,
    holm_bonferroni_correction,
    paired_bootstrap_test,
    wilcoxon_signed_rank,
)


class TestConvergenceTracker:
    """Tests for the convergence tracking system."""

    def test_logging(self) -> None:
        """Should correctly store logged metrics."""
        tracker = ConvergenceTracker()
        tracker.log(step=0, exploitability=1.0)
        tracker.log(step=10, exploitability=0.5)
        tracker.log(step=20, exploitability=0.25)

        assert len(tracker.exploitabilities) == 3
        assert len(tracker.step_indices) == 3
        assert tracker.step_indices == [0, 10, 20]

    def test_detect_linear_convergence(self) -> None:
        """Should detect linear convergence from exponential decay."""
        tracker = ConvergenceTracker()

        # Simulate exponential decay: ε_t = exp(-0.1 * t)
        for t in range(100):
            tracker.log(step=t, exploitability=np.exp(-0.1 * t))

        result = tracker.estimate_convergence_rate("exploitability")
        assert result.is_linear, f"Should detect linear convergence, R²={result.r_squared}"
        assert abs(result.rate - 0.1) < 0.02, f"Rate should be ~0.1, got {result.rate}"

    def test_detect_non_convergence(self) -> None:
        """Should NOT detect linear convergence from oscillating data."""
        tracker = ConvergenceTracker()
        for t in range(50):
            tracker.log(step=t, exploitability=abs(np.sin(t * 0.5)) + 0.1)

        result = tracker.estimate_convergence_rate("exploitability")
        assert not result.is_linear, "Should not detect linear convergence in oscillating data"

    def test_verify_linear_convergence_function(self) -> None:
        """verify_linear_convergence should be True for exponential decay."""
        tracker = ConvergenceTracker()
        for t in range(50):
            tracker.log(step=t, exploitability=np.exp(-0.05 * t))

        assert verify_linear_convergence(tracker, min_points=10)

    def test_to_dict(self) -> None:
        """Should export as a serializable dict."""
        tracker = ConvergenceTracker()
        tracker.log(step=0, exploitability=1.0, residual=0.5)
        tracker.log(step=1, exploitability=0.5, residual=0.25)

        d = tracker.to_dict()
        assert "step" in d
        assert "exploitability" in d
        assert len(d["step"]) == 2

    def test_window_parameter(self) -> None:
        """Should allow analyzing only recent data."""
        tracker = ConvergenceTracker()

        # First half: no convergence; second half: convergence
        for t in range(50):
            tracker.log(step=t, exploitability=1.0)  # Flat
        for t in range(50, 100):
            tracker.log(step=t, exploitability=np.exp(-0.1 * (t - 50)))

        result_all = tracker.estimate_convergence_rate("exploitability")
        result_recent = tracker.estimate_convergence_rate("exploitability", window=50)

        # Recent window should show better linear fit
        assert result_recent.r_squared > result_all.r_squared


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_ci_contains_true_mean(self) -> None:
        """CI should contain the true mean with high probability."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        data = rng.normal(true_mean, 1.0, size=100)

        result = bootstrap_ci(data, confidence=0.95)
        assert result.ci_lower <= true_mean <= result.ci_upper

    def test_wider_ci_for_lower_confidence(self) -> None:
        """99% CI should be wider than 90% CI."""
        data = np.random.default_rng(42).normal(0, 1, 50)

        ci_90 = bootstrap_ci(data, confidence=0.90)
        ci_99 = bootstrap_ci(data, confidence=0.99)

        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90

    def test_empty_data(self) -> None:
        """Should handle empty data gracefully."""
        result = bootstrap_ci([])
        assert np.isnan(result.mean)


class TestWilcoxon:
    """Tests for Wilcoxon signed-rank test."""

    def test_identical_methods_not_significant(self) -> None:
        """Same data should not show significance."""
        data = [0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.17, 0.19, 0.14, 0.16]
        result = wilcoxon_signed_rank(data, data)
        assert not result.significant

    def test_clearly_different_methods(self) -> None:
        """Clearly different methods should be significant."""
        method_a = [0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.17, 0.19, 0.14, 0.16]
        method_b = [0.5, 0.6, 0.55, 0.58, 0.52, 0.62, 0.57, 0.59, 0.54, 0.56]

        result = wilcoxon_signed_rank(method_a, method_b)
        assert result.significant, f"Should be significant, p={result.p_value}"

    def test_effect_size_bounded(self) -> None:
        """Effect size should be non-negative."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.5, 2.5, 3.5, 4.5, 5.5]
        result = wilcoxon_signed_rank(a, b)
        assert result.effect_size >= 0


class TestPairedBootstrap:
    """Tests for paired bootstrap test."""

    def test_identical_not_significant(self) -> None:
        """Identical methods should not be significant."""
        data = [0.1, 0.2, 0.15, 0.18, 0.12]
        result = paired_bootstrap_test(data, data)
        assert not result.significant

    def test_different_significant(self) -> None:
        """Clearly different methods should be significant."""
        a = [0.1, 0.2, 0.15, 0.18, 0.12, 0.22, 0.17, 0.19, 0.14, 0.16]
        b = [0.5, 0.6, 0.55, 0.58, 0.52, 0.62, 0.57, 0.59, 0.54, 0.56]
        result = paired_bootstrap_test(a, b)
        assert result.significant


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni multiple testing correction."""

    def test_single_significant(self) -> None:
        """Single significant p-value should remain significant."""
        assert holm_bonferroni_correction([0.01]) == [True]

    def test_single_not_significant(self) -> None:
        """Single non-significant p-value should remain not significant."""
        assert holm_bonferroni_correction([0.1]) == [False]

    def test_correction_is_conservative(self) -> None:
        """After correction, fewer tests should be significant than without."""
        p_values = [0.01, 0.03, 0.04, 0.06]
        uncorrected = [p < 0.05 for p in p_values]
        corrected = holm_bonferroni_correction(p_values, alpha=0.05)
        assert sum(corrected) <= sum(uncorrected)

    def test_empty(self) -> None:
        """Empty input should return empty output."""
        assert holm_bonferroni_correction([]) == []
