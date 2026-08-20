"""Tests for convergence diagnostics and statistical testing.

Validates:
    1. ConvergenceTracker correctly logs and analyzes metrics
    2. Linear convergence detection works on known exponential decay
    3. Bootstrap CI has correct coverage
    4. Wilcoxon test detects significant differences
    5. Holm-Bonferroni correction controls FWER
"""

import numpy as np

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

    def test_rate_within_theoretical_bounds(self) -> None:
        """Verify convergence rate matches expected theoretical O(η·τ) prediction.

        This test ensures that linear convergence is not just exponential decay,
        but decay at the theoretically predicted rate. A convergence tracker with
        exponential decay at rate r should validate only if r matches theory.

        Theory: For mirror descent with magnetic regularization, the convergence
        rate should be O(η·τ) where η is stepsize and τ is magnetic strength.
        """
        # Scenario: theoretical rate = η·τ = 0.3 * 0.01 = 0.003
        eta = 0.3
        tau = 0.01
        theory_rate = eta * tau
        theory_upper_bound = theory_rate * 10  # Allow 10x variation

        # Case 1: Empirical rate matches theory (should validate)
        tracker_good = ConvergenceTracker()
        for t in range(100):
            tracker_good.log(step=t, exploitability=np.exp(-theory_rate * t))

        result_good = tracker_good.estimate_convergence_rate("exploitability")
        # Should detect exponential decay
        assert result_good.r_squared > 0.95, "Should detect exponential decay"
        # Rate should be close to theory
        assert abs(result_good.rate - theory_rate) < theory_rate * 0.1, (
            f"Rate {result_good.rate} should be close to theory {theory_rate}"
        )

        # Case 2: Exponential decay but rate way too fast (should fail validation)
        tracker_bad = ConvergenceTracker()
        empirical_rate_wrong = 0.5  # 166x too fast
        for t in range(100):
            tracker_bad.log(step=t, exploitability=np.exp(-empirical_rate_wrong * t))

        result_bad = tracker_bad.estimate_convergence_rate("exploitability")
        # Should detect exponential decay (R² is good)
        assert result_bad.r_squared > 0.95, "Should detect exponential decay"
        # But rate is way outside theory bounds
        assert result_bad.rate > theory_upper_bound, (
            f"Rate {result_bad.rate} should be > theoretical upper bound {theory_upper_bound}"
        )

        # Case 3: Too slow (below theory) also fails
        tracker_slow = ConvergenceTracker()
        empirical_rate_slow = 0.00001  # 300x too slow
        for t in range(100):
            tracker_slow.log(step=t, exploitability=np.exp(-empirical_rate_slow * t))

        result_slow = tracker_slow.estimate_convergence_rate("exploitability")
        assert result_slow.r_squared > 0.95, "Should detect exponential decay"
        assert result_slow.rate < theory_rate / 100, (
            f"Rate {result_slow.rate} should be << theoretical lower bound {theory_rate}"
        )


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_bootstrap_ci_coverage_probability(self) -> None:
        """Bootstrap CI should achieve ~95% coverage over repeated experiments.

        NOT a guarantee that any single CI contains the true mean, but ~95% of
        CIs should across repeated sampling. This validates the probabilistic
        guarantee that defines a 95% confidence interval.
        """
        true_mean = 5.0
        n_trials = 100
        coverage_count = 0

        # Repeat the experiment with different random samples
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            data = rng.normal(true_mean, 1.0, size=100)
            result = bootstrap_ci(data, confidence=0.95, seed=seed)

            if result.ci_lower <= true_mean <= result.ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # Allow statistical margin: 95% ± 1.96*sqrt(0.95*0.05/100) ≈ ±4.3%
        # Expect 88-100 successes out of 100 trials
        assert 0.88 <= coverage_rate <= 1.0, (
            f"Coverage rate {coverage_rate:.1%} outside expected range [88%, 100%]. "
            f"95% CI should contain true mean in ~95 of 100 experiments."
        )

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

    def test_wilcoxon_small_n_matches_scipy_exact(self) -> None:
        """Wilcoxon p-value must match scipy exact for all n, especially n < 10."""
        import scipy.stats

        # Test case from defect report: n=5 with uniform differences
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a + 0.5

        result = wilcoxon_signed_rank(a, b, alpha=0.05)
        scipy_exact = scipy.stats.wilcoxon(a, b, alternative="two-sided", method="exact")

        # Core assertion: p-values must match (allowing small numerical tolerance)
        np.testing.assert_allclose(
            result.p_value,
            scipy_exact.pvalue,
            rtol=1e-5,
            err_msg=f"n=5 approximation error too large: {abs(result.p_value - scipy_exact.pvalue)}",
        )

        # Critical: significance decision must match
        assert result.significant == (scipy_exact.pvalue < result.alpha), (
            f"n=5 disagreement on significance: our p={result.p_value:.6f}, "
            f"scipy p={scipy_exact.pvalue:.6f}"
        )

        # Test boundary cases n=5,7,9,10,15,20
        for n in [5, 7, 9, 10, 15, 20]:
            a = np.arange(1.0, n + 1.0)
            b = a + 0.5  # Uniform difference

            result = wilcoxon_signed_rank(a, b, alpha=0.05)
            scipy_exact = scipy.stats.wilcoxon(a, b, alternative="two-sided", method="exact")

            # For all n, p-values and significance must match.
            # Use looser tolerance for extreme p-values where relative error is large.
            if scipy_exact.pvalue < 1e-4:
                # For very small p-values, use absolute tolerance
                np.testing.assert_allclose(
                    result.p_value,
                    scipy_exact.pvalue,
                    atol=1e-3,
                    err_msg=f"n={n}: error = {abs(result.p_value - scipy_exact.pvalue)}",
                )
            else:
                np.testing.assert_allclose(
                    result.p_value,
                    scipy_exact.pvalue,
                    rtol=5e-3,
                    err_msg=f"n={n}: error = {abs(result.p_value - scipy_exact.pvalue)}",
                )
            assert result.significant == (scipy_exact.pvalue < 0.05), (
                f"n={n}: significance mismatch (ours={result.significant}, "
                f"scipy={scipy_exact.pvalue < 0.05})"
            )


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
