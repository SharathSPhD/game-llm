"""Statistical testing framework for game-theoretic experiments.

Provides rigorous statistical tests for comparing algorithms:
    - Bootstrap confidence intervals for exploitability/convergence
    - Wilcoxon signed-rank test for paired method comparisons
    - Convergence rate comparison with confidence intervals
    - Multiple testing correction (Bonferroni, Holm)

These are essential for the paper: every claim about one method being
"better" than another must be backed by a statistical test.

References:
    [1] Efron & Tibshirani "An Introduction to the Bootstrap" (1993)
    [2] Wilcoxon "Individual Comparisons by Ranking Methods" (1945)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result.

    Attributes:
        mean: Sample mean.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        confidence: Confidence level (e.g., 0.95).
        n_bootstrap: Number of bootstrap samples used.
        std_error: Standard error of the mean.
    """

    mean: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_bootstrap: int
    std_error: float


@dataclass
class PairedTestResult:
    """Result of a paired statistical test.

    Attributes:
        statistic: Test statistic value.
        p_value: P-value (probability of observing this result under H0).
        significant: Whether the test is significant at the given alpha.
        alpha: Significance level.
        effect_size: Estimated effect size (e.g., Cohen's d).
        method_a_mean: Mean of method A.
        method_b_mean: Mean of method B.
        test_name: Name of the test used.
    """

    statistic: float
    p_value: float
    significant: bool
    alpha: float
    effect_size: float
    method_a_mean: float
    method_b_mean: float
    test_name: str


def bootstrap_ci(
    data: NDArray[np.floating] | list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for the mean.

    Uses the percentile method for CI construction.

    Args:
        data: 1D array of observations.
        confidence: Confidence level (0 to 1).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        BootstrapCI with mean, CI bounds, and diagnostics.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    if n == 0:
        return BootstrapCI(
            mean=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            std_error=float("nan"),
        )

    rng = np.random.default_rng(seed)

    # Bootstrap resampling
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        mean=float(np.mean(data)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        std_error=float(np.std(boot_means)),
    )


def wilcoxon_signed_rank(
    method_a: NDArray[np.floating] | list[float],
    method_b: NDArray[np.floating] | list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> PairedTestResult:
    """Wilcoxon signed-rank test for paired comparisons.

    Non-parametric test that doesn't assume normal distributions.
    Tests whether the median of differences is zero.

    Args:
        method_a: Observations from method A (e.g., final exploitability
            across 10 seeds).
        method_b: Observations from method B.
        alpha: Significance level.
        alternative: "two-sided", "greater", or "less".

    Returns:
        PairedTestResult with test statistics and significance.
    """
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)

    assert len(a) == len(b), "Paired test requires equal-length arrays"
    n = len(a)

    differences = a - b

    # Remove zero differences
    nonzero_mask = differences != 0
    d = differences[nonzero_mask]
    n_eff = len(d)

    if n_eff == 0:
        return PairedTestResult(
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            effect_size=0.0,
            method_a_mean=float(np.mean(a)),
            method_b_mean=float(np.mean(b)),
            test_name="wilcoxon_signed_rank",
        )

    # Rank the absolute differences
    abs_d = np.abs(d)
    ranks = _rank(abs_d)

    # Signed ranks
    positive_ranks = ranks[d > 0].sum()
    negative_ranks = ranks[d < 0].sum()

    # Test statistic: W = min(W+, W-)
    w_plus = positive_ranks
    w_minus = negative_ranks

    if alternative == "two-sided":
        w_stat = min(w_plus, w_minus)
    elif alternative == "greater":
        w_stat = w_minus  # Small W- means A > B
    else:  # "less"
        w_stat = w_plus  # Small W+ means A < B

    # Normal approximation for p-value (valid for n >= 10)
    mean_w = n_eff * (n_eff + 1) / 4
    std_w = np.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)

    if std_w > 0:
        z = (w_stat - mean_w) / std_w
        if alternative == "two-sided":
            p_value = 2 * _normal_cdf(-abs(z))
        else:
            p_value = _normal_cdf(z)
    else:
        p_value = 1.0

    # Effect size: r = Z / sqrt(N)
    effect_size = abs(z) / np.sqrt(n) if std_w > 0 else 0.0

    return PairedTestResult(
        statistic=float(w_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(effect_size),
        method_a_mean=float(np.mean(a)),
        method_b_mean=float(np.mean(b)),
        test_name="wilcoxon_signed_rank",
    )


def paired_bootstrap_test(
    method_a: NDArray[np.floating] | list[float],
    method_b: NDArray[np.floating] | list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int | None = 42,
) -> PairedTestResult:
    """Paired bootstrap test for method comparison.

    Tests whether the mean difference between methods is significantly
    different from zero using bootstrap resampling.

    Args:
        method_a: Observations from method A.
        method_b: Observations from method B.
        n_bootstrap: Number of bootstrap samples.
        alpha: Significance level.
        seed: Random seed.

    Returns:
        PairedTestResult with bootstrap-based significance.
    """
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)

    assert len(a) == len(b), "Paired test requires equal-length arrays"
    n = len(a)

    differences = a - b
    observed_mean_diff = np.mean(differences)

    rng = np.random.default_rng(seed)

    # Bootstrap the differences
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(differences, size=n, replace=True)
        boot_diffs[i] = np.mean(sample)

    # Two-sided p-value: proportion of bootstrap samples on the other side of 0
    centered = boot_diffs - observed_mean_diff  # Center at 0 under H0
    p_value = float(np.mean(np.abs(centered) >= abs(observed_mean_diff)))

    # Effect size: Cohen's d
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    cohens_d = abs(observed_mean_diff) / max(pooled_std, 1e-12)

    return PairedTestResult(
        statistic=float(observed_mean_diff),
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(cohens_d),
        method_a_mean=float(np.mean(a)),
        method_b_mean=float(np.mean(b)),
        test_name="paired_bootstrap",
    )


def holm_bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[bool]:
    """Holm-Bonferroni method for multiple testing correction.

    Controls the family-wise error rate (FWER) when running multiple
    statistical tests simultaneously.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Desired FWER.

    Returns:
        List of booleans indicating significance after correction.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    significant = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            # Once we fail to reject, stop (step-down procedure)
            break

    return significant


# --- Internal helpers ---


def _rank(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute ranks with average tie-breaking."""
    sorted_idx = np.argsort(data)
    ranks = np.empty_like(data, dtype=np.float64)
    ranks[sorted_idx] = np.arange(1, len(data) + 1, dtype=np.float64)

    # Handle ties: assign average rank
    unique_vals = np.unique(data)
    for val in unique_vals:
        mask = data == val
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    return ranks


def _normal_cdf(z: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    import math

    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
