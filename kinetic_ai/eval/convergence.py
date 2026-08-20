"""Convergence diagnostics for game-theoretic algorithms.

Provides tools to measure, analyze, and visualize convergence behavior
of MMD and other equilibrium-finding algorithms.

Key capabilities:
    - Track convergence metrics over training steps
    - Verify linear convergence rate (expected for MMD → QRE)
    - Statistical confidence on convergence claims
    - Export metrics for visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor


@dataclass
class ConvergenceTracker:
    """Tracks convergence metrics over training steps.

    Attributes:
        exploitabilities: NashConv values at each logged step.
        residuals: Fixed-point residuals (for DEQ).
        step_indices: Step numbers for each measurement.
        utilities: Expected utilities at each step.
    """

    exploitabilities: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)
    utilities: list[float] = field(default_factory=list)
    wall_times: list[float] = field(default_factory=list)

    def log(
        self,
        step: int,
        exploitability: float | None = None,
        residual: float | None = None,
        utility: float | None = None,
        wall_time: float | None = None,
    ) -> None:
        """Record a convergence measurement."""
        self.step_indices.append(step)
        if exploitability is not None:
            self.exploitabilities.append(exploitability)
        if residual is not None:
            self.residuals.append(residual)
        if utility is not None:
            self.utilities.append(utility)
        if wall_time is not None:
            self.wall_times.append(wall_time)

    def estimate_convergence_rate(
        self, metric: str = "exploitability", window: int | None = None
    ) -> ConvergenceRateResult:
        """Estimate the convergence rate via log-linear regression.

        For linear convergence, log(metric) should decrease linearly:
            log(ε_t) ≈ log(ε_0) - r·t

        where r is the convergence rate.

        Args:
            metric: Which metric to analyze ("exploitability" or "residual").
            window: If specified, use only the last `window` data points.

        Returns:
            ConvergenceRateResult with rate estimate and diagnostics.
        """
        if metric == "exploitability":
            values = self.exploitabilities
        elif metric == "residual":
            values = self.residuals
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if len(values) < 3:
            return ConvergenceRateResult(
                rate=0.0, r_squared=0.0, is_linear=False, num_points=len(values)
            )

        if window is not None:
            values = values[-window:]
            steps = list(range(len(values)))
        else:
            steps = list(range(len(values)))

        # Filter out zeros/negatives for log
        valid = [(s, v) for s, v in zip(steps, values) if v > 0]
        if len(valid) < 3:
            return ConvergenceRateResult(
                rate=0.0, r_squared=0.0, is_linear=False, num_points=len(valid)
            )

        steps_arr = np.array([s for s, _ in valid])
        log_values = np.log(np.array([v for _, v in valid]))

        # Linear regression: log(ε) = a + b·t
        # Use numpy polyfit for simplicity
        coeffs = np.polyfit(steps_arr, log_values, 1)
        rate = -coeffs[0]  # Negative slope = convergence rate

        # R² for goodness of fit
        predicted = np.polyval(coeffs, steps_arr)
        ss_res = np.sum((log_values - predicted) ** 2)
        ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
        r_squared = 1.0 - (ss_res / max(ss_tot, 1e-12))

        return ConvergenceRateResult(
            rate=float(rate),
            r_squared=float(r_squared),
            is_linear=r_squared > 0.9 and rate > 0,
            num_points=len(valid),
        )

    def to_dict(self) -> dict[str, list[float]]:
        """Export all metrics as a dict (for JSON/CSV serialization)."""
        return {
            "step": [float(s) for s in self.step_indices],
            "exploitability": self.exploitabilities,
            "residual": self.residuals,
            "utility": self.utilities,
            "wall_time": self.wall_times,
        }


@dataclass
class ConvergenceRateResult:
    """Result of convergence rate estimation.

    Attributes:
        rate: Estimated convergence rate (positive = converging).
        r_squared: R² of the log-linear fit (1.0 = perfect linear convergence).
        is_linear: Whether convergence appears to be linear (R² > 0.9).
        num_points: Number of data points used.
    """

    rate: float
    r_squared: float
    is_linear: bool
    num_points: int


def verify_linear_convergence(
    tracker: ConvergenceTracker,
    metric: str = "exploitability",
    r_squared_threshold: float = 0.9,
    min_points: int = 10,
) -> bool:
    """Verify that an algorithm exhibits linear convergence.

    This is the key claim for MMD: it converges linearly to QRE.

    Args:
        tracker: ConvergenceTracker with logged metrics.
        metric: Which metric to check.
        r_squared_threshold: Minimum R² for "linear" classification.
        min_points: Minimum data points required.

    Returns:
        True if linear convergence is confirmed.
    """
    result = tracker.estimate_convergence_rate(metric)
    return result.is_linear and result.num_points >= min_points and result.rate > 0
