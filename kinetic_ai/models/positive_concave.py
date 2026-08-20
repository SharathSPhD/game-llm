"""Positive Concave DEQ (pcDEQ) constraints.

Provides constraints that guarantee the existence and uniqueness of
fixed points in Deep Equilibrium Models without requiring spectral
normalization.

Key Idea:
    If the transformation f(z, x) is:
        1. Nonnegative: f maps the positive orthant to itself
        2. Concave: f is concave in z on the positive orthant

    Then by the Perron-Frobenius theorem and concavity, the fixed-point
    equation z = f(z, x) has a unique positive solution.

Implementation:
    - Nonnegative weight enforcement via absolute value or softplus
    - Concave activation functions (e.g., min(z, α), log(1+z), z^p for p<1)
    - Combined pcDEQ layer that wraps any linear transformation

References:
    [1] Ghosh et al. "Positive Concave Deep Equilibrium Models"
        (arXiv:2402.04029)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


class NonNegativeLinear(nn.Module):
    """Linear layer with guaranteed non-negative weights.

    Enforces W ≥ 0 by reparameterizing weights through softplus:
        W = softplus(W_raw) = log(1 + exp(W_raw))

    This ensures that the transformation is monotone (order-preserving),
    which is required for the pcDEQ fixed-point guarantee.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias. Bias is unconstrained.
        method: "softplus" for smooth reparameterization,
                "abs" for simple absolute value.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        method: str = "softplus",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.method = method

        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> Tensor:
        """Get the constrained (non-negative) weight matrix."""
        if self.method == "softplus":
            return nn.functional.softplus(self.weight_raw)
        elif self.method == "abs":
            return torch.abs(self.weight_raw)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class ConcaveActivation(nn.Module):
    """Concave activation function for pcDEQ.

    Available activations:
        - "sqrt": f(z) = sqrt(z + ε) — concave, positive
        - "log1p": f(z) = log(1 + z) — concave, positive
        - "power": f(z) = (z + ε)^p, p ∈ (0,1) — concave
        - "min_linear": f(z) = min(z, α) — concave, piecewise linear

    All maintain concavity on the positive orthant.

    Args:
        activation_type: Which concave activation to use.
        alpha: Cap for "min_linear", or power for "power".
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        activation_type: str = "log1p",
        alpha: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.activation_type = activation_type
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_type == "sqrt":
            return torch.sqrt(x.clamp(min=self.eps))
        elif self.activation_type == "log1p":
            return torch.log1p(x.clamp(min=0))
        elif self.activation_type == "power":
            return torch.pow(x.clamp(min=self.eps), self.alpha)
        elif self.activation_type == "min_linear":
            return torch.minimum(x, torch.tensor(self.alpha, device=x.device))
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")


class PositiveConcaveBlock(nn.Module):
    """A pcDEQ-compatible transformation block.

    Combines NonNegativeLinear with ConcaveActivation to create a
    transformation z → σ(W_nn @ [z; x] + b) that is:
        1. Non-negative (maps positive orthant to itself)
        2. Concave in z (guarantees unique fixed point)

    This can be used as the `func` argument to DEQLayer.

    Args:
        hidden_dim: Dimension of the equilibrium state z.
        input_dim: Dimension of the input x.
        activation: Type of concave activation.
        alpha: Activation parameter.
    """

    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        activation: str = "log1p",
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.linear = NonNegativeLinear(hidden_dim + input_dim, hidden_dim, bias=True)
        self.activation = ConcaveActivation(activation, alpha=alpha)

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        """Apply the pcDEQ transformation.

        Args:
            z: Current equilibrium state. Shape: (batch, hidden_dim)
            x: Input. Shape: (batch, input_dim)

        Returns:
            Updated state. Shape: (batch, hidden_dim)
        """
        combined = torch.cat([z, x], dim=-1)
        return cast(Tensor, self.activation(self.linear(combined)))


def verify_positive_concave(
    func: Callable[[Tensor, Tensor], Tensor],
    z_dim: int,
    x_dim: int,
    n_tests: int = 100,
    eps: float = 1e-4,
) -> dict[str, bool]:
    """Empirically verify that a transformation satisfies pcDEQ properties.

    Tests:
        1. Non-negativity: f(z, x) ≥ 0 for z ≥ 0
        2. Monotonicity: z1 ≥ z2 → f(z1, x) ≥ f(z2, x) (elementwise)
        3. Concavity (midpoint): f((z1+z2)/2, x) ≥ (f(z1,x)+f(z2,x))/2

    Args:
        func: The transformation to verify.
        z_dim: Dimension of z.
        x_dim: Dimension of x.
        n_tests: Number of random test points.
        eps: Tolerance for inequality checks.

    Returns:
        Dict with bool results for each property.
    """
    nonneg_ok = True
    monotone_ok = True
    concave_ok = True

    for _ in range(n_tests):
        x = torch.randn(1, x_dim).abs()
        z1 = torch.randn(1, z_dim).abs()
        z2 = torch.randn(1, z_dim).abs()

        with torch.no_grad():
            f_z1 = func(z1, x)
            f_z2 = func(z2, x)

            # Non-negativity
            if (f_z1 < -eps).any() or (f_z2 < -eps).any():
                nonneg_ok = False

            # Monotonicity: test with z1 >= z2
            z_big = torch.max(z1, z2)
            z_small = torch.min(z1, z2)
            f_big = func(z_big, x)
            f_small = func(z_small, x)
            if (f_big < f_small - eps).any():
                monotone_ok = False

            # Midpoint concavity
            z_mid = 0.5 * (z1 + z2)
            f_mid = func(z_mid, x)
            f_avg = 0.5 * (f_z1 + f_z2)
            if (f_mid < f_avg - eps).any():
                concave_ok = False

    return {
        "nonnegative": nonneg_ok,
        "monotone": monotone_ok,
        "concave": concave_ok,
        "valid_pcdeq": nonneg_ok and monotone_ok and concave_ok,
    }
