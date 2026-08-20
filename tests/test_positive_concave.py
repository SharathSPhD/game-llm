"""Tests for Positive Concave DEQ constraints.

Validates:
    1. NonNegativeLinear produces non-negative weights
    2. ConcaveActivation satisfies concavity on positive orthant
    3. PositiveConcaveBlock satisfies pcDEQ properties
    4. verify_positive_concave correctly identifies valid/invalid blocks
"""

import pytest
import torch

from kinetic_ai.models.positive_concave import (
    ConcaveActivation,
    NonNegativeLinear,
    PositiveConcaveBlock,
    verify_positive_concave,
)


class TestNonNegativeLinear:
    """Tests for the non-negative weight linear layer."""

    def test_weights_non_negative_softplus(self) -> None:
        """Softplus parameterization should produce non-negative weights."""
        layer = NonNegativeLinear(5, 3, method="softplus")
        assert (layer.weight >= 0).all(), "Softplus weights should be non-negative"

    def test_weights_non_negative_abs(self) -> None:
        """Absolute value parameterization should produce non-negative weights."""
        layer = NonNegativeLinear(5, 3, method="abs")
        assert (layer.weight >= 0).all(), "Abs weights should be non-negative"

    def test_forward_shape(self) -> None:
        """Forward pass should produce correct output shape."""
        layer = NonNegativeLinear(5, 3)
        x = torch.randn(2, 5)
        out = layer(x)
        assert out.shape == (2, 3)

    def test_weights_gradient_flows(self) -> None:
        """Gradients should flow through the weight reparameterization."""
        layer = NonNegativeLinear(5, 3, method="softplus")
        x = torch.randn(2, 5)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight_raw.grad is not None
        assert not torch.all(layer.weight_raw.grad == 0)


class TestConcaveActivation:
    """Tests for concave activation functions."""

    @pytest.mark.parametrize("activation_type", ["sqrt", "log1p", "power", "min_linear"])
    def test_non_negative_output(self, activation_type: str) -> None:
        """All activations should produce non-negative output on non-negative input."""
        alpha = 0.5 if activation_type == "power" else 1.0
        act = ConcaveActivation(activation_type, alpha=alpha)
        x = torch.rand(20).abs() * 5
        out = act(x)
        assert (out >= 0).all(), f"{activation_type} should be non-negative"

    @pytest.mark.parametrize("activation_type", ["sqrt", "log1p", "power", "min_linear"])
    def test_midpoint_concavity(self, activation_type: str) -> None:
        """f((x1+x2)/2) ≥ (f(x1)+f(x2))/2 for concave functions."""
        alpha = 0.5 if activation_type == "power" else 2.0
        act = ConcaveActivation(activation_type, alpha=alpha)

        for _ in range(50):
            x1 = torch.rand(10).abs() * 5 + 0.01
            x2 = torch.rand(10).abs() * 5 + 0.01

            f_mid = act(0.5 * (x1 + x2))
            f_avg = 0.5 * (act(x1) + act(x2))

            assert (f_mid >= f_avg - 1e-5).all(), (
                f"{activation_type} violates midpoint concavity"
            )


class TestPositiveConcaveBlock:
    """Tests for the full pcDEQ block."""

    def test_forward_shape(self) -> None:
        """Forward pass should produce correct output shape."""
        block = PositiveConcaveBlock(hidden_dim=8, input_dim=4)
        z = torch.randn(2, 8).abs()
        x = torch.randn(2, 4).abs()
        out = block(z, x)
        assert out.shape == (2, 8)

    def test_output_non_negative(self) -> None:
        """Output should be non-negative for non-negative input."""
        block = PositiveConcaveBlock(hidden_dim=8, input_dim=4, activation="log1p")
        z = torch.randn(10, 8).abs()
        x = torch.randn(10, 4).abs()
        out = block(z, x)
        assert (out >= 0).all(), "pcDEQ output should be non-negative"

    def test_verify_positive_concave_passes(self) -> None:
        """The pcDEQ block should pass the positive concave verification."""
        block = PositiveConcaveBlock(hidden_dim=4, input_dim=2, activation="log1p")
        result = verify_positive_concave(block, z_dim=4, x_dim=2, n_tests=50)
        assert result["nonnegative"], "Should be non-negative"
        assert result["valid_pcdeq"], f"Should be valid pcDEQ, got {result}"


class TestVerifyPositiveConcave:
    """Tests for the verification utility."""

    def test_strictly_convex_fails_concavity(self) -> None:
        """Strictly convex quadratic should fail concavity check.

        Tests that verify_positive_concave correctly identifies functions
        that violate concavity. A quadratic f(z) = z^2 is strictly convex,
        not concave, and should fail the midpoint concavity test:
            f((z1+z2)/2) < (f(z1)+f(z2))/2  for strict convexity
        """

        def quadratic_transform(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Strictly convex: f(z) = z^2 + 0.1 (ignoring x)
            return z ** 2 + 0.1

        result = verify_positive_concave(quadratic_transform, z_dim=3, x_dim=2, n_tests=100)
        assert not result["concave"], (
            "Quadratic (strictly convex) should fail concavity check. "
            f"Got result={result}"
        )

    def test_identity_fails_concavity(self) -> None:
        """A quadratic (convex) transform should fail concavity."""

        def quadratic_transform(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return z ** 2 + 0.1  # Convex, not concave

        result = verify_positive_concave(quadratic_transform, z_dim=3, x_dim=2, n_tests=100)
        assert not result["concave"], "Quadratic should fail concavity"
