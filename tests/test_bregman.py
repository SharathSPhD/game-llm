"""Tests for the Bregman divergence library.

Tests mathematical properties that must hold for any valid Bregman divergence:
    1. Non-negativity: D_Φ(x || y) ≥ 0
    2. Identity: D_Φ(x || x) = 0
    3. Mirror map invertibility: ∇Φ*(∇Φ(x)) ≈ x (on the constraint set)
    4. Simplex membership: softmax always returns valid probabilities
"""

import pytest
import torch

from kinetic_ai.optim.bregman import DilatedEntropy, Euclidean, NegativeEntropy


class TestNegativeEntropy:
    """Tests for the negative entropy Bregman divergence (KL divergence)."""

    def setup_method(self) -> None:
        self.bregman = NegativeEntropy()

    def test_phi_uniform(self) -> None:
        """Φ(uniform) should equal -log(n) (maximum entropy, negated)."""
        n = 4
        x = torch.ones(n) / n
        expected = -torch.log(torch.tensor(float(n)))
        assert torch.isclose(self.bregman.phi(x), expected, atol=1e-6)

    def test_divergence_non_negative(self) -> None:
        """D_Φ(x || y) ≥ 0 for all valid distributions."""
        for _ in range(20):
            x = torch.softmax(torch.randn(5), dim=-1)
            y = torch.softmax(torch.randn(5), dim=-1)
            assert self.bregman.divergence(x, y) >= -1e-7

    def test_divergence_identity(self) -> None:
        """D_Φ(x || x) = 0."""
        x = torch.softmax(torch.randn(5), dim=-1)
        assert torch.isclose(self.bregman.divergence(x, x), torch.tensor(0.0), atol=1e-6)

    def test_mirror_map_invertibility(self) -> None:
        """∇Φ*(∇Φ(x)) should recover x (on the simplex)."""
        x = torch.softmax(torch.randn(5), dim=-1)
        y = self.bregman.grad_phi(x)
        x_recovered = self.bregman.grad_phi_star(y)
        assert torch.allclose(x, x_recovered, atol=1e-5)

    def test_grad_phi_star_returns_simplex(self) -> None:
        """∇Φ* should always return a valid probability distribution."""
        for _ in range(20):
            y = torch.randn(10)  # Arbitrary dual-space point
            x = self.bregman.grad_phi_star(y)
            assert torch.all(x >= 0), "Negative probabilities"
            assert torch.isclose(x.sum(), torch.tensor(1.0), atol=1e-6), "Not normalized"

    def test_divergence_is_kl(self) -> None:
        """The divergence should match the KL divergence formula."""
        x = torch.softmax(torch.randn(5), dim=-1)
        y = torch.softmax(torch.randn(5), dim=-1)
        kl = torch.sum(x * (torch.log(x) - torch.log(y)))
        assert torch.isclose(self.bregman.divergence(x, y), kl, atol=1e-5)

    def test_batch_support(self) -> None:
        """Should handle batched inputs."""
        x = torch.softmax(torch.randn(3, 5), dim=-1)
        y = torch.softmax(torch.randn(3, 5), dim=-1)
        div = self.bregman.divergence(x, y)
        assert div.shape == (3,)
        assert torch.all(div >= -1e-7)


class TestEuclidean:
    """Tests for the Euclidean (L2) Bregman divergence."""

    def setup_method(self) -> None:
        self.bregman = Euclidean()

    def test_divergence_is_l2(self) -> None:
        """D_Φ(x || y) = ½||x - y||²."""
        x = torch.randn(5)
        y = torch.randn(5)
        expected = 0.5 * torch.sum((x - y) ** 2)
        assert torch.isclose(self.bregman.divergence(x, y), expected, atol=1e-6)

    def test_mirror_map_is_identity(self) -> None:
        """∇Φ(x) = x and ∇Φ*(y) = y for Euclidean."""
        x = torch.randn(5)
        assert torch.allclose(self.bregman.grad_phi(x), x)
        assert torch.allclose(self.bregman.grad_phi_star(x), x)

    def test_divergence_identity(self) -> None:
        """D_Φ(x || x) = 0."""
        x = torch.randn(5)
        assert torch.isclose(self.bregman.divergence(x, x), torch.tensor(0.0), atol=1e-6)


class TestDilatedEntropy:
    """Tests for the dilated entropy Bregman divergence."""

    def setup_method(self) -> None:
        self.info_sets = [3, 2, 4]
        self.bregman = DilatedEntropy(info_set_sizes=self.info_sets)

    def test_divergence_non_negative(self) -> None:
        """D_Φ(x || y) ≥ 0."""
        total_dim = sum(self.info_sets)
        for _ in range(20):
            # Build valid strategy (per-info-set simplex)
            parts_x = [torch.softmax(torch.randn(s), dim=-1) for s in self.info_sets]
            parts_y = [torch.softmax(torch.randn(s), dim=-1) for s in self.info_sets]
            x = torch.cat(parts_x)
            y = torch.cat(parts_y)
            assert self.bregman.divergence(x, y) >= -1e-6

    def test_divergence_identity(self) -> None:
        """D_Φ(x || x) = 0."""
        parts = [torch.softmax(torch.randn(s), dim=-1) for s in self.info_sets]
        x = torch.cat(parts)
        assert torch.isclose(self.bregman.divergence(x, x), torch.tensor(0.0), atol=1e-6)

    def test_mirror_map_invertibility(self) -> None:
        """∇Φ*(∇Φ(x)) ≈ x on the treeplex."""
        parts = [torch.softmax(torch.randn(s), dim=-1) for s in self.info_sets]
        x = torch.cat(parts)
        y = self.bregman.grad_phi(x)
        x_recovered = self.bregman.grad_phi_star(y)
        assert torch.allclose(x, x_recovered, atol=1e-4)

    def test_per_info_set_simplex_preserved(self) -> None:
        """∇Φ* should return per-info-set valid simplexes."""
        y = torch.randn(sum(self.info_sets))
        x = self.bregman.grad_phi_star(y)
        offset = 0
        for size in self.info_sets:
            part = x[offset : offset + size]
            assert torch.all(part >= 0), f"Negative probs in info set of size {size}"
            assert torch.isclose(part.sum(), torch.tensor(1.0), atol=1e-5)
            offset += size
