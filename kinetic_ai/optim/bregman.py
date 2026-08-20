"""Bregman divergence library for mirror descent algorithms.

Provides pluggable distance-generating functions (mirror maps) and their
associated Bregman divergences. These are the mathematical core of
Magnetic Mirror Descent.

Mathematical Background:
    A Bregman divergence D_Φ(x || y) = Φ(x) - Φ(y) - ⟨∇Φ(y), x - y⟩
    where Φ is a strictly convex, differentiable function (the "mirror map").

    Mirror descent operates in dual space:
        y_{t+1} = ∇Φ(x_t) - η·g_t          (dual update)
        x_{t+1} = ∇Φ*(y_{t+1})              (primal recovery)

    where ∇Φ* is the gradient of the convex conjugate (inverse mirror map).

Key Implementations:
    - NegativeEntropy: Φ(x) = Σ x_i log(x_i), maps to/from simplex via softmax
    - Euclidean: Φ(x) = ½||x||², reduces mirror descent to standard GD
    - DilatedEntropy: For extensive-form game treeplexes

References:
    [1] Sokota et al. "A Unified Approach to RL, QRE, and Two-Player
        Zero-Sum Games" (NeurIPS 2023, arXiv:2206.05825)
    [2] Hoda et al. "Smoothing Techniques for Computing Nash Equilibria
        of Sequential Games" (Mathematics of Operations Research, 2010)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BregmanDivergence(ABC):
    """Abstract base class for Bregman divergence / mirror map pairs.

    Subclasses must implement:
        - phi(x): The generating function Φ(x)
        - grad_phi(x): ∇Φ(x), the mirror map (primal → dual)
        - grad_phi_star(y): ∇Φ*(y), the inverse mirror map (dual → primal)
        - divergence(x, y): D_Φ(x || y), the Bregman divergence
    """

    @abstractmethod
    def phi(self, x: Tensor) -> Tensor:
        """Evaluate the generating function Φ(x).

        Args:
            x: Point in primal space.

        Returns:
            Scalar value of Φ(x).
        """
        ...

    @abstractmethod
    def grad_phi(self, x: Tensor) -> Tensor:
        """Compute ∇Φ(x), mapping from primal to dual space.

        Args:
            x: Point in primal space.

        Returns:
            Gradient ∇Φ(x) in dual space.
        """
        ...

    @abstractmethod
    def grad_phi_star(self, y: Tensor) -> Tensor:
        """Compute ∇Φ*(y), mapping from dual to primal space.

        This is the inverse mirror map. For negative entropy, this is softmax.

        Args:
            y: Point in dual space.

        Returns:
            Point in primal space (on the constraint set).
        """
        ...

    def divergence(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the Bregman divergence D_Φ(x || y).

        D_Φ(x || y) = Φ(x) - Φ(y) - ⟨∇Φ(y), x - y⟩

        Args:
            x: First point in primal space.
            y: Second point in primal space (center of divergence).

        Returns:
            Non-negative scalar divergence value.
        """
        return self.phi(x) - self.phi(y) - torch.sum(self.grad_phi(y) * (x - y))


class NegativeEntropy(BregmanDivergence):
    """Negative entropy generating function on the probability simplex.

    Φ(x) = Σ_i x_i log(x_i)   (negative entropy)
    ∇Φ(x) = log(x) + 1         (log mapping, primal → dual)
    ∇Φ*(y) = softmax(y)        (softmax, dual → primal)
    D_Φ(x || y) = KL(x || y)   (KL divergence)

    This is the correct mirror map for mirror descent on the simplex,
    and is the foundation of Magnetic Mirror Descent for normal-form games.

    Args:
        eps: Small constant for numerical stability in log computations.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def phi(self, x: Tensor) -> Tensor:
        """Φ(x) = Σ x_i log(x_i) (negative entropy)."""
        x_safe = x.clamp(min=self.eps)
        return torch.sum(x_safe * torch.log(x_safe), dim=-1)

    def grad_phi(self, x: Tensor) -> Tensor:
        """∇Φ(x) = log(x) + 1."""
        x_safe = x.clamp(min=self.eps)
        return torch.log(x_safe) + 1.0

    def grad_phi_star(self, y: Tensor) -> Tensor:
        """∇Φ*(y) = softmax(y - 1) = softmax(y) (shift-invariant).

        Maps from unconstrained dual space back to the probability simplex.
        """
        # softmax is shift-invariant so the -1 cancels
        return torch.softmax(y, dim=-1)

    def divergence(self, x: Tensor, y: Tensor) -> Tensor:
        """D_Φ(x || y) = KL(x || y) = Σ x_i log(x_i / y_i)."""
        x_safe = x.clamp(min=self.eps)
        y_safe = y.clamp(min=self.eps)
        return torch.sum(x_safe * (torch.log(x_safe) - torch.log(y_safe)), dim=-1)


class Euclidean(BregmanDivergence):
    """Euclidean (L2) generating function.

    Φ(x) = ½ ||x||²
    ∇Φ(x) = x       (identity mirror map)
    ∇Φ*(y) = y      (identity inverse)
    D_Φ(x || y) = ½ ||x - y||²

    When used with mirror descent, this reduces to standard gradient descent.
    Provided as a baseline for comparison.
    """

    def phi(self, x: Tensor) -> Tensor:
        """Φ(x) = ½ ||x||²."""
        return 0.5 * torch.sum(x**2, dim=-1)

    def grad_phi(self, x: Tensor) -> Tensor:
        """∇Φ(x) = x (identity)."""
        return x

    def grad_phi_star(self, y: Tensor) -> Tensor:
        """∇Φ*(y) = y (identity)."""
        return y

    def divergence(self, x: Tensor, y: Tensor) -> Tensor:
        """D_Φ(x || y) = ½ ||x - y||²."""
        return 0.5 * torch.sum((x - y) ** 2, dim=-1)


class DilatedEntropy(BregmanDivergence):
    """Dilated entropy for sequence-form (extensive-form) game treeplexes.

    For extensive-form games, the strategy space is a treeplex (Cartesian
    product of simplexes connected by a tree structure). Standard negative
    entropy does NOT provide 1-strong convexity w.r.t. the treeplex L1 norm.

    Dilated entropy constructs a weighted combination of local entropies
    at each information set, with weights chosen to achieve 1-strong convexity.

    This implementation supports a simplified two-level tree structure:
    a root decision point followed by K information sets, each with A_k actions.

    For full EFG support, the tree structure and weights must be computed
    from the game tree. See Hoda et al. (2010) and Kroer et al. (2020).

    Args:
        info_set_sizes: List of action counts at each information set.
        weights: Per-info-set entropy weights. If None, computed for 1-strong convexity.
        eps: Numerical stability constant.

    References:
        [1] Hoda et al. "Smoothing Techniques for Computing Nash Equilibria
            of Sequential Games" (MOR 2010)
        [2] Kroer et al. "Faster Algorithms for Extensive-Form Game Solving
            via Improved Smoothing" (ICML 2018)
    """

    def __init__(
        self,
        info_set_sizes: list[int],
        weights: list[float] | None = None,
        eps: float = 1e-10,
    ) -> None:
        self.info_set_sizes = info_set_sizes
        self.eps = eps

        if weights is not None:
            self.weights = weights
        else:
            # Default: uniform weights that achieve 1-strong convexity
            # For a simple tree, weight = 1 / (depth * max_branching)
            depth = len(info_set_sizes)
            max_branch = max(info_set_sizes) if info_set_sizes else 1
            self.weights = [1.0 / (depth * max_branch)] * len(info_set_sizes)

    def _split_strategy(self, x: Tensor) -> list[Tensor]:
        """Split a flat strategy vector into per-information-set simplexes."""
        parts = []
        offset = 0
        for size in self.info_set_sizes:
            parts.append(x[..., offset : offset + size])
            offset += size
        return parts

    def phi(self, x: Tensor) -> Tensor:
        """Weighted sum of local negative entropies at each information set."""
        parts = self._split_strategy(x)
        total = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for w, part in zip(self.weights, parts):
            part_safe = part.clamp(min=self.eps)
            total = total + w * torch.sum(part_safe * torch.log(part_safe), dim=-1)
        return total

    def grad_phi(self, x: Tensor) -> Tensor:
        """Gradient: weighted log at each information set."""
        parts = self._split_strategy(x)
        grads = []
        for w, part in zip(self.weights, parts):
            part_safe = part.clamp(min=self.eps)
            grads.append(w * (torch.log(part_safe) + 1.0))
        return torch.cat(grads, dim=-1)

    def grad_phi_star(self, y: Tensor) -> Tensor:
        """Inverse mirror map: per-information-set softmax with weights."""
        parts = self._split_strategy(y)
        primals = []
        for w, part in zip(self.weights, parts):
            # Invert the weighting: y_i = w * (log(x_i) + 1) → x = softmax(y/w)
            primals.append(torch.softmax(part / max(w, self.eps), dim=-1))
        return torch.cat(primals, dim=-1)
