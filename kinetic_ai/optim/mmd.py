"""Magnetic Mirror Descent (MMD) Optimizer.

This implements the actual MMD algorithm from Sokota et al. (2023),
operating in dual space with proper Bregman divergences.

Mathematical Formulation:
    Standard Mirror Descent update (dual space):
        y_{t+1} = ∇Φ(x_t) - η · g_t

    Magnetic Mirror Descent adds a magnetic regularization:
        y_{t+1} = ∇Φ(x_t) - η · g_t - η · τ · (∇Φ(x_t) - ∇Φ(x_ref))
                = (1 - η·τ)·∇Φ(x_t) - η·g_t + η·τ·∇Φ(x_ref)

    Primal recovery:
        x_{t+1} = ∇Φ*(y_{t+1})

    Where:
        Φ    = distance-generating function (mirror map)
        ∇Φ   = mirror map (primal → dual)
        ∇Φ*  = inverse mirror map (dual → primal)
        η    = learning rate
        τ    = magnetic strength (pull toward reference)
        x_ref = reference policy (the "magnet")
        g_t  = gradient (or utility gradient) at step t

    For the simplex (negative entropy mirror map):
        ∇Φ(x) = log(x) + 1
        ∇Φ*(y) = softmax(y)
        This means the update is:
            log(x_{t+1}) ∝ (1 - η·τ)·log(x_t) - η·g_t + η·τ·log(x_ref)
            x_{t+1} = softmax of the above

    Convergence Properties (Sokota et al. 2023):
        With FIXED reference x_ref:
            MMD converges linearly to the τ-regularized QRE with rationality λ = 1/τ.

        With PERIODIC reference updates (Regularized Nash Dynamics):
            The sequence of τ-regularized QRE fixed points traces a path toward
            the unregularized Nash Equilibrium as reference resets occur.

        Stepsize Conditions:
            For convergence on zero-sum games with simultaneous updates:
            η must be sufficiently small relative to τ and game structure.
            Sequential (alternating) updates enable larger stepsizes.

Two Operating Modes:
    1. Strategy-space mode: Directly optimizes probability distributions
       (strategies) on the simplex/treeplex. Used for game solving.
    2. Parameter-space mode: Optimizes neural network parameters with
       a Bregman-regularized update toward a reference model. Used for
       LLM alignment (Magnetic Preference Optimization).

References:
    [1] Sokota et al. "A Unified Approach to RL, QRE, and Two-Player
        Zero-Sum Games" (NeurIPS 2023)
    [2] ryan-dorazio/mmd-dilated (GitHub reference implementation)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import overload

import torch
from torch import Tensor
from torch.optim import Optimizer

from kinetic_ai.config import BregmanType, MMDConfig
from kinetic_ai.optim.bregman import (
    BregmanDivergence,
    DilatedEntropy,
    Euclidean,
    NegativeEntropy,
)


def create_bregman(bregman_type: BregmanType, **kwargs: object) -> BregmanDivergence:
    """Factory for creating Bregman divergence instances from config enum."""
    if bregman_type == BregmanType.NEGATIVE_ENTROPY:
        return NegativeEntropy()
    elif bregman_type == BregmanType.EUCLIDEAN:
        return Euclidean()
    elif bregman_type == BregmanType.DILATED_ENTROPY:
        info_set_sizes = kwargs.get("info_set_sizes", [2, 2])
        assert isinstance(info_set_sizes, list)
        return DilatedEntropy(info_set_sizes=info_set_sizes)
    else:
        raise ValueError(f"Unknown Bregman type: {bregman_type}")


class MagneticMirrorDescent(Optimizer):
    """Magnetic Mirror Descent optimizer.

    This implements MMD with proper Bregman divergences, operating in dual
    space. Unlike the naive L2-proximal version, this:
        - Uses mirror maps (log/softmax for simplex, dilated entropy for EFGs)
        - Operates in dual space with proper Bregman geometry
        - Supports reference policy updates (Regularized Nash Dynamics)

    Args:
        params: Iterable of parameters to optimize.
        config: MMD configuration dataclass.
        bregman: Bregman divergence to use. If None, created from config.
        reference_params: Optional reference parameters. If None, initial
            parameters are used as the reference ("magnet").

    Example (parameter-space mode for neural networks)::

        model = MyModel()
        config = MMDConfig(lr=0.01, tau=0.1, bregman_type=BregmanType.EUCLIDEAN)
        optimizer = MagneticMirrorDescent(model.parameters(), config=config)

        for step in range(num_steps):
            optimizer.zero_grad()
            loss = compute_loss(model)
            loss.backward()
            optimizer.step()

    Example (strategy-space mode for game solving)::

        # Strategy on a 3-action simplex
        strategy = torch.tensor([1/3, 1/3, 1/3], requires_grad=False)
        bregman = NegativeEntropy()
        config = MMDConfig(lr=0.1, tau=0.05)

        for step in range(num_steps):
            utility_grad = compute_utility_gradient(strategy)
            strategy = mmd_strategy_update(strategy, utility_grad, reference,
                                           bregman, config)
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        config: MMDConfig | None = None,
        bregman: BregmanDivergence | None = None,
        reference_params: Iterator[Tensor] | None = None,
    ) -> None:
        if config is None:
            config = MMDConfig()

        if config.lr < 0.0:
            raise ValueError(f"Invalid learning rate: {config.lr}")
        if config.tau < 0.0:
            raise ValueError(f"Invalid magnetic strength (tau): {config.tau}")

        self.config = config
        self.bregman = bregman or create_bregman(config.bregman_type)

        defaults = dict(lr=config.lr, tau=config.tau)
        super().__init__(params, defaults)

        # Store reference parameters (the "magnet")
        self._reference_state: list[Tensor] = []
        self._step_count = 0

        if reference_params is not None:
            for p_ref in reference_params:
                self._reference_state.append(p_ref.detach().clone())
        else:
            for group in self.param_groups:
                for p in group["params"]:
                    self._reference_state.append(p.detach().clone())

    def _update_reference(self) -> None:
        """Update the reference policy to the current policy.

        This implements Regularized Nash Dynamics (RND): by periodically
        resetting the magnet, the sequence of QREs traces a path to the
        unregularized Nash Equilibrium.
        """
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                self._reference_state[idx] = p.detach().clone()
                idx += 1

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform a single MMD optimization step.

        The update rule in parameter space:
            1. Compute dual-space coordinates: y_t = ∇Φ(θ_t)
            2. Compute reference dual coordinates: y_ref = ∇Φ(θ_ref)
            3. Magnetic dual update:
               y_{t+1} = y_t - η·g_t - η·τ·(y_t - y_ref)
                       = (1 - η·τ)·y_t - η·g_t + η·τ·y_ref
            4. Primal recovery: θ_{t+1} = ∇Φ*(y_{t+1})
        """
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss_result = closure()
                # Convert Tensor result to float if needed
                loss = float(loss_result.item()) if isinstance(loss_result, Tensor) else loss_result

        ref_idx = 0
        for group in self.param_groups:
            lr = group["lr"]
            tau = group["tau"]

            for p in group["params"]:
                if p.grad is None:
                    ref_idx += 1
                    continue

                grad = p.grad.data
                p_ref = self._reference_state[ref_idx]

                # Step 1: Map current params to dual space
                y_current = self.bregman.grad_phi(p.data)

                # Step 2: Map reference params to dual space
                y_ref = self.bregman.grad_phi(p_ref)

                # Step 3: Magnetic dual-space update
                # y_{t+1} = (1 - η·τ)·y_t - η·g_t + η·τ·y_ref
                y_next = (1.0 - lr * tau) * y_current - lr * grad + lr * tau * y_ref

                # Step 4: Map back to primal space
                p.data.copy_(self.bregman.grad_phi_star(y_next))

                ref_idx += 1

        # Optionally update reference (Regularized Nash Dynamics)
        self._step_count += 1
        if (
            self.config.reference_update_interval > 0
            and self._step_count % self.config.reference_update_interval == 0
        ):
            self._update_reference()

        return loss


def mmd_strategy_update(
    strategy: Tensor,
    utility_gradient: Tensor,
    reference: Tensor,
    bregman: BregmanDivergence,
    lr: float,
    tau: float,
) -> Tensor:
    """Functional MMD update for direct strategy optimization.

    This is the pure functional form used for game solving (not NN training).
    Operates directly on probability distributions.

    Args:
        strategy: Current strategy (probability distribution on simplex).
        utility_gradient: Gradient of utility w.r.t. strategy.
            Note: for maximization, this should be the utility gradient
            (not negated), and we ADD it in the dual update.
        reference: Reference strategy (the "magnet").
        bregman: Bregman divergence / mirror map.
        lr: Learning rate.
        tau: Magnetic strength.

    Returns:
        Updated strategy on the simplex.
    """
    # Map to dual space
    y_current = bregman.grad_phi(strategy)
    y_ref = bregman.grad_phi(reference)

    # Magnetic dual update (note: + for maximization of utility)
    y_next = (1.0 - lr * tau) * y_current + lr * utility_gradient + lr * tau * y_ref

    # Map back to primal (simplex)
    return bregman.grad_phi_star(y_next)
