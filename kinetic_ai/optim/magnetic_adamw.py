"""MagneticAdamW: AdamW with Magnetic Proximal Pull.

Combines standard AdamW update (adaptive learning rate, weight decay) with a
magnetic pull toward a reference point. This implements the practical form of
magnetic/KL-anchored preference optimization suitable for LM training.

The update is:
    1. Perform standard AdamW: θ' ← AdamW_step(θ)
    2. Apply magnetic pull: θ ← θ' − lr·τ·(θ' − θ_ref)

Reference modes:
    - 'ema': θ_ref ← β·θ_ref + (1−β)·θ each step (exponential moving average)
    - 'periodic': snapshot θ_ref every K steps (Regularized Nash Dynamics)

References:
    - AdamW: https://arxiv.org/abs/1711.05101 (Loshchilov & Hutter)
    - Magnetic mirror descent: decision 0003-magnetic-adamw.md
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class MagneticAdamW(Optimizer):
    """AdamW optimizer with magnetic proximal pull toward a reference point.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for running average of gradient and squared gradient
               (default: (0.9, 0.999)).
        eps: Term added to denominator for numerical stability (default: 1e-8).
        weight_decay: Weight decay coefficient (default: 0.0).
        tau: Magnetic strength (pull toward reference). tau=0 recovers AdamW.
             (default: 0.0).
        ref_mode: Reference mode - 'ema' (exponential moving average) or 'periodic'
                 (snapshot every K steps). (default: 'ema').
        ref_beta: Exponential moving average coefficient for 'ema' mode
                 (default: 0.999).
        ref_interval: Snapshot interval for 'periodic' mode (default: 10).

    Shape:
        - Input: parameters of any shape.
        - Output: updated parameters with magnetic pull applied.

    Example:
        >>> model = nn.Linear(10, 5)
        >>> opt = MagneticAdamW(model.parameters(), lr=1e-3, tau=0.01)
        >>> loss = model(x).sum()
        >>> loss.backward()
        >>> opt.step()
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        tau: float = 0.0,
        ref_mode: str = "ema",
        ref_beta: float = 0.999,
        ref_interval: int = 10,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if tau < 0.0:
            raise ValueError(f"Invalid tau value: {tau}")
        if ref_mode not in ("ema", "periodic"):
            raise ValueError(f"Invalid ref_mode: {ref_mode}. Must be 'ema' or 'periodic'.")
        if not (0.0 <= ref_beta < 1.0):
            raise ValueError(f"Invalid ref_beta: {ref_beta}")
        if ref_interval <= 0:
            raise ValueError(f"Invalid ref_interval: {ref_interval}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            tau=tau,
            ref_mode=ref_mode,
            ref_beta=ref_beta,
            ref_interval=ref_interval,
        )
        super().__init__(params, defaults)

        self.tau = tau
        self.ref_mode = ref_mode
        self.ref_beta = ref_beta
        self.ref_interval = ref_interval

        # Initialize reference state (will be populated on first step)
        # Maps parameter id (int) to reference tensor
        self.ref_state: dict[int, Tensor] | None = None
        self.step_counter = 0

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("eps", 1e-8)
            group.setdefault("tau", 0.0)
            group.setdefault("ref_mode", "ema")
            group.setdefault("ref_beta", 0.999)
            group.setdefault("ref_interval", 10)

    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            None (following torch.optim.Optimizer convention).
        """
        if closure is not None:
            closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "MagneticAdamW does not support sparse gradients"
                    )

                wd = group["weight_decay"]
                if wd != 0:
                    grad = grad.add(p, alpha=wd)

                # Get state for this parameter
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Compute adaptive learning rate term
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])

                # Standard AdamW update
                step_size = group["lr"] / bias_correction1
                p_new = p.data.add(exp_avg / denom, alpha=-step_size)

                # Apply magnetic proximal pull if tau > 0
                tau = group["tau"]
                if tau > 0:
                    # Initialize reference on first step
                    if self.ref_state is None:
                        self.ref_state = {}

                    # Get or initialize reference for this parameter
                    param_id = id(p)
                    if param_id not in self.ref_state:
                        self.ref_state[param_id] = p.data.clone()

                    ref = self.ref_state[param_id]

                    # Magnetic pull: p ← p' − lr·τ·(p' − p_ref)
                    # This pulls the updated parameter back toward the reference
                    p_new = p_new.add(p_new - ref, alpha=-group["lr"] * tau)

                    # Update reference based on mode
                    if group["ref_mode"] == "ema":
                        # EMA: ref ← β·ref + (1−β)·p_new
                        ref.mul_(group["ref_beta"]).add_(p_new, alpha=1 - group["ref_beta"])
                    elif (
                        group["ref_mode"] == "periodic"
                        and self.step_counter % group["ref_interval"] == 0
                    ):
                        # Periodic: snapshot every ref_interval steps
                        ref.copy_(p_new)

                # Update parameter
                p.data.copy_(p_new)

        # Increment global step counter for periodic mode
        self.step_counter += 1


def magnetic_adamw_step(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    state_step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    tau: float,
    ref: Tensor | None = None,
) -> Tensor:
    """Functional form of a single MagneticAdamW step.

    This is a lower-level interface for advanced use cases where you need
    fine-grained control over the parameter updates.

    Args:
        param: Parameter to update.
        grad: Gradient of loss w.r.t. parameter.
        exp_avg: Running average of gradient.
        exp_avg_sq: Running average of squared gradient.
        state_step: Current step number (1-indexed).
        lr: Learning rate.
        beta1: Coefficient for gradient moving average.
        beta2: Coefficient for squared gradient moving average.
        eps: Numerical stability term.
        weight_decay: Weight decay coefficient.
        tau: Magnetic strength (0 for no magnetic pull).
        ref: Reference point for magnetic pull. If None, no pull applied.

    Returns:
        Updated parameter tensor (not in-place).
    """
    # Apply weight decay
    if weight_decay != 0:
        grad = grad.add(param, alpha=weight_decay)

    # Update biased first and second moment estimates
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    bias_correction1 = 1 - beta1 ** state_step
    bias_correction2 = 1 - beta2 ** state_step

    # Compute denom for adaptive LR
    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)

    # AdamW step
    step_size = lr / bias_correction1
    p_new = param.add_(exp_avg / denom, alpha=-step_size)

    # Apply magnetic pull if tau > 0 and reference is provided
    if tau > 0 and ref is not None:
        p_new = p_new.add_(p_new - ref, alpha=-lr * tau)

    return p_new
