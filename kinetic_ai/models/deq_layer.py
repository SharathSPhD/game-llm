"""Deep Equilibrium (DEQ) Layer with proper fixed-point solvers.

Replaces explicit depth with an implicit fixed-point computation:
    z* = f(z*, x)  where z* is the equilibrium state

The forward pass finds z* via iterative root-finding (Anderson acceleration
or Broyden's method). The backward pass uses the Implicit Function Theorem
to compute gradients analytically at the fixed point, without unrolling.

Key Properties:
    - O(1) memory footprint regardless of effective depth
    - Equivalent to an infinitely deep network
    - Gradients computed via IFT, not backpropagation through iterations

Solver Options:
    - Picard: Simple fixed-point iteration z ← f(z, x). Linear convergence
      at best, can diverge for non-contractive maps. Baseline only.
    - Anderson Acceleration: Extrapolates from a history of m previous
      iterates to accelerate convergence. Default and recommended.
    - Broyden's Method: Quasi-Newton solver for the root-finding problem
      g(z) = f(z,x) - z = 0. Quadratic convergence near the solution.

Stability:
    - Spectral normalization can enforce the Lipschitz constant < 1,
      guaranteeing the existence and uniqueness of the fixed point.
    - Positive Concave constraints (pcDEQ) provide even stronger guarantees.

References:
    [1] Bai et al. "Deep Equilibrium Models" (NeurIPS 2019, arXiv:1909.01377)
    [2] Bai et al. "Positive Concave Deep Equilibrium Models" (arXiv:2402.04029)
    [3] Anderson, D.G. "Iterative procedures for nonlinear integral
        equations" (J. ACM, 1965)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor

from kinetic_ai.config import DEQConfig, SolverType

# ---------------------------------------------------------------------------
# Fixed-point solvers
# ---------------------------------------------------------------------------


def _picard_iteration(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    z_init: Tensor,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, dict[str, object]]:
    """Simple fixed-point iteration: z ← f(z, x).

    Linear convergence at best. Provided as baseline.

    Convergence is gated on RELATIVE residual: rel = ||z_next - z|| / (||z_next|| + eps)
    This ensures convergence is achievable at batch scale (F14 fix).
    Both absolute and relative residuals are recorded for analysis.
    """
    z = z_init.clone()
    residuals: list[float] = []
    rel_residuals: list[float] = []
    eps = 1e-8

    for i in range(max_iter):  # noqa: B007  (used after loop)
        z_next = f(z, x)
        abs_residual = torch.norm(z_next - z).item()
        norm_z_next = torch.norm(z_next).item()
        rel_residual = abs_residual / (norm_z_next + eps)
        residuals.append(abs_residual)
        rel_residuals.append(rel_residual)
        z = z_next
        if rel_residual < tol:
            break

    info = {
        "iterations": i + 1,
        "residuals": residuals,
        "rel_residuals": rel_residuals,
        "converged": rel_residual < tol,
    }
    return z, info


def _anderson_acceleration(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    z_init: Tensor,
    max_iter: int,
    tol: float,
    m: int = 5,
    beta: float = 1.0,
) -> tuple[Tensor, dict[str, object]]:
    """Anderson acceleration for fixed-point problems.

    Maintains a history of m previous iterates and their residuals,
    then solves a least-squares problem to find optimal mixing coefficients.

    This typically provides superlinear convergence, significantly faster
    than Picard iteration.

    Convergence is gated on RELATIVE residual: rel = ||z_next - z|| / (||z_next|| + eps)
    This ensures convergence is achievable at batch scale (F14 fix).
    Both absolute and relative residuals are recorded for analysis.

    Args:
        f: The fixed-point map z ← f(z, x).
        x: Input/context tensor.
        z_init: Initial guess for the fixed point.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        m: History size (number of previous iterates to use).
        beta: Mixing/damping coefficient (1.0 = no damping).

    Returns:
        Tuple of (fixed_point, info_dict).
    """
    z = z_init.clone()
    bsz = z.shape[0] if z.dim() > 1 else 1
    flat_dim = z.numel() // bsz

    # Flatten for linear algebra
    z_flat = z.reshape(bsz, flat_dim)

    # History buffers: store (z, g(z)) pairs where g(z) = f(z,x) - z
    X_history: list[Tensor] = []  # z values
    F_history: list[Tensor] = []  # f(z,x) values

    residuals: list[float] = []
    rel_residuals: list[float] = []
    eps = 1e-8

    for k in range(max_iter):  # noqa: B007  (used after loop)
        f_z = f(z.reshape(z_init.shape), x).reshape(bsz, flat_dim)
        g_z = f_z - z_flat  # residual

        abs_residual = torch.norm(g_z).item()
        norm_f_z = torch.norm(f_z).item()
        rel_residual = abs_residual / (norm_f_z + eps)
        residuals.append(abs_residual)
        rel_residuals.append(rel_residual)

        if rel_residual < tol:
            z = z_flat.reshape(z_init.shape)
            break

        # Store in history
        X_history.append(z_flat.clone())
        F_history.append(f_z.clone())

        # Trim history to size m
        if len(X_history) > m:
            X_history.pop(0)
            F_history.pop(0)

        n_hist = len(X_history)

        if n_hist < 2:
            # Not enough history, do a simple Picard step
            z_flat = (1 - beta) * z_flat + beta * f_z
        else:
            # Build the differences matrix
            # ΔG_k = [g_{k} - g_{k-1}, g_{k-1} - g_{k-2}, ...]
            G = torch.stack(
                [F_history[i] - X_history[i] for i in range(n_hist)], dim=-1
            )  # (bsz, flat_dim, n_hist)
            dG = G[..., 1:] - G[..., :-1]  # (bsz, flat_dim, n_hist-1)

            # Solve least squares: min ||dG @ α - g_k||²
            # Using normal equations: (dG^T dG) α = dG^T g_k
            g_current = g_z.unsqueeze(-1)  # (bsz, flat_dim, 1)

            # For numerical stability, use torch.linalg.lstsq
            # Reshape for batch lstsq: (bsz, flat_dim, n_hist-1)
            try:
                # alpha shape: (bsz, n_hist-1, 1)
                result = torch.linalg.lstsq(dG, g_current)
                alpha = result.solution
            except RuntimeError:
                # Fallback to Picard if lstsq fails
                z_flat = (1 - beta) * z_flat + beta * f_z
                continue

            # Compute the accelerated iterate
            # z_{k+1} = (1-β) * (z_k - ΔX @ α) + β * (f(z_k) - ΔF @ α)
            dX = torch.stack(X_history, dim=-1)  # (bsz, flat_dim, n_hist)
            dX = dX[..., 1:] - dX[..., :-1]  # (bsz, flat_dim, n_hist-1)
            dF = torch.stack(F_history, dim=-1)
            dF = dF[..., 1:] - dF[..., :-1]

            z_correction = torch.bmm(dX, alpha).squeeze(-1)
            f_correction = torch.bmm(dF, alpha).squeeze(-1)

            z_flat = (
                (1 - beta) * (z_flat - z_correction) + beta * (f_z - f_correction)
            )

        z = z_flat.reshape(z_init.shape)

    info = {
        "iterations": min(k + 1, max_iter),
        "residuals": residuals,
        "rel_residuals": rel_residuals,
        "converged": (len(rel_residuals) > 0 and rel_residuals[-1] < tol),
    }
    return z.reshape(z_init.shape), info


def _broyden_solver(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    z_init: Tensor,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, dict[str, object]]:
    """Broyden's method for solving g(z) = f(z,x) - z = 0.

    Quasi-Newton method that approximates the inverse Jacobian.
    Provides quadratic convergence near the solution.

    Uses the "good Broyden" update with Sherman-Morrison for efficient
    inverse Jacobian maintenance.

    Convergence is gated on RELATIVE residual: rel = ||z_next - z|| / (||z_next|| + eps)
    This ensures convergence is achievable at batch scale (F14 fix).
    Both absolute and relative residuals are recorded for analysis.
    """
    bsz = z_init.shape[0] if z_init.dim() > 1 else 1
    flat_dim = z_init.numel() // bsz

    z = z_init.clone().reshape(bsz, flat_dim)

    # Initial residual
    g = f(z.reshape(z_init.shape), x).reshape(bsz, flat_dim) - z

    abs_residual = torch.norm(g).item()
    norm_f_init = torch.norm(f(z.reshape(z_init.shape), x).reshape(bsz, flat_dim)).item()
    eps = 1e-8
    rel_residual = abs_residual / (norm_f_init + eps)
    residuals: list[float] = [abs_residual]
    rel_residuals: list[float] = [rel_residual]

    # Initialize inverse Jacobian approximation as -I
    # (good for contraction mappings where J ≈ 0)
    J_inv = -torch.eye(flat_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
    J_inv = J_inv.expand(bsz, -1, -1).clone()

    for k in range(max_iter):  # noqa: B007  (used after loop)
        if rel_residuals[-1] < tol:
            break

        # Newton-like step: Δz = -J_inv @ g
        delta_z = -torch.bmm(J_inv, g.unsqueeze(-1)).squeeze(-1)

        # Update z
        z_new = z + delta_z

        # New residual
        g_new = f(z_new.reshape(z_init.shape), x).reshape(bsz, flat_dim) - z_new

        abs_residual = torch.norm(g_new).item()
        norm_f_new = torch.norm(f(z_new.reshape(z_init.shape), x).reshape(bsz, flat_dim)).item()
        rel_residual = abs_residual / (norm_f_new + eps)
        residuals.append(abs_residual)
        rel_residuals.append(rel_residual)

        # Broyden update to J_inv (Sherman-Morrison)
        delta_g = g_new - g  # (bsz, flat_dim)

        # u = Δz - J_inv @ Δg
        u = delta_z - torch.bmm(J_inv, delta_g.unsqueeze(-1)).squeeze(-1)

        # denominator = Δz^T @ J_inv @ Δg
        denom = torch.bmm(
            delta_z.unsqueeze(1), torch.bmm(J_inv, delta_g.unsqueeze(-1))
        ).squeeze(-1).squeeze(-1)

        # Avoid division by zero
        mask = denom.abs() > 1e-12
        if mask.any():
            # J_inv += (u ⊗ (Δz^T @ J_inv)) / denom
            # The mask ensures |denom| > 1e-12, so we can divide safely.
            # Importantly, we do NOT clamp the denominator, which would flip
            # the sign for negative values and violate Sherman-Morrison formula.
            numerator = torch.bmm(
                u.unsqueeze(-1),
                torch.bmm(delta_z.unsqueeze(1), J_inv),
            )
            J_inv = J_inv + numerator / denom.unsqueeze(-1).unsqueeze(-1)

        z = z_new
        g = g_new

    info = {
        "iterations": min(k + 1, max_iter),
        "residuals": residuals,
        "rel_residuals": rel_residuals,
        "converged": rel_residuals[-1] < tol,
    }
    return z.reshape(z_init.shape), info


# ---------------------------------------------------------------------------
# DEQ autograd function
# ---------------------------------------------------------------------------


class DEQFixedPoint(autograd.Function):
    """Custom autograd function for DEQ forward/backward passes.

    Forward: Find z* such that f(z*, x) = z* using the configured solver.
    Backward: Use the Implicit Function Theorem to compute gradients.

    The IFT gives us:
        dL/dx = dL/dz* · (I - df/dz*)^{-1} · df/dx

    We compute (I - df/dz*)^{-1} · v via solving the linear system
    (I - (df/dz*)^T) · g = v using fixed-point iteration on the adjoint.
    """

    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        func: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        z_init: Tensor,
        solver_type: SolverType,
        max_iter: int,
        tol: float,
        anderson_m: int,
        anderson_beta: float,
        jfb: bool,
    ) -> Tensor:
        """Find the fixed point z* = f(z*, x)."""
        # Select solver
        with torch.no_grad():
            if solver_type == SolverType.ANDERSON:
                z_star, info = _anderson_acceleration(
                    func, x, z_init, max_iter, tol, m=anderson_m, beta=anderson_beta
                )
            elif solver_type == SolverType.BROYDEN:
                z_star, info = _broyden_solver(func, x, z_init, max_iter, tol)
            else:  # PICARD
                z_star, info = _picard_iteration(func, x, z_init, max_iter, tol)

        # Save for backward
        ctx.save_for_backward(z_star.detach(), x.detach())
        ctx.func = func  # type: ignore[attr-defined]
        ctx.max_iter = max_iter  # type: ignore[attr-defined]
        ctx.tol = tol  # type: ignore[attr-defined]
        ctx.jfb = jfb  # type: ignore[attr-defined]
        ctx.solver_info = info  # type: ignore[attr-defined]

        return z_star

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[None, Tensor, None, None, None, None, None, None, None]:
        """Compute gradients via the Implicit Function Theorem.

        We need to solve: g = (I - (df/dz)^T)^{-1} · grad_output

        If jfb=True, we use the Jacobian-Free Backprop approximation
        (just use grad_output directly, skipping the linear solve).
        """
        z_star, x = ctx.saved_tensors  # type: ignore[attr-defined]
        func = ctx.func  # type: ignore[attr-defined]
        jfb = ctx.jfb  # type: ignore[attr-defined]

        z_star = z_star.detach().requires_grad_()
        x = x.detach().requires_grad_()

        if jfb:
            # Jacobian-Free Backprop: skip the adjoint solve
            # Just compute df/dx at z* with grad_output as the vector
            with torch.enable_grad():
                f_val = func(z_star, x)
            grad_x = autograd.grad(f_val, x, grad_output, retain_graph=False)[0]
            return None, grad_x, None, None, None, None, None, None, None

        # Full IFT backward: solve adjoint fixed-point
        with torch.enable_grad():
            f_val = func(z_star, x)

        # Fixed-point iteration on the adjoint:
        # g_{k+1} = grad_output + (df/dz)^T · g_k
        g = grad_output.clone()
        for _ in range(ctx.max_iter):  # type: ignore[attr-defined]
            with torch.enable_grad():
                vjp_z = autograd.grad(f_val, z_star, g, retain_graph=True)[0]
            g_next = grad_output + vjp_z
            residual = torch.norm(g_next - g).item()
            g = g_next
            if residual < ctx.tol:  # type: ignore[attr-defined]
                break

        # Now compute df/dx using the solved adjoint vector g
        with torch.enable_grad():
            grad_x = autograd.grad(f_val, x, g, retain_graph=False)[0]

        return None, grad_x, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# DEQ Layer (nn.Module)
# ---------------------------------------------------------------------------


class DEQLayer(nn.Module):
    """Deep Equilibrium Layer.

    Replaces explicit depth with an implicit fixed-point solver.
    Acts as an infinitely deep network with O(1) memory.

    Args:
        func: The fixed-point map z = f(z, x). Must take (z, x) and return
            a tensor with the same shape as z.
        config: DEQ configuration (solver type, tolerances, etc.).

    Example::

        def transform(z, x):
            return torch.tanh(linear(torch.cat([z, x], dim=-1)))

        deq = DEQLayer(transform, config=DEQConfig(solver=SolverType.ANDERSON))
        z_star = deq(x)  # Finds the equilibrium state

    Attributes:
        last_info: Dict with solver diagnostics from the last forward pass
            (iterations, residuals, convergence status).
    """

    def __init__(
        self,
        func: Callable[[Tensor, Tensor], Tensor],
        config: DEQConfig | None = None,
    ) -> None:
        super().__init__()
        self.func = func
        self.config = config or DEQConfig()
        self.last_info: dict[str, object] = {}

    def forward(self, x: Tensor) -> Tensor:
        """Find the equilibrium state z* = f(z*, x).

        Args:
            x: Input tensor. The fixed point z* will have the same batch
                dimensions as x.

        Returns:
            The equilibrium state z*.
        """
        # Initialize z with zeros
        with torch.no_grad():
            z_init = torch.zeros_like(x)
            # Determine output shape from a test forward
            test_out = self.func(z_init, x)
            z_init = torch.zeros_like(test_out)

            # Call solver directly to capture solver diagnostics
            if self.config.solver == SolverType.ANDERSON:
                z_star, info = _anderson_acceleration(
                    self.func,
                    x,
                    z_init,
                    self.config.max_iter,
                    self.config.tol,
                    m=self.config.anderson_m,
                    beta=self.config.anderson_beta,
                )
            elif self.config.solver == SolverType.BROYDEN:
                z_star, info = _broyden_solver(
                    self.func, x, z_init, self.config.max_iter, self.config.tol
                )
            else:  # PICARD
                z_star, info = _picard_iteration(
                    self.func, x, z_init, self.config.max_iter, self.config.tol
                )

            # Store solver info for external inspection
            self.last_info = info

        # Apply DEQFixedPoint for autograd (it will re-solve to be safe,
        # but could be optimized to reuse z_star)
        z_star = cast(
            Tensor,
            DEQFixedPoint.apply(
                self.func,
                x,
                z_init,
                self.config.solver,
                self.config.max_iter,
                self.config.tol,
                self.config.anderson_m,
                self.config.anderson_beta,
                self.config.jfb,
            ),
        )

        return z_star


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to all linear/conv layers in a module.

    This ensures the transformation f has Lipschitz constant ≤ 1,
    which guarantees the existence and uniqueness of the fixed point
    (Banach fixed-point theorem).

    Args:
        module: The nn.Module containing the transformation layers.

    Returns:
        The module with spectral normalization applied.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            setattr(module, name, nn.utils.parametrizations.spectral_norm(child))
        else:
            apply_spectral_norm(child)
    return module
