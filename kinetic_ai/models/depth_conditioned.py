"""Depth conditioning: one block that behaves differently at each iteration.

F45 measured what weight tying costs at equal compute — the shared block reaches
96.8% of a twelve-layer explicit transformer with 2.70 times fewer parameters —
and located the residual 3.2% precisely. Twelve applications of one static map
cannot express what twelve distinct maps can, because every application is the
same function.

The TRIZ session for this gap (matrix cell 36/26, principles Local Quality and
Dynamics) points at differentiating the map in time rather than in parameters.
The block is modulated by a small per-iteration scale and shift, so the same
weights implement a different function at depth three than at depth nine, and the
fixed point becomes the equilibrium of a periodically time-varying system rather
than of a static one. The cost is two vectors of width ``d_model`` per iteration:
roughly eighteen thousand parameters against the seventy-eight million that
untying the block would add, and no extra arithmetic at all, since modulation is
elementwise.

This is deliberately the smallest intervention that could close the gap. If a
few thousand parameters of depth-specific behaviour recover most of what twelve
separate blocks provide, the finding is that tying loses expressiveness rather
than capacity, and the remedy is cheap. If they recover nothing, the loss is
capacity and no modulation scheme will help, which is equally worth knowing.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DepthFiLM(nn.Module):
    """Per-iteration scale and shift for a weight-tied block.

    Args:
        d_model: Width of the hidden state being modulated.
        max_depth: Number of iterations that receive their own modulation.
            Iterations beyond this reuse the final one, so a model trained at
            twelve iterations still runs at twenty without an index error — the
            anytime property F24 established has to survive this change.

    The scale is parameterised around one and the shift around zero, so an
    untrained module is the identity and the modulated model starts exactly
    where the unmodulated one does. Without that the comparison would confound
    depth conditioning with a different initialisation.
    """

    def __init__(self, d_model: int, max_depth: int = 12) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.gamma = nn.Parameter(torch.ones(max_depth, d_model))
        self.beta = nn.Parameter(torch.zeros(max_depth, d_model))

    def forward(self, z: Tensor, step: int) -> Tensor:
        idx = min(step, self.max_depth - 1)
        return self.gamma[idx] * z + self.beta[idx]

    def extra_repr(self) -> str:
        return f"max_depth={self.max_depth}, params={self.gamma.numel() * 2}"


class DepthConditionedBlock(nn.Module):
    """Wraps a tied block so that its map depends on the iteration index.

    The wrapped block is left untouched, which matters for the comparison: the
    depth-conditioned model and the plain tied model share an architecture and
    an initialisation, and differ only by a modulation that begins as the
    identity. Any quality difference is therefore attributable to depth
    conditioning rather than to a second architectural change smuggled in
    alongside it.
    """

    def __init__(self, block: nn.Module, d_model: int, max_depth: int = 12) -> None:
        super().__init__()
        self.block = block
        self.film = DepthFiLM(d_model, max_depth)
        self._step = 0

    def reset(self) -> None:
        """Begin a new solve. Callers that iterate manually must call this."""
        self._step = 0

    def forward(self, z: Tensor, x: Tensor) -> Tensor:
        out: Tensor = self.block(z, x)
        out = self.film(out, self._step)
        self._step += 1
        return out


def count_conditioning_parameters(d_model: int, max_depth: int = 12) -> int:
    """What depth conditioning costs, for reporting beside what it buys."""
    return 2 * d_model * max_depth
