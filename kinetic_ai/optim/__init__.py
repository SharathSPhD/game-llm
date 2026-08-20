"""Optimization module: Magnetic Mirror Descent and Bregman divergences."""

from kinetic_ai.optim.bregman import (
    BregmanDivergence,
    DilatedEntropy,
    Euclidean,
    NegativeEntropy,
)
from kinetic_ai.optim.mmd import MagneticMirrorDescent, mmd_strategy_update

__all__ = [
    "BregmanDivergence",
    "NegativeEntropy",
    "Euclidean",
    "DilatedEntropy",
    "MagneticMirrorDescent",
    "mmd_strategy_update",
]
