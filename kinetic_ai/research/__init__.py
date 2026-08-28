"""Autoresearch: choosing what to measure next.

The programme's cycle is driven by which experiment reduces the most uncertainty
about what to build, weighed against what the experiment costs. This package
holds that decision as code rather than as judgement, so it can be inspected and
argued with.
"""

from kinetic_ai.research.efe import (
    HYPOTHESIS_TEXT,
    EFEAgent,
    EFEScore,
    Experiment,
    Hypothesis,
    standing_candidates,
)

__all__ = [
    "EFEAgent",
    "EFEScore",
    "Experiment",
    "Hypothesis",
    "HYPOTHESIS_TEXT",
    "standing_candidates",
]
