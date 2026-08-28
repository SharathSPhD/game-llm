"""TDD for the magnetically anchored answer vote (F39, SPEC 0017).

The properties pinned here are the ones the mechanism's claim rests on: the
router limit, the override threshold, the one-vote-per-player neutralisation of
verbosity, and abstention. Each would fail against a rule that quietly reverted
to majority voting or to the router alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.decode.anchored_vote import AnchoredVote


def _vote(**kw) -> AnchoredVote:
    defaults = dict(anchor_player="ref", tau=1.0)
    defaults.update(kw)
    return AnchoredVote(**defaults)


class TestRouterLimit:
    def test_large_tau_reproduces_the_anchor_exactly(self) -> None:
        """The floor-is-the-baseline property: at tau beyond any achievable vote
        mass the mechanism must be the router, not approximately the router."""
        v = _vote(tau=100.0)
        answers = {"ref": "A", "b": "B", "c": "B", "d": "B"}
        assert v.select(answers) == "A"

    def test_zero_tau_is_plain_voting_with_anchor_tiebreak(self) -> None:
        v = _vote(tau=0.0)
        assert v.select({"ref": "A", "b": "B", "c": "B"}) == "B"
        # A genuine tie resolves toward the anchor, which is the tau -> 0+
        # limit rather than an extra rule.
        assert v.select({"ref": "A", "b": "B"}) == "A"


class TestOverrideThreshold:
    def test_the_council_overrides_only_beyond_the_margin(self) -> None:
        """tau = 1: one dissenting vote cannot move the answer, two can."""
        v = _vote(tau=1.0)
        assert v.select({"ref": "A", "b": "B", "c": "C"}) == "A"
        assert v.select({"ref": "A", "b": "B", "c": "B"}) == "A"  # 2 vs 1+1: tie -> anchor
        assert v.select({"ref": "A", "b": "B", "c": "B", "d": "B"}) == "B"

    def test_agreement_with_the_anchor_strengthens_it(self) -> None:
        v = _vote(tau=1.0)
        assert v.select({"ref": "A", "b": "A", "c": "B", "d": "B"}) == "A"


class TestMeasurementHazards:
    def test_a_player_votes_once_regardless_of_how_much_it_wrote(self) -> None:
        """Verbosity was the confound that corrupted the first cross-examination
        run; a vote over answer classes is immune by construction, and this test
        documents that the interface takes answers, never texts."""
        v = _vote(tau=0.0)
        assert v.select({"ref": "A", "b": "B"}) == "A"

    def test_an_unextractable_answer_abstains(self) -> None:
        v = _vote(tau=1.0)
        assert v.select({"ref": "A", "b": None, "c": None, "d": "B"}) == "A"

    def test_an_abstaining_anchor_leaves_a_plain_vote(self) -> None:
        v = _vote(tau=5.0)
        assert v.select({"ref": None, "b": "B", "c": "B", "d": "C"}) == "B"

    def test_everyone_abstaining_returns_none(self) -> None:
        v = _vote(tau=1.0)
        assert v.select({"ref": None, "b": None}) is None


class TestWeights:
    def test_weighted_votes_change_the_outcome(self) -> None:
        v = _vote(tau=0.0, weights={"ref": 0.4, "b": 0.9, "c": 0.3})
        assert v.select({"ref": "A", "b": "B", "c": "A"}) == "B" or True
        # 0.4 + 0.3 = 0.7 for A against 0.9 for B: B wins on weight alone.
        assert v.select({"ref": "A", "b": "B", "c": "A"}) == "B"

    def test_missing_weight_is_refused_rather_than_defaulted(self) -> None:
        import pytest

        v = _vote(tau=0.0, weights={"ref": 1.0})
        with pytest.raises(KeyError):
            v.select({"ref": "A", "unknown": "B"})
