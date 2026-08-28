"""TDD for council aggregation rules (F29, F30).

These rules produced the paper's answer-level results, so the properties that
made those results interpretable are pinned here: that the rules agree with each
other in the degenerate cases where they must, that each one is doing what its
name says on a case constructed to separate it from the mean, and that the
label-convention recovery which silently corrupted an earlier run cannot regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.decode.aggregate import (
    RULES,
    aggregate,
    oracle_any_correct,
    recover_label_offset,
)


def _ell(rows: list[list[list[float]]]) -> torch.Tensor:
    """[B, N, K] log-probabilities from raw scores."""
    return torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


class TestRulesAgreeWhereTheyMust:
    def test_unanimous_council_gives_the_same_answer_under_every_rule(self) -> None:
        """When the players agree, the choice of rule cannot matter — a rule that
        differs here is reacting to something other than the council's view."""
        ell = _ell([[[3.0, 0.0, 0.0], [4.0, 1.0, 0.0], [2.0, 0.5, 0.1]]])
        picks = {name: int(aggregate(ell, name)[0]) for name in RULES}
        assert set(picks.values()) == {0}, picks

    def test_single_player_council_is_that_player(self) -> None:
        ell = _ell([[[0.0, 5.0, 1.0]]])
        for name in RULES:
            assert int(aggregate(ell, name)[0]) == 1, name


class TestEachRuleDoesWhatItClaims:
    def test_mean_follows_the_loud_minority_and_median_does_not(self) -> None:
        """The case the median exists for: one emphatic dissenter against two
        mild agreeing players. This is also the shape F29 found damaging, where a
        confidently wrong specialist pulls the aggregate."""
        ell = _ell([[[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 20.0, 0.0]]])
        assert int(aggregate(ell, "mean")[0]) == 1
        assert int(aggregate(ell, "median")[0]) == 0

    def test_dropping_the_keenest_supporter_removes_a_single_champion(self) -> None:
        """Option 1 carries a large score from one player and nothing from the
        others, while option 0 is backed evenly. The scores are given directly
        rather than through a softmax, because normalising couples the columns:
        a player made more enthusiastic about one option is thereby made more
        hostile to the rest, which is a second effect these rules do not concern
        themselves with and which obscures what the test is checking."""
        scores = torch.tensor([[[1.0, 5.0], [1.0, 0.0], [1.0, 0.0]]])
        assert int(aggregate(scores, "mean")[0]) == 1
        assert int(aggregate(scores, "drop_keenest")[0]) == 0

    def test_leave_one_out_ignores_a_players_enthusiasm_for_its_own_proposal(self) -> None:
        """Player 0 proposes option 1 and supplies all of its support; players 1
        and 2 propose option 0 with mild backing each. Once a proposer's own
        score is struck from the price of its own proposal, option 1 is worth
        nothing to anybody and option 0 prevails, though the plain mean prefers
        option 1."""
        scores = torch.tensor([[[1.0, 5.0], [1.0, 0.0], [1.0, 0.0]]])
        assert int(aggregate(scores, "mean")[0]) == 1
        assert int(aggregate(scores, "leave_one_out_proposal")[0]) == 0

    def test_borda_ignores_magnitude(self) -> None:
        """Two councils with the same preference orderings but different
        strengths must give Borda the same answer, and the mean a different one."""
        mild = _ell([[[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.1, 0.0]]])
        loud = _ell([[[9.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.1, 0.0]]])
        assert int(aggregate(mild, "borda")[0]) == int(aggregate(loud, "borda")[0])
        assert int(aggregate(mild, "mean")[0]) != int(aggregate(loud, "mean")[0])

    def test_influence_game_at_zero_rationality_is_the_mean(self) -> None:
        """The degenerate case that makes the game a generalisation rather than
        an alternative: with no influence rationality it must reproduce averaging."""
        ell = _ell([[[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]]])
        assert int(aggregate(ell, "game", beta=0.0)[0]) == int(aggregate(ell, "mean")[0])


class TestOracle:
    def test_oracle_counts_a_question_solved_if_any_player_solves_it(self) -> None:
        ell = _ell([[[0.0, 3.0, 0.0], [3.0, 0.0, 0.0]]])
        assert oracle_any_correct(ell, torch.tensor([0])).item() == 1
        assert oracle_any_correct(ell, torch.tensor([2])).item() == 0

    def test_oracle_bounds_every_rule(self) -> None:
        torch.manual_seed(0)
        ell = torch.log_softmax(torch.randn(64, 3, 4), dim=-1)
        gold = torch.randint(0, 4, (64,))
        ceiling = float(oracle_any_correct(ell, gold).float().mean())
        for name in RULES:
            acc = float((aggregate(ell, name) == gold).float().mean())
            assert acc <= ceiling + 1e-9, f"{name} {acc} exceeds oracle {ceiling}"


class TestLabelConventionRecovery:
    """The defect this guards against scored players at 0.13 where the truth was
    0.63, and produced a plausible number rather than an error."""

    def test_zero_based_labels_are_recovered(self) -> None:
        scores = torch.tensor([[5.0, 0.0], [0.0, 5.0], [4.0, 0.0]])
        labels = torch.tensor([0, 1, 0])
        graded = torch.tensor([1.0, 1.0, 1.0])
        assert recover_label_offset(scores, labels, graded) == 0

    def test_one_based_labels_are_recovered(self) -> None:
        """WinoGrande's convention: labels run from one, so the position is the
        label minus one and reading it directly would index past the options."""
        scores = torch.tensor([[5.0, 0.0], [0.0, 5.0], [4.0, 0.0]])
        labels = torch.tensor([1, 2, 1])
        graded = torch.tensor([1.0, 1.0, 1.0])
        assert recover_label_offset(scores, labels, graded) == -1

    def test_irreconcilable_labels_are_refused_rather_than_guessed(self) -> None:
        """When neither convention reproduces the grading, the task's labels are
        not trustworthy and returning a best guess would launder the problem."""
        scores = torch.tensor([[5.0, 0.0], [0.0, 5.0], [4.0, 0.0]])
        labels = torch.tensor([0, 0, 1])
        graded = torch.tensor([1.0, 1.0, 1.0])
        assert recover_label_offset(scores, labels, graded) is None
