"""TDD for the Expected Free Energy experiment selector.

The first version of this agent scored every action identically, which made it
decoration rather than a selector, and its tests passed anyway because they only
checked that a number came back. The tests here are written so that the same
stub could not survive them: each one would fail against an agent that ignores
the action, ignores cost, or collapses the two terms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.research.efe import (
    EFEAgent,
    Experiment,
    Hypothesis,
    standing_candidates,
)


def _sharp(hyp: Hypothesis, cost: float = 0.0, payoff: float = 1.0) -> Experiment:
    """An experiment that separates one hypothesis cleanly."""
    return Experiment(
        name=f"sharp_{hyp.name}_{cost}",
        gpu_hours=cost,
        diagnosticity={hyp: (0.95, 0.05)},
        payoff={hyp: payoff},
    )


def _blunt(hyp: Hypothesis, cost: float = 0.0) -> Experiment:
    """An experiment whose outcome is the same whether or not the hypothesis holds."""
    return Experiment(
        name=f"blunt_{hyp.name}_{cost}",
        gpu_hours=cost,
        diagnosticity={hyp: (0.5, 0.5)},
        payoff={hyp: 1.0},
    )


class TestTheAgentActuallyDiscriminates:
    """The failure that made the first implementation worthless."""

    def test_different_experiments_receive_different_scores(self) -> None:
        agent = EFEAgent()
        scores = {s.experiment: s.total for s in agent.rank(standing_candidates())}
        assert len(set(round(v, 9) for v in scores.values())) == len(scores), (
            f"an agent that scores every action alike cannot select: {scores}"
        )

    def test_an_uninformative_experiment_yields_no_information(self) -> None:
        """Equal likelihoods under both branches must give exactly zero gain."""
        agent = EFEAgent()
        assert agent.epistemic_value(_blunt(Hypothesis.VERIFICATION_BEATS_AGGREGATION)) == pytest.approx(0.0, abs=1e-12)

    def test_a_sharp_experiment_beats_a_blunt_one_of_equal_cost(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        sharp = agent.expected_free_energy(_sharp(hyp, cost=1.0))
        blunt = agent.expected_free_energy(_blunt(hyp, cost=1.0))
        assert sharp.epistemic > blunt.epistemic
        assert sharp.total < blunt.total

    def test_information_gain_is_largest_when_the_agent_is_most_unsure(self) -> None:
        """A hypothesis already all but settled is not worth another experiment."""
        hyp = Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING
        unsure = EFEAgent({hyp: 0.5}).epistemic_value(_sharp(hyp))
        settled = EFEAgent({hyp: 0.97}).epistemic_value(_sharp(hyp))
        assert unsure > settled


class TestCostEntersTheDecision:
    def test_the_cheaper_of_two_equally_sharp_experiments_wins(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING
        cheap = Experiment(
            name="cheap", gpu_hours=0.0,
            diagnosticity={hyp: (0.9, 0.1)}, payoff={hyp: 1.0},
        )
        dear = Experiment(
            name="dear", gpu_hours=40.0,
            diagnosticity={hyp: (0.9, 0.1)}, payoff={hyp: 1.0},
        )
        assert agent.select([dear, cheap]).name == "cheap"

    def test_a_large_enough_cost_overturns_a_sharper_experiment(self) -> None:
        """Cost must be able to change the ordering, not merely tie-break it."""
        agent = EFEAgent()
        hyp = Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING
        sharper_but_costly = Experiment(
            name="sharper", gpu_hours=100.0,
            diagnosticity={hyp: (0.95, 0.05)}, payoff={hyp: 1.0},
        )
        duller_but_free = Experiment(
            name="duller", gpu_hours=0.0,
            diagnosticity={hyp: (0.75, 0.25)}, payoff={hyp: 1.0},
        )
        assert agent.select([sharper_but_costly, duller_but_free]).name == "duller"
        # And with cost disregarded the ordering reverses, which shows the
        # previous assertion was decided by cost rather than by diagnosticity.
        indifferent = EFEAgent(cost_weight=0.0)
        assert indifferent.select([sharper_but_costly, duller_but_free]).name == "sharper"


class TestBothTermsAreRealAndSeparable:
    def test_the_two_terms_are_reported_separately_and_differ(self) -> None:
        agent = EFEAgent()
        score = agent.expected_free_energy(standing_candidates()[0])
        assert score.epistemic > 0
        assert score.pragmatic > 0
        assert score.epistemic != score.pragmatic

    def test_payoff_moves_the_ranking_with_diagnosticity_held_fixed(self) -> None:
        """Two experiments equally informative but unequally consequential."""
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        valuable = Experiment(
            name="valuable", gpu_hours=1.0,
            diagnosticity={hyp: (0.9, 0.1)}, payoff={hyp: 5.0},
        )
        idle = Experiment(
            name="idle", gpu_hours=1.0,
            diagnosticity={hyp: (0.9, 0.1)}, payoff={hyp: 0.0},
        )
        assert agent.expected_free_energy(valuable).epistemic == pytest.approx(
            agent.expected_free_energy(idle).epistemic
        )
        assert agent.select([idle, valuable]).name == "valuable"

    def test_total_is_the_stated_combination(self) -> None:
        agent = EFEAgent()
        s = agent.expected_free_energy(standing_candidates()[0])
        assert s.total == pytest.approx(-(s.epistemic + s.pragmatic) + s.cost)


class TestBeliefUpdate:
    def test_a_positive_outcome_raises_the_belief_it_bears_on(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        before = agent.beliefs[hyp]
        agent.observe(_sharp(hyp), hyp, positive=True)
        assert agent.beliefs[hyp] > before

    def test_a_negative_outcome_lowers_it(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        agent.observe(_sharp(hyp), hyp, positive=False)
        assert agent.beliefs[hyp] < 0.5

    def test_other_hypotheses_are_left_alone(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        others = {h: p for h, p in agent.beliefs.items() if h is not hyp}
        agent.observe(_sharp(hyp), hyp, positive=True)
        for h, p in others.items():
            assert agent.beliefs[h] == p

    def test_a_blunt_experiment_teaches_nothing(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        agent.observe(_blunt(hyp), hyp, positive=True)
        assert agent.beliefs[hyp] == pytest.approx(0.5)

    def test_updating_on_an_experiment_that_carries_no_evidence_is_refused(self) -> None:
        """Silently permitting this would let the agent invent information."""
        agent = EFEAgent()
        with pytest.raises(ValueError, match="no evidence"):
            agent.observe(
                _sharp(Hypothesis.VERIFICATION_BEATS_AGGREGATION),
                Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING,
                positive=True,
            )

    def test_entropy_falls_as_evidence_accumulates(self) -> None:
        agent = EFEAgent()
        hyp = Hypothesis.VERIFICATION_BEATS_AGGREGATION
        start = agent.total_entropy()
        for _ in range(3):
            agent.observe(_sharp(hyp), hyp, positive=True)
        assert agent.total_entropy() < start

    def test_convergence_counts_confident_refutation_as_settled(self) -> None:
        """Most of this programme's hypotheses were settled by being refuted."""
        agent = EFEAgent({h: 0.5 for h in Hypothesis})
        for hyp in Hypothesis:
            for _ in range(6):
                agent.observe(_sharp(hyp), hyp, positive=False)
        assert agent.has_converged()
        assert all(p < 0.05 for p in agent.beliefs.values())


class TestAgainstTheRealRecord:
    """The agent should reproduce a choice the programme made and got right."""

    def test_it_prefers_the_free_simulation_over_the_training_run(self) -> None:
        """After F30, the aggregation rules were exhausted and the open question
        was whether better players would help. Two actions bore on it: an offline
        simulation costing no GPU time, and a distillation run costing eight
        hours. The simulation was chosen, and it settled the question for free.
        An agent that cannot recover that ordering is not encoding the
        programme's actual practice."""
        agent = EFEAgent(
            {
                Hypothesis.VERIFICATION_BEATS_AGGREGATION: 0.2,
                Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING: 0.4,
                Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING: 0.65,
                Hypothesis.COMPLEMENTARITY_GENERALISES: 0.25,
            }
        )
        candidates = standing_candidates()
        ranked = [s.experiment for s in agent.rank(candidates)]
        assert ranked.index("exp24_coupling_simulation") < ranked.index(
            "teacher_distillation_run"
        ), f"ranking was {ranked}"

    def test_settling_a_question_stops_it_being_asked_again(self) -> None:
        """Once the simulation has refuted the better-players route, the agent
        should stop proposing experiments about it."""
        agent = EFEAgent()
        candidates = standing_candidates()
        sim = next(e for e in candidates if e.name == "exp24_coupling_simulation")
        distil = next(e for e in candidates if e.name == "teacher_distillation_run")

        before = agent.expected_free_energy(distil).epistemic
        for _ in range(5):
            agent.observe(sim, Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING, positive=False)
        after = agent.expected_free_energy(distil).epistemic
        assert after < before

    def test_the_report_names_the_hypotheses_and_the_ranking(self) -> None:
        agent = EFEAgent()
        text = agent.report(standing_candidates())
        assert "VERIFICATION_BEATS_AGGREGATION" in text
        assert "exp23_cross_examination" in text
        assert "epistemic" in text


class TestConstruction:
    def test_certainty_is_refused_as_a_starting_belief(self) -> None:
        with pytest.raises(ValueError, match="strictly inside"):
            EFEAgent({Hypothesis.VERIFICATION_BEATS_AGGREGATION: 1.0})

    def test_a_likelihood_outside_zero_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="probability"):
            Experiment(
                name="bad", gpu_hours=0.0,
                diagnosticity={Hypothesis.VERIFICATION_BEATS_AGGREGATION: (1.4, 0.1)},
            )

    def test_negative_cost_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            Experiment(
                name="bad", gpu_hours=-1.0,
                diagnosticity={Hypothesis.VERIFICATION_BEATS_AGGREGATION: (0.9, 0.1)},
            )

    def test_selecting_from_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no candidate"):
            EFEAgent().select([])


class TestSettledQuestionsStopBeingWorthAsking:
    """Found by running the loop: an experiment whose hypothesis is already
    settled kept ranking first, because confirming a belief already held scored
    as progress. Progress means changing what would be done next."""

    def test_pragmatic_value_falls_sharply_as_a_hypothesis_settles(self) -> None:
        """The scaling is 4q(1-q), so a belief at 0.97 keeps about a ninth of the
        value it had at even odds. The threshold below is deliberately loose: the
        property under test is that settling costs most of the value, not that it
        costs some particular fraction, and pinning an exact constant would make
        the test a restatement of the formula rather than a check on it."""
        hyp = Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING
        exp = _sharp(hyp, cost=0.0, payoff=1.0)
        open_q = EFEAgent({hyp: 0.5}).pragmatic_value(exp)
        settled = EFEAgent({hyp: 0.97}).pragmatic_value(exp)
        assert settled < 0.3 * open_q

    def test_an_open_question_outranks_a_settled_one_at_equal_cost(self) -> None:
        settled = Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING
        still_open = Hypothesis.COMPLEMENTARITY_GENERALISES
        agent = EFEAgent({settled: 0.97, still_open: 0.5})
        chosen = agent.select([
            _sharp(settled, cost=1.0, payoff=1.0),
            _sharp(still_open, cost=1.0, payoff=1.0),
        ])
        assert chosen.diagnosticity == {still_open: (0.95, 0.05)}
