"""Expected Free Energy over candidate experiments.

The programme's practice has been to prefer cheap probes that resolve a lot and
to defer expensive ones that resolve little, and to justify that preference
after the fact. This module makes the preference a computation instead, so that
the choice of next experiment can be inspected, disagreed with, and checked
against what was actually run.

The construction follows the active-inference treatment used in this group's
earlier work: an agent holds beliefs over hidden state, and scores each
available action by the Expected Free Energy it would incur, which decomposes
into an epistemic term rewarding expected information gain and a pragmatic term
rewarding expected progress toward what is wanted. The action minimising the
total is selected.

Two things make this a real selector rather than a label. The likelihood of an
outcome is conditioned on the *action*, so an experiment that cannot discriminate
a hypothesis contributes no information about it and an experiment that can
contributes a great deal; without that conditioning every action scores alike and
the agent is decorative. And the cost of running an experiment enters the score
directly, so a probe that resolves slightly less for no GPU time can and does
outrank a training run that resolves slightly more for eight hours of it. That
ordering is the one the programme arrived at by judgement, and reproducing it
from the arithmetic is what the retrospective test in the suite checks.

Beliefs are held as independent marginals over binary hypotheses rather than as a
joint distribution. The hypotheses concern different parts of the system —
whether verification beats aggregation, whether better players help, whether the
solve is affordable, whether the measured complementarity generalises — and
treating them as independent keeps the update legible at the cost of ignoring
correlations that no measurement has yet estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Hypothesis(IntEnum):
    """The binary questions whose answers decide what to build.

    Each is a claim that a measurement could settle, and each maps to a branch
    of the plan: the programme proceeds differently depending on which are true.
    """

    VERIFICATION_BEATS_AGGREGATION = 0
    BETTER_PLAYERS_RAISE_THE_CEILING = 1
    SOLVE_IS_AFFORDABLE_AT_SERVING = 2
    COMPLEMENTARITY_GENERALISES = 3


#: Human-readable statement of each hypothesis, for reporting.
HYPOTHESIS_TEXT: dict[Hypothesis, str] = {
    Hypothesis.VERIFICATION_BEATS_AGGREGATION: (
        "Selecting among generated candidates by cross-examination beats the best "
        "single player by more than seed noise"
    ),
    Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING: (
        "Building or distilling stronger domain specialists raises what the "
        "council can reach, rather than the aggregation rule being the constraint"
    ),
    Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING: (
        "The equilibrium solve and its forward passes fit within a small multiple "
        "of single-model serving latency"
    ),
    Hypothesis.COMPLEMENTARITY_GENERALISES: (
        "The complementarity measured on this council holds for other model "
        "families rather than being an artefact of one"
    ),
}


@dataclass(frozen=True)
class Experiment:
    """A candidate action: something that could be run, and what it would tell.

    Attributes:
        name: Identifier used in reports and rankings.
        gpu_hours: What running it costs. Zero for offline analysis over results
            already on disk, which is the class of action this project has
            repeatedly found underrated.
        diagnosticity: Per hypothesis, the probability of observing a positive
            outcome when the hypothesis is true and when it is false. Equal
            values mean the experiment cannot discriminate that hypothesis and
            it therefore yields no information about it; the further apart they
            are, the sharper the experiment. Hypotheses absent from the mapping
            are untouched by this experiment.
        payoff: Per hypothesis, how much confirming it advances the goal of
            beating the baseline. This is what makes the agent prefer resolving
            questions that matter over questions that are merely open.
        description: What the experiment does, for the report.
    """

    name: str
    gpu_hours: float
    diagnosticity: dict[Hypothesis, tuple[float, float]]
    payoff: dict[Hypothesis, float] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        for hyp, (p_true, p_false) in self.diagnosticity.items():
            for p in (p_true, p_false):
                if not 0.0 <= p <= 1.0:
                    raise ValueError(
                        f"{self.name}: likelihood for {hyp.name} must be a "
                        f"probability, got {p}"
                    )
        if self.gpu_hours < 0:
            raise ValueError(f"{self.name}: gpu_hours cannot be negative")


def _bernoulli_kl(posterior: float, prior: float) -> float:
    """KL divergence between two Bernoulli distributions, in nats."""
    p = float(np.clip(posterior, 1e-12, 1 - 1e-12))
    q = float(np.clip(prior, 1e-12, 1 - 1e-12))
    return float(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q)))


@dataclass
class EFEScore:
    """The decomposition behind one action's score.

    Both terms are kept separate because they answer different questions and can
    disagree: an experiment may be highly informative about something that does
    not matter, or highly valuable to confirm while telling us little we did not
    already believe. Collapsing them into one number hides that.
    """

    experiment: str
    epistemic: float
    pragmatic: float
    cost: float
    total: float

    def __str__(self) -> str:
        return (
            f"{self.experiment}: G={self.total:+.4f} "
            f"(epistemic {self.epistemic:.4f}, pragmatic {self.pragmatic:.4f}, "
            f"cost {self.cost:.4f})"
        )


class EFEAgent:
    """Selects the next experiment by minimising Expected Free Energy.

    Args:
        beliefs: Initial probability that each hypothesis is true. Defaults to
            complete ignorance at one half apiece.
        cost_weight: How many nats of information one GPU-hour is worth trading
            away. Raising it makes the agent more parsimonious; the default is
            calibrated so that a whole day of GPU time is worth roughly one nat,
            which reproduces this programme's revealed preference for cheap
            probes without making expensive experiments unreachable.
    """

    def __init__(
        self,
        beliefs: dict[Hypothesis, float] | None = None,
        cost_weight: float = 1.0 / 24.0,
    ) -> None:
        self.beliefs: dict[Hypothesis, float] = {
            h: 0.5 for h in Hypothesis
        }
        if beliefs:
            for hyp, p in beliefs.items():
                if not 0.0 < p < 1.0:
                    raise ValueError(
                        f"belief for {hyp.name} must lie strictly inside (0, 1), "
                        f"got {p}; certainty admits no update"
                    )
                self.beliefs[hyp] = p
        self.cost_weight = cost_weight
        self.history: list[tuple[str, Hypothesis, bool]] = []
        self._entropy_trace: list[float] = [self.total_entropy()]

    # ── belief state ────────────────────────────────────────────────────────

    def total_entropy(self) -> float:
        """Summed binary entropy over the hypotheses, in nats.

        This is the quantity the epistemic term is trying to reduce, so tracking
        it across a campaign shows whether the programme is actually learning or
        merely running experiments.
        """
        total = 0.0
        for p in self.beliefs.values():
            q = float(np.clip(p, 1e-12, 1 - 1e-12))
            total += -(q * np.log(q) + (1 - q) * np.log(1 - q))
        return total

    # ── scoring ─────────────────────────────────────────────────────────────

    def epistemic_value(self, experiment: Experiment) -> float:
        """Expected information gain, in nats, from running this experiment.

        Averaged over the outcomes the experiment could produce, weighted by how
        likely each is under current belief. An experiment whose positive rate is
        the same whether or not a hypothesis holds contributes exactly zero here,
        which is the property that makes the ranking depend on the action.
        """
        gain = 0.0
        for hyp, (p_true, p_false) in experiment.diagnosticity.items():
            prior = self.beliefs[hyp]
            p_pos = prior * p_true + (1 - prior) * p_false
            p_pos = float(np.clip(p_pos, 1e-12, 1 - 1e-12))

            post_pos = prior * p_true / p_pos
            post_neg = prior * (1 - p_true) / (1 - p_pos)

            gain += p_pos * _bernoulli_kl(post_pos, prior)
            gain += (1 - p_pos) * _bernoulli_kl(post_neg, prior)
        return gain

    def pragmatic_value(self, experiment: Experiment) -> float:
        """Expected progress toward the goal, in the same units.

        An experiment earns this by being likely to confirm something that
        matters. A probe that can only ever return bad news still has epistemic
        value — knowing a route is closed redirects effort — but it earns little
        here, which is the intended asymmetry.

        The value is scaled by how unsettled the hypothesis still is, via
        ``4q(1-q)``, which is one at maximum uncertainty and zero at either
        certainty. Without that scaling the agent keeps proposing experiments
        whose answers it already has: running this loop for real surfaced exactly
        that, with a latency measurement still ranked first after the
        measurement had been made and the hypothesis driven to 0.944, because
        confirming something already believed still scored as progress. Progress
        means changing what would be done next, and an outcome that is already
        expected changes nothing.
        """
        value = 0.0
        for hyp, payoff in experiment.payoff.items():
            if hyp not in experiment.diagnosticity:
                continue
            p_true, p_false = experiment.diagnosticity[hyp]
            prior = self.beliefs[hyp]
            p_confirms = prior * p_true + (1 - prior) * p_false
            still_open = 4.0 * prior * (1.0 - prior)
            value += payoff * p_confirms * still_open
        return value

    def expected_free_energy(self, experiment: Experiment) -> EFEScore:
        """Score one experiment. Lower is better."""
        epistemic = self.epistemic_value(experiment)
        pragmatic = self.pragmatic_value(experiment)
        cost = self.cost_weight * experiment.gpu_hours
        return EFEScore(
            experiment=experiment.name,
            epistemic=epistemic,
            pragmatic=pragmatic,
            cost=cost,
            total=-(epistemic + pragmatic) + cost,
        )

    def rank(self, experiments: list[Experiment]) -> list[EFEScore]:
        """Score every candidate, best first."""
        return sorted(
            (self.expected_free_energy(e) for e in experiments),
            key=lambda s: s.total,
        )

    def select(self, experiments: list[Experiment]) -> Experiment:
        """The experiment to run next."""
        if not experiments:
            raise ValueError("no candidate experiments to choose between")
        best = self.rank(experiments)[0]
        return next(e for e in experiments if e.name == best.experiment)

    # ── learning ────────────────────────────────────────────────────────────

    def observe(
        self, experiment: Experiment, hypothesis: Hypothesis, positive: bool
    ) -> float:
        """Fold an experiment's outcome into the belief over one hypothesis.

        Args:
            experiment: What was run; its diagnosticity supplies the likelihood.
            hypothesis: Which question the outcome bears on.
            positive: Whether the outcome supported the hypothesis.

        Returns:
            The information actually gained, in nats, which will differ from the
            expected gain and is worth recording — a campaign whose realised
            gains fall persistently short of expectation has a miscalibrated
            model of its own experiments.
        """
        if hypothesis not in experiment.diagnosticity:
            raise ValueError(
                f"{experiment.name} carries no evidence about {hypothesis.name}; "
                "updating on it would invent information"
            )
        p_true, p_false = experiment.diagnosticity[hypothesis]
        prior = self.beliefs[hypothesis]

        like_true = p_true if positive else 1 - p_true
        like_false = p_false if positive else 1 - p_false
        evidence = prior * like_true + (1 - prior) * like_false
        if evidence <= 0:
            raise ValueError(
                f"{experiment.name} assigns zero probability to the observed "
                f"outcome for {hypothesis.name}; the likelihood is wrong"
            )
        posterior = prior * like_true / evidence

        self.beliefs[hypothesis] = float(np.clip(posterior, 1e-9, 1 - 1e-9))
        self.history.append((experiment.name, hypothesis, positive))
        self._entropy_trace.append(self.total_entropy())
        return _bernoulli_kl(self.beliefs[hypothesis], prior)

    def has_converged(self, threshold: float = 0.05) -> bool:
        """Whether every hypothesis is settled enough to stop asking.

        Settled means confident either way: a hypothesis driven to near-certainly
        false is as much a result as one driven to near-certainly true, and the
        programme's record contains more of the former.
        """
        return all(
            p < threshold or p > 1 - threshold for p in self.beliefs.values()
        )

    def report(self, experiments: list[Experiment]) -> str:
        """A human-readable ranking, for the application and the journal."""
        lines = [
            f"belief entropy {self.total_entropy():.4f} nats over "
            f"{len(self.beliefs)} hypotheses",
        ]
        for hyp, p in self.beliefs.items():
            lines.append(f"  P({hyp.name}) = {p:.3f}")
        lines.append("ranked candidates, lowest expected free energy first:")
        for score in self.rank(experiments):
            lines.append(f"  {score}")
        return "\n".join(lines)


def standing_candidates() -> list[Experiment]:
    """The experiments actually available to this programme, with their costs.

    Diagnosticity values are judgements about how sharply each experiment would
    separate its hypothesis, not measurements, and they are the part of this
    module most worth disagreeing with. They are stated here rather than buried
    so that a disagreement can be expressed as an edit.
    """
    return [
        Experiment(
            name="exp23_cross_examination",
            gpu_hours=3.0,
            diagnosticity={
                Hypothesis.VERIFICATION_BEATS_AGGREGATION: (0.85, 0.15),
                Hypothesis.COMPLEMENTARITY_GENERALISES: (0.6, 0.4),
            },
            payoff={Hypothesis.VERIFICATION_BEATS_AGGREGATION: 1.0},
            description=(
                "Every player writes a full solution and prices every peer's; "
                "selection rules operate on the resulting valuations"
            ),
        ),
        Experiment(
            name="exp24_coupling_simulation",
            gpu_hours=0.0,
            diagnosticity={
                Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING: (0.75, 0.25),
            },
            payoff={Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING: 0.5},
            description=(
                "Offline sweep over confidence-competence coupling on stored "
                "scores, costing no GPU time"
            ),
        ),
        Experiment(
            name="teacher_distillation_run",
            gpu_hours=8.0,
            diagnosticity={
                Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING: (0.8, 0.2),
            },
            payoff={Hypothesis.BETTER_PLAYERS_RAISE_THE_CEILING: 1.0},
            description=(
                "Distil a domain specialist and re-measure the council's ceiling"
            ),
        ),
        Experiment(
            name="serving_latency_measurement",
            gpu_hours=0.5,
            diagnosticity={
                Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING: (0.9, 0.1),
            },
            payoff={Hypothesis.SOLVE_IS_AFFORDABLE_AT_SERVING: 0.8},
            description=(
                "Wall-clock the council against single-model serving on matched "
                "prompts"
            ),
        ),
        Experiment(
            name="second_model_family_check",
            gpu_hours=2.0,
            diagnosticity={
                Hypothesis.COMPLEMENTARITY_GENERALISES: (0.85, 0.15),
            },
            payoff={Hypothesis.COMPLEMENTARITY_GENERALISES: 0.6},
            description=(
                "Repeat the complementarity measurement on a council drawn from "
                "a different model family"
            ),
        ),
    ]
