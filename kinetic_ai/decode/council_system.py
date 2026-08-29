"""The deployed council: route on measured priors, fall back on extraction failure.

This is the configuration that beat the baseline single model by 8.33 points on
360 pre-registered questions (F41), and it is deliberately small. A domain
router sends each prompt to whichever player the ladder measured to be strongest
on that domain; if that player's answer cannot be parsed — which happened on
16.1% of questions — the remaining players' answers are taken and the majority
class is returned.

It is worth being exact about the provenance, because the interesting part of
the construction is not what survived. The system is the ``tau -> infinity``
limit of the magnetically anchored vote: the council never overrides an
answering champion, which is what the confirmation measured as the correct
setting after finding that overriding at any finite threshold neither helped nor
hurt. The equilibrium construction supplied the form and the guarantee that this
limit contains the baseline, which is why the system cannot lose to the model it
routes on. It did not supply an operating point strictly inside the interval.

Cost is 1.25 expected generations per request against a single model's one,
since additional players are consulted only when the champion's answer fails to
parse.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class CouncilSystem:
    """Route to a per-domain champion, with council redundancy on failure.

    Args:
        champions: Domain to player, fixed in advance from ladder measurements.
            Deriving these from the questions being scored would be hindsight
            and is the reason they are passed in rather than computed.
        fallback_order: Optional per-domain order in which to consult the
            remaining players when the champion abstains. Defaults to majority
            vote over every non-abstaining player, which is what was measured.
    """

    champions: dict[str, str]
    fallback_order: dict[str, list[str]] | None = None

    def select(self, domain: str, answers: dict[str, str | None]) -> str | None:
        """The system's answer class, or ``None`` if every player abstained."""
        champion = self.champions.get(domain)
        if champion is not None and answers.get(champion) is not None:
            return answers[champion]

        if self.fallback_order is not None:
            for player in self.fallback_order.get(domain, []):
                if answers.get(player) is not None:
                    return answers[player]
            return None

        votes = Counter(a for a in answers.values() if a is not None)
        return votes.most_common(1)[0][0] if votes else None

    def players_consulted(
        self, domain: str, answers: dict[str, str | None]
    ) -> int:
        """How many players a request actually needed.

        Reported alongside accuracy because the system's case rests on costing
        barely more than one model, and an implementation that quietly consulted
        every player on every request would have the accuracy without the
        argument.
        """
        champion = self.champions.get(domain)
        if champion is not None and answers.get(champion) is not None:
            return 1
        return len(answers)


def expected_generations(
    rows: list[tuple[str, dict[str, str | None]]], system: CouncilSystem
) -> float:
    """Mean players consulted per request over a set of (domain, answers) rows."""
    if not rows:
        return 0.0
    return sum(system.players_consulted(d, a) for d, a in rows) / len(rows)


def make_generator_router(
    generate: Callable[[str, str], str | None],
) -> Callable[[str, str, list[str]], tuple[str | None, int]]:
    """Wrap a generation function so players are called lazily, in order.

    The measured cost figure assumes the champion is generated first and the
    others only on failure. A serving implementation that generated everything
    up front would report the same accuracy at four times the cost, so the lazy
    order is part of the result rather than an optimisation of it.
    """

    def run(prompt: str, champion: str, others: list[str]) -> tuple[str | None, int]:
        answer = generate(prompt, champion)
        if answer is not None:
            return answer, 1
        collected = []
        for player in others:
            got = generate(prompt, player)
            if got is not None:
                collected.append(got)
        if not collected:
            return None, 1 + len(others)
        votes = Counter(collected)
        return votes.most_common(1)[0][0], 1 + len(others)

    return run
