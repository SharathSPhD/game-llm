"""TDD for the deployed council system (F41).

The properties pinned are the ones the confirmed claim depends on: the champion
is authoritative when it answers, redundancy engages only when it does not, and
the cost accounting reflects lazy consultation rather than assuming it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.decode.council_system import (
    CouncilSystem,
    expected_generations,
    make_generator_router,
)

CH = {"math": "mathematician", "general": "generalist"}


def _sys() -> CouncilSystem:
    return CouncilSystem(champions=CH)


class TestChampionIsAuthoritative:
    def test_an_answering_champion_is_never_overridden(self) -> None:
        """The tau -> infinity property: three dissenters do not move the answer.
        This is what the confirmation measured as correct, and a regression here
        would silently reintroduce the refuted anchoring regime."""
        s = _sys()
        answers = {"mathematician": "A", "generalist": "B", "c": "B", "d": "B"}
        assert s.select("math", answers) == "A"

    def test_the_champion_is_chosen_per_domain(self) -> None:
        s = _sys()
        answers = {"mathematician": "A", "generalist": "B"}
        assert s.select("math", answers) == "A"
        assert s.select("general", answers) == "B"


class TestRedundancyOnlyOnFailure:
    def test_majority_of_the_rest_when_the_champion_abstains(self) -> None:
        s = _sys()
        answers = {"mathematician": None, "generalist": "B", "c": "B", "d": "C"}
        assert s.select("math", answers) == "B"

    def test_none_when_everyone_abstains(self) -> None:
        s = _sys()
        assert s.select("math", {"mathematician": None, "generalist": None}) is None

    def test_an_explicit_fallback_order_is_respected(self) -> None:
        s = CouncilSystem(champions=CH, fallback_order={"math": ["d", "generalist"]})
        answers = {"mathematician": None, "generalist": "B", "c": "B", "d": "C"}
        assert s.select("math", answers) == "C"

    def test_an_unknown_domain_falls_back_rather_than_failing(self) -> None:
        s = _sys()
        answers = {"mathematician": "A", "generalist": "B", "c": "B"}
        assert s.select("physics", answers) == "B"


class TestCostAccounting:
    def test_one_player_is_consulted_when_the_champion_answers(self) -> None:
        s = _sys()
        assert s.players_consulted("math", {"mathematician": "A", "generalist": "B"}) == 1

    def test_all_players_are_consulted_when_it_does_not(self) -> None:
        s = _sys()
        assert s.players_consulted("math", {"mathematician": None, "generalist": "B"}) == 2

    def test_expected_generations_tracks_the_failure_rate(self) -> None:
        """Four requests, one champion failure over four players: 1.75 expected."""
        s = _sys()
        rows = [
            ("math", {"mathematician": "A", "generalist": "B", "c": "B", "d": "B"}),
            ("math", {"mathematician": "A", "generalist": "B", "c": "B", "d": "B"}),
            ("math", {"mathematician": "A", "generalist": "B", "c": "B", "d": "B"}),
            ("math", {"mathematician": None, "generalist": "B", "c": "B", "d": "B"}),
        ]
        assert expected_generations(rows, s) == 1.75

    def test_the_lazy_router_does_not_generate_others_when_unnecessary(self) -> None:
        """The cost claim assumes the champion runs first and the rest only on
        failure; a wrapper that generated everything up front would report the
        same accuracy at four times the cost."""
        calls: list[str] = []

        def generate(prompt: str, player: str) -> str | None:
            calls.append(player)
            return "A" if player == "champ" else "B"

        run = make_generator_router(generate)
        answer, used = run("q", "champ", ["x", "y", "z"])
        assert answer == "A"
        assert used == 1
        assert calls == ["champ"]

    def test_the_lazy_router_consults_the_rest_on_failure(self) -> None:
        def generate(prompt: str, player: str) -> str | None:
            return None if player == "champ" else "B"

        run = make_generator_router(generate)
        answer, used = run("q", "champ", ["x", "y"])
        assert answer == "B"
        assert used == 3
