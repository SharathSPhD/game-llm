"""The magnetically anchored answer vote (F39, SPEC 0017).

The one mechanism in this programme that has beaten the domain router held-out,
promoted from the experiment scripts so the served system and the paper's
numbers rest on the same tested object.

The construction inverts the relationship every earlier rule had with the
baseline. Instead of competing with the router, the router's choice is the
reference policy of a quantal response over answer equivalence classes, pulled
by a magnetic term of strength ``tau`` — the policy-space proximal anchor of
magnetic mirror descent, with the incumbent as the magnet. Over discrete classes
the QRE argmax reduces to a thresholded vote: the council's answer moves away
from the router's only when the net vote margin against it exceeds ``tau``. At
large ``tau`` the mechanism is exactly the router, so its floor is the baseline
by construction; at ``tau = 0`` it is answer-consistency voting with ties
resolved toward the anchor, which is the limit rather than an extra rule.

Voting over answer classes rather than tokens or text scores is load-bearing:
players that do not share a tokenizer still share answers, and a verbose
derivation casts exactly one vote, so the two measurement hazards documented in
F37 and the first cross-examination run cannot recur here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnchoredVote:
    """Select an answer class under a magnetic anchor toward a reference player.

    Args:
        anchor_player: The player whose answer receives the magnetic bonus —
            in deployment, whichever player a domain router fixed in advance
            would have chosen for this prompt.
        tau: The net weighted vote margin the rest of the council must exceed
            to move the answer off the anchor. The offline grid was flat from
            0.25 to 2.5 (F39); 1.0 is the pre-registered deployment value.
        weights: Optional per-player vote weights, fixed in advance (the ladder
            accuracies of F28/F33). Absent players are an error rather than a
            silent default, because a mis-keyed player name would otherwise
            vote with an arbitrary weight and the failure would be invisible.
    """

    anchor_player: str
    tau: float = 1.0
    weights: dict[str, float] | None = field(default=None)

    def select(self, answers: dict[str, str | None]) -> str | None:
        """The council's answer, given each player's extracted answer class.

        A player whose answer could not be extracted abstains. If every player
        abstains there is nothing to select and the caller must handle ``None``
        rather than receive an invented answer.
        """
        scores: dict[str, float] = {}
        for player, cls in answers.items():
            if cls is None:
                continue
            w = 1.0 if self.weights is None else self.weights[player]
            scores[cls] = scores.get(cls, 0.0) + w

        anchor_cls = answers.get(self.anchor_player)
        if anchor_cls is not None:
            scores[anchor_cls] = scores.get(anchor_cls, 0.0) + self.tau

        if not scores:
            return None
        best = max(scores.values())
        tied = [c for c, s in scores.items() if s >= best - 1e-9]
        if anchor_cls in tied:
            return anchor_cls
        return tied[0]
