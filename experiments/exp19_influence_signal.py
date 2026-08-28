"""Where the extractable signal in a council actually is (SPEC 0016, follow-up).

exp18 established that solving the influence game buys nothing over uniform
averaging, and that raising the influence rationality makes matters steadily
worse. It did not establish why, and the difference matters: if the equilibrium
machinery is at fault the paradigm is in trouble, whereas if the *payoff* is at
fault then the machinery is fine and the game was mis-specified.

The suspicion this file tests is that self-agreement is the wrong payoff.
Weighting a player by how well the current consensus scores under its own logits
rewards confidence, and confidence is not competence — a specialist that is
wrong four times in five is still emphatic, and the game hands it influence
precisely when it is emphatic. Nothing in the construction lets competence in.

Four interventions are measured against uniform averaging on the same stored
scores, with per-player competence estimated on one half of each task and every
number reported on the other half, so no weight is fitted on what it is scored
against.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp18_equilibrium_mc import PLAYERS, SHORT, Example, collect  # noqa: E402


def _stack(rows: list[Example], normalised: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Group examples by option count so each group solves as one batch."""
    by_k: dict[int, list[Example]] = {}
    for row in rows:
        by_k.setdefault(row.scores(normalised).shape[-1], []).append(row)
    out = []
    for _, group in sorted(by_k.items()):
        ell = torch.stack([torch.log_softmax(r.scores(normalised), dim=-1) for r in group])
        gold = torch.tensor([r.gold for r in group])
        out.append((ell, gold))
    return out


def _acc(pred: torch.Tensor, gold: torch.Tensor) -> tuple[int, int]:
    return int((pred == gold).sum().item()), int(gold.numel())


def uniform_average(ell: torch.Tensor) -> torch.Tensor:
    return ell.mean(dim=1).argmax(dim=-1)


def competence_average(ell: torch.Tensor, comp: torch.Tensor, gamma: float) -> torch.Tensor:
    """Fixed weights from measured reliability rather than per-token confidence.

    This is the simplest way to let competence into the aggregation, and it is
    the honest control for the equilibrium: if a constant weight vector fitted on
    held-out accuracy captures the available gain, then a per-token game that
    does not know accuracy cannot be credited for that gain.
    """
    w = F.softmax(gamma * comp, dim=0).view(1, -1, 1)
    return (w * ell).sum(dim=1).argmax(dim=-1)


def entropy_influence(ell: torch.Tensor, beta: float) -> torch.Tensor:
    """Influence from per-example sharpness instead of agreement with consensus.

    A different reading of "who should decide here": the player whose
    distribution is most peaked on this question, regardless of whether it agrees
    with anyone. It is still a pure confidence signal, so it is expected to
    inherit the same defect, and measuring it separates "the payoff was the wrong
    *kind* of confidence" from "confidence of any kind is the wrong signal".
    """
    p = ell.exp()
    ent = -(p * ell).sum(dim=-1)
    w = F.softmax(-beta * ent, dim=1).unsqueeze(-1)
    return (w * ell).sum(dim=1).argmax(dim=-1)


def competence_game(
    ell: torch.Tensor, comp: torch.Tensor, beta: float, gamma: float, eta: float, iters: int
) -> torch.Tensor:
    """The influence game with competence entering the payoff.

    Identical to the equilibrium of ADR 0008 except that a player's payoff is its
    agreement with the consensus *plus* a standing term for how reliable it has
    proved, so influence can no longer be won by confidence alone. Setting gamma
    to zero recovers exactly the game exp18 measured.
    """
    log_y = torch.log_softmax(ell.mean(dim=1), dim=-1)
    prior = gamma * comp.view(1, -1)
    for _ in range(iters):
        y = log_y.exp()
        payoff = torch.einsum("bv,bnv->bn", y, ell)
        w = F.softmax(beta * payoff + prior, dim=-1)
        target = torch.einsum("bn,bnv->bv", w, ell)
        log_y = (1.0 - eta) * log_y + eta * target
        log_y = log_y - torch.logsumexp(log_y, dim=-1, keepdim=True)
    return log_y.argmax(dim=-1)


def dissent_game(
    ell: torch.Tensor, beta: float, eta: float, iters: int
) -> torch.Tensor:
    """The influence game with the payoff reversed.

    If rewarding agreement with the consensus is what damages the aggregate, then
    penalising it should either help or fail in an informative way. A player that
    disagrees with the emerging consensus gains influence here, which is the
    minority-report reading of the same game.
    """
    log_y = torch.log_softmax(ell.mean(dim=1), dim=-1)
    for _ in range(iters):
        y = log_y.exp()
        payoff = torch.einsum("bv,bnv->bn", y, ell)
        w = F.softmax(-beta * payoff, dim=-1)
        target = torch.einsum("bn,bnv->bv", w, ell)
        log_y = (1.0 - eta) * log_y + eta * target
        log_y = log_y - torch.logsumexp(log_y, dim=-1, keepdim=True)
    return log_y.argmax(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/agree")
    ap.add_argument("--out", default="results/scale/exp19_influence_signal.json")
    args = ap.parse_args()

    per_task, _ = collect(Path(args.root))

    # Split every task in half by document order: competence is estimated on the
    # fit half and every reported number comes from the eval half.
    fit: list[Example] = []
    ev: list[Example] = []
    for rows in per_task.values():
        half = len(rows) // 2
        fit.extend(rows[:half])
        ev.extend(rows[half:])

    n_players = len(PLAYERS)
    hits = torch.zeros(n_players)
    for row in fit:
        picks = row.scores(True).argmax(dim=-1)
        hits += (picks == row.gold).float()
    comp = hits / max(len(fit), 1)

    groups = _stack(ev, normalised=True)
    n_eval = sum(int(g.numel()) for _, g in groups)

    def score(fn: Any) -> float:
        c = 0
        for ell, gold in groups:
            c += _acc(fn(ell), gold)[0]
        return c / n_eval

    singles = torch.zeros(n_players)
    oracle = 0
    for ell, gold in groups:
        picks = ell.argmax(dim=-1)
        singles += (picks == gold.unsqueeze(1)).float().sum(dim=0)
        oracle += int((picks == gold.unsqueeze(1)).any(dim=1).sum().item())

    report: dict[str, Any] = {
        "n_fit": len(fit),
        "n_eval": n_eval,
        "competence_on_fit_half": {SHORT[p]: round(float(c), 4) for p, c in zip(PLAYERS, comp, strict=True)},
        "eval_singles": {SHORT[p]: round(float(s) / n_eval, 4) for p, s in zip(PLAYERS, singles, strict=True)},
        "oracle_any_correct": round(oracle / n_eval, 4),
        "uniform_average": round(score(uniform_average), 4),
        "arms": {},
    }
    base = report["uniform_average"]
    report["stderr"] = round(math.sqrt(base * (1 - base) / n_eval), 4)

    for gamma in (2.0, 5.0, 10.0, 20.0, 40.0):
        report["arms"][f"competence_average_gamma{gamma:g}"] = round(
            score(lambda e, g=gamma: competence_average(e, comp, g)), 4
        )
    for beta in (0.5, 2.0, 8.0):
        report["arms"][f"entropy_influence_beta{beta:g}"] = round(
            score(lambda e, b=beta: entropy_influence(e, b)), 4
        )
        report["arms"][f"dissent_game_beta{beta:g}"] = round(
            score(lambda e, b=beta: dissent_game(e, b, 0.5, 32)), 4
        )
    for beta in (0.25, 2.0):
        for gamma in (5.0, 20.0, 40.0):
            report["arms"][f"competence_game_beta{beta:g}_gamma{gamma:g}"] = round(
                score(lambda e, b=beta, g=gamma: competence_game(e, comp, b, g, 0.5, 32)), 4
            )

    best_name = max(report["arms"], key=lambda k: report["arms"][k])
    report["best_arm"] = {"name": best_name, "acc": report["arms"][best_name]}
    report["margin_over_averaging"] = round(report["arms"][best_name] - base, 4)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
