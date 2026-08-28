"""Aggregation rules that discount a player's vote for its own proposal.

F29 located the defect precisely: influence follows agreement with the consensus,
which rewards confidence, and a player that is confidently wrong is rewarded
exactly when it is most wrong. Every rule tested there let each player vote for
its own answer with its own enthusiasm.

Mechanism design has a standard answer to that, and it is the one this project
already validated in another setting (F6): value a proposal by what it is worth
to *everyone else*. A player cannot then buy influence with self-belief, because
its own belief is excluded from the valuation of the thing it proposed. The
rules below are variations on that idea, together with the robust-statistics
neighbours that make the same move bluntly — trimming the most enthusiastic voter,
or taking the median instead of the mean.

None of this costs a forward pass, so it is worth settling here before the
sequential arena, where the same question costs GPU hours to ask.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp18_equilibrium_mc import PLAYERS, SHORT, Example, collect  # noqa: E402


def _grouped(rows: list[Example]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    by_k: dict[int, list[Example]] = {}
    for row in rows:
        by_k.setdefault(row.scores(True).shape[-1], []).append(row)
    return [
        (
            torch.stack([torch.log_softmax(r.scores(True), dim=-1) for r in group]),
            torch.tensor([r.gold for r in group]),
        )
        for _, group in sorted(by_k.items())
    ]


def mean_rule(ell: torch.Tensor) -> torch.Tensor:
    return ell.mean(dim=1).argmax(dim=-1)


def median_rule(ell: torch.Tensor) -> torch.Tensor:
    """A single outlier player cannot move the median, however emphatic it is."""
    return ell.median(dim=1).values.argmax(dim=-1)


def trimmed_rule(ell: torch.Tensor) -> torch.Tensor:
    """Drop each option's most enthusiastic supporter before averaging.

    The crudest form of "your own vote does not count for you": whoever is most
    excited about an option is excluded from scoring it, whether or not that
    player would have proposed it.
    """
    top = ell.max(dim=1, keepdim=True).values
    return ((ell.sum(dim=1, keepdim=True) - top) / (ell.shape[1] - 1)).squeeze(1).argmax(dim=-1)


def leave_one_out_proposal(ell: torch.Tensor) -> torch.Tensor:
    """Each player proposes its own best option; the others price it.

    The proposer's own score is excluded from the valuation of its proposal, so
    a player wins only when the rest of the council independently finds its
    answer plausible. This is the second-price intuition of F6 applied to
    answers rather than tokens: what you propose is worth what it is worth to
    everybody else.
    """
    proposals = ell.argmax(dim=-1)                                    # [B, N]
    totals = ell.sum(dim=1)                                           # [B, K]
    own = ell.gather(2, proposals.unsqueeze(-1)).squeeze(-1)          # [B, N]
    prop_total = totals.gather(1, proposals)                          # [B, N]
    value = prop_total - own                                          # others only
    best = value.argmax(dim=1)                                        # [B]
    return proposals.gather(1, best.unsqueeze(1)).squeeze(1)


def leave_one_out_all(ell: torch.Tensor) -> torch.Tensor:
    """The same exclusion applied to every option, not only proposed ones.

    For each option the score is the total minus whichever player likes it most,
    which asks: setting aside its keenest advocate, does the council still
    prefer this answer?
    """
    totals = ell.sum(dim=1)
    keenest = ell.max(dim=1).values
    return (totals - keenest).argmax(dim=-1)


def borda_rule(ell: torch.Tensor) -> torch.Tensor:
    """Rank-based voting, which discards magnitude entirely.

    If the damage in F29 came from *how strongly* players scored rather than
    *what they preferred*, then a rule that keeps only the ordering should
    recover it. If Borda also fails to beat the mean, magnitude was not the
    problem.
    """
    ranks = ell.argsort(dim=-1).argsort(dim=-1).float()
    return ranks.sum(dim=1).argmax(dim=-1)


RULES = {
    "mean": mean_rule,
    "median": median_rule,
    "trimmed_drop_keenest": trimmed_rule,
    "leave_one_out_proposal": leave_one_out_proposal,
    "leave_one_out_all_options": leave_one_out_all,
    "borda": borda_rule,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/agree")
    ap.add_argument("--out", default="results/scale/exp21_verification_market.json")
    args = ap.parse_args()

    per_task, _ = collect(Path(args.root))
    rows: list[Example] = []
    for r in per_task.values():
        rows.extend(r)
    groups = _grouped(rows)
    n = sum(int(g.numel()) for _, g in groups)

    singles = torch.zeros(len(PLAYERS))
    oracle = 0
    for ell, gold in groups:
        picks = ell.argmax(dim=-1)
        singles += (picks == gold.unsqueeze(1)).float().sum(dim=0)
        oracle += int((picks == gold.unsqueeze(1)).any(dim=1).sum().item())

    report: dict[str, Any] = {
        "n": n,
        "singles": {
            SHORT[p]: round(float(s) / n, 4)
            for p, s in zip(PLAYERS, singles, strict=True)
        },
        "best_single": round(float(singles.max()) / n, 4),
        "oracle_any_correct": round(oracle / n, 4),
        "rules": {},
    }
    for name, fn in RULES.items():
        c = sum(int((fn(ell) == gold).sum().item()) for ell, gold in groups)
        report["rules"][name] = round(c / n, 4)

    base = report["rules"]["mean"]
    report["stderr"] = round(math.sqrt(base * (1 - base) / n), 4)
    # Paired comparison against the mean: a difference is only interesting
    # relative to how often the two rules disagree at all, since agreeing
    # predictions contribute nothing to the variance of the difference.
    report["vs_mean"] = {}
    for name, fn in RULES.items():
        if name == "mean":
            continue
        disagree = 0
        wins = 0
        losses = 0
        for ell, gold in groups:
            a, b = mean_rule(ell), fn(ell)
            d = a != b
            disagree += int(d.sum().item())
            wins += int(((b == gold) & d).sum().item())
            losses += int(((a == gold) & d).sum().item())
        se = math.sqrt(max(wins + losses, 1))
        report["vs_mean"][name] = {
            "delta": round((wins - losses) / n, 4),
            "disagreements": disagree,
            "wins": wins,
            "losses": losses,
            "z": round((wins - losses) / se, 2),
        }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
