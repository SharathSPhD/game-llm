"""Calibrate the players, then play the same game (SPEC 0016, ADR 0008).

Every rule tried so far changed how votes are combined, and every one landed
within noise of a plain average. This file changes the votes instead.

The argument is that "confidence is not competence" — F29's diagnosis — is a
statement about the *inputs*, not the rule. The influence game is well posed if a
player's confidence means what it claims: a player that says 0.9 should be right
nine times in ten. Then agreement with the consensus really does track being
right, the payoff becomes meaningful, and the mechanism that damaged the
aggregate becomes the mechanism that should help it. If instead the weak players
are simply loud, no combination rule can repair them from outside, which is
exactly the pattern of results so far.

Each player gets one scalar temperature, fitted on half of each task to minimise
negative log-likelihood of the correct answer, and the whole comparison is then
re-run on the other half with the calibrated scores in place of the raw ones.
One number per player is deliberately the smallest possible intervention: if it
works, the effect is attributable to calibration and not to a fitted aggregator.
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


def fit_temperatures(
    groups: list[tuple[torch.Tensor, torch.Tensor]], n_players: int
) -> torch.Tensor:
    """One temperature per player, fitted by maximum likelihood on the fit half.

    Temperature is the right knob because it moves confidence without moving
    preference: the ranking of options under a player is unchanged, so anything
    the calibrated council gains comes from *how strongly* players speak rather
    than from changing what they would have said alone.
    """
    log_t = torch.zeros(n_players, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.zeros(())
        count = 0
        for ell, gold in groups:
            scaled = ell / log_t.exp().view(1, -1, 1)
            logp = torch.log_softmax(scaled, dim=-1)
            tgt = gold.view(-1, 1, 1).expand(-1, ell.shape[1], 1)
            loss = loss + (-logp.gather(2, tgt).squeeze(-1)).sum()
            count += ell.shape[0] * ell.shape[1]
        (loss / count).backward()
        opt.step()
    return log_t.detach().exp()


def expected_calibration_error(
    groups: list[tuple[torch.Tensor, torch.Tensor]], player: int, bins: int = 10
) -> float:
    """How far a player's stated confidence is from its realised accuracy."""
    confs, hits = [], []
    for ell, gold in groups:
        p = ell[:, player].exp()
        conf, pred = p.max(dim=-1)
        confs.append(conf)
        hits.append((pred == gold).float())
    conf = torch.cat(confs)
    hit = torch.cat(hits)
    err, n = 0.0, conf.numel()
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += float(m.sum()) / n * abs(float(conf[m].mean()) - float(hit[m].mean()))
    return err


def influence_game(
    ell: torch.Tensor, beta: float, tau: float, eta: float, iters: int
) -> torch.Tensor:
    ref = torch.log_softmax(ell.mean(dim=1), dim=-1)
    log_y = ref.clone()
    for _ in range(iters):
        y = log_y.exp()
        payoff = torch.einsum("bv,bnv->bn", y, ell)
        w = (
            F.softmax(beta * payoff, dim=-1)
            if beta != 0
            else torch.full_like(payoff, 1.0 / payoff.shape[-1])
        )
        target = torch.einsum("bn,bnv->bv", w, ell) + tau * ref
        log_y = (1.0 - eta) * log_y + eta * target
        log_y = log_y - torch.logsumexp(log_y, dim=-1, keepdim=True)
    return log_y.argmax(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/agree")
    ap.add_argument("--out", default="results/scale/exp22_calibrated_game.json")
    args = ap.parse_args()

    per_task, _ = collect(Path(args.root))
    fit_rows: list[Example] = []
    eval_rows: list[Example] = []
    for rows in per_task.values():
        half = len(rows) // 2
        fit_rows.extend(rows[:half])
        eval_rows.extend(rows[half:])

    fit_groups = _grouped(fit_rows)
    eval_groups = _grouped(eval_rows)
    n_eval = sum(int(g.numel()) for _, g in eval_groups)
    n_players = len(PLAYERS)

    temps = fit_temperatures(fit_groups, n_players)
    cal_groups = [
        (torch.log_softmax(ell / temps.view(1, -1, 1), dim=-1), gold)
        for ell, gold in eval_groups
    ]

    def score(groups: list[tuple[torch.Tensor, torch.Tensor]], fn: Any) -> float:
        return sum(int((fn(e) == g).sum().item()) for e, g in groups) / n_eval

    report: dict[str, Any] = {
        "n_eval": n_eval,
        "temperatures": {
            SHORT[p]: round(float(t), 3) for p, t in zip(PLAYERS, temps, strict=True)
        },
        "ece_before": {
            SHORT[p]: round(expected_calibration_error(eval_groups, i), 4)
            for i, p in enumerate(PLAYERS)
        },
        "ece_after": {
            SHORT[p]: round(expected_calibration_error(cal_groups, i), 4)
            for i, p in enumerate(PLAYERS)
        },
        "raw": {},
        "calibrated": {},
    }

    mean_fn = lambda e: e.mean(dim=1).argmax(dim=-1)  # noqa: E731
    report["raw"]["average"] = round(score(eval_groups, mean_fn), 4)
    report["calibrated"]["average"] = round(score(cal_groups, mean_fn), 4)

    for beta in (0.25, 1.0, 2.0, 4.0, 8.0):
        key = f"game_beta{beta:g}"
        report["raw"][key] = round(
            score(eval_groups, lambda e, b=beta: influence_game(e, b, 0.0, 0.5, 32)), 4
        )
        report["calibrated"][key] = round(
            score(cal_groups, lambda e, b=beta: influence_game(e, b, 0.0, 0.5, 32)), 4
        )

    base = report["raw"]["average"]
    report["stderr"] = round(math.sqrt(base * (1 - base) / n_eval), 4)
    best_cal = max(report["calibrated"], key=lambda k: report["calibrated"][k])
    report["best_calibrated"] = {"name": best_cal, "acc": report["calibrated"][best_cal]}
    report["margin_over_raw_average"] = round(
        report["calibrated"][best_cal] - base, 4
    )

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
