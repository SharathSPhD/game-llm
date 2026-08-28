"""Context-dependent competence as the influence signal (SPEC 0016, follow-up).

exp19 found that a player's measured reliability is the only signal that moves
the aggregate, and that every confidence-derived signal — agreement with the
consensus, sharpness, dissent — is worth nothing. But a constant reliability
weight is just weighted ensembling, and a weighted ensemble is not a new way to
compute a token; it is an old one with better constants. The gap between the two
is where this file looks.

The move is to make competence a function of the question rather than a
constant. A player's chance of being right on *this* input is partly legible
from the shape of the scores every player produced, without any label: how peaked
the player is, how far it stands from the others, whether the field agrees. If
that function can be estimated at all, then influence can follow predicted
competence instead of confidence, and the quantity being solved for is no longer
available to a fixed-weight ensemble.

The ceiling worth aiming at is the per-example oracle — some player is right on
83% of these questions while the best single player is right on 63%, so the
information is present in the council and the whole difficulty is extraction.

Everything is fitted on one half of each task and reported on the other.
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


def features(ell: torch.Tensor) -> torch.Tensor:
    """Label-free descriptors of each player's position on one question.

    ``ell`` is ``[B, N, K]`` log-probabilities. The descriptors are deliberately
    cheap and available at decode time: how concentrated a player is, how far it
    is from the field, and how much the field agrees with itself. Nothing here
    depends on knowing the answer, which is what makes the gate usable at
    inference rather than only in analysis.
    """
    p = ell.exp()
    ent = -(p * ell).sum(dim=-1)                                  # [B, N]
    top2 = p.topk(min(2, p.shape[-1]), dim=-1).values
    margin = top2[..., 0] - (top2[..., 1] if top2.shape[-1] > 1 else 0.0)
    mean_log = ell.mean(dim=1, keepdim=True)
    kl_to_field = (p * (ell - mean_log)).sum(dim=-1)              # [B, N]
    field_ent = -(mean_log.exp() * mean_log).sum(dim=-1)          # [B, 1]
    agree = torch.einsum("bnk,bk->bn", p, mean_log.exp().squeeze(1))
    n_opts = torch.full_like(ent, float(ell.shape[-1]))
    return torch.stack(
        [
            ent,
            margin,
            kl_to_field,
            agree,
            field_ent.expand_as(ent),
            ent - field_ent.expand_as(ent),
            torch.log(n_opts),
        ],
        dim=-1,
    )  # [B, N, F]


class Gate(torch.nn.Module):
    """Predicts, per player, the chance that this player has the question right.

    One shared linear map over the descriptors plus a per-player bias: the bias
    carries the standing reliability that exp19 already showed matters, and the
    shared weights carry whatever of the per-question variation is legible. Kept
    deliberately small — with a few thousand fitting examples, anything larger
    would fit the fitting half rather than the signal.
    """

    def __init__(self, n_features: int, n_players: int) -> None:
        super().__init__()
        self.w = torch.nn.Linear(n_features, 1, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(n_players))
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.scale * self.w(feats).squeeze(-1) + self.bias


def _grouped(rows: list[Example]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    by_k: dict[int, list[Example]] = {}
    for row in rows:
        by_k.setdefault(row.scores(True).shape[-1], []).append(row)
    out = []
    for _, group in sorted(by_k.items()):
        ell = torch.stack([torch.log_softmax(r.scores(True), dim=-1) for r in group])
        gold = torch.tensor([r.gold for r in group])
        out.append((ell, gold))
    return out


def fit_gate(groups: list[tuple[torch.Tensor, torch.Tensor]], n_players: int, seed: int) -> Gate:
    torch.manual_seed(seed)
    gate = Gate(n_features=7, n_players=n_players)
    opt = torch.optim.Adam(gate.parameters(), lr=0.05)
    feats = [features(ell) for ell, _ in groups]
    # Standardise per feature so the descriptors, which live on very different
    # scales, do not make the fit depend on the optimiser's step size.
    allf = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats])
    mu, sd = allf.mean(0), allf.std(0).clamp_min(1e-6)
    feats = [(f - mu) / sd for f in feats]
    targets = [
        (ell.argmax(dim=-1) == gold.unsqueeze(1)).float() for ell, gold in groups
    ]
    for _ in range(400):
        opt.zero_grad()
        loss = torch.stack(
            [
                F.binary_cross_entropy_with_logits(gate(f), t)
                for f, t in zip(feats, targets, strict=True)
            ]
        ).mean()
        loss.backward()
        opt.step()
    gate.mu, gate.sd = mu, sd  # type: ignore[assignment]
    return gate


def gate_scores(gate: Gate, ell: torch.Tensor) -> torch.Tensor:
    f = (features(ell) - gate.mu) / gate.sd  # type: ignore[operator]
    return gate(f)


def gate_average(gate: Gate, ell: torch.Tensor, gamma: float) -> torch.Tensor:
    w = F.softmax(gamma * gate_scores(gate, ell), dim=-1).unsqueeze(-1)
    return (w * ell).sum(dim=1).argmax(dim=-1)


def gate_game(
    gate: Gate, ell: torch.Tensor, beta: float, gamma: float, eta: float, iters: int
) -> torch.Tensor:
    """The equilibrium with predicted competence in the payoff.

    The solve is the one ADR 0008 specifies; only the payoff has changed, so a
    player now gains influence for being *probably right here* as well as for
    agreeing with the consensus. If this beats ``gate_average`` the solve is
    contributing something a fixed-weight blend cannot, and if it does not, the
    honest reading is that the gate is doing the work.
    """
    prior = gamma * gate_scores(gate, ell)
    log_y = torch.log_softmax(ell.mean(dim=1), dim=-1)
    for _ in range(iters):
        y = log_y.exp()
        payoff = torch.einsum("bv,bnv->bn", y, ell)
        w = F.softmax(beta * payoff + prior, dim=-1)
        target = torch.einsum("bn,bnv->bv", w, ell)
        log_y = (1.0 - eta) * log_y + eta * target
        log_y = log_y - torch.logsumexp(log_y, dim=-1, keepdim=True)
    return log_y.argmax(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/agree")
    ap.add_argument("--out", default="results/scale/exp20_learned_competence.json")
    ap.add_argument("--seeds", type=int, default=3)
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

    def score(fn: Any) -> float:
        c = 0
        for ell, gold in eval_groups:
            c += int((fn(ell) == gold).sum().item())
        return c / n_eval

    singles = torch.zeros(n_players)
    oracle = 0
    for ell, gold in eval_groups:
        picks = ell.argmax(dim=-1)
        singles += (picks == gold.unsqueeze(1)).float().sum(dim=0)
        oracle += int((picks == gold.unsqueeze(1)).any(dim=1).sum().item())

    base = score(lambda e: e.mean(dim=1).argmax(dim=-1))
    report: dict[str, Any] = {
        "n_fit": len(fit_rows),
        "n_eval": n_eval,
        "eval_singles": {
            SHORT[p]: round(float(s) / n_eval, 4)
            for p, s in zip(PLAYERS, singles, strict=True)
        },
        "best_single": round(float(singles.max()) / n_eval, 4),
        "oracle_any_correct": round(oracle / n_eval, 4),
        "uniform_average": round(base, 4),
        "stderr": round(math.sqrt(base * (1 - base) / n_eval), 4),
        "seeds": {},
    }

    for seed in range(42, 42 + args.seeds):
        gate = fit_gate(fit_groups, n_players, seed)
        arms: dict[str, float] = {}
        for gamma in (1.0, 2.0, 4.0, 8.0):
            arms[f"gate_average_gamma{gamma:g}"] = round(
                score(lambda e, g=gamma, gt=gate: gate_average(gt, e, g)), 4
            )
            arms[f"gate_game_beta0.25_gamma{gamma:g}"] = round(
                score(lambda e, g=gamma, gt=gate: gate_game(gt, e, 0.25, g, 0.5, 32)), 4
            )
        report["seeds"][str(seed)] = arms

    # Report each arm as a mean over seeds, since the gate fit is the only
    # stochastic part and a single fit could flatter or spoil any one arm.
    names = list(report["seeds"][str(42)])
    means = {
        n: round(sum(report["seeds"][s][n] for s in report["seeds"]) / args.seeds, 4)
        for n in names
    }
    report["arm_means"] = means
    best = max(means, key=lambda k: means[k])
    report["best_arm"] = {"name": best, "acc": means[best]}
    report["margin_over_averaging"] = round(means[best] - base, 4)
    report["fraction_of_oracle_gap_closed"] = round(
        (means[best] - base) / max(report["oracle_any_correct"] - base, 1e-9), 4
    )

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
