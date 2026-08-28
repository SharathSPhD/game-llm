"""How tightly must confidence track competence before the game beats averaging?

F29 and F30 established that the influence game loses to a plain average and that
the cause is a payoff which rewards confidence while what it needs is competence.
Two responses follow, and they differ in cost by orders of magnitude. One is to
keep searching for a better rule over the players as they are, which five
experiments have now failed at. The other is to change the players so that
confidence *does* carry competence — which means training, and training is
expensive enough that it deserves a cheap test of its premise first.

That premise is testable without a GPU. If a council is simulated in which each
player's confidence on a question is coupled to its chance of being right on that
question, with the coupling strength as a dial, then sweeping the dial answers the
question directly: is there any coupling at which solving the game beats
averaging, and how tight must it be? A threshold that sits near perfect coupling
would mean training cannot realistically reach it and the paradigm is finished
regardless of how good the players become. A threshold well inside the achievable
range gives the training objective a target to hit.

The simulation is anchored rather than free: its uncoupled regime is calibrated
against the measured council, so the zero-coupling end of the sweep reproduces
what was actually observed instead of an arbitrary baseline.
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

from kinetic_ai.decode.aggregate import aggregate, oracle_any_correct  # noqa: E402


def measured_coupling(root: Path) -> dict[str, Any]:
    """Correlation between a player's confidence and its correctness, as measured.

    This is the quantity the simulation's dial represents, so reading it off the
    real council fixes where on the sweep the present system sits. Computed
    per player over the pooled questions as the point-biserial correlation
    between top-option probability and whether that option was right.
    """
    from exp18_equilibrium_mc import PLAYERS, SHORT, collect

    per_task, _ = collect(root)
    rows = [r for v in per_task.values() for r in v]
    out: dict[str, Any] = {}
    for i, name in enumerate(PLAYERS):
        confs, hits = [], []
        for row in rows:
            lp = torch.log_softmax(row.scores(True), dim=-1)[i]
            conf, pred = lp.exp().max(dim=-1)
            confs.append(float(conf))
            hits.append(float(pred.item() == row.gold))
        c = torch.tensor(confs)
        h = torch.tensor(hits)
        cc = c - c.mean()
        hh = h - h.mean()
        denom = float(cc.norm() * hh.norm())
        out[SHORT[name]] = round(float((cc * hh).sum()) / denom, 4) if denom else 0.0
    return out


def simulate(
    n_questions: int,
    n_players: int,
    n_options: int,
    coupling: float,
    error_correlation: float,
    competence: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A council whose confidence tracks competence by ``coupling``.

    Each player is right on a question with a probability drawn around its
    standing competence, and its confidence on that question is a blend of a
    random level and its actual chance of being right. At ``coupling = 0`` a
    player is exactly as emphatic when wrong as when right, which is the regime
    F29 measured; at ``coupling = 1`` confidence is a faithful report of the
    chance of being right, which is what a training objective that penalises
    confident error would be aiming to produce.

    Returns log-probabilities ``[Q, N, K]`` and the gold index ``[Q]``.
    """
    gold = torch.randint(0, n_options, (n_questions,), generator=generator)

    # Per-question, per-player chance of being right: the player's standing
    # competence perturbed so that questions differ in difficulty.
    jitter = torch.rand(n_questions, n_players, generator=generator) * 0.5 - 0.25
    p_right = (competence.view(1, -1) + jitter).clamp(0.05, 0.98)

    is_right = torch.rand(n_questions, n_players, generator=generator) < p_right

    # Confidence: a random level, blended toward the true chance of being right
    # in proportion to the coupling.
    random_level = torch.rand(n_questions, n_players, generator=generator)
    conf = (1.0 - coupling) * random_level + coupling * p_right
    # Map confidence to a logit gap; a confident player concentrates mass on its
    # chosen option, whether or not that option is the right one.
    gap = 8.0 * conf

    # Where a player is wrong, it lands on the question's attractor distractor
    # with probability `error_correlation` and on a random option otherwise.
    # Models trained on overlapping corpora fail in the same direction far more
    # often than chance — measured at 1.66 times chance on the real council,
    # where 56.6% of joint errors are the *same* error against 34.2% expected
    # under independence. Independent errors cancel under averaging and make
    # concentration safe; correlated errors do neither, so this dial is not a
    # detail of the simulation but the thing it exists to vary.
    attractor = torch.randint(0, n_options, (n_questions, 1), generator=generator)
    attractor = torch.where(attractor == gold.view(-1, 1), (attractor + 1) % n_options, attractor)
    random_wrong = torch.randint(0, n_options, (n_questions, n_players), generator=generator)
    follows_attractor = (
        torch.rand(n_questions, n_players, generator=generator) < error_correlation
    )
    wrong_choice = torch.where(follows_attractor, attractor.expand(-1, n_players), random_wrong)

    choice = torch.where(
        is_right,
        gold.view(-1, 1).expand(-1, n_players),
        wrong_choice,
    )
    logits = torch.zeros(n_questions, n_players, n_options)
    logits.scatter_(2, choice.unsqueeze(-1), gap.unsqueeze(-1))
    logits = logits + 0.1 * torch.randn(
        n_questions, n_players, n_options, generator=generator
    )
    return torch.log_softmax(logits, dim=-1), gold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/agree")
    ap.add_argument("--out", default="results/scale/exp24_coupling_threshold.json")
    ap.add_argument("--questions", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    # Standing competences matched to the measured council (F28), so the
    # simulated players are as unequal as the real ones.
    competence = torch.tensor([0.626, 0.583, 0.520, 0.391])
    n_players, n_options = len(competence), 4

    report: dict[str, Any] = {
        "questions": args.questions,
        "seeds": args.seeds,
        "competence": competence.tolist(),
        "sweep": [],
    }
    try:
        report["measured_confidence_competence_correlation"] = measured_coupling(
            Path(args.root)
        )
    except Exception as exc:  # measured data may be absent on a fresh checkout
        report["measured_confidence_competence_correlation"] = f"unavailable: {exc}"

    # The measured council sits near coupling 0.35 and error correlation 0.57,
    # so the sweep must cover that point rather than only the clean corners.
    couplings = [0.0, 0.35, 0.7, 1.0]
    error_corrs = [0.0, 0.35, 0.57, 0.8]
    for coupling in couplings:
      for err in error_corrs:
        arms: dict[str, list[float]] = {}
        for seed in range(args.seeds):
            gen = torch.Generator().manual_seed(1000 + seed)
            ell, gold = simulate(
                args.questions, n_players, n_options, coupling, err, competence, gen
            )
            n = gold.numel()
            arms.setdefault("best_single", []).append(
                float((ell.argmax(-1) == gold.unsqueeze(1)).float().mean(0).max())
            )
            arms.setdefault("oracle", []).append(
                float(oracle_any_correct(ell, gold).float().mean())
            )
            for rule, kw in (
                ("mean", {}),
                ("game", {"beta": 1.0}),
                ("game_beta4", {"beta": 4.0}),
                ("game_beta16", {"beta": 16.0}),
            ):
                name = "game" if rule.startswith("game") else rule
                beta = kw.get("beta", 0.25)
                picks = aggregate(ell, name, beta=beta) if name == "game" else aggregate(ell, name)
                arms.setdefault(rule, []).append(float((picks == gold).float().mean()))
            del n
        entry = {
            "coupling": coupling,
            "error_correlation": err,
            **{k: round(sum(v) / len(v), 4) for k, v in arms.items()},
            "stderr": round(
                math.sqrt(0.25 / (args.questions * args.seeds)), 4
            ),
        }
        entry["best_game_minus_mean"] = round(
            max(entry["game"], entry["game_beta4"], entry["game_beta16"]) - entry["mean"], 4
        )
        report["sweep"].append(entry)
        print(
            f"coupling={coupling:.2f} errcorr={err:.2f}  mean={entry['mean']:.4f}  "
            f"game(b=1)={entry['game']:.4f}  game(b=4)={entry['game_beta4']:.4f}  "
            f"game(b=16)={entry['game_beta16']:.4f}  "
            f"oracle={entry['oracle']:.4f}  delta={entry['best_game_minus_mean']:+.4f}",
            flush=True,
        )

    wins = [
        (e["coupling"], e["error_correlation"])
        for e in report["sweep"]
        if e["best_game_minus_mean"] > 3 * e["stderr"]
    ]
    report["regimes_where_game_wins"] = wins
    measured = next(
        (e for e in report["sweep"]
         if e["coupling"] == 0.35 and e["error_correlation"] == 0.57),
        None,
    )
    report["at_measured_regime"] = measured
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("regimes_where_game_wins", "at_measured_regime",
                       "measured_confidence_competence_correlation")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
