"""The magnetically anchored answer vote (TRIZ cycle 28, SPEC 0016).

Every mechanism tested so far competed with the domain router and lost or tied
(F34, F36, F38). The TRIZ session inverted the dependency (principle 13): stop
building a council that competes with the router and make the router the
*reference policy* of the game, so the council can move the answer only when its
collective evidence overcomes a magnetic pull toward the incumbent. This is the
kinetic core aimed at the baseline instead of against it — the MMD magnet in
policy space (F21's correction) with the router as the magnet, a QRE over the
discrete space of answer equivalence classes, and ladder-fixed weights as
truthful bids.

Answer classes are the intermediary (principle 24) that neutralises the two
measurement artefacts this programme has already paid for: players that do not
share a tokenizer still share answers, and a verbose derivation casts exactly
one vote, so the length confound that corrupted the first cross-examination run
cannot recur here.

Over discrete classes the QRE argmax reduces to a thresholded vote,

    score(c) = sum_i w_i [a_i in c] + tau [c = router's class],

so tau is the number of net weighted votes the council needs against the anchor
before the answer moves. At tau large the mechanism IS the router — its floor is
the bar by construction, which no previously tested rule had — and at tau zero
it is weighted answer-consistency voting.

The test costs nothing: it runs over the candidates exp23 already generated,
with (weighting, tau) fitted on one seed and evaluated on the other two, all
three folds reported. Nothing here touches a GPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp16_auction_real import extract_answer, is_correct  # noqa: E402

#: Per-domain reference player — the router of F34, fixed from the ladder.
ROUTER = {
    "math": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "general": "Qwen/Qwen2.5-1.5B-Instruct",
}

#: Ladder accuracies per domain (F28 corrected by F33), fixed in advance. Used
#: as vote weights in the "ladder" weighting: a player's vote counts for what
#: the ladder measured it to be worth on that domain. These are measurements
#: taken before any exp23 run existed, so they carry no hindsight.
LADDER_WEIGHTS = {
    "math": {
        "Qwen/Qwen2.5-Math-1.5B-Instruct": 0.795,
        "Qwen/Qwen2.5-1.5B-Instruct": 0.595,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": 0.510,
        "Qwen/Qwen3-1.7B": 0.450,
    },
    "general": {
        "Qwen/Qwen2.5-1.5B-Instruct": 0.626,
        "Qwen/Qwen3-1.7B": 0.583,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct": 0.520,
        "Qwen/Qwen2.5-Math-1.5B-Instruct": 0.391,
    },
}

TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
WEIGHTINGS = ["uniform", "ladder"]


def _normalise(ans: str | None, kind: str) -> str | None:
    """Collapse an extracted answer to its equivalence class key."""
    if ans is None:
        return None
    if kind == "number":
        try:
            return f"{float(ans):.6g}"
        except ValueError:
            return None
    return ans.strip().upper() or None


def load_seed(candidates_path: Path, records_path: Path) -> list[dict[str, Any]]:
    """One question per row: gold, domain, and each player's answer class."""
    cands = json.loads(candidates_path.read_text())
    records = json.loads(records_path.read_text())
    rows = []
    for i, rec in enumerate(records):
        kind = "number" if rec["domain"] == "math" else "letter"
        answers = {
            player: _normalise(extract_answer(text, kind), kind)
            for player, text in cands[str(i)].items()
        }
        rows.append(
            {
                "domain": rec["domain"],
                "kind": kind,
                "gold": rec["gold"],
                "answers": answers,
                "router_class": answers.get(ROUTER[rec["domain"]]),
            }
        )
    return rows


def anchored_vote(row: dict[str, Any], weighting: str, tau: float) -> str | None:
    """Select an answer class under the magnetic anchor.

    A player whose answer could not be extracted abstains rather than voting for
    an empty class. Ties resolve toward the anchor when it is among the tied
    classes, which is the tau -> 0+ limit rather than an extra rule.
    """
    scores: dict[str, float] = {}
    for player, cls in row["answers"].items():
        if cls is None:
            continue
        w = 1.0 if weighting == "uniform" else LADDER_WEIGHTS[row["domain"]][player]
        scores[cls] = scores.get(cls, 0.0) + w
    anchor = row["router_class"]
    if anchor is not None:
        scores[anchor] = scores.get(anchor, 0.0) + tau
    if not scores:
        return None
    best = max(scores.values())
    tied = [c for c, s in scores.items() if s >= best - 1e-9]
    if anchor in tied:
        return anchor
    return tied[0]


def accuracy(rows: list[dict[str, Any]], weighting: str, tau: float) -> float:
    hits = sum(
        1
        for r in rows
        if (c := anchored_vote(r, weighting, tau)) is not None
        and is_correct(c, r["gold"], r["kind"])
    )
    return hits / len(rows)


def router_accuracy(rows: list[dict[str, Any]]) -> float:
    hits = sum(
        1
        for r in rows
        if r["router_class"] is not None
        and is_correct(r["router_class"], r["gold"], r["kind"])
    )
    return hits / len(rows)


def paired_vs_router(
    rows: list[dict[str, Any]], weighting: str, tau: float
) -> dict[str, float]:
    wins = losses = 0
    for r in rows:
        m = anchored_vote(r, weighting, tau)
        mech = m is not None and is_correct(m, r["gold"], r["kind"])
        rout = r["router_class"] is not None and is_correct(
            r["router_class"], r["gold"], r["kind"]
        )
        wins += int(mech and not rout)
        losses += int(rout and not mech)
    se = math.sqrt(max(wins + losses, 1))
    return {"wins": wins, "losses": losses, "z": (wins - losses) / se}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/exp23")
    ap.add_argument("--out", default="results/scale/exp27_anchored_vote.json")
    args = ap.parse_args()

    root = Path(args.root)
    seeds = {}
    for rp in sorted(root.glob("records_seed*.json")):
        seed = rp.stem.replace("records_", "")
        seeds[seed] = load_seed(root / f"candidates_{seed}.json", rp)

    report: dict[str, Any] = {"seeds": sorted(seeds), "router": ROUTER}

    # Sanity: at tau far above any achievable vote mass the mechanism must
    # reproduce the router exactly wherever the router answered at all.
    for rows in seeds.values():
        for r in rows:
            if r["router_class"] is not None:
                assert anchored_vote(r, "uniform", 100.0) == r["router_class"]

    # The full grid on all data, labelled as in-sample: transparency about the
    # surface, never the claim.
    pooled = [r for rows in seeds.values() for r in rows]
    report["router_accuracy_pooled"] = router_accuracy(pooled)
    report["in_sample_grid"] = {
        f"{w}/tau={t:g}": round(accuracy(pooled, w, t), 4)
        for w, t in itertools.product(WEIGHTINGS, TAU_GRID)
    }

    # The claim: fit (weighting, tau) on one seed, evaluate on the other two.
    folds = []
    for fit_seed in sorted(seeds):
        fit_rows = seeds[fit_seed]
        test_rows = [r for s, rows in seeds.items() if s != fit_seed for r in rows]
        w_best, t_best = max(
            itertools.product(WEIGHTINGS, TAU_GRID),
            key=lambda wt: accuracy(fit_rows, *wt),
        )
        held = accuracy(test_rows, w_best, t_best)
        rout = router_accuracy(test_rows)
        folds.append(
            {
                "fit_seed": fit_seed,
                "chosen": {"weighting": w_best, "tau": t_best},
                "held_out_accuracy": round(held, 4),
                "held_out_router": round(rout, 4),
                "margin": round(held - rout, 4),
                "paired": paired_vs_router(test_rows, w_best, t_best),
                "per_domain": {
                    dom: {
                        "mechanism": round(
                            accuracy(
                                [r for r in test_rows if r["domain"] == dom],
                                w_best,
                                t_best,
                            ),
                            4,
                        ),
                        "router": round(
                            router_accuracy(
                                [r for r in test_rows if r["domain"] == dom]
                            ),
                            4,
                        ),
                    }
                    for dom in ("math", "general")
                },
            }
        )
    report["folds"] = folds
    report["mean_held_out_margin"] = round(
        sum(f["margin"] for f in folds) / len(folds), 4
    )
    total_w = sum(f["paired"]["wins"] for f in folds)
    total_l = sum(f["paired"]["losses"] for f in folds)
    report["pooled_paired"] = {
        "wins": total_w,
        "losses": total_l,
        "z": round((total_w - total_l) / math.sqrt(max(total_w + total_l, 1)), 2),
        "note": (
            "test folds overlap (each seed appears in two folds), so this z "
            "overstates independence; the per-fold numbers are the claim"
        ),
    }

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
