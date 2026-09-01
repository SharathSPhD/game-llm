#!/usr/bin/env python3
"""efe_rank.py — rank candidate experiments by Expected Free Energy.

Usage:
    efe_rank.py <candidates.json> [--cost-weight W] [--json]
    efe_rank.py --example > candidates.json

Reads a belief state and a set of candidate experiments, prints the ranking with
the epistemic term, the pragmatic term and the cost shown separately, and exits
non-zero if every candidate scores alike — because an agent that cannot separate
its actions is not selecting between them, and a cycle runner should stop rather
than proceed on a ranking that means nothing.

Standard library only, so it runs anywhere Python does.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

EPS = 1e-12


def _clip(p: float) -> float:
    return min(max(p, EPS), 1.0 - EPS)


def bernoulli_kl(posterior: float, prior: float) -> float:
    """KL divergence between two Bernoulli distributions, in nats."""
    p, q = _clip(posterior), _clip(prior)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def score(
    candidate: dict[str, Any], beliefs: dict[str, float], cost_weight: float
) -> dict[str, Any]:
    """Expected Free Energy for one candidate. Lower is better."""
    epistemic = 0.0
    pragmatic = 0.0
    contributions: dict[str, dict[str, float]] = {}

    for hyp, spec in candidate.get("diagnosticity", {}).items():
        if hyp not in beliefs:
            raise SystemExit(
                f"candidate {candidate['name']!r} refers to unknown hypothesis "
                f"{hyp!r}; declare it in \"beliefs\" or remove it"
            )
        p_true, p_false = float(spec[0]), float(spec[1])
        prior = beliefs[hyp]

        p_pos = _clip(prior * p_true + (1 - prior) * p_false)
        post_pos = prior * p_true / p_pos
        post_neg = prior * (1 - p_true) / (1 - p_pos)

        gain = p_pos * bernoulli_kl(post_pos, prior) + (1 - p_pos) * bernoulli_kl(
            post_neg, prior
        )
        value = float(candidate.get("payoff", {}).get(hyp, 0.0)) * p_pos

        epistemic += gain
        pragmatic += value
        contributions[hyp] = {"epistemic": gain, "pragmatic": value}

    cost = cost_weight * float(candidate.get("cost", 0.0))
    return {
        "name": candidate["name"],
        "epistemic": epistemic,
        "pragmatic": pragmatic,
        "cost": cost,
        "total": -(epistemic + pragmatic) + cost,
        "per_hypothesis": contributions,
        "description": candidate.get("description", ""),
    }


def entropy(beliefs: dict[str, float]) -> float:
    """Summed binary entropy of the belief state, in nats."""
    total = 0.0
    for p in beliefs.values():
        q = _clip(p)
        total += -(q * math.log(q) + (1 - q) * math.log(1 - q))
    return total


EXAMPLE = {
    "beliefs": {
        "verification_beats_aggregation": 0.2,
        "better_players_raise_the_ceiling": 0.4,
        "solve_is_affordable_at_serving": 0.65,
    },
    "cost_weight": 0.0417,
    "cost_units": "GPU-hours; weight set so one day of GPU time trades against one nat",
    "candidates": [
        {
            "name": "offline_reanalysis",
            "cost": 0.0,
            "diagnosticity": {"better_players_raise_the_ceiling": [0.75, 0.25]},
            "payoff": {"better_players_raise_the_ceiling": 0.5},
            "description": "Sweep already-stored scores; no new compute",
        },
        {
            "name": "training_run",
            "cost": 8.0,
            "diagnosticity": {"better_players_raise_the_ceiling": [0.8, 0.2]},
            "payoff": {"better_players_raise_the_ceiling": 1.0},
            "description": "Train a specialist and re-measure the ceiling",
        },
        {
            "name": "latency_measurement",
            "cost": 0.5,
            "diagnosticity": {"solve_is_affordable_at_serving": [0.9, 0.1]},
            "payoff": {"solve_is_affordable_at_serving": 0.8},
            "description": "Wall-clock the system against a single-model baseline",
        },
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidates", nargs="?", help="JSON file describing the cycle")
    ap.add_argument("--cost-weight", type=float, default=None,
                    help="override the weight in the file")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--example", action="store_true",
                    help="print a template input file and exit")
    args = ap.parse_args()

    if args.example:
        print(json.dumps(EXAMPLE, indent=2))
        return 0
    if not args.candidates:
        ap.error("give a candidates file, or --example to print a template")

    with open(args.candidates) as fh:
        spec = json.load(fh)

    beliefs = {k: float(v) for k, v in spec["beliefs"].items()}
    for hyp, p in beliefs.items():
        if not 0.0 < p < 1.0:
            raise SystemExit(
                f"belief for {hyp!r} is {p}; certainty admits no update, so it "
                "cannot be a hypothesis under test"
            )

    cost_weight = args.cost_weight
    if cost_weight is None:
        cost_weight = float(spec.get("cost_weight", 0.0))

    ranked = sorted(
        (score(c, beliefs, cost_weight) for c in spec["candidates"]),
        key=lambda s: s["total"],
    )

    identical = len({round(s["total"], 9) for s in ranked}) == 1 and len(ranked) > 1

    if args.json:
        print(json.dumps({
            "entropy_nats": entropy(beliefs),
            "cost_weight": cost_weight,
            "degenerate": identical,
            "ranking": ranked,
        }, indent=2))
    else:
        print(f"belief entropy {entropy(beliefs):.4f} nats "
              f"over {len(beliefs)} hypotheses")
        for hyp, p in beliefs.items():
            print(f"  P({hyp}) = {p:.3f}")
        print(f"cost weight {cost_weight:g} per unit "
              f"({spec.get('cost_units', 'unspecified units')})")
        print("\nranked, lowest expected free energy first:")
        width = max(len(s["name"]) for s in ranked)
        for s in ranked:
            print(f"  {s['name']:<{width}}  G={s['total']:+.4f}   "
                  f"epistemic {s['epistemic']:.4f}   "
                  f"pragmatic {s['pragmatic']:.4f}   "
                  f"cost {s['cost']:.4f}")
        if ranked:
            print(f"\nrun next: {ranked[0]['name']}")
            if ranked[0]["description"]:
                print(f"          {ranked[0]['description']}")

    if identical:
        print(
            "\nREFUSED: every candidate scored identically, so this ranking "
            "selects nothing. Almost always the likelihoods do not depend on the "
            "action — check that diagnosticity differs between candidates and "
            "that each pair (p_true, p_false) is not equal.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
