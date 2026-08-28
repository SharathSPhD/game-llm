"""The bar a council has to clear: a domain router fixed in advance.

A council is only worth its machinery if it beats the cheapest thing that
exploits the same structure. On a mixed arena that cheapest thing is a domain
router: classify the prompt, send it to whichever player the baseline ladder
already showed to be best on that domain, and stop. It needs no aggregation, no
solve, no second forward pass, and on a mixed set of mathematics and
multiple-choice knowledge the classification is decidable from format alone.

Reporting a council against the best single player, as the earlier experiments
did, therefore flatters it. The best single player is the wrong bar because it
is not what a competent engineer would build; the router is. This script
computes the router's score from any exp23 record file, using the per-domain
champions the ladder fixed in advance rather than the champions of the run being
scored, so the router gets no hindsight the council does not also get.

Run it over each seed's records and compare against every rule that seed
measured.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

#: Per-domain champions taken from the baseline ladder (findings F28 and F33),
#: fixed before any of these runs and not re-derived from the data being scored.
LADDER_CHAMPIONS = {
    "math": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "general": "Qwen/Qwen2.5-1.5B-Instruct",
}

RULES = ("cross_exam", "leave_one_out", "self_preference", "equilibrium")


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every rule in a records file, plus the router and the single players."""
    n = len(records)
    if n == 0:
        raise ValueError("no records to score")

    players = list(records[0]["singles"])
    singles = {p: sum(1 for r in records if r["singles"][p]) / n for p in players}
    best_player = max(singles, key=lambda p: singles[p])

    missing = {r["domain"] for r in records} - set(LADDER_CHAMPIONS)
    if missing:
        raise ValueError(
            f"no ladder champion declared for domain(s) {sorted(missing)}; "
            "add one rather than letting the router silently skip them"
        )
    router = sum(
        1 for r in records if r["singles"][LADDER_CHAMPIONS[r["domain"]]]
    ) / n

    rules = {
        rule: sum(1 for r in records if r[rule]["correct"]) / n
        for rule in RULES
        if rule in records[0]
    }
    oracle = sum(1 for r in records if any(r["singles"].values())) / n

    per_domain: dict[str, Any] = {}
    for dom in sorted({r["domain"] for r in records}):
        rs = [r for r in records if r["domain"] == dom]
        per_domain[dom] = {
            "n": len(rs),
            "router_picks": LADDER_CHAMPIONS[dom],
            "router": sum(1 for r in rs if r["singles"][LADDER_CHAMPIONS[dom]]) / len(rs),
            **{
                rule: sum(1 for r in rs if r[rule]["correct"]) / len(rs)
                for rule in RULES
                if rule in rs[0]
            },
        }

    # A margin is only meaningful against the spread of the comparison. The
    # paired standard error uses only the questions where the two rules differ,
    # since agreeing answers contribute nothing to the variance of the gap.
    beats_router = {}
    for rule in rules:
        wins = sum(
            1 for r in records
            if r[rule]["correct"] and not r["singles"][LADDER_CHAMPIONS[r["domain"]]]
        )
        losses = sum(
            1 for r in records
            if not r[rule]["correct"] and r["singles"][LADDER_CHAMPIONS[r["domain"]]]
        )
        se = math.sqrt(max(wins + losses, 1))
        beats_router[rule] = {
            "margin": rules[rule] - router,
            "wins": wins,
            "losses": losses,
            "z": (wins - losses) / se,
        }

    return {
        "n": n,
        "singles": singles,
        "best_single": singles[best_player],
        "best_single_name": best_player,
        "domain_router": router,
        "router_over_best_single": router - singles[best_player],
        "rules": rules,
        "oracle_any_player_correct": oracle,
        "rules_vs_router": beats_router,
        "per_domain": per_domain,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+", help="exp23 records_seed*.json files")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report: dict[str, Any] = {"per_seed": {}}
    for path in args.records:
        recs = json.loads(Path(path).read_text())
        report["per_seed"][Path(path).stem] = score_records(recs)

    seeds = list(report["per_seed"].values())
    if seeds:
        keys = set(seeds[0]["rules"])
        report["mean"] = {
            "domain_router": sum(s["domain_router"] for s in seeds) / len(seeds),
            "best_single": sum(s["best_single"] for s in seeds) / len(seeds),
            "oracle": sum(s["oracle_any_player_correct"] for s in seeds) / len(seeds),
            "rules": {
                k: sum(s["rules"][k] for s in seeds) / len(seeds) for k in keys
            },
        }
        m = report["mean"]
        print(f"{'':24s}{'mean':>9}")
        print(f"{'best single player':24s}{m['best_single']:9.4f}")
        print(f"{'domain router (ladder)':24s}{m['domain_router']:9.4f}   <- the bar")
        for k in sorted(m["rules"], key=lambda k: -m["rules"][k]):
            flag = "  BEATS ROUTER" if m["rules"][k] > m["domain_router"] else ""
            print(f"{k:24s}{m['rules'][k]:9.4f}{flag}")
        print(f"{'oracle (any player)':24s}{m['oracle']:9.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
