"""Does the surviving system generalise beyond one model family? (SPEC 0017 companion)

F40 reduced the council's demonstrated value to two ingredients: routing on
measured per-domain priors, and redundancy against single-model extraction
failure. Both were established on one family of players. The remaining open
hypothesis in the autoresearch belief state is whether that structure is a
property of councils or an artefact of the Qwen 1.5B generation, and this
experiment answers it on a council chosen to be maximally unlike the first:
four different families, four different tokenizers, sizes from 1B to 7B.

The tokenizer heterogeneity is the point rather than an obstacle. Answer-class
mechanisms are the only ones in this programme that such a council can run at
all — token-level aggregation across four vocabularies is undefined (F37) — so a
positive result here is also evidence for the answer-class design itself.

Protocol, fixed before any of this data exists: seed 50 is calibration and
defines the per-domain champions by accuracy, standing in for the ladder; seeds
51 to 53 are evaluation. On the evaluation seeds four systems are compared —
best single member, the plain champion router, the champion router with
majority fallback on extraction failure (the fair bar of F40), and the anchored
vote at the pre-registered uniform weighting and tau = 1. Success for the
generality claim is the fallback router beating the best single member on the
evaluation seeds; the anchored vote is reported against the fallback router
under F40's decomposition, with no expectation of separation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp16_auction_real import build_tasks, extract_answer, is_correct  # noqa: E402
from exp23_cross_examination import generate_solution  # noqa: E402

#: Council B: four families, four tokenizers, 1B to 7B, all ungated.
PLAYERS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-1B-Instruct",
]

CALIBRATION_SEED = 50
EVAL_SEEDS = [51, 52, 53]


def _normalise(ans: str | None, kind: str) -> str | None:
    if ans is None:
        return None
    if kind == "number":
        try:
            return f"{float(ans):.6g}"
        except ValueError:
            return None
    return ans.strip().upper() or None


@torch.no_grad()
def generate_seed(seed: int, cfg: dict, device: str, out: Path) -> None:
    """All players' candidates for one seed, stored like exp23's."""
    cfg = dict(cfg)
    cfg["seed"] = seed
    torch.manual_seed(seed)
    tasks = build_tasks(cfg, shuffle_math=True)
    for task in tasks:
        if task["kind"] == "letter":
            task["prompt"] = task["prompt"].replace(
                "Reply with ONLY the single letter of the correct option in "
                "\\boxed{}. Do not explain.",
                "Reason briefly, then give the single letter of the correct "
                "option in \\boxed{}.",
            )
    budget = {
        "number": int(cfg["generation"]["max_new_tokens_math"]),
        "letter": int(cfg["generation"]["max_new_tokens_general"]),
    }
    candidates: dict[int, dict[str, str]] = {i: {} for i in range(len(tasks))}
    for name in PLAYERS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map=device
        ).eval()
        for i, task in enumerate(tasks):
            candidates[i][name] = generate_solution(
                model, tok, task["prompt"], budget[task["kind"]], device
            )
        del model
        torch.cuda.empty_cache()
        print(f"seed {seed} generated: {name}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"candidates_seed{seed}.json").write_text(
        json.dumps({str(k): v for k, v in candidates.items()})
    )
    (out / f"tasks_seed{seed}.json").write_text(
        json.dumps([{k: t[k] for k in ("domain", "gold", "kind")} for t in tasks])
    )


def load_rows(out: Path, seed: int) -> list[dict[str, Any]]:
    cands = json.loads((out / f"candidates_seed{seed}.json").read_text())
    tasks = json.loads((out / f"tasks_seed{seed}.json").read_text())
    rows = []
    for i, t in enumerate(tasks):
        answers = {
            p: _normalise(extract_answer(txt, t["kind"]), t["kind"])
            for p, txt in cands[str(i)].items()
        }
        rows.append({**t, "answers": answers})
    return rows


def _right(cls: str | None, r: dict[str, Any]) -> bool:
    return cls is not None and is_correct(cls, r["gold"], r["kind"])


def analyse(out: Path) -> dict[str, Any]:
    calib = load_rows(out, CALIBRATION_SEED)
    champions: dict[str, str] = {}
    for dom in ("math", "general"):
        sub = [r for r in calib if r["domain"] == dom]
        champions[dom] = max(
            PLAYERS, key=lambda p: sum(1 for r in sub if _right(r["answers"][p], r))
        )

    def router(r: dict[str, Any]) -> str | None:
        return r["answers"][champions[r["domain"]]]

    def fallback(r: dict[str, Any]) -> str | None:
        c = router(r)
        if c is not None:
            return c
        votes = Counter(v for v in r["answers"].values() if v is not None)
        return votes.most_common(1)[0][0] if votes else None

    def anchored(r: dict[str, Any], tau: float = 1.0) -> str | None:
        scores: dict[str, float] = {}
        for cls in r["answers"].values():
            if cls is not None:
                scores[cls] = scores.get(cls, 0.0) + 1.0
        anchor = router(r)
        if anchor is not None:
            scores[anchor] = scores.get(anchor, 0.0) + tau
        if not scores:
            return None
        best = max(scores.values())
        tied = [c for c, s in scores.items() if s >= best - 1e-9]
        return anchor if anchor in tied else tied[0]

    rows = [r for s in EVAL_SEEDS for r in load_rows(out, s)]
    n = len(rows)
    singles = {
        p: sum(1 for r in rows if _right(r["answers"][p], r)) / n for p in PLAYERS
    }
    best_p = max(singles, key=lambda p: singles[p])
    systems = {
        "best_single": [_right(r["answers"][best_p], r) for r in rows],
        "plain_router": [_right(router(r), r) for r in rows],
        "fallback_router": [_right(fallback(r), r) for r in rows],
        "anchored_tau1": [_right(anchored(r), r) for r in rows],
    }
    fb, bs = systems["fallback_router"], systems["best_single"]
    w = sum(1 for a, b in zip(fb, bs, strict=True) if a and not b)
    lo = sum(1 for a, b in zip(fb, bs, strict=True) if b and not a)
    return {
        "champions": champions,
        "calibration_n": len(calib),
        "eval_n": n,
        "singles": {p: round(v, 4) for p, v in singles.items()},
        "best_single_name": best_p,
        "systems": {k: round(sum(v) / n, 4) for k, v in systems.items()},
        "fallback_vs_best_single": {
            "wins": w,
            "losses": lo,
            "z": round((w - lo) / math.sqrt(max(w + lo, 1)), 2),
        },
        "oracle_any_player": round(
            sum(1 for r in rows if any(_right(c, r) for c in r["answers"].values()))
            / n,
            4,
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp23_cross_exam.yaml")
    ap.add_argument("--out", default="results/scale/exp28")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda:0")
    out = Path(args.out)

    if not args.analyse_only:
        for seed in [CALIBRATION_SEED, *EVAL_SEEDS]:
            if not (out / f"candidates_seed{seed}.json").exists():
                generate_seed(seed, cfg, device, out)
                print(f"=== seed {seed} complete ===", flush=True)

    report = analyse(out)
    (out / "results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
