"""Score an uptrained KineticLM against its base (SPEC 0011 gates).

Runs the same lm-evaluation-harness invocation that produced the recorded base
rates, so retention is a same-harness, same-machine, same-sample comparison
rather than a number lifted from a model card. Also reports the parameter
saving and the inference-budget sweep, which are the other two halves of the
H7 gate.

Usage:
    python exp15_eval.py --checkpoint results/exp15_kinetic/checkpoint \
        --output results/scale/exp15_eval [--limit 300] [--skip-harness]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.kinetic_lm import count_unique_params, load_kinetic  # noqa: E402

# Base rates measured on this project's own hardware with the identical harness
# invocation (0-shot, 300-sample limit, batch 8) — see
# results/scale/baselines/qwen3_1p7b_screen/.
BASE_RATES: dict[str, float] = {
    "arc_challenge/acc,none": 0.4067,
    "arc_challenge/acc_norm,none": 0.4433,
    "hellaswag/acc,none": 0.43,
    "hellaswag/acc_norm,none": 0.5067,
    "gsm8k/exact_match,flexible-extract": 0.4567,
}
HEADLINE = [
    "arc_challenge/acc_norm,none",
    "hellaswag/acc_norm,none",
    "gsm8k/exact_match,flexible-extract",
]
TASKS = "arc_challenge,hellaswag,gsm8k"


def flatten(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for task, metrics in results.get("results", {}).items():
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in key:
                flat[f"{task}/{key}"] = float(value)
    return flat


@torch.no_grad()
def heldout_ppl(model: Any, tokens: torch.Tensor, device: str, batches: int, bs: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(batches):
        batch = tokens[i * bs : (i + 1) * bs].to(device)
        if batch.numel() == 0:
            break
        total += model(batch, labels=batch).loss.item()
        count += 1
    return math.exp(total / max(count, 1))


def run_harness(checkpoint: Path, out_dir: Path, limit: int, batch_size: int) -> dict[str, float]:
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint},dtype=bfloat16,device_map=cuda:0",
        "--tasks", TASKS,
        "--num_fewshot", "0",
        "--limit", str(limit),
        "--batch_size", str(batch_size),
        "--output_path", str(out_dir / "harness"),
    ]
    print("[harness] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"lm_eval failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    files = sorted((out_dir / "harness").rglob("results_*.json"))
    if not files:
        raise RuntimeError("lm_eval produced no results file")
    return flatten(json.loads(files[-1].read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--token-cache", default=None,
                    help="token cache .pt from exp15 for the held-out perplexity sweep")
    ap.add_argument("--skip-harness", action="store_true")
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B",
                    help="base model scored on the SAME held-out tokens, so the "
                         "perplexity ratio is like-for-like")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_kinetic(checkpoint, dtype=torch.bfloat16).to(device).eval()
    n_params = count_unique_params(model)
    full_depth = model.recursion_depth
    print(f"[model] {n_params/1e9:.3f}B unique params | recursion depth {full_depth}", flush=True)

    budget_ppl: dict[str, float] = {}
    if args.token_cache and Path(args.token_cache).exists():
        blob = torch.load(args.token_cache, weights_only=True)
        heldout = blob["tokens"][:64]
        for d in sorted({max(1, full_depth // 4), max(1, full_depth // 2), full_depth}):
            model.set_recursion_depth(d)
            budget_ppl[str(d)] = heldout_ppl(model, heldout, device, 8, 4)
            print(f"  budget {d:3d}: held-out ppl {budget_ppl[str(d)]:.3f}", flush=True)
        model.set_recursion_depth(full_depth)

    base_ppl: float | None = None
    if args.token_cache and Path(args.token_cache).exists():
        from transformers import AutoModelForCausalLM

        blob = torch.load(args.token_cache, weights_only=True)
        heldout = blob["tokens"][:64]
        base = (
            AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        base_ppl = heldout_ppl(base, heldout, device, 8, 4)
        print(f"[base] {args.base_model} held-out ppl {base_ppl:.3f} (same tokens)", flush=True)
        del base
        torch.cuda.empty_cache()

    scores: dict[str, float] = {}
    retention: dict[str, float] = {}
    if not args.skip_harness:
        del model
        torch.cuda.empty_cache()
        scores = run_harness(checkpoint, out_dir, args.limit, args.batch_size)
        for key, base in BASE_RATES.items():
            if key in scores and base > 0:
                retention[key] = scores[key] / base

    headline = [retention[k] for k in HEADLINE if k in retention]
    mean_retention = sum(headline) / len(headline) if headline else None

    results = {
        "experiment": "exp15_eval",
        "spec": "0011",
        "checkpoint": str(checkpoint),
        "unique_params": n_params,
        "recursion_depth": full_depth,
        "harness": {
            "tasks": TASKS, "num_fewshot": 0, "limit": args.limit,
            "note": "identical invocation to the recorded base run on the same machine",
        },
        "base_rates": BASE_RATES,
        "scores": scores,
        "retention_per_metric": retention,
        "mean_retention_headline": mean_retention,
        "heldout_ppl_by_budget": budget_ppl,
        "base_heldout_ppl": base_ppl,
        "ppl_ratio_vs_base": (
            budget_ppl[str(full_depth)] / base_ppl
            if base_ppl and str(full_depth) in budget_ppl
            else None
        ),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== exp15 evaluation ===")
    for key in HEADLINE:
        if key in scores:
            print(f"  {key:42} {scores[key]:.4f}  base {BASE_RATES[key]:.4f}  "
                  f"retention {retention[key]:.3f}")
    if mean_retention is not None:
        print(f"  mean headline retention: {mean_retention:.3f}")
    if budget_ppl:
        print(f"  budget sweep (held-out ppl): {budget_ppl}")
    if base_ppl:
        full = budget_ppl.get(str(full_depth))
        print(f"  base held-out ppl {base_ppl:.3f}"
              + (f" | converted/base ratio {full/base_ppl:.2f}x" if full else ""))


if __name__ == "__main__":
    main()
