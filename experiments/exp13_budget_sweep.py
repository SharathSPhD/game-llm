"""B1 budget-sweep rider (SPEC 0010 amendment): evaluate an anytime-trained
EqLM checkpoint at solver budgets {4, 8, 12} on the BLiMP subset.

Usage:
    python exp13_budget_sweep.py --checkpoint results/exp13_seed42/checkpoints/B1.pt \
        --output results/exp13_seed42/budget_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from experiments.exp05_eqlm_pretrain import create_gpt2_tokenizer_fn  # noqa: E402
except ImportError:
    from exp05_eqlm_pretrain import create_gpt2_tokenizer_fn  # type: ignore[no-redef]  # noqa: E402
from kinetic_ai.eval.blimp import evaluate_blimp_subset, load_blimp_subset  # noqa: E402
from kinetic_ai.models.eqlm import load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--budgets", default="4,8,12")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_checkpoint(args.checkpoint).to(device).eval()
    tokenizer_fn = create_gpt2_tokenizer_fn("gpt2")
    blimp = load_blimp_subset(num_phenomena=5, pairs_per_phenomenon=200)

    results = {}
    for budget in [int(b) for b in args.budgets.split(",")]:
        model.deq.config.max_iter = budget
        r = evaluate_blimp_subset(model, blimp, tokenizer_fn, device=device)
        results[str(budget)] = r
        print(f"budget {budget}: blimp {r['accuracy']:.3f}", flush=True)

    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
