"""Damage probe for the final leap (SPEC 0020): what does gentle tying cost?

F35 measured conversion at twelve-fold reuse and found it unrecoverable at
affordable budgets. SPEC 0020 proposes two-fold reuse — ten cores applied twice,
outer layers kept explicit — on the argument that damage grows steeply with how
much is tied and a gentler surgery starts close enough to the base model for the
available budget to finish the job. That argument is a hypothesis about the
starting point, and it is checkable in minutes per configuration by measuring
perplexity immediately after surgery, before any training at all.

The probe exists to spend hours before the programme spends a week. SPEC 0020
fixes the kill criterion it feeds: initial damage above five times base
perplexity fails the gate.
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

from kinetic_ai.models.kinetic_lm import KineticConfig, convert_to_kinetic  # noqa: E402

BASE = "Qwen/Qwen2.5-1.5B-Instruct"

#: The configurations SPEC 0020 considers, gentlest first. Each is (n_pre,
#: n_post, n_cores, recursion_depth); the middle layer count must equal
#: n_cores * (layers each core replaces), and reuse is what the depth multiplies.
CONFIGS = [
    {"name": "pair_tying_10x2", "n_pre": 4, "n_post": 4, "n_cores": 10, "recursion_depth": 2},
    {"name": "quad_tying_5x4", "n_pre": 4, "n_post": 4, "n_cores": 5, "recursion_depth": 4},
    {"name": "pair_tying_12x2_thin_shell", "n_pre": 2, "n_post": 2, "n_cores": 12, "recursion_depth": 2},
]


@torch.no_grad()
def held_out_ppl(model: Any, tok: Any, texts: list[str], device: str) -> float:
    """Perplexity on fixed neutral text, identical for every configuration."""
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        if ids.shape[1] < 8:
            continue
        out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += float(out.loss) * n
        total_tokens += n
    return math.exp(total_nll / max(total_tokens, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/scale/exp37_damage_probe.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-texts", type=int, default=40)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    tok = AutoTokenizer.from_pretrained(BASE)
    texts = [
        r["text"]
        for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if len(r["text"]) > 400
    ][: args.n_texts]

    report: dict[str, Any] = {"base_model": BASE, "n_texts": len(texts), "configs": {}}

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map=device
    ).eval()
    base_params = sum(p.numel() for p in model.parameters())
    base_ppl = held_out_ppl(model, tok, texts, device)
    report["base"] = {"ppl": round(base_ppl, 3), "params_M": round(base_params / 1e6, 1)}
    print(f"base: ppl {base_ppl:.2f}, {base_params/1e6:.0f}M params", flush=True)
    del model
    torch.cuda.empty_cache()

    for cfg in CONFIGS:
        # A fresh copy per configuration: conversion mutates the model in place,
        # and reusing a mutated model would measure compounded surgeries.
        model = AutoModelForCausalLM.from_pretrained(
            BASE, dtype=torch.bfloat16, device_map=device
        ).eval()
        kc = KineticConfig(
            n_pre=cfg["n_pre"], n_post=cfg["n_post"], n_cores=cfg["n_cores"],
            recursion_depth=cfg["recursion_depth"], init_strategy="average",
        )
        try:
            model = convert_to_kinetic(model, kc)
        except Exception as exc:  # noqa: BLE001 - a failed surgery is a result
            report["configs"][cfg["name"]] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"{cfg['name']}: conversion failed — {exc}", flush=True)
            del model
            torch.cuda.empty_cache()
            continue
        from kinetic_ai.models.kinetic_lm import count_unique_params

        unique = count_unique_params(model)
        ppl = held_out_ppl(model, tok, texts, device)
        report["configs"][cfg["name"]] = {
            "ppl": round(ppl, 3),
            "damage_ratio": round(ppl / base_ppl, 3),
            "unique_params_M": round(unique / 1e6, 1),
            "param_fraction": round(unique / base_params, 4),
            "passes_gate": bool(ppl / base_ppl <= 5.0),
        }
        print(
            f"{cfg['name']}: ppl {ppl:.2f} ({ppl/base_ppl:.2f}x base), "
            f"{unique/1e6:.0f}M unique ({unique/base_params:.1%}), "
            f"gate {'PASS' if ppl/base_ppl <= 5.0 else 'FAIL'}",
            flush=True,
        )
        del model
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
