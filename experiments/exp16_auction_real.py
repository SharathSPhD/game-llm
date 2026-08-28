"""Experiment 16 — H9 (SPEC 0013): truthful auction decoding over real specialists.

Retests the claim that failed at toy scale (F23) under conditions that remove
both of that experiment's weaknesses:

  * the specialists are genuinely strong and genuinely different
    (Qwen2.5-{Math,Coder,Instruct}-1.5B — verified to share one tokenizer, a
    hard requirement for token-level aggregation);
  * the metric is objective task accuracy, not a judge LM whose style prior
    dominated the F23 comparison.

Systems compared, all in CLOSED-LOOP generation:
  S_*      each specialist alone
  ENS      uniform logit-average ensemble
  AUC      per-token second-price auction, bid = own max-prob confidence
  AUC_CTX  context-aware bids (the F23 follow-up): the bid is fixed per
           sequence from each model's confidence on the PROMPT, testing
           whether prompt-level signal fixes the style drift token-level
           bidding suffered
  ORACLE   upper bound: always the specialist that owns the domain

All systems decode from an identical prompt string under one shared chat
template, so the only difference is who chooses the token.

Usage:
    python exp16_auction_real.py --config configs/exp16_auction.yaml \
        --output results/exp16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.decode.equilibrium import (  # noqa: E402
    EquilibriumConfig,
    solve_equilibrium,
)


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


# ── task construction ────────────────────────────────────────────────────────


def build_tasks(cfg: dict) -> list[dict]:
    """Mixed-domain prompt set with objective answers."""
    from datasets import load_dataset

    n_math = int(cfg["tasks"]["n_gsm8k"])
    n_gen = int(cfg["tasks"]["n_mmlu"])
    tasks: list[dict] = []

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for row in gsm.select(range(n_math)):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        tasks.append(
            {
                "domain": "math",
                "prompt": (
                    f"{row['question']}\n\n"
                    "Solve step by step, then end with the final numeric answer "
                    "in \\boxed{}."
                ),
                "gold": gold,
                "kind": "number",
            }
        )

    mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=cfg["seed"])
    for row in mmlu.select(range(n_gen)):
        choices = "\n".join(f"{c}. {t}" for c, t in zip("ABCD", row["choices"]))
        tasks.append(
            {
                "domain": "general",
                "prompt": (
                    f"{row['question']}\n{choices}\n\n"
                    "Reply with ONLY the single letter of the correct option in "
                    "\\boxed{}. Do not explain."
                ),
                "gold": "ABCD"[int(row["answer"])],
                "kind": "letter",
            }
        )
    return tasks


def extract_answer(text: str, kind: str) -> str | None:
    """Pull the final answer, tolerating the formats these models actually use.

    Priority: \\boxed{...} (what Qwen math models emit) > an explicit ####
    marker > the last candidate in the trailing text.
    """
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    candidates = [boxed[-1]] if boxed else []
    if "####" in text:
        candidates.append(text.split("####")[-1])
    candidates.append(text[-200:])

    for cand in candidates:
        if kind == "number":
            nums = re.findall(r"-?\d+(?:\.\d+)?", cand.replace(",", "").replace("$", ""))
            if nums:
                return nums[-1]
        else:
            letters = re.findall(r"\b([ABCD])\b", cand.upper())
            if letters:
                return letters[-1]
    return None


def is_correct(pred: str | None, gold: str, kind: str) -> bool:
    if pred is None:
        return False
    if kind == "number":
        try:
            return abs(float(pred) - float(gold)) < 1e-4
        except ValueError:
            return False
    return pred.strip().upper() == gold.strip().upper()


# ── decoding ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_system(
    system: str,
    models: dict[str, Any],
    order: list[str],
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: set[int],
    device: str,
    ctx_bids: torch.Tensor | None = None,
    trace_limit: int = 0,
    eq_config: EquilibriumConfig | None = None,
) -> tuple[list[int], list[dict]]:
    """Closed-loop greedy generation under one aggregation rule."""
    eq_config = eq_config or EquilibriumConfig()
    ctx = input_ids.to(device)
    caches: dict[str, Any] = {name: None for name in order}
    generated: list[int] = []
    traces: list[dict] = []
    prev_eq: torch.Tensor | None = None  # warm start for equilibrium decoding

    for pos in range(max_new_tokens):
        logits: dict[str, torch.Tensor] = {}
        for name in order:
            model = models[name]
            step_in = ctx if caches[name] is None else ctx[:, -1:]
            out = model(step_in, past_key_values=caches[name], use_cache=True)
            caches[name] = out.past_key_values
            logits[name] = out.logits[:, -1, :].float()

        if system in models:
            nxt = int(logits[system].argmax(dim=-1).item())
            winner = system
            payment = None
        elif system == "ENS":
            stacked = torch.stack([logits[n] for n in order])
            nxt = int(stacked.mean(0).argmax(dim=-1).item())
            winner, payment = "ENS", None
        elif system.startswith("EQ"):
            # Equilibrium decoding (ADR 0008): the token distribution is the
            # solved QRE of the influence game among players, warm-started from
            # the previous position because consecutive equilibria are close.
            stacked = torch.stack([logits[n][0] for n in order])  # [N, V]
            y, eq_info = solve_equilibrium(
                stacked, eq_config, y_init=prev_eq, return_info=True
            )
            prev_eq = y
            nxt = int(y.argmax().item())
            winner, payment = system, None
            if pos < trace_limit:
                traces.append({
                    "position": pos,
                    "eq_iterations": eq_info["iterations"],
                    "eq_converged": eq_info["converged"],
                    "token": nxt,
                })
        elif system in ("AUC", "AUC_CTX"):
            if system == "AUC":
                bids = torch.tensor(
                    [F.softmax(logits[n], dim=-1).max().item() for n in order]
                )
            else:
                assert ctx_bids is not None
                bids = ctx_bids
            top = int(torch.argmax(bids).item())
            winner = order[top]
            payment = float(torch.sort(bids, descending=True).values[1].item())
            nxt = int(logits[winner].argmax(dim=-1).item())
            if pos < trace_limit:
                traces.append(
                    {
                        "position": pos,
                        "bids": [float(b) for b in bids.tolist()],
                        "winner": top,
                        "winner_name": winner,
                        "payment": payment,
                        "token": nxt,
                    }
                )
        else:
            raise ValueError(f"unknown system {system}")

        generated.append(nxt)
        if nxt in eos_ids:
            break
        ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)

    return generated, traces


@torch.no_grad()
def prompt_confidences(
    models: dict[str, Any], order: list[str], input_ids: torch.Tensor, device: str
) -> torch.Tensor:
    """Context-aware bid: exp(-mean NLL) of each model on the prompt itself."""
    ids = input_ids.to(device)
    vals = []
    for name in order:
        logits = models[name](ids).logits
        logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        nll = -logp.gather(2, ids[:, 1:].unsqueeze(2)).mean()
        vals.append(float(torch.exp(-nll).item()))
    return torch.tensor(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")
    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    result_file = out_dir / f"results_seed{seed}.json"
    if result_file.exists():
        print(f"[resume] seed {seed} complete")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec_cfg: dict[str, str] = cfg["specialists"]
    order = list(spec_cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["chat_template_from"])
    vocabs = set()
    models: dict[str, Any] = {}
    for name, repo in spec_cfg.items():
        models[name] = (
            AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(device).eval()
        )
        vocabs.add(len(AutoTokenizer.from_pretrained(repo)))
    if len(vocabs) != 1:
        raise RuntimeError(f"specialists must share a tokenizer; got vocab sizes {vocabs}")
    print(f"[models] {order} loaded, shared vocab {vocabs.pop()}", flush=True)

    tasks = build_tasks(cfg)
    print(f"[tasks] {len(tasks)} prompts: "
          f"{sum(t['domain']=='math' for t in tasks)} math / "
          f"{sum(t['domain']=='general' for t in tasks)} general", flush=True)

    eq_variants: dict[str, EquilibriumConfig] = {}
    for name, params in (cfg.get("equilibrium") or {}).items():
        eq_variants[name] = EquilibriumConfig(**params)
    systems = order + ["ENS", "AUC", "AUC_CTX", "ORACLE"] + list(eq_variants)
    oracle_map: dict[str, str] = cfg["oracle_map"]
    eos_ids = {tokenizer.eos_token_id} | {
        tokenizer.convert_tokens_to_ids(t)
        for t in cfg.get("extra_eos", [])
        if tokenizer.convert_tokens_to_ids(t) is not None
    }
    max_new_by_kind = {
        "number": int(cfg["generation"].get("max_new_tokens_math", 512)),
        "letter": int(cfg["generation"].get("max_new_tokens_general", 160)),
    }
    trace_limit = int(cfg["generation"].get("trace_positions", 0))

    records: list[dict] = []
    all_traces: list[dict] = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(chat, return_tensors="pt").input_ids
        ctx_bids = prompt_confidences(models, order, ids, device)

        row: dict[str, Any] = {"idx": i, "domain": task["domain"], "gold": task["gold"]}
        for system in systems:
            effective = oracle_map[task["domain"]] if system == "ORACLE" else system
            gen, traces = generate_system(
                effective if effective in models else system,
                models,
                order,
                ids,
                max_new_by_kind[task["kind"]],
                eos_ids,
                device,
                ctx_bids=ctx_bids,
                trace_limit=trace_limit if (system in ("AUC",) and i == 0) else 0,
                eq_config=eq_variants.get(system),
            )
            text = tokenizer.decode(gen, skip_special_tokens=True)
            pred = extract_answer(text, task["kind"])
            row[system] = {
                "correct": is_correct(pred, task["gold"], task["kind"]),
                "pred": pred,
                "n_tokens": len(gen),
            }
            if traces:
                all_traces = traces
        row["ctx_bids"] = [round(b, 5) for b in ctx_bids.tolist()]
        records.append(row)
        if (i + 1) % 5 == 0:
            done = {
                s: sum(r[s]["correct"] for r in records) / len(records) for s in systems
            }
            print(f"  [{i+1}/{len(tasks)}] {({k: round(v,3) for k,v in done.items()})} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)

    def acc(system: str, domain: str | None = None) -> float:
        rows = [r for r in records if domain is None or r["domain"] == domain]
        return sum(r[system]["correct"] for r in rows) / max(len(rows), 1)

    summary = {
        s: {
            "overall": acc(s),
            "math": acc(s, "math"),
            "general": acc(s, "general"),
        }
        for s in systems
    }
    best_single = max(order, key=lambda s: summary[s]["overall"])
    best_eq = max(eq_variants, key=lambda s: summary[s]["overall"]) if eq_variants else None
    verdict = (
        "MET"
        if summary["AUC"]["overall"] > summary[best_single]["overall"]
        else (
            "PARTIAL"
            if summary["AUC_CTX"]["overall"] > summary[best_single]["overall"]
            else "MISSED"
        )
    )

    results = {
        "experiment": "exp16_auction_real",
        "spec": "0013",
        "seed": seed,
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "specialists": spec_cfg,
        "n_tasks": len(tasks),
        "summary": summary,
        "best_single": best_single,
        "best_equilibrium": best_eq,
        "equilibrium_configs": {k: vars(v) for k, v in eq_variants.items()},
        "h9_score": verdict,
        "records": records,
        "wall_clock_s": time.time() - t0,
    }
    result_file.write_text(json.dumps(results, indent=2))
    (out_dir / f"traces_seed{seed}.json").write_text(json.dumps(all_traces, indent=2))

    print("\n=== exp16 summary (closed-loop, objective accuracy) ===")
    for s in systems:
        v = summary[s]
        print(f"  {s:9} overall {v['overall']:.3f}  math {v['math']:.3f}  general {v['general']:.3f}")
    print(f"  best single = {best_single} | H9 {verdict}")


if __name__ == "__main__":
    main()
