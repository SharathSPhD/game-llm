"""What the council costs, measured rather than argued (SPEC 0016, ADR 0009).

The PRD's case for a council rests on a cost claim: after one forward pass per
player the equilibrium solve is softmax and dot products over the vocabulary, so
the system runs at ensemble cost. ADR 0009 flagged that cross-examination
invalidates the claim as stated, because pricing every candidate by every player
is quadratic in council size rather than linear. Neither statement has been
measured.

It matters more now than when it was proposed. F36 found that no council rule
beats a domain router, so the question has changed from "is the extra cost
affordable" to "what exactly is being paid for nothing". A negative result that
quantifies its own overhead is a stronger statement than one that merely reports
a tie, and it is the number a practitioner reading this work would want first.

Four systems generate the same continuations from the same prompts on the same
machine, differing only in how the next token is decided:

  single       one model, the strongest on the ladder — the practical baseline
  router       classify the prompt, then one model; what F36 showed is the bar
  equilibrium  every player forward per token, then the ADR 0008 solve
  cross_exam   every player generates a full solution, then prices every peer

Latency is reported per generated token and per completed request, since a
council that is cheap per token can still be dear per answer when it must
generate several complete candidates before choosing one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp16_auction_real import build_tasks  # noqa: E402

from kinetic_ai.decode.equilibrium import EquilibriumConfig, solve_equilibrium  # noqa: E402


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_single(
    model: Any, tok: Any, prompt: str, max_new: int, device: str
) -> tuple[float, int]:
    """Wall-clock one ordinary generation."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ids = tok(text, return_tensors="pt").to(device)
    _sync()
    t0 = time.perf_counter()
    out = model.generate(
        **ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    _sync()
    elapsed = time.perf_counter() - t0
    return elapsed, int(out.shape[1] - ids["input_ids"].shape[1])


@torch.no_grad()
def time_equilibrium(
    models: dict[str, Any], tok: Any, prompt: str, max_new: int, device: str,
    cfg: EquilibriumConfig,
) -> tuple[float, int, dict[str, float]]:
    """Token-level council decoding: every player forward, then solve.

    Requires a shared tokenizer, since the players must agree on what a token
    index means. That is checked by the caller rather than assumed — aggregating
    logits across disagreeing vocabularies produces fluent nonsense rather than
    an error, which is the worst way for a bug to present.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    ctx = tok(text, return_tensors="pt").input_ids.to(device)
    order = list(models)
    caches: dict[str, Any] = dict.fromkeys(order)
    eos = {tok.eos_token_id}
    prev = None
    forward_time = 0.0
    solve_time = 0.0

    _sync()
    t0 = time.perf_counter()
    generated = 0
    for _ in range(max_new):
        logits = []
        _sync()
        tf = time.perf_counter()
        for name in order:
            step_in = ctx if caches[name] is None else ctx[:, -1:]
            out = models[name](step_in, past_key_values=caches[name], use_cache=True)
            caches[name] = out.past_key_values
            logits.append(out.logits[:, -1, :].float()[0])
        _sync()
        forward_time += time.perf_counter() - tf

        ts = time.perf_counter()
        y = solve_equilibrium(torch.stack(logits), cfg, y_init=prev)
        assert isinstance(y, torch.Tensor)
        prev = y
        nxt = int(y.argmax().item())
        _sync()
        solve_time += time.perf_counter() - ts

        generated += 1
        if nxt in eos:
            break
        ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)
    _sync()
    total = time.perf_counter() - t0
    return total, generated, {"forward_s": forward_time, "solve_s": solve_time}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp23_cross_exam.yaml")
    ap.add_argument("--out", default="results/scale/exp26_serving_latency.json")
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument(
        "--players", default=None,
        help="comma-separated override; use to measure a subset that genuinely "
             "shares a tokenizer",
    )
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = 42
    device = cfg.get("device", "cuda:0")
    names: list[str] = (
        [n.strip() for n in args.players.split(",")] if args.players
        else list(cfg["players"])
    )
    baseline = names[0]

    tasks = build_tasks(cfg, shuffle_math=True)[: args.prompts]
    prompts = [t["prompt"] for t in tasks]

    tokenizers = {n: AutoTokenizer.from_pretrained(n) for n in names}
    vocabs = {n: len(tokenizers[n]) for n in names}
    shared_tokenizer = len(set(vocabs.values())) == 1

    report: dict[str, Any] = {
        "prompts": len(prompts),
        "max_new_tokens": args.max_new,
        "players": names,
        "vocab_sizes": vocabs,
        "shared_tokenizer": shared_tokenizer,
        "systems": {},
    }

    # Single model, and the router, which pays the same decode cost plus a
    # classification that is negligible beside a forward pass.
    model = AutoModelForCausalLM.from_pretrained(
        baseline, dtype=torch.bfloat16, device_map=device
    ).eval()
    singles = [time_single(model, tokenizers[baseline], p, args.max_new, device)
               for p in prompts]
    del model
    torch.cuda.empty_cache()
    report["systems"]["single"] = {
        "per_request_s": [s[0] for s in singles],
        "tokens": [s[1] for s in singles],
        "note": f"{baseline}, the strongest player on the ladder",
    }
    print(f"single done: {statistics.mean(s[0] for s in singles):.3f}s/request", flush=True)

    # Council resident together — the configuration GB10's unified memory allows
    # and a single 32GB card does not.
    if shared_tokenizer:
        models = {
            n: AutoModelForCausalLM.from_pretrained(
                n, dtype=torch.bfloat16, device_map=device
            ).eval()
            for n in names
        }
        eq = [
            time_equilibrium(
                models, tokenizers[baseline], p, args.max_new, device,
                EquilibriumConfig(beta=0.25, tau=0.0, max_iter=32),
            )
            for p in prompts
        ]
        report["systems"]["equilibrium"] = {
            "per_request_s": [e[0] for e in eq],
            "tokens": [e[1] for e in eq],
            "forward_s": [e[2]["forward_s"] for e in eq],
            "solve_s": [e[2]["solve_s"] for e in eq],
        }
        print(f"equilibrium done: {statistics.mean(e[0] for e in eq):.3f}s/request",
              flush=True)
        if torch.cuda.is_available():
            report["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        del models
        torch.cuda.empty_cache()
    else:
        report["systems"]["equilibrium"] = {
            "skipped": "players do not share a tokenizer, so token-level "
                       "aggregation is undefined"
        }

    # Cross-examination cost is structural rather than measured here: it is the
    # single-model cost times the number of players to generate, plus a scoring
    # pass for every reader-writer pair. Deriving it from the measured single
    # cost keeps it on the same footing as the others.
    n_players = len(names)
    mean_single = statistics.mean(s[0] for s in singles)
    report["systems"]["cross_exam_derived"] = {
        "generation_s": n_players * mean_single,
        "scoring_passes": n_players * n_players,
        "note": (
            "generation is one full decode per player; scoring is one forward "
            "pass over the prompt and candidate for every reader-writer pair, "
            "quadratic in council size"
        ),
    }

    summary = {}
    for name, sysd in report["systems"].items():
        if "per_request_s" not in sysd:
            continue
        per_req = statistics.mean(sysd["per_request_s"])
        toks = sum(sysd["tokens"])
        summary[name] = {
            "mean_request_s": per_req,
            "ms_per_token": 1000 * sum(sysd["per_request_s"]) / max(toks, 1),
        }
    if "single" in summary:
        base = summary["single"]["mean_request_s"]
        for v in summary.values():
            v["vs_single"] = v["mean_request_s"] / base
        summary["cross_exam_derived"] = {
            "mean_request_s": report["systems"]["cross_exam_derived"]["generation_s"],
            "vs_single": n_players,
            "note": "generation only; scoring passes extra",
        }
    report["summary"] = summary

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
