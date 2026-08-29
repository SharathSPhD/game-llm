"""Does answer-stability under resampling identify who is right? (TRIZ cycle 29)

F30 and F32 established that whether a player is right on a given question is
not legible from the shape of its scores, and every rule that tried to read it
there failed. F42 sharpened the consequence: the council holds the answer — the
oracle sits 13 to 25 points above the best single member — and the only signal
that ever paid was a per-domain prior measurable in advance, which supplies
nothing at all when one member dominates every domain.

The impossibility argument leaves exactly one route open. Every refuted rule
reweighted *one observation* per player. Stability under perturbation is a
*second* observation: ask a player the same question several times under
resampling and paraphrase, and see whether it says the same thing. A player that
answers identically five times has told you something its single-pass logits
could not, and a player that flips between three answers has told you something
its confident-looking softmax actively concealed.

This is the empirical form of the rationality parameter the quantal response
framework treats as given, and it makes truthful bidding verifiable in the sense
F6 required: a player cannot claim a stability it does not exhibit, because the
bid is measured rather than reported.

This file tests the precondition only, and is deliberately cheap. If stability
does not separate correct answers from incorrect ones, no weighting built on it
can work and nothing further is spent. The test runs on the council where one
member dominates both domains, because that is where routing is worth exactly
nothing and any gain must come from a per-question signal.
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
from exp28_second_family import PLAYERS, _normalise  # noqa: E402

#: Paraphrase prefixes: perturbing the prompt as well as the sampler, so
#: stability measures robustness of the answer rather than of the decoder alone.
PARAPHRASES = [
    "",
    "Work carefully. ",
    "Take your time and check each step. ",
    "Solve this problem. ",
    "Answer the following, showing your reasoning. ",
]


@torch.no_grad()
def sample_answers(
    model: Any, tok: Any, prompt: str, kind: str, max_new: int, device: str, k: int
) -> list[str | None]:
    """k answers from one player under resampling and paraphrase."""
    out: list[str | None] = []
    for i in range(k):
        text = tok.apply_chat_template(
            [{"role": "user", "content": PARAPHRASES[i % len(PARAPHRASES)] + prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tok(text, return_tensors="pt").to(device)
        gen = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        body = tok.decode(
            gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        )
        out.append(_normalise(extract_answer(body, kind), kind))
    return out


def stability(answers: list[str | None]) -> float:
    """Fraction of samples agreeing with the player's own modal answer.

    Abstentions count against stability: a player that fails to produce a
    parseable answer has not demonstrated it knows anything, and treating that
    as neutral would let unreliability hide.
    """
    if not answers:
        return 0.0
    votes = Counter(a for a in answers if a is not None)
    if not votes:
        return 0.0
    return votes.most_common(1)[0][1] / len(answers)


def modal(answers: list[str | None]) -> str | None:
    votes = Counter(a for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp23_cross_exam.yaml")
    ap.add_argument("--out", default="results/scale/exp29_stability")
    ap.add_argument("--seed", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-tasks", type=int, default=60)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = args.seed
    device = cfg.get("device", "cuda:0")
    torch.manual_seed(args.seed)

    tasks = build_tasks(cfg, shuffle_math=True)
    for t in tasks:
        if t["kind"] == "letter":
            t["prompt"] = t["prompt"].replace(
                "Reply with ONLY the single letter of the correct option in "
                "\\boxed{}. Do not explain.",
                "Reason briefly, then give the single letter of the correct "
                "option in \\boxed{}.",
            )
    # Balanced subset across both domains.
    half = args.n_tasks // 2
    math_t = [t for t in tasks if t["kind"] == "number"][:half]
    gen_t = [t for t in tasks if t["kind"] == "letter"][:half]
    tasks = math_t + gen_t

    budget = {
        "number": int(cfg["generation"]["max_new_tokens_math"]),
        "letter": int(cfg["generation"]["max_new_tokens_general"]),
    }

    samples: dict[int, dict[str, list[str | None]]] = {
        i: {} for i in range(len(tasks))
    }
    for name in PLAYERS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map=device
        ).eval()
        for i, t in enumerate(tasks):
            samples[i][name] = sample_answers(
                model, tok, t["prompt"], t["kind"], budget[t["kind"]], device, args.k
            )
        del model
        torch.cuda.empty_cache()
        print(f"sampled: {name}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"samples_seed{args.seed}.json").write_text(
        json.dumps({str(k): v for k, v in samples.items()})
    )
    (out / f"tasks_seed{args.seed}.json").write_text(
        json.dumps([{k: t[k] for k in ("domain", "gold", "kind")} for t in tasks])
    )

    # The precondition: does stability separate right from wrong?
    buckets: dict[str, list[int]] = {}
    per_player: dict[str, dict[str, Any]] = {}
    for name in PLAYERS:
        s_right, s_wrong = [], []
        for i, t in enumerate(tasks):
            answers = samples[i][name]
            m = modal(answers)
            correct = m is not None and is_correct(m, t["gold"], t["kind"])
            (s_right if correct else s_wrong).append(stability(answers))
        per_player[name] = {
            "n_correct": len(s_right),
            "n_wrong": len(s_wrong),
            "mean_stability_when_right": round(
                sum(s_right) / len(s_right), 4
            ) if s_right else None,
            "mean_stability_when_wrong": round(
                sum(s_wrong) / len(s_wrong), 4
            ) if s_wrong else None,
        }
        for v in s_right:
            buckets.setdefault("right", []).append(v)
        for v in s_wrong:
            buckets.setdefault("wrong", []).append(v)

    r, w = buckets.get("right", []), buckets.get("wrong", [])
    sep = None
    if r and w:
        mr, mw = sum(r) / len(r), sum(w) / len(w)
        vr = sum((x - mr) ** 2 for x in r) / max(len(r) - 1, 1)
        vw = sum((x - mw) ** 2 for x in w) / max(len(w) - 1, 1)
        se = math.sqrt(vr / len(r) + vw / len(w))
        sep = {
            "mean_when_right": round(mr, 4),
            "mean_when_wrong": round(mw, 4),
            "difference": round(mr - mw, 4),
            "welch_t": round((mr - mw) / se, 2) if se > 0 else None,
        }

    report = {
        "seed": args.seed,
        "k": args.k,
        "n_tasks": len(tasks),
        "players": PLAYERS,
        "per_player": per_player,
        "pooled_separation": sep,
        "precondition_met": bool(sep and sep["welch_t"] and sep["welch_t"] >= 3.0),
    }
    (out / "probe_results.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
