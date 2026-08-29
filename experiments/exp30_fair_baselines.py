"""The baselines the council actually has to beat (cycle 29, fairness audit).

The confirmed eight-point margin is measured against the strongest single member
of the council decoded greedily once. That is a legitimate baseline and it is not
a sufficient one, because the council spends more than one generation per request
and holds more than one model in memory. Two comparisons are owed, and both are
capable of destroying the claim, which is why they are run.

The first is matched compute. The deployed system uses 1.258 expected
generations per request; a practitioner given that budget would not build a
council, they would sample the single best model several times and take the
majority answer. Self-consistency at two, three and five samples brackets and
exceeds the council's compute, so if the council's advantage is really an
advantage of spending more compute, this measurement will show it.

The second is matched capacity. The council holds 6.34B parameters resident
against the baseline's 1.54B, a factor of four. A single model of comparable
total size is therefore the honest capacity-matched comparison, and
Qwen2.5-7B-Instruct is the nearest general-purpose model to that total.

Both run on the same confirmation questions, with the same prompts, the same
extraction and the same scoring as the result they are testing, so nothing but
the system under test differs.
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
from exp27_anchored_vote import _normalise, load_seed  # noqa: E402

BASELINE = "Qwen/Qwen2.5-1.5B-Instruct"
CAPACITY_MATCHED = "Qwen/Qwen2.5-7B-Instruct"
SELF_CONSISTENCY_K = [2, 3, 5]


@torch.no_grad()
def generate(model: Any, tok: Any, prompt: str, max_new: int, device: str,
             sample: bool) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(device)
    kw: dict[str, Any] = {"do_sample": False}
    if sample:
        kw = {"do_sample": True, "temperature": 0.7, "top_p": 0.95}
    out = model.generate(
        **ids, max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw,
    )
    return tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)


def _right(cls: str | None, r: dict[str, Any]) -> bool:
    return cls is not None and is_correct(cls, r["gold"], r["kind"])


def paired(a: list[bool], b: list[bool]) -> dict[str, float]:
    w = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    lo = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return {"wins": w, "losses": lo,
            "z": round((w - lo) / math.sqrt(max(w + lo, 1)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/exp23_confirm")
    ap.add_argument("--out", default="results/scale/exp30_fair_baselines.json")
    ap.add_argument("--config", default="configs/exp23_cross_exam.yaml")
    ap.add_argument("--skip-capacity", action="store_true")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda:0")
    budget = {
        "number": int(cfg["generation"]["max_new_tokens_math"]),
        "letter": int(cfg["generation"]["max_new_tokens_general"]),
    }

    # The stored records carry no prompt, so each seed's task set is rebuilt
    # from the same seeded construction that produced it and then checked
    # question by question against the stored gold answers. A silent misalignment
    # here would regenerate against the wrong questions and produce a baseline
    # that looks weak for reasons having nothing to do with the model, so the
    # alignment is asserted rather than assumed.
    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    for rp in sorted(root.glob("records_seed*.json")):
        seed_tag = rp.stem.replace("records_", "")
        seed_rows = load_seed(root / f"candidates_{seed_tag}.json", rp)
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = int(seed_tag.replace("seed", ""))
        tasks = build_tasks(seed_cfg, shuffle_math=True)
        for t in tasks:
            if t["kind"] == "letter":
                t["prompt"] = t["prompt"].replace(
                    "Reply with ONLY the single letter of the correct option in "
                    "\\boxed{}. Do not explain.",
                    "Reason briefly, then give the single letter of the correct "
                    "option in \\boxed{}.",
                )
        if len(tasks) != len(seed_rows):
            raise SystemExit(
                f"{seed_tag}: rebuilt {len(tasks)} tasks for {len(seed_rows)} records"
            )
        for t, r in zip(tasks, seed_rows, strict=True):
            if str(t["gold"]).strip() != str(r["gold"]).strip():
                raise SystemExit(
                    f"{seed_tag}: rebuilt task set does not align with the stored "
                    f"records (gold {t['gold']!r} vs {r['gold']!r})"
                )
            r["prompt"] = t["prompt"]
        rows.extend(seed_rows)
    n = len(rows)

    # The two systems already measured, recomputed here so every number in the
    # comparison table comes from one place.
    def council(r: dict[str, Any]) -> str | None:
        c = r["router_class"]
        if c is not None:
            return c
        v = Counter(x for x in r["answers"].values() if x is not None)
        return v.most_common(1)[0][0] if v else None

    systems: dict[str, list[bool]] = {
        "baseline_single_greedy": [_right(r["answers"][BASELINE], r) for r in rows],
        "council_system": [_right(council(r), r) for r in rows],
    }
    cost: dict[str, float] = {
        "baseline_single_greedy": 1.0,
        "council_system": 1.258,
    }

    # Matched compute: self-consistency on the single best model.
    tok = AutoTokenizer.from_pretrained(BASELINE)
    model = AutoModelForCausalLM.from_pretrained(
        BASELINE, dtype=torch.bfloat16, device_map=device
    ).eval()
    max_k = max(SELF_CONSISTENCY_K)
    samples: list[list[str | None]] = []
    for i, r in enumerate(rows):
        got = [
            _normalise(
                extract_answer(
                    generate(model, tok, r["prompt"], budget[r["kind"]], device, True),
                    r["kind"],
                ),
                r["kind"],
            )
            for _ in range(max_k)
        ]
        samples.append(got)
        if (i + 1) % 60 == 0:
            print(f"self-consistency: {i + 1}/{n} questions", flush=True)
    del model
    torch.cuda.empty_cache()

    for k in SELF_CONSISTENCY_K:
        picks = []
        for got in samples:
            v = Counter(a for a in got[:k] if a is not None)
            picks.append(v.most_common(1)[0][0] if v else None)
        systems[f"self_consistency_k{k}"] = [
            _right(p, r) for p, r in zip(picks, rows, strict=True)
        ]
        cost[f"self_consistency_k{k}"] = float(k)

    # Matched capacity: one model of comparable total size.
    if not args.skip_capacity:
        try:
            tok7 = AutoTokenizer.from_pretrained(CAPACITY_MATCHED)
            m7 = AutoModelForCausalLM.from_pretrained(
                CAPACITY_MATCHED, dtype=torch.bfloat16, device_map=device
            ).eval()
            picks = []
            for i, r in enumerate(rows):
                txt = generate(m7, tok7, r["prompt"], budget[r["kind"]], device, False)
                picks.append(_normalise(extract_answer(txt, r["kind"]), r["kind"]))
                if (i + 1) % 60 == 0:
                    print(f"capacity-matched: {i + 1}/{n}", flush=True)
            del m7
            torch.cuda.empty_cache()
            systems["capacity_matched_7b_greedy"] = [
                _right(p, r) for p, r in zip(picks, rows, strict=True)
            ]
            cost["capacity_matched_7b_greedy"] = 1.0
        except Exception as exc:  # noqa: BLE001 - report rather than abort the run
            print(f"capacity-matched arm unavailable: {type(exc).__name__}: {exc}")

    council_hits = systems["council_system"]
    report = {
        "n": n,
        "accuracy": {k: round(sum(v) / n, 4) for k, v in systems.items()},
        "generations_per_request": cost,
        "council_vs": {
            k: paired(council_hits, v)
            for k, v in systems.items()
            if k != "council_system"
        },
        "note": (
            "Same questions, prompts, extraction and scoring as the confirmation "
            "run. Self-consistency arms reuse one set of five samples, so k=2 and "
            "k=3 are prefixes of k=5 rather than independent draws."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
