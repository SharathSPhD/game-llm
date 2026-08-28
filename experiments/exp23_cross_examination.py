"""Cross-examination: players price each other's reasoning, not each other's votes.

F29 and F30 closed the answer-level arena. Eleven aggregation rules — voting,
market, robust-statistics, competence-weighted, calibrated — all landed at or
below a plain average of the players' option scores, while some player answered
correctly on 83% of questions against the best aggregate's 64%. The explanation
that survived every test is that all those rules reweight one fixed body of
evidence, and reweighting cannot recover what reweighting discards.

This file changes what is on the table. Each player writes out a full solution,
and then every player scores every other player's solution. A chain of reasoning
is evidence its reader did not have: a model that would not have found the
substitution can still recognise that the substitution goes through. That is the
one channel by which a council can learn something at inference time rather than
merely re-weigh what it already believed, and it is the sequential mechanism the
answer-level arena structurally cannot contain.

Six selection rules run over the same generated candidates, so they differ only
in how the cross-examination scores are used:

  best-single      each player answers alone — the bar from the ladder
  self-preference  each candidate scored only by its own author
  cross-exam       every player scores every candidate, summed
  leave-one-out    a candidate is priced by everyone except its author (F6)
  equilibrium      the influence game of ADR 0008 played over candidates
  majority         self-consistency: the most common extracted answer
  oracle           any candidate correct, the ceiling the council could reach

Scoring is over *text*, so players need not share a tokenizer — each reads a
peer's solution through its own. That is a property the token-level decoder does
not have, and it widens which models can sit on a council.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp16_auction_real import build_tasks, extract_answer, is_correct  # noqa: E402

from kinetic_ai.decode.equilibrium import EquilibriumConfig, solve_equilibrium  # noqa: E402


@torch.no_grad()
def generate_solution(
    model: Any, tok: Any, prompt: str, max_new_tokens: int, device: str
) -> str:
    """One player's own attempt, prompted the way it was instruction-tuned.

    The chat template matters more than it looks: F28 found the ladder's GSM8K
    column measured whether a model emits a particular answer convention rather
    than whether it can do the arithmetic, purely because the raw prompt format
    was not the one these checkpoints were tuned on.
    """
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok(text, return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    return tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)


@torch.no_grad()
def score_solution(
    model: Any, tok: Any, prompt: str, solution: str, device: str
) -> float:
    """Mean log-probability a player assigns to a solution it is shown.

    Per-token rather than total, because candidates differ in length and a sum
    would price brevity rather than quality. This is the reader's valuation of
    the writer's reasoning, and it is the quantity the market below clears on.
    """
    head = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    head_ids = tok(head, return_tensors="pt").input_ids.to(device)
    full_ids = tok(head + solution, return_tensors="pt").input_ids.to(device)
    if full_ids.shape[1] <= head_ids.shape[1]:
        return float("-inf")
    logits = model(full_ids).logits[:, :-1].float()
    targets = full_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    answer_logp = logp[:, head_ids.shape[1] - 1 :]
    return float(answer_logp.mean().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp23_cross_exam.yaml")
    ap.add_argument("--out", default="results/scale/exp23")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["seed"] = args.seed
    device = cfg.get("device", "cuda:0")
    torch.manual_seed(args.seed)

    names: list[str] = list(cfg["players"])
    # Seeded on both halves: greedy decoding makes a fixed slice identical
    # across seeds, which would leave the mathematics half — the domain where
    # the council is most heterogeneous — with no seed variation at all.
    tasks = build_tasks(cfg, shuffle_math=True)

    # The inherited multiple-choice prompt asks for a bare letter and forbids
    # explanation, which makes it unusable here. Seed 42 showed why: the general
    # model complied and emitted "\boxed{B}" in ten characters while the
    # mathematics model ignored the instruction and wrote 444 characters of
    # derivation, so a per-token valuation compared a handful of tokens against
    # hundreds of highly predictable ones and preferred the verbose answer on 52
    # of 60 questions — selecting the player that was right on 12 of them over
    # players right on 27 and 28. That measures instruction compliance and
    # length, not reasoning quality. Cross-examination prices reasoning, so the
    # arena has to contain reasoning on both halves for the mechanism to be
    # under test at all.
    for task in tasks:
        if task["kind"] == "letter":
            task["prompt"] = (
                task["prompt"]
                .replace(
                    "Reply with ONLY the single letter of the correct option in "
                    "\\boxed{}. Do not explain.",
                    "Reason briefly, then give the single letter of the correct "
                    "option in \\boxed{}.",
                )
            )

    budget = {
        "number": int(cfg["generation"]["max_new_tokens_math"]),
        "letter": int(cfg["generation"]["max_new_tokens_general"]),
    }

    # Players are loaded one at a time for generation, then one at a time again
    # for scoring, so peak memory is one model rather than the whole council.
    candidates: dict[int, dict[str, str]] = {i: {} for i in range(len(tasks))}
    for name in names:
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
        print(f"generated: {name}", flush=True)

    scores: dict[int, dict[str, dict[str, float]]] = {
        i: {r: {} for r in names} for i in range(len(tasks))
    }
    for reader in names:
        tok = AutoTokenizer.from_pretrained(reader)
        model = AutoModelForCausalLM.from_pretrained(
            reader, dtype=torch.bfloat16, device_map=device
        ).eval()
        for i, task in enumerate(tasks):
            for writer in names:
                scores[i][reader][writer] = score_solution(
                    model, tok, task["prompt"], candidates[i][writer], device
                )
        del model
        torch.cuda.empty_cache()
        print(f"scored by: {reader}", flush=True)

    results: dict[str, int] = Counter()
    per_domain: dict[str, Counter] = {}
    records = []
    for i, task in enumerate(tasks):
        kind, gold = task["kind"], task["gold"]
        preds = {n: extract_answer(candidates[i][n], kind) for n in names}
        right = {n: is_correct(preds[n], gold, kind) for n in names}

        mat = torch.tensor(
            [[scores[i][r][w] for w in names] for r in names], dtype=torch.float32
        )  # [reader, writer]

        total = mat.sum(dim=0)
        own = torch.tensor([mat[j, j] for j in range(len(names))])
        loo = total - own

        picks = {
            "cross_exam": names[int(total.argmax())],
            "leave_one_out": names[int(loo.argmax())],
            "self_preference": names[int(own.argmax())],
        }

        # The influence game over candidates: readers are the players and the
        # candidates are the options, so this is exactly ADR 0008's construction
        # applied where the options carry information the readers did not have.
        ell = torch.log_softmax(mat, dim=-1)
        y = solve_equilibrium(ell, EquilibriumConfig(beta=cfg["equilibrium"]["beta"],
                                                     tau=cfg["equilibrium"]["tau"]))
        assert isinstance(y, torch.Tensor)
        picks["equilibrium"] = names[int(y.argmax())]

        votes = Counter(p for p in preds.values() if p is not None)
        maj = votes.most_common(1)[0][0] if votes else None

        row: dict[str, Any] = {
            "domain": task["domain"],
            "gold": gold,
            # Recorded per example so that a valuation preferring long or
            # short candidates is visible in the results rather than needing
            # to be discovered by reading generations.
            "lengths": {n: len(candidates[i][n]) for n in names},
        }
        for rule, winner in picks.items():
            ok = right[winner]
            results[rule] += int(ok)
            row[rule] = {"winner": winner, "correct": ok}
        results["majority"] += int(is_correct(maj, gold, kind))
        results["oracle"] += int(any(right.values()))
        for n in names:
            results[f"single::{n}"] += int(right[n])
        row["singles"] = right
        records.append(row)

        d = per_domain.setdefault(task["domain"], Counter())
        d["n"] += 1
        for rule in list(picks) + ["majority", "oracle"]:
            key = rule
            if rule == "majority":
                d[key] += int(is_correct(maj, gold, kind))
            elif rule == "oracle":
                d[key] += int(any(right.values()))
            else:
                d[key] += int(right[picks[rule]])

    n = len(tasks)
    # Length by domain, and the length of what each rule picked: if a rule's
    # winners are systematically longer or shorter than the field, its valuation
    # is keyed on verbosity and the accuracy figure should not be read as a
    # statement about reasoning.
    length_bias: dict[str, Any] = {}
    for rule in ("cross_exam", "leave_one_out", "self_preference", "equilibrium"):
        picked, field = [], []
        for rec in records:
            picked.append(rec["lengths"][rec[rule]["winner"]])
            field.extend(rec["lengths"].values())
        length_bias[rule] = {
            "mean_length_of_winner": round(sum(picked) / max(len(picked), 1), 1),
            "mean_length_of_field": round(sum(field) / max(len(field), 1), 1),
        }

    report: dict[str, Any] = {
        "n_tasks": n,
        "length_bias": length_bias,
        "seed": args.seed,
        "players": names,
        "accuracy": {k: round(v / n, 4) for k, v in sorted(results.items())},
        "per_domain": {
            dom: {k: round(v / c["n"], 4) for k, v in c.items() if k != "n"}
            | {"n": c["n"]}
            for dom, c in per_domain.items()
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_seed{args.seed}.json").write_text(json.dumps(report, indent=2))
    (out / f"candidates_seed{args.seed}.json").write_text(
        json.dumps({str(k): v for k, v in candidates.items()}, indent=2)
    )
    (out / f"records_seed{args.seed}.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
