"""Equilibrium decoding over answer options, measured offline (SPEC 0016).

The decisive question for the paradigm is whether solving the influence game
beats averaging the players. Answering it on generated text costs a GPU pass per
parameter setting, which limits any sweep to a handful of prompts. Multiple
choice removes that cost: each player's per-option loglikelihoods are computed
once, stored, and then aggregated offline as many times as needed. The game is
the same one the decoder plays, with the K answer options standing in for the
vocabulary, so ``beta = 0`` is still exactly uniform ensembling and large
``beta`` still approaches routing.

Two properties of this arena are worth stating because they are not
concessions. Aggregating over options needs no shared tokenizer, so players that
could never share a decode loop can still be compared here. And the comparison
runs over thousands of examples rather than the tens a generation sweep affords,
which is what makes a small margin distinguishable from noise at all.

What it cannot show is behaviour over a sequence: warm starting, drift, and the
cost of solving per token are all properties of decoding that this measurement
does not reach. Those belong to the generative arm.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.decode.equilibrium import EquilibriumConfig, solve_equilibrium

PLAYERS = [
    "Qwen_Qwen3-1.7B",
    "Qwen_Qwen2.5-1.5B-Instruct",
    "Qwen_Qwen2.5-Math-1.5B-Instruct",
    "Qwen_Qwen2.5-Coder-1.5B-Instruct",
]
SHORT = {
    "Qwen_Qwen3-1.7B": "qwen3-1.7b",
    "Qwen_Qwen2.5-1.5B-Instruct": "qwen2.5-base",
    "Qwen_Qwen2.5-Math-1.5B-Instruct": "qwen2.5-math",
    "Qwen_Qwen2.5-Coder-1.5B-Instruct": "qwen2.5-coder",
}


def _load_samples(root: Path, player: str) -> dict[str, dict[int, dict[str, Any]]]:
    """Read one player's logged samples, keyed by task and document id."""
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    base = root / player
    for path in sorted(base.rglob("samples_*.jsonl")):
        task = path.name[len("samples_") :].rsplit("_", 1)[0]
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                out[task][rec["doc_id"]] = rec
    return out


@dataclass
class Example:
    """One question, scored by every player.

    ``raw`` holds the loglikelihood sums as the harness computed them and
    ``normed`` divides each by its option's length, which is the ``acc_norm``
    convention; keeping both lets the aggregation be checked against whichever
    metric the task is normally reported under.
    """

    raw: torch.Tensor
    normed: torch.Tensor
    gold: int

    def scores(self, normalised: bool) -> torch.Tensor:
        return self.normed if normalised else self.raw


def _example_matrix(records: list[dict[str, Any]], gold: int) -> Example | None:
    """Per-player option scores for one example.

    Returns raw loglikelihood sums, the same scores normalised per option by
    completion length, and the gold index. Length normalisation is the
    ``acc_norm`` convention, and it matters here because options of unequal
    length otherwise favour whichever player is most confident about long
    strings rather than most correct.
    """
    n_opts = {len(r["filtered_resps"]) for r in records}
    if len(n_opts) != 1:
        return None
    raw = torch.tensor(
        [[float(resp[0]) for resp in r["filtered_resps"]] for r in records],
        dtype=torch.float32,
    )
    lengths = torch.tensor(
        [[max(len(str(c)), 1) for c in _completions(records[0])]],
        dtype=torch.float32,
    )
    if not 0 <= gold < raw.shape[-1]:
        return None
    return Example(raw=raw, normed=raw / lengths, gold=gold)


def _completions(rec: dict[str, Any]) -> list[str]:
    """The candidate continuations, in the order their scores were recorded.

    The harness writes these either as a mapping ``gen_args_N -> {arg_0: context,
    arg_1: continuation}`` or as bare pairs, depending on version, and the order
    of the mapping is the order of ``filtered_resps``.
    """
    args = rec.get("arguments", {})
    if isinstance(args, dict):
        out = []
        for value in args.values():
            if isinstance(value, dict):
                out.append(str(value.get("arg_1", "")))
            else:
                out.append(str(value[1]))
        return out
    return [str(a[1]) for a in args]


def _raw_target(rec: dict[str, Any]) -> int | None:
    """The task's answer label as an integer, before any convention is assumed.

    Tasks disagree about what the integer means — ARC and HellaSwag label options
    from zero, WinoGrande from one — so this deliberately does not resolve the
    label into a position. ``_calibrate_offset`` does that from evidence.
    """
    # The harness's own ``target`` is authoritative and is consulted first. The
    # raw document fields are a fallback only, because they can disagree with it
    # under a different convention within a single task: ARC records carry
    # ``target`` "1" alongside ``answerKey`` 2 for the same question, and reading
    # the document field first mixes two conventions inside one task, which no
    # single offset can then repair.
    tgt = rec.get("target")
    if isinstance(tgt, int):
        return tgt
    if isinstance(tgt, str):
        if tgt.strip().isdigit():
            return int(tgt.strip())
        if len(tgt.strip()) == 1 and tgt.strip().upper() in "ABCDE":
            return "ABCDE".index(tgt.strip().upper())
        # Matching the target against the continuations only identifies an
        # option when exactly one matches. WinoGrande varies the *context*
        # between its two options and repeats the same continuation in both, so
        # a first-match lookup would silently always return option zero.
        comps = [c.strip() for c in _completions(rec)]
        if comps.count(tgt.strip()) == 1:
            return comps.index(tgt.strip())
    doc = rec.get("doc", {})
    for key in ("label", "answer", "gold", "answerKey"):
        if key in doc:
            val = doc[key]
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                if val.strip().isdigit():
                    return int(val.strip())
                if len(val.strip()) == 1 and val.strip().upper() in "ABCDE":
                    return "ABCDE".index(val.strip().upper())
    return None


def _calibrate_offset(records: list[dict[str, Any]]) -> int | None:
    """Recover a task's label convention from the harness's own scoring.

    Every logged record carries both the model's option scores and the ``acc``
    the harness awarded, so the correct label-to-position mapping is the one
    under which "the top-scoring option is the labelled one" agrees with the acc
    the harness actually recorded. Testing that, rather than assuming zero-based
    labels, is what catches an off-by-one: WinoGrande labels from one, and
    reading its labels as positions scored the players at 0.13 against a true
    0.63 — an error large enough to invert every conclusion, and silent, because
    a wrong index still produces a plausible-looking number.
    """
    best, best_hits = None, -1
    for offset in (0, -1):
        hits = 0
        total = 0
        for rec in records:
            tgt = _raw_target(rec)
            if tgt is None or "acc" not in rec:
                continue
            gold = tgt + offset
            scores = [float(r[0]) for r in rec["filtered_resps"]]
            if not 0 <= gold < len(scores):
                total += 1
                continue
            predicted = max(range(len(scores)), key=lambda i: scores[i])
            hits += int((predicted == gold) == (float(rec["acc"]) == 1.0))
            total += 1
        if total and hits > best_hits:
            best, best_hits = offset, hits
        if total:
            _CALIBRATION.setdefault(id(records), {})[offset] = hits / total
    return best


_CALIBRATION: dict[int, dict[int, float]] = {}


def collect(root: Path) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Align every player on the examples all of them scored.

    Returns the per-task examples and a provenance record of how each task's
    label convention was resolved, so a later reader can see that the mapping
    was established rather than assumed.
    """
    loaded = {p: _load_samples(root, p) for p in PLAYERS}
    tasks = set.intersection(*(set(v) for v in loaded.values()))
    per_task: dict[str, list[Example]] = {}
    provenance: dict[str, Any] = {}
    for task in sorted(tasks):
        ids = sorted(set.intersection(*(set(loaded[p][task]) for p in PLAYERS)))
        anchor = [loaded[PLAYERS[0]][task][i] for i in ids]
        offset = _calibrate_offset(anchor)
        agreement = _CALIBRATION.get(id(anchor), {})
        if offset is None or agreement.get(offset, 0.0) < 0.99:
            provenance[task] = {"dropped": True, "agreement": agreement}
            continue
        provenance[task] = {"offset": offset, "agreement": agreement[offset]}
        rows: list[Example] = []
        for doc_id in ids:
            records = [loaded[p][task][doc_id] for p in PLAYERS]
            tgt = _raw_target(records[0])
            if tgt is None:
                continue
            got = _example_matrix(records, tgt + offset)
            if got is not None:
                rows.append(got)
        if rows:
            per_task[task] = rows
    return per_task, provenance


def _as_player_logits(scores: torch.Tensor) -> torch.Tensor:
    """Put players on a common scale.

    Raw loglikelihood sums differ in magnitude between models for reasons that
    have nothing to do with which option is right, and the influence payoff is
    an inner product, so an uncalibrated player would win weight by sheer scale.
    Converting each player's scores to log-probabilities over the options is the
    faithful analogue of the decoder, where a softmax over the vocabulary is
    applied to every player before the game begins.
    """
    return torch.log_softmax(scores, dim=-1)


def evaluate(rows: list[Example], normalised: bool) -> dict[str, Any]:
    n = len(rows)
    singles = [0] * len(PLAYERS)
    oracle = 0
    majority = 0
    for row in rows:
        scores, gold = row.scores(normalised), row.gold
        picks = scores.argmax(dim=-1)
        hit = False
        for i, pick in enumerate(picks.tolist()):
            if pick == gold:
                singles[i] += 1
                hit = True
        oracle += int(hit)
        votes = torch.bincount(picks, minlength=scores.shape[-1])
        majority += int(votes.argmax().item() == gold)
    return {
        "n": n,
        "singles": {SHORT[p]: s / n for p, s in zip(PLAYERS, singles, strict=True)},
        "best_single": max(singles) / n,
        "best_single_name": SHORT[PLAYERS[singles.index(max(singles))]],
        "oracle_any_correct": oracle / n,
        "majority_vote": majority / n,
    }


def aggregate(rows: list[Example], cfg: EquilibriumConfig, normalised: bool) -> float:
    correct = 0
    for row in rows:
        ell = _as_player_logits(row.scores(normalised))
        y = solve_equilibrium(ell, cfg)
        assert isinstance(y, torch.Tensor)
        correct += int(y.argmax().item() == row.gold)
    return correct / len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/scale/agree")
    ap.add_argument("--out", default="results/scale/exp18_equilibrium_mc.json")
    ap.add_argument("--normalised", action="store_true", default=True)
    args = ap.parse_args()

    per_task, provenance = collect(Path(args.root))
    if not per_task:
        print("no aligned samples found", file=sys.stderr)
        return 1

    pooled: list[Example] = []
    for rows in per_task.values():
        pooled.extend(rows)

    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    taus = [0.0, 0.05, 0.1, 0.25, 0.5]

    report: dict[str, Any] = {
        "players": [SHORT[p] for p in PLAYERS],
        "tasks": {t: len(r) for t, r in per_task.items()},
        "label_calibration": provenance,
        "pooled": evaluate(pooled, args.normalised),
        "per_task": {t: evaluate(r, args.normalised) for t, r in per_task.items()},
        "sweep": [],
    }

    for beta in betas:
        for tau in taus:
            cfg = EquilibriumConfig(tau=tau, beta=beta, eta=0.5, max_iter=64, tol=1e-6)
            acc = aggregate(pooled, cfg, args.normalised)
            report["sweep"].append({"beta": beta, "tau": tau, "acc": acc})
            print(f"beta={beta:6.2f} tau={tau:4.2f}  pooled acc={acc:.4f}", flush=True)

    best = max(report["sweep"], key=lambda r: r["acc"])
    report["best"] = best
    base = next(r for r in report["sweep"] if r["beta"] == 0.0 and r["tau"] == 0.0)
    report["averaging"] = base
    report["margin_over_averaging"] = best["acc"] - base["acc"]

    # A margin is only meaningful against the spread of a coin flip on the same
    # number of examples, so record the standard error the comparison must clear.
    n = report["pooled"]["n"]
    p = base["acc"]
    report["stderr"] = math.sqrt(p * (1 - p) / n)

    report["per_task_best"] = {}
    for task, rows in per_task.items():
        cfg = EquilibriumConfig(
            tau=best["tau"], beta=best["beta"], eta=0.5, max_iter=64, tol=1e-6
        )
        avg = EquilibriumConfig(tau=0.0, beta=0.0, eta=0.5, max_iter=64, tol=1e-6)
        report["per_task_best"][task] = {
            "equilibrium": aggregate(rows, cfg, args.normalised),
            "averaging": aggregate(rows, avg, args.normalised),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("best", "averaging", "margin_over_averaging", "stderr")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
