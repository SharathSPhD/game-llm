"""Build the static data the replay-only app serves (SPEC 0025, ADR 0011).

With the serving host away, every number the app shows is pre-recorded from
the results tree at build time and committed under apps/web/data/. Run from
the repo root after any results change:

    .venv/bin/python scripts/build_app_data.py

Outputs (all JSON, all committed):
  apps/web/data/runs.json          run registry: one row per results.json
  apps/web/data/leaderboard.json   baseline ladder (Qwen council players)
  apps/web/data/ladder_exp40.json  the 1B twin's public ladder (F55)
  apps/web/data/council.json       F41/F54 council record, per seed and rule
  apps/web/data/results.json       findings mirror (copied from site/src/data)
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "apps" / "web" / "data"


def _load(p: Path) -> Any:
    return json.loads(p.read_text())


def _round(x: Any, nd: int = 4) -> Any:
    return round(float(x), nd) if isinstance(x, (int, float)) else x


# ── run registry ────────────────────────────────────────────────────────────

def _headline(data: dict[str, Any]) -> dict[str, Any]:
    """A compact, shape-tolerant summary of one results.json."""
    head: dict[str, Any] = {}
    if isinstance(data.get("metrics"), dict):
        head.update({k: _round(v) for k, v in list(data["metrics"].items())[:4]})
    arms = data.get("arms")
    if isinstance(arms, dict):
        for name, arm in arms.items():
            if isinstance(arm, dict):
                for key in ("blimp_accuracy", "accuracy", "heldout_ppl", "final_loss"):
                    if key in arm:
                        head[f"{name}.{key}"] = _round(arm[key])
                        break
    for key in ("accuracy", "best_single_name", "oracle_any_player"):
        if key in data and not isinstance(data[key], dict):
            head[key] = _round(data[key])
    return head


def build_runs() -> list[dict[str, Any]]:
    rows = []
    for p in sorted(RESULTS.rglob("results.json")):
        try:
            data = _load(p)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        rel = str(p.parent.relative_to(RESULTS))
        rows.append({
            "dir": rel,
            "experiment": data.get("experiment") or rel.split("/")[-1],
            "spec": data.get("spec"),
            "seed": data.get("seed"),
            "config_hash": data.get("config_hash", "unknown"),
            "git_commit": data.get("git_commit", "unknown"),
            "headline": _headline(data),
        })
    return rows


# ── baseline ladder (mirrors apps/web/lib/leaderboard-data.ts) ──────────────

LADDER_MODELS = [
    ("Qwen_Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen_Qwen2.5-Coder-1.5B-Instruct", "Qwen2.5-Coder-1.5B-Instruct", 1.5),
    ("Qwen_Qwen2.5-Math-1.5B-Instruct", "Qwen2.5-Math-1.5B-Instruct", 1.5),
    ("Qwen_Qwen3-1.7B", "Qwen3-1.7B", 1.7),
]


def _latest_results(d: Path) -> dict[str, Any] | None:
    files = sorted(d.rglob("results_*.json"))
    return _load(files[-1]) if files else None


def build_leaderboard() -> list[dict[str, Any]]:
    rows = []
    for d, name, size in LADDER_MODELS:
        data = _latest_results(RESULTS / "scale" / "ladder" / d)
        if not data:
            continue
        r = data.get("results", {})
        mmlu = [v["acc,none"] for k, v in r.items() if k.startswith("mmlu_") and "acc,none" in v]
        gsm = None
        fixed = _latest_results(RESULTS / "scale" / "gsm8k_fixed" / d)
        if fixed and "gsm8k" in fixed.get("results", {}):
            gsm = fixed["results"]["gsm8k"].get("exact_match,flexible-extract")
        elif "gsm8k" in r:
            gsm = r["gsm8k"].get("exact_match,flexible-extract")
        mmlu_acc = statistics.mean(mmlu) if mmlu else None
        cfg = data.get("config", {})
        sha = str(cfg.get("sha") or data.get("git_hash") or "unknown")[:8]
        rows.append({
            "model_name": name, "size_b": size, "source": "Alibaba Qwen",
            "mmlu_acc": _round(mmlu_acc) if mmlu_acc is not None else None,
            "arc_challenge_acc": _round(r.get("arc_challenge", {}).get("acc,none")),
            "hellaswag_acc": _round(r.get("hellaswag", {}).get("acc,none")),
            "gsm8k_flexible": _round(gsm) if gsm is not None else None,
            "mixed_arena": _round((mmlu_acc + gsm) / 2) if mmlu_acc is not None and gsm is not None else None,
            "config_sha": sha, "git_commit": sha, "machine": "GB10",
        })
    return rows


# ── the twin's public ladder (exp40) ────────────────────────────────────────

LADDER_TASKS = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "sciq", "lambada_openai")
PRETTY = {
    "milestone_explicit_500M": ("Explicit twin 913M", "0.5B"),
    "milestone_explicit_1B": ("Explicit twin 913M", "1B"),
    "milestone_explicit_2p5B": ("Explicit twin 913M", "2.5B"),
    "milestone_tied_500M": ("Tied EqLM 158M resident", "0.5B"),
    "milestone_tied_1B": ("Tied EqLM 158M resident", "1B"),
    "milestone_tied_2p5B": ("Tied EqLM 158M resident", "2.5B"),
    "rung_EleutherAI_pythia-410m": ("Pythia-410m", "300B"),
    "rung_EleutherAI_pythia-1b": ("Pythia-1b", "300B"),
    "rung_HuggingFaceTB_SmolLM2-360M": ("SmolLM2-360M", "4T"),
    "rung_TinyLlama_TinyLlama_v1.1": ("TinyLlama-1.1B", "3T"),
}


def build_ladder_exp40() -> dict[str, Any]:
    rows = []
    for stem, (name, tokens) in PRETTY.items():
        p = RESULTS / "scale" / "exp40" / f"{stem}.json"
        if not p.exists():
            continue
        tasks = _load(p)["tasks"]
        accs = {t: _round(tasks[t]["acc"], 3) for t in LADDER_TASKS if isinstance(tasks.get(t), dict) and "acc" in tasks[t]}
        six = [accs[t] for t in LADDER_TASKS[:6] if t in accs]
        rows.append({"model": name, "tokens": tokens, "ours": stem.startswith("milestone"),
                     "mean6": _round(sum(six) / len(six), 3) if six else None, **accs})
    return {"finding": "F55", "chance_note": "Six multiple-choice tasks; chance is about 0.33 on the mean. "
            "Both twin arms are at chance at 2.5B tokens.", "rows": rows}


# ── council record (F41 confirmation seeds, F54 fair baselines) ─────────────

def build_council() -> dict[str, Any]:
    out: dict[str, Any] = {"findings": ["F41", "F54"], "seeds": [], "fair_baselines": None,
                           "note": "Per-question winners and rule accuracies from the pre-registered "
                                   "confirmation arena (SPEC 0017); no per-token influence traces were "
                                   "recorded, so the equilibrium view replays answer-level outcomes."}
    conf = RESULTS / "scale" / "exp23_confirm"
    for rp in sorted(conf.glob("results_seed*.json")):
        data = _load(rp)
        seed = data.get("seed") or int(re.findall(r"\d+", rp.stem)[-1])
        recs_p = conf / f"records_seed{seed}.json"
        sample = []
        if recs_p.exists():
            for i, rec in enumerate(_load(recs_p)[:40]):
                sample.append({
                    "i": i, "domain": rec.get("domain"),
                    "winners": {rule: (rec.get(rule) or {}).get("winner") for rule in ("equilibrium", "cross_exam", "leave_one_out", "self_preference")},
                    "correct": {rule: (rec.get(rule) or {}).get("correct") for rule in ("equilibrium", "cross_exam", "leave_one_out", "self_preference")},
                    "singles": rec.get("singles"),
                })
        out["seeds"].append({
            "seed": seed, "n_tasks": data.get("n_tasks"), "players": data.get("players"),
            "accuracy": {k: _round(v) for k, v in (data.get("accuracy") or {}).items()},
            "per_domain": data.get("per_domain"), "sample": sample,
        })
    fb = RESULTS / "scale" / "exp30_fair_baselines.json"
    if fb.exists():
        out["fair_baselines"] = _load(fb)
    anchored = RESULTS / "scale" / "exp27_confirmation.json"
    if anchored.exists():
        a = _load(anchored)
        out["anchored_confirmation"] = {k: a[k] for k in ("seeds", "router", "per_seed") if k in a}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs.json").write_text(json.dumps({"runs": build_runs()}, indent=1) + "\n")
    (OUT / "leaderboard.json").write_text(json.dumps(build_leaderboard(), indent=1) + "\n")
    (OUT / "ladder_exp40.json").write_text(json.dumps(build_ladder_exp40(), indent=1) + "\n")
    (OUT / "council.json").write_text(json.dumps(build_council(), indent=1) + "\n")
    site = REPO / "site" / "src" / "data" / "results.json"
    if site.exists():
        shutil.copy(site, OUT / "results.json")
    for f in ("runs.json", "leaderboard.json", "ladder_exp40.json", "council.json", "results.json"):
        print(f"{f}: {(OUT / f).stat().st_size} bytes")


if __name__ == "__main__":
    main()
