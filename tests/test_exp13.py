"""Smoke tests for exp13 (H6: contraction-at-width) — real CLI, tiny, CPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent

TINY_MC = {
    "vocab_size": 50257, "d_model": 16, "n_heads": 2, "d_ff": 32,
    "max_seq_len": 32, "deq_max_iter": 6, "deq_tol": 1e-3,
    "map_form": "postln", "dropout": 0.0,
}


@pytest.fixture(scope="module")
def exp13_setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("exp13")
    torch.manual_seed(0)
    tokens = torch.randint(0, 50257, (40, 16))
    torch.save({"tensor": tokens, "num_seqs": 40}, root / "cache.pt")

    cfg = {
        "seed": 7,
        "device": "cpu",
        "data": {"token_cache_file": str(root / "cache.pt"), "batch_size": 4},
        "training": {"num_steps": 4, "lr": 1e-3, "weight_decay": 0.0,
                     "grad_clip": 1.0, "log_every": 2},
        "eval": {"telemetry_batches": 2,
                 "blimp": {"num_phenomena": 2, "pairs_per_phenomenon": 5}},
        "control": {"note": "exp10 A3 seed42", "blimp_accuracy": 0.537},
        "arms": {
            "B1": {"kind": "anytime", "model_config": dict(TINY_MC),
                   "supervise_at": [2, 4, 6],
                   "supervise_weights": [0.15, 0.3, 1.0]},
            "B2": {"kind": "trajpen", "model_config": dict(TINY_MC),
                   "lambda_c": 0.1, "gamma": 0.9},
            "B3": {"kind": "core",
                   "model_config": {**TINY_MC, "d_core": 8, "n_heads_core": 2,
                                    "d_ff_core": 16, "n_enc": 1, "n_dec": 1}},
        },
    }
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = root / "out"
    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp13_contraction_at_width.py"),
         "--config", str(cfg_path), "--output", str(out)],
        capture_output=True, text=True, timeout=900, cwd=str(REPO),
    )
    return {"proc": proc, "out": out, "cfg_path": cfg_path}


def test_cli_runs_green(exp13_setup) -> None:
    proc = exp13_setup["proc"]
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-3000:]}"


def test_all_arms_present_with_metrics(exp13_setup) -> None:
    r = json.loads((exp13_setup["out"] / "results.json").read_text())
    assert r["spec"] == "0010"
    assert set(r["arms"]) == {"B1", "B2", "B3"}
    for arm in r["arms"].values():
        assert 0.0 <= arm["blimp_accuracy"] <= 1.0
        assert 0.0 <= arm["solver_convergence_rate"] <= 1.0
        assert arm["solver_mean_iterations"] > 0
        assert arm["final_loss"] > 0


def test_trajpen_records_lipschitz(exp13_setup) -> None:
    r = json.loads((exp13_setup["out"] / "results.json").read_text())
    tail = r["arms"]["B2"]["lipschitz_curve_tail"]
    assert tail and all(v > 0 for v in tail)


def test_core_arm_uses_core_model(exp13_setup) -> None:
    r = json.loads((exp13_setup["out"] / "results.json").read_text())
    assert r["arms"]["B3"]["kind"] == "core"


def test_resume_skips(exp13_setup) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp13_contraction_at_width.py"),
         "--config", str(exp13_setup["cfg_path"]), "--output", str(exp13_setup["out"])],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
    )
    assert proc.returncode == 0 and proc.stdout.count("[resume]") == 3
