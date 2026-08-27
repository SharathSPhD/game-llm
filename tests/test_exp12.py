"""Smoke tests for exp12 (H4: auction decoding) — real CLI, tiny models, CPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def exp12_setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("exp12")
    # Two synthetic "domains" with distinct vocabulary distributions.
    dom_a = "\n".join("the cat sat on the mat and the dog ran away" for _ in range(300))
    dom_b = "\n".join("quantum systems evolve under unitary operators" for _ in range(300))
    (root / "dom_a.txt").write_text(dom_a)
    (root / "dom_b.txt").write_text(dom_b)

    cfg = {
        "seed": 5,
        "device": "cpu",
        "domains": {
            "a": {"name": "toy_speech", "file": str(root / "dom_a.txt")},
            "b": {"name": "toy_wiki", "file": str(root / "dom_b.txt")},
        },
        "data": {"heldout_frac": 0.2, "max_eval_windows": 4},
        "model": {
            "vocab_size": 50257, "d_model": 16, "n_heads": 2, "d_ff": 32,
            "max_seq_len": 32, "n_layers": 2,
        },
        "training": {"steps": 30, "batch_size": 4, "lr": 1e-3},
        "eval": {"trace_positions": 40},
        "save_checkpoints": False,
    }
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = root / "out"

    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp12_auction_decoding.py"),
         "--config", str(cfg_path), "--output", str(out)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO),
    )
    return {"proc": proc, "out": out, "cfg_path": cfg_path}


def test_cli_runs_green(exp12_setup) -> None:
    proc = exp12_setup["proc"]
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-3000:]}"


def test_results_structure(exp12_setup) -> None:
    results = json.loads((exp12_setup["out"] / "results_seed5.json").read_text())
    assert results["spec"] == "0008"
    ppls = results["eval"]["perplexity_mixed"]
    assert set(ppls) == {"S_A", "S_B", "ENS", "AUC"}
    for v in ppls.values():
        assert 0 < v < 1e9
    assert results["h4_score"] in {"MET", "PARTIAL", "MISSED"}
    assert 0.0 <= results["eval"]["auction_win_frac_a"] <= 1.0


def test_vectorized_auction_matches_validated_mechanism(exp12_setup) -> None:
    """The closed-form auction path must agree with TokenAuction (F6)."""
    results = json.loads((exp12_setup["out"] / "results_seed5.json").read_text())
    assert results["eval"]["trace_mechanism_mismatches"] == 0


def test_traces_written_for_playground(exp12_setup) -> None:
    traces = json.loads((exp12_setup["out"] / "traces_seed5.json").read_text())
    assert len(traces) > 0
    t = traces[0]
    assert set(t) == {"position", "bids", "winner", "payment", "target_token"}
    assert t["winner"] in (0, 1)
    # second-price: payment equals the losing bid
    assert t["payment"] == pytest.approx(t["bids"][1 - t["winner"]], abs=1e-5)


def test_resume_skips_completed_seed(exp12_setup) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp12_auction_decoding.py"),
         "--config", str(exp12_setup["cfg_path"]), "--output", str(exp12_setup["out"])],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    assert proc.returncode == 0
    assert "[resume]" in proc.stdout
