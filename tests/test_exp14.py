"""Smoke tests for exp14 (H5: autoregressive auction) — real CLI, tiny, CPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.eqlm import EqLMConfig, ExplicitLM, save_checkpoint

REPO = Path(__file__).resolve().parent.parent


def _tiny(seed: int, layers: int = 2) -> ExplicitLM:
    torch.manual_seed(seed)
    return ExplicitLM(
        config=EqLMConfig(vocab_size=50257, d_model=16, n_heads=2, d_ff=32, max_seq_len=64),
        n_layers=layers,
    )


@pytest.fixture(scope="module")
def exp14_setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("exp14")
    save_checkpoint(_tiny(0), root / "spec_a.pt")
    save_checkpoint(_tiny(1), root / "spec_b.pt")
    save_checkpoint(_tiny(2, layers=3), root / "judge.pt")

    (root / "dom_a.txt").write_text("\n".join(f"the cat sat on mat number {i}." for i in range(120)))
    (root / "dom_b.txt").write_text("\n".join(f"wikipedia article section {i} text." for i in range(120)))

    cfg = {
        "seed": 7,
        "device": "cpu",
        "specialists": {"a": str(root / "spec_a.pt"), "b": str(root / "spec_b.pt")},
        "judge": str(root / "judge.pt"),
        "domains": {"a": {"file": str(root / "dom_a.txt")}, "b": {"file": str(root / "dom_b.txt")}},
        "data": {"heldout_frac": 0.2},
        "generation": {"prefix_len": 8, "gen_len": 6, "prompts_per_domain": 3},
    }
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = root / "out"
    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp14_autoregressive_auction.py"),
         "--config", str(cfg_path), "--output", str(out)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO),
    )
    return {"proc": proc, "out": out, "cfg_path": cfg_path}


def test_cli_runs_green(exp14_setup) -> None:
    proc = exp14_setup["proc"]
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-3000:]}"


def test_results_structure(exp14_setup) -> None:
    r = json.loads((exp14_setup["out"] / "results_seed7.json").read_text())
    assert r["spec"] == "0009"
    assert set(r["systems"]) == {"S_A", "S_B", "ENS", "AUC"}
    for s in r["systems"].values():
        assert s["judge_nll_per_token"] > 0
        assert s["judge_nll_domain_a"] > 0 and s["judge_nll_domain_b"] > 0
        assert 0.0 <= s["repetition_3gram"] <= 1.0
    assert r["h5_score"] in ("MET", "MISSED")


def test_auction_traces_second_price(exp14_setup) -> None:
    traces = json.loads((exp14_setup["out"] / "gen_traces_seed7.json").read_text())
    assert traces, "auction must emit traces"
    for t in traces:
        assert t["payment"] == pytest.approx(min(t["bids"]), abs=1e-6)
        assert t["winner"] == (0 if t["bids"][0] >= t["bids"][1] else 1)


def test_resume_skips(exp14_setup) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp14_autoregressive_auction.py"),
         "--config", str(exp14_setup["cfg_path"]), "--output", str(exp14_setup["out"])],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )
    assert proc.returncode == 0 and "[resume]" in proc.stdout
