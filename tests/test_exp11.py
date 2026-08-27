"""Smoke tests for exp11 (H3: MPO vs DPO) — real CLI, tiny models, CPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM, save_checkpoint

REPO = Path(__file__).resolve().parent.parent

PAIRS = [
    {"sentence_good": "The cats sleep on the mat.", "sentence_bad": "The cats sleeps on the mat.", "UID": "u1"},
    {"sentence_good": "She reads a long book.", "sentence_bad": "She read a long books.", "UID": "u1"},
    {"sentence_good": "Dogs bark at strangers.", "sentence_bad": "Dogs barks at strangers.", "UID": "u2"},
    {"sentence_good": "He walks to the store.", "sentence_bad": "He walk to the store.", "UID": "u2"},
    {"sentence_good": "Birds fly over the lake.", "sentence_bad": "Birds flies over the lake.", "UID": "u3"},
    {"sentence_good": "They open the old door.", "sentence_bad": "They opens the old door.", "UID": "u3"},
]


@pytest.fixture(scope="module")
def exp11_setup(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("exp11")
    torch.manual_seed(0)

    explicit = ExplicitLM(
        config=EqLMConfig(vocab_size=50257, d_model=16, n_heads=2, d_ff=32, max_seq_len=64),
        n_layers=2,
    )
    save_checkpoint(explicit, root / "explicit.pt")
    eqlm = EqLM(
        config=EqLMConfig(
            vocab_size=50257, d_model=16, n_heads=2, d_ff=32, max_seq_len=64,
            deq_max_iter=3, map_form="postln",
        )
    )
    save_checkpoint(eqlm, root / "eqlm.pt")

    (root / "pairs.json").write_text(json.dumps(PAIRS))
    cfg = {
        "seed": 7,
        "device": "cpu",
        "bases": {
            "explicit": {"checkpoint": str(root / "explicit.pt")},
            "eqlm": {"checkpoint": str(root / "eqlm.pt")},
        },
        "pairs": {"json_file": str(root / "pairs.json"), "train_uids": ["u1", "u2"]},
        "training": {"beta": 0.1, "lr": 1e-3, "epochs": 2, "batch_size": 4, "max_seq_len": 64},
        # P1 is MagneticAdamW with tau=0 (= decoupled AdamW, same code path)
        # so the arms differ ONLY in the magnet strength.
        "arms": {
            "P1": {"optimizer": "magnetic", "tau": 0.0},
            "P2": {"optimizer": "magnetic", "tau": 50.0},
        },
    }
    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = root / "out"

    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp11_mpo_dpo.py"),
         "--config", str(cfg_path), "--output", str(out)],
        capture_output=True, text=True, timeout=600, cwd=str(REPO),
    )
    return {"proc": proc, "out": out, "cfg_path": cfg_path}


def test_cli_runs_green(exp11_setup) -> None:
    proc = exp11_setup["proc"]
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-3000:]}"


def test_results_structure(exp11_setup) -> None:
    results = json.loads((exp11_setup["out"] / "results.json").read_text())
    assert results["spec"] == "0007"
    assert set(results["runs"]) == {"explicit_P1", "explicit_P2", "eqlm_P1", "eqlm_P2"}
    for run in results["runs"].values():
        assert 0.0 <= run["after"]["heldout_acc"] <= 1.0
        assert run["after"]["kl_to_ref"] >= 0.0
        assert run["num_train_steps"] > 0
        assert run["final_loss"] is not None


def test_strong_magnet_reduces_kl_drift(exp11_setup) -> None:
    """Mechanics, not science: tau=50 must pin weights nearer the base than
    tau=0 (same optimizer code path), hence strictly smaller KL-to-ref."""
    results = json.loads((exp11_setup["out"] / "results.json").read_text())
    for base in ("explicit", "eqlm"):
        kl_dpo = results["runs"][f"{base}_P1"]["after"]["kl_to_ref"]
        kl_mpo = results["runs"][f"{base}_P2"]["after"]["kl_to_ref"]
        assert kl_mpo < kl_dpo, f"{base}: magnet KL {kl_mpo} !< DPO KL {kl_dpo}"


def test_resume_skips_completed_arms(exp11_setup) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp11_mpo_dpo.py"),
         "--config", str(exp11_setup["cfg_path"]), "--output", str(exp11_setup["out"])],
        capture_output=True, text=True, timeout=300, cwd=str(REPO),
    )
    assert proc.returncode == 0
    assert proc.stdout.count("[resume]") == 4
