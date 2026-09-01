"""The twin trainer's promises, tested at toy scale on CPU.

A multi-day run earns trust from properties provable in seconds: the anytime
depths reduce to the registered [6, 11, 16] at depth sixteen, the schedule is
warmup-stable-decay in token space, a resumed run reaches the same weights as
an uninterrupted one, milestones are recorded exactly once, and a checkpoint
refuses to resume onto a pack other than the one it was trained on.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp39_twin_1b import lr_at, main, supervise_depths  # noqa: E402

VOCAB = 211
SHARD_TOKENS = 60_000
HOLDOUT_TOKENS = 20_000


def make_pack(pack_dir: Path, seed: int = 7) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shards = []
    hashes = []
    for i in range(2):
        arr = rng.integers(0, VOCAB, size=SHARD_TOKENS, dtype=np.uint16)
        f = pack_dir / f"shard_{i:05d}.bin"
        arr.tofile(f)
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        hashes.append(digest)
        shards.append({"file": f.name, "sha256": digest, "tokens": SHARD_TOKENS})
    hold = rng.integers(0, VOCAB, size=HOLDOUT_TOKENS, dtype=np.uint16)
    hf = pack_dir / "holdout.bin"
    hold.tofile(hf)
    manifest = {
        "dataset": "synthetic", "dataset_config": "test", "tokenizer": "none",
        "vocab_size": VOCAB, "eos_id": 0, "dtype": "uint16",
        "shard_tokens": SHARD_TOKENS, "shards": shards,
        "holdout": {"file": "holdout.bin",
                    "sha256": hashlib.sha256(hf.read_bytes()).hexdigest(),
                    "tokens": HOLDOUT_TOKENS},
        "total_train_tokens": 2 * SHARD_TOKENS,
        "pack_hash": hashlib.sha256("".join(hashes).encode()).hexdigest(),
    }
    (pack_dir / "manifest.json").write_text(json.dumps(manifest))


def run_trainer(pack: Path, out: Path, arm: str, target: int,
                extra: list[str] | None = None) -> int:
    argv = [
        "exp39", "--arm", arm, "--pack-dir", str(pack), "--out-dir", str(out),
        "--target-tokens", str(target), "--seq-len", "16", "--d-model", "32",
        "--n-heads", "4", "--d-ff", "64", "--depth", "4",
        "--vocab-size", "256", "--micro-batch", "4", "--grad-accum", "2",
        "--warmup-tokens", "64", "--ckpt-tokens", "256",
        "--milestones", "256", "512", "--heldout-batches", "2",
        "--device", "cpu", "--log-every", "1",
    ] + (extra or [])
    old = sys.argv
    sys.argv = argv
    try:
        return main()
    finally:
        sys.argv = old


def test_supervise_depths_registered_values() -> None:
    assert supervise_depths(16) == [6, 11, 16]
    assert supervise_depths(4) == [2, 3, 4]


def test_lr_schedule_is_wsd() -> None:
    peak, floor, warm = 3e-4, 3e-5, 100
    assert lr_at(49, peak, floor, warm, None, None) == pytest.approx(peak / 2, rel=0.03)
    assert lr_at(500, peak, floor, warm, None, None) == peak
    assert lr_at(500, peak, floor, warm, 800, 1000) == peak
    assert lr_at(1000, peak, floor, warm, 800, 1000) == floor
    mid = lr_at(900, peak, floor, warm, 800, 1000)
    assert floor < mid < peak


@pytest.mark.parametrize("arm", ["explicit", "tied"])
def test_smoke_train_writes_artifacts(tmp_path: Path, arm: str) -> None:
    pack = tmp_path / "pack"
    make_pack(pack)
    out = tmp_path / f"out_{arm}"
    rc = run_trainer(pack, out, arm, target=768)
    assert rc == 0
    assert (out / "ckpt_latest.pt").exists()
    lines = (out / "milestones.jsonl").read_text().splitlines()
    recorded = [json.loads(ln)["milestone_tokens"] for ln in lines]
    assert recorded == [256, 512]
    log = [json.loads(ln) for ln in (out / "train_log.jsonl").read_text().splitlines()]
    assert all(np.isfinite(e["loss"]) for e in log)
    assert (out / "ckpt_256.pt").exists() and (out / "ckpt_512.pt").exists()


def test_resume_matches_uninterrupted(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    make_pack(pack)
    out_a = tmp_path / "a"
    assert run_trainer(pack, out_a, "tied", target=768) == 0
    out_b = tmp_path / "b"
    assert run_trainer(pack, out_b, "tied", target=384) == 0
    assert run_trainer(pack, out_b, "tied", target=768) == 0
    a = torch.load(out_a / "ckpt_latest.pt", weights_only=True)
    b = torch.load(out_b / "ckpt_latest.pt", weights_only=True)
    assert a["tokens_seen"] == b["tokens_seen"]
    assert a["cursor"] == b["cursor"]
    for k, va in a["state_dict"].items():
        assert torch.allclose(va, b["state_dict"][k], atol=1e-6), k


def test_checkpoint_refuses_foreign_pack(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    make_pack(pack)
    out = tmp_path / "out"
    assert run_trainer(pack, out, "tied", target=384) == 0
    pack2 = tmp_path / "pack2"
    make_pack(pack2, seed=8)
    with pytest.raises(SystemExit, match="refused"):
        run_trainer(pack2, out, "tied", target=768)


def test_preflight_reports_and_roundtrips(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    make_pack(pack)
    out = tmp_path / "pf"
    rc = run_trainer(pack, out, "explicit", target=10**9,
                     extra=["--preflight", "3", "--preflight-accum", "2"])
    assert rc == 0
    rep = json.loads((out / "preflight.json").read_text())
    assert rep["resume_roundtrip_exact"] is True
    assert rep["median_tok_s"] > 0


def test_intervention_flags(tmp_path: Path) -> None:
    """SPEC 0024: block-lr scaling trains the block group at a scaled rate,
    final-only supervision trains without the anytime triple, and the scale
    is refused for the explicit arm, which has no tied block."""
    pack = tmp_path / "pack"
    make_pack(pack)
    out = tmp_path / "i1"
    rc = run_trainer(pack, out, "tied", target=256,
                     extra=["--block-lr-scale", "0.25"])
    assert rc == 0
    out2 = tmp_path / "i2"
    rc = run_trainer(pack, out2, "tied", target=256,
                     extra=["--supervise-final-only"])
    assert rc == 0
    with pytest.raises(SystemExit, match="tied arm"):
        run_trainer(pack, tmp_path / "bad", "explicit", target=256,
                    extra=["--block-lr-scale", "0.25"])
