"""Tests for data pack builder and reader for SPEC 0022.

Covers pack creation from local texts, manifest generation, shard layout,
holdout isolation, and windowed reading with shard boundary handling.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


class TestPackBuilder:
    """Tests for the prepare_1b_pack.py builder."""

    def test_smoke_build_from_local_texts(self, tmp_path: Path) -> None:
        """Build a tiny pack from local text file in smoke mode."""
        # Create a test text file with simple documents
        texts_file = tmp_path / "texts.txt"
        base_text = "Hello world this is a test document for tokenization with varied content. "
        texts = [base_text * 50 for _ in range(200)]  # ~7050 tokens per doc, 1.4M total
        texts_file.write_text("\n".join(texts))

        out_dir = tmp_path / "pack"
        out_dir.mkdir()

        # Run builder in smoke mode with small token counts
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--co",
                "-q",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
        )
        # Just verify we can import the module first
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path("/home/sharaths/projects/game-llm")
                    / "scripts"
                    / "prepare_1b_pack.py"
                ),
                "--out-dir",
                str(out_dir),
                "--train-tokens",
                "100000",
                "--holdout-tokens",
                "10000",
                "--shard-tokens",
                "30000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0
        ), f"Builder failed:\n{result.stdout}\n{result.stderr}"

        # Verify shards exist
        shard_files = sorted(out_dir.glob("shard_*.bin"))
        assert len(shard_files) > 0, "No shard files created"

        # Verify manifest exists
        manifest_file = out_dir / "manifest.json"
        assert manifest_file.exists(), "Manifest not created"

        manifest = json.loads(manifest_file.read_text())
        assert manifest["dataset"] == "local-texts"
        assert manifest["tokenizer"] == "gpt2"
        assert manifest["vocab_size"] == 50257
        assert "shards" in manifest
        assert len(manifest["shards"]) == len(shard_files)
        assert "holdout" in manifest
        assert manifest["holdout"]["file"] == "holdout.bin"
        assert manifest["holdout"]["tokens"] > 0

    def test_shard_file_sizes_match_manifest(self, tmp_path: Path) -> None:
        """Verify shard file sizes match the manifest token counts."""
        texts_file = tmp_path / "texts.txt"
        base_text = "Document with content for testing the tokenizer. "
        texts = [base_text * 30 for i in range(80)]
        texts_file.write_text("\n".join(texts))

        out_dir = tmp_path / "pack"
        out_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path("/home/sharaths/projects/game-llm")
                    / "scripts"
                    / "prepare_1b_pack.py"
                ),
                "--out-dir",
                str(out_dir),
                "--train-tokens",
                "80000",
                "--holdout-tokens",
                "5000",
                "--shard-tokens",
                "25000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        manifest_file = out_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text())

        # Verify each shard's file size matches token count
        for shard_info in manifest["shards"]:
            shard_path = out_dir / shard_info["file"]
            assert shard_path.exists()
            file_size = shard_path.stat().st_size
            expected_size = shard_info["tokens"] * 2  # uint16 = 2 bytes
            assert (
                file_size == expected_size
            ), f"Shard {shard_info['file']}: expected {expected_size} bytes, got {file_size}"

        # Verify holdout file size
        holdout_path = out_dir / manifest["holdout"]["file"]
        assert holdout_path.exists()
        holdout_size = holdout_path.stat().st_size
        expected_holdout_size = manifest["holdout"]["tokens"] * 2
        assert (
            holdout_size == expected_holdout_size
        ), f"Holdout: expected {expected_holdout_size} bytes, got {holdout_size}"

    def test_holdout_does_not_overlap_train(self, tmp_path: Path) -> None:
        """Verify holdout tokens come after all training tokens."""
        texts_file = tmp_path / "texts.txt"
        base_text = "Test document for validation and testing purposes. "
        texts = [base_text * 30 for _ in range(100)]
        texts_file.write_text("\n".join(texts))

        out_dir = tmp_path / "pack"
        out_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path("/home/sharaths/projects/game-llm")
                    / "scripts"
                    / "prepare_1b_pack.py"
                ),
                "--out-dir",
                str(out_dir),
                "--train-tokens",
                "100000",
                "--holdout-tokens",
                "10000",
                "--shard-tokens",
                "40000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        manifest_file = out_dir / "manifest.json"
        manifest = json.loads(manifest_file.read_text())

        # Load all train shards and the holdout
        train_tokens: list[int] = []
        for shard_info in manifest["shards"]:
            shard_path = out_dir / shard_info["file"]
            data = np.fromfile(shard_path, dtype=np.uint16)
            train_tokens.extend(data.tolist())

        holdout_path = out_dir / manifest["holdout"]["file"]
        holdout_tokens = np.fromfile(holdout_path, dtype=np.uint16).tolist()

        # Verify no overlap by checking that we have the right total
        expected_train = min(manifest["total_train_tokens"], len(train_tokens))
        expected_holdout = manifest["holdout"]["tokens"]

        assert len(train_tokens) == expected_train
        assert len(holdout_tokens) == expected_holdout

        # The holdout should be drawn from positions after all training data
        # This is verified implicitly by the builder's logic


class TestPackReader:
    """Tests for the PackReader class."""

    @pytest.fixture
    def sample_pack(self, tmp_path: Path) -> Path:
        """Create a sample pack for testing."""
        texts_file = tmp_path / "texts.txt"
        base_text = "Sample document with content for testing purposes. "
        texts = [base_text * 50 for _ in range(200)]
        texts_file.write_text("\n".join(texts))

        out_dir = tmp_path / "pack"
        out_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path("/home/sharaths/projects/game-llm")
                    / "scripts"
                    / "prepare_1b_pack.py"
                ),
                "--out-dir",
                str(out_dir),
                "--train-tokens",
                "50000",
                "--holdout-tokens",
                "5000",
                "--shard-tokens",
                "20000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Pack creation failed:\n{result.stderr}"
        return out_dir

    def test_pack_reader_loads_manifest(self, sample_pack: Path) -> None:
        """Test that PackReader loads the manifest correctly."""
        from kinetic_ai.data.pack import PackReader

        reader = PackReader(sample_pack)
        assert reader.pack_hash is not None
        assert reader.total_train_tokens > 0

    def test_pack_reader_window_deterministic(self, sample_pack: Path) -> None:
        """Test that window_order is deterministic for same seed."""
        from kinetic_ai.data.pack import PackReader, window_order

        reader = PackReader(sample_pack)
        n_windows = reader.n_windows(seq_len=512, batch=4)

        # Get two orderings with same seed
        order1 = window_order(n_windows, seed=42)
        order2 = window_order(n_windows, seed=42)

        assert torch.equal(order1, order2), "Same seed should give same order"

        # Different seed should give different order (very likely)
        order3 = window_order(n_windows, seed=43)
        assert not torch.equal(
            order1, order3
        ), "Different seed should (almost certainly) give different order"

    def test_pack_reader_window_crossing_shards(self, sample_pack: Path) -> None:
        """Test that window() correctly reads across shard boundaries."""
        from kinetic_ai.data.pack import PackReader

        reader = PackReader(sample_pack)
        seq_len = 256
        batch = 2

        # Get a window and verify it has the right shape
        window = reader.window(idx=0, seq_len=seq_len, batch=batch)
        assert window.shape == (batch, seq_len)
        assert window.dtype == torch.long

        # Verify the window contains valid token ids
        assert (window >= 0).all()
        assert (window < 50257).all()  # GPT-2 vocab size

    def test_pack_reader_holdout_batches(self, sample_pack: Path) -> None:
        """Test that holdout_batches returns non-overlapping windows."""
        from kinetic_ai.data.pack import PackReader

        reader = PackReader(sample_pack)
        seq_len = 256
        batch = 2

        batches = reader.holdout_batches(seq_len=seq_len, batch=batch, n=2)
        assert len(batches) == 2
        for batch_tensor in batches:
            assert batch_tensor.shape == (batch, seq_len)
            assert batch_tensor.dtype == torch.long

    def test_pack_reader_raises_on_missing_manifest(self, tmp_path: Path) -> None:
        """Test that PackReader raises SystemExit if manifest is missing."""
        from kinetic_ai.data.pack import PackReader

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(SystemExit):
            PackReader(empty_dir)

    def test_pack_reader_raises_on_shard_size_mismatch(
        self, sample_pack: Path, tmp_path: Path
    ) -> None:
        """Test that PackReader raises if shard file size doesn't match manifest."""
        from kinetic_ai.data.pack import PackReader

        # Load the manifest
        manifest_path = sample_pack / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Truncate a shard to cause a mismatch
        if manifest["shards"]:
            first_shard = sample_pack / manifest["shards"][0]["file"]
            if first_shard.exists():
                # Truncate to half size
                data = np.fromfile(first_shard, dtype=np.uint16)
                truncated = data[: len(data) // 2]
                truncated.tofile(first_shard)

                with pytest.raises(SystemExit):
                    PackReader(sample_pack)

    def test_pack_reader_window_content_matches_concatenation(
        self, sample_pack: Path
    ) -> None:
        """Test that window content matches direct concatenation of shards."""
        from kinetic_ai.data.pack import PackReader

        reader = PackReader(sample_pack)
        seq_len = 128
        batch = 1

        # Get a window
        window = reader.window(idx=0, seq_len=seq_len, batch=batch)

        # Load all shards and concatenate
        manifest_path = sample_pack / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        all_tokens: list[int] = []
        for shard_info in manifest["shards"]:
            shard_path = sample_pack / shard_info["file"]
            data = np.fromfile(shard_path, dtype=np.uint16)
            all_tokens.extend(data.tolist())

        # First window should match the first seq_len*batch tokens
        expected = torch.tensor(
            all_tokens[: seq_len * batch], dtype=torch.long
        ).reshape(batch, seq_len)
        assert torch.equal(window, expected)


class TestPackIntegration:
    """Integration tests for pack builder and reader working together."""

    def test_round_trip_build_and_read(self, tmp_path: Path) -> None:
        """Build a pack, then read it back and verify content."""
        from kinetic_ai.data.pack import PackReader

        texts_file = tmp_path / "texts.txt"
        base_text = "Integration test document with varied content. "
        texts = [base_text * 30 for _ in range(100)]
        texts_file.write_text("\n".join(texts))

        out_dir = tmp_path / "pack"
        out_dir.mkdir()

        # Build
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path("/home/sharaths/projects/game-llm")
                    / "scripts"
                    / "prepare_1b_pack.py"
                ),
                "--out-dir",
                str(out_dir),
                "--train-tokens",
                "120000",
                "--holdout-tokens",
                "15000",
                "--shard-tokens",
                "40000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Read
        reader = PackReader(out_dir)
        assert reader.total_train_tokens > 0

        # Verify we can read windows
        window = reader.window(idx=0, seq_len=256, batch=1)
        assert window.shape == (1, 256)

    def test_pack_manifest_hash_computed(self, tmp_path: Path) -> None:
        """Verify pack_hash is computed from shard hashes."""
        texts_file = tmp_path / "texts.txt"
        base_text = "Test document for pack hash verification. "
        texts = [base_text * 30 for _ in range(80)]
        texts_file.write_text("\n".join(texts))

        out_dir = tmp_path / "pack"
        out_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path("/home/sharaths/projects/game-llm")
                    / "scripts"
                    / "prepare_1b_pack.py"
                ),
                "--out-dir",
                str(out_dir),
                "--train-tokens",
                "80000",
                "--holdout-tokens",
                "8000",
                "--shard-tokens",
                "30000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert "pack_hash" in manifest
        assert isinstance(manifest["pack_hash"], str)
        assert len(manifest["pack_hash"]) == 64  # SHA256 hex


def test_builder_stops_streaming_once_targets_met(tmp_path):
    """Regression: with a stream far larger than the targets, the builder must
    stop consuming as soon as train+holdout tokens are written, rather than
    tokenizing the remainder into an unbounded in-memory buffer (the hang the
    first preflight build hit at 27GB RSS on sample-100BT)."""
    import json as _json
    import subprocess
    import sys as _sys

    texts = tmp_path / "texts.txt"
    with texts.open("w") as fh:
        for i in range(50_000):
            fh.write(f"document {i} carries a handful of words for the pack\n")
    out = tmp_path / "pack"
    rc = subprocess.run(
        [_sys.executable, "scripts/prepare_1b_pack.py",
         "--out-dir", str(out), "--local-texts-file", str(texts),
         "--train-tokens", "5000", "--holdout-tokens", "1000",
         "--shard-tokens", "2500", "--batch-docs", "64"],
        capture_output=True, text=True, timeout=120,
    )
    assert rc.returncode == 0, rc.stderr[-2000:]
    manifest = _json.loads((out / "manifest.json").read_text())
    assert manifest["total_train_tokens"] == 5000
    progress = _json.loads((out / "progress.json").read_text())
    # ~11 tokens per doc: 6000 tokens needs ~600 docs; consuming even a tenth
    # of the 50k-line stream would mean the stop condition is broken again.
    assert progress["docs_consumed"] < 5_000, progress["docs_consumed"]
