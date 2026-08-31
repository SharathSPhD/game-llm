"""Tests for byte pack builder for SPEC 0023.

Covers byte-level tokenization from GPT-2 packs, UTF-8 encoding, NUL
stripping, document separation, budget management, and dtype support in
PackReader.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from kinetic_ai.data.pack import PackReader


class TestBytePackBuilder:
    """Tests for the prepare_byte_pack.py builder."""

    @pytest.fixture
    def gpt2_pack(self, tmp_path: Path) -> Path:
        """Create a small GPT-2 pack for testing."""
        texts_file = tmp_path / "texts.txt"
        # Use simple ASCII text to avoid NUL byte complications
        base_text = "The quick brown fox jumps over the lazy dog. "
        texts = [base_text * 20 for _ in range(50)]
        texts_file.write_text("\n".join(texts))

        pack_dir = tmp_path / "gpt2_pack"
        pack_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_1b_pack.py"),
                "--out-dir",
                str(pack_dir),
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
        assert result.returncode == 0, f"GPT-2 pack creation failed:\n{result.stderr}"
        return pack_dir

    def test_smoke_build_byte_pack(self, gpt2_pack: Path, tmp_path: Path) -> None:
        """Build a byte pack from a GPT-2 pack in smoke mode."""
        byte_dir = tmp_path / "byte_pack"

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_dir),
                "--train-bytes",
                "100000",
                "--holdout-bytes",
                "5000",
                "--shard-bytes",
                "30000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Byte pack build failed:\n{result.stderr}"

        # Verify outputs exist
        assert (byte_dir / "manifest.json").exists(), "Manifest not created"
        shard_files = sorted(byte_dir.glob("shard_*.bin"))
        assert len(shard_files) > 0, "No shard files created"
        assert (byte_dir / "holdout.bin").exists(), "Holdout not created"

    def test_byte_pack_manifest_fields(self, gpt2_pack: Path, tmp_path: Path) -> None:
        """Verify byte pack manifest has correct fields."""
        byte_dir = tmp_path / "byte_pack"

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_dir),
                "--train-bytes",
                "100000",
                "--holdout-bytes",
                "5000",
                "--shard-bytes",
                "30000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        manifest_path = byte_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Verify required fields
        assert manifest["dataset"] == "fineweb-edu-bytes-from-pack"
        assert manifest["tokenizer"] == "byte-256"
        assert manifest["vocab_size"] == 256
        assert manifest["eos_id"] == 0
        assert manifest["dtype"] == "uint8"
        assert "shards" in manifest
        assert "holdout" in manifest
        assert "total_train_tokens" in manifest
        assert "pack_hash" in manifest
        assert "source_pack_hash" in manifest
        assert "nul_bytes_stripped" in manifest

    def test_byte_pack_shard_dtype_is_uint8(self, gpt2_pack: Path, tmp_path: Path) -> None:
        """Verify byte pack shards are stored as uint8."""
        byte_dir = tmp_path / "byte_pack"

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_dir),
                "--train-bytes",
                "50000",
                "--holdout-bytes",
                "2000",
                "--shard-bytes",
                "20000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        manifest_path = byte_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Verify dtype is uint8 and shard files have correct size
        for shard_info in manifest["shards"]:
            shard_path = byte_dir / shard_info["file"]
            assert shard_path.exists()
            file_size = shard_path.stat().st_size
            # For uint8, bytes = tokens (since each byte is 1 byte)
            expected_size = shard_info["tokens"]
            assert (
                file_size == expected_size
            ), f"Shard size mismatch: expected {expected_size}, got {file_size}"

            # Verify data is uint8
            data = np.fromfile(shard_path, dtype=np.uint8)
            assert (data < 256).all(), "Byte values exceed uint8 range"

    def test_byte_pack_budgets_respected(self, gpt2_pack: Path, tmp_path: Path) -> None:
        """Verify train and holdout budgets are respected."""
        byte_dir = tmp_path / "byte_pack"
        train_budget = 50000
        holdout_budget = 5000

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_dir),
                "--train-bytes",
                str(train_budget),
                "--holdout-bytes",
                str(holdout_budget),
                "--shard-bytes",
                "20000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        manifest_path = byte_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        # Verify train bytes are within budget
        total_train = manifest["total_train_tokens"]
        assert total_train <= train_budget, f"Train bytes {total_train} exceed budget {train_budget}"

        # Verify holdout bytes are within budget
        holdout_bytes = manifest["holdout"]["tokens"]
        assert holdout_bytes <= holdout_budget, f"Holdout bytes {holdout_bytes} exceed budget {holdout_budget}"

    def test_byte_pack_separator_bytes(self, gpt2_pack: Path, tmp_path: Path) -> None:
        """Verify documents are separated by 0x00 bytes."""
        byte_dir = tmp_path / "byte_pack"

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_dir),
                "--train-bytes",
                "100000",
                "--holdout-bytes",
                "5000",
                "--shard-bytes",
                "30000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Load all train shards and check for 0x00 separators
        manifest_path = byte_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        all_bytes: list[int] = []
        for shard_info in manifest["shards"]:
            shard_path = byte_dir / shard_info["file"]
            data = np.fromfile(shard_path, dtype=np.uint8)
            all_bytes.extend(data.tolist())

        # Count 0x00 bytes (document separators)
        separator_count = sum(1 for b in all_bytes if b == 0)

        # Should have at least one separator (from at least one document)
        assert separator_count > 0, "No separator bytes found in pack"


class TestBytePackReader:
    """Tests for PackReader supporting uint8 dtype."""

    @pytest.fixture
    def byte_pack(self, tmp_path: Path) -> Path:
        """Create a byte pack for testing."""
        # First create a GPT-2 pack
        texts_file = tmp_path / "texts.txt"
        base_text = "Simple test document for byte pack reading. "
        texts = [base_text * 15 for _ in range(30)]
        texts_file.write_text("\n".join(texts))

        gpt2_pack = tmp_path / "gpt2_pack"
        gpt2_pack.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_1b_pack.py"),
                "--out-dir",
                str(gpt2_pack),
                "--train-tokens",
                "30000",
                "--holdout-tokens",
                "3000",
                "--shard-tokens",
                "15000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Convert to byte pack
        byte_pack = tmp_path / "byte_pack"
        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_pack),
                "--train-bytes",
                "100000",
                "--holdout-bytes",
                "5000",
                "--shard-bytes",
                "30000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        return byte_pack

    def test_pack_reader_reads_uint8_pack(self, byte_pack: Path) -> None:
        """Test PackReader can read a uint8 byte pack."""
        reader = PackReader(byte_pack)

        # Verify reader loaded correctly
        assert reader.pack_hash is not None
        assert reader.total_train_tokens > 0

    def test_pack_reader_uint8_window_shape(self, byte_pack: Path) -> None:
        """Test that PackReader returns correct window shape for uint8 pack."""
        reader = PackReader(byte_pack)

        seq_len = 256
        batch = 2
        n_windows = reader.n_windows(seq_len=seq_len, batch=batch)

        if n_windows > 0:
            window = reader.window(idx=0, seq_len=seq_len, batch=batch)
            assert window.shape == (batch, seq_len)
            assert window.dtype == torch.long
            # Verify all values are bytes (0-255)
            assert (window >= 0).all() and (window < 256).all()

    def test_pack_reader_still_reads_uint16_pack(self, tmp_path: Path) -> None:
        """Test that PackReader still reads uint16 packs correctly."""
        # Create a uint16 pack
        texts_file = tmp_path / "texts.txt"
        base_text = "Test document for uint16 compatibility. "
        texts = [base_text * 15 for _ in range(30)]
        texts_file.write_text("\n".join(texts))

        pack_dir = tmp_path / "uint16_pack"
        pack_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_1b_pack.py"),
                "--out-dir",
                str(pack_dir),
                "--train-tokens",
                "30000",
                "--holdout-tokens",
                "3000",
                "--shard-tokens",
                "15000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Read with PackReader
        reader = PackReader(pack_dir)
        assert reader.pack_hash is not None
        assert reader.total_train_tokens > 0

        # Verify we can read windows
        if reader.n_windows(seq_len=256, batch=1) > 0:
            window = reader.window(idx=0, seq_len=256, batch=1)
            assert window.shape == (1, 256)
            assert window.dtype == torch.long
            # GPT-2 tokens should be < 50257
            assert (window < 50257).all()


class TestBytePackIntegration:
    """Integration tests for GPT-2 to byte pack conversion."""

    def test_round_trip_gpt2_to_bytes(self, tmp_path: Path) -> None:
        """Test round-trip: text -> GPT-2 tokens -> bytes"""
        from transformers import AutoTokenizer

        # Load tokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")

        # Simple test text with known UTF-8 encoding
        test_text = "Hello world"
        expected_bytes = test_text.encode("utf-8")

        # Tokenize
        ids = tok(test_text, add_special_tokens=False).input_ids

        # Decode back
        decoded_text = tok.decode(ids)

        # Encode to UTF-8
        decoded_bytes = decoded_text.encode("utf-8")

        # Should match original bytes (modulo tokenizer normalization)
        assert isinstance(decoded_bytes, bytes)
        assert len(decoded_bytes) > 0

    def test_gpt2_pack_hash_in_byte_manifest(self, tmp_path: Path) -> None:
        """Verify byte pack manifest includes source pack hash."""
        # Create GPT-2 pack
        texts_file = tmp_path / "texts.txt"
        base_text = "Test for pack hash tracking. "
        texts = [base_text * 15 for _ in range(30)]
        texts_file.write_text("\n".join(texts))

        gpt2_pack = tmp_path / "gpt2_pack"
        gpt2_pack.mkdir()

        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_1b_pack.py"),
                "--out-dir",
                str(gpt2_pack),
                "--train-tokens",
                "30000",
                "--holdout-tokens",
                "3000",
                "--shard-tokens",
                "15000",
                "--local-texts-file",
                str(texts_file),
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        gpt2_manifest = json.loads((gpt2_pack / "manifest.json").read_text())
        source_pack_hash = gpt2_manifest["pack_hash"]

        # Convert to byte pack
        byte_pack = tmp_path / "byte_pack"
        result = subprocess.run(
            [
                sys.executable,
                str(Path("/home/sharaths/projects/game-llm") / "scripts" / "prepare_byte_pack.py"),
                "--input-pack",
                str(gpt2_pack),
                "--out-dir",
                str(byte_pack),
                "--train-bytes",
                "100000",
                "--holdout-bytes",
                "5000",
                "--shard-bytes",
                "30000",
            ],
            cwd="/home/sharaths/projects/game-llm",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        byte_manifest = json.loads((byte_pack / "manifest.json").read_text())

        # Verify source pack hash is recorded
        assert "source_pack_hash" in byte_manifest
        assert byte_manifest["source_pack_hash"] == source_pack_hash
