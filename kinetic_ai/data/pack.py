"""Reader for tokenized data packs built by SPEC 0022.

Provides memory-mapped access to packed token shards with deterministic
windowing and holdout isolation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch


def window_order(n_windows: int, seed: int) -> torch.Tensor:
    """Generate deterministic random window ordering.

    Args:
        n_windows: Total number of windows available.
        seed: Random seed for reproducibility.

    Returns:
        torch.Tensor: Permutation of window indices.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n_windows, generator=g)


class PackReader:
    """Read tokenized data from a pack built by prepare_1b_pack.py.

    Memory-maps all shard files and provides windowed access with automatic
    shard boundary handling. Holdout data is kept separate and never mixes
    with training data.
    """

    def __init__(self, pack_dir: str | Path) -> None:
        """Load a pack and verify integrity.

        Args:
            pack_dir: Path to directory containing manifest.json and shards.

        Raises:
            SystemExit: If manifest is missing or any shard has size mismatch.
        """
        self.pack_dir = Path(pack_dir)
        manifest_path = self.pack_dir / "manifest.json"

        if not manifest_path.exists():
            print(f"ERROR: manifest.json not found in {self.pack_dir}", file=sys.stderr)
            raise SystemExit(1)

        manifest_text = manifest_path.read_text()
        self.manifest = cast(dict, json.loads(manifest_text))

        # Determine dtype from manifest (default uint16 for backward compatibility)
        dtype_str = self.manifest.get("dtype", "uint16")
        if dtype_str == "uint8":
            dtype = np.uint8
            bytes_per_token = 1
        elif dtype_str == "uint16":
            dtype = np.uint16
            bytes_per_token = 2
        else:
            print(f"ERROR: unknown dtype '{dtype_str}'", file=sys.stderr)
            raise SystemExit(1)

        # Load and verify all training shards
        self._shards: list[np.memmap] = []
        self._shard_offsets: list[int] = []
        offset = 0

        for shard_info in self.manifest["shards"]:
            shard_path = self.pack_dir / shard_info["file"]
            if not shard_path.exists():
                print(
                    f"ERROR: shard file missing: {shard_path}",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            file_size = shard_path.stat().st_size
            expected_tokens = shard_info["tokens"]
            expected_size = expected_tokens * bytes_per_token

            if file_size != expected_size:
                print(
                    f"ERROR: shard {shard_info['file']} size mismatch: "
                    f"expected {expected_size} bytes, got {file_size} bytes",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            # Memory-map in read-only mode
            data = np.memmap(shard_path, dtype=dtype, mode="r")
            self._shards.append(data)
            self._shard_offsets.append(offset)
            offset += len(data)

        self._total_train_tokens = offset

        # Load holdout
        holdout_info = self.manifest["holdout"]
        holdout_path = self.pack_dir / holdout_info["file"]
        if not holdout_path.exists():
            print(
                f"ERROR: holdout file missing: {holdout_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)

        file_size = holdout_path.stat().st_size
        expected_tokens = holdout_info["tokens"]
        expected_size = expected_tokens * bytes_per_token

        if file_size != expected_size:
            print(
                f"ERROR: holdout size mismatch: "
                f"expected {expected_size} bytes, got {file_size} bytes",
                file=sys.stderr,
            )
            raise SystemExit(1)

        # Handle empty holdout files (can happen in smoke tests)
        if file_size == 0:
            self._holdout = np.array([], dtype=dtype)
        else:
            self._holdout = np.memmap(holdout_path, dtype=dtype, mode="r")

    @property
    def total_train_tokens(self) -> int:
        """Total number of training tokens across all shards."""
        return self._total_train_tokens

    @property
    def pack_hash(self) -> str:
        """SHA256 hash of concatenated shard hashes."""
        return cast(str, self.manifest["pack_hash"])

    def n_windows(self, seq_len: int, batch: int) -> int:
        """Compute the number of non-overlapping windows available.

        Args:
            seq_len: Sequence length per sample.
            batch: Batch size.

        Returns:
            Number of complete windows.
        """
        span = seq_len * batch
        return self._total_train_tokens // span

    def window(
        self,
        idx: int,
        seq_len: int,
        batch: int,
    ) -> torch.Tensor:
        """Read a window of tokens from the pack.

        Handles spans that cross shard boundaries by reading from multiple
        shards as needed.

        Args:
            idx: Window index (0-based).
            seq_len: Sequence length per sample.
            batch: Batch size (number of samples per window).

        Returns:
            torch.LongTensor of shape [batch, seq_len] containing the window.
        """
        span = seq_len * batch
        start_idx = idx * span
        end_idx = start_idx + span

        # Read across shard boundaries if needed
        tokens: list[int] = []
        pos = start_idx

        while pos < end_idx:
            # Find which shard contains pos
            shard_idx = 0
            for i, offset in enumerate(self._shard_offsets):
                if offset <= pos < (offset + len(self._shards[i])):
                    shard_idx = i
                    break
                elif offset > pos:
                    shard_idx = i - 1
                    break

            if shard_idx < 0:
                shard_idx = 0

            shard = self._shards[shard_idx]
            shard_offset = self._shard_offsets[shard_idx]
            local_idx = pos - shard_offset

            # Read as much as we can from this shard
            remaining_in_shard = len(shard) - local_idx
            needed = end_idx - pos
            to_read = min(remaining_in_shard, needed)

            tokens.extend(shard[local_idx : local_idx + to_read].tolist())
            pos += to_read

        # Convert to tensor and reshape
        token_array = np.array(tokens[:span], dtype=np.int64)
        return torch.from_numpy(token_array).reshape(batch, seq_len).long()

    def holdout_batches(
        self,
        seq_len: int,
        batch: int,
        n: int,
    ) -> list[torch.Tensor]:
        """Read n non-overlapping windows from the holdout set.

        Args:
            seq_len: Sequence length per sample.
            batch: Batch size.
            n: Number of windows to read.

        Returns:
            List of n torch.LongTensor of shape [batch, seq_len].
        """
        span = seq_len * batch
        batches: list[torch.Tensor] = []

        for i in range(n):
            start = i * span
            end = start + span

            if end > len(self._holdout):
                break

            tokens = self._holdout[start:end]
            token_array = np.array(tokens, dtype=np.int64)
            batch_tensor = torch.from_numpy(token_array).reshape(batch, seq_len).long()
            batches.append(batch_tensor)

        return batches
