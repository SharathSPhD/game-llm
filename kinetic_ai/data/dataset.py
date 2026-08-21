"""Dataset loading and token stream building for BabyLM pretraining.

Loads BabyLM-2026-Strict-Small dataset from HuggingFace cache
and builds token streams for model training.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import torch
from datasets import Dataset, load_dataset


def load_babylm_dataset(
    subset: str = "BabyLM-2026-Strict-Small",
    split: str = "train",
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Load BabyLM dataset from HuggingFace cache (offline).

    Args:
        subset: Dataset name (default: BabyLM-2026-Strict-Small).
        split: Which split to load (default: "train").
        cache_dir: Cache directory (default: ~/.cache/huggingface/hub).
        max_samples: Limit number of samples (for smoke testing).

    Returns:
        HuggingFace Dataset object.

    Raises:
        FileNotFoundError: If dataset not found in cache.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    # Load from cache (offline)
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        dataset = load_dataset(
            f"BabyLM-community/{subset}",
            split=split,
            cache_dir=cache_dir,
        )

        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return dataset
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load {subset} from cache at {cache_dir}. "
            f"Please ensure the dataset is in HF cache. Error: {e}"
        ) from e


def build_token_stream(
    dataset: Dataset,
    tokenizer_fn: Any,
    seq_len: int = 128,
    max_tokens: int | None = None,
    cache_dir: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Build a token stream from dataset text using tokenizer.

    Concatenates all texts, tokenizes, and chunks into sequences of seq_len.
    Returns token IDs as a tensor (suitable for model training).

    Args:
        dataset: HuggingFace Dataset with 'text' field.
        tokenizer_fn: Function that takes a string and returns token list.
        seq_len: Sequence length for chunking.
        max_tokens: Maximum tokens to include (for smoke testing).

    Returns:
        Tuple of (token_tensor [total_tokens, seq_len], num_sequences).
    """
    # Optional disk cache: tokenizing millions of tokens costs minutes per run.
    # Key = (fingerprint of the text column, seq_len, max_tokens). tokenizer_fn is
    # not hashable; callers must use distinct cache_dirs for distinct tokenizers.
    cache_path = None
    if cache_dir is not None:
        import hashlib
        from pathlib import Path

        h = hashlib.sha256()
        n = len(dataset)
        h.update(f"{n}:{seq_len}:{max_tokens}".encode())
        # Fingerprint on a sample of rows (hashing every row costs as much as tokenizing)
        for idx in range(0, n, max(1, n // 64)):
            h.update(str(dataset[idx].get("text", ""))[:256].encode())
        cache_path = Path(cache_dir) / f"tokstream_{h.hexdigest()[:24]}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            return cached["tensor"], int(cached["num_seqs"])

    # Tokenize sample-by-sample, accumulating until max_tokens is reached.
    # (Tokenizing per sample avoids tokenizer max-length limits and lets us
    # stop early instead of processing the whole corpus.)
    tokens: list[int] = []
    for text in dataset["text"]:
        if not text:
            continue
        t = tokenizer_fn(text)
        if isinstance(t, str):
            t = [int(x) if x.isdigit() else 0 for x in t.split()]
        elif not isinstance(t, list):
            t = list(t)
        tokens.extend(int(x) if isinstance(x, (int, float)) else 0 for x in t)
        if max_tokens and len(tokens) >= max_tokens:
            break

    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    # Chunk into sequences
    num_seqs = len(tokens) // seq_len
    if num_seqs == 0:
        num_seqs = 1
        # Pad if needed
        tokens = tokens + [0] * (seq_len - len(tokens))

    token_tensor = torch.tensor(
        tokens[: num_seqs * seq_len],
        dtype=torch.long,
    ).reshape(num_seqs, seq_len)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tensor": token_tensor, "num_seqs": num_seqs}, cache_path)

    return token_tensor, num_seqs


class BabyLMDataLoader:
    """Simple data loader for BabyLM token streams.

    Provides batching and iteration over token sequences.
    """

    def __init__(
        self,
        token_tensor: torch.Tensor,
        batch_size: int = 32,
        shuffle: bool = True,
        device: str | None = None,
    ):
        """Initialize data loader.

        Args:
            token_tensor: Token tensor [num_seqs, seq_len].
            batch_size: Batch size.
            shuffle: Whether to shuffle batches.
            device: Device to load tensors on (e.g., 'cuda').
        """
        self.token_tensor = token_tensor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or "cpu"
        self.num_seqs = token_tensor.shape[0]
        self.indices = list(range(self.num_seqs))

        if shuffle:
            import random
            random.shuffle(self.indices)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over batches of token sequences."""
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = self.token_tensor[batch_indices].to(self.device)
            yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        return (self.num_seqs + self.batch_size - 1) // self.batch_size
