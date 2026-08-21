"""Unit tests for data loading and tokenization (TDD).

Tests the kinetic_ai.data module: dataset loading, tokenization,
and token stream building.
"""

from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from kinetic_ai.data import (  # noqa: E402
    build_token_stream,
    load_babylm_dataset,
    load_or_build_tokenizer,
)

_HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
requires_hf_cache = pytest.mark.skipif(
    not (_HF_HUB / "datasets--BabyLM-community--BabyLM-2026-Strict-Small").exists(),
    reason="BabyLM HF cache not present on this machine",
)


class TestTokenizer:
    """Test tokenizer loading and building."""

    @requires_hf_cache
    def test_load_gpt2_tokenizer(self):
        """Test loading GPT-2 tokenizer from cache."""
        token2id, id2token, choice = load_or_build_tokenizer()

        assert isinstance(token2id, dict)
        assert isinstance(id2token, dict)
        assert len(token2id) > 0
        assert len(id2token) == len(token2id)
        assert choice in ["gpt2", "custom_bpe", "minimal"]

    def test_tokenizer_consistency(self):
        """Test that token2id and id2token are consistent."""
        token2id, id2token, _ = load_or_build_tokenizer()

        for tok_str, tok_id in list(token2id.items())[:10]:
            assert id2token[tok_id] == tok_str, f"Mismatch for token {tok_str}"

    def test_tokenizer_vocab_size(self):
        """Test that tokenizer has reasonable vocab size."""
        token2id, id2token, choice = load_or_build_tokenizer()

        # GPT-2 has ~50k tokens; custom BPE can be smaller
        if choice == "gpt2":
            assert 40000 < len(token2id) < 60000
        else:
            assert len(token2id) > 0


class TestBabyLMDataset:
    """Test BabyLM dataset loading."""

    @requires_hf_cache
    def test_load_babylm_dataset(self):
        """Test loading BabyLM from cache."""
        dataset = load_babylm_dataset(max_samples=100)

        assert len(dataset) > 0
        assert "text" in dataset.column_names
        assert isinstance(dataset[0]["text"], str)

    def test_babylm_max_samples(self):
        """Test that max_samples limit is respected."""
        max_n = 50
        dataset = load_babylm_dataset(max_samples=max_n)

        assert len(dataset) == max_n


class TestTokenStream:
    """Test token stream building."""

    def test_build_token_stream(self):
        """Test building token stream from dataset."""
        from datasets import Dataset

        # Create minimal dataset
        dataset = Dataset.from_dict({
            "text": ["hello world", "foo bar baz", "test sentence"]
        })

        def simple_tokenizer(text):
            return [ord(word[0]) for word in text.split()]

        token_tensor, num_seqs = build_token_stream(
            dataset,
            simple_tokenizer,
            seq_len=4,
            max_tokens=100,
        )

        assert isinstance(token_tensor, torch.Tensor)
        assert token_tensor.shape[1] == 4  # seq_len dimension
        assert num_seqs == token_tensor.shape[0]

    def test_token_stream_shape(self):
        """Test that token stream has correct shape."""
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "text": ["a b c d e f g h i j"] * 10
        })

        def simple_tokenizer(text):
            return [i for i, _ in enumerate(text.split())]

        token_tensor, num_seqs = build_token_stream(
            dataset,
            simple_tokenizer,
            seq_len=8,
        )

        assert token_tensor.shape == (num_seqs, 8)
        assert token_tensor.dtype == torch.long


class TestDataLoader:
    """Test BabyLM data loader."""

    def test_dataloader_batching(self):
        """Test that data loader correctly batches data."""
        from kinetic_ai.data import BabyLMDataLoader

        token_tensor = torch.randint(0, 100, (32, 16))  # 32 seqs, 16 tokens
        loader = BabyLMDataLoader(token_tensor, batch_size=8, shuffle=False)

        batches = list(loader)
        assert len(batches) == 4  # 32 seqs / 8 batch_size
        assert batches[0].shape == (8, 16)

    def test_dataloader_to_device(self):
        """Test that data loader moves tensors to device."""
        from kinetic_ai.data import BabyLMDataLoader

        token_tensor = torch.randint(0, 100, (16, 8))
        loader = BabyLMDataLoader(
            token_tensor,
            batch_size=4,
            device="cpu",
        )

        batch = next(iter(loader))
        assert batch.device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTokenStreamCoverage:
    """Regression: build_token_stream must not cap input at 1000 samples (F9 blocker)."""

    def test_consumes_beyond_1000_samples(self) -> None:
        from datasets import Dataset

        # 2000 samples of 10 tokens each; requesting 15000 tokens requires >1000 samples
        ds = Dataset.from_dict({"text": ["a b c d e f g h i j"] * 2000})
        tokenizer_fn = lambda text: [1] * len(text.split())  # noqa: E731
        tensor, num_seqs = build_token_stream(ds, tokenizer_fn, seq_len=100, max_tokens=15000)
        assert tensor.numel() >= 15000 - 100, f"stream too small: {tensor.numel()}"

    def test_max_tokens_respected(self) -> None:
        from datasets import Dataset

        ds = Dataset.from_dict({"text": ["a b c"] * 50})
        tokenizer_fn = lambda text: [1] * len(text.split())  # noqa: E731
        tensor, _ = build_token_stream(ds, tokenizer_fn, seq_len=10, max_tokens=60)
        assert tensor.numel() <= 60


class TestTokenStreamCache:
    """Disk cache: identical (texts-hash, seq_len, max_tokens) must not re-tokenize."""

    def test_cache_roundtrip(self, tmp_path) -> None:
        from datasets import Dataset

        ds = Dataset.from_dict({"text": ["a b c d e f g h"] * 200})
        calls = {"n": 0}

        def tokenizer_fn(text):
            calls["n"] += 1
            return [1] * len(text.split())

        t1, n1 = build_token_stream(
            ds, tokenizer_fn, seq_len=8, max_tokens=800, cache_dir=str(tmp_path)
        )
        first_calls = calls["n"]
        assert first_calls > 0
        t2, n2 = build_token_stream(
            ds, tokenizer_fn, seq_len=8, max_tokens=800, cache_dir=str(tmp_path)
        )
        assert calls["n"] == first_calls, "second call should hit the cache"
        assert torch.equal(t1, t2) and n1 == n2

    def test_cache_key_differs_on_params(self, tmp_path) -> None:
        from datasets import Dataset

        ds = Dataset.from_dict({"text": ["a b c d"] * 100})
        fn = lambda text: [1] * len(text.split())  # noqa: E731
        t1, _ = build_token_stream(ds, fn, seq_len=4, max_tokens=200, cache_dir=str(tmp_path))
        t2, _ = build_token_stream(ds, fn, seq_len=8, max_tokens=200, cache_dir=str(tmp_path))
        assert t1.shape[1] == 4 and t2.shape[1] == 8
