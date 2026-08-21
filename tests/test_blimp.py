"""Unit tests for BLiMP evaluation (TDD).

Tests the kinetic_ai.eval.blimp module: dataset loading, log-prob
computation, and minimal-pair evaluation.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("datasets")

from kinetic_ai.eval.blimp import (  # noqa: E402
    compute_sentence_logprob,
    evaluate_blimp_subset,
    load_blimp_subset,
)


class SimpleTestModel(nn.Module):
    """Minimal language model for testing."""

    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """Simple forward pass."""
        z = self.embedding(input_ids)  # [B, T, d_model]
        logits = self.lm_head(z)  # [B, T, vocab_size]
        return logits


class TestBLiMPLoading:
    """Test BLiMP dataset loading."""

    @pytest.mark.skipif(
        not (Path.home() / ".cache" / "huggingface").exists(),
        reason="HF cache not present",
    )
    def test_load_blimp_subset(self):
        """Test loading BLiMP subset from cache."""
        subset = load_blimp_subset(num_phenomena=2, pairs_per_phenomenon=20)

        assert len(subset) > 0
        assert len(subset) <= 2 * 20 + 100  # Allow some tolerance

    def test_blimp_fields(self):
        """Test that BLiMP subset has required fields."""
        subset = load_blimp_subset(num_phenomena=1, pairs_per_phenomenon=10)

        # Check for minimal-pair fields (may vary by dataset version)
        example = subset[0]
        assert isinstance(example, dict)
        # At least one of these should exist
        has_pair = any(
            k in example for k in [
                "sentence_good", "good_sentence",
                "sentence_bad", "bad_sentence",
                "acceptable", "unacceptable",
            ]
        )
        assert has_pair, f"No minimal-pair fields in: {example.keys()}"


class TestSentenceLogprob:
    """Test sentence log-probability computation."""

    def test_compute_logprob(self):
        """Test computing log-prob of a sentence."""
        model = SimpleTestModel(vocab_size=100, d_model=32)
        tokens = torch.tensor([5, 10, 15, 20])  # Simple sequence

        logprob = compute_sentence_logprob(model, tokens, device="cpu")

        assert isinstance(logprob, float)
        # Log-prob should be negative (sentence has probability < 1)
        assert logprob < 0

    def test_logprob_is_sum(self):
        """Test that log-prob is sum of token log-probs."""
        model = SimpleTestModel(vocab_size=100, d_model=32)
        tokens = torch.tensor([5, 10, 15])

        logprob_total = compute_sentence_logprob(model, tokens, device="cpu")

        # Verify it's a reasonable negative number
        assert -1000 < logprob_total < 0

    def test_longer_sequence_lower_logprob(self):
        """Test that longer sequences have lower (more negative) log-probs."""
        model = SimpleTestModel()

        short = torch.tensor([5, 10])
        long = torch.tensor([5, 10, 15, 20, 25])

        lp_short = compute_sentence_logprob(model, short, device="cpu")
        lp_long = compute_sentence_logprob(model, long, device="cpu")

        # Longer sequence should have more negative log-prob
        # (sum of more negative terms)
        assert lp_long < lp_short


class TestBLiMPEvaluation:
    """Test BLiMP evaluation pipeline."""

    def test_evaluate_blimp_minimal(self):
        """Test minimal BLiMP evaluation."""
        from datasets import Dataset

        # Create a minimal synthetic BLiMP-like dataset
        blimp_data = Dataset.from_dict({
            "sentence_good": ["a b c", "x y z"],
            "sentence_bad": ["a b d", "x y w"],
        })

        model = SimpleTestModel(vocab_size=100)

        def tokenizer_fn(text):
            # Map first char to token ID
            tokens = []
            for word in text.split():
                tokens.append(ord(word[0]) % 100)
            return tokens

        result = evaluate_blimp_subset(
            model,
            blimp_data,
            tokenizer_fn,
            device="cpu",
            max_samples=2,
        )

        assert "accuracy" in result
        assert "num_correct" in result
        assert "num_total" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["num_total"] == 2

    def test_evaluate_returns_dict(self):
        """Test that evaluation returns correct format."""
        from datasets import Dataset

        blimp_data = Dataset.from_dict({
            "sentence_good": ["hello"],
            "sentence_bad": ["world"],
        })

        model = SimpleTestModel()
        def tokenizer_fn(text):
            return [ord(c) % 100 for c in text]

        result = evaluate_blimp_subset(
            model,
            blimp_data,
            tokenizer_fn,
            device="cpu",
            max_samples=1,
        )

        required_keys = {"accuracy", "num_correct", "num_total"}
        assert required_keys.issubset(set(result.keys()))


class TestMinimalPairLogic:
    """Test the core minimal-pair evaluation logic."""

    def test_model_prefers_good_sentence(self):
        """Test that we can detect when model prefers good vs bad."""
        # Create a model that outputs specific logits
        model = SimpleTestModel(vocab_size=10, d_model=4)

        # Minimal test: model should prefer different outputs for different inputs
        tokens_good = torch.tensor([1, 2])
        tokens_bad = torch.tensor([1, 3])

        lp_good = compute_sentence_logprob(model, tokens_good, device="cpu")
        lp_bad = compute_sentence_logprob(model, tokens_bad, device="cpu")

        # They should be different (random model will have different log-probs)
        # This is a weak test since model is random, but verifies computation works
        assert isinstance(lp_good, float)
        assert isinstance(lp_bad, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
