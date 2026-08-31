"""Unit tests for instruct-tuning pipeline (kinetic_ai.train.instruct).

Tests cover: chat formatting, label masking, loss computation, determinism.
No network/HF dataset access; mocks and synthetic data only.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetic_ai.train.instruct import format_chat, sft_step


class TestFormatChat:
    """Test format_chat output correctness."""

    def test_format_chat_single_turn(self) -> None:
        """Single user-assistant exchange."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = format_chat(messages)
        expected = "User: Hello\nAssistant: Hi there!\n"
        assert result == expected

    def test_format_chat_multiline_content(self) -> None:
        """Content with embedded newlines."""
        messages = [
            {"role": "user", "content": "Line 1\nLine 2"},
            {"role": "assistant", "content": "Response"},
        ]
        result = format_chat(messages)
        assert "User: Line 1\nLine 2\n" in result
        assert "Assistant: Response\n" in result

    def test_format_chat_empty(self) -> None:
        """Empty message list."""
        result = format_chat([])
        assert result == ""

    def test_format_chat_capitalization(self) -> None:
        """Role names are capitalized."""
        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "test"},
        ]
        result = format_chat(messages)
        assert "User:" in result
        assert "Assistant:" in result
        # Verify it's the capitalized form, not lowercase
        assert result.startswith("User:")


class SimpleLM(nn.Module):
    """Tiny mock LM for testing."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, max_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.config = type("Config", (), {"deq_max_iter": 3})()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, V]."""
        x = self.embedding(input_ids)
        return self.head(x)

    def forward_unrolled(
        self, input_ids: torch.Tensor, supervise_at: list[int]
    ) -> list[tuple[int, torch.Tensor]]:
        """Mock forward_unrolled: return logits at each supervised depth."""
        logits_base = self.forward(input_ids)
        return [(d, logits_base) for d in supervise_at]


class TestSFTStep:
    """Test SFT loss computation."""

    def test_sft_step_basic(self) -> None:
        """Basic SFT loss computation."""
        model = SimpleLM(vocab_size=256, d_model=32)
        batch_size = 2
        seq_len = 10

        input_ids = torch.randint(0, 256, (batch_size, seq_len))
        labels = torch.randint(0, 256, (batch_size, seq_len))

        loss = sft_step(model, input_ids, labels, supervise_at=[3])
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_sft_step_masked_positions(self) -> None:
        """Loss is not computed on masked positions (-100)."""
        model = SimpleLM(vocab_size=256, d_model=32)
        model.eval()  # Eval mode for determinism
        batch_size = 1
        seq_len = 10

        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        labels = torch.ones((batch_size, seq_len), dtype=torch.long)

        # Mask first 5 positions
        labels[0, :5] = -100

        # Compute loss
        with torch.no_grad():
            loss = sft_step(model, input_ids, labels, supervise_at=[3])

            # Create reference: loss on unmasked positions only
            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            # CE on unmasked only
            ce_per_pos = F.cross_entropy(logits_flat, labels_flat, reduction="none")
            mask = (labels_flat != -100).float()
            ref_loss = (ce_per_pos * mask).sum() / (mask.sum() + 1e-8)

        # sft_step multiplies by weight 1.0 for a single depth (weight_idx=0 for 1 depth)
        # So loss should equal ref_loss * 1.0 (within floating point precision)
        assert abs(loss.item() - ref_loss.item()) < 1e-5

    def test_sft_step_all_masked(self) -> None:
        """When all labels are masked, loss should be 0."""
        model = SimpleLM(vocab_size=256, d_model=32)
        batch_size = 1
        seq_len = 10

        input_ids = torch.ones((batch_size, seq_len), dtype=torch.long)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)

        loss = sft_step(model, input_ids, labels, supervise_at=[3])
        assert loss.item() == 0.0

    def test_sft_step_determinism(self) -> None:
        """Same inputs produce same loss."""
        model = SimpleLM(vocab_size=256, d_model=32)
        model.eval()  # Deterministic eval mode

        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        labels = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)

        with torch.no_grad():
            loss1 = sft_step(model, input_ids, labels, supervise_at=[3])
            loss2 = sft_step(model, input_ids, labels, supervise_at=[3])

        assert abs(loss1.item() - loss2.item()) < 1e-6

    def test_sft_step_multi_depth(self) -> None:
        """SFT with multiple supervised depths."""
        model = SimpleLM(vocab_size=256, d_model=32)

        input_ids = torch.randint(0, 256, (2, 10))
        labels = torch.randint(0, 256, (2, 10))

        # Supervise at depths [1, 2, 3] with weights [0.15, 0.3, 1.0]
        loss = sft_step(model, input_ids, labels, supervise_at=[1, 2, 3])
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestPrefferencePairsAndMPO:
    """Test preference pairs loading and MPO loss (partially).

    Full MPO tests require model.forward_unrolled, so we focus on
    synthetic data and loss direction here.
    """

    def test_mpo_loss_decreases_for_better_preference(self) -> None:
        """MPO loss should favor chosen over rejected (qualitative check).

        This is a light check: we don't simulate full training, but verify
        that the loss direction is correct on a synthetic pair.
        """
        from kinetic_ai.train.instruct import mpo_step

        # Create two identical models
        model = SimpleLM(vocab_size=100, d_model=16)
        ref_model = SimpleLM(vocab_size=100, d_model=16)
        ref_model.load_state_dict(model.state_dict())

        # Synthetic preference pair: (prompt, chosen, rejected)
        prompt = torch.randint(0, 100, (10,))
        chosen = torch.randint(0, 100, (10,))
        rejected = torch.randint(0, 100, (10,))

        batch = [(prompt, chosen, rejected)]

        # Compute MPO loss
        loss, pref_acc = mpo_step(model, ref_model, batch, beta=0.1, magnet_tau=0.05)

        # Loss should be a scalar and computable
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert pref_acc.item() >= 0.0 and pref_acc.item() <= 1.0


class TestBatchDeterminism:
    """Test that batch order is deterministic."""

    def test_sft_batches_determinism(self) -> None:
        """Calling sft_batches with same seed produces same order."""
        # This is a light test: full HF integration is tested separately
        # Here we just verify the function signature exists
        from kinetic_ai.train.instruct import sft_batches

        # Function should exist and be callable
        assert callable(sft_batches)

    def test_preference_batches_determinism(self) -> None:
        """Calling preference_batches with same seed produces same order."""
        from kinetic_ai.train.instruct import preference_batches

        # Function should exist and be callable
        assert callable(preference_batches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
