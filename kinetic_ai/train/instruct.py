"""Instruct-tuning pipeline for EqLM with SFT and MPO (Magnetic Preference Optimization).

This module provides data loading, batch processing, and training step functions for:
  - SFT (Supervised Fine-Tuning): plain cross-entropy with assistant-only loss masking
  - MPO: preference optimization via DPO loss + magnetic anchor to reference model

Design choices:
  - Chat format: plain text "User: {..}\nAssistant: {..}\n" with EOS between turns
  - No special tokens: GPT-2 vocab (50257) only
  - Anytime supervision: forward_unrolled at depths [0.375, 0.6875, 1.0] × max_iter
    with weights [0.15, 0.3, 1.0] (F24 regime from exp39)
  - Magnetic anchor: KL(policy || reference) regularizer on chosen seqs with weight magnet_tau
    (reuse from MMD: log-softmax space with Bregman geometry, per exp17_pma_dpo.py)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor

# Type aliases for clarity
BatchData: TypeAlias = tuple[Tensor, Tensor]  # (input_ids, labels)
PreferencePair: TypeAlias = tuple[Tensor, Tensor, Tensor]  # (prompt, chosen, rejected)


@dataclass
class InstructConfig:
    """Configuration for instruct-tuning."""
    vocab_size: int = 50257
    seq_len: int = 128
    batch_size: int = 8
    seed: int = 42


def format_chat(messages: list[dict[str, str]]) -> str:
    """Format a list of messages into plain-text chat format.

    Each message is a dict with "role" (str) and "content" (str).
    Format: "User: {content}\nAssistant: {content}\n" with no special tokens.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                 Roles are typically 'user' or 'assistant'.

    Returns:
        Formatted plain-text string.
    """
    result = []
    for msg in messages:
        role = msg["role"].capitalize()  # 'user' -> 'User', 'assistant' -> 'Assistant'
        content = msg["content"]
        result.append(f"{role}: {content}\n")
    return "".join(result)


def sft_batches(
    dataset_name: str,
    tok: Any,  # HF tokenizer (PreTrainedTokenizer)
    seq_len: int,
    batch: int,
    seed: int,
    limit: int | None = None,
) -> Iterator[BatchData]:
    """Stream SFT data as (input_ids, labels) with assistant-only loss masking.

    Loads HuggingFaceTB/smoltalk (config="all", split="train"), formats as chat,
    tokenizes, and yields packed batches with label masking (-100 on user tokens).

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceTB/smoltalk").
        tok: HuggingFace tokenizer (must have encode_plus and eos_token_id).
        seq_len: Sequence length for truncation/padding.
        batch: Batch size (number of sequences per batch).
        seed: Random seed for reproducibility.
        limit: Optional max number of examples to load (for testing).

    Yields:
        (input_ids, labels) tensors:
          - input_ids: [batch, seq_len] token IDs
          - labels: [batch, seq_len] with -100 where user tokens (to mask loss)
    """
    from datasets import load_dataset

    # Load dataset
    try:
        # SmolTalk: each example has 'messages' list of {'role': 'user'/'assistant', 'content': str}
        # Use 'all' config for full SmolTalk dataset
        ds = load_dataset(dataset_name, name="all", split="train")
    except Exception:
        # Fallback if dataset name or split changes
        print(f"Warning: could not load {dataset_name}, trying alternate ID")
        raise

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    # Build batches
    batch_buffer = []
    for example in ds:
        if "messages" not in example:
            continue

        # Format chat
        text = format_chat(example["messages"])

        # Tokenize
        input_ids = tok.encode(
            text,
            truncation=True,
            max_length=seq_len,
        )
        # Pad to seq_len
        if len(input_ids) < seq_len:
            input_ids = input_ids + [tok.eos_token_id] * (seq_len - len(input_ids))
        else:
            input_ids = input_ids[:seq_len]

        # Create labels with assistant-only masking
        # Mask all tokens until we see "Assistant:" to only compute loss on assistant outputs
        labels = input_ids.copy()

        # Scan for "Assistant:" and mask everything before it
        user_end = text.find("Assistant:")
        if user_end >= 0:
            # Tokenize user part and mask those indices
            user_part = text[:user_end]
            user_tokens = tok.encode(user_part, return_tensors=None)
            user_len = len(user_tokens)

            # Mask user tokens
            for idx in range(min(user_len, len(labels))):
                labels[idx] = -100

        batch_buffer.append((input_ids, labels))

        if len(batch_buffer) >= batch:
            # Yield batch
            ids_batch = torch.tensor(
                [b[0][:seq_len] for b in batch_buffer[:batch]], dtype=torch.long
            )
            labels_batch = torch.tensor(
                [b[1][:seq_len] for b in batch_buffer[:batch]], dtype=torch.long
            )
            yield (ids_batch, labels_batch)
            batch_buffer = batch_buffer[batch:]


def sft_step(
    model: Any,  # EqLM model with forward_unrolled method
    input_ids: Tensor,
    labels: Tensor,
    supervise_at: list[int] | None = None,
) -> Tensor:
    """Compute SFT loss with anytime supervision (F24 regime).

    Applies forward_unrolled at specified depths with weighted cross-entropy.
    Loss is computed ONLY on unmasked label positions (labels != -100).

    Weights [0.15, 0.3, 1.0] correspond to the depths in supervise_at (per F24 regime).

    Args:
        model: EqLM model with forward_unrolled(input_ids, supervise_at) method.
        input_ids: Token IDs [B, T].
        labels: Target token IDs [B, T] with -100 for masked positions.
        supervise_at: Depths to supervise (e.g., [6, 11, 16]). If None, uses [model.config.deq_max_iter].

    Returns:
        Scalar loss tensor (mean over batch and unmasked positions).
    """
    if supervise_at is None:
        # Type ignore: accessing model.config.deq_max_iter (not all models have this)
        supervise_at = [model.config.deq_max_iter]  # type: ignore

    # Forward unrolled: returns list of (depth, logits) tuples (sorted by depth)
    outputs = model.forward_unrolled(input_ids, supervise_at=supervise_at)  # type: ignore

    # Weights for depths: [0.15, 0.3, 1.0] for standard F24 (3 depths)
    # Generalize to handle arbitrary number of depths
    num_depths = len(outputs)
    if num_depths == 1:
        weights = [1.0]
    elif num_depths == 2:
        weights = [0.3, 1.0]
    else:  # num_depths >= 3
        weights = [0.15, 0.3, 1.0] + [1.0] * (num_depths - 3)

    total_loss: Tensor | float = 0.0
    for idx, (_depth, logits) in enumerate(outputs):
        weight = weights[idx]

        # Cross-entropy on unmasked positions only
        # Flatten for CE: [B*T, V]
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Compute CE
        loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")

        # Mask: set loss to 0 where labels == -100
        mask = (labels_flat != -100).float()
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        total_loss = (weight * loss if isinstance(total_loss, float) else
                      total_loss + weight * loss)

    return total_loss if isinstance(total_loss, Tensor) else torch.tensor(total_loss)


def preference_batches(
    dataset_name: str,
    tok: Any,  # HF tokenizer (PreTrainedTokenizer)
    seq_len: int,
    batch: int,
    seed: int,
    limit: int | None = None,
) -> Iterator[list[PreferencePair]]:
    """Stream preference pairs (prompt, chosen, rejected) from HuggingFaceH4/ultrafeedback_binarized.

    Args:
        dataset_name: HF dataset name (e.g., "HuggingFaceH4/ultrafeedback_binarized").
        tok: HuggingFace tokenizer.
        seq_len: Max sequence length.
        batch: Batch size (number of pairs per batch).
        seed: Random seed.
        limit: Optional max number of examples.

    Yields:
        List of (prompt_ids, chosen_ids, rejected_ids) tensors, each [seq_len].
    """
    from datasets import load_dataset

    # Load preference dataset
    try:
        ds = load_dataset(
            dataset_name,
            split="train_prefs",
            trust_remote_code=True,
        )
    except Exception:
        # Try alternate split name
        ds = load_dataset(
            dataset_name,
            split="train_sft",
            trust_remote_code=True,
        )

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    batch_buffer = []
    for example in ds:
        # UltraFeedback has 'prompt' (str) and 'chosen', 'rejected' (lists of messages)
        prompt_text = example.get("prompt", "")
        chosen_msgs = example.get("chosen", [])
        rejected_msgs = example.get("rejected", [])

        # Format messages as text
        if isinstance(chosen_msgs, list) and chosen_msgs:
            chosen_text = format_chat(chosen_msgs)
        else:
            chosen_text = ""

        if isinstance(rejected_msgs, list) and rejected_msgs:
            rejected_text = format_chat(rejected_msgs)
        else:
            rejected_text = ""

        if not (prompt_text and chosen_text and rejected_text):
            continue

        # Combine prompt + response for full context
        # For now, just use the response text directly
        # Tokenize
        prompt_ids = torch.tensor(
            tok.encode(prompt_text)[:seq_len],
            dtype=torch.long,
        )
        chosen_ids = torch.tensor(
            tok.encode(chosen_text)[:seq_len],
            dtype=torch.long,
        )
        rejected_ids = torch.tensor(
            tok.encode(rejected_text)[:seq_len],
            dtype=torch.long,
        )

        # Pad to seq_len
        def pad_to(t: Tensor, length: int, pad_id: int = 0) -> Tensor:
            if t.shape[0] < length:
                pad = torch.full((length - t.shape[0],), pad_id, dtype=torch.long)
                return torch.cat([t, pad])
            return t

        prompt_ids = pad_to(prompt_ids, seq_len)
        chosen_ids = pad_to(chosen_ids, seq_len)
        rejected_ids = pad_to(rejected_ids, seq_len)

        batch_buffer.append((prompt_ids, chosen_ids, rejected_ids))

        if len(batch_buffer) >= batch:
            yield batch_buffer[:batch]
            batch_buffer = batch_buffer[batch:]


def mpo_step(
    model: Any,  # EqLM model
    ref_model: Any,  # Reference EqLM model
    batch: list[PreferencePair],
    beta: float = 0.1,
    magnet_tau: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Compute MPO loss: DPO + magnetic anchor to reference model.

    MPO combines:
      1. DPO loss: log-sigmoid(beta * (chosen_logprob - rejected_logprob))
      2. Magnetic anchor: KL(policy || reference) on chosen sequences
         with weight magnet_tau (Bregman geometry, per MMD formulation)

    Args:
        model: Policy model (EqLM).
        ref_model: Reference/frozen model (same architecture as model).
        batch: List of (prompt_ids, chosen_ids, rejected_ids) pairs.
        beta: DPO beta (scale on log-ratio).
        magnet_tau: Magnetic anchor weight (strength of pull to reference).

    Returns:
        (loss, preference_accuracy) where:
          - loss: scalar tensor for backward pass
          - preference_accuracy: fraction of pairs where chosen > rejected
    """
    device = next(model.parameters()).device

    model.train()
    ref_model.eval()

    batch_size = len(batch)
    total_dpo_loss: Tensor | float = 0.0
    total_kl_loss: Tensor | float = 0.0
    correct_pref = 0

    for prompt_ids, chosen_ids, rejected_ids in batch:
        prompt_ids = prompt_ids.to(device)
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)

        # Compute log probabilities
        with torch.no_grad():
            # Reference model: frozen
            ref_chosen_logits = ref_model(chosen_ids.unsqueeze(0))
            ref_rejected_logits = ref_model(rejected_ids.unsqueeze(0))

            ref_chosen_lps = F.log_softmax(ref_chosen_logits, dim=-1)
            ref_rejected_lps = F.log_softmax(ref_rejected_logits, dim=-1)

            # Sequence log probabilities
            ref_chosen_logprob = (
                ref_chosen_lps[0, :-1, :].gather(1, chosen_ids[1:].unsqueeze(1)).sum()
            )
            ref_rejected_logprob = (
                ref_rejected_lps[0, :-1, :].gather(1, rejected_ids[1:].unsqueeze(1)).sum()
            )

        # Policy model: with gradients
        chosen_logits = model(chosen_ids.unsqueeze(0))
        rejected_logits = model(rejected_ids.unsqueeze(0))

        chosen_lps = F.log_softmax(chosen_logits, dim=-1)
        rejected_lps = F.log_softmax(rejected_logits, dim=-1)

        # Sequence log probabilities
        chosen_logprob = (
            chosen_lps[0, :-1, :].gather(1, chosen_ids[1:].unsqueeze(1)).sum()
        )
        rejected_logprob = (
            rejected_lps[0, :-1, :].gather(1, rejected_ids[1:].unsqueeze(1)).sum()
        )

        # DPO loss
        logits_diff = chosen_logprob - rejected_logprob
        ref_logits_diff = ref_chosen_logprob - ref_rejected_logprob

        # log-sigmoid of the preference margin
        dpo_loss = -torch.nn.functional.logsigmoid(
            beta * (logits_diff - ref_logits_diff)
        )
        total_dpo_loss = total_dpo_loss + dpo_loss

        # Preference accuracy
        if (chosen_logprob > rejected_logprob).item():
            correct_pref += 1

        # Magnetic anchor: KL(policy || reference) on chosen sequences
        # KL = sum_t P(policy) * (log P(policy) - log P(reference))
        # For each token position, KL = softmax(policy_logits) * (policy_lps - ref_lps)
        kl_per_pos = torch.exp(chosen_lps[0, :-1, :]) * (
            chosen_lps[0, :-1, :] - ref_chosen_lps[0, :-1, :]
        )
        kl = kl_per_pos.sum()  # Sum over all positions and vocabulary
        total_kl_loss = total_kl_loss + kl

    # Average over batch
    if isinstance(total_dpo_loss, float):
        avg_dpo_loss = torch.tensor(total_dpo_loss) / batch_size
    else:
        avg_dpo_loss = total_dpo_loss / batch_size

    if isinstance(total_kl_loss, float):
        avg_kl_loss = torch.tensor(total_kl_loss) / batch_size
    else:
        avg_kl_loss = total_kl_loss / batch_size

    # Combined loss
    loss = avg_dpo_loss + magnet_tau * avg_kl_loss

    # Preference accuracy
    pref_accuracy = correct_pref / batch_size

    return loss, torch.tensor(pref_accuracy, dtype=torch.float32)
