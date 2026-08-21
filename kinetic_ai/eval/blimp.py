"""BLiMP (Blimp, a Benchmark of Linguistic Minimal Pairs) evaluation.

Evaluates language models on minimal-pair tasks where the model should
assign higher log-probability to the acceptable (correct) sentence than
the unacceptable (incorrect) minimal-pair sentence.

Reference: Warstadt et al. 2020, https://arxiv.org/abs/1901.05257
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset


def load_blimp_subset(
    num_phenomena: int = 5,
    pairs_per_phenomenon: int = 100,
    cache_dir: str | None = None,
) -> Dataset:
    """Load a subset of BLiMP for SMOKE testing.

    Selects num_phenomena × pairs_per_phenomenon samples from the
    nyu-mll/blimp dataset. BLiMP is organized by linguistic phenomena
    (e.g., 'adjunct_island', 'anaphor_gender_agreement', etc.).

    Args:
        num_phenomena: Number of linguistic phenomena to include.
        pairs_per_phenomenon: Pairs per phenomenon.
        cache_dir: Cache directory (default: ~/.cache/huggingface/hub).

    Returns:
        HuggingFace Dataset with subset of minimal pairs.

    Raises:
        FileNotFoundError: If BLiMP not in cache.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        # BLiMP requires a config (phenomenon) to be loaded
        # Load the first num_phenomena configs
        phenomena_to_load = [
            "adjunct_island",
            "anaphor_gender_agreement",
            "anaphor_number_agreement",
            "animate_subject_passive",
            "animate_subject_trans",
        ][:num_phenomena]

        all_data = []
        for phenom in phenomena_to_load:
            try:
                dataset = load_dataset(
                    "nyu-mll/blimp",
                    phenom,
                    cache_dir=cache_dir,
                    split="train",
                )
                # Take pairs_per_phenomenon samples
                n_to_take = min(pairs_per_phenomenon, len(dataset))
                samples = dataset.select(range(n_to_take))
                all_data.extend(samples)
            except Exception:
                # Skip if phenomenon not available
                pass

        if not all_data:
            raise FileNotFoundError("Could not load any BLiMP phenomena")

        subset = Dataset.from_list(all_data)
        return subset

    except Exception as e:
        raise FileNotFoundError(
            f"Could not load BLiMP from cache at {cache_dir}. Error: {e}"
        ) from e


def compute_sentence_logprob(
    model: nn.Module,
    tokens: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Compute log-probability of a sentence under a language model.

    Sums log-probs of all tokens in the sequence (left-to-right).
    For minimal-pair evaluation: log p(sentence) = sum_t log p(t | t<t).

    Args:
        model: Language model with forward(input_ids) -> logits signature.
        tokens: Token IDs [seq_len], 1D tensor.
        device: Device to run on.

    Returns:
        Total log-probability (sum of all token log-probs).
    """
    tokens = tokens.unsqueeze(0).to(device)  # [1, seq_len]
    model.eval()

    with torch.no_grad():
        # Forward pass: logits [1, seq_len, vocab_size]
        logits = model(tokens)

        # Get log-probs for each position
        # Shift: predict token t+1 from tokens up to t
        logits_shifted = logits[0, :-1, :]  # [seq_len-1, vocab_size]
        targets = tokens[0, 1:]  # [seq_len-1]

        # Compute log-softmax
        log_probs = torch.nn.functional.log_softmax(logits_shifted, dim=-1)

        # Gather log-probs for target tokens
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1))  # [seq_len-1, 1]

        # Sum log-probs
        total_logprob = target_log_probs.sum().item()

    return total_logprob


def evaluate_blimp_subset(
    model: nn.Module,
    blimp_subset: Dataset,
    tokenizer_fn: Any,
    device: str = "cpu",
    max_samples: int | None = None,
) -> dict:
    """Evaluate model on BLiMP subset via minimal-pair scoring.

    For each (acceptable, unacceptable) pair, model should assign
    higher log-prob to acceptable sentence. Accuracy = fraction correct.

    Args:
        model: Language model.
        blimp_subset: BLiMP Dataset with fields: 'sentence_good', 'sentence_bad'.
        tokenizer_fn: Function to tokenize strings -> token IDs (list of ints).
        device: Device to run on.
        max_samples: Limit number of pairs (for speed).

    Returns:
        Dict with keys:
            - accuracy: Fraction of pairs where model prefers correct sentence.
            - num_correct: Number of correct pairs.
            - num_total: Total number of pairs evaluated.
    """
    model.eval()
    correct = 0
    total = 0

    for i, example in enumerate(blimp_subset):
        if max_samples and i >= max_samples:
            break

        # Get sentences
        try:
            sent_good = example.get("sentence_good") or example.get("good_sentence")
            sent_bad = example.get("sentence_bad") or example.get("bad_sentence")
        except (KeyError, AttributeError):
            # Skip if no sentence fields
            continue

        if not sent_good or not sent_bad:
            continue

        # Tokenize
        try:
            tokens_good = tokenizer_fn(sent_good)
            tokens_bad = tokenizer_fn(sent_bad)

            if isinstance(tokens_good, str):
                tokens_good = [int(t) if t.isdigit() else 0 for t in tokens_good.split()]
            if isinstance(tokens_bad, str):
                tokens_bad = [int(t) if t.isdigit() else 0 for t in tokens_bad.split()]

            tokens_good = torch.tensor(tokens_good, dtype=torch.long)
            tokens_bad = torch.tensor(tokens_bad, dtype=torch.long)

            if len(tokens_good) < 2 or len(tokens_bad) < 2:
                continue
        except Exception:
            continue

        # Compute log-probs
        lp_good = compute_sentence_logprob(model, tokens_good, device)
        lp_bad = compute_sentence_logprob(model, tokens_bad, device)

        # Check if model prefers good sentence
        if lp_good > lp_bad:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": total,
    }
