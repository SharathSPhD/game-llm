"""Tokenizer loading and building for language model pretraining.

Provides utilities to load GPT-2 tokenizer from HF cache or build
a custom BPE tokenizer from a text corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from transformers import GPT2Tokenizer, PreTrainedTokenizer


def load_gpt2_tokenizer() -> PreTrainedTokenizer | None:
    """Load GPT-2 tokenizer from HuggingFace cache if available.

    Returns:
        GPT2Tokenizer or None if not available.
    """
    try:
        # Try loading from cache (offline mode)
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            # Load GPT-2 tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(
                "gpt2",
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
            return cast(PreTrainedTokenizer, tokenizer)
    except Exception:
        pass
    return None


def build_bpe_tokenizer(
    texts: list[str],
    vocab_size: int = 8192,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build a simple BPE tokenizer via tokenizers library.

    This is a minimal implementation for SMOKE testing.
    For production, use HF tokenizers.Tokenizer with BPE trainer.

    Args:
        texts: List of text samples to train on.
        vocab_size: Target vocabulary size.

    Returns:
        Tuple of (token2id dict, id2token dict).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
    from tokenizers.trainers import BpeTrainer

    # Create tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.post_processor = ByteLevelProcessor()

    # Trainer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>"])

    # Train on texts
    tokenizer.train_from_iterator(texts, trainer)

    # Build token2id and id2token
    token2id = tokenizer.get_vocab()
    id2token = {v: k for k, v in token2id.items()}

    return token2id, id2token


def load_or_build_tokenizer(
    texts: list[str] | None = None,
    vocab_size: int = 8192,
) -> tuple[dict[str, int], dict[int, str], str]:
    """Load GPT-2 tokenizer or build custom BPE tokenizer.

    Prefers GPT-2 from cache (offline). Falls back to building custom BPE
    if texts are provided. If neither is possible, returns a minimal vocab.

    Args:
        texts: Optional list of texts to train custom BPE on.
        vocab_size: Vocabulary size for custom BPE (default 8192).

    Returns:
        Tuple of (token2id, id2token, tokenizer_choice_string).
    """
    # Try GPT-2 first (offline)
    gpt2_tok = load_gpt2_tokenizer()
    if gpt2_tok is not None:
        # Convert GPT-2 tokenizer to our format
        token2id = gpt2_tok.encoder
        id2token = {v: k for k, v in token2id.items()}
        return token2id, id2token, "gpt2"

    # Fall back to custom BPE if texts provided
    if texts:
        token2id, id2token = build_bpe_tokenizer(texts, vocab_size)
        return token2id, id2token, "custom_bpe"

    # Minimal fallback
    token2id = {f"<tok_{i}>": i for i in range(vocab_size)}
    id2token = {i: f"<tok_{i}>" for i in range(vocab_size)}
    return token2id, id2token, "minimal"
