"""Build a production data pack for SPEC 0022: the 1B twin run.

Streams FineWeb-Edu (or local text files for testing) through GPT-2 tokenizer,
writes tokenized data to uint16 memmap shards, and generates a manifest with
sha256 checksums, token counts, and resumability state. Holdout tokens are
drawn from the stream tail and never trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np


def compute_file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 of a file, streaming to avoid memory overhead."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_progress(progress_path: Path) -> dict[str, Any]:
    """Load resumability state from progress.json."""
    if progress_path.exists():
        return cast(dict[str, Any], json.loads(progress_path.read_text()))
    return {
        "docs_consumed": 0,
        "tokens_written": 0,
        "shards_done": 0,
        "carry_tokens": [],
    }


def save_progress(
    progress_path: Path,
    docs_consumed: int,
    tokens_written: int,
    shards_done: int,
    carry_tokens: list[int],
) -> None:
    """Save resumability state to progress.json."""
    progress_path.write_text(
        json.dumps(
            {
                "docs_consumed": docs_consumed,
                "tokens_written": tokens_written,
                "shards_done": shards_done,
                "carry_tokens": carry_tokens,
            }
        )
    )


def main() -> int:
    """Build the data pack."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    ap = argparse.ArgumentParser(
        description="Build a production data pack for SPEC 0022."
    )
    ap.add_argument(
        "--out-dir",
        default="data/pack_1b",
        help="Output directory for shards and manifest",
    )
    ap.add_argument(
        "--train-tokens",
        type=int,
        default=10_500_000_000,
        help="Number of training tokens to produce",
    )
    ap.add_argument(
        "--holdout-tokens",
        type=int,
        default=20_000_000,
        help="Number of holdout tokens (from stream tail)",
    )
    ap.add_argument(
        "--shard-tokens",
        type=int,
        default=500_000_000,
        help="Tokens per shard file",
    )
    ap.add_argument(
        "--batch-docs",
        type=int,
        default=512,
        help="Documents per tokenizer batch",
    )
    ap.add_argument(
        "--local-texts-file",
        type=str,
        default=None,
        help="For testing: read documents from plain text file (one per line) "
        "instead of HF streaming",
    )
    ap.add_argument(
        "--dataset-override",
        type=str,
        default=None,
        help="For testing: override dataset name",
    )
    ap.add_argument(
        "--date",
        type=str,
        default=None,
        help="ISO date string for manifest (omitted if not provided)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / "progress.json"
    progress = load_progress(progress_path)

    # Load tokenizer
    from transformers import AutoTokenizer

    print("Loading GPT-2 tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = len(tok)
    eos_id = tok.eos_token_id

    # Verify vocab size fits in uint16
    assert (
        vocab_size < 65536
    ), f"Vocab size {vocab_size} does not fit in uint16"
    print(f"Vocabulary size: {vocab_size}, EOS ID: {eos_id}", flush=True)

    # Open or create data stream
    if args.local_texts_file:
        # For testing: read from local file
        with open(args.local_texts_file) as f:
            all_lines = [line.rstrip("\n") for line in f if line.strip()]

        def stream_docs() -> Any:
            for line in all_lines:
                yield {"text": line}

        dataset_name = args.dataset_override or "local-texts"
    else:
        # Production: stream from HF
        from datasets import load_dataset

        print("Loading HuggingFaceFW/fineweb-edu (sample-100BT)...", flush=True)
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-100BT",
            split="train",
            streaming=True,
        )

        def stream_docs() -> Any:
            yield from ds

        dataset_name = "HuggingFaceFW/fineweb-edu"

    # Skip already-consumed documents
    docs_consumed = progress["docs_consumed"]
    tokens_written = progress["tokens_written"]
    shards_done = progress["shards_done"]
    carry_tokens = progress.get("carry_tokens", [])

    print(
        f"Resuming: {docs_consumed} docs consumed, "
        f"{tokens_written} tokens written, "
        f"{shards_done} shards completed",
        flush=True,
    )

    stream = stream_docs()
    for _ in range(docs_consumed):
        next(stream)

    # Manifest accumulator
    shards_info: list[dict[str, Any]] = []

    # Main tokenization loop
    print("Beginning tokenization and packing...", flush=True)
    t_start = time.time()
    current_shard_idx = shards_done
    buf: list[int] = carry_tokens.copy()
    train_complete = False

    for doc_idx, doc in enumerate(stream):
        if doc_idx % (args.batch_docs * 100) == 0 and doc_idx > 0 and not train_complete:
            elapsed = time.time() - t_start
            tokens_per_sec = tokens_written / max(elapsed, 1e-6)
            remaining_tokens = args.train_tokens + args.holdout_tokens - (
                tokens_written + len(buf)
            )
            eta_sec = remaining_tokens / max(tokens_per_sec, 1e-6)
            eta_hours = eta_sec / 3600
            print(
                f"  {doc_idx} docs, {tokens_written/1e9:.2f}B tokens, "
                f"{tokens_per_sec/1e6:.2f}M tok/s, "
                f"ETA {eta_hours:.1f}h",
                flush=True,
            )

        # Tokenize and append
        text = doc.get("text", "")
        if not text:
            continue

        ids = tok(text, add_special_tokens=False).input_ids
        buf.extend(ids)
        buf.append(eos_id)

        # Write shard when buffer is large enough (only if training not complete)
        if not train_complete:
            while len(buf) >= args.shard_tokens:
                shard_tokens = buf[: args.shard_tokens]
                buf = buf[args.shard_tokens :]

                shard_path = out_dir / f"shard_{current_shard_idx:05d}.bin"
                data = np.array(shard_tokens, dtype=np.uint16)

                # Verify all ids fit in uint16
                assert (data < 65536).all(), "Token id >= 65536 in shard"

                data.tofile(shard_path)
                sha256 = compute_file_sha256(shard_path)
                shards_info.append(
                    {
                        "file": f"shard_{current_shard_idx:05d}.bin",
                        "sha256": sha256,
                        "tokens": len(shard_tokens),
                    }
                )

                current_shard_idx += 1
                tokens_written += len(shard_tokens)

                # Update progress
                save_progress(
                    progress_path,
                    docs_consumed + doc_idx + 1,
                    tokens_written,
                    current_shard_idx - shards_done,
                    buf,
                )

                # Check if training tokens are complete
                if tokens_written >= args.train_tokens:
                    train_complete = True
                    break

    # Handle remaining training tokens from buffer
    if tokens_written < args.train_tokens and buf:
        remaining_needed = args.train_tokens - tokens_written
        shard_tokens = buf[:remaining_needed]
        buf = buf[remaining_needed:]

        if shard_tokens:
            shard_path = out_dir / f"shard_{current_shard_idx:05d}.bin"
            data = np.array(shard_tokens, dtype=np.uint16)
            assert (data < 65536).all()
            data.tofile(shard_path)
            sha256 = compute_file_sha256(shard_path)
            shards_info.append(
                {
                    "file": f"shard_{current_shard_idx:05d}.bin",
                    "sha256": sha256,
                    "tokens": len(shard_tokens),
                }
            )
            tokens_written += len(shard_tokens)
            current_shard_idx += 1

    # Collect holdout tokens from remaining buffer + stream
    holdout_tokens: list[int] = buf.copy()
    print(
        f"Collected {len(holdout_tokens)} tokens from carry buffer, "
        f"reading holdout from stream...",
        flush=True,
    )

    # Continue streaming for holdout
    for doc in stream:
        text = doc.get("text", "")
        if not text:
            continue

        ids = tok(text, add_special_tokens=False).input_ids
        holdout_tokens.extend(ids)
        holdout_tokens.append(eos_id)

        if len(holdout_tokens) >= args.holdout_tokens:
            break

    # Write holdout
    holdout_data = holdout_tokens[: args.holdout_tokens]
    holdout_path = out_dir / "holdout.bin"
    data = np.array(holdout_data, dtype=np.uint16)
    assert (data < 65536).all()
    data.tofile(holdout_path)
    holdout_sha256 = compute_file_sha256(holdout_path)

    print(f"Wrote {len(holdout_data)} holdout tokens", flush=True)

    # Compute pack hash from shard hashes
    pack_hash_input = "".join(s["sha256"] for s in shards_info)
    pack_hash = hashlib.sha256(pack_hash_input.encode()).hexdigest()

    # Write manifest
    manifest = {
        "dataset": dataset_name,
        "dataset_config": "sample-100BT" if not args.local_texts_file else None,
        "tokenizer": "gpt2",
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "dtype": "uint16",
        "shard_tokens": args.shard_tokens,
        "shards": shards_info,
        "holdout": {
            "file": "holdout.bin",
            "sha256": holdout_sha256,
            "tokens": len(holdout_data),
        },
        "total_train_tokens": tokens_written,
        "pack_hash": pack_hash,
    }
    if args.date:
        manifest["created"] = args.date

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t_start
    print(
        f"\nPack complete: {tokens_written/1e9:.2f}B train tokens, "
        f"{len(holdout_data)/1e6:.1f}M holdout tokens, "
        f"{len(shards_info)} shards, "
        f"{elapsed/3600:.1f}h elapsed",
        flush=True,
    )
    print(f"Manifest written to {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
