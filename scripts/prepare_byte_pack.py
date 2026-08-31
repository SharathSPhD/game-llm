"""Build a byte-level data pack for SPEC 0023 by decoding an existing GPT-2 pack.

GPT-2 BPE is byte-reversible, so the byte-level content is identical to the
source pack by construction. Documents are split at GPT-2 EOS, decoded to text,
encoded to UTF-8 bytes, with NUL bytes (0x00) stripped from text and used as
document separators instead.

Output format: uint8 shards in the same manifest/shard layout as the input pack,
with dtype field set to "uint8". Budgets are specified in bytes rather than tokens.

Resumability is not required; the full pack build takes ~1-2 hours and is
re-runnable from scratch.
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
from transformers import AutoTokenizer


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


def load_gpt2_pack_manifest(pack_dir: Path) -> dict[str, Any]:
    """Load manifest from a GPT-2 pack."""
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found in {pack_dir}")
    return cast(dict[str, Any], json.loads(manifest_path.read_text()))


def stream_gpt2_ids(pack_dir: Path, manifest: dict[str, Any]) -> Any:
    """Stream token ids from GPT-2 pack shards in order.

    Yields individual token ids from all training shards.
    """
    for shard_info in manifest["shards"]:
        shard_path = pack_dir / shard_info["file"]
        if not shard_path.exists():
            raise SystemExit(f"Shard file missing: {shard_path}")
        data = np.fromfile(shard_path, dtype=np.uint16)
        for token_id in data:
            yield int(token_id)


def decode_and_encode_batch(
    ids_batch: list[int],
    tokenizer: AutoTokenizer,
) -> tuple[list[bytes], int]:
    """Decode a batch of token ids to bytes.

    Args:
        ids_batch: List of GPT-2 token ids.
        tokenizer: GPT-2 tokenizer.

    Returns:
        Tuple of (byte chunks per doc, total nul bytes stripped).
    """
    # Split batch by documents (EOS id = 50256)
    eos_id = tokenizer.eos_token_id
    docs_in_batch: list[list[int]] = [[]]

    for token_id in ids_batch:
        if token_id == eos_id:
            docs_in_batch.append([])
        else:
            docs_in_batch[-1].append(token_id)

    # Decode all docs in the batch
    doc_ids_list = [d for d in docs_in_batch if d]  # Skip empty docs
    if not doc_ids_list:
        return [], 0

    # Batch decode for speed
    decoded_texts = tokenizer.batch_decode(
        doc_ids_list,
        skip_special_tokens=True,
    )

    result_bytes: list[bytes] = []
    nul_stripped = 0

    for text in decoded_texts:
        # Strip NUL bytes (0x00) from text before encoding
        text_no_nul = text.replace("\x00", "")
        nul_stripped += len(text) - len(text_no_nul)

        # Encode to UTF-8
        doc_bytes = text_no_nul.encode("utf-8")

        result_bytes.append(doc_bytes)

    return result_bytes, nul_stripped


def main() -> int:
    """Build the byte-level data pack."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    ap = argparse.ArgumentParser(
        description="Build a byte-level data pack from a GPT-2 pack for SPEC 0023."
    )
    ap.add_argument(
        "--input-pack",
        default="data/pack_1b",
        help="Input directory containing GPT-2 pack",
    )
    ap.add_argument(
        "--out-dir",
        default="data/pack_byte",
        help="Output directory for byte pack shards and manifest",
    )
    ap.add_argument(
        "--train-bytes",
        type=int,
        default=5_000_000_000,
        help="Number of training bytes to produce",
    )
    ap.add_argument(
        "--holdout-bytes",
        type=int,
        default=20_000_000,
        help="Number of holdout bytes (from stream tail)",
    )
    ap.add_argument(
        "--shard-bytes",
        type=int,
        default=500_000_000,
        help="Bytes per shard file",
    )
    ap.add_argument(
        "--batch-docs",
        type=int,
        default=512,
        help="Documents per tokenizer batch",
    )
    args = ap.parse_args()

    input_pack = Path(args.input_pack)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GPT-2 pack manifest
    print("Loading GPT-2 pack manifest...", flush=True)
    gpt2_manifest = load_gpt2_pack_manifest(input_pack)
    source_pack_hash = gpt2_manifest["pack_hash"]

    # Load tokenizer
    print("Loading GPT-2 tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    eos_id = tok.eos_token_id
    print(f"GPT-2 EOS ID: {eos_id}", flush=True)

    # Manifest accumulator for byte pack
    shards_info: list[dict[str, Any]] = []
    nul_bytes_total = 0

    # Main decoding loop
    print("Beginning decoding and packing...", flush=True)
    t_start = time.time()
    current_shard_idx = 0
    buf: list[int] = []  # Buffer for bytes

    # Stream GPT-2 ids and decode
    train_complete = False
    holdout_complete = False
    train_bytes_written = 0
    holdout_bytes: list[int] = []
    ids_batch: list[int] = []
    docs_in_batch = 0

    for token_id in stream_gpt2_ids(input_pack, gpt2_manifest):
        # Stop if both budgets are met
        if train_complete and holdout_complete:
            break

        # Accumulate tokens for batch processing
        ids_batch.append(token_id)
        if token_id == eos_id:
            docs_in_batch += 1

        # Process batch when we have enough documents or if end of stream is near
        if docs_in_batch >= args.batch_docs:
            # Decode batch
            doc_bytes_list, nul_stripped = decode_and_encode_batch(ids_batch, tok)
            nul_bytes_total += nul_stripped

            # Add documents to appropriate buffer
            for doc_bytes in doc_bytes_list:
                if not train_complete:
                    buf.extend(doc_bytes)
                    buf.append(0)  # Separator byte
                    if len(buf) >= args.shard_bytes:
                        # Write shard
                        shard_tokens = buf[: args.shard_bytes]
                        buf = buf[args.shard_bytes :]

                        shard_path = out_dir / f"shard_{current_shard_idx:05d}.bin"
                        data = np.array(shard_tokens, dtype=np.uint8)
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
                        train_bytes_written += len(shard_tokens)

                        if train_bytes_written % (250_000_000) < args.shard_bytes:
                            elapsed = time.time() - t_start
                            bytes_per_sec = train_bytes_written / max(elapsed, 1e-6)
                            print(
                                f"  {train_bytes_written/1e9:.2f}B bytes written, "
                                f"{bytes_per_sec/1e6:.2f}M bytes/s",
                                flush=True,
                            )

                        if train_bytes_written >= args.train_bytes:
                            train_complete = True
                            break
                else:
                    # Collect holdout bytes
                    holdout_bytes.extend(doc_bytes)
                    holdout_bytes.append(0)  # Separator byte
                    if len(holdout_bytes) >= args.holdout_bytes:
                        holdout_complete = True
                        break

            ids_batch = []
            docs_in_batch = 0

    # Handle remaining training bytes from buffer
    if train_bytes_written < args.train_bytes and buf:
        remaining_needed = args.train_bytes - train_bytes_written
        shard_tokens = buf[:remaining_needed]
        buf = buf[remaining_needed:]

        if shard_tokens:
            shard_path = out_dir / f"shard_{current_shard_idx:05d}.bin"
            data = np.array(shard_tokens, dtype=np.uint8)
            data.tofile(shard_path)
            sha256 = compute_file_sha256(shard_path)
            shards_info.append(
                {
                    "file": f"shard_{current_shard_idx:05d}.bin",
                    "sha256": sha256,
                    "tokens": len(shard_tokens),
                }
            )
            train_bytes_written += len(shard_tokens)
            current_shard_idx += 1

    # Process any remaining batched documents
    if ids_batch:
        doc_bytes_list, nul_stripped = decode_and_encode_batch(ids_batch, tok)
        nul_bytes_total += nul_stripped

        for doc_bytes in doc_bytes_list:
            if not train_complete:
                buf.extend(doc_bytes)
                buf.append(0)  # Separator byte
                if len(buf) >= args.shard_bytes:
                    shard_tokens = buf[: args.shard_bytes]
                    buf = buf[args.shard_bytes :]

                    shard_path = out_dir / f"shard_{current_shard_idx:05d}.bin"
                    data = np.array(shard_tokens, dtype=np.uint8)
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
                    train_bytes_written += len(shard_tokens)

                if train_bytes_written >= args.train_bytes:
                    train_complete = True
            else:
                holdout_bytes.extend(doc_bytes)
                holdout_bytes.append(0)  # Separator byte
                if len(holdout_bytes) >= args.holdout_bytes:
                    holdout_complete = True
                    break

    # Write final training shard if buffer has remaining bytes
    if buf and not train_complete:
        shard_path = out_dir / f"shard_{current_shard_idx:05d}.bin"
        data = np.array(buf, dtype=np.uint8)
        data.tofile(shard_path)
        sha256 = compute_file_sha256(shard_path)
        shards_info.append(
            {
                "file": f"shard_{current_shard_idx:05d}.bin",
                "sha256": sha256,
                "tokens": len(buf),
            }
        )
        train_bytes_written += len(buf)
        current_shard_idx += 1

    # Write holdout
    holdout_data = holdout_bytes[: args.holdout_bytes]
    holdout_path = out_dir / "holdout.bin"
    data = np.array(holdout_data, dtype=np.uint8)
    data.tofile(holdout_path)
    holdout_sha256 = compute_file_sha256(holdout_path)

    print(f"Wrote {len(holdout_data)} holdout bytes", flush=True)

    # Compute pack hash from shard hashes
    pack_hash_input = "".join(s["sha256"] for s in shards_info)
    pack_hash = hashlib.sha256(pack_hash_input.encode()).hexdigest()

    # Write manifest
    manifest = {
        "dataset": "fineweb-edu-bytes-from-pack",
        "tokenizer": "byte-256",
        "vocab_size": 256,
        "eos_id": 0,
        "dtype": "uint8",
        "shard_bytes": args.shard_bytes,
        "shards": shards_info,
        "holdout": {
            "file": "holdout.bin",
            "sha256": holdout_sha256,
            "tokens": len(holdout_data),
        },
        "total_train_tokens": train_bytes_written,
        "pack_hash": pack_hash,
        "source_pack_hash": source_pack_hash,
        "nul_bytes_stripped": nul_bytes_total,
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t_start
    print(
        f"\nByte pack complete: {train_bytes_written/1e9:.2f}B train bytes, "
        f"{len(holdout_data)/1e6:.1f}M holdout bytes, "
        f"{len(shards_info)} shards, "
        f"{nul_bytes_total} nul bytes stripped, "
        f"{elapsed/3600:.1f}h elapsed",
        flush=True,
    )
    print(f"Manifest written to {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
