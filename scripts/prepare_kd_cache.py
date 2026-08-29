"""Materialise a bounded FineWeb-Edu token cache for the KD pilot (SPEC 0021).

Streaming from the Hub mid-training stalled the pilot inside the container, so
the data is fetched once here, tokenized with the teacher's tokenizer, and
written as a flat tensor the trainer memory-maps. Determinism comes free: a file
is the same bytes for every arm.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=650_000_000)
    ap.add_argument("--out", default="data/cache/kd_fineweb_qwen.pt")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
        streaming=True,
    )
    buf: list[int] = []
    for i, row in enumerate(ds):
        buf.extend(tok(row["text"]).input_ids)
        buf.append(tok.eos_token_id)
        if i % 2000 == 0:
            print(f"{i} docs, {len(buf)/1e6:.1f}M tokens", flush=True)
        if len(buf) >= args.tokens:
            break
    t = torch.tensor(buf[: args.tokens], dtype=torch.int32)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, args.out)
    print(f"wrote {t.numel()/1e6:.1f}M tokens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
