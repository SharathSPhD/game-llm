"""What the equilibrium formulation actually saves: memory (cycle 32).

F45 established the exchange rate at equal compute — the tied block reaches 0.958
of an explicit transformer's quality with 2.70 times fewer parameters. That is a
statement about storage, and storage is where a fixed-point model's advantage
should be structural rather than incidental: an explicit stack must hold the
weights of every layer, while a tied block holds one set and revisits it, and an
explicit stack's activation memory during a forward pass grows with depth while
an equilibrium solve can overwrite its iterate in place.

This file measures both, because a memory claim asserted from architecture is
worth nothing next to a memory claim measured on a device. Three quantities are
reported for each model: resident weight memory, peak allocated memory during a
forward pass at several sequence lengths and batch sizes, and the largest batch
that fits inside a stated budget. The last is the one a practitioner deploying to
a small device actually needs, and it is the one an architecture diagram cannot
tell them.

The comparison is like for like: the same evaluation, the same sequence lengths,
the same dtype, on the same card, with the tied model at the width that makes its
per-iteration compute equal to an explicit layer's, so nothing is traded silently
for the memory being saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp31_adaptive_depth import load_model  # noqa: E402


def _reset(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def weight_bytes(model: torch.nn.Module) -> int:
    """Bytes held by parameters and buffers — what must be resident to serve."""
    p = sum(t.numel() * t.element_size() for t in model.parameters())
    b = sum(t.numel() * t.element_size() for t in model.buffers())
    return p + b


@torch.no_grad()
def peak_forward_bytes(
    model: torch.nn.Module, batch: int, seq: int, device: str, vocab: int
) -> int | None:
    """Peak allocation during one forward pass, weights excluded.

    Measured as the difference between peak and the baseline already resident,
    so the number is the transient cost of running the model rather than the
    cost of having loaded it. Returns None if the configuration does not fit,
    which is itself the answer for a memory-limited device.
    """
    _reset(device)
    base = torch.cuda.memory_allocated() if device.startswith("cuda") else 0
    ids = torch.randint(0, max(vocab - 1, 1), (batch, seq), device=device)
    try:
        model(ids)
    except torch.cuda.OutOfMemoryError:
        _reset(device)
        return None
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    _reset(device)
    return int(peak - base)


def largest_batch(
    model: torch.nn.Module, seq: int, device: str, vocab: int, budget_bytes: int
) -> int:
    """Largest batch whose forward pass fits inside a stated budget.

    Doubling then bisecting, because the quantity a deployer cares about is a
    threshold rather than a curve, and the threshold is what decides whether a
    model runs on a given device at all.
    """
    lo, hi = 0, 1
    while hi <= 4096:
        got = peak_forward_bytes(model, hi, seq, device, vocab)
        if got is None or got > budget_bytes:
            break
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        got = peak_forward_bytes(model, mid, seq, device, vocab)
        if got is not None and got <= budget_bytes:
            lo = mid
        else:
            hi = mid
    return lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqlm", required=True)
    ap.add_argument("--explicit", required=True)
    ap.add_argument("--out", default="results/scale/exp35_memory_profile.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget-mb", type=int, default=512,
                    help="a small-device activation budget, in megabytes")
    args = ap.parse_args()

    device = args.device
    report: dict[str, Any] = {"device": device, "budget_mb": args.budget_mb, "models": {}}

    for tag, path in (("eqlm_tied", args.eqlm), ("explicit", args.explicit)):
        model = load_model(Path(path), device)
        vocab = int(model.config.vocab_size)
        entry: dict[str, Any] = {
            "checkpoint": path,
            "class": type(model).__name__,
            "d_model": int(model.config.d_model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "weight_bytes": weight_bytes(model),
            "forward_peak_bytes": {},
        }
        # Sequence lengths are capped at what the checkpoint was trained to
        # support; exceeding max_seq_len indexes past the position table and
        # fails as a device-side assert rather than a clear error.
        max_seq = int(getattr(model.config, "max_seq_len", 128))
        for seq in [s for s in (64, 128, 256) if s <= max_seq]:
            for batch in (1, 4, 16):
                got = peak_forward_bytes(model, batch, seq, device, vocab)
                entry["forward_peak_bytes"][f"b{batch}_s{seq}"] = got
        probe_seq = min(128, max_seq)
        entry["probe_seq"] = probe_seq
        entry["largest_batch_at_budget"] = {
            f"seq{probe_seq}": largest_batch(
                model, probe_seq, device, vocab, args.budget_mb * 1024**2
            )
        }
        report["models"][tag] = entry
        print(f"{tag}: weights {entry['weight_bytes']/1e6:.1f}MB, "
              f"largest batch {list(entry['largest_batch_at_budget'].values())[0]}",
              flush=True)
        del model
        _reset(device)

    e, x = report["models"]["eqlm_tied"], report["models"]["explicit"]
    report["comparison"] = {
        "weight_memory_ratio": round(e["weight_bytes"] / x["weight_bytes"], 4),
        "weight_memory_saved_mb": round((x["weight_bytes"] - e["weight_bytes"]) / 1e6, 1),
        "forward_peak_ratio": {
            k: (round(e["forward_peak_bytes"][k] / x["forward_peak_bytes"][k], 4)
                if e["forward_peak_bytes"].get(k) and x["forward_peak_bytes"].get(k)
                else None)
            for k in set(e["forward_peak_bytes"]) & set(x["forward_peak_bytes"])
        },
        "batch_ratio_at_budget": round(
            list(e["largest_batch_at_budget"].values())[0]
            / max(list(x["largest_batch_at_budget"].values())[0], 1), 3
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
