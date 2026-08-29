"""The KD pilot (SPEC 0021): does distillation accelerate a from-scratch tied model?

Everything downstream — a month of compute, possibly money — hangs on one
number: how much a frozen teacher's logits speed up a from-scratch tied student
against plain cross-entropy at identical tokens. The Minitron line reports
order-of-magnitude token savings for explicit students; nothing published says
whether that transfers to a weight-tied fixed-point student, whose training
dynamics F24 showed to be regime-sensitive. So the pilot measures it on this
stack before the month is committed.

Two arms, one student architecture, identical data order (same seed, same
stream), identical steps: cross-entropy alone, then cross-entropy plus a KL term
against the teacher's temperature-softened logits. The gate, fixed in SPEC 0021
before this file ran: the KD arm must cut held-out perplexity by at least 15%
relative to the CE arm, or the leap closes at F51's verdict.

The student uses the teacher's tokenizer, which is what makes logit
distillation well-defined, and the anytime unrolled supervision of F24, which is
the training regime under which tying is known to work.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.eqlm import EqLM, EqLMConfig  # noqa: E402

TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"
SUP_AT = [4, 8, 12]
SUP_W = [0.15, 0.3, 1.0]


CACHE_FILE = "data/cache/kd_fineweb_qwen.pt"
_CACHE: dict[str, torch.Tensor] = {}


def stream_batches(
    tok: Any, seq_len: int, batch: int, device: str, seed: int
) -> Any:
    """Packed token batches from the pre-built cache, deterministic in order.

    Streaming from the Hub mid-training stalled inside the container, so the
    corpus is materialised once by scripts/prepare_kd_cache.py and read here as
    a flat tensor. Both arms draw the same shuffled window order from the same
    seed, so the comparison sees byte-identical data. The held-out seed (999)
    reserves the final windows, which the training seeds never reach.
    """
    if "t" not in _CACHE:
        _CACHE["t"] = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
    t = _CACHE["t"]
    span = seq_len * batch
    n_windows = t.numel() // span
    g = torch.Generator().manual_seed(seed)
    if seed == 999:
        order = torch.arange(n_windows - 64, n_windows)
    else:
        order = torch.randperm(n_windows - 64, generator=g)
    for idx in order.tolist():
        chunk = t[idx * span : (idx + 1) * span].to(device=device, dtype=torch.long)
        yield chunk.view(batch, seq_len)


@torch.no_grad()
def held_out_ppl(model: EqLM, batches: list[torch.Tensor]) -> float:
    total, count = 0.0, 0
    for ids in batches:
        outs = model.forward_unrolled(ids, supervise_at=[SUP_AT[-1]])
        logits = outs[-1][1]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        )
        total += float(loss) * ids.numel()
        count += ids.numel()
    return math.exp(total / max(count, 1))


def train_arm(
    kd: bool,
    steps: int,
    seq_len: int,
    batch: int,
    device: str,
    seed: int,
    heldout: list[torch.Tensor],
    teacher: Any,
    tok: Any,
    kd_weight: float,
    kd_temp: float,
    log: list[dict[str, Any]],
) -> tuple[EqLM, float]:
    torch.manual_seed(seed)
    cfg = EqLMConfig(
        vocab_size=len(tok), d_model=768, n_heads=12, d_ff=3072,
        max_seq_len=seq_len, deq_max_iter=12, solver="anderson",
        map_form="postln", spectral_norm=True, residual_damping=0.2,
        dropout=0.1,
    )
    model = EqLM(config=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    stream = stream_batches(tok, seq_len, batch, device, seed)
    t0 = time.time()
    for step in range(steps):
        ids = next(stream)
        outs = model.forward_unrolled(ids, supervise_at=SUP_AT)
        parts = []
        for w, (_, lg) in zip(SUP_W, outs, strict=True):
            ce = F.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1)
            )
            parts.append(w * ce)
        loss = torch.stack(parts).sum() / sum(SUP_W)

        if kd:
            with torch.no_grad():
                # Qwen pads its output head beyond the tokenizer (151936 vs
                # 151665); the padded ids are never produced by real text, so
                # slicing to the student's vocabulary and renormalising under
                # log_softmax loses nothing the data can express.
                t_logits = teacher(ids).logits[:, :-1, : outs[-1][1].shape[-1]]
            s_logits = outs[-1][1][:, :-1]
            # KL between temperature-softened distributions, scaled by T^2 in
            # the standard way so gradients keep their magnitude as T varies.
            # Flattened to (tokens, vocab) so batchmean yields per-token KL,
            # the same scale as cross-entropy; on [B, T, V] batchmean divides by
            # batch alone and the KD term dwarfs CE by the sequence length.
            v = s_logits.shape[-1]
            t_lp = F.log_softmax(t_logits / kd_temp, dim=-1).reshape(-1, v)
            s_lp = F.log_softmax(s_logits / kd_temp, dim=-1).reshape(-1, v)
            kl = F.kl_div(s_lp, t_lp, log_target=True, reduction="batchmean")
            loss = loss + kd_weight * (kd_temp**2) * kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log.append({"arm": "kd" if kd else "ce", "step": step,
                        "loss": round(float(loss), 4),
                        "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    ppl = held_out_ppl(model.eval(), heldout)
    return model, ppl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/exp38_kd_pilot.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    tok = AutoTokenizer.from_pretrained(TEACHER)
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER, dtype=torch.bfloat16, device_map=device
    ).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Held-out batches drawn from a disjoint seed so neither arm trains on them.
    heldout = []
    ho = stream_batches(tok, args.seq_len, args.batch, device, seed=999)
    for _ in range(8):
        heldout.append(next(ho))

    tokens = args.steps * args.seq_len * args.batch
    log: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "teacher": TEACHER,
        "tokens_per_arm": tokens,
        "steps": args.steps,
        "gate": "KD must reduce held-out ppl by >= 15% vs CE (SPEC 0021)",
    }

    _, ce_ppl = train_arm(False, args.steps, args.seq_len, args.batch, device,
                          42, heldout, teacher, tok, args.kd_weight,
                          args.kd_temp, log)
    report["ce_ppl"] = round(ce_ppl, 3)
    print(f"CE arm held-out ppl: {ce_ppl:.2f}", flush=True)

    _, kd_ppl = train_arm(True, args.steps, args.seq_len, args.batch, device,
                          42, heldout, teacher, tok, args.kd_weight,
                          args.kd_temp, log)
    report["kd_ppl"] = round(kd_ppl, 3)
    reduction = 1.0 - kd_ppl / ce_ppl
    report["ppl_reduction"] = round(reduction, 4)
    report["gate_passed"] = bool(reduction >= 0.15)
    report["log"] = log
    print(f"KD arm held-out ppl: {kd_ppl:.2f}  reduction {reduction:.1%}  "
          f"gate {'PASS' if report['gate_passed'] else 'FAIL'}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
