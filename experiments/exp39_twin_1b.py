"""The twin at 1B (SPEC 0022): compute-matched tying at deployment scale.

One trainer serves both arms so the comparison cannot drift: the explicit
sixteen-layer transformer and the tied block iterated sixteen times share this
file's data order, schedule, precision, and checkpoint discipline, and differ
only in which forward function the arm name selects. The tied arm trains under
the anytime-unrolled regime of F24 — supervision at three depths with the
final depth dominant — because that is the regime under which tying is known
to work; its extra head evaluations (~13% training FLOPs) are disclosed in the
spec and charged to the tied arm.

Scale mechanics, fixed here and in the spec before any 1B step ran: bf16
autocast over fp32 master weights; per-block gradient checkpointing so the
32GB card holds sixteen levels of depth at sequence length 2048; a
warmup-stable-decay schedule in token space so both arms see identical
learning rates at identical tokens during the twin phase while the tied arm's
extension can anneal later without retroactively unbalancing the comparison;
data consumed as fixed 4x2048-token units from one shuffled order so arms with
different micro-batch sizes still read byte-identical streams; checkpoints
every 500M tokens with optimizer state, data cursor, RNG, and the pack hash,
so a resume is a continuation and a checkpoint can never silently resume onto
a different pack.

The preflight mode exists because extrapolated throughput is how schedules
die: it measures median post-warmup tokens/sec at the real geometry, exercises
a save/resume round trip, and projects wall-clock for the registered budgets.
The GO rule lives in SPEC 0022, not here.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.data.pack import PackReader, window_order  # noqa: E402
from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM  # noqa: E402

#: Data is consumed in fixed units of UNIT_BATCH rows so both arms follow the
#: identical shuffled stream regardless of their micro-batch sizes.
UNIT_BATCH = 4

SUP_WEIGHTS = [0.15, 0.3, 1.0]

MILESTONES_DEFAULT = [
    500_000_000, 1_000_000_000, 2_500_000_000,
    5_000_000_000, 7_500_000_000, 10_000_000_000,
]


def supervise_depths(depth: int) -> list[int]:
    """The F24 anytime depths scaled to the arm's depth: 6/11/16 at depth 16."""
    a = max(1, round(depth * 0.375))
    b = max(a + 1, round(depth * 0.6875))
    return [min(a, depth), min(b, depth), depth]


def embed(model: Any, ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.long)
    return model.embedding(ids) + model.pos_embedding(positions)


def head(model: Any, z: torch.Tensor) -> torch.Tensor:
    h = model.ln_final(z)
    return model.lm_head(h) / (model.config.d_model**0.5)


def explicit_logits(model: ExplicitLM, ids: torch.Tensor, use_ckpt: bool) -> torch.Tensor:
    """The explicit stack, reimplemented from the model's own modules so the
    per-layer gradient checkpointing the class does not offer can wrap each
    block; the mathematics is the class's forward exactly (x injection zero).
    """
    z = embed(model, ids)
    x0 = torch.zeros_like(z)
    for layer in model.layers:
        z = checkpoint(layer, z, x0, use_reentrant=False) if use_ckpt else layer(z, x0)
    return head(model, z)


def tied_outputs(
    model: EqLM, ids: torch.Tensor, depths: list[int], use_ckpt: bool
) -> list[tuple[int, torch.Tensor]]:
    """forward_unrolled with per-iteration checkpointing; same computation."""
    x = embed(model, ids)
    z = x
    outs: list[tuple[int, torch.Tensor]] = []
    for k in range(1, depths[-1] + 1):
        z = checkpoint(model.block, z, x, use_reentrant=False) if use_ckpt else model.block(z, x)
        if k in depths:
            outs.append((k, head(model, z)))
    return outs


def next_token_ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        ids[:, 1:].reshape(-1),
    )


def arm_loss(
    arm: str, model: Any, ids: torch.Tensor, depths: list[int], use_ckpt: bool
) -> torch.Tensor:
    if arm == "explicit":
        return next_token_ce(explicit_logits(model, ids, use_ckpt), ids)
    outs = tied_outputs(model, ids, depths, use_ckpt)
    parts = [
        w * next_token_ce(lg, ids)
        for w, (_, lg) in zip(SUP_WEIGHTS, outs, strict=True)
    ]
    return torch.stack(parts).sum() / sum(SUP_WEIGHTS)


def lr_at(
    tokens: int, peak: float, floor: float, warmup: int,
    decay_start: int | None, decay_end: int | None,
) -> float:
    """Warmup-stable-decay in token space (identical for both arms while both
    train; only the extension passes a decay window)."""
    if tokens < warmup:
        return peak * (tokens + 1) / warmup
    if decay_start is None or decay_end is None or tokens < decay_start:
        return peak
    if tokens >= decay_end:
        return floor
    frac = (tokens - decay_start) / (decay_end - decay_start)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * frac))


def build_model(arm: str, args: argparse.Namespace, device: str) -> Any:
    cfg = EqLMConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, max_seq_len=args.seq_len, deq_max_iter=args.depth,
        deq_tol=1e-3, solver="anderson", map_form="postln",
        spectral_norm=True, residual_damping=0.2, dropout=0.0, sdpa=args.sdpa,
    )
    torch.manual_seed(args.seed)
    if arm == "explicit":
        model: Any = ExplicitLM(config=cfg, n_layers=args.depth)
    else:
        model = EqLM(config=cfg)
    return model.to(device)


def build_optimizer(model: Any, args: argparse.Namespace) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    fused = "cuda" in args.device and torch.cuda.is_available()
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=fused,
    )


@torch.no_grad()
def held_out_ppl(
    arm: str, model: Any, batches: list[torch.Tensor], depths: list[int],
    device: str, autocast: bool,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for ids_cpu in batches:
        ids = ids_cpu.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            if arm == "explicit":
                logits = explicit_logits(model, ids, use_ckpt=False)
            else:
                logits = tied_outputs(model, ids, depths, use_ckpt=False)[-1][1]
            loss = next_token_ce(logits, ids)
        n = ids[:, 1:].numel()
        total += float(loss) * n
        count += n
    model.train()
    return math.exp(total / max(count, 1))


def save_state(
    path: Path, model: Any, opt: torch.optim.AdamW, args: argparse.Namespace,
    tokens_seen: int, cursor: int, pack_hash: str,
) -> None:
    """Atomic write: the previous latest survives any crash mid-save."""
    blob: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config_dict": dataclasses.asdict(model.config),
        "model_class": type(model).__name__,
        "opt_state": opt.state_dict(),
        "tokens_seen": tokens_seen,
        "cursor": cursor,
        "pack_hash": pack_hash,
        "arm": args.arm,
        "order_seed": args.seed,
        "torch_rng": torch.get_rng_state(),
    }
    if isinstance(model, ExplicitLM):
        blob["n_layers"] = len(model.layers)
    if torch.cuda.is_available() and "cuda" in args.device:
        blob["cuda_rng"] = torch.cuda.get_rng_state(args.device)
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_state(
    path: Path, model: Any, opt: torch.optim.AdamW, args: argparse.Namespace,
    pack_hash: str,
) -> tuple[int, int]:
    blob = torch.load(path, map_location=args.device, weights_only=True)
    if blob["pack_hash"] != pack_hash:
        raise SystemExit(
            f"checkpoint {path} was trained on pack {blob['pack_hash'][:12]}, "
            f"but the mounted pack is {pack_hash[:12]}; a resume across packs "
            "would silently change the data and is refused"
        )
    if blob["arm"] != args.arm:
        raise SystemExit(f"checkpoint is arm {blob['arm']!r}, not {args.arm!r}")
    model.load_state_dict(blob["state_dict"])
    opt.load_state_dict(blob["opt_state"])
    torch.set_rng_state(blob["torch_rng"].cpu())
    if "cuda_rng" in blob and torch.cuda.is_available() and "cuda" in args.device:
        torch.cuda.set_rng_state(blob["cuda_rng"].cpu(), args.device)
    return int(blob["tokens_seen"]), int(blob["cursor"])


def main() -> int:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--arm", required=True, choices=["explicit", "tied"])
    ap.add_argument("--pack-dir", default="data/pack_1b")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--target-tokens", type=int, default=2_500_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--depth", type=int, default=16,
                    help="explicit layers / tied iterations (compute-matched)")
    ap.add_argument("--vocab-size", type=int, default=50304,
                    help="model vocab, padded past GPT-2's 50257 to a multiple "
                         "of 64 for matmul throughput; ids never reach the pad")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="rows per forward; default 8 explicit / 4 tied "
                         "(the tied arm holds three supervision logit tensors)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="default sized so one step is ~1.05M tokens")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-tokens", type=int, default=250_000_000)
    ap.add_argument("--decay-start-tokens", type=int, default=None)
    ap.add_argument("--decay-end-tokens", type=int, default=None)
    ap.add_argument("--ckpt-tokens", type=int, default=500_000_000)
    ap.add_argument("--milestones", type=int, nargs="*", default=MILESTONES_DEFAULT)
    ap.add_argument("--heldout-batches", type=int, default=32)
    ap.add_argument("--resume", default="auto",
                    help="'auto' resumes ckpt_latest.pt in out-dir if present; "
                         "'none' starts fresh; else a checkpoint path")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--no-sdpa", dest="sdpa", action="store_false")
    ap.add_argument("--preflight", type=int, default=0,
                    help="run N optimizer steps, measure median tok/s, test "
                         "save/resume, write a report, and exit")
    ap.add_argument("--preflight-accum", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    if args.micro_batch is None:
        args.micro_batch = 8 if args.arm == "explicit" else 4
    if args.micro_batch % UNIT_BATCH != 0:
        raise SystemExit(f"--micro-batch must be a multiple of {UNIT_BATCH}")
    if args.grad_accum is None:
        args.grad_accum = max(1, round(1_048_576 / (args.micro_batch * args.seq_len)))
    if args.preflight:
        args.grad_accum = args.preflight_accum

    device = args.device
    use_cuda = "cuda" in device and torch.cuda.is_available()
    use_ckpt = not args.no_grad_checkpoint
    depths = supervise_depths(args.depth)

    out_dir = Path(args.out_dir or f"results/scale/exp39/{args.arm}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = PackReader(args.pack_dir)
    if reader.manifest["vocab_size"] > args.vocab_size:
        raise SystemExit("pack vocab exceeds model vocab")
    n_units = reader.n_windows(args.seq_len, UNIT_BATCH)
    order = window_order(n_units, args.seed)
    heldout = reader.holdout_batches(args.seq_len, UNIT_BATCH, args.heldout_batches)

    model = build_model(args.arm, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optimizer(model, args)

    tokens_seen, cursor = 0, 0
    latest = out_dir / "ckpt_latest.pt"
    if args.resume == "auto" and latest.exists():
        tokens_seen, cursor = load_state(latest, model, opt, args, reader.pack_hash)
        print(f"resumed at {tokens_seen/1e9:.3f}B tokens (cursor {cursor})", flush=True)
    elif args.resume not in ("auto", "none"):
        tokens_seen, cursor = load_state(Path(args.resume), model, opt, args, reader.pack_hash)
        print(f"resumed {args.resume} at {tokens_seen/1e9:.3f}B tokens", flush=True)

    log_path = out_dir / "train_log.jsonl"
    milestones_path = out_dir / "milestones.jsonl"
    done_milestones = set()
    if milestones_path.exists():
        for line in milestones_path.read_text().splitlines():
            done_milestones.add(json.loads(line)["milestone_tokens"])

    print(json.dumps({
        "arm": args.arm, "params": n_params, "depth": args.depth,
        "d_model": args.d_model, "seq_len": args.seq_len,
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
        "tokens_per_step": args.micro_batch * args.seq_len * args.grad_accum,
        "target_tokens": args.target_tokens, "pack_hash": reader.pack_hash[:12],
        "grad_checkpoint": use_ckpt, "sdpa": args.sdpa, "n_units": n_units,
    }), flush=True)

    micro_units = args.micro_batch // UNIT_BATCH

    def next_micro() -> torch.Tensor | None:
        nonlocal cursor
        if cursor + micro_units > n_units:
            return None
        idxs = order[cursor : cursor + micro_units].tolist()
        cursor += micro_units
        rows = [reader.window(i, args.seq_len, UNIT_BATCH) for i in idxs]
        return torch.cat(rows, dim=0).to(device, non_blocking=True)

    model.train()
    t_start = time.time()
    next_ckpt = (tokens_seen // args.ckpt_tokens + 1) * args.ckpt_tokens
    ema_loss: float | None = None
    micro_times: list[float] = []
    step = 0
    step_loss = 0.0

    def emergency(reason: str) -> None:
        save_state(out_dir / "ckpt_emergency.pt", model, opt, args,
                   tokens_seen, cursor, reader.pack_hash)
        print(f"ABORT: {reason} at {tokens_seen} tokens; emergency checkpoint "
              f"written", flush=True)

    while tokens_seen < args.target_tokens:
        lr = lr_at(tokens_seen, args.lr, args.lr_min, args.warmup_tokens,
                   args.decay_start_tokens, args.decay_end_tokens)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            ids = next_micro()
            if ids is None:
                print("pack exhausted before target tokens", flush=True)
                save_state(latest, model, opt, args, tokens_seen, cursor,
                           reader.pack_hash)
                return 0
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
                loss = arm_loss(args.arm, model, ids, depths, use_ckpt)
            if not torch.isfinite(loss):
                emergency(f"non-finite loss {float(loss)}")
                return 3
            (loss / args.grad_accum).backward()
            if use_cuda:
                torch.cuda.synchronize(device)
            micro_times.append(time.time() - t0)
            if len(micro_times) > 10_000:
                del micro_times[:5_000]
            step_loss += float(loss.detach()) / args.grad_accum
            tokens_seen += ids.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1

        ema_loss = step_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * step_loss
        if step > 100 and step_loss > 2.0 * ema_loss:
            print(f"WARN loss spike: {step_loss:.3f} vs ema {ema_loss:.3f} "
                  f"at {tokens_seen} tokens", flush=True)

        if step % args.log_every == 0 or args.preflight:
            recent = micro_times[-args.grad_accum * args.log_every:]
            tok_s = (args.micro_batch * args.seq_len) / (
                sum(recent) / max(len(recent), 1))
            entry = {
                "step": step, "tokens": tokens_seen,
                "loss": round(step_loss, 4), "lr": round(lr, 8),
                "tok_s": round(tok_s, 1),
                "mem_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                if use_cuda else 0.0,
                "min": round((time.time() - t_start) / 60, 1),
            }
            with log_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(entry, flush=True)

        if args.preflight and step >= args.preflight:
            return preflight_report(args, model, opt, reader, heldout, depths,
                                    micro_times, out_dir, tokens_seen, cursor)

        if tokens_seen >= next_ckpt:
            save_state(latest, model, opt, args, tokens_seen, cursor,
                       reader.pack_hash)
            for m in args.milestones:
                if tokens_seen >= m and m not in done_milestones:
                    keep = out_dir / f"ckpt_{m}.pt"
                    shutil.copy2(latest, keep)
                    ppl = held_out_ppl(args.arm, model, heldout, depths,
                                       device, autocast=use_cuda)
                    rec = {"milestone_tokens": m, "tokens_seen": tokens_seen,
                           "heldout_ppl": round(ppl, 4), "arm": args.arm,
                           "checkpoint": str(keep)}
                    with milestones_path.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"MILESTONE {rec}", flush=True)
                    done_milestones.add(m)
            next_ckpt += args.ckpt_tokens

        if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
            save_state(latest, model, opt, args, tokens_seen, cursor,
                       reader.pack_hash)
            print(f"paused at {tokens_seen} tokens (--max-hours reached)",
                  flush=True)
            return 0

    save_state(latest, model, opt, args, tokens_seen, cursor, reader.pack_hash)
    ppl = held_out_ppl(args.arm, model, heldout, depths, device,
                       autocast=use_cuda)
    print(json.dumps({"done": True, "tokens": tokens_seen,
                      "heldout_ppl": round(ppl, 4)}), flush=True)
    return 0


def preflight_report(
    args: argparse.Namespace, model: Any, opt: torch.optim.AdamW,
    reader: PackReader, heldout: list[torch.Tensor], depths: list[int],
    micro_times: list[float], out_dir: Path, tokens_seen: int, cursor: int,
) -> int:
    """Median measured throughput, memory, and a save/resume round trip."""
    warm = micro_times[5:] or micro_times
    warm_sorted = sorted(warm)
    median = warm_sorted[len(warm_sorted) // 2]
    tok_s = args.micro_batch * args.seq_len / median
    ck = out_dir / "ckpt_preflight.pt"
    save_state(ck, model, opt, args, tokens_seen, cursor, reader.pack_hash)
    fresh = build_model(args.arm, args, args.device)
    fresh_opt = build_optimizer(fresh, args)
    load_state(ck, fresh, fresh_opt, args, reader.pack_hash)
    fresh.eval()
    model.eval()
    ids = heldout[0].to(args.device)
    with torch.no_grad():
        if args.arm == "explicit":
            a = explicit_logits(model, ids, use_ckpt=False)
            b = explicit_logits(fresh, ids, use_ckpt=False)
        else:
            a = tied_outputs(model, ids, depths, use_ckpt=False)[-1][1]
            b = tied_outputs(fresh, ids, depths, use_ckpt=False)[-1][1]
    resume_ok = bool(torch.equal(a, b))
    use_cuda = "cuda" in args.device and torch.cuda.is_available()
    report = {
        "arm": args.arm,
        "median_tok_s": round(tok_s, 1),
        "peak_mem_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 2)
        if use_cuda else 0.0,
        "tokens_per_step": args.micro_batch * args.seq_len * args.grad_accum,
        "projected_days": {
            "arm_2.5B": round(2.5e9 / tok_s / 86400, 2),
            "twin_both_arms_2.5B": round(2 * 2.5e9 / tok_s / 86400, 2),
            "extension_7.5B": round(7.5e9 / tok_s / 86400, 2),
        },
        "resume_roundtrip_exact": resume_ok,
        "micro_batch": args.micro_batch,
        "grad_checkpoint": not args.no_grad_checkpoint,
        "sdpa": args.sdpa,
    }
    (out_dir / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if resume_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
