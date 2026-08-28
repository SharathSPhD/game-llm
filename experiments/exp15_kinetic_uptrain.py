"""Experiment 15 — H7 (SPEC 0011): uptrain a converted KineticLM.

Recovers a block-recursive KineticLM (F25 operating point) toward its base
model using the kinetic objective:

    L = alpha * KL(student || frozen teacher)  +  (1 - alpha) * CE(tokens)
        [+ anytime term: the same loss at a REDUCED recursion depth, applied
         stochastically, so one checkpoint serves several inference budgets —
         the F24/B1 property carried to real scale]

Teacher distillation is what published recursive-uptraining recipes use;
the anytime term is ours. Both are logged separately so their contributions
stay separable.

Usage:
    python exp15_kinetic_uptrain.py --config configs/exp15_kinetic_1p7b.yaml \
        --output results/exp15_kinetic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.kinetic_lm import (  # noqa: E402
    KineticConfig,
    convert_to_kinetic,
    count_unique_params,
)


def get_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    tr = cfg["training"]
    for k in ("lr", "weight_decay", "grad_clip", "alpha_kl", "anytime_prob", "warmup_frac"):
        if isinstance(tr.get(k), str):
            tr[k] = float(tr[k])
    return cfg


def build_token_batches(cfg: dict, tokenizer) -> torch.Tensor:
    """Stream a corpus into fixed-length blocks, cached on disk.

    Cache key covers dataset id/split/config, block size and token budget, so a
    changed corpus can never silently reuse another run's tokens.
    """
    dcfg = cfg["data"]
    seq_len = int(dcfg["seq_len"])
    max_tokens = int(dcfg["max_tokens"])
    key = hashlib.sha256(
        json.dumps(
            {k: dcfg.get(k) for k in ("dataset", "name", "split", "seq_len", "max_tokens")},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache = Path(dcfg.get("cache_dir", "data/scale_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"tokens_{key}.pt"
    if cache_file.exists():
        blob = torch.load(cache_file, weights_only=True)
        print(f"[data] cache hit {cache_file} -> {tuple(blob['tokens'].shape)}", flush=True)
        return blob["tokens"]

    from datasets import load_dataset

    print(f"[data] streaming {dcfg['dataset']} for {max_tokens/1e6:.0f}M tokens...", flush=True)
    ds = load_dataset(
        dcfg["dataset"], name=dcfg.get("name"), split=dcfg.get("split", "train"), streaming=True
    )
    buf: list[int] = []
    eos = tokenizer.eos_token_id
    for row in ds:
        text = row.get("text") or row.get("content") or ""
        if not text:
            continue
        buf.extend(tokenizer(text, add_special_tokens=False).input_ids)
        if eos is not None:
            buf.append(eos)
        if len(buf) >= max_tokens:
            break
    n = (len(buf) // seq_len) * seq_len
    tokens = torch.tensor(buf[:n], dtype=torch.long).view(-1, seq_len)
    torch.save({"tokens": tokens}, cache_file)
    print(f"[data] built {tuple(tokens.shape)} -> {cache_file}", flush=True)
    return tokens


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Token-level forward KL from teacher to student."""
    s = F.log_softmax(student_logits / temp, dim=-1)
    t = F.log_softmax(teacher_logits / temp, dim=-1)
    return F.kl_div(s, t, log_target=True, reduction="batchmean") * (temp**2)


def ce_loss(logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
    )


@torch.no_grad()
def eval_ppl(model, tokens: torch.Tensor, device: str, n_batches: int, bs: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(n_batches):
        batch = tokens[i * bs : (i + 1) * bs].to(device)
        if batch.numel() == 0:
            break
        total += model(batch, labels=batch).loss.item()
        count += 1
    model.train()
    return math.exp(total / max(count, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")
    tr = cfg["training"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokens = build_token_batches(cfg, tokenizer)
    n_hold = int(cfg["data"].get("heldout_batches", 16)) * int(tr["batch_size"])
    heldout, train_tokens = tokens[:n_hold], tokens[n_hold:]

    kinetic_cfg_dict = dict(cfg["kinetic"])
    anytime_depths_cfg = kinetic_cfg_dict.pop("anytime_depths", None)
    kin_cfg = KineticConfig(**kinetic_cfg_dict)
    student = convert_to_kinetic(
        AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16), kin_cfg
    ).to(device)
    n_student = count_unique_params(student)

    teacher = None
    if tr["alpha_kl"] > 0:
        teacher = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    n_base = sum(p.numel() for p in (teacher or student).parameters()) if teacher else None

    if tr.get("gradient_checkpointing", True):
        student.gradient_checkpointing_enable()
        student.config.use_cache = False

    print(
        f"[model] student {n_student/1e9:.3f}B unique params"
        + (f" ({100*n_student/n_base:.0f}% of base)" if n_base else "")
        + f" | depth {student.recursion_depth} | teacher {'on' if teacher else 'off'}",
        flush=True,
    )

    decay = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(decay, lr=tr["lr"], weight_decay=tr["weight_decay"], fused=True)
    total_steps = int(tr["num_steps"])
    warmup = max(1, int(total_steps * tr.get("warmup_frac", 0.02)))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warmup
        if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    bs = int(tr["batch_size"])
    accum = int(tr.get("grad_accum", 1))
    full_depth = student.recursion_depth
    anytime_depths = [
        d for d in (anytime_depths_cfg or [max(1, full_depth // 2)]) if d != full_depth
    ] or [max(1, full_depth // 2)]
    rng = random.Random(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])

    ppl0 = eval_ppl(student, heldout, device, 4, bs)
    print(f"[eval] pre-uptraining held-out ppl {ppl0:.2f}", flush=True)

    log: list[dict] = []
    t0 = time.time()
    student.train()
    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        step_loss = step_kd = step_ce = step_any = 0.0
        for _ in range(accum):
            idx = torch.randint(0, train_tokens.shape[0], (bs,), generator=gen)
            batch = train_tokens[idx].to(device)
            logits = student(batch).logits
            loss_ce = ce_loss(logits, batch)
            loss = (1.0 - tr["alpha_kl"]) * loss_ce
            step_ce += loss_ce.item()
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(batch).logits
                loss_kd = kd_loss(logits, t_logits, tr.get("kd_temp", 1.0))
                loss = loss + tr["alpha_kl"] * loss_kd
                step_kd += loss_kd.item()
            # Anytime term (ours): same objective at a reduced budget.
            if tr["anytime_prob"] > 0 and rng.random() < tr["anytime_prob"]:
                d = rng.choice(anytime_depths)
                student.set_recursion_depth(d)
                try:
                    a_logits = student(batch).logits
                    loss_any = ce_loss(a_logits, batch)
                    loss = loss + tr.get("anytime_weight", 0.3) * loss_any
                    step_any += loss_any.item()
                finally:
                    student.set_recursion_depth(full_depth)
            (loss / accum).backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(student.parameters(), tr["grad_clip"])
        opt.step()
        sched.step()

        if step % int(tr.get("log_every", 20)) == 0 or step == total_steps - 1:
            rec = {
                "step": step,
                "loss": step_loss / accum,
                "ce": step_ce / accum,
                "kd": step_kd / accum,
                "anytime": step_any / accum,
                "lr": sched.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }
            log.append(rec)
            print(
                f"  step {step:5d} loss {rec['loss']:7.3f} ce {rec['ce']:7.3f} "
                f"kd {rec['kd']:6.3f} any {rec['anytime']:6.3f} "
                f"lr {rec['lr']:.2e} {rec['elapsed_s']/60:5.1f}min",
                flush=True,
            )
        if tr.get("eval_every") and step > 0 and step % int(tr["eval_every"]) == 0:
            p = eval_ppl(student, heldout, device, 4, bs)
            print(f"  [eval] step {step} held-out ppl {p:.2f}", flush=True)
            log.append({"step": step, "heldout_ppl": p})
            (out_dir / "log.json").write_text(json.dumps(log, indent=2))
            student.save_pretrained(out_dir / "checkpoint")
            tokenizer.save_pretrained(out_dir / "checkpoint")

    ppl_final = eval_ppl(student, heldout, device, 8, bs)
    budget_ppl = {}
    for d in sorted({full_depth, *anytime_depths}):
        student.set_recursion_depth(d)
        budget_ppl[str(d)] = eval_ppl(student, heldout, device, 4, bs)
    student.set_recursion_depth(full_depth)

    student.save_pretrained(out_dir / "checkpoint")
    tokenizer.save_pretrained(out_dir / "checkpoint")
    results: dict[str, Any] = {
        "experiment": "exp15_kinetic_uptrain",
        "spec": "0011",
        "seed": cfg["seed"],
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "base_model": base_id,
        "kinetic": cfg["kinetic"],
        "student_params": n_student,
        "base_params": n_base,
        "param_pct": (100 * n_student / n_base) if n_base else None,
        "tokens_seen": total_steps * bs * accum * int(cfg["data"]["seq_len"]),
        "heldout_ppl_pre": ppl0,
        "heldout_ppl_post": ppl_final,
        "heldout_ppl_by_budget": budget_ppl,
        "wall_clock_s": time.time() - t0,
        "log": log,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(
        f"\n=== exp15 done: ppl {ppl0:.1f} -> {ppl_final:.1f} | budgets {budget_ppl} "
        f"| {results['tokens_seen']/1e6:.0f}M tokens | {results['wall_clock_s']/3600:.2f}h ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
