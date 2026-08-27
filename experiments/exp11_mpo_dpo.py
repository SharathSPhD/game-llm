"""Experiment 11 — H3 (SPEC 0007): Magnetic Preference Optimization vs DPO.

BLiMP minimal pairs ARE preference pairs: (sentence_good > sentence_bad).
Fine-tune the trained exp10 checkpoints with the standard DPO loss under two
optimizers at matched budgets:

  P1  DPO: AdamW
  P2a MPO: MagneticAdamW, ref_mode="fixed" (magnet = frozen base), tau small
  P2b MPO: same, tau larger

Split is BY PHENOMENON (train on 60% of phenomena, hold out the rest), so
held-out accuracy measures generalization to unseen phenomena, not
memorization of trained pairs.

Metrics per (base model x arm): held-out pair accuracy (win-rate proxy),
train-domain pair accuracy (catastrophic-drift check), and mean per-token
KL-to-reference on held-out good sentences (reward-hacking drift proxy).

H3 scoring (pre-registered): MPO >= DPO held-out accuracy AND lower KL drift
=> MET; either half alone => PARTIAL; neither => MISSED.

Usage:
    python exp11_mpo_dpo.py --config configs/exp11_mpo_dpo.yaml --output results/exp11
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from kinetic_ai.models.eqlm import load_checkpoint  # noqa: E402
from kinetic_ai.optim.magnetic_adamw import MagneticAdamW  # noqa: E402


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
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tr = cfg["training"]
    for key in ("lr", "beta"):
        if isinstance(tr.get(key), str):
            tr[key] = float(tr[key])
    for arm in cfg["arms"].values():
        if isinstance(arm.get("tau"), str):
            arm["tau"] = float(arm["tau"])
    return cfg


def create_tokenizer_fn():
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    return lambda text: tok.encode(text)


def split_pairs_by_phenomenon(
    pairs: list[dict], train_uids: list[str]
) -> tuple[list[dict], list[dict]]:
    """Phenomenon-level split: held-out phenomena are never trained on."""
    train = [p for p in pairs if p["UID"] in train_uids]
    heldout = [p for p in pairs if p["UID"] not in train_uids]
    if not train or not heldout:
        raise ValueError(
            f"Bad split: {len(train)} train / {len(heldout)} held-out pairs "
            f"for train_uids={train_uids}"
        )
    return train, heldout


def tokenize_pairs(
    pairs: list[dict], tokenizer_fn, max_len: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for p in pairs:
        g = torch.tensor(tokenizer_fn(p["sentence_good"])[:max_len], dtype=torch.long)
        b = torch.tensor(tokenizer_fn(p["sentence_bad"])[:max_len], dtype=torch.long)
        if len(g) >= 2 and len(b) >= 2:
            out.append((g, b))
    return out


def sequence_logprob(
    model: torch.nn.Module, tokens: torch.Tensor, device: str
) -> torch.Tensor:
    """Sum of next-token log-probs for one sequence. Differentiable."""
    tokens = tokens.unsqueeze(0).to(device)
    logits = model(tokens)
    logp = F.log_softmax(logits[0, :-1, :], dim=-1)
    return logp.gather(1, tokens[0, 1:].unsqueeze(1)).sum()


@torch.no_grad()
def pair_accuracy(
    model: torch.nn.Module,
    tok_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> float:
    model.eval()
    correct = sum(
        1
        for g, b in tok_pairs
        if sequence_logprob(model, g, device) > sequence_logprob(model, b, device)
    )
    return correct / len(tok_pairs)


@torch.no_grad()
def mean_kl_to_reference(
    model: torch.nn.Module,
    reference: torch.nn.Module,
    sentences: list[torch.Tensor],
    device: str,
) -> float:
    """Mean per-token KL(pi_model || pi_ref) over held-out sentences."""
    model.eval()
    reference.eval()
    total_kl, total_tok = 0.0, 0
    for tokens in sentences:
        t = tokens.unsqueeze(0).to(device)
        logp_m = F.log_softmax(model(t)[0, :-1, :], dim=-1)
        logp_r = F.log_softmax(reference(t)[0, :-1, :], dim=-1)
        kl = F.kl_div(logp_r, logp_m, log_target=True, reduction="sum")
        total_kl += kl.item()
        total_tok += logp_m.shape[0]
    return total_kl / max(total_tok, 1)


@torch.no_grad()
def precompute_ref_logprobs(
    reference: torch.nn.Module,
    tok_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> list[tuple[float, float]]:
    reference.eval()
    return [
        (
            sequence_logprob(reference, g, device).item(),
            sequence_logprob(reference, b, device).item(),
        )
        for g, b in tok_pairs
    ]


def dpo_finetune(
    model: torch.nn.Module,
    train_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ref_logprobs: list[tuple[float, float]],
    optimizer: torch.optim.Optimizer,
    beta: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> list[float]:
    """Standard DPO loss; the optimizer choice is the ONLY arm difference."""
    rng = random.Random(seed)
    order = list(range(len(train_pairs)))
    loss_curve: list[float] = []
    model.train()
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            losses = []
            for i in idx:
                g, b = train_pairs[i]
                ref_g, ref_b = ref_logprobs[i]
                lp_g = sequence_logprob(model, g, device)
                lp_b = sequence_logprob(model, b, device)
                margin = beta * ((lp_g - ref_g) - (lp_b - ref_b))
                losses.append(-F.logsigmoid(margin))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            loss_curve.append(loss.item())
    return loss_curve


def build_optimizer(arm_cfg: dict, model: torch.nn.Module, lr: float):
    if arm_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    if arm_cfg["optimizer"] == "magnetic":
        return MagneticAdamW(
            model.parameters(), lr=lr, tau=arm_cfg["tau"], ref_mode="fixed"
        )
    raise ValueError(f"Unknown optimizer: {arm_cfg['optimizer']}")


def run_arm(
    base_name: str,
    arm_name: str,
    arm_cfg: dict,
    cfg: dict,
    checkpoint_path: str,
    data: dict[str, Any],
    device: str,
    out_dir: Path,
) -> dict:
    tag = f"{base_name}_{arm_name}"
    arm_file = out_dir / f"arm_{tag}.json"
    if arm_file.exists():
        print(f"[resume] {tag} already complete, skipping")
        return json.loads(arm_file.read_text())

    tr = cfg["training"]
    torch.manual_seed(cfg["seed"])
    model = load_checkpoint(checkpoint_path).to(device)
    reference = load_checkpoint(checkpoint_path).to(device)
    for p in reference.parameters():
        p.requires_grad_(False)

    print(f"[{tag}] baseline metrics...")
    before = {
        "heldout_acc": pair_accuracy(model, data["heldout"], device),
        "train_acc": pair_accuracy(model, data["train"], device),
    }

    optimizer = build_optimizer(arm_cfg, model, tr["lr"])
    t0 = time.time()
    loss_curve = dpo_finetune(
        model,
        data["train"],
        data["ref_logprobs"],
        optimizer,
        beta=tr["beta"],
        epochs=tr["epochs"],
        batch_size=tr["batch_size"],
        device=device,
        seed=cfg["seed"],
    )
    train_time = time.time() - t0

    print(f"[{tag}] post-finetune metrics...")
    after = {
        "heldout_acc": pair_accuracy(model, data["heldout"], device),
        "train_acc": pair_accuracy(model, data["train"], device),
        "kl_to_ref": mean_kl_to_reference(
            model, reference, data["kl_sentences"], device
        ),
    }

    result = {
        "base": base_name,
        "arm": arm_name,
        "optimizer": arm_cfg["optimizer"],
        "tau": arm_cfg.get("tau"),
        "before": before,
        "after": after,
        "final_loss": loss_curve[-1] if loss_curve else None,
        "loss_curve_head": loss_curve[:5],
        "loss_curve_tail": loss_curve[-5:],
        "num_train_steps": len(loss_curve),
        "train_time_sec": train_time,
    }
    arm_file.write_text(json.dumps(result, indent=2))

    del model, reference, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    pairs = json.loads(Path(cfg["pairs"]["json_file"]).read_text())
    train_raw, heldout_raw = split_pairs_by_phenomenon(
        pairs, cfg["pairs"]["train_uids"]
    )
    tokenizer_fn = create_tokenizer_fn()
    max_len = cfg["training"]["max_seq_len"]
    data: dict[str, Any] = {
        "train": tokenize_pairs(train_raw, tokenizer_fn, max_len),
        "heldout": tokenize_pairs(heldout_raw, tokenizer_fn, max_len),
    }
    # KL drift is measured on held-out GOOD sentences (unseen text domain).
    data["kl_sentences"] = [g for g, _ in data["heldout"]]
    print(
        f"pairs: {len(data['train'])} train / {len(data['heldout'])} held-out "
        f"(train UIDs: {cfg['pairs']['train_uids']})"
    )

    results = {
        "experiment": "exp11_mpo_dpo",
        "spec": "0007",
        "seed": cfg["seed"],
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "split": {
            "train_uids": cfg["pairs"]["train_uids"],
            "n_train": len(data["train"]),
            "n_heldout": len(data["heldout"]),
        },
        "runs": {},
    }

    for base_name, base_cfg in cfg["bases"].items():
        # Reference logprobs depend only on the base model: compute once per base.
        ckpt = base_cfg["checkpoint"]
        reference = load_checkpoint(ckpt).to(device)
        for p in reference.parameters():
            p.requires_grad_(False)
        print(f"[{base_name}] precomputing reference logprobs...")
        data["ref_logprobs"] = precompute_ref_logprobs(
            reference, data["train"], device
        )
        del reference
        if device == "cuda":
            torch.cuda.empty_cache()

        for arm_name, arm_cfg in cfg["arms"].items():
            results["runs"][f"{base_name}_{arm_name}"] = run_arm(
                base_name, arm_name, arm_cfg, cfg, ckpt, data, device, out_dir
            )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nresults written to {out_dir / 'results.json'}")
    for tag, r in results["runs"].items():
        print(
            f"  {tag:22} heldout {r['before']['heldout_acc']:.3f}->"
            f"{r['after']['heldout_acc']:.3f}  train {r['before']['train_acc']:.3f}->"
            f"{r['after']['train_acc']:.3f}  KL {r['after']['kl_to_ref']:.5f}"
        )


if __name__ == "__main__":
    main()
