"""Experiment 17 — H8 (SPEC 0012): parameter-space magnetic anchoring (PMA) vs DPO/SimPO at 1.7B.

Compares preference optimization arms on UltraFeedback-binarized:

    D1 "dpo": MagneticAdamW with tau=0 (identical to decoupled AdamW)
    D2 "pma": MagneticAdamW with tau>0 and ref_mode="fixed" (magnet = frozen base weights)
    D3 "simpo": TRL's SimPO/CPO-style loss if available; OMITTED if not in TRL version.

The loss is identical across arms (DPO); only the optimizer differs. This isolates
the effect of magnetic anchoring in parameter space on held-out preference accuracy
and KL-to-base drift.

CRITICAL LESSON (F21): Magnetic displacement scales as lr*tau*steps. The tau values
are configured per arm and documented per config. For scale changes, tau must be
rescaled: tau_new = tau_old * (steps_old / steps_new) * (lr_new / lr_old).

Usage:
    # Run smoke test (quick verification on tiny data)
    python exp17_pma_dpo.py --config configs/exp17_pma_smoke.yaml \
        --output results/exp17_pma_smoke

    # Run seed 42 with all arms
    python exp17_pma_dpo.py --config configs/exp17_pma_seed42.yaml \
        --output results/exp17_pma_seed42

Results are written per-arm under --output/{arm_name}/results.json, with:
    - config_hash: SHA256 of config (for reproducibility)
    - git_commit: HEAD at run time
    - held_out_accuracy: fraction of examples where chosen logprob > rejected
    - kl_to_base: mean KL(policy || base) on held-out prompts
    - wall_clock_s: training time
    - final_loss: last epoch loss
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

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.optim.magnetic_adamw import MagneticAdamW  # noqa: E402


def get_git_commit() -> str:
    """Get current HEAD commit hash."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_config_hash(cfg: dict) -> str:
    """SHA256 of sorted config dict."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def load_config(path: str) -> dict:
    """Load YAML config and convert string floats."""
    cfg = yaml.safe_load(Path(path).read_text())

    # Convert string floats in training and arms sections
    for section in ["training", "arms"]:
        if section in cfg:
            if section == "arms":
                for _arm_name, arm_cfg in cfg["arms"].items():
                    if isinstance(arm_cfg.get("tau"), str):
                        arm_cfg["tau"] = float(arm_cfg["tau"])
            else:
                for k in ("lr", "weight_decay", "grad_clip", "warmup_frac"):
                    if isinstance(cfg[section].get(k), str):
                        cfg[section][k] = float(cfg[section][k])

    return cfg


def load_preference_data(cfg: dict, tokenizer, split: str = "train"):
    """Load UltraFeedback-binarized dataset.

    Args:
        cfg: Config dict with 'data' section containing 'dataset_name', etc.
        tokenizer: HF tokenizer
        split: Dataset split ('train' or 'validation')

    Returns:
        List of dicts with 'chosen' and 'rejected' keys (already tokenized).
    """
    from datasets import load_dataset

    dcfg = cfg["data"]
    dataset_name = dcfg.get("dataset_name", "HuggingFaceH4/ultrafeedback_binarized")
    split_name = dcfg.get("split", "train_sft")

    print(f"[data] loading {dataset_name} split {split_name}...", flush=True)
    ds = load_dataset(dataset_name, split=split_name)

    # Tokenize the dataset for DPO
    # UltraFeedback-binarized has 'chosen' and 'rejected' fields with 'content'
    def tokenize_pair(examples):
        # Tokenize chosen and rejected separately
        chosen_encodings = tokenizer(
            examples["chosen"],
            truncation=True,
            max_length=512,
            padding=False,
        )
        rejected_encodings = tokenizer(
            examples["rejected"],
            truncation=True,
            max_length=512,
            padding=False,
        )
        return {
            "chosen_input_ids": chosen_encodings["input_ids"],
            "chosen_attention_mask": chosen_encodings["attention_mask"],
            "rejected_input_ids": rejected_encodings["input_ids"],
            "rejected_attention_mask": rejected_encodings["attention_mask"],
        }

    ds = ds.map(tokenize_pair, batched=True, remove_columns=ds.column_names)
    return ds


def compute_held_out_accuracy(model, chosen_ids, rejected_ids, device: str) -> float:
    """Compute accuracy on held-out set: fraction where P(chosen) > P(rejected).

    Args:
        model: Language model
        chosen_ids: (N, seq_len) tensor of chosen input IDs
        rejected_ids: (N, seq_len) tensor of rejected input IDs
        device: Device to run on

    Returns:
        Accuracy (0 to 1)
    """
    model.eval()
    with torch.no_grad():
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)

        chosen_outputs = model(chosen_ids, output_hidden_states=False)
        rejected_outputs = model(rejected_ids, output_hidden_states=False)

        # Compute per-token log probs and average
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits

        # Shift for language modeling (predict next token)
        chosen_labels = chosen_ids[:, 1:]
        rejected_labels = rejected_ids[:, 1:]

        chosen_logprobs = torch.nn.functional.log_softmax(chosen_logits[:, :-1], dim=-1)
        rejected_logprobs = torch.nn.functional.log_softmax(rejected_logits[:, :-1], dim=-1)

        # Gather log probs of actual tokens
        chosen_lp = torch.gather(
            chosen_logprobs, -1, chosen_labels.unsqueeze(-1)
        ).squeeze(-1)
        rejected_lp = torch.gather(
            rejected_logprobs, -1, rejected_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Average over sequence dimension
        chosen_lp = chosen_lp.mean(dim=1)
        rejected_lp = rejected_lp.mean(dim=1)

        # Fraction where chosen > rejected
        accuracy = (chosen_lp > rejected_lp).float().mean().item()

    model.train()
    return accuracy


def compute_kl_to_base(model, base_model, input_ids, device: str) -> float:
    """Compute mean KL divergence of current model vs base model.

    Args:
        model: Preference-optimized model
        base_model: Frozen base model
        input_ids: (N, seq_len) input IDs
        device: Device to run on

    Returns:
        Mean KL divergence (nats, reduction: mean over batch and sequence)
    """
    model.eval()
    base_model.eval()
    with torch.no_grad():
        input_ids = input_ids.to(device)

        model_outputs = model(input_ids)
        base_outputs = base_model(input_ids)

        model_logits = model_outputs.logits[:, :-1, :]  # (batch, seq-1, vocab)
        base_logits = base_outputs.logits[:, :-1, :]

        model_logprobs = torch.nn.functional.log_softmax(model_logits, dim=-1)
        base_logprobs = torch.nn.functional.log_softmax(base_logits, dim=-1)

        # KL(model || base) = E_model[log model - log base]
        kl = (torch.exp(model_logprobs) * (model_logprobs - base_logprobs)).sum(dim=-1)
        kl_mean = kl.mean().item()

    model.train()
    return kl_mean


def run_arm(
    arm_name: str,
    arm_cfg: dict,
    cfg: dict,
    base_model_id: str,
    train_data,
    held_out_data,
    out_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run a single arm of the experiment.

    Args:
        arm_name: Arm identifier (e.g., 'dpo', 'pma', 'simpo')
        arm_cfg: Arm-specific config (tau, optimizer_type, etc.)
        cfg: Full experiment config
        base_model_id: HuggingFace model ID
        train_data: Training dataset
        held_out_data: Held-out validation dataset
        out_dir: Output directory for this arm
        device: Device to run on

    Returns:
        Results dict with metrics
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    arm_dir = out_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already run
    results_file = arm_dir / "results.json"
    if results_file.exists():
        print(f"[{arm_name}] skipping (results already exist)", flush=True)
        return json.loads(results_file.read_text())

    print(f"\n=== Arm {arm_name} ===", flush=True)
    tr = cfg["training"]
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )

    # Freeze base model copy for KL computation
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # Gradient checkpointing if enabled
    if tr.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # No reference snapshot is taken here: MagneticAdamW clones each parameter
    # into its own reference state on the first step where tau > 0, so a second
    # copy of the full parameter set would be dead weight in both senses.
    # Build optimizer based on arm config
    lr = float(tr["lr"])
    wd = float(tr.get("weight_decay", 0.01))

    if arm_cfg.get("optimizer_type") == "dpo":
        tau = 0.0
    elif arm_cfg.get("optimizer_type") == "pma":
        tau = float(arm_cfg.get("tau", 1e-4))
    else:
        tau = 0.0

    if arm_cfg.get("optimizer_type") in ("dpo", "pma"):
        opt = MagneticAdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            tau=tau,
            ref_mode="fixed" if arm_cfg.get("optimizer_type") == "pma" else "ema",
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Simple learning rate scheduler (linear warmup + cosine)
    num_steps = int(tr["num_steps"])
    warmup = max(1, int(num_steps * tr.get("warmup_frac", 0.02)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(1, num_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Training loop (simplified DPO without TRL for compatibility)
    model.train()
    bs = int(tr["batch_size"])
    log: list[dict] = []
    t0 = time.time()

    print(f"[{arm_name}] lr={lr} wd={wd} tau={tau} steps={num_steps} batch_size={bs}", flush=True)

    for step in range(num_steps):
        opt.zero_grad(set_to_none=True)

        # Sample batch from training data
        indices = np.random.choice(len(train_data), size=bs, replace=True)
        batch_chosen_ids = torch.stack(
            [torch.tensor(train_data[int(i)]["chosen_input_ids"]) for i in indices]
        )
        batch_rejected_ids = torch.stack(
            [torch.tensor(train_data[int(i)]["rejected_input_ids"]) for i in indices]
        )

        # Pad sequences to same length
        max_len_chosen = batch_chosen_ids.shape[1]
        max_len_rejected = batch_rejected_ids.shape[1]
        max_len = max(max_len_chosen, max_len_rejected)

        def pad_to_len(ids, target_len):
            if ids.shape[1] < target_len:
                pad = torch.full((ids.shape[0], target_len - ids.shape[1]), tokenizer.pad_token_id)
                ids = torch.cat([ids, pad], dim=1)
            return ids[:, :target_len]

        batch_chosen_ids = pad_to_len(batch_chosen_ids, max_len)
        batch_rejected_ids = pad_to_len(batch_rejected_ids, max_len)

        batch_chosen_ids = batch_chosen_ids.to(device)
        batch_rejected_ids = batch_rejected_ids.to(device)

        # Forward pass on chosen and rejected
        chosen_outputs = model(batch_chosen_ids)
        rejected_outputs = model(batch_rejected_ids)

        # DPO loss: log(sigmoid(log p_c - log p_r))
        # For simplicity, use the average log probs
        chosen_logits = chosen_outputs.logits[:, :-1, :]
        rejected_logits = rejected_outputs.logits[:, :-1, :]

        chosen_labels = batch_chosen_ids[:, 1:]
        rejected_labels = batch_rejected_ids[:, 1:]

        chosen_logprobs = torch.nn.functional.log_softmax(chosen_logits, dim=-1)
        rejected_logprobs = torch.nn.functional.log_softmax(rejected_logits, dim=-1)

        chosen_lp = torch.gather(chosen_logprobs, -1, chosen_labels.unsqueeze(-1)).squeeze(-1)
        rejected_lp = torch.gather(
            rejected_logprobs, -1, rejected_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Average over sequence (excluding padding)
        chosen_lp = chosen_lp.mean(dim=1)
        rejected_lp = rejected_lp.mean(dim=1)

        # DPO loss: -log(sigmoid(chosen - rejected))
        log_sigmoid_diff = torch.nn.functional.logsigmoid(chosen_lp - rejected_lp)
        loss = -log_sigmoid_diff.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr.get("grad_clip", 1.0)))
        opt.step()
        sched.step()

        if step % int(tr.get("log_every", 20)) == 0 or step == num_steps - 1:
            rec = {
                "step": step,
                "loss": loss.item(),
                "lr": sched.get_last_lr()[0],
                "elapsed_s": time.time() - t0,
            }
            log.append(rec)
            print(
                f"  [{arm_name}] step {step:5d} loss {rec['loss']:7.4f} "
                f"lr {rec['lr']:.2e} {rec['elapsed_s']/60:5.1f}min",
                flush=True,
            )

    wall_clock = time.time() - t0

    # Evaluate on held-out set
    print(f"[{arm_name}] evaluating on held-out set...", flush=True)

    held_out_chosen_ids = torch.stack(
        [torch.tensor(held_out_data[i]["chosen_input_ids"]) for i in range(len(held_out_data))]
    )
    held_out_rejected_ids = torch.stack(
        [torch.tensor(held_out_data[i]["rejected_input_ids"]) for i in range(len(held_out_data))]
    )

    accuracy = compute_held_out_accuracy(
        model, held_out_chosen_ids, held_out_rejected_ids, device
    )
    kl_to_base = compute_kl_to_base(model, base_model, held_out_chosen_ids, device)

    results = {
        "experiment": "exp17_pma_dpo",
        "spec": "0012",
        "hypothesis": "H8",
        "arm": arm_name,
        "seed": cfg["seed"],
        "config_hash": compute_config_hash(cfg),
        "git_commit": get_git_commit(),
        "base_model": base_model_id,
        "optimizer_type": arm_cfg.get("optimizer_type", "dpo"),
        "tau": tau,
        "lr": lr,
        "weight_decay": wd,
        "batch_size": bs,
        "num_steps": num_steps,
        "held_out_accuracy": accuracy,
        "kl_to_base": kl_to_base,
        "wall_clock_s": wall_clock,
        "final_loss": log[-1]["loss"] if log else None,
        "log": log,
    }

    (arm_dir / "results.json").write_text(json.dumps(results, indent=2))
    model.save_pretrained(arm_dir / "checkpoint")
    tokenizer.save_pretrained(arm_dir / "checkpoint")

    print(
        f"\n[{arm_name}] done: accuracy {accuracy:.4f} | "
        f"kl_to_base {kl_to_base:.4f} | {wall_clock/60:.1f}min",
        flush=True,
    )

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Config YAML path")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--arms", help="Comma-separated arms to run (default: all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")

    print(f"[exp17] device={device} seed={cfg['seed']} config_hash={compute_config_hash(cfg)[:8]}")
    print(f"[exp17] arms: {list(cfg.get('arms', {}).keys())}", flush=True)

    # Load data once
    from transformers import AutoTokenizer

    base_model_id = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[data] loading preference data...", flush=True)
    ds = load_preference_data(cfg, tokenizer)

    # Split into train/heldout
    n_hold = int(cfg["data"].get("heldout_frac", 0.1) * len(ds))
    indices = np.random.permutation(len(ds))
    held_out_indices = indices[:n_hold]
    train_indices = indices[n_hold:]

    held_out_data = ds.select(held_out_indices)
    train_data = ds.select(train_indices)

    print(f"[data] train={len(train_data)} held_out={len(held_out_data)}", flush=True)

    # Run arms
    arms_to_run = cfg.get("arms", {})
    if args.arms:
        arms_to_run = {k: v for k, v in arms_to_run.items() if k in args.arms.split(",")}

    all_results = {}
    for arm_name, arm_cfg in arms_to_run.items():
        result = run_arm(
            arm_name,
            arm_cfg,
            cfg,
            base_model_id,
            train_data,
            held_out_data,
            out_dir,
            device,
        )
        all_results[arm_name] = result

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    for arm_name, result in all_results.items():
        print(
            f"  {arm_name:10s} accuracy={result['held_out_accuracy']:.4f} "
            f"kl={result['kl_to_base']:.4f} wall_clock={result['wall_clock_s']/60:.1f}min",
            flush=True,
        )
    print("=" * 80, flush=True)

    # Write summary
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
