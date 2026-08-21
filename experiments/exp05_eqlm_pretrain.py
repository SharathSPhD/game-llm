"""SPEC 0004: EqLM Pretraining on BabyLM (Tier B SMOKE).

Trains three arms (A1=ExplicitLM, A2=EqLM, A3=EqLM+MMD) on BabyLM 2026 strict-small.
Logs loss curves, memory, and wall time per arm. Evaluates on BLiMP subset.

Usage:
    python experiments/exp05_eqlm_pretrain.py --config configs/exp05_smoke.yaml --output results/exp05_eqlm_pretrain
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from kinetic_ai.config import BregmanType, MMDConfig
from kinetic_ai.data import (
    BabyLMDataLoader,
    build_token_stream,
    load_babylm_dataset,
    load_or_build_tokenizer,
)
from kinetic_ai.eval.blimp import evaluate_blimp_subset, load_blimp_subset
from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM, count_params
from kinetic_ai.optim.mmd import MagneticMirrorDescent


def create_gpt2_tokenizer_fn(tokenizer_choice: str):
    """Create a tokenizer function from GPT-2 or HF tokenizers."""
    try:
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
        return lambda text: tok.encode(text)
    except Exception:
        # Fallback: simple space-split + int conversion
        return lambda text: [int(t) if t.isdigit() else 0 for t in text.split()]


def load_config(config_path: str) -> dict:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Convert string floats to actual floats (YAML quirk)
    if "training" in cfg and "lr" in cfg["training"] and isinstance(cfg["training"]["lr"], str):
        cfg["training"]["lr"] = float(cfg["training"]["lr"])

    # Fix MMD tau if it's a string
    if (
        "arms" in cfg
        and "A3" in cfg["arms"]
        and "mmd_config" in cfg["arms"]["A3"]
        and "tau" in cfg["arms"]["A3"]["mmd_config"]
        and isinstance(cfg["arms"]["A3"]["mmd_config"]["tau"], str)
    ):
        cfg["arms"]["A3"]["mmd_config"]["tau"] = float(cfg["arms"]["A3"]["mmd_config"]["tau"])

    # Fix deq_tol if it's a string
    for arm in ["A2", "A3"]:
        if (
            arm in cfg["arms"]
            and "config" in cfg["arms"][arm]
            and "deq_tol" in cfg["arms"][arm]["config"]
            and isinstance(cfg["arms"][arm]["config"]["deq_tol"], str)
        ):
            cfg["arms"][arm]["config"]["deq_tol"] = float(cfg["arms"][arm]["config"]["deq_tol"])

    return cfg


def train_arm(
    arm_name: str,
    model: torch.nn.Module,
    train_loader: BabyLMDataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_steps: int,
    log_every: int,
) -> dict:
    """Train one model arm.

    Args:
        arm_name: Name of the arm (A1, A2, A3).
        model: Model to train.
        train_loader: Data loader.
        optimizer: Optimizer.
        device: Device.
        num_steps: Total steps to train.
        log_every: Log frequency.

    Returns:
        Dict with loss_curve, total_time, peak_memory_mb.
    """
    model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)

    loss_curve = []
    start_time = time.time()

    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break

        batch = batch.to(device)
        B, T = batch.shape

        # Shift: predict token t from tokens < t
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        # Forward
        optimizer.zero_grad()
        logits = model(input_ids)  # [B, T-1, vocab_size]

        # Loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )

        # Backward
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            loss_curve.append({"step": step, "loss": loss.item()})
            print(f"  {arm_name} step {step:4d} | loss {loss.item():.4f}")

        step += 1

    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "loss_curve": loss_curve,
        "total_time_sec": elapsed,
        "peak_memory_mb": peak_memory,
        "total_steps": step,
    }


def main():
    """Main pipeline: data → train → eval → results."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/exp05_smoke.yaml")
    parser.add_argument("--output", default="results/exp05_eqlm_pretrain")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(args.config)
    print(f"\nConfig: {json.dumps(cfg, indent=2)}")

    # Save resolved config
    resolved_cfg_path = output_dir / "config.yaml"
    with open(resolved_cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Saved resolved config to {resolved_cfg_path}")

    # Set seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)

    # ========================================================================
    # DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Loading data and tokenizer")
    print("=" * 70)

    dataset = load_babylm_dataset(
        subset=cfg["data"]["dataset"],
        max_samples=None,  # Use all available
    )
    print(f"Loaded dataset: {len(dataset)} samples")

    # Load or build tokenizer
    token2id, id2token, tokenizer_choice = load_or_build_tokenizer(texts=None)
    actual_vocab_size = len(token2id)
    print(f"Tokenizer: {tokenizer_choice} | vocab_size {actual_vocab_size}")

    tokenizer_fn = create_gpt2_tokenizer_fn(tokenizer_choice)

    # Override config vocab size with actual tokenizer vocab size
    cfg["arms"]["A1"]["config"]["vocab_size"] = actual_vocab_size
    cfg["arms"]["A2"]["config"]["vocab_size"] = actual_vocab_size
    cfg["arms"]["A3"]["config"]["vocab_size"] = actual_vocab_size

    # Build token stream (smoke: limit to subset_size)
    token_tensor, num_seqs = build_token_stream(
        dataset,
        tokenizer_fn,
        seq_len=cfg["data"]["seq_len"],
        max_tokens=cfg["data"]["subset_size"],
    )
    print(f"Built token stream: {num_seqs} sequences of length {cfg['data']['seq_len']}")

    # Data loader
    train_loader = BabyLMDataLoader(
        token_tensor,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        device=device,
    )
    print(f"Data loader: {len(train_loader)} batches")

    # ========================================================================
    # BUILD MODELS & OPTIMIZERS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Building models")
    print("=" * 70)

    results = {
        "arms": {},
        "config_hash": "",
        "git_commit": "",
        "tokenizer_choice": tokenizer_choice,
        "notes": "SMOKE test for SPEC 0004; tiny models for 30min GPU budget.",
    }

    # A1: ExplicitLM (baseline)
    print("\n--- Arm A1: ExplicitLM (Baseline) ---")
    a1_config_dict = cfg["arms"]["A1"]["config"].copy()
    a1_n_layers = a1_config_dict.pop("n_layers", 4)
    a1_cfg = EqLMConfig(**a1_config_dict)
    a1_model = ExplicitLM(config=a1_cfg, n_layers=a1_n_layers)
    a1_params = count_params(a1_model)
    print(f"A1 params: {a1_params:,}")

    # A2: EqLM (param-matched)
    print("\n--- Arm A2: EqLM (Param-matched) ---")
    a2_cfg = EqLMConfig(**cfg["arms"]["A2"]["config"])
    a2_model = EqLM(config=a2_cfg)
    a2_params = count_params(a2_model)
    print(f"A2 params: {a2_params:,}")
    print(f"A2/A1 param ratio: {a2_params / a1_params:.2f}")

    # A3: EqLM + MMD
    print("\n--- Arm A3: EqLM + MMD (Magnetic anchor) ---")
    a3_cfg = EqLMConfig(**cfg["arms"]["A3"]["config"])
    a3_model = EqLM(config=a3_cfg)
    a3_params = count_params(a3_model)
    print(f"A3 params: {a3_params:,}")
    print(f"A3/A1 param ratio: {a3_params / a1_params:.2f}")

    # ========================================================================
    # TRAIN
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Training (sequential)")
    print("=" * 70)

    num_steps = len(train_loader) * cfg["training"]["num_epochs"]
    print(f"Total steps per arm: {num_steps}")

    # Arm A1
    print("\n--- Training A1 ---")
    a1_opt = torch.optim.AdamW(a1_model.parameters(), lr=cfg["training"]["lr"])
    a1_results = train_arm("A1", a1_model, train_loader, a1_opt, device, num_steps, cfg["training"]["log_every"])
    results["arms"]["A1"] = {
        **a1_results,
        "num_params": a1_params,
        "model_name": "ExplicitLM",
    }
    print(f"A1 done: {a1_results['total_time_sec']:.1f}s, peak memory {a1_results['peak_memory_mb']:.0f}MB")

    # Arm A2
    print("\n--- Training A2 ---")
    a2_opt = torch.optim.AdamW(a2_model.parameters(), lr=cfg["training"]["lr"])
    a2_results = train_arm("A2", a2_model, train_loader, a2_opt, device, num_steps, cfg["training"]["log_every"])
    results["arms"]["A2"] = {
        **a2_results,
        "num_params": a2_params,
        "model_name": "EqLM",
    }
    print(f"A2 done: {a2_results['total_time_sec']:.1f}s, peak memory {a2_results['peak_memory_mb']:.0f}MB")

    # Arm A3
    print("\n--- Training A3 (MMD) ---")
    mmd_cfg = MMDConfig(
        lr=cfg["training"]["lr"],
        tau=cfg["arms"]["A3"]["mmd_config"].get("tau", 1e-2),
        bregman_type=BregmanType.EUCLIDEAN,
        reference_update_interval=cfg["arms"]["A3"]["mmd_config"].get("reference_update_interval", 0),
    )
    # Move model to device first
    a3_model = a3_model.to(device)
    a3_opt = MagneticMirrorDescent(a3_model.parameters(), config=mmd_cfg)
    # Move reference state to device
    a3_opt._reference_state = [ref.to(device) for ref in a3_opt._reference_state]
    a3_results = train_arm("A3", a3_model, train_loader, a3_opt, device, num_steps, cfg["training"]["log_every"])
    results["arms"]["A3"] = {
        **a3_results,
        "num_params": a3_params,
        "model_name": "EqLM+MMD",
        "mmd_tau": mmd_cfg.tau,
    }
    print(f"A3 done: {a3_results['total_time_sec']:.1f}s, peak memory {a3_results['peak_memory_mb']:.0f}MB")

    # ========================================================================
    # EVAL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: BLiMP-subset evaluation")
    print("=" * 70)

    blimp_subset = load_blimp_subset(
        num_phenomena=cfg["eval"]["blimp"]["num_phenomena"],
        pairs_per_phenomenon=cfg["eval"]["blimp"]["pairs_per_phenomenon"],
    )
    print(f"Loaded BLiMP subset: {len(blimp_subset)} pairs")

    for arm_name, model in [("A1", a1_model), ("A2", a2_model), ("A3", a3_model)]:
        print(f"\nEvaluating {arm_name}...")
        try:
            eval_result = evaluate_blimp_subset(
                model,
                blimp_subset,
                tokenizer_fn,
                device=device,
                max_samples=min(100, len(blimp_subset)),  # Smoke: max 100 samples
            )
            results["arms"][arm_name]["blimp_accuracy"] = eval_result["accuracy"]
            results["arms"][arm_name]["blimp_num_correct"] = eval_result["num_correct"]
            results["arms"][arm_name]["blimp_num_total"] = eval_result["num_total"]
            print(f"  BLiMP accuracy: {eval_result['accuracy']:.3f}")
        except Exception as e:
            print(f"  BLiMP eval failed: {e}")
            results["arms"][arm_name]["blimp_accuracy"] = None
            results["arms"][arm_name]["blimp_error"] = str(e)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: Saving results")
    print("=" * 70)

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        # Convert tensors to lists for JSON serialization
        results_json = results.copy()
        json.dump(results_json, f, indent=2)
    print(f"Saved results to {results_json_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for arm in ["A1", "A2", "A3"]:
        arm_data = results["arms"][arm]
        print(f"\n{arm}: {arm_data['model_name']}")
        print(f"  Params: {arm_data['num_params']:,}")
        print(f"  Time: {arm_data['total_time_sec']:.1f}s")
        print(f"  Peak Memory: {arm_data['peak_memory_mb']:.0f}MB")
        if "blimp_accuracy" in arm_data and arm_data["blimp_accuracy"] is not None:
            print(f"  BLiMP Accuracy: {arm_data['blimp_accuracy']:.3f} ({arm_data['blimp_num_correct']}/{arm_data['blimp_num_total']})")

    print("\n✓ SMOKE test complete!")


if __name__ == "__main__":
    main()
