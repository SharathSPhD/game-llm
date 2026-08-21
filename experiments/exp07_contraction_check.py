"""EXP07: Contraction Check for EqLM-v3 (PostLN Fixed Points).

Compares four arms over 300 training steps on real BabyLM stream:
  A1: ExplicitLM reference — parameter-matched baseline
  A2: EqLM-v2 (spectral_norm=True + damping, residual form) — contraction-enforced
  A3: EqLM-v1 (spectral_norm=False, residual form) — non-contractive baseline
  A4: EqLM-v3 (postln=True, spectral_norm=True) — bona fide fixed points (F14 fix)

Primary metrics:
  - Final loss (A4 final ≤ A3 final + 5% per preregistration)
  - Solver convergence rate (A4 rel-convergence > 80% at tol 1e-2 per preregistration)
  - Mean solver iterations (A4 should decrease convergence latency vs v1/v2)
  - Wall time (A4 expected ~2-3x A1 due to fixed-point solving)

Usage:
    # CPU dry-run (3 steps, no GPU):
    python experiments/exp07_contraction_check.py --config configs/exp07_dry_run.yaml --output results/exp07_contraction_check --device cpu

    # GPU smoke (300 steps):
    python experiments/exp07_contraction_check.py --config configs/exp07_smoke.yaml --output results/exp07_contraction_check --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kinetic_ai.data import (
    BabyLMDataLoader,
    build_token_stream,
    load_babylm_dataset,
    load_or_build_tokenizer,
)
from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM, count_params


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

    # Convert string floats in training section
    for key in ["lr"]:
        if key in cfg.get("training", {}):
            cfg["training"][key] = float(cfg["training"][key])

    # Convert model config strings to proper types
    if "model" in cfg:
        model_cfg = cfg["model"]
        for key in ["deq_tol"]:
            if key in model_cfg and isinstance(model_cfg[key], str):
                model_cfg[key] = float(model_cfg[key])

    return cfg


def train_arm(
    arm_name: str,
    model: torch.nn.Module,
    train_loader: BabyLMDataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_steps: int,
    log_every: int,
    grad_clip: float = 1.0,
) -> dict:
    """Train one arm, tracking DEQ solver stats.

    Returns:
        Dict with loss_curve, times, solver_stats (if EqLM arm).
    """
    model.to(device)
    model.train()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    loss_curve = []
    start_time = time.time()
    solver_iterations = []
    solver_convergence = []
    step = 0

    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break

            batch = batch.to(device)
            B, T = batch.shape

            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Track solver stats
            if hasattr(model, "deq") and hasattr(model.deq, "last_info"):
                info = model.deq.last_info
                if isinstance(info, dict) and "iterations" in info:
                    solver_iterations.append(info["iterations"])
                    solver_convergence.append(float(info.get("converged", False)))

            if step % log_every == 0:
                loss_curve.append({"step": step, "loss": loss.item()})
                print(f"  {arm_name} step {step:3d} | loss {loss.item():.4f}")

            step += 1

        # Reload data if needed
        if step < num_steps:
            train_loader.indices = list(range(train_loader.num_seqs))
            if train_loader.shuffle:
                import random
                random.shuffle(train_loader.indices)

    elapsed = time.time() - start_time
    peak_memory = 0.0
    if device.startswith("cuda"):
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = {
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1]["loss"] if loss_curve else float("inf"),
        "total_time_sec": elapsed,
        "peak_memory_mb": peak_memory,
        "total_steps": step,
    }

    if solver_iterations:
        result["solver_mean_iterations"] = float(np.mean(solver_iterations))
        result["solver_max_iterations"] = int(np.max(solver_iterations))
        result["solver_min_iterations"] = int(np.min(solver_iterations))
        result["solver_convergence_rate"] = float(np.mean(solver_convergence))

    return result


def get_git_info() -> dict:
    """Get git commit and diff info."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return {"commit": commit}
    except Exception:
        return {"commit": "unknown"}


def compute_config_hash(cfg: dict) -> str:
    """SHA256 hash of config."""
    cfg_str = json.dumps(cfg, sort_keys=True, indent=2)
    return hashlib.sha256(cfg_str.encode()).hexdigest()


def plot_loss_curves(arms_data: dict, output_dir: Path) -> None:
    """Plot loss curves for all arms."""
    colors = {"A1": "#E69F00", "A2": "#56B4E9", "A3": "#009E73", "A4": "#CC79A7"}

    plt.figure(figsize=(12, 6))

    for arm_name in ["A1", "A2", "A3", "A4"]:
        if arm_name not in arms_data:
            continue
        loss_curve = arms_data[arm_name]["loss_curve"]
        if not loss_curve:
            continue

        steps = [pt["step"] for pt in loss_curve]
        losses = [pt["loss"] for pt in loss_curve]

        plt.plot(
            steps, losses, marker="o", label=arm_name, color=colors.get(arm_name), linewidth=2
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("EXP07: Contraction Check — Loss Curves", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf_path = output_dir / "loss_curves.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=150)
    print(f"Saved loss curve figure to {pdf_path}")
    plt.close()


def main():
    """Main pipeline: load data → train arms → log results."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/exp07_smoke.yaml")
    parser.add_argument("--output", default="results/exp07_contraction_check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    cfg = load_config(args.config)
    print(f"\nConfig:\n{json.dumps(cfg, indent=2)}\n")

    config_hash = compute_config_hash(cfg)
    git_info = get_git_info()

    # ========== Data ==========
    print("Loading BabyLM data...")
    token2id, id2token, tokenizer_choice = load_or_build_tokenizer(texts=None)
    actual_vocab_size = len(token2id)
    print(f"Tokenizer: {tokenizer_choice} | vocab_size {actual_vocab_size}")

    tokenizer_fn = create_gpt2_tokenizer_fn(tokenizer_choice)

    dataset = load_babylm_dataset("BabyLM-2026-Strict-Small")
    token_tensor, num_seqs = build_token_stream(
        dataset,
        tokenizer_fn,
        seq_len=cfg["data"].get("seq_len", 128),
        max_tokens=cfg["data"].get("max_tokens", 1_300_000),
    )
    train_loader = BabyLMDataLoader(
        token_tensor=token_tensor,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        device=device,
    )

    # ========== Hyperparameters ==========
    lr = cfg["training"]["lr"]
    num_steps = cfg["training"]["num_steps"]
    log_every = cfg["training"].get("log_every", 10)

    # ========== Train A1: ExplicitLM (baseline) ==========
    print("\n" + "=" * 60)
    print("ARM A1: ExplicitLM (Baseline)")
    print("=" * 60)

    base_config = EqLMConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        deq_max_iter=cfg["model"]["deq_max_iter"],
        deq_tol=cfg["model"]["deq_tol"],
        solver=cfg["model"].get("solver", "anderson"),
        dropout=cfg["model"].get("dropout", 0.1),
    )

    explicit_model = ExplicitLM(base_config, n_layers=cfg["model"].get("n_layers_explicit", 3))
    a1_params = count_params(explicit_model)
    print(f"A1 params: {a1_params:,}")

    a1_optimizer = torch.optim.AdamW(explicit_model.parameters(), lr=lr)
    a1_result = train_arm("A1", explicit_model, train_loader, a1_optimizer, device, num_steps, log_every)

    # ========== Train A2: EqLM-v2 (spectral norm + damping) ==========
    print("\n" + "=" * 60)
    print("ARM A2: EqLM-v2 (spectral_norm=True + damping)")
    print("=" * 60)

    eqlm_v2_config = EqLMConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        deq_max_iter=cfg["model"]["deq_max_iter"],
        deq_tol=cfg["model"]["deq_tol"],
        solver=cfg["model"].get("solver", "anderson"),
        dropout=cfg["model"].get("dropout", 0.1),
        spectral_norm=True,
        residual_damping=0.2,
    )

    eqlm_v2_model = EqLM(eqlm_v2_config)
    a2_params = count_params(eqlm_v2_model)
    print(f"A2 params: {a2_params:,}")

    a2_optimizer = torch.optim.AdamW(eqlm_v2_model.parameters(), lr=lr)
    a2_result = train_arm("A2", eqlm_v2_model, train_loader, a2_optimizer, device, num_steps, log_every)

    # ========== Train A3: EqLM-v1 (spectral norm=False, baseline) ==========
    print("\n" + "=" * 60)
    print("ARM A3: EqLM-v1 (spectral_norm=False)")
    print("=" * 60)

    eqlm_v1_config = EqLMConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        deq_max_iter=cfg["model"]["deq_max_iter"],
        deq_tol=cfg["model"]["deq_tol"],
        solver=cfg["model"].get("solver", "anderson"),
        dropout=cfg["model"].get("dropout", 0.1),
        spectral_norm=False,
        residual_damping=1.0,  # No damping = full residual
        map_form="residual",  # Explicitly v1 form
    )

    eqlm_v1_model = EqLM(eqlm_v1_config)
    a3_params = count_params(eqlm_v1_model)
    print(f"A3 params: {a3_params:,}")

    a3_optimizer = torch.optim.AdamW(eqlm_v1_model.parameters(), lr=lr)
    a3_result = train_arm("A3", eqlm_v1_model, train_loader, a3_optimizer, device, num_steps, log_every)

    # ========== Train A4: EqLM-v3 (postln map form, F14 fix) ==========
    print("\n" + "=" * 60)
    print("ARM A4: EqLM-v3 (map_form='postln', bona fide fixed points)")
    print("=" * 60)

    eqlm_v3_config = EqLMConfig(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        max_seq_len=cfg["model"]["max_seq_len"],
        deq_max_iter=cfg["model"]["deq_max_iter"],
        deq_tol=cfg["model"]["deq_tol"],
        solver=cfg["model"].get("solver", "anderson"),
        dropout=cfg["model"].get("dropout", 0.1),
        spectral_norm=True,  # v3 uses spectral norm
        residual_damping=cfg["model"].get("residual_damping", 0.2),  # From config
        map_form="postln",  # v3: postln map form with bounded iterates
    )

    eqlm_v3_model = EqLM(eqlm_v3_config)
    a4_params = count_params(eqlm_v3_model)
    print(f"A4 params: {a4_params:,}")

    a4_optimizer = torch.optim.AdamW(eqlm_v3_model.parameters(), lr=lr)
    a4_result = train_arm("A4", eqlm_v3_model, train_loader, a4_optimizer, device, num_steps, log_every)

    # ========== Aggregate Results ==========
    arms_data = {"A1": a1_result, "A2": a2_result, "A3": a3_result, "A4": a4_result}

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nFinal Losses:")
    print(f"  A1 (ExplicitLM):       {a1_result['final_loss']:.4f}")
    print(f"  A2 (EqLM-v2):          {a2_result['final_loss']:.4f}")
    print(f"  A3 (EqLM-v1):          {a3_result['final_loss']:.4f}")
    print(f"  A4 (EqLM-v3 postln):   {a4_result['final_loss']:.4f}")

    if "solver_convergence_rate" in a2_result:
        print("\nSolver Convergence Rates (rel-residual @ tol 1e-2):")
        print(f"  A2 (EqLM-v2): {a2_result['solver_convergence_rate']:.1%}")
        print(f"  A3 (EqLM-v1): {a3_result.get('solver_convergence_rate', 'N/A')}")
        print(f"  A4 (EqLM-v3): {a4_result.get('solver_convergence_rate', 'N/A')}")

    if "solver_mean_iterations" in a2_result:
        print("\nMean Solver Iterations:")
        print(f"  A2 (EqLM-v2): {a2_result['solver_mean_iterations']:.1f}")
        print(f"  A3 (EqLM-v1): {a3_result.get('solver_mean_iterations', 'N/A')}")
        print(f"  A4 (EqLM-v3): {a4_result.get('solver_mean_iterations', 'N/A')}")

    print("\nWall Times:")
    print(f"  A1 (ExplicitLM): {a1_result['total_time_sec']:.1f}s")
    print(f"  A2 (EqLM-v2):    {a2_result['total_time_sec']:.1f}s ({a2_result['total_time_sec']/a1_result['total_time_sec']:.1f}x)")
    print(f"  A3 (EqLM-v1):    {a3_result['total_time_sec']:.1f}s ({a3_result['total_time_sec']/a1_result['total_time_sec']:.1f}x)")
    print(f"  A4 (EqLM-v3):    {a4_result['total_time_sec']:.1f}s ({a4_result['total_time_sec']/a1_result['total_time_sec']:.1f}x)")

    # Preregistered checks
    print("\nPRE-REGISTERED CRITERIA:")
    v3_final = a4_result["final_loss"]
    v1_final = a3_result["final_loss"]
    criterion_a4_loss = v3_final <= v1_final * 1.05
    print(f"  [1] A4 final ≤ A3 final + 5%: {v3_final:.4f} <= {v1_final * 1.05:.4f} = {criterion_a4_loss}")

    if "solver_convergence_rate" in a4_result:
        criterion_a4_conv = a4_result["solver_convergence_rate"] > 0.80
        print(f"  [2] A4 rel-convergence > 80%: {a4_result['solver_convergence_rate']:.1%} = {criterion_a4_conv}")
    else:
        criterion_a4_conv = None

    # Legacy criteria for v2
    v2_final = a2_result["final_loss"]
    criterion1 = v2_final <= v1_final * 1.05
    print(f"  [Legacy] A2 final ≤ A3 final + 5%: {v2_final:.4f} <= {v1_final * 1.05:.4f} = {criterion1}")

    if "solver_convergence_rate" in a2_result:
        criterion2 = a2_result["solver_convergence_rate"] > 0.80
        print(f"  [Legacy] A2 convergence > 80%: {a2_result['solver_convergence_rate']:.1%} = {criterion2}")
    else:
        criterion2 = None

    # Save results
    results = {
        "config_hash": config_hash,
        "git": git_info,
        "arms": {
            "A1": {
                "name": "ExplicitLM (Baseline)",
                "params": a1_params,
                "final_loss": float(a1_result["final_loss"]),
                "wall_time_sec": float(a1_result["total_time_sec"]),
                "peak_memory_mb": float(a1_result["peak_memory_mb"]),
            },
            "A2": {
                "name": "EqLM-v2 (spectral_norm=True, damping=0.2, residual form)",
                "params": a2_params,
                "final_loss": float(a2_result["final_loss"]),
                "wall_time_sec": float(a2_result["total_time_sec"]),
                "peak_memory_mb": float(a2_result["peak_memory_mb"]),
                "solver_convergence_rate": float(a2_result.get("solver_convergence_rate", 0.0)),
                "solver_mean_iterations": float(a2_result.get("solver_mean_iterations", 0.0)),
            },
            "A3": {
                "name": "EqLM-v1 (spectral_norm=False, residual form)",
                "params": a3_params,
                "final_loss": float(a3_result["final_loss"]),
                "wall_time_sec": float(a3_result["total_time_sec"]),
                "peak_memory_mb": float(a3_result["peak_memory_mb"]),
                "solver_convergence_rate": float(a3_result.get("solver_convergence_rate", 0.0)),
                "solver_mean_iterations": float(a3_result.get("solver_mean_iterations", 0.0)),
            },
            "A4": {
                "name": "EqLM-v3 (spectral_norm=True, postln form, bona fide fixed points)",
                "params": a4_params,
                "final_loss": float(a4_result["final_loss"]),
                "wall_time_sec": float(a4_result["total_time_sec"]),
                "peak_memory_mb": float(a4_result["peak_memory_mb"]),
                "solver_convergence_rate": float(a4_result.get("solver_convergence_rate", 0.0)),
                "solver_mean_iterations": float(a4_result.get("solver_mean_iterations", 0.0)),
            },
        },
        "preregistered_criteria": {
            "a4_final_within_5pct_of_a3": criterion_a4_loss,
            "a4_rel_convergence_above_80pct": criterion_a4_conv,
            "a2_final_within_5pct_of_a3": criterion1,
            "a2_convergence_above_80pct": criterion2,
        },
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save config
    cfg_path = output_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Saved config to {cfg_path}")

    # Plot
    plot_loss_curves(arms_data, output_dir)

    print("\n" + "=" * 60)
    print("EXP07 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
