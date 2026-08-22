"""EXP08: Solver-Aware Auxiliary Loss for Learning Contraction.

Tests whether training with an auxiliary loss that penalizes the solver residual
enables EqLM to LEARN contraction, without explicit architectural constraints.

Compares four arms over 300 training steps on real BabyLM stream:
  B0: EqLM-v3 (postln) with lambda_aux=0.0 (control)
  B1: EqLM-v3 (postln) with lambda_aux=0.01
  B2: EqLM-v3 (postln) with lambda_aux=0.1 (primary treatment)
  B3: EqLM-v3 (postln) with lambda_aux=1.0

Primary metrics:
  - Final CE loss (all arms should be within 5% of B0 per preregistration)
  - Solver rel-residual at exit (mean over last 50 steps)
  - Solver convergence rate (at tol 1e-2)
  - Wall time

Preregistered hypothesis:
  Exists an arm with (a) rel-residual < 0.5x control AND (b) CE within 5% of control.
  If true: solver-aware loss enables learning of contraction.

Usage:
    # CPU dry-run (3 steps, no GPU):
    python experiments/exp08_solver_aware.py --config configs/exp08_dry_run.yaml --output results/exp08_solver_aware --device cpu

    # GPU smoke (300 steps):
    python experiments/exp08_solver_aware.py --config configs/exp08_smoke.yaml --output results/exp08_solver_aware --device cuda
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
from kinetic_ai.models.eqlm import EqLM, EqLMConfig, count_params


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
    lambda_aux: float = 0.0,
    grad_clip: float = 1.0,
) -> dict:
    """Train one arm, tracking DEQ solver stats and auxiliary residuals.

    Returns:
        Dict with loss_curve, times, solver_stats, residual_stats.
    """
    model.to(device)
    model.train()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    loss_curve = []
    aux_residuals = []
    solver_iterations = []
    solver_convergence = []
    solver_rel_residuals = []
    step = 0
    start_time = time.time()

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

            # Cross-entropy loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

            # Auxiliary residual loss (if lambda_aux > 0)
            aux_loss = torch.tensor(0.0, device=device)
            if hasattr(model, "last_aux_residual") and model.last_aux_residual is not None:
                aux_loss = model.last_aux_residual
                aux_residuals.append(aux_loss.item())
            else:
                aux_residuals.append(0.0)

            # Total loss
            total_loss = ce_loss + lambda_aux * aux_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Track solver stats
            if hasattr(model, "deq") and hasattr(model.deq, "last_info"):
                info = model.deq.last_info
                if isinstance(info, dict) and "iterations" in info:
                    solver_iterations.append(info["iterations"])
                    solver_convergence.append(float(info.get("converged", False)))
                    if "rel_residuals" in info and info["rel_residuals"]:
                        solver_rel_residuals.append(info["rel_residuals"][-1])

            if step % log_every == 0:
                loss_curve.append(
                    {
                        "step": step,
                        "ce_loss": ce_loss.item(),
                        "aux_loss": aux_loss.item() if aux_loss.item() > 0 else 0.0,
                        "total_loss": total_loss.item(),
                    }
                )
                print(
                    f"  {arm_name} step {step:3d} | CE {ce_loss.item():.4f} | "
                    f"aux {aux_loss.item():.6f} | total {total_loss.item():.4f}"
                )

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
        "final_ce_loss": loss_curve[-1]["ce_loss"] if loss_curve else float("inf"),
        "final_aux_loss": loss_curve[-1]["aux_loss"] if loss_curve else 0.0,
        "final_total_loss": loss_curve[-1]["total_loss"] if loss_curve else float("inf"),
        "total_time_sec": elapsed,
        "peak_memory_mb": peak_memory,
        "total_steps": step,
    }

    # Solver statistics
    if solver_iterations:
        result["solver_mean_iterations"] = float(np.mean(solver_iterations))
        result["solver_max_iterations"] = int(np.max(solver_iterations))
        result["solver_min_iterations"] = int(np.min(solver_iterations))
        result["solver_convergence_rate"] = float(np.mean(solver_convergence))

    # Residual statistics: mean over last 50 steps
    if solver_rel_residuals:
        last_50 = solver_rel_residuals[-50:] if len(solver_rel_residuals) >= 50 else solver_rel_residuals
        result["solver_rel_residual_mean_last50"] = float(np.mean(last_50))
        result["solver_rel_residual_final"] = float(solver_rel_residuals[-1])
        result["solver_rel_residual_min"] = float(np.min(solver_rel_residuals))

    # Auxiliary residual statistics
    if aux_residuals:
        last_50_aux = aux_residuals[-50:] if len(aux_residuals) >= 50 else aux_residuals
        result["aux_residual_mean_last50"] = float(np.mean(last_50_aux))
        result["aux_residual_final"] = float(aux_residuals[-1])

    return result


def get_git_info() -> dict:
    """Get git commit info."""
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
    colors = {"B0": "#1f77b4", "B1": "#ff7f0e", "B2": "#2ca02c", "B3": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CE loss
    for arm_name in ["B0", "B1", "B2", "B3"]:
        if arm_name not in arms_data:
            continue
        loss_curve = arms_data[arm_name]["loss_curve"]
        if not loss_curve:
            continue

        steps = [pt["step"] for pt in loss_curve]
        ce_losses = [pt["ce_loss"] for pt in loss_curve]

        axes[0].plot(
            steps,
            ce_losses,
            marker="o",
            label=arm_name,
            color=colors.get(arm_name),
            linewidth=2,
        )

    axes[0].set_xlabel("Training Step", fontsize=12)
    axes[0].set_ylabel("CE Loss", fontsize=12)
    axes[0].set_title("EXP08: Solver-Aware Loss — CE Loss Curves", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Auxiliary loss (only for arms with lambda_aux > 0)
    for arm_name in ["B1", "B2", "B3"]:
        if arm_name not in arms_data:
            continue
        loss_curve = arms_data[arm_name]["loss_curve"]
        if not loss_curve:
            continue

        steps = [pt["step"] for pt in loss_curve]
        aux_losses = [pt["aux_loss"] for pt in loss_curve]

        axes[1].plot(
            steps,
            aux_losses,
            marker="o",
            label=arm_name,
            color=colors.get(arm_name),
            linewidth=2,
        )

    axes[1].set_xlabel("Training Step", fontsize=12)
    axes[1].set_ylabel("Auxiliary Residual Loss", fontsize=12)
    axes[1].set_title("EXP08: Auxiliary Residual Over Training", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    pdf_path = output_dir / "loss_curves.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=150)
    print(f"Saved loss curve figure to {pdf_path}")
    plt.close()


def main():
    """Main pipeline: load data → train arms → log results."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", default="configs/exp08_smoke.yaml")
    parser.add_argument("--output", default="results/exp08_solver_aware")
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

    try:
        from transformers import GPT2Tokenizer
        def tokenizer_fn(text):
            return GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True).encode(text)
    except Exception:
        def tokenizer_fn(text):
            return [int(t) if t.isdigit() else 0 for t in text.split()]

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

    # Base config for all arms (EqLM-v3 postln)
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
        spectral_norm=True,
        residual_damping=cfg["model"].get("residual_damping", 0.2),
        map_form="postln",
        aux_residual=True,  # Enable auxiliary residual computation
    )

    arms_results = {}

    # ===== B0: lambda_aux = 0.0 (control) =====
    print("\n" + "=" * 60)
    print("ARM B0: EqLM-v3 (postln) with lambda_aux=0.0 (Control)")
    print("=" * 60)

    cfg_b0 = EqLMConfig(
        **{
            k: v
            for k, v in base_config.__dict__.items()
            if k not in ["lambda_aux"]
        },
        lambda_aux=0.0,
    )
    model_b0 = EqLM(cfg_b0)
    params_b0 = count_params(model_b0)
    print(f"B0 params: {params_b0:,}")

    optimizer_b0 = torch.optim.AdamW(model_b0.parameters(), lr=lr)
    result_b0 = train_arm(
        "B0",
        model_b0,
        train_loader,
        optimizer_b0,
        device,
        num_steps,
        log_every,
        lambda_aux=0.0,
    )
    arms_results["B0"] = result_b0

    # ===== B1: lambda_aux = 0.01 =====
    print("\n" + "=" * 60)
    print("ARM B1: EqLM-v3 (postln) with lambda_aux=0.01")
    print("=" * 60)

    cfg_b1 = EqLMConfig(
        **{
            k: v
            for k, v in base_config.__dict__.items()
            if k not in ["lambda_aux"]
        },
        lambda_aux=0.01,
    )
    model_b1 = EqLM(cfg_b1)
    params_b1 = count_params(model_b1)
    print(f"B1 params: {params_b1:,}")

    optimizer_b1 = torch.optim.AdamW(model_b1.parameters(), lr=lr)
    result_b1 = train_arm(
        "B1",
        model_b1,
        train_loader,
        optimizer_b1,
        device,
        num_steps,
        log_every,
        lambda_aux=0.01,
    )
    arms_results["B1"] = result_b1

    # ===== B2: lambda_aux = 0.1 (primary treatment) =====
    print("\n" + "=" * 60)
    print("ARM B2: EqLM-v3 (postln) with lambda_aux=0.1 (Primary)")
    print("=" * 60)

    cfg_b2 = EqLMConfig(
        **{
            k: v
            for k, v in base_config.__dict__.items()
            if k not in ["lambda_aux"]
        },
        lambda_aux=0.1,
    )
    model_b2 = EqLM(cfg_b2)
    params_b2 = count_params(model_b2)
    print(f"B2 params: {params_b2:,}")

    optimizer_b2 = torch.optim.AdamW(model_b2.parameters(), lr=lr)
    result_b2 = train_arm(
        "B2",
        model_b2,
        train_loader,
        optimizer_b2,
        device,
        num_steps,
        log_every,
        lambda_aux=0.1,
    )
    arms_results["B2"] = result_b2

    # ===== B3: lambda_aux = 1.0 =====
    print("\n" + "=" * 60)
    print("ARM B3: EqLM-v3 (postln) with lambda_aux=1.0")
    print("=" * 60)

    cfg_b3 = EqLMConfig(
        **{
            k: v
            for k, v in base_config.__dict__.items()
            if k not in ["lambda_aux"]
        },
        lambda_aux=1.0,
    )
    model_b3 = EqLM(cfg_b3)
    params_b3 = count_params(model_b3)
    print(f"B3 params: {params_b3:,}")

    optimizer_b3 = torch.optim.AdamW(model_b3.parameters(), lr=lr)
    result_b3 = train_arm(
        "B3",
        model_b3,
        train_loader,
        optimizer_b3,
        device,
        num_steps,
        log_every,
        lambda_aux=1.0,
    )
    arms_results["B3"] = result_b3

    # ========== Aggregate Results ==========
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nFinal CE Losses:")
    for arm in ["B0", "B1", "B2", "B3"]:
        loss = arms_results[arm]["final_ce_loss"]
        print(f"  {arm}: {loss:.4f}")

    print("\nSolver Rel-Residual at Exit (mean over last 50 steps):")
    for arm in ["B0", "B1", "B2", "B3"]:
        res = arms_results[arm].get("solver_rel_residual_mean_last50", "N/A")
        if isinstance(res, float):
            print(f"  {arm}: {res:.6f}")
        else:
            print(f"  {arm}: {res}")

    print("\nSolver Convergence Rates (% @ tol 1e-3):")
    for arm in ["B0", "B1", "B2", "B3"]:
        conv = arms_results[arm].get("solver_convergence_rate", "N/A")
        if isinstance(conv, float):
            print(f"  {arm}: {conv:.1%}")
        else:
            print(f"  {arm}: {conv}")

    print("\nWall Times:")
    base_time = arms_results["B0"]["total_time_sec"]
    for arm in ["B0", "B1", "B2", "B3"]:
        t = arms_results[arm]["total_time_sec"]
        ratio = t / base_time if base_time > 0 else 1.0
        print(f"  {arm}: {t:.1f}s ({ratio:.2f}x B0)")

    # Preregistered check
    print("\nPRE-REGISTERED CRITERIA:")
    b0_loss = arms_results["B0"]["final_ce_loss"]
    b0_residual = arms_results["B0"].get("solver_rel_residual_mean_last50", float("inf"))

    criterion_met = False
    for arm in ["B1", "B2", "B3"]:
        arm_loss = arms_results[arm]["final_ce_loss"]
        arm_residual = arms_results[arm].get("solver_rel_residual_mean_last50", float("inf"))

        # Check: (a) residual < 0.5x control AND (b) CE within 5% of control
        has_lower_residual = arm_residual < 0.5 * b0_residual if b0_residual < float("inf") else False
        has_low_ce = arm_loss <= b0_loss * 1.05

        if has_lower_residual and has_low_ce:
            criterion_met = True
            print(
                f"  ✓ {arm} satisfies criterion: residual {arm_residual:.6f} < "
                f"0.5x control ({0.5 * b0_residual:.6f}) AND CE {arm_loss:.4f} <= "
                f"1.05x control ({b0_loss * 1.05:.4f})"
            )
        else:
            if not has_lower_residual:
                print(
                    f"  ✗ {arm} residual criterion failed: {arm_residual:.6f} >= "
                    f"0.5x control ({0.5 * b0_residual:.6f})"
                )
            if not has_low_ce:
                print(
                    f"  ✗ {arm} CE criterion failed: {arm_loss:.4f} > "
                    f"1.05x control ({b0_loss * 1.05:.4f})"
                )

    print(f"\n  [Overall] Criterion Met: {criterion_met}")

    # Save results
    results = {
        "config_hash": config_hash,
        "git": git_info,
        "criterion_met": criterion_met,
        "arms": {
            "B0": {
                "name": "EqLM-v3 (postln, lambda_aux=0.0, control)",
                "params": params_b0,
                "final_ce_loss": float(result_b0["final_ce_loss"]),
                "final_aux_loss": float(result_b0["final_aux_loss"]),
                "wall_time_sec": float(result_b0["total_time_sec"]),
                "peak_memory_mb": float(result_b0["peak_memory_mb"]),
                "solver_rel_residual_mean_last50": float(result_b0.get("solver_rel_residual_mean_last50", 0.0)),
                "solver_convergence_rate": float(result_b0.get("solver_convergence_rate", 0.0)),
            },
            "B1": {
                "name": "EqLM-v3 (postln, lambda_aux=0.01)",
                "params": params_b1,
                "final_ce_loss": float(result_b1["final_ce_loss"]),
                "final_aux_loss": float(result_b1["final_aux_loss"]),
                "wall_time_sec": float(result_b1["total_time_sec"]),
                "peak_memory_mb": float(result_b1["peak_memory_mb"]),
                "solver_rel_residual_mean_last50": float(result_b1.get("solver_rel_residual_mean_last50", 0.0)),
                "solver_convergence_rate": float(result_b1.get("solver_convergence_rate", 0.0)),
            },
            "B2": {
                "name": "EqLM-v3 (postln, lambda_aux=0.1, primary)",
                "params": params_b2,
                "final_ce_loss": float(result_b2["final_ce_loss"]),
                "final_aux_loss": float(result_b2["final_aux_loss"]),
                "wall_time_sec": float(result_b2["total_time_sec"]),
                "peak_memory_mb": float(result_b2["peak_memory_mb"]),
                "solver_rel_residual_mean_last50": float(result_b2.get("solver_rel_residual_mean_last50", 0.0)),
                "solver_convergence_rate": float(result_b2.get("solver_convergence_rate", 0.0)),
            },
            "B3": {
                "name": "EqLM-v3 (postln, lambda_aux=1.0)",
                "params": params_b3,
                "final_ce_loss": float(result_b3["final_ce_loss"]),
                "final_aux_loss": float(result_b3["final_aux_loss"]),
                "wall_time_sec": float(result_b3["total_time_sec"]),
                "peak_memory_mb": float(result_b3["peak_memory_mb"]),
                "solver_rel_residual_mean_last50": float(result_b3.get("solver_rel_residual_mean_last50", 0.0)),
                "solver_convergence_rate": float(result_b3.get("solver_convergence_rate", 0.0)),
            },
        },
    }

    # Save JSON results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Plot loss curves
    plot_loss_curves(arms_results, output_dir)


if __name__ == "__main__":
    main()
