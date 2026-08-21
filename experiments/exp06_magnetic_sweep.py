"""EXP06: tau/reference mini-sweep on EqLM+MagneticAdamW.

Tests which (tau, ref_mode) preserves learning while showing measurable drift reduction.
Grid: tau in {1e-4, 1e-3, 1e-2} x ref_mode in {ema(0.999), periodic(100)} + AdamW baseline.
300 steps each to fit budget on GB10.

Metrics:
  - final_loss
  - weight_drift_from_init (L2 norm distance from initialization)
  - solver convergence rate (for EqLM arms)
  - wall time

Prereg (results.json):
  - Exists a cell with final loss within 10% of AdamW baseline
  - AND drift-from-init < 0.9x baseline drift

Output:
  - results.json with config sha, git commit, per-arm metrics
  - loss_vs_tau.pdf (final loss vs tau, grouped by ref_mode)
  - drift_vs_tau.pdf (weight drift vs tau, grouped by ref_mode)

Usage:
    python experiments/exp06_magnetic_sweep.py --config configs/exp06_magnetic_sweep.yaml --output results/exp06_magnetic_sweep
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

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
from kinetic_ai.models.eqlm import (
    EqLM,
    EqLMConfig,
    ExplicitLM,
    count_params,
    match_explicit_width,
)
from kinetic_ai.optim.magnetic_adamw import MagneticAdamW


def load_config(config_path: str) -> dict:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Normalize config values (handle YAML string quirks)
    if "training" in cfg:
        for key in ["lr", "weight_decay", "grad_clip"]:
            if key in cfg["training"] and isinstance(cfg["training"][key], str):
                cfg["training"][key] = float(cfg["training"][key])

    for arm_name, arm_cfg in cfg.get("arms", {}).items():
        if "config" in arm_cfg:
            for key in ["deq_tol"]:
                if key in arm_cfg["config"] and isinstance(arm_cfg["config"][key], str):
                    arm_cfg["config"][key] = float(arm_cfg["config"][key])
        for key in ["magnetic_adamw_tau"]:
            if key in arm_cfg and isinstance(arm_cfg[key], str):
                arm_cfg[key] = float(arm_cfg[key])

    return cfg


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def compute_config_hash(cfg: dict) -> str:
    """Compute SHA256 hash of resolved config."""
    cfg_str = json.dumps(cfg, sort_keys=True, indent=2)
    return hashlib.sha256(cfg_str.encode()).hexdigest()


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
    """Train one arm and track metrics."""
    model.train()
    device_obj = torch.device(device)

    loss_curve = []
    solver_iterations = []
    start_time = time.time()
    peak_memory_mb = 0.0

    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break

        # Get batch
        input_ids = batch["input_ids"].to(device_obj)
        target_ids = batch["target_ids"].to(device_obj)

        # Forward pass
        logits = model(input_ids)  # [B, T, vocab_size]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Metrics
        loss_curve.append(loss.item())
        if step % log_every == 0:
            print(f"  {arm_name} step {step}: loss={loss.item():.4f}")

        # Track solver stats if DEQ model
        if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
            layer = model.transformer.layer
            if hasattr(layer, "deq_block") and hasattr(layer.deq_block, "solver_iterations"):
                solver_iterations.append(layer.deq_block.solver_iterations)

        # Peak memory tracking
        if device == "cuda":
            peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated() / 1e6)

        step += 1

    total_time = time.time() - start_time

    result = {
        "final_loss": loss_curve[-1] if loss_curve else float("nan"),
        "loss_curve": loss_curve,
        "total_steps": step,
        "total_time_sec": total_time,
        "peak_memory_mb": peak_memory_mb,
    }

    if solver_iterations:
        result["solver_mean_iterations"] = float(np.mean(solver_iterations))
        result["solver_convergence_rate"] = float(
            np.mean([1 if it >= 12 else 0 for it in solver_iterations])  # Assume max_iter=12
        )
        result["solver_min_iterations"] = float(np.min(solver_iterations))
        result["solver_max_iterations"] = float(np.max(solver_iterations))

    return result


def compute_weight_drift(model: torch.nn.Module, init_state: dict[str, torch.Tensor]) -> float:
    """Compute L2 norm of weight drift from initialization."""
    total_drift = 0.0
    for name, param in model.named_parameters():
        if name in init_state:
            drift = torch.norm(param - init_state[name]).item()
            total_drift += drift ** 2
    return float(np.sqrt(total_drift))


def plot_loss_vs_tau(
    results_by_arm: dict[str, dict],
    output_dir: Path,
) -> None:
    """Plot final loss vs tau, grouped by ref_mode."""
    tau_values = []
    loss_ema = []
    loss_periodic = []

    # Extract data from results
    for arm_name in sorted(results_by_arm.keys()):
        if arm_name == "Baseline":
            continue

        arm_data = results_by_arm[arm_name]
        tau = arm_data.get("magnetic_adamw_tau")
        ref_mode = arm_data.get("magnetic_adamw_ref_mode", "unknown")
        final_loss = arm_data.get("final_loss")

        if tau is not None and final_loss is not None:
            if ref_mode == "ema":
                if tau not in tau_values:
                    tau_values.append(tau)
                    loss_ema.append(final_loss)
            elif ref_mode == "periodic":
                if tau not in tau_values:
                    loss_periodic.append(final_loss)

    # Baseline loss
    baseline_loss = results_by_arm.get("Baseline", {}).get("final_loss")

    # Plot
    plt.figure(figsize=(10, 6))
    if loss_ema:
        plt.plot(tau_values[: len(loss_ema)], loss_ema, "o-", label="EMA", linewidth=2, markersize=8)
    if loss_periodic:
        plt.plot(tau_values[: len(loss_periodic)], loss_periodic, "s-", label="Periodic", linewidth=2, markersize=8)
    if baseline_loss is not None:
        plt.axhline(baseline_loss, color="red", linestyle="--", label="Baseline (AdamW)", linewidth=2)

    plt.xscale("log")
    plt.xlabel("tau (magnetic strength)")
    plt.ylabel("Final Loss")
    plt.title("EqLM+MagneticAdamW: Loss vs tau")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_vs_tau.pdf", dpi=150)
    plt.close()


def plot_drift_vs_tau(
    results_by_arm: dict[str, dict],
    output_dir: Path,
) -> None:
    """Plot weight drift vs tau, grouped by ref_mode."""
    tau_values = []
    drift_ema = []
    drift_periodic = []

    # Extract data from results
    for arm_name in sorted(results_by_arm.keys()):
        if arm_name == "Baseline":
            continue

        arm_data = results_by_arm[arm_name]
        tau = arm_data.get("magnetic_adamw_tau")
        ref_mode = arm_data.get("magnetic_adamw_ref_mode", "unknown")
        drift = arm_data.get("weight_drift_from_init")

        if tau is not None and drift is not None:
            if ref_mode == "ema":
                if tau not in tau_values:
                    tau_values.append(tau)
                    drift_ema.append(drift)
            elif ref_mode == "periodic":
                if tau not in tau_values:
                    drift_periodic.append(drift)

    # Baseline drift
    baseline_drift = results_by_arm.get("Baseline", {}).get("weight_drift_from_init")

    # Plot
    plt.figure(figsize=(10, 6))
    if drift_ema:
        plt.plot(tau_values[: len(drift_ema)], drift_ema, "o-", label="EMA", linewidth=2, markersize=8)
    if drift_periodic:
        plt.plot(tau_values[: len(drift_periodic)], drift_periodic, "s-", label="Periodic", linewidth=2, markersize=8)
    if baseline_drift is not None:
        plt.axhline(baseline_drift, color="red", linestyle="--", label="Baseline (AdamW)", linewidth=2)
        plt.axhline(baseline_drift * 0.9, color="red", linestyle=":", label="Baseline * 0.9 (target)", linewidth=1)

    plt.xscale("log")
    plt.xlabel("tau (magnetic strength)")
    plt.ylabel("Weight Drift (L2 norm from init)")
    plt.title("EqLM+MagneticAdamW: Weight Drift vs tau")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "drift_vs_tau.pdf", dpi=150)
    plt.close()


def main() -> None:
    """Run exp06: tau/reference mini-sweep."""
    parser = argparse.ArgumentParser(description="EXP06: tau/reference mini-sweep")
    parser.add_argument("--config", type=str, default="configs/exp06_magnetic_sweep.yaml")
    parser.add_argument("--output", type=str, default="results/exp06_magnetic_sweep")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    config_hash = compute_config_hash(cfg)
    git_commit = get_git_commit()

    print("\n" + "=" * 70)
    print("EXP06: tau/reference mini-sweep")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print(f"Config SHA256: {config_hash}")
    print(f"Git commit: {git_commit}")

    # ========================================================================
    # STAGE 1: Load data and tokenizer
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Loading data and tokenizer")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_fn = load_or_build_tokenizer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        prefer_gpt2=cfg["tokenizer"].get("prefer_gpt2", False),
    )
    print(f"Tokenizer vocab size: {cfg['tokenizer']['vocab_size']}")

    # Load dataset
    print(f"Loading dataset: {cfg['data']['dataset']}")
    dataset = load_babylm_dataset(subset_size=cfg["data"].get("subset_size"))

    # Build token stream and dataloader
    token_stream = build_token_stream(dataset, tokenizer_fn)
    train_loader = BabyLMDataLoader(
        token_stream,
        seq_len=cfg["data"]["seq_len"],
        batch_size=cfg["data"]["batch_size"],
        device=device,
    )
    print(f"Data loader: {cfg['data']['batch_size']} batch, {cfg['data']['seq_len']} seq_len")

    # ========================================================================
    # STAGE 2: Initialize models for all arms
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Initializing models")
    print("=" * 70)

    torch.manual_seed(cfg["training"]["seed"])

    results = {
        "config_hash": config_hash,
        "git_commit": git_commit,
        "arms": {},
    }

    models_and_inits = {}  # Store (model, init_state) for each arm

    # Baseline: ExplicitLM with AdamW
    if "Baseline" in cfg["arms"]:
        print("\nInitializing Baseline (ExplicitLM)...")
        arm_cfg = cfg["arms"]["Baseline"]["config"]
        baseline_model = ExplicitLM(
            vocab_size=arm_cfg["vocab_size"],
            d_model=arm_cfg["d_model"],
            n_heads=arm_cfg["n_heads"],
            d_ff=arm_cfg["d_ff"],
            n_layers=4,
            max_seq_len=arm_cfg["max_seq_len"],
            dropout=arm_cfg.get("dropout", 0.1),
        ).to(device)
        baseline_params = count_params(baseline_model)
        print(f"  Params: {baseline_params:,}")
        models_and_inits["Baseline"] = (
            baseline_model,
            {k: v.clone().detach() for k, v in baseline_model.named_parameters()},
        )

    # All other arms: EqLM with different magnetic configs
    for arm_name in sorted(cfg["arms"].keys()):
        if arm_name == "Baseline":
            continue

        print(f"\nInitializing {arm_name}...")
        arm_cfg = cfg["arms"][arm_name]["config"]

        # Match width to baseline
        if "Baseline" in cfg["arms"]:
            base_cfg = cfg["arms"]["Baseline"]["config"]
            matched_d_model, matched_d_ff = match_explicit_width(
                base_cfg["d_model"],
                base_cfg["n_heads"],
                base_cfg["d_ff"],
            )
            arm_cfg["d_model"] = matched_d_model
            arm_cfg["d_ff"] = matched_d_ff
        else:
            matched_d_model, matched_d_ff = match_explicit_width(
                arm_cfg.get("d_model", 192),
                arm_cfg.get("n_heads", 4),
                arm_cfg.get("d_ff", 512),
            )
            arm_cfg["d_model"] = matched_d_model
            arm_cfg["d_ff"] = matched_d_ff

        eqlm_cfg = EqLMConfig(
            vocab_size=arm_cfg["vocab_size"],
            d_model=arm_cfg["d_model"],
            n_heads=arm_cfg["n_heads"],
            d_ff=arm_cfg["d_ff"],
            max_seq_len=arm_cfg["max_seq_len"],
            deq_max_iter=arm_cfg.get("deq_max_iter", 12),
            deq_tol=arm_cfg.get("deq_tol", 1e-3),
            solver=arm_cfg.get("solver", "anderson"),
            jfb=arm_cfg.get("jfb", False),
            dropout=arm_cfg.get("dropout", 0.1),
        )
        model = EqLM(eqlm_cfg).to(device)
        num_params = count_params(model)
        print(f"  Params: {num_params:,}")
        models_and_inits[arm_name] = (
            model,
            {k: v.clone().detach() for k, v in model.named_parameters()},
        )

    # ========================================================================
    # STAGE 3: Train all arms (sequential)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Training all arms (sequential)")
    print("=" * 70)

    num_steps = cfg["training"]["num_steps"]
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    log_every = cfg["training"]["log_every"]

    for arm_name in sorted(cfg["arms"].keys()):
        print(f"\n--- Training {arm_name} ({num_steps} steps) ---")

        model, init_state = models_and_inits[arm_name]
        arm_cfg = cfg["arms"][arm_name]

        # Create optimizer
        if arm_cfg["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"].get("weight_decay", 0.01),
            )
        elif arm_cfg["optimizer"] == "magnetic_adamw":
            tau = arm_cfg.get("magnetic_adamw_tau", 0.0)
            ref_mode = arm_cfg.get("magnetic_adamw_ref_mode", "ema")
            ref_beta = arm_cfg.get("magnetic_adamw_ref_beta", 0.999)
            ref_interval = arm_cfg.get("magnetic_adamw_ref_interval", 10)

            optimizer = MagneticAdamW(
                model.parameters(),
                lr=cfg["training"]["lr"],
                weight_decay=cfg["training"].get("weight_decay", 0.01),
                tau=tau,
                ref_mode=ref_mode,
                ref_beta=ref_beta,
                ref_interval=ref_interval,
            )
        else:
            raise ValueError(f"Unknown optimizer: {arm_cfg['optimizer']}")

        # Train
        arm_results = train_arm(
            arm_name,
            model,
            train_loader,
            optimizer,
            device,
            num_steps,
            log_every,
            grad_clip=grad_clip,
        )

        # Compute weight drift
        weight_drift = compute_weight_drift(model, init_state)
        arm_results["weight_drift_from_init"] = weight_drift

        # Store results
        results["arms"][arm_name] = {
            **arm_results,
            "model_name": arm_cfg.get("name", arm_name),
            "num_params": count_params(model),
            "optimizer": arm_cfg["optimizer"],
        }

        # Add magnetic config if present
        if arm_cfg["optimizer"] == "magnetic_adamw":
            results["arms"][arm_name]["magnetic_adamw_tau"] = arm_cfg.get("magnetic_adamw_tau", 0.0)
            results["arms"][arm_name]["magnetic_adamw_ref_mode"] = arm_cfg.get("magnetic_adamw_ref_mode", "ema")

        print(
            f"  Done: {arm_results['total_time_sec']:.1f}s, "
            f"loss {arm_results['final_loss']:.4f}, "
            f"drift {weight_drift:.4f}"
        )

    # ========================================================================
    # STAGE 4: Compute prereg metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: Computing prereg metrics")
    print("=" * 70)

    baseline_loss = results["arms"].get("Baseline", {}).get("final_loss")
    baseline_drift = results["arms"].get("Baseline", {}).get("weight_drift_from_init")

    print(f"Baseline loss: {baseline_loss:.4f}")
    print(f"Baseline drift: {baseline_drift:.4f}")

    # Prereg check 1: exists cell with loss within 10% of baseline
    loss_within_10pct = False
    for arm_name, arm_data in results["arms"].items():
        if arm_name == "Baseline":
            continue
        final_loss = arm_data.get("final_loss")
        if final_loss is not None and baseline_loss is not None:
            loss_diff_pct = abs(final_loss - baseline_loss) / baseline_loss * 100
            if loss_diff_pct <= 10.0:
                loss_within_10pct = True
                print(f"  ✓ {arm_name}: loss {final_loss:.4f} (within 10% of baseline)")

    # Prereg check 2: exists cell with drift < 0.9x baseline
    drift_below_90pct = False
    for arm_name, arm_data in results["arms"].items():
        if arm_name == "Baseline":
            continue
        drift = arm_data.get("weight_drift_from_init")
        if drift is not None and baseline_drift is not None:
            if drift < baseline_drift * 0.9:
                drift_below_90pct = True
                print(f"  ✓ {arm_name}: drift {drift:.4f} (below 90% of baseline)")

    results["prereg"] = {
        "loss_within_10pct_of_baseline": loss_within_10pct,
        "drift_below_90pct_of_baseline": drift_below_90pct,
        "baseline_loss": float(baseline_loss) if baseline_loss is not None else None,
        "baseline_drift": float(baseline_drift) if baseline_drift is not None else None,
    }

    print(f"\nPrereg check 1 (loss within 10%): {loss_within_10pct}")
    print(f"Prereg check 2 (drift < 90%): {drift_below_90pct}")

    # ========================================================================
    # STAGE 5: Generate plots and save results
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: Generating plots and saving results")
    print("=" * 70)

    try:
        plot_loss_vs_tau(results["arms"], output_dir)
        print("  Saved loss_vs_tau.pdf")
    except Exception as e:
        print(f"  Warning: Failed to plot loss vs tau: {e}")

    try:
        plot_drift_vs_tau(results["arms"], output_dir)
        print("  Saved drift_vs_tau.pdf")
    except Exception as e:
        print(f"  Warning: Failed to plot drift vs tau: {e}")

    # Save results JSON
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results.json")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for arm_name in sorted(results["arms"].keys()):
        arm_data = results["arms"][arm_name]
        print(f"\n{arm_name}:")
        print(f"  Loss: {arm_data.get('final_loss', 'N/A'):.4f}")
        print(f"  Drift: {arm_data.get('weight_drift_from_init', 'N/A'):.4f}")
        print(f"  Time: {arm_data.get('total_time_sec', 'N/A'):.1f}s")
        if "magnetic_adamw_tau" in arm_data:
            print(f"  tau: {arm_data['magnetic_adamw_tau']}, ref_mode: {arm_data.get('magnetic_adamw_ref_mode')}")

    print("\n" + "=" * 70)
    print("✓ EXP06 complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
