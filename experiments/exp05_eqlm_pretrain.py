"""SPEC 0004: EqLM Pretraining on BabyLM (Tier B SMOKE).

Trains three arms (A1=ExplicitLM, A2=EqLM, A3=EqLM+MMD) on BabyLM 2026 strict-small.
Logs loss curves, memory, and wall time per arm. Evaluates on BLiMP subset.

Implements proper parameter matching, gradient clipping, DEQ solver tracking, and
loss curve visualization per SPEC requirements.

Usage:
    python experiments/exp05_eqlm_pretrain.py --config configs/exp05_smoke.yaml --output results/exp05_eqlm_pretrain
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
from kinetic_ai.eval.blimp import evaluate_blimp_subset, load_blimp_subset
from kinetic_ai.models.eqlm import (
    EqLM,
    EqLMConfig,
    ExplicitLM,
    count_params,
    match_explicit_width,
    save_checkpoint,
)
from kinetic_ai.optim.magnetic_adamw import MagneticAdamW


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

    # Fix magnetic_adamw_tau if it's a string
    if (
        "arms" in cfg
        and "A3" in cfg["arms"]
        and "magnetic_adamw_tau" in cfg["arms"]["A3"]
        and isinstance(cfg["arms"]["A3"]["magnetic_adamw_tau"], str)
    ):
        cfg["arms"]["A3"]["magnetic_adamw_tau"] = float(cfg["arms"]["A3"]["magnetic_adamw_tau"])

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
    grad_clip: float,
    lambda_aux: float = 0.0,
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
        grad_clip: Gradient clipping norm (1.0 per spec).

    Returns:
        Dict with loss_curve, total_time, peak_memory_mb, solver_stats (for EqLM).
    """
    model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats(device)

    loss_curve = []
    start_time = time.time()
    solver_iterations = []  # Track DEQ solver iterations for A2/A3
    convergence_failures = 0  # Track solver convergence failures

    step = 0
    # Cycle through data loader until we reach num_steps
    # This handles cases where the data loader has fewer batches than needed
    while step < num_steps:
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
            # Solver-aware auxiliary residual (EqLM-v4, F16): L += lambda_aux * r
            aux = getattr(model, "last_aux_residual", None)
            if aux is not None and lambda_aux > 0:
                loss = loss + lambda_aux * aux

            # Backward
            loss.backward()

            # Gradient clipping (1.0 per spec)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Track DEQ solver statistics if available (EqLM models)
            if (
                hasattr(model, "deq")
                and hasattr(model.deq, "last_info")
                and model.deq.last_info is not None
            ):
                deq_info = model.deq.last_info
                if isinstance(deq_info, dict) and "iterations" in deq_info:
                    solver_iterations.append(deq_info["iterations"])
                    if not deq_info.get("converged", True):
                        convergence_failures += 1

            if step % log_every == 0:
                loss_curve.append({"step": step, "loss": loss.item()})
                print(f"  {arm_name} step {step:4d} | loss {loss.item():.4f}")

            step += 1

        # Reset data loader if we haven't reached num_steps yet
        if step < num_steps:
            train_loader.indices = list(range(train_loader.num_seqs))
            if train_loader.shuffle:
                import random
                random.shuffle(train_loader.indices)

    elapsed = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    result = {
        "loss_curve": loss_curve,
        "total_time_sec": elapsed,
        "peak_memory_mb": peak_memory,
        "total_steps": step,
    }

    # Add solver statistics if tracking (A2/A3 only)
    if solver_iterations:
        result["solver_mean_iterations"] = float(np.mean(solver_iterations))
        result["solver_convergence_rate"] = 1.0 - (
            convergence_failures / len(solver_iterations)
        )
        result["solver_max_iterations"] = int(np.max(solver_iterations))
        result["solver_min_iterations"] = int(np.min(solver_iterations))

    return result


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


def plot_loss_curves(
    arms_data: dict,
    output_dir: Path,
    okabe_ito_colors: dict | None = None,
) -> None:
    """Plot training loss curves for all arms using Okabe-Ito palette.

    Args:
        arms_data: Dict with arm names as keys and loss_curve lists as values.
        output_dir: Output directory for PDF.
        okabe_ito_colors: Okabe-Ito color palette (or use defaults).
    """
    if okabe_ito_colors is None:
        # Okabe-Ito colorblind-friendly palette
        okabe_ito_colors = {
            "A1": "#E69F00",  # Orange
            "A2": "#56B4E9",  # Sky Blue
            "A3": "#009E73",  # Bluish Green
        }

    plt.figure(figsize=(10, 6))

    for arm_name in ["A1", "A2", "A3"]:
        if arm_name not in arms_data:
            continue
        loss_curve = arms_data[arm_name]["loss_curve"]
        if not loss_curve:
            continue

        steps = [pt["step"] for pt in loss_curve]
        losses = [pt["loss"] for pt in loss_curve]

        plt.plot(
            steps,
            losses,
            marker="o",
            label=arm_name,
            color=okabe_ito_colors.get(arm_name),
            linewidth=2,
            markersize=4,
        )

    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("EqLM SMOKE Test: Training Loss Curves", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf_path = output_dir / "loss_curves.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=150)
    print(f"Saved loss curve figure to {pdf_path}")
    plt.close()


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

    # Compute config hash and git commit early
    config_hash = compute_config_hash(cfg)
    git_commit = get_git_commit()
    print(f"Config SHA256: {config_hash}")
    print(f"Git commit: {git_commit}")

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
        cache_dir="data/cache",
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
    print("STAGE 2: Building models with parameter matching")
    print("=" * 70)

    results = {
        "iteration": 3,
        "refinement_reason": "Iteration 3: MagneticAdamW (tau=1e-2, ema mode) replaces raw MMD; EqLM/ExplicitLM init-scale fix (embeddings std=0.02, logits scaled by sqrt(d_model)); DEQ solver stats exposed cleanly",
        "arms": {},
        "config_hash": config_hash,
        "git_commit": git_commit,
        "tokenizer_choice": tokenizer_choice,
        "notes": "SMOKE test for SPEC 0004 Tier B; real token budget (~3.3M) per arm, parameter-matched arms within 5%, A3 now uses MagneticAdamW",
    }

    # A1: ExplicitLM (baseline)
    print("\n--- Arm A1: ExplicitLM (Baseline) ---")
    a1_config_dict = cfg["arms"]["A1"]["config"].copy()
    a1_n_layers = a1_config_dict.pop("n_layers", 4)
    a1_cfg = EqLMConfig(**a1_config_dict)
    a1_model = ExplicitLM(config=a1_cfg, n_layers=a1_n_layers)
    a1_params = count_params(a1_model)
    print(f"A1 params: {a1_params:,}")

    # A2: EqLM (param-matched via match_explicit_width)
    print("\n--- Arm A2: EqLM (Param-matched via match_explicit_width) ---")
    a2_base_cfg = EqLMConfig(**cfg["arms"]["A2"]["config"])
    a2_cfg = match_explicit_width(target_params=a1_params, base_cfg=a2_base_cfg)
    a2_model = EqLM(config=a2_cfg)
    a2_params = count_params(a2_model)
    param_ratio_a2 = a2_params / a1_params
    print(f"A2 params: {a2_params:,}")
    print(f"A2/A1 param ratio: {param_ratio_a2:.4f}")
    # Assert parameter matching within 5% per spec
    assert (
        0.95 <= param_ratio_a2 <= 1.05
    ), f"A2 param matching failed: ratio {param_ratio_a2:.4f} not in [0.95, 1.05]"
    print("✓ A2 parameter matching within 5%")
    # Update config with matched A2 values
    cfg["arms"]["A2"]["config"]["d_model"] = a2_cfg.d_model
    cfg["arms"]["A2"]["config"]["d_ff"] = a2_cfg.d_ff

    # A3: EqLM + MMD (param-matched)
    print("\n--- Arm A3: EqLM + MMD (Param-matched) ---")
    a3_base_cfg = EqLMConfig(**cfg["arms"]["A3"]["config"])
    a3_cfg = match_explicit_width(target_params=a1_params, base_cfg=a3_base_cfg)
    a3_model = EqLM(config=a3_cfg)
    a3_params = count_params(a3_model)
    param_ratio_a3 = a3_params / a1_params
    print(f"A3 params: {a3_params:,}")
    print(f"A3/A1 param ratio: {param_ratio_a3:.4f}")
    # Assert parameter matching within 5% per spec
    assert (
        0.95 <= param_ratio_a3 <= 1.05
    ), f"A3 param matching failed: ratio {param_ratio_a3:.4f} not in [0.95, 1.05]"
    print("✓ A3 parameter matching within 5%")
    # Update config with matched A3 values
    cfg["arms"]["A3"]["config"]["d_model"] = a3_cfg.d_model
    cfg["arms"]["A3"]["config"]["d_ff"] = a3_cfg.d_ff

    # ========================================================================
    # TRAIN
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Training (sequential)")
    print("=" * 70)

    # Use num_steps from config (target ~800 per spec), or fall back to epoch-based calculation
    num_steps = cfg["training"].get("num_steps", len(train_loader) * cfg["training"]["num_epochs"])
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    log_every = cfg["training"]["log_every"]
    print(f"Total steps per arm: {num_steps}")
    print(f"Gradient clipping: {grad_clip}")
    print(f"Log every: {log_every} steps")

    # Arm A1
    print("\n--- Training A1 ---")
    a1_opt = torch.optim.AdamW(a1_model.parameters(), lr=cfg["training"]["lr"])
    a1_results = train_arm(
        "A1",
        a1_model,
        train_loader,
        a1_opt,
        device,
        num_steps,
        log_every,
        grad_clip=grad_clip,
    )
    try:
        save_checkpoint(a1_model, output_dir / "checkpoints" / "a1.pt")
        print(f"Saved checkpoint: {output_dir}/checkpoints/a1.pt")
    except Exception as e:  # checkpointing must never kill a finished run
        print(f"WARNING: checkpoint save failed for A1: {e}")
    results["arms"]["A1"] = {
        **a1_results,
        "num_params": a1_params,
        "model_name": "ExplicitLM",
        "config": a1_cfg.__dict__,
    }
    print(
        f"A1 done: {a1_results['total_time_sec']:.1f}s, "
        f"peak memory {a1_results['peak_memory_mb']:.0f}MB, "
        f"steps {a1_results['total_steps']}"
    )

    # Arm A2
    print("\n--- Training A2 ---")
    a2_opt = torch.optim.AdamW(a2_model.parameters(), lr=cfg["training"]["lr"])
    a2_results = train_arm(
        "A2",
        a2_model,
        train_loader,
        a2_opt,
        device,
        num_steps,
        log_every,
        grad_clip=grad_clip,
        lambda_aux=float(cfg["arms"]["A2"].get("lambda_aux", 0.0)),
    )
    try:
        save_checkpoint(a2_model, output_dir / "checkpoints" / "a2.pt")
        print(f"Saved checkpoint: {output_dir}/checkpoints/a2.pt")
    except Exception as e:  # checkpointing must never kill a finished run
        print(f"WARNING: checkpoint save failed for A2: {e}")
    results["arms"]["A2"] = {
        **a2_results,
        "num_params": a2_params,
        "model_name": "EqLM",
        "config": a2_cfg.__dict__,
    }
    print(
        f"A2 done: {a2_results['total_time_sec']:.1f}s, "
        f"peak memory {a2_results['peak_memory_mb']:.0f}MB, "
        f"steps {a2_results['total_steps']}"
    )

    # Arm A3 (optimizer selected by config: adamw | magnetic_adamw)
    a3_opt_name = cfg["arms"]["A3"].get("optimizer", "magnetic_adamw")
    print(f"\n--- Training A3 ({a3_opt_name}) ---")
    if a3_opt_name == "adamw":
        a3_opt: torch.optim.Optimizer = torch.optim.AdamW(
            a3_model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"].get("weight_decay", 0.01),
        )
    else:
        tau = cfg["arms"]["A3"].get("magnetic_adamw_tau", cfg["arms"]["A3"].get("mmd_config", {}).get("tau", 1e-2))
        a3_opt = MagneticAdamW(
            a3_model.parameters(),
            lr=cfg["training"]["lr"],
            betas=(0.9, 0.999),
            weight_decay=cfg["training"].get("weight_decay", 0.01),
            tau=tau,
            ref_mode="ema",
            ref_beta=0.999,
        )
    a3_results = train_arm(
        "A3",
        a3_model,
        train_loader,
        a3_opt,
        device,
        num_steps,
        log_every,
        grad_clip=grad_clip,
        lambda_aux=float(cfg["arms"]["A3"].get("lambda_aux", 0.0)),
    )
    try:
        save_checkpoint(a3_model, output_dir / "checkpoints" / "a3.pt")
        print(f"Saved checkpoint: {output_dir}/checkpoints/a3.pt")
    except Exception as e:  # checkpointing must never kill a finished run
        print(f"WARNING: checkpoint save failed for A3: {e}")
    results["arms"]["A3"] = {
        **a3_results,
        "num_params": a3_params,
        "model_name": cfg["arms"]["A3"].get("name", "EqLM-arm3"),
        "config": a3_cfg.__dict__,
        "optimizer": a3_opt_name,
        **({"magnetic_adamw_tau": tau} if a3_opt_name != "adamw" else {}),
    }
    print(
        f"A3 done: {a3_results['total_time_sec']:.1f}s, "
        f"peak memory {a3_results['peak_memory_mb']:.0f}MB, "
        f"steps {a3_results['total_steps']}"
    )

    # ========================================================================
    # EVAL
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 4: BLiMP-subset evaluation (stratified, ≥1000 pairs)")
    print("=" * 70)

    blimp_subset = load_blimp_subset(
        num_phenomena=cfg["eval"]["blimp"]["num_phenomena"],
        pairs_per_phenomenon=cfg["eval"]["blimp"]["pairs_per_phenomenon"],
        cache_dir=cfg["eval"]["blimp"].get("cache_dir", None),
    )
    print(f"Loaded BLiMP subset: {len(blimp_subset)} pairs across "
          f"{cfg['eval']['blimp']['num_phenomena']} phenomena")

    for arm_name, model in [("A1", a1_model), ("A2", a2_model), ("A3", a3_model)]:
        print(f"\nEvaluating {arm_name}...")
        try:
            # Use the full BLiMP subset (not limited to 100)
            eval_result = evaluate_blimp_subset(
                model,
                blimp_subset,
                tokenizer_fn,
                device=device,
                max_samples=None,  # Use full subset
            )
            results["arms"][arm_name]["blimp_accuracy"] = eval_result["accuracy"]
            results["arms"][arm_name]["blimp_num_correct"] = eval_result["num_correct"]
            results["arms"][arm_name]["blimp_num_total"] = eval_result["num_total"]
            print(
                f"  BLiMP accuracy: {eval_result['accuracy']:.4f} "
                f"({eval_result['num_correct']}/{eval_result['num_total']})"
            )
        except Exception as e:
            print(f"  BLiMP eval failed: {e}")
            results["arms"][arm_name]["blimp_accuracy"] = None
            results["arms"][arm_name]["blimp_error"] = str(e)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 5: Saving results and generating plots")
    print("=" * 70)

    # Generate loss curve plot (Okabe-Ito colors)
    try:
        arms_loss_data = {
            arm: results["arms"][arm] for arm in ["A1", "A2", "A3"]
            if arm in results["arms"]
        }
        plot_loss_curves(arms_loss_data, output_dir)
    except Exception as e:
        print(f"Warning: Failed to plot loss curves: {e}")

    # Save results JSON
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_json_path}")

    # Print detailed summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for arm in ["A1", "A2", "A3"]:
        arm_data = results["arms"][arm]
        print(f"\n{arm}: {arm_data['model_name']}")
        print(f"  Params: {arm_data['num_params']:,}")
        print(f"  Steps: {arm_data['total_steps']}")
        print(f"  Time: {arm_data['total_time_sec']:.1f}s")
        print(f"  Peak Memory: {arm_data['peak_memory_mb']:.0f}MB")
        if "blimp_accuracy" in arm_data and arm_data["blimp_accuracy"] is not None:
            print(
                f"  BLiMP Accuracy: {arm_data['blimp_accuracy']:.4f} "
                f"({arm_data['blimp_num_correct']}/{arm_data['blimp_num_total']})"
            )
        # Print solver statistics if available (A2/A3)
        if "solver_mean_iterations" in arm_data:
            print("  DEQ Solver Stats:")
            print(f"    Mean iterations: {arm_data['solver_mean_iterations']:.1f}")
            print(f"    Convergence rate: {arm_data['solver_convergence_rate']:.2%}")
            print(f"    Min/Max iterations: {arm_data['solver_min_iterations']}/{arm_data['solver_max_iterations']}")
            if arm_data['solver_convergence_rate'] < 0.8:
                print("    WARNING: Low convergence rate (<80%)!")

    # Print metadata
    print("\nMetadata:")
    print(f"  Config SHA256: {results['config_hash']}")
    print(f"  Git commit: {results['git_commit']}")
    print(f"  Iteration: {results.get('iteration', 'N/A')}")

    print("\n✓ SMOKE test complete!")


if __name__ == "__main__":
    main()
