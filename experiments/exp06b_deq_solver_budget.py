"""EXP06b: DEQ Solver Budget Study (phantom-gradient characterization).

On same EqLM smoke model: train 300 steps with max_iter in {12, 24, 48} and tol 1e-3 (AdamW).
Characterizes whether unconverged (phantom-gradient) training hurts final loss.

Metrics (per max_iter):
  - final_loss
  - solver_convergence_rate (% of steps where solver converged)
  - solver_mean_iterations (average iterations per forward pass)
  - solver_mean_residual (mean residual at exit)
  - wall_time (total training time)

Prereg:
  - Report the loss-vs-budget curve honestly (does loss degrade with unconverged solver?)
  - Characterize whether training succeeds (loss decreases) despite phantom gradients

Output:
  - results.json with per-budget metrics
  - solver_budget_curve.pdf (loss and convergence rate vs max_iter)

Usage:
    python experiments/exp06b_deq_solver_budget.py --output results/exp06_magnetic_sweep/solver_budget
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from exp05_eqlm_pretrain import create_gpt2_tokenizer_fn  # noqa: E402

from kinetic_ai.data import (
    BabyLMDataLoader,
    build_token_stream,
    load_babylm_dataset,
    load_or_build_tokenizer,
)
from kinetic_ai.models.eqlm import (
    EqLM,
    EqLMConfig,
    count_params,
)


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


def train_with_solver_budget(
    model: torch.nn.Module,
    train_loader: BabyLMDataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_steps: int,
    log_every: int,
    grad_clip: float = 1.0,
) -> dict:
    """Train and track solver statistics."""
    model.train()
    device_obj = torch.device(device)

    loss_curve = []
    solver_convergences = []  # 1 if converged, 0 if not
    solver_iterations_list = []
    solver_residuals = []
    start_time = time.time()
    peak_memory_mb = 0.0

    step = 0
    for batch in train_loader:
        if step >= num_steps:
            break

        # Get batch: loader yields [B, T] token tensors; shift for next-token LM
        batch = batch.to(device_obj)
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

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
            print(f"  Step {step}: loss={loss.item():.4f}")

        # Track solver stats (exp05 pattern: model.deq.last_info)
        _info = getattr(getattr(model, "deq", None), "last_info", None)
        if _info:
            solver_iterations_list.append(int(_info.get("iterations", 0)))
            solver_convergences.append(1 if _info.get("converged") else 0)
            _res = _info.get("residuals")
            if _res:
                solver_residuals.append(float(_res[-1]))

        # Peak memory
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

    if solver_convergences:
        result["solver_convergence_rate"] = float(np.mean(solver_convergences))

    if solver_iterations_list:
        result["solver_mean_iterations"] = float(np.mean(solver_iterations_list))
        result["solver_min_iterations"] = float(np.min(solver_iterations_list))
        result["solver_max_iterations"] = float(np.max(solver_iterations_list))

    if solver_residuals:
        result["solver_mean_residual"] = float(np.mean(solver_residuals))
        result["solver_min_residual"] = float(np.min(solver_residuals))
        result["solver_max_residual"] = float(np.max(solver_residuals))

    return result


def plot_solver_budget_curve(
    results_by_budget: dict[int, dict],
    output_dir: Path,
) -> None:
    """Plot loss and convergence rate vs solver budget (max_iter)."""
    budgets = sorted(results_by_budget.keys())
    losses = [results_by_budget[b].get("final_loss") for b in budgets]
    convergence_rates = [results_by_budget[b].get("solver_convergence_rate", 0.0) for b in budgets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs budget
    ax1.plot(budgets, losses, "o-", linewidth=2, markersize=10, color="#0173B2")
    ax1.set_xlabel("max_iter (solver budget)")
    ax1.set_ylabel("Final Loss")
    ax1.set_title("Loss vs Solver Budget (max_iter)")
    ax1.grid(True, alpha=0.3)

    # Convergence rate vs budget
    ax2.plot(budgets, convergence_rates, "s-", linewidth=2, markersize=10, color="#DE8F05")
    ax2.set_xlabel("max_iter (solver budget)")
    ax2.set_ylabel("Convergence Rate (%)")
    ax2.set_ylim([0, 1.05])
    ax2.set_title("DEQ Solver Convergence Rate vs Budget")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "solver_budget_curve.pdf", dpi=150)
    plt.close()

    print("  Saved solver_budget_curve.pdf")


def main() -> None:
    """Run exp06b: DEQ solver budget study."""
    parser = argparse.ArgumentParser(description="EXP06b: DEQ Solver Budget Study")
    parser.add_argument("--output", type=str, default="results/exp06_magnetic_sweep/solver_budget")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()

    print("\n" + "=" * 70)
    print("EXP06b: DEQ Solver Budget Study (phantom-gradient characterization)")
    print("=" * 70)
    print(f"Output: {output_dir}")
    print(f"Git commit: {git_commit}")

    # ========================================================================
    # STAGE 1: Setup (data, tokenizer, device)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 1: Loading data and setup")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tokenizer (same working path as exp05)
    token2id, _id2token, tokenizer_choice = load_or_build_tokenizer(texts=None)
    actual_vocab_size = len(token2id)
    print(f"Tokenizer: {tokenizer_choice} | vocab_size {actual_vocab_size}")
    tokenizer_fn = create_gpt2_tokenizer_fn(tokenizer_choice)

    # Dataset (same as exp05)
    print("Loading dataset...")
    dataset = load_babylm_dataset(subset="BabyLM-2026-Strict-Small", max_samples=None)
    print(f"Loaded dataset: {len(dataset)} samples")

    # Data loader
    token_tensor, _num_seqs = build_token_stream(
        dataset, tokenizer_fn, seq_len=128, max_tokens=3300000
    )
    train_loader = BabyLMDataLoader(
        token_tensor,
        batch_size=32,
        shuffle=True,
        device=device,
    )
    print("Data loader: batch=32, seq_len=128")

    # ========================================================================
    # STAGE 2: Train with different solver budgets
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Training with different solver budgets")
    print("=" * 70)

    torch.manual_seed(42)
    results = {
        "git_commit": git_commit,
        "solver_budgets": {},
    }

    num_steps = 300
    grad_clip = 1.0
    log_every = 25

    for max_iter in [12, 24, 48]:
        print(f"\n--- Training with max_iter={max_iter} ---")

        # Create fresh EqLM model for this budget
        eqlm_cfg = EqLMConfig(
            vocab_size=actual_vocab_size,
            d_model=192,
            n_heads=4,
            d_ff=512,
            max_seq_len=128,
            deq_max_iter=max_iter,  # Variable budget
            deq_tol=1e-3,
            solver="anderson",
            jfb=False,
            dropout=0.1,
        )
        model = EqLM(eqlm_cfg).to(device)
        num_params = count_params(model)
        print(f"  Model params: {num_params:,}")

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )

        # Train
        budget_results = train_with_solver_budget(
            model,
            train_loader,
            optimizer,
            device,
            num_steps,
            log_every,
            grad_clip=grad_clip,
        )

        results["solver_budgets"][max_iter] = {
            **budget_results,
            "model_config": {
                "d_model": 192,
                "n_heads": 4,
                "d_ff": 512,
                "max_iter": max_iter,
                "tol": 1e-3,
            },
        }

        print(
            f"  Done: {budget_results['total_time_sec']:.1f}s, "
            f"loss {budget_results['final_loss']:.4f}, "
            f"convergence {budget_results.get('solver_convergence_rate', 0.0):.1%}"
        )

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

    # ========================================================================
    # STAGE 3: Analysis and plots
    # ========================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: Analysis and plots")
    print("=" * 70)

    # Compute loss diff from baseline (max_iter=12)
    baseline_loss = results["solver_budgets"][12].get("final_loss")
    print(f"\nBaseline (max_iter=12) loss: {baseline_loss:.4f}")

    for max_iter in sorted(results["solver_budgets"].keys()):
        data = results["solver_budgets"][max_iter]
        loss = data.get("final_loss")
        convergence = data.get("solver_convergence_rate", 0.0)
        mean_iter = data.get("solver_mean_iterations", 0.0)

        loss_diff = (loss - baseline_loss) / baseline_loss * 100 if baseline_loss else 0.0
        print(
            f"max_iter={max_iter}: loss={loss:.4f} ({loss_diff:+.1f}%), "
            f"convergence={convergence:.1%}, mean_iter={mean_iter:.1f}"
        )

    # Preregs
    print("\n--- Prereg Metrics ---")
    results["prereg"] = {
        "baseline_loss": float(baseline_loss) if baseline_loss else None,
        "loss_degradation_at_max_iter_24": float(
            (results["solver_budgets"][24].get("final_loss", baseline_loss) - baseline_loss) / baseline_loss
            if baseline_loss
            else 0.0
        ),
        "loss_degradation_at_max_iter_48": float(
            (results["solver_budgets"][48].get("final_loss", baseline_loss) - baseline_loss) / baseline_loss
            if baseline_loss
            else 0.0
        ),
        "convergence_rate_at_max_iter_12": float(results["solver_budgets"][12].get("solver_convergence_rate", 0.0)),
        "convergence_rate_at_max_iter_24": float(results["solver_budgets"][24].get("solver_convergence_rate", 0.0)),
        "convergence_rate_at_max_iter_48": float(results["solver_budgets"][48].get("solver_convergence_rate", 0.0)),
    }

    # Plot
    try:
        plot_solver_budget_curve(results["solver_budgets"], output_dir)
    except Exception as e:
        print(f"  Warning: Failed to plot: {e}")

    # Save results
    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved results.json")

    # ========================================================================
    # SUMMARY AND INTERPRETATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Phantom-Gradient Impact on Training")
    print("=" * 70)

    converged_12 = results["solver_budgets"][12].get("solver_convergence_rate", 0.0)
    converged_24 = results["solver_budgets"][24].get("solver_convergence_rate", 0.0)
    converged_48 = results["solver_budgets"][48].get("solver_convergence_rate", 0.0)

    loss_12 = results["solver_budgets"][12].get("final_loss")
    loss_24 = results["solver_budgets"][24].get("final_loss")
    loss_48 = results["solver_budgets"][48].get("final_loss")

    print("\nConvergence rates:")
    print(f"  max_iter=12: {converged_12:.1%}")
    print(f"  max_iter=24: {converged_24:.1%}")
    print(f"  max_iter=48: {converged_48:.1%}")

    print("\nFinal losses:")
    print(f"  max_iter=12: {loss_12:.4f}")
    print(f"  max_iter=24: {loss_24:.4f}")
    print(f"  max_iter=48: {loss_48:.4f}")

    if converged_12 < 0.5:
        print("\n[FINDING] max_iter=12 is MOSTLY UNCONVERGED (phantom-gradient training)")
        if abs((loss_24 - loss_12) / loss_12) < 0.05:
            print(f"[FINDING] Yet loss barely improves at max_iter=24 ({(loss_24 - loss_12) / loss_12:.1%})")
            print("[CONCLUSION] Phantom gradients are tolerable for EqLM smoke training")
        else:
            print(f"[FINDING] Loss degrades significantly at max_iter=24 ({(loss_24 - loss_12) / loss_12:.1%})")
            print("[CONCLUSION] Phantom gradients harm training; need higher budget for Tier B")

    print("\n✓ EXP06b complete!")


if __name__ == "__main__":
    main()
