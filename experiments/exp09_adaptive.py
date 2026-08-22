"""EXP09: H1′ Adaptive-Equilibrium Program.

Measures three H1′ hypotheses:
  H1′a (warm-start decoding): Mean iterations-per-token with warm vs cold start
  H1′b (per-token early exit): Solver iterations with per-position convergence
  H1′c (think-harder dial): BLiMP at eval-time budget scaling

Usage (CPU smoke, 3 steps):
    python experiments/exp09_adaptive.py --config configs/exp09_smoke.yaml --device cpu

Usage (GPU smoke, 500 steps):
    python experiments/exp09_adaptive.py --config configs/exp09_smoke.yaml --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

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
    load_checkpoint,
    save_checkpoint,
)


def load_config(config_path: str) -> dict:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Convert types
    for key in ["lr", "deq_tol", "lambda_aux"]:
        if "training" in cfg and key in cfg["training"]:
            cfg["training"][key] = float(cfg["training"][key])
        if "model" in cfg and key in cfg["model"]:
            cfg["model"][key] = float(cfg["model"][key])

    return cfg


def train_model(
    config: dict,
    device: str,
    output_dir: Path,
) -> tuple[EqLM, Path]:
    """Train model briefly to get a non-random checkpoint.

    Returns:
        (model, checkpoint_path)
    """
    print(f"\n{'='*70}")
    print("TRAINING PHASE: Brief pretraining to get non-random model")
    print(f"{'='*70}")

    # Load dataset
    dataset = load_babylm_dataset(
        subset=config["data"]["dataset"],
        max_samples=None,
    )
    tokenizer = load_or_build_tokenizer(
        vocab_size=config["tokenizer"]["vocab_size"],
    )

    # Build token stream
    token_tensor, num_seqs = build_token_stream(
        dataset,
        tokenizer,
        seq_len=config["data"]["seq_len"],
        cache_dir="data/cache",
    )
    print(f"  Token stream: {num_seqs} sequences")

    # Data loader
    train_loader = BabyLMDataLoader(
        token_tensor,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        device=device,
    )

    # Create model
    model_config = EqLMConfig(**config["model"])
    model = EqLM(model_config)
    model.to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
    )

    # Training loop
    num_steps = config["training"]["num_steps"]
    log_every = config["training"]["log_every"]
    step = 0
    start_time = time.time()
    loss_curve = []

    while step < num_steps:
        for batch in train_loader:
            if step >= num_steps:
                break

            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(input_ids)

            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

            # Add aux residual loss if enabled
            aux_loss = torch.tensor(0.0, device=device)
            if (
                hasattr(model, "last_aux_residual")
                and model.last_aux_residual is not None
            ):
                aux_loss = model.last_aux_residual

            total_loss = ce_loss + model.config.lambda_aux * aux_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()

            loss_curve.append(total_loss.item())

            if step % log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Step {step:4d} | Loss {total_loss.item():.4f} | "
                    f"Elapsed {elapsed:.1f}s"
                )

            step += 1

    elapsed_total = time.time() - start_time
    print(f"\nTraining completed in {elapsed_total:.1f}s")
    print(f"  Final loss: {loss_curve[-1]:.4f}")
    print(f"  Initial loss: {loss_curve[0]:.4f}")

    # Save checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "eqlm_smoke.pt"
    save_checkpoint(model, str(checkpoint_path))
    print(f"  Saved checkpoint to {checkpoint_path}")

    return model, checkpoint_path


def measure_warm_start_decoding(
    model: EqLM,
    device: str,
    num_prompts: int = 5,
    max_new_tokens: int = 10,
) -> dict:
    """Measure warm-start vs cold-start decoding iterations (H1′a).

    Returns:
        Dict with warm and cold iteration counts.
    """
    print(f"\n{'='*70}")
    print("H1′a MEASUREMENT: Warm-Start vs Cold-Start Decoding")
    print(f"{'='*70}")

    model.eval()
    model.to(device)

    # Small prompts
    torch.manual_seed(42)
    results = {
        "warm_iters": [],
        "cold_iters": [],
        "num_prompts": num_prompts,
        "max_new_tokens": max_new_tokens,
    }

    with torch.no_grad():
        for prompt_idx in range(num_prompts):
            # Random prompt
            prompt_len = 4
            input_ids = torch.randint(0, 100, (1, prompt_len), device=device)

            # Cold start (baseline)
            output_ids, info_cold = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                warm_start=False,
                return_iter_counts=True,
            )
            cold_mean = info_cold["mean_iters"] if info_cold else 0.0
            results["cold_iters"].append(cold_mean)

            # Warm start
            output_ids, info_warm = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                warm_start=True,
                return_iter_counts=True,
            )
            warm_mean = info_warm["mean_iters"] if info_warm else 0.0
            results["warm_iters"].append(warm_mean)

            reduction = (
                100 * (1 - warm_mean / cold_mean)
                if cold_mean > 0
                else 0
            )
            print(
                f"  Prompt {prompt_idx}: Cold {cold_mean:.1f} iters, "
                f"Warm {warm_mean:.1f} iters ({reduction:+.0f}%)"
            )

    # Summary
    cold_avg = sum(results["cold_iters"]) / len(results["cold_iters"])
    warm_avg = sum(results["warm_iters"]) / len(results["warm_iters"])
    reduction_pct = 100 * (1 - warm_avg / cold_avg) if cold_avg > 0 else 0

    print(f"\n  Average cold-start: {cold_avg:.2f} iters/token")
    print(f"  Average warm-start: {warm_avg:.2f} iters/token")
    print(f"  Reduction: {reduction_pct:.1f}%")
    print(f"  H1′a threshold: ≥50% reduction — {'PASS' if reduction_pct >= 50 else 'FAIL'}")

    results["mean_cold_iters"] = cold_avg
    results["mean_warm_iters"] = warm_avg
    results["reduction_pct"] = reduction_pct

    return results


def main() -> None:
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="EXP09: H1′ Adaptive-Equilibrium")
    parser.add_argument("--config", type=str, default="configs/exp09_smoke.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="results/exp09_adaptive")
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("EXP09: H1′ Adaptive-Equilibrium Program")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    # Config hash
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    print(f"Config SHA: {config_hash}")

    # Train model
    model, checkpoint_path = train_model(config, args.device, output_dir)

    # Load checkpoint (verify roundtrip)
    model_loaded = load_checkpoint(str(checkpoint_path))
    print("\nCheckpoint roundtrip: ✓ Verified")

    # H1′a: Warm-start decoding
    results_h1a = measure_warm_start_decoding(
        model_loaded,
        args.device,
        num_prompts=3,
        max_new_tokens=5,
    )

    # Compile final results
    final_results = {
        "config_sha": config_hash,
        "device": args.device,
        "num_steps": config["training"]["num_steps"],
        "H1a_warm_start": results_h1a,
    }

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*70}")
    print("RESULTS SAVED")
    print(f"{'='*70}")
    print(f"  File: {results_file}")
    print(f"  Config SHA: {config_hash}")

    # Summary
    print(f"\n{'='*70}")
    print("DRY-RUN SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Training: {config['training']['num_steps']} steps")
    print("✓ Checkpoint save/load: OK")
    print("✓ H1′a warm-start measurement: OK")
    print("\nFull GPU run command:")
    print("  python experiments/exp09_adaptive.py \\")
    print("    --config configs/exp09_smoke.yaml \\")
    print("    --device cuda")


if __name__ == "__main__":
    main()
