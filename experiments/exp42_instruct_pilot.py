"""Experiment 42: Instruct-tuning pilot (SFT + MPO) at 121M scale.

Phase 3 implementation: two-arm comparison on a small checkpoint to validate
machinery and provide directional signal before the 1B twin.

PILOT SCOPE: 121M EqLM, max_seq_len=128, GPT-2 vocab (50257).
This is a mechanical proof-of-concept, not a scale claim. Results are directional
at best; do not oversell.

Arms:
  sft:   Plain SFT on SmolTalk, anytime supervision at [6, 11, 12].
  mpo:   SFT output as reference, DPO + magnetic anchor on UltraFeedback.

CLI:
    python exp42_instruct_pilot.py --arm sft --steps 300 --batch 8 --out results/scale/exp42
    python exp42_instruct_pilot.py --arm mpo --steps 300 --batch 8 --ref results/scale/exp42/sft_final.pt --out results/scale/exp42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.eqlm import EqLM, load_checkpoint, save_checkpoint  # noqa: E402
from kinetic_ai.train.instruct import (  # noqa: E402
    mpo_step,
    preference_batches,
    sft_batches,
    sft_step,
)


def supervise_depths(depth: int) -> list[int]:
    """Anytime depths scaled to the arm's depth (F24 regime from exp39)."""
    a = max(1, round(depth * 0.375))
    b = max(a + 1, round(depth * 0.6875))
    return [min(a, depth), min(b, depth), depth]


def generate_samples(
    model: EqLM,
    prompt: str,
    num_samples: int = 1,
    max_len: int = 32,
    device: str = "cuda",
) -> list[str]:
    """Greedy generation from a prompt using the model.

    Args:
        model: EqLM model.
        prompt: Text prompt.
        num_samples: Number of samples (for now, just 1 - deterministic).
        max_len: Max new tokens.
        device: Device to run on.

    Returns:
        List of generated text strings.
    """
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

    # Encode prompt
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        # Generate
        output_ids, info = model.generate(
            prompt_ids,
            max_new_tokens=max_len,
            warm_start=False,
            return_iter_counts=True,
        )

    # Decode
    generated_texts = []
    for ids in output_ids:
        text = tok.decode(ids.cpu().tolist(), skip_special_tokens=True)
        generated_texts.append(text)

    return generated_texts


def compute_held_out_perplexity(
    model: EqLM,
    dataset_name: str,
    batch_size: int = 8,
    seq_len: int = 128,
    device: str = "cuda",
    limit: int = 100,
) -> float:
    """Compute perplexity on held-out SFT data.

    Args:
        model: EqLM model.
        dataset_name: HF dataset name.
        batch_size: Batch size.
        seq_len: Sequence length.
        device: Device.
        limit: Max examples to evaluate.

    Returns:
        Perplexity (scalar).
    """
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    try:
        for batch_data in sft_batches(
            dataset_name, tok, seq_len, batch_size, seed=43, limit=limit
        ):
            input_ids, labels = batch_data
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                # Compute loss on unmasked positions
                logits = model(input_ids)  # [B, T, V]
                logits_flat = logits.view(-1, logits.shape[-1])
                labels_flat = labels.view(-1)

                loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")
                mask = (labels_flat != -100).float()
                masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

                total_loss += masked_loss.item()
                num_batches += 1

                if num_batches * batch_size >= limit:
                    break
    except Exception as e:
        print(f"Warning: could not load dataset for perplexity: {e}")
        return float("nan")

    avg_loss = total_loss / max(num_batches, 1)
    ppl = math.exp(avg_loss)
    return ppl


def train_sft_arm(
    checkpoint_path: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Train SFT arm on SmolTalk.

    Returns:
        Dict with metrics: final_loss, ppl, samples.
    """
    from transformers import GPT2Tokenizer

    print("[SFT] Loading checkpoint...")
    model_loaded = load_checkpoint(checkpoint_path)
    model = model_loaded.to(device)  # type: ignore
    model.train()

    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Anytime depths
    supervise_at = supervise_depths(model.config.deq_max_iter)
    print(f"[SFT] Supervising at depths: {supervise_at}")

    # Training loop
    loss_history = []
    print(f"[SFT] Training for {steps} steps...")
    step = 0

    try:
        for batch_data in sft_batches(
            "HuggingFaceTB/smoltalk",
            tok,
            seq_len,
            batch_size,
            seed=42,
            limit=None,
        ):
            if step >= steps:
                break

            input_ids, labels = batch_data
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = sft_step(model, input_ids, labels, supervise_at=supervise_at)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())

            if (step + 1) % 50 == 0:
                recent_loss = sum(loss_history[-10:]) / 10.0
                print(f"  Step {step + 1}/{steps}: loss={recent_loss:.4f}")

            step += 1

    except Exception as e:
        print(f"Warning: dataset loading failed: {e}")
        print(f"[SFT] Completed {step} steps before stopping")

    # Save checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = out_dir / "sft_final.pt"
    print(f"[SFT] Saving checkpoint to {final_ckpt_path}...")
    save_checkpoint(model, str(final_ckpt_path))  # type: ignore

    # Compute perplexity
    print("[SFT] Computing held-out perplexity...")
    ppl = compute_held_out_perplexity(
        model, "HuggingFaceTB/smoltalk", batch_size, seq_len, device, limit=100
    )

    # Generate samples
    print("[SFT] Generating samples...")
    fixed_prompts = [
        "Hello, my name is",
        "The weather today is",
        "I love to",
    ]
    samples = {}
    for prompt in fixed_prompts:
        try:
            generated = generate_samples(model, prompt, num_samples=1, max_len=32, device=device)
            samples[prompt] = generated[0] if generated else ""
        except Exception as e:
            print(f"Warning: generation failed for '{prompt}': {e}")
            samples[prompt] = ""

    results = {
        "arm": "sft",
        "steps": step,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "ppl": ppl,
        "samples": samples,
    }

    return results


def train_mpo_arm(
    sft_ckpt_path: str,
    base_ckpt_path: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    beta: float,
    magnet_tau: float,
    device: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Train MPO arm using SFT checkpoint as reference.

    Returns:
        Dict with metrics: final_loss, pref_acc_before, pref_acc_after, samples.
    """
    from transformers import GPT2Tokenizer

    print("[MPO] Loading SFT checkpoint as policy...")
    model_loaded = load_checkpoint(sft_ckpt_path)
    model = model_loaded.to(device)  # type: ignore
    model.train()

    print("[MPO] Loading base checkpoint as reference...")
    ref_model_loaded = load_checkpoint(base_ckpt_path)
    ref_model = ref_model_loaded.to(device)  # type: ignore
    ref_model.eval()

    tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    # Training loop
    loss_history = []
    pref_acc_history = []
    print(f"[MPO] Training for {steps} steps...")
    step = 0

    try:
        for batch in preference_batches(
            "HuggingFaceH4/ultrafeedback_binarized",
            tok,
            seq_len,
            batch_size,
            seed=42,
            limit=None,
        ):
            if step >= steps:
                break

            optimizer.zero_grad()
            loss, pref_acc = mpo_step(
                model, ref_model, batch, beta=beta, magnet_tau=magnet_tau
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_history.append(loss.item())
            pref_acc_history.append(pref_acc.item())

            if (step + 1) % 50 == 0:
                recent_loss = sum(loss_history[-10:]) / 10.0
                recent_acc = sum(pref_acc_history[-10:]) / 10.0
                print(
                    f"  Step {step + 1}/{steps}: loss={recent_loss:.4f}, pref_acc={recent_acc:.4f}"
                )

            step += 1

    except Exception as e:
        print(f"Warning: dataset loading failed: {e}")
        print(f"[MPO] Completed {step} steps before stopping")

    # Save checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = out_dir / "mpo_final.pt"
    print(f"[MPO] Saving checkpoint to {final_ckpt_path}...")
    save_checkpoint(model, str(final_ckpt_path))  # type: ignore

    # Generate samples
    print("[MPO] Generating samples...")
    fixed_prompts = [
        "Hello, my name is",
        "The weather today is",
        "I love to",
    ]
    samples = {}
    for prompt in fixed_prompts:
        try:
            generated = generate_samples(model, prompt, num_samples=1, max_len=32, device=device)
            samples[prompt] = generated[0] if generated else ""
        except Exception as e:
            print(f"Warning: generation failed for '{prompt}': {e}")
            samples[prompt] = ""

    results = {
        "arm": "mpo",
        "steps": step,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "pref_acc": sum(pref_acc_history) / len(pref_acc_history) if pref_acc_history else 0.0,
        "samples": samples,
    }

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Instruct-tuning pilot (SFT + MPO) at 121M scale"
    )
    parser.add_argument("--arm", choices=["sft", "mpo"], required=True, help="Which arm to train")
    parser.add_argument(
        "--checkpoint",
        default="results/scale/ckpt/eqlm_anytime_seed42.pt",
        help="Path to base checkpoint",
    )
    parser.add_argument("--ref", help="Path to SFT checkpoint (for MPO arm)")
    parser.add_argument("--steps", type=int, default=300, help="Number of training steps")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (for MPO)")
    parser.add_argument(
        "--magnet-tau", type=float, default=0.05, help="Magnetic anchor weight (for MPO)"
    )
    parser.add_argument("--device", default="cuda:0", help="Device to train on")
    parser.add_argument("--out", default="results/scale/exp42", help="Output directory")
    parser.add_argument("--limit-examples", type=int, default=None, help="Max examples to load")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = Path(args.out) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.arm == "sft":
        results = train_sft_arm(
            args.checkpoint,
            args.steps,
            args.batch,
            args.seq_len,
            args.lr,
            device,
            out_dir,
        )
    else:  # mpo
        if not args.ref:
            parser.error("--ref is required for mpo arm")
        results = train_mpo_arm(
            args.ref,
            args.checkpoint,
            args.steps,
            args.batch,
            args.seq_len,
            args.lr,
            args.beta,
            args.magnet_tau,
            device,
            out_dir,
        )

    # Save results
    results_path = out_dir / "results.json"
    print(f"\nSaving results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[{args.arm.upper()}] Complete.")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
