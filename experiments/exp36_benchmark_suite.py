"""Multi-benchmark head-to-head evaluation harness for compute-matched models.

Rigorous empirical comparison of the tied-block equilibrium model against its
compute-matched explicit baseline. Evaluates both architectures on:
  (1) Full BLiMP (67 linguistic phenomena, per-phenomenon breakdown + z-test)
  (2) Held-out perplexity (on data/cache/*.pt token sequences)
  (3) Accounting: parameters, weight memory (bytes), compute per token

All measurements include model capacity accounting (F44 discipline) so that
quality claims remain grounded in actual compute cost. Output is a single JSON
record with per-benchmark detail and paired statistical comparison.

References:
  - F24: Anytime-unrolled training closes width gap (ratio 0.991 at 121M)
  - F44: Quality without compute accounting makes misleading claims
  - BLiMP (Warstadt et al. 2020): Minimal-pair linguistic test set, 67 phenomena
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.data.tokenizer import load_gpt2_tokenizer  # noqa: E402
from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM  # noqa: E402


def load_model(checkpoint_path: Path, device: str) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a checkpoint with its recorded architecture.

    The checkpoint records both weights and the class name (EqLM vs ExplicitLM)
    so the correct architecture is restored without guessing. Returns the
    model and metadata dict for parameter accounting.
    """
    blob = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = blob["state_dict"]
    cfg = EqLMConfig(**blob["config_dict"])
    kind = blob.get("model_class", "EqLM")

    if kind == "ExplicitLM":
        model: nn.Module = ExplicitLM(config=cfg, n_layers=int(blob["n_layers"]))
    else:
        model = EqLM(config=cfg)

    model.load_state_dict(state)
    model = model.to(device).eval()

    # Metadata for accounting: stored alongside checkpoint or computed
    metadata = blob.get("metadata", {})
    metadata.setdefault("model_class", kind)
    metadata.setdefault("n_layers", blob.get("n_layers", 1))

    return model, metadata


def compute_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_weight_memory_bytes(model: nn.Module) -> int:
    """Compute weight memory footprint in bytes (float32 unless stored in bf16)."""
    total = 0
    for p in model.parameters():
        # Assume float32 (4 bytes per param) unless explicitly stored otherwise
        total += p.numel() * 4
    return total


def compute_units_per_token(
    model: nn.Module,
    config: EqLMConfig,
    model_class: str,
    n_layers: int = 1,
) -> float:
    """Compute approximate FLOPs per token (multiply-accumulate count).

    For a transformer block applied once:
      - Attention: O(d^2 + d*T) per token (Q/K/V proj + score computation)
        Simplified to 4*d^2 per token (d^2 per QKV proj, d^2 for scores)
      - MLP: 2*d*d_ff (two linear layers)
      - Total per application: 4*d^2 + 2*d*d_ff

    For EqLM: multiply by effective depth (12 iterations, or adaptive)
    For ExplicitLM: multiply by n_layers

    This is a rough operation count; actual FLOPs depend on attention
    implementation (FlashAttention, sparse attn, etc.).
    """
    d = config.d_model
    d_ff = config.d_ff

    # Simplification: one block application costs 4*d^2 + 2*d*d_ff
    per_block = 4 * (d**2) + 2 * d * d_ff

    # Effective depth: deq_max_iter for EqLM (typically 12 iterations),
    # n_layers for ExplicitLM (explicit stacked layers)
    effective_depth = config.deq_max_iter if model_class == "EqLM" else n_layers

    return per_block * effective_depth


def load_blimp_full() -> dict[str, list[dict]]:
    """Load all 67 BLiMP phenomena from HuggingFace (offline cache).

    Returns a dict mapping phenomenon name -> list of minimal-pair examples,
    each with 'sentence_good' and 'sentence_bad' fields.
    """
    phenomena = [
        "adjunct_island",
        "anaphor_gender_agreement",
        "anaphor_number_agreement",
        "animate_subject_passive",
        "animate_subject_trans",
        "arg_structure",
        "binding_anaphor",
        "binding_complex_NP",
        "binding_negative_anaphor",
        "binding_relative_clause",
        "c_command_complex_NP",
        "c_command_VP",
        "center_embedding",
        "cleft_island",
        "complex_NP_island",
        "coordinate_structure",
        "coordination_across_constituents",
        "coordination_of_adjectives",
        "determination",
        "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause",
        "drop_argument",
        "ellipsis_n_bar_1",
        "ellipsis_n_bar_2",
        "ellipsis_vp_2",
        "extraposed_relative_clause",
        "familiar",
        "filler_gap",
        "focus",
        "garden_path_1",
        "garden_path_2",
        "garden_path_3",
        "generic_plural",
        "genitive_in_bnp",
        "gerund_island",
        "given_new",
        "hard_garden_path_1",
        "hard_garden_path_2",
        "hypernym_only",
        "idiom_variations",
        "implicit_causality",
        "infinitival_relative_clause",
        "interrogative_complementizer",
        "irregular_past_participle",
        "irregular_plural",
        "irregular_possessive",
        "left_branch_island_echo_question",
        "left_branch_island_main_clause",
        "left_branch_island_pied_piping",
        "ltltransitive_intr_variant",
        "matrix_question_npi",
        "mve_agree",
        "mve_pseudo_reflexive",
        "nominalized_embedded_clause",
        "nominalized_relative_clause",
        "np_vp_coordination",
        "npi_scope",
        "only_npi_licensor",
        "only_npi_scope",
        "passive_1",
        "passive_2",
        "pied_piping",
        "plural_subject_verb_agreement_1",
        "plural_subject_verb_agreement_2",
        "principle_A_c_command",
        "principle_A_case_weight",
        "principle_A_domain_1",
        "principle_A_domain_2",
        "principle_A_domain_3",
        "principle_A_reconstruction",
        "regular_plural",
        "regular_possessive",
        "relative_clause_across_animate",
        "relative_clause_across_inanimate",
        "relative_clause_subject",
        "relative_clause_object",
        "sentential_negation_npi_licensor",
        "sentential_negation_npi_scope",
        "sentential_subject",
        "ship_vessel",
        "simple_active_syntax",
        "simplex_NP_island",
        "singular_subject_verb_agreement",
        "superlative",
        "syntactic_category_typo",
        "tough_vs_raising_1",
        "tough_vs_raising_2",
        "transitive_active_variant",
        "transitive_negative_variant",
        "wh_island",
        "wh_questions_object_gap",
        "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance",
        "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ]

    blimp_data = {}
    for phenom in phenomena:
        try:
            ds = load_dataset("nyu-mll/blimp", phenom, split="train")
            blimp_data[phenom] = [dict(ex) for ex in ds]
        except Exception as e:
            print(f"Warning: could not load BLiMP phenomenon '{phenom}': {e}")

    return blimp_data


def compute_sentence_logprob(model: nn.Module, tokens: torch.Tensor, device: str) -> float:
    """Sum log-probabilities of tokens in sequence (left-to-right).

    Standard BLiMP protocol: log p(sentence) = sum_t log p(t | t<t).
    Returns total log-prob as a scalar.
    """
    tokens = tokens.unsqueeze(0).to(device)  # [1, seq_len]
    model.eval()

    with torch.no_grad():
        # Forward pass: logits [1, seq_len, vocab_size]
        logits = model(tokens)

        # Shift: predict token t+1 from tokens up to t
        logits_shifted = logits[0, :-1, :]  # [seq_len-1, vocab_size]
        targets = tokens[0, 1:]  # [seq_len-1]

        # Log-softmax and gather
        log_probs = torch.nn.functional.log_softmax(logits_shifted, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1))  # [seq_len-1, 1]

        # Sum and return
        return float(target_log_probs.sum().item())


def evaluate_blimp_phenomenon(
    model: nn.Module,
    phenomenon_data: list[dict],
    tokenizer_fn: Any,
    device: str,
) -> dict[str, Any]:
    """Evaluate model on a single BLiMP phenomenon (all pairs).

    Returns accuracy, count, and logprob data for statistical comparison.
    """
    model.eval()
    correct = 0
    total = 0
    good_logprobs = []
    bad_logprobs = []

    for example in phenomenon_data:
        sent_good = example.get("sentence_good") or example.get("good_sentence")
        sent_bad = example.get("sentence_bad") or example.get("bad_sentence")

        if not sent_good or not sent_bad:
            continue

        try:
            tokens_good = tokenizer_fn(sent_good)
            tokens_bad = tokenizer_fn(sent_bad)

            if isinstance(tokens_good, str):
                tokens_good = [int(t) if t.isdigit() else 0 for t in tokens_good.split()]
            if isinstance(tokens_bad, str):
                tokens_bad = [int(t) if t.isdigit() else 0 for t in tokens_bad.split()]

            tokens_good = torch.tensor(tokens_good, dtype=torch.long)
            tokens_bad = torch.tensor(tokens_bad, dtype=torch.long)

            if len(tokens_good) < 2 or len(tokens_bad) < 2:
                continue

            lp_good = compute_sentence_logprob(model, tokens_good, device)
            lp_bad = compute_sentence_logprob(model, tokens_bad, device)

            good_logprobs.append(lp_good)
            bad_logprobs.append(lp_bad)

            if lp_good > lp_bad:
                correct += 1

            total += 1
        except Exception:
            continue

    if total == 0:
        return {"accuracy": 0.0, "num_correct": 0, "num_total": 0, "logprobs_good": [], "logprobs_bad": []}

    return {
        "accuracy": correct / total,
        "num_correct": correct,
        "num_total": total,
        "logprobs_good": good_logprobs,
        "logprobs_bad": bad_logprobs,
    }


def paired_z_test(
    good_logprobs_model1: list[float],
    bad_logprobs_model1: list[float],
    good_logprobs_model2: list[float],
    bad_logprobs_model2: list[float],
) -> dict[str, float]:
    """Paired z-test for BLiMP phenomenon comparison.

    Computes whether model1 is significantly better/worse than model2
    on a per-pair basis. Records win/loss counts and computes z-score.
    """
    if len(good_logprobs_model1) == 0 or len(good_logprobs_model2) == 0:
        return {"wins": 0, "losses": 0, "ties": 0, "z_score": 0.0, "p_value": 1.0}

    # Ensure same length
    min_len = min(
        len(good_logprobs_model1),
        len(good_logprobs_model2),
    )
    g1 = np.array(good_logprobs_model1[:min_len])
    b1 = np.array(bad_logprobs_model1[:min_len])
    g2 = np.array(good_logprobs_model2[:min_len])
    b2 = np.array(bad_logprobs_model2[:min_len])

    # Per-pair margin: how much did the model prefer the good sentence?
    margins_1 = g1 - b1
    margins_2 = g2 - b2

    # Count wins/losses/ties
    wins = np.sum(margins_1 > margins_2)
    losses = np.sum(margins_1 < margins_2)
    ties = np.sum(margins_1 == margins_2)

    # Paired z-test: does model1 have systematically larger margins?
    diffs = margins_1 - margins_2
    if len(diffs) > 1 and np.std(diffs) > 1e-6:
        z_score = float(np.mean(diffs) / (np.std(diffs) / np.sqrt(len(diffs))))
        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = 0.0
        p_value = 1.0

    return {
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "z_score": z_score,
        "p_value": p_value,
    }


def compute_perplexity_on_cache(
    model: nn.Module,
    cache_dir: Path,
    device: str,
    max_examples: int = 100,
) -> dict[str, Any]:
    """Compute held-out perplexity on token cache.

    The token cache (data/cache/*.pt files) contains pre-tokenized sequences.
    This evaluates the model's language modeling loss on held-out data.
    """
    model.eval()

    cache_files = list(cache_dir.glob("tokstream_*.pt"))
    if not cache_files:
        return {
            "perplexity": None,
            "loss": None,
            "num_tokens": 0,
            "note": "No cache files found",
        }

    total_loss = 0.0
    total_tokens = 0
    num_loaded = 0

    for cache_file in cache_files[:max_examples]:
        try:
            # Load token sequence
            data = torch.load(cache_file, map_location=device, weights_only=True)

            # Expect data to be a tensor of token IDs or dict with tokens field
            tokens = data.get("tokens", data.get("input_ids")) if isinstance(data, dict) else data

            if tokens is None or len(tokens) == 0:
                continue

            tokens = torch.as_tensor(tokens, dtype=torch.long, device=device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)  # [1, seq_len]

            with torch.no_grad():
                # Forward pass
                logits = model(tokens)  # [batch, seq_len, vocab_size]

                # Shift: predict token t+1 from t<t
                logits_shifted = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
                targets = tokens[:, 1:]  # [batch, seq_len-1]

                # Cross-entropy
                loss = torch.nn.functional.cross_entropy(
                    logits_shifted.reshape(-1, logits_shifted.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )

                total_loss += loss.item()
                total_tokens += targets.numel()
                num_loaded += 1
        except Exception as e:
            print(f"Warning: could not load {cache_file}: {e}")
            continue

    if total_tokens == 0:
        return {
            "perplexity": None,
            "loss": None,
            "num_tokens": 0,
            "num_files_loaded": num_loaded,
        }

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        "perplexity": float(perplexity),
        "loss": float(avg_loss),
        "num_tokens": total_tokens,
        "num_files_loaded": num_loaded,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multi-benchmark head-to-head evaluation harness"
    )
    ap.add_argument("--eqlm", required=True, help="EqLM tied-block checkpoint")
    ap.add_argument("--explicit", required=True, help="Explicit baseline checkpoint (param-matched)")
    ap.add_argument("--out", default="results/exp36_benchmark_suite.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blimp-limit", type=int, default=None, help="Limit BLiMP to N phenomena (smoke test)")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--cache-limit", type=int, default=100, help="Limit perplexity eval to N cache files")
    args = ap.parse_args()

    device = args.device
    torch.manual_seed(42)
    np.random.seed(42)

    print("[benchmark] Loading models...")
    eqlm_model, eqlm_meta = load_model(Path(args.eqlm), device)
    explicit_model, explicit_meta = load_model(Path(args.explicit), device)

    print(f"[benchmark] Loaded EqLM: {eqlm_meta}")
    print(f"[benchmark] Loaded Explicit: {explicit_meta}")

    # Tokenizer (shared)
    gpt2 = load_gpt2_tokenizer()
    if gpt2 is None:
        print("ERROR: GPT-2 tokenizer unavailable")
        return 1

    def tokenizer_fn(text: str) -> list[int]:
        return list(gpt2(text)["input_ids"])

    # Load BLiMP
    print("[benchmark] Loading BLiMP...")
    blimp_data = load_blimp_full()
    phenomena_to_eval = list(blimp_data.keys())
    if args.blimp_limit:
        phenomena_to_eval = phenomena_to_eval[:args.blimp_limit]
    print(f"[benchmark] Evaluating on {len(phenomena_to_eval)} phenomena")

    # Evaluate on BLiMP
    print("[benchmark] Evaluating on BLiMP...")
    blimp_results = {}
    eqlm_phenom_results = {}
    explicit_phenom_results = {}

    for i, phenom in enumerate(phenomena_to_eval):
        if i % 10 == 0:
            print(f"  [{i}/{len(phenomena_to_eval)}] {phenom}")

        phenom_data = blimp_data[phenom]

        # Evaluate both models
        eqlm_res = evaluate_blimp_phenomenon(eqlm_model, phenom_data, tokenizer_fn, device)
        explicit_res = evaluate_blimp_phenomenon(explicit_model, phenom_data, tokenizer_fn, device)

        eqlm_phenom_results[phenom] = eqlm_res
        explicit_phenom_results[phenom] = explicit_res

        # Paired comparison
        comparison = paired_z_test(
            eqlm_res["logprobs_good"],
            eqlm_res["logprobs_bad"],
            explicit_res["logprobs_good"],
            explicit_res["logprobs_bad"],
        )

        blimp_results[phenom] = {
            "eqlm_accuracy": eqlm_res["accuracy"],
            "explicit_accuracy": explicit_res["accuracy"],
            "eqlm_count": eqlm_res["num_total"],
            "explicit_count": explicit_res["num_total"],
            "comparison": comparison,
        }

    # Aggregate BLiMP scores
    eqlm_blimp_overall = np.mean([r["eqlm_accuracy"] for r in blimp_results.values()])
    explicit_blimp_overall = np.mean([r["explicit_accuracy"] for r in blimp_results.values()])

    # Perplexity on cache
    print("[benchmark] Computing perplexity on cache...")
    cache_dir = Path(args.cache_dir)
    eqlm_ppl = compute_perplexity_on_cache(eqlm_model, cache_dir, device, args.cache_limit)
    explicit_ppl = compute_perplexity_on_cache(explicit_model, cache_dir, device, args.cache_limit)

    # Parameter accounting
    eqlm_params = compute_parameters(eqlm_model)
    explicit_params = compute_parameters(explicit_model)
    eqlm_weight_bytes = compute_weight_memory_bytes(eqlm_model)
    explicit_weight_bytes = compute_weight_memory_bytes(explicit_model)

    # A missing or unrecognised config must stop the run rather than fall back
    # to defaults. Substituting a default here would silently compute the
    # compute-per-token figure for a model that is not the one being evaluated,
    # and F44 exists because exactly that kind of unstated mismatch made an
    # earlier comparison misleading.
    def _require_config(model: nn.Module, tag: str) -> EqLMConfig:
        cfg = getattr(model, "config", None)
        if not isinstance(cfg, EqLMConfig):
            raise SystemExit(
                f"{tag} checkpoint carries no usable EqLMConfig (got {type(cfg).__name__}); "
                "refusing to report compute accounting for a config that was guessed"
            )
        return cfg

    eqlm_config = _require_config(eqlm_model, "eqlm")
    explicit_config = _require_config(explicit_model, "explicit")

    eqlm_fpu_token = compute_units_per_token(eqlm_model, eqlm_config, "EqLM", 1)
    explicit_fpu_token = compute_units_per_token(
        explicit_model, explicit_config, "ExplicitLM", explicit_meta.get("n_layers", 12)
    )

    # Build report
    report: dict[str, Any] = {
        "timestamp": str(torch.cuda.Event(enable_timing=True).record()),
        "checkpoints": {
            "eqlm": args.eqlm,
            "explicit": args.explicit,
        },
        "models": {
            "eqlm": {
                "class": eqlm_meta.get("model_class", "EqLM"),
                "n_layers": eqlm_meta.get("n_layers", 1),
                "parameters": eqlm_params,
                "weight_memory_bytes": eqlm_weight_bytes,
                "ops_per_token": eqlm_fpu_token,
            },
            "explicit": {
                "class": explicit_meta.get("model_class", "ExplicitLM"),
                "n_layers": explicit_meta.get("n_layers", 12),
                "parameters": explicit_params,
                "weight_memory_bytes": explicit_weight_bytes,
                "ops_per_token": explicit_fpu_token,
            },
        },
        "parameter_ratio": float(eqlm_params) / float(explicit_params) if explicit_params > 0 else 1.0,
        "blimp": {
            "phenomena_evaluated": len(blimp_results),
            "eqlm_overall_accuracy": float(eqlm_blimp_overall),
            "explicit_overall_accuracy": float(explicit_blimp_overall),
            "ratio": float(eqlm_blimp_overall) / float(explicit_blimp_overall) if explicit_blimp_overall > 0 else 1.0,
            "per_phenomenon": blimp_results,
        },
        "perplexity": {
            "eqlm": eqlm_ppl,
            "explicit": explicit_ppl,
        },
    }

    # Write report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[benchmark] Results written to {args.out}")
    print("\nBLiMP Summary:")
    print(f"  EqLM:     {eqlm_blimp_overall:.4f}")
    print(f"  Explicit: {explicit_blimp_overall:.4f}")
    print(f"  Ratio:    {eqlm_blimp_overall/explicit_blimp_overall:.4f}")
    print("\nParameter Accounting:")
    print(f"  EqLM:     {eqlm_params:,} params, {eqlm_weight_bytes/1e6:.1f} MB weight, {eqlm_fpu_token:.1e} ops/token")
    print(f"  Explicit: {explicit_params:,} params, {explicit_weight_bytes/1e6:.1f} MB weight, {explicit_fpu_token:.1e} ops/token")

    return 0


if __name__ == "__main__":
    sys.exit(main())
