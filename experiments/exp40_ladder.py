"""Milestone eval harness: universal score for our checkpoints and HF models.

Scores BOTH our EqLM/ExplicitLM checkpoints and public HF causal LMs on the same
multiple-choice loglikelihood protocol (lm-eval style), producing one JSON per
invocation. The ladder enables direct numerical comparison: accuracy metrics stay
identical when the model changes; only the logits differ. This protocol is the
foundation for Arm T/E comparison and public baseline positioning.

Core scorer: score_continuation(model_fn, tok_fn, context, continuation, device)
returns (sum_log_p_of_continuation_tokens, n_continuation_tokens). The loglikelihood
is fed to multiple-choice and full-LM evaluation pipelines that all use the same
scorer — that invariant is what makes a ladder trustworthy.

References:
  - lm-eval (https://github.com/EleutherAI/lm-evaluation-harness) for protocol
  - SPEC 0022 "Eval ladder" section for registered tasks and gate criteria
  - exp39_twin_1b.py for our checkpoint loaders and tied/explicit forward paths
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.models.eqlm import EqLM, EqLMConfig, ExplicitLM  # noqa: E402


@dataclasses.dataclass
class ModelAdapter:
    """Protocol for scoring: unified interface hides implementation details."""

    name: str
    logits_fn: Callable[[torch.Tensor], torch.Tensor]
    tokenize: Callable[[str], list[int]]
    max_len: int
    device: str

    def score_ids(
        self, context_ids: list[int], continuation_ids: list[int]
    ) -> tuple[float, int]:
        """Score a continuation given a context.

        Returns: (sum_log_p_of_continuation, len_continuation_ids)
        """
        # Concatenate and check truncation. At least one context token must
        # survive truncation, because the first continuation token is scored
        # from the logits at the last context position.
        all_ids = context_ids + continuation_ids
        if len(all_ids) > self.max_len:
            keep = max(1, self.max_len - len(continuation_ids))
            context_ids = context_ids[-keep:]
            all_ids = context_ids + continuation_ids
        if not context_ids:
            raise ValueError("score_ids requires a non-empty context")

        # One forward pass on full sequence
        all_tensor = torch.tensor([all_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.logits_fn(all_tensor)  # [1, seq_len, vocab_size]

        # A causal LM's logits at position i are the distribution over the
        # token at position i+1, so continuation token k (absolute position
        # n_context+k) is scored from logits[n_context+k-1]. Reading logits at
        # the token's own position instead scores nothing meaningful — the
        # first rung sweep returned exact chance on every task this way.
        n_context = len(context_ids)
        continuation_logits = logits[
            0, n_context - 1 : n_context - 1 + len(continuation_ids), :
        ]

        # Log-softmax and gather
        log_probs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
        scores = []
        for i, token_id in enumerate(continuation_ids):
            scores.append(log_probs[i, token_id].item())

        return sum(scores), len(continuation_ids)


def load_our_checkpoint(
    ckpt_path: str | Path, device: str, iters: int | None = None
) -> ModelAdapter:
    """Load an EqLM or ExplicitLM checkpoint and return a scoring adapter.

    Args:
        ckpt_path: Path to torch.load'ed checkpoint.
        device: CUDA device string.
        iters: For tied models, number of block iterations (default: config.deq_max_iter).

    Returns:
        ModelAdapter ready for scoring.
    """
    ckpt_path = Path(ckpt_path)
    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = blob["state_dict"]
    cfg = EqLMConfig(**blob["config_dict"])
    kind = blob.get("model_class", "EqLM")
    n_layers = int(blob.get("n_layers", 1))

    model = ExplicitLM(config=cfg, n_layers=n_layers) if kind == "ExplicitLM" else EqLM(config=cfg)
    model.load_state_dict(state)
    model = model.to(device).eval()

    # Tokenizer: GPT-2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Prepare the forward function
    if kind == "ExplicitLM":
        # Import explicit_logits from exp39
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from exp39_twin_1b import explicit_logits  # noqa: E402

        def logits_fn(ids: torch.Tensor) -> torch.Tensor:
            return explicit_logits(model, ids, use_ckpt=False)

    else:
        # Tied model: import tied_outputs
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from exp39_twin_1b import tied_outputs  # noqa: E402

        # Use iters from config if not specified
        if iters is None:
            iters = cfg.deq_max_iter

        def logits_fn(ids: torch.Tensor) -> torch.Tensor:
            # Unroll to iters iterations and take the final output
            depths = [iters]
            outs = tied_outputs(model, ids, depths, use_ckpt=False)
            return outs[0][1]  # Return just the logits tensor

    # Slice logits if vocab is padded
    tokenizer_vocab_size = len(tokenizer)

    def logits_fn_sliced(ids: torch.Tensor) -> torch.Tensor:
        logits = logits_fn(ids)
        # Slice to tokenizer vocab if padded
        if logits.shape[-1] > tokenizer_vocab_size:
            logits = logits[:, :, :tokenizer_vocab_size]
        return logits

    return ModelAdapter(
        name=ckpt_path.stem,
        logits_fn=logits_fn_sliced,
        tokenize=lambda text: tokenizer.encode(text),
        max_len=cfg.max_seq_len,
        device=device,
    )


def load_hf_model(name: str, device: str) -> ModelAdapter:
    """Load a HuggingFace causal LM and return a scoring adapter.

    Args:
        name: HuggingFace model ID (e.g., "EleutherAI/pythia-70m").
        device: CUDA device string.

    Returns:
        ModelAdapter ready for scoring.
    """
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)

    # Infer max_len from config
    max_len = getattr(model.config, "max_position_embeddings", 2048)

    def logits_fn(ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(ids, output_hidden_states=False)
        return out.logits

    return ModelAdapter(
        name=name,
        logits_fn=logits_fn,
        tokenize=lambda text: tokenizer.encode(text),
        max_len=max_len,
        device=device,
    )


def score_continuation(adapter: ModelAdapter, context: str, continuation: str) -> tuple[float, int]:
    """Score a continuation given a context using the adapter.

    Args:
        adapter: ModelAdapter instance.
        context: Context string.
        continuation: Continuation string.

    Returns:
        (log_prob_sum, n_continuation_tokens)
    """
    context_ids = adapter.tokenize(context)
    continuation_ids = adapter.tokenize(continuation)
    return adapter.score_ids(context_ids, continuation_ids)


# ============================================================================
# Task loaders: each returns list[{"context": str, "options": [str, ...], "gold": int}]
# ============================================================================


def load_arc_easy(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """ARC-Easy: multiple choice reading comprehension."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    examples = []
    for ex in ds:
        question = ex["question"]
        choices = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        # answerKey is "A", "B", "C", "D" or "1", "2", "3", "4"
        gold_idx = ord(answer_key.upper()) - ord("A") if answer_key.isalpha() else int(answer_key) - 1
        context = f"Question: {question}\nAnswer:"
        options = [f" {c}" for c in choices]
        examples.append({"context": context, "options": options, "gold": gold_idx})
    return sample_examples(examples, max_examples, seed)


def load_arc_challenge(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """ARC-Challenge: multiple choice reading comprehension (harder)."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    examples = []
    for ex in ds:
        question = ex["question"]
        choices = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        gold_idx = ord(answer_key.upper()) - ord("A") if answer_key.isalpha() else int(answer_key) - 1
        context = f"Question: {question}\nAnswer:"
        options = [f" {c}" for c in choices]
        examples.append({"context": context, "options": options, "gold": gold_idx})
    return sample_examples(examples, max_examples, seed)


def load_hellaswag(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """HellaSwag: activity prediction."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    examples = []
    for ex in ds:
        context = ex["ctx"]  # activity context string
        endings = ex["endings"]
        gold = int(ex["label"])
        options = [f" {end}" for end in endings]
        examples.append({"context": context, "options": options, "gold": gold})
    return sample_examples(examples, max_examples, seed)


def load_piqa(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """PIQA: physical common sense reasoning."""
    # The canonical piqa repo ships a loading script, which datasets >= 3
    # refuses to execute; the hub's auto-converted parquet branch carries the
    # same rows and loads everywhere.
    ds = load_dataset(
        "parquet",
        data_files={
            "validation": "hf://datasets/ybisk/piqa@refs/convert/parquet/"
            "plain_text/validation/*.parquet"
        },
    )["validation"]
    examples = []
    for ex in ds:
        goal = ex["goal"]
        sol1 = ex["sol1"]
        sol2 = ex["sol2"]
        label = ex["label"]
        context = f"Question: {goal}\nAnswer:"
        options = [f" {sol1}", f" {sol2}"]
        examples.append({"context": context, "options": options, "gold": label})
    return sample_examples(examples, max_examples, seed)


def winogrande_example(
    sentence: str, option1: str, option2: str, answer: str
) -> dict | None:
    """Partial-evaluation form: each option fills the blank to make its own
    context, and the sentence's shared suffix is the scored continuation. The
    first version scored one identical (context, suffix) pair twice, which
    reduced every model to the gold marginal (0.493 across the whole sweep)."""
    if "_" not in sentence:
        return None
    prefix, suffix = sentence.split("_", 1)
    return {
        "contexts": [prefix + option1, prefix + option2],
        "options": [suffix, suffix],
        "gold": 0 if answer == "1" else 1,
    }


def load_winogrande(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """WinoGrande: coreference resolution via loglikelihood of filled pronoun context."""
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    examples = []
    for ex in ds:
        made = winogrande_example(
            ex["sentence"], ex["option1"], ex["option2"], ex["answer"]
        )
        if made is not None:
            examples.append(made)

    return sample_examples(examples, max_examples, seed)


def load_sciq(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """SciQ: science question answering with distractors."""
    ds = load_dataset("allenai/sciq", split="test")
    examples = []
    rng = random.Random(seed)
    for ex in ds:
        question = ex["question"]
        support = ex["support"]
        correct_answer = ex["correct_answer"]
        distractors = [ex["distractor1"], ex["distractor2"], ex["distractor3"]]

        # Truncate support to last 600 chars
        if len(support) > 600:
            support = support[-600:]

        context = f"{support}\nQuestion: {question}\nAnswer:"

        # Shuffle options with correct answer
        options = [correct_answer] + distractors
        gold = 0  # correct_answer is now at index 0
        rng.shuffle(options)
        gold = options.index(correct_answer)

        options = [f" {opt}" for opt in options]
        examples.append({"context": context, "options": options, "gold": gold})

    return sample_examples(examples, max_examples, seed)


def load_lambada_openai(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """LAMBADA (openai version): predict last word of passage."""
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    examples = []
    for ex in ds:
        text = ex["text"]
        # Split into context (all but last word) and continuation (last word)
        words = text.rsplit(None, 1)
        if len(words) != 2:
            continue
        context = words[0]
        continuation = " " + words[1]  # lm-eval convention
        examples.append({"context": context, "continuation": continuation, "lambada": True})
    return sample_examples(examples, max_examples, seed)


def load_mmlu(max_examples: int | None = None, seed: int = 42) -> list[dict]:
    """MMLU: massive multitask language understanding (sampled across all subjects)."""
    ds = load_dataset("cais/mmlu", "all", split="test")
    examples = []
    for ex in ds:
        question = ex["question"]
        choices = ex["choices"]
        answer = ex["answer"]  # 0-3 index
        context = f"Question: {question}\nAnswer:"
        options = [f" {c}" for c in choices]
        examples.append({"context": context, "options": options, "gold": answer})
    return sample_examples(examples, max_examples, seed)


def sample_examples(examples: list[dict], max_examples: int | None, seed: int) -> list[dict]:
    """Uniformly sample up to max_examples using seeded RNG, without replacement."""
    if max_examples is None or len(examples) <= max_examples:
        return examples
    rng = random.Random(seed)
    return rng.sample(examples, max_examples)


# ============================================================================
# Multiple choice and language model evaluation
# ============================================================================


def eval_multiple_choice(adapter: ModelAdapter, examples: list[dict]) -> dict[str, float]:
    """Evaluate multiple-choice accuracy and normalized accuracy."""
    n = 0
    acc = 0
    acc_norm = 0
    for ex in examples:
        options = ex["options"]
        gold = ex["gold"]
        # Most tasks share one context across options; winogrande's partial-
        # evaluation convention inverts that — each option fills the blank to
        # form its own context and the shared suffix is what gets scored — so
        # an example may carry per-option contexts instead.
        contexts = ex.get("contexts") or [ex["context"]] * len(options)

        scores = []
        for ctx, option in zip(contexts, options, strict=True):
            log_p, _ = score_continuation(adapter, ctx, option)
            scores.append(log_p)

        # Raw accuracy: argmax of log sums
        pred = int(np.argmax(scores))
        if pred == gold:
            acc += 1

        # Normalized accuracy: argmax of log sums divided by byte length
        norm_scores = [s / len(opt.encode("utf-8")) for s, opt in zip(scores, options, strict=True)]
        pred_norm = int(np.argmax(norm_scores))
        if pred_norm == gold:
            acc_norm += 1

        n += 1

    return {"acc": acc / n if n > 0 else 0.0, "acc_norm": acc_norm / n if n > 0 else 0.0, "n": n}


def eval_lambada(adapter: ModelAdapter, examples: list[dict]) -> dict[str, float]:
    """Evaluate LAMBADA: exact match via greedy token prediction at each position."""
    n = 0
    correct = 0
    for ex in examples:
        context = ex["context"]
        continuation = ex["continuation"]
        continuation_ids = adapter.tokenize(continuation)

        # Check if model predicts the correct continuation greedily at each position
        context_ids = adapter.tokenize(context)
        all_ids = context_ids + continuation_ids

        if len(all_ids) > adapter.max_len:
            keep = max(1, adapter.max_len - len(continuation_ids))
            context_ids = context_ids[-keep:]
            all_ids = context_ids + continuation_ids

        all_tensor = torch.tensor([all_ids], dtype=torch.long, device=adapter.device)
        with torch.no_grad():
            logits = adapter.logits_fn(all_tensor)

        # Same shift as score_ids: the prediction for continuation token k
        # lives at logits position n_context + k - 1.
        n_context = len(context_ids)
        continuation_logits = logits[
            0, n_context - 1 : n_context - 1 + len(continuation_ids), :
        ]
        greedy_preds = torch.argmax(continuation_logits, dim=-1).cpu().tolist()

        if greedy_preds == continuation_ids:
            correct += 1
        n += 1

    return {"acc": correct / n if n > 0 else 0.0, "n": n}


# ============================================================================
# CLI and main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Eval ladder: unified scoring for our models and HF baselines."
    )
    # Model selection (XOR)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=str, help="Path to our checkpoint")
    model_group.add_argument("--hf-model", type=str, help="HuggingFace model ID")

    # Task selection
    parser.add_argument(
        "--tasks",
        type=str,
        default="core",
        help="Comma-separated task list or 'core' (all except mmlu) or 'all'",
    )

    # Options
    parser.add_argument("--iters", type=int, default=None, help="Tied model iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    parser.add_argument("--max-examples", type=int, default=1000, help="Max examples per task")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    # Load model
    if args.checkpoint:
        adapter = load_our_checkpoint(args.checkpoint, args.device, iters=args.iters)
        model_type = "checkpoint"
    else:
        adapter = load_hf_model(args.hf_model, args.device)
        model_type = "hf"

    # Parse tasks
    task_map = {
        "arc_easy": load_arc_easy,
        "arc_challenge": load_arc_challenge,
        "hellaswag": load_hellaswag,
        "piqa": load_piqa,
        "winogrande": load_winogrande,
        "sciq": load_sciq,
        "lambada_openai": load_lambada_openai,
        "mmlu": load_mmlu,
    }
    core_tasks = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "sciq", "lambada_openai"]

    if args.tasks == "core":
        task_names = core_tasks
    elif args.tasks == "all":
        task_names = list(task_map.keys())
    else:
        task_names = args.tasks.split(",")

    # Evaluate
    results = {"model": adapter.name, "model_type": model_type, "tasks": {}}
    if model_type == "checkpoint" and args.iters:
        results["iters"] = args.iters

    for task_name in task_names:
        if task_name not in task_map:
            print(f"Warning: unknown task {task_name}")
            continue
        print(f"Loading {task_name}...")
        # One task's loader breaking (a dataset repo changing format, a hub
        # outage) must cost that task alone, not the whole ladder: the first
        # rung sweep lost all four models to a single piqa failure.
        try:
            examples = task_map[task_name](args.max_examples, args.seed)
            print(f"Evaluating {task_name} ({len(examples)} examples)...")
            if task_name == "lambada_openai":
                task_results = eval_lambada(adapter, examples)
            else:
                task_results = eval_multiple_choice(adapter, examples)
        except Exception as exc:  # noqa: BLE001 - a failed task is a result
            task_results = {"error": f"{type(exc).__name__}: {exc}"}
        results["tasks"][task_name] = task_results
        print(f"  {task_name}: {task_results}")

    # Write output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
