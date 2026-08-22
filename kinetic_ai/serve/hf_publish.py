"""Publish trained models to Hugging Face Hub.

This module provides utilities to push checkpoints saved by kinetic_ai.models.eqlm
to Hugging Face Hub with auto-generated model cards including:
  - Architecture summary from checkpoint config
  - Provenance block (config sha, git commit, run directory)
  - Metrics table pulled from sibling results.json
  - Links to paper, GitHub, and findings.md
  - License and disclaimer

Usage:
  publish_checkpoint_to_hf(
    checkpoint_path="results/exp09/checkpoints/model.pt",
    repo_id="kinetic-ai/eqlm-babylm-10m",
  )

Auth:
  - Requires HF_TOKEN env var or huggingface-hub login (hf_hub_download reads both)
  - HfApi().whoami() must succeed; raises RuntimeError if not authenticated
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, upload_file


def generate_model_card(
    title: str,
    architecture_summary: str,
    config_sha: str,
    git_commit: str,
    run_dir: str,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Generate a model card markdown for a checkpoint.

    Args:
        title: Model name / title for the card
        architecture_summary: Brief architecture description (e.g., "EqLM-768, DEQ, 12 layers")
        config_sha: Config hash for reproducibility (from results.json)
        git_commit: Git commit SHA (from results.json)
        run_dir: Results directory path (e.g., "results/exp09_adaptive")
        metrics: Dictionary of metrics to include in card (e.g., {"val_loss": 8.7, "blip": 0.92})

    Returns:
        Markdown string for the model card.
    """
    if metrics is None:
        metrics = {}

    # Build metrics table
    metrics_table = ""
    if metrics:
        metrics_table = "\n## Metrics\n\n| Metric | Value |\n|--------|-------|\n"
        for key, value in metrics.items():
            # Format numeric values nicely
            if isinstance(value, float):
                formatted = f"{value:.4f}" if value < 1000 else f"{value:,.0f}"
            else:
                formatted = str(value)
            metrics_table += f"| {key} | {formatted} |\n"

    card = f"""---
license: mit
---

# {title}

This is an equilibrium language model from the **Kinetic AI** research project.

## Architecture

{architecture_summary}

## Provenance

- **Config Hash:** `{config_sha}`
- **Git Commit:** `{git_commit}`
- **Run Directory:** `{run_dir}`

All numbers in this model card trace to documented runs; see [research/memory/findings.md](https://github.com/SharathSPhD/game-llm/blob/main/research/memory/findings.md) for validation details.
{metrics_table}

## Disclaimer

This is research-stage code. Numbers reported here are from `results.json` in the named run directory and trace to `research/memory/findings.md`. For baseline comparisons and metric definitions, consult the paper at https://github.com/SharathSPhD/game-llm/tree/main/paper.

## Links

- [GitHub Repository](https://github.com/SharathSPhD/game-llm)
- [Paper (arXiv/Site)](https://github.com/SharathSPhD/game-llm/tree/main/paper)
- [Validated Findings](https://github.com/SharathSPhD/game-llm/blob/main/research/memory/findings.md)
- [Kinetic AI Home](https://kinetic.kinetic-ai.workers.dev)
"""
    return card


def get_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load checkpoint metadata without materializing the model.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.

    Returns:
        Dictionary with keys: config, model_class, n_layers (if applicable).

    Raises:
        FileNotFoundError: Checkpoint does not exist.
        RuntimeError: Checkpoint load fails or config is malformed.
    """
    import torch

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load with weights_only=False since we trust our own checkpoints
    # but only extract metadata, not instantiate the model
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") from e

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Checkpoint is not a dict: {type(ckpt)}")

    # Extract metadata without instantiating model
    config = ckpt.get("config")
    model_class = ckpt.get("model_class")

    if config is None or model_class is None:
        raise RuntimeError("Checkpoint missing 'config' or 'model_class' key")

    metadata = {"config": config, "model_class": model_class}

    # For ExplicitLM, include layer count
    if "n_layers" in ckpt:
        metadata["n_layers"] = ckpt["n_layers"]

    return metadata


def get_metrics_from_results_json(run_dir: str | Path) -> dict[str, Any]:
    """Extract metrics from sibling results.json file.

    Args:
        run_dir: Results directory (parent of checkpoints/ subdirectory).

    Returns:
        Dictionary of metrics found in results.json['arms'][...]. Empty dict if
        results.json does not exist or is malformed.
    """
    run_dir = Path(run_dir)
    results_json_path = run_dir / "results.json"

    if not results_json_path.exists():
        return {}

    try:
        with open(results_json_path) as f:
            results = json.load(f)
    except Exception:
        return {}

    # Extract metrics from the arms section
    metrics = {}

    # Get config_hash and git_commit for provenance
    if "config_hash" in results:
        metrics["config_sha"] = results["config_hash"]
    if "git_commit" in results:
        metrics["git_commit"] = results["git_commit"]

    # Extract key metrics from arms (usually first arm)
    if "arms" in results and isinstance(results["arms"], dict):
        for _arm_name, arm_data in results["arms"].items():
            if isinstance(arm_data, dict):
                # Pull out common metrics
                for key in ["final_loss", "val_loss", "test_loss", "blip", "metrics"]:
                    if key in arm_data:
                        val = arm_data[key]
                        if key == "metrics" and isinstance(val, dict):
                            # Flatten nested metrics
                            metrics.update(val)
                        else:
                            metrics[key] = val
                # Only take first arm for simplicity
                break

    return metrics


def publish_checkpoint_to_hf(
    checkpoint_path: str | Path,
    repo_id: str,
) -> str:
    """Publish a checkpoint to Hugging Face Hub.

    Args:
        checkpoint_path: Path to checkpoint (must be inside results/).
        repo_id: Repo ID in format "owner/name" (e.g., "kinetic-ai/eqlm-babylm-10m").

    Returns:
        URL of the published repo on Hugging Face Hub.

    Raises:
        ValueError: checkpoint_path outside results/, repo_id malformed, HF auth fails.
        FileNotFoundError: Checkpoint or run directory does not exist.
        RuntimeError: HF API call fails.
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    # Verify checkpoint is inside results/ (security: no traversal)
    results_dir = Path("results").resolve()
    try:
        checkpoint_path.relative_to(results_dir)
    except ValueError as e:
        raise ValueError(
            f"Checkpoint path {checkpoint_path} is outside results/ directory. "
            "For security, only checkpoints inside results/ can be published."
        ) from e

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Validate repo_id format
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise ValueError(f"repo_id must be in format 'owner/name', got: {repo_id}")

    # Get checkpoint metadata
    metadata = get_checkpoint_metadata(checkpoint_path)
    config = metadata["config"]
    model_class = metadata["model_class"]

    # Get run directory (parent of checkpoints/)
    run_dir = checkpoint_path.parent.parent

    # Extract metrics from results.json
    metrics = get_metrics_from_results_json(run_dir)

    # Verify HF authentication
    api = HfApi()
    try:
        api.whoami()
    except Exception as e:
        raise RuntimeError(
            f"Hugging Face authentication failed: {e}. "
            "Ensure HF_TOKEN is set or you are logged in via `huggingface-cli login`."
        ) from e

    # Build architecture summary from config
    if hasattr(config, "d_model"):
        arch_parts = [f"d_model={config.d_model}"]
        if hasattr(config, "n_heads"):
            arch_parts.append(f"n_heads={config.n_heads}")
        if hasattr(config, "deq_max_iter"):
            arch_parts.append(f"DEQ (max_iter={config.deq_max_iter})")
        elif model_class == "ExplicitLM" and "n_layers" in metadata:
            arch_parts.append(f"{metadata['n_layers']} layers")
        architecture_summary = ", ".join(arch_parts)
    else:
        architecture_summary = f"Config: {config}"

    # Generate model card
    config_sha = metrics.pop("config_sha", "unknown")
    git_commit = metrics.pop("git_commit", "unknown")

    model_card_md = generate_model_card(
        title=f"{model_class} — {repo_id.split('/')[-1]}",
        architecture_summary=architecture_summary,
        config_sha=config_sha,
        git_commit=git_commit,
        run_dir=str(run_dir),
        metrics=metrics,
    )

    # Create repo on HF (exist_ok=True allows publishing to existing repo)
    try:
        repo_url = api.create_repo(repo_id, exist_ok=True, private=False).url
    except Exception as e:
        raise RuntimeError(f"Failed to create HF repo {repo_id}: {e}") from e

    # Upload checkpoint
    try:
        upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="model.pt",
            repo_id=repo_id,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to upload checkpoint to {repo_id}: {e}") from e

    # Upload model card as README
    try:
        # Write card to temp file and upload
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(model_card_md)
            card_path = f.name

        upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=repo_id,
        )

        # Clean up temp file
        Path(card_path).unlink()
    except Exception as e:
        raise RuntimeError(f"Failed to upload model card to {repo_id}: {e}") from e

    return repo_url
