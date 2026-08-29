"""SafeTensors export for EqLM.

SafeTensors is the native PyTorch tensor format with perfect fidelity to the
training checkpoint. No format conversion overhead; no parameter duplication.

Recommended for: PyTorch-based downstream use, parameter studies, fine-tuning,
and any use case where Python is available.
"""

from __future__ import annotations

from pathlib import Path

import torch

from kinetic_ai.models.eqlm import EqLM, EqLMConfig


def export_to_safetensors(
    model: EqLM, output_path: str | Path, save_config: bool = True
) -> None:
    """Export EqLM to SafeTensors format.

    Saves model weights with full fidelity. Metadata (config, solver parameters,
    tokenizer info) stored as JSON alongside the tensors.

    Args:
        model: EqLM model to export.
        output_path: Path to output .safetensors file.
        save_config: If True, save config as JSON in output_path.parent / "config.json".

    Returns:
        None. Saves files to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file, save_model
    except ImportError as e:
        raise ImportError(
            "safetensors is required for this export. Install with: pip install safetensors"
        ) from e

    # Save using save_model which handles weight tying correctly
    # (embeds it in the safetensors metadata so no duplication on disk)
    try:
        save_model(model, str(output_path))
    except Exception:
        # Fallback: manually extract unique tensors to avoid duplication error
        state_dict = model.state_dict()
        # Only save embedding (lm_head.weight is tied to it)
        save_dict = {}
        seen_tensors = set()
        for k, v in state_dict.items():
            tensor_id = id(v.data.data_ptr())
            if tensor_id not in seen_tensors:
                save_dict[k] = v.detach().cpu()
                seen_tensors.add(tensor_id)
            # Skip lm_head.weight if it's the same as embedding.weight
            elif k == "lm_head.weight":
                continue
        save_file(save_dict, str(output_path))

    # Save metadata
    if save_config:
        import json

        config_path = output_path.parent / "config.json"
        from dataclasses import asdict

        metadata = {
            "model_class": "EqLM",
            "config": asdict(model.config),
            "format": "safetensors",
            "notes": (
                "Weight-tied fixed-point transformer. Single block solved to fixed point "
                "via Anderson acceleration (adaptive depth 1-12 iterations). "
                "Load with: from kinetic_ai.models.eqlm import EqLMConfig; "
                "from safetensors.torch import load_file; "
                "model = EqLM(EqLMConfig(...)); model.load_state_dict(load_file('weights.safetensors'))"
            ),
        }
        with open(config_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_from_safetensors(
    weights_path: str | Path, config_path: str | Path | None = None
) -> EqLM:
    """Load EqLM from SafeTensors format.

    Args:
        weights_path: Path to .safetensors file.
        config_path: Path to config.json. If None, looks in weights_path.parent.

    Returns:
        EqLM model with loaded weights.
    """
    weights_path = Path(weights_path)

    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise ImportError(
            "safetensors is required for this export. Install with: pip install safetensors"
        ) from e

    # Infer config path
    if config_path is None:
        config_path = weights_path.parent / "config.json"
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Ensure config.json was saved alongside the weights."
        )

    # Load metadata
    import json

    with open(config_path) as f:
        metadata = json.load(f)

    # Reconstruct model
    cfg = EqLMConfig(**metadata["config"])
    model = EqLM(cfg)

    # Load weights
    state_dict = load_file(str(weights_path))

    # Restore weight tying: if embedding.weight was saved but lm_head.weight was not,
    # manually tie them
    if "embedding.weight" in state_dict and "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["embedding.weight"]

    model.load_state_dict(state_dict)

    return model


def validate_safetensors_export(model: EqLM, export_path: str | Path) -> bool:
    """Validate SafeTensors export by round-trip and numerical fidelity.

    Args:
        model: Original EqLM model.
        export_path: Path to exported .safetensors file.

    Returns:
        True if export is valid and fidelity is within tolerance.

    Raises:
        AssertionError if validation fails.
    """
    # Load exported model
    loaded_model = load_from_safetensors(export_path)

    # Put both models in eval mode to disable dropout
    model.eval()
    loaded_model.eval()

    # Test input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    # Forward passes
    with torch.no_grad():
        orig_logits = model(input_ids)
        loaded_logits = loaded_model(input_ids)

    # Check fidelity
    assert torch.allclose(
        orig_logits, loaded_logits, atol=1e-5, rtol=1e-4
    ), f"SafeTensors export fidelity check failed. Max diff: {(orig_logits - loaded_logits).abs().max().item()}"

    return True
