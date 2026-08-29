"""ONNX export for EqLM (fixed iteration count).

ONNX is a portable intermediate representation supported by many runtimes
(ONNX Runtime, TensorRT, CoreML, etc.). This exporter traces a fixed-iteration
forward pass and exports the resulting computation graph.

LIMITATIONS:
- The exported model runs exactly num_iters iterations (typically 2–12).
- No convergence criterion; iteration count is baked into the graph.
- Solver telemetry (iteration counts, convergence info) is unavailable.
- Weight tying is preserved in ONNX but may or may not be optimized away
  by the target runtime.

ADVANTAGES OVER GGUF:
- No parameter duplication (weight tying is preserved).
- Portable to any ONNX-compliant runtime.
- Smaller file size than unrolled GGUF.

TRADEOFF:
- Fixed depth means quality depends on the chosen num_iters.
- Different from the adaptive solver on distribution shift.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from kinetic_ai.models.eqlm import EqLM


def export_to_onnx(
    model: EqLM,
    output_path: str | Path,
    num_iters: int = 4,
    opset_version: int = 17,
) -> dict[str, Any]:
    """Export EqLM to ONNX format with fixed iteration unrolling.

    Args:
        model: EqLM model to export.
        output_path: Path to output .onnx file.
        num_iters: Number of fixed iterations to unroll (typically 2–12).
        opset_version: ONNX opset version (17 is current standard).

    Returns:
        Dictionary with export metadata: original_params, file_size_mb,
        num_iters, opset_version.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = model.config

    # Create a wrapper that unrolls to exactly num_iters
    class UnrolledEqLM(torch.nn.Module):
        """Unrolled version of EqLM for tracing."""

        def __init__(self, eqlm_model: EqLM, num_unroll: int):
            super().__init__()
            self.eqlm = eqlm_model
            self.num_unroll = num_unroll

        def forward(self, input_ids: Tensor) -> Tensor:
            batch_size, seq_len = input_ids.shape

            # Embeddings
            z = self.eqlm.embedding(input_ids)
            positions = torch.arange(
                seq_len, device=input_ids.device, dtype=torch.long
            )
            z = z + self.eqlm.pos_embedding(positions)

            # Unroll block num_unroll times
            x = torch.zeros_like(z)
            for _ in range(self.num_unroll):
                z = self.eqlm.block(z, x)

            # Output projection
            z = self.eqlm.ln_final(z)
            logits = cast(
                Tensor, self.eqlm.lm_head(z) / (self.eqlm.config.d_model**0.5)
            )
            return logits

    unrolled_model = UnrolledEqLM(model, num_iters)
    unrolled_model.eval()

    # Dummy input for tracing
    batch_size, seq_len = 1, 32
    dummy_input = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    # Trace and export
    try:
        with torch.no_grad():
            traced = torch.jit.trace(unrolled_model, dummy_input)

        torch.onnx.export(
            traced,
            dummy_input,
            str(output_path),
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e

    # Save metadata
    metadata_path = output_path.parent / (output_path.stem + "_metadata.json")
    metadata = {
        "model_class": "EqLM",
        "export_type": "onnx",
        "num_iters": num_iters,
        "opset_version": opset_version,
        "config": {
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
        },
        "notes": (
            "Fixed-iteration unrolled ONNX export. Runs exactly num_iters iterations; "
            "no convergence criterion. Weight tying is preserved but may be optimized "
            "away by the target runtime."
        ),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Compute metrics
    file_size_bytes = output_path.stat().st_size if output_path.exists() else 0
    file_size_mb = file_size_bytes / (1024 * 1024)
    original_params = sum(p.numel() for p in model.parameters())

    return {
        "original_params": original_params,
        "file_size_mb": file_size_mb,
        "num_iters": num_iters,
        "opset_version": opset_version,
        "info": (
            f"ONNX export unrolled {num_iters} iterations. "
            f"File size: {file_size_mb:.1f} MB. "
            "Use any ONNX runtime (ONNX Runtime, TensorRT, CoreML, etc.)."
        ),
    }


def validate_onnx_export(
    model: EqLM, onnx_path: str | Path, num_iters: int = 4
) -> bool:
    """Validate ONNX export by comparing outputs with unrolled model.

    Args:
        model: Original EqLM model.
        onnx_path: Path to exported ONNX file.
        num_iters: Number of unroll iterations (must match export).

    Returns:
        True if outputs match within tolerance.

    Raises:
        AssertionError if validation fails.
    """
    try:
        import onnxruntime as rt
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for ONNX validation. "
            "Install with: pip install onnxruntime"
        ) from e

    # Load ONNX model
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Test input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len))

    # ONNX forward
    input_name = sess.get_inputs()[0].name
    onnx_output = sess.run(None, {input_name: input_ids.numpy()})[0]

    # Unrolled model forward
    with torch.no_grad():
        z = model.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        z = z + model.pos_embedding(positions)

        x = torch.zeros_like(z)
        for _ in range(num_iters):
            z = model.block(z, x)

        z = model.ln_final(z)
        expected_output = (
            model.lm_head(z) / (model.config.d_model**0.5)
        ).numpy()

    # Compare
    max_diff = abs(onnx_output - expected_output).max()
    assert (
        max_diff < 1e-4
    ), f"ONNX fidelity check failed. Max diff: {max_diff}"

    return True
