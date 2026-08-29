"""EqLM export infrastructure.

This module exports the equilibrium language model to formats suitable for
distribution and inference outside the Kinetic AI training environment.

Architecture constraint: EqLM is a weight-tied fixed-point transformer, not a
stacked model. A single transformer block is applied iteratively via Anderson
acceleration (adaptive depth, 1-12 iterations) to solve z* = f(z*, x).

This constraint makes naive GGUF (llama.cpp) export problematic: llama.cpp expects
a fixed sequence of layers. Three export paths are provided:

1. SAFETENSORS (recommended): Preserves all structure, loads natively in PyTorch.
   No size overhead; full fidelity to the training checkpoint. Requires PyTorch.

2. GGUF with unrolling (honest but costly): Unrolls the block into N fixed layers
   with weight duplication (llama.cpp does not support weight sharing at the
   serialization level). File size grows by factor ≈ N; still readable by llama.cpp
   but loses the parameter-efficiency advantage that makes weight tying worthwhile.
   Most sensible at N=2 or N=3 (anytime depths from training).

3. ONNX (fixed iteration count): Exports a traced ONNX graph that runs exactly
   N iterations (no convergence criterion). Portable to any ONNX runtime, but
   loses adaptive depth and solver telemetry.

All exports are tested for numerical fidelity: outputs match the original model
to within floating-point tolerance (atol=1e-5, rtol=1e-4).
"""

from kinetic_ai.serve.export.gguf_export import export_to_gguf, verify_gguf_fidelity
from kinetic_ai.serve.export.onnx_export import export_to_onnx
from kinetic_ai.serve.export.safetensors_export import (
    export_to_safetensors,
    load_from_safetensors,
)

__all__ = [
    "export_to_safetensors",
    "load_from_safetensors",
    "export_to_gguf",
    "verify_gguf_fidelity",
    "export_to_onnx",
]
