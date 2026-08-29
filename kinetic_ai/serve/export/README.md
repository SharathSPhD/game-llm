# EqLM Export Infrastructure

Kinetic AI's EqLM (Equilibrium Language Model) is a weight-tied deep-equilibrium transformer. This module exports it to formats suitable for distribution and inference outside this codebase.

## Architecture Overview

EqLM differs fundamentally from standard stacked transformers:

- **Single Block**: One transformer block (EqLMBlock) applied iteratively.
- **Fixed-Point Solver**: Anderson acceleration (adaptive depth 1–12 iterations) solves z* = f(z*, x).
- **Weight Tying**: The block's parameters are used at every iteration; the lm_head shares weights with the token embedding.

This design achieves parameter efficiency: 120.7M parameters (versus ~124M for a 12-layer explicit baseline) while solving to fixed-point convergence.

## What Cannot Be Exported

**Solver telemetry**: The adaptive iteration count and convergence criterion are runtime properties of the Anderson solver, not the model itself. Any export to a format that doesn't support dynamic computation depth loses this property.

**Full parameter efficiency**: Formats like GGUF that don't support weight sharing or tensor aliasing will duplicate the block weights if unrolled to multiple iterations, negating the parameter savings.

## Export Options

### 1. SafeTensors (Recommended)

**Format**: `.safetensors` (PyTorch native, no overhead)

**Advantages**:
- Perfect fidelity to training checkpoint.
- No parameter duplication; preserves weight tying.
- Zero file size overhead.
- Loads natively in any PyTorch environment.

**Disadvantages**:
- Requires Python and PyTorch.
- Does not work with llama.cpp or other C++ runtimes without wrapping.

**Usage**:

```python
from kinetic_ai.serve.export import export_to_safetensors, load_from_safetensors
from kinetic_ai.models.eqlm import load_checkpoint

# Load model
model = load_checkpoint("results/scale/ckpt/eqlm_anytime_seed42.pt")

# Export
export_to_safetensors(model, "eqlm_model.safetensors")

# Load
loaded_model = load_from_safetensors("eqlm_model.safetensors")
logits = loaded_model(input_ids)
```

**When to Use**: Downstream research, fine-tuning, parameter studies, any case where PyTorch is available.

---

### 2. GGUF (with weight unrolling)

**Format**: `.gguf` (llama.cpp compatible)

**Advantages**:
- Compatible with llama.cpp and ecosystem (lm-studio, Ollama, etc.).
- Smaller binary footprint than Python-based inference.

**Disadvantages**:
- **Fundamental format mismatch**: llama.cpp expects a fixed sequence of layers; EqLM is iterative.
- **Parameter duplication**: To work around this, the tied block must be unrolled into N sequential layers with duplicated weights.
- **File size inflation**: Unrolling N times multiplies block parameters by N (approximately 2–12× on disk).
- **No convergence criterion**: llama.cpp applies exactly N fixed iterations; the adaptive solver is lost.

**Cost Analysis**:

| Config | Block Params | Unroll (N=2) | Unroll (N=4) | Unroll (N=12) |
|--------|-------------|--------------|-------------|--------------|
| Original | 34.8M | 69.6M | 139.2M | 417.6M |
| Disk Cost | 1× | 2× | 4× | 12× |

The full model (120.7M) unrolled to N iterations becomes (85.9M embeddings) + (34.8M × N block).

**Usage**:

```python
from kinetic_ai.serve.export import export_to_gguf

# Export with 4 unroll iterations
result = export_to_gguf(model, "eqlm_model.gguf", num_unroll_iters=4)
print(result["warning"])  # Warns about file size cost

# Run with llama.cpp
# $ llama-cli -m eqlm_model.gguf -n 256
```

**When to Use**: Only if target deployment is llama.cpp-exclusive and binary footprint (and the ability to run on CPU) is more important than model size.

**Honest Assessment**: GGUF export works but violates the integrity principle of the export system. The unrolled model runs exactly N fixed iterations, not the adaptive solver. Quality at N=4 may differ substantially from the adaptive solver model. This is acceptable for anytime inference (where you have a budget for 4 iterations) but not a faithful representation of the trained system.

---

### 3. ONNX (Fixed iteration count)

**Format**: `.onnx` (portable intermediate representation)

**Advantages**:
- Portable to any ONNX-compliant runtime (ONNX Runtime, TensorRT, CoreML, etc.).
- No parameter duplication; weight tying is preserved in the graph.
- Smaller file size than unrolled GGUF.

**Disadvantages**:
- Fixed iteration count (baked into the computation graph).
- No convergence criterion or solver telemetry.
- Requires a tracing pass (may not capture all dynamic behavior).

**Usage**:

```python
from kinetic_ai.serve.export import export_to_onnx

# Export with 4 fixed iterations
result = export_to_onnx(model, "eqlm_model.onnx", num_iters=4)
print(result["info"])  # Reports iteration count and file size

# Run with ONNX Runtime
import onnxruntime as rt
sess = rt.InferenceSession("eqlm_model.onnx", providers=["CPUExecutionProvider"])
logits = sess.run(None, {"input_ids": input_ids.numpy()})[0]
```

**When to Use**: Portable inference where ONNX runtimes are available and fixed depth is acceptable.

---

## Comparison Table

| Property | SafeTensors | GGUF | ONNX |
|----------|-------------|------|------|
| File Size | 1× | 2–12× | 1× |
| PyTorch Native | ✓ | ✗ | ✗ |
| llama.cpp Compatible | ✗ | ✓ | ✗ |
| ONNX Runtime Compatible | ✗ | ✗ | ✓ |
| Weight Tying Preserved | ✓ | ✗ (duplicated) | ✓ |
| Solver Telemetry | ✓ | ✗ | ✗ |
| Fidelity to Training | ✓ (perfect) | ⚠ (unrolled) | ⚠ (fixed depth) |

---

## Distribution Eligibility

Kinetic AI checked submission requirements for three platforms:

### OpenRouter

**Status**: Not eligible for their model marketplace.

**Reason**: OpenRouter accepts LLMs trained primarily on instruction-following datasets. EqLM is a 121M-parameter research model (46M compute-matched variant), pre-trained on BabyLM-scale Wikipedia and BookCorpus. It has not been instruction-tuned. OpenRouter explicitly states preference for models 7B+ parameters unless they are specialized (code, math) or instruction-tuned to high quality.

**Verdict**: Would require instruction-tuning and evaluation on their benchmarks first. Not a plug-and-play submission.

### LM Studio

**Status**: Technically possible but with caveats.

**Requirements**:
- Accepts `.gguf` format (or `.ggml`, `.safetensors`).
- Supports arbitrary architectures if described in GGUF metadata.
- No minimum model size requirement.

**Caveats**:
- LM Studio's UI targets consumer-facing models (7B–70B range); smaller models are runnable but not prominently featured.
- Community discoverability is low for research-scale models.
- No official review or curation step; authors self-submit via their model registry.

**Verdict**: Can submit via their registry at https://lmstudio.ai, but expect minimal visibility.

### Ollama

**Status**: Similar to LM Studio; technically possible.

**Requirements**:
- Accepts `.gguf` format via direct pull or manual `Modelfile` submission.
- Ollama's library favors models 7B+ that are broadly useful (instruction-tuned, general-purpose).
- No explicit size restriction but strong cultural preference for larger models.

**Process**:
- Create a `Modelfile` and submit to Ollama's registry via GitHub PR (ollama/ollama-library).
- Or run locally via `ollama pull huggingface.co/username/model-name`.

**Caveats**:
- EqLM is a research model without instruction-tuning or safety alignment. Ollama's community may reject submissions that don't meet minimum capability bars.
- The 121M size (46M compute-matched) is well below Ollama's typical model range.

**Verdict**: Submission is possible but likely to face slow curation and low adoption.

---

## Recommendation

**For public distribution**: SafeTensors export to Hugging Face Hub.

1. Export to SafeTensors: `export_to_safetensors(model, "eqlm.safetensors")`
2. Create a Hugging Face repository (e.g., `kinetic-ai/eqlm-121m`).
3. Upload `.safetensors` and `config.json`.
4. Add a `README.md` describing:
   - The model's architecture (weight-tied fixed-point DEQ transformer).
   - Limitations (121M parameters, 128 token context, BabyLM pre-training).
   - Usage example in PyTorch.
   - Citation and license.

Hugging Face's model card infrastructure supports:
- Arbitrary architecture descriptions.
- Detailed model cards with figures and tables.
- Community comments and discussions.
- Integration with inference endpoints (if applicable for research models).

**For research**: Keep SafeTensors as the primary format. ONNX is acceptable for portable inference in specific deployment targets. GGUF is honest but costly; use only if llama.cpp is the target runtime.

---

## Testing

All exports are tested for round-trip fidelity:

```bash
pytest tests/test_export.py -v
```

Tests verify:
- SafeTensors round-trip (export and load).
- Forward pass equivalence (outputs match original model to within 1e-5 absolute tolerance).
- Config metadata preservation.

---

## API Reference

### SafeTensors

```python
export_to_safetensors(model, output_path, save_config=True)
load_from_safetensors(weights_path, config_path=None) -> EqLM
validate_safetensors_export(model, export_path) -> bool
```

### GGUF

```python
export_to_gguf(model, output_path, num_unroll_iters=4, quantize_type=None) -> dict
verify_gguf_fidelity(model, gguf_path, num_iters=4) -> bool
```

### ONNX

```python
export_to_onnx(model, output_path, num_iters=4, opset_version=17) -> dict
validate_onnx_export(model, onnx_path, num_iters=4) -> bool
```

---

## References

**EqLM Architecture**:
- `kinetic_ai/models/eqlm.py`: Model definition (EqLM, EqLMBlock, EqLMCore).
- `research/memory/findings.md`: F24, F44, F45, F47 document the parameter-compute tradeoff.

**Export Formats**:
- SafeTensors: https://github.com/huggingface/safetensors
- GGUF: https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/format.py
- ONNX: https://onnx.ai

**Distribution Platforms**:
- Hugging Face Hub: https://huggingface.co
- OpenRouter: https://openrouter.ai (API partnerships for LLMs)
- LM Studio: https://lmstudio.ai (desktop inference UI)
- Ollama: https://ollama.ai (CLI + library)

---

**Last Updated**: 2026-08-29
