"""Tests for export infrastructure.

Tests round-trip fidelity for SafeTensors and ONNX exports.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from kinetic_ai.models.eqlm import EqLM, EqLMConfig
from kinetic_ai.serve.export.safetensors_export import (
    export_to_safetensors,
    load_from_safetensors,
    validate_safetensors_export,
)


def _has_safetensors():
    """Check if safetensors is available."""
    try:
        import safetensors  # noqa: F401
        return True
    except ImportError:
        return False


def _has_onnxruntime():
    """Check if onnxruntime is available."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def small_eqlm_model():
    """Fixture: small EqLM for testing."""
    cfg = EqLMConfig(
        vocab_size=256,
        d_model=128,
        n_heads=4,
        d_ff=512,
        max_seq_len=64,
        deq_max_iter=4,
        spectral_norm=False,  # Faster for tests
    )
    return EqLM(cfg)


def test_safetensors_export_and_load(small_eqlm_model):
    """Test SafeTensors export and round-trip loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.safetensors"

        # Export
        export_to_safetensors(small_eqlm_model, export_path, save_config=True)
        assert export_path.exists()

        # Load
        loaded_model = load_from_safetensors(export_path)
        assert loaded_model.config.d_model == small_eqlm_model.config.d_model
        assert loaded_model.config.vocab_size == small_eqlm_model.config.vocab_size


def test_safetensors_fidelity(small_eqlm_model):
    """Test SafeTensors export preserves model outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.safetensors"

        # Export and validate
        export_to_safetensors(small_eqlm_model, export_path)
        is_valid = validate_safetensors_export(small_eqlm_model, export_path)
        assert is_valid


def test_safetensors_forward_equivalence(small_eqlm_model):
    """Test exported model produces identical outputs."""
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, small_eqlm_model.config.vocab_size, (batch_size, seq_len))

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.safetensors"
        export_to_safetensors(small_eqlm_model, export_path)
        loaded_model = load_from_safetensors(export_path)

        # Put both models in eval mode to disable dropout
        small_eqlm_model.eval()
        loaded_model.eval()

        with torch.no_grad():
            orig_output = small_eqlm_model(input_ids)
            loaded_output = loaded_model(input_ids)

        assert torch.allclose(orig_output, loaded_output, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not _has_onnxruntime(), reason="onnxruntime not installed"
)
def test_onnx_export_and_validate(small_eqlm_model):
    """Test ONNX export creates a valid model."""
    pytest.importorskip("onnx")
    from kinetic_ai.serve.export.onnx_export import export_to_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.onnx"

        # Export
        result = export_to_onnx(small_eqlm_model, export_path, num_iters=2)
        assert export_path.exists()
        assert result["num_iters"] == 2
        assert result["file_size_mb"] > 0


@pytest.mark.skipif(
    not _has_safetensors(), reason="safetensors not installed"
)
def test_safetensors_metadata_saved(small_eqlm_model):
    """Test that config.json is saved alongside weights."""
    pytest.importorskip("safetensors")
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "model.safetensors"
        export_to_safetensors(small_eqlm_model, export_path, save_config=True)

        config_path = export_path.parent / "config.json"
        assert config_path.exists()

        import json
        with open(config_path) as f:
            metadata = json.load(f)
        assert metadata["model_class"] == "EqLM"
        assert metadata["format"] == "safetensors"
