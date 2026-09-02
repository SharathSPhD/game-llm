"""Pytest configuration and shared fixtures."""

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(42)
    yield


# The suite never edits research/memory/state.json. It exports a private state
# file with the lock free, so job submission works in tests while a training job
# keeps the real lock; a killed run cannot leave the repository unlocked.
# Set at import time because app/server.py builds its executor on import.
_PRIVATE_STATE_DIR = tempfile.mkdtemp(prefix="kinetic-test-state-")
_PRIVATE_STATE = Path(_PRIVATE_STATE_DIR) / "state.json"
_PRIVATE_STATE.write_text(json.dumps({"gpu_lock": False}))
os.environ.setdefault("KINETIC_STATE_FILE", str(_PRIVATE_STATE))
