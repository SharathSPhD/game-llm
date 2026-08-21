"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(42)
    yield


@pytest.fixture(scope="session", autouse=True)
def disable_gpu_lock_for_tests():
    """Temporarily disable GPU lock in state.json for testing.

    The lock is set during GPU training; for testing we need to disable it
    to allow job submission. We restore it after tests complete.
    """
    state_file = Path(__file__).parent.parent / "research" / "memory" / "state.json"

    if not state_file.exists():
        yield
        return

    # Read original state
    with open(state_file) as f:
        original_state = json.load(f)

    # Temporarily disable lock
    original_lock = original_state.get("gpu_lock", False)
    original_state["gpu_lock"] = False

    with open(state_file, "w") as f:
        json.dump(original_state, f, indent=2)

    yield

    # Restore original lock state
    original_state["gpu_lock"] = original_lock
    with open(state_file, "w") as f:
        json.dump(original_state, f, indent=2)
