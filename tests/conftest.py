"""Pytest configuration and shared fixtures."""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility across all tests."""
    torch.manual_seed(42)
    yield
