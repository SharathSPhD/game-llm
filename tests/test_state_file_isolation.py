"""The GPU lock must never be edited by the test suite.

On 2026-09-02 a reviewer's pytest run was killed by a timeout; the session
fixture that had rewritten research/memory/state.json to gpu_lock=false never
reached its teardown, and the repository's lock read "free" while a training
job held the GPU. Tests now point the lock readers at a private file through
KINETIC_STATE_FILE and leave the repository's state alone.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.serve.executor import STATE_FILE_ENV, LocalExecutor  # noqa: E402
from kinetic_ai.serve.profile import load_profile  # noqa: E402

REPO_STATE = Path(__file__).resolve().parent.parent / "research" / "memory" / "state.json"


def test_suite_runs_against_a_private_state_file() -> None:
    private = os.environ.get(STATE_FILE_ENV)
    assert private, "conftest must export a private state file"
    assert Path(private).resolve() != REPO_STATE.resolve()


def test_executor_default_follows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    locked = tmp_path / "locked.json"
    locked.write_text(json.dumps({"gpu_lock": True}))
    monkeypatch.setenv(STATE_FILE_ENV, str(locked))
    assert LocalExecutor()._is_gpu_locked() is True
    locked.write_text(json.dumps({"gpu_lock": False}))
    assert LocalExecutor()._is_gpu_locked() is False


def test_executor_env_is_read_at_check_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """app/server.py constructs its executor at import; the env must still win."""
    ex = LocalExecutor()
    locked = tmp_path / "late.json"
    locked.write_text(json.dumps({"gpu_lock": True}))
    monkeypatch.setenv(STATE_FILE_ENV, str(locked))
    assert ex._is_gpu_locked() is True


def test_explicit_state_file_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit.json"
    explicit.write_text(json.dumps({"gpu_lock": True}))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"gpu_lock": False}))
    monkeypatch.setenv(STATE_FILE_ENV, str(other))
    assert LocalExecutor(state_file=str(explicit))._is_gpu_locked() is True


def test_profile_lock_file_follows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"gpu_lock": True}))
    monkeypatch.setenv(STATE_FILE_ENV, str(f))
    assert load_profile("rtx5090").gpu_lock_file == f
