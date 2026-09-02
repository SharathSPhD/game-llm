"""TDD for host-independent serving profiles (ADR 0010 §4).

Serving must run on whichever machine is available — the GB10 when it is
here, the RTX 5090 while it is away — with the difference confined to a YAML
profile selected by one environment variable. The rules pinned here are the
ones that make the switch safe: the 5090 profile must never touch the GPU
while a training job holds the lock, every profile must resolve to a concrete
device, and an unknown profile must fail loudly rather than fall back to a
host that is not there.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinetic_ai.serve.profile import (  # noqa: E402
    PROFILE_ENV,
    ServeProfile,
    list_profiles,
    load_profile,
    resolve_device,
)

REPO = Path(__file__).resolve().parent.parent


class TestProfilesShipped:
    def test_both_hosts_have_a_profile(self) -> None:
        names = list_profiles()
        assert "gb10" in names and "rtx5090" in names

    def test_every_profile_loads_with_required_fields(self) -> None:
        for name in list_profiles():
            p = load_profile(name)
            assert isinstance(p, ServeProfile)
            assert p.name == name
            assert p.host and 1024 <= p.port <= 65535
            assert p.device in {"auto", "cuda", "cpu"}
            assert p.max_resident_gb > 0
            assert p.allowed_origins, f"{name} must list explicit origins"
            assert "*" not in p.allowed_origins

    def test_5090_profile_fits_beside_a_training_job(self) -> None:
        p = load_profile("rtx5090")
        # 32 GB card; the trainer holds ~25 GB, so the serving budget must
        # leave it room rather than assume the GB10's 121 GB.
        assert p.max_resident_gb <= 6
        assert p.gpu_lock_file is not None


class TestSelection:
    def test_env_selects_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROFILE_ENV, "gb10")
        assert load_profile().name == "gb10"
        monkeypatch.setenv(PROFILE_ENV, "rtx5090")
        assert load_profile().name == "rtx5090"

    def test_unknown_profile_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROFILE_ENV, "dgx-h100")
        with pytest.raises(KeyError, match="dgx-h100"):
            load_profile()

    def test_env_overrides_scalar_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROFILE_ENV, "rtx5090")
        monkeypatch.setenv("KINETIC_SERVE_PORT", "8123")
        assert load_profile().port == 8123


class TestDeviceResolution:
    def _state(self, tmp_path: Path, locked: bool) -> Path:
        f = tmp_path / "state.json"
        f.write_text(json.dumps({"gpu_lock": locked}))
        return f

    def test_auto_yields_cpu_while_training_holds_the_lock(self, tmp_path: Path) -> None:
        p = load_profile("rtx5090")
        p = ServeProfile(**{**p.__dict__, "device": "auto", "gpu_lock_file": self._state(tmp_path, True)})
        assert resolve_device(p, cuda_available=True) == "cpu"

    def test_auto_yields_cuda_when_lock_is_free(self, tmp_path: Path) -> None:
        p = load_profile("rtx5090")
        p = ServeProfile(**{**p.__dict__, "device": "auto", "gpu_lock_file": self._state(tmp_path, False)})
        assert resolve_device(p, cuda_available=True) == "cuda"

    def test_auto_yields_cpu_without_cuda(self, tmp_path: Path) -> None:
        p = load_profile("gb10")
        p = ServeProfile(**{**p.__dict__, "device": "auto", "gpu_lock_file": None})
        assert resolve_device(p, cuda_available=False) == "cpu"

    def test_explicit_cpu_is_honoured_even_with_cuda(self) -> None:
        p = load_profile("gb10")
        p = ServeProfile(**{**p.__dict__, "device": "cpu"})
        assert resolve_device(p, cuda_available=True) == "cpu"

    def test_missing_lock_file_counts_as_free(self, tmp_path: Path) -> None:
        p = load_profile("rtx5090")
        p = ServeProfile(**{**p.__dict__, "device": "auto", "gpu_lock_file": tmp_path / "absent.json"})
        assert resolve_device(p, cuda_available=True) == "cuda"
