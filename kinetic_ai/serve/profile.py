"""Host-independent serving profiles (ADR 0010 §4).

Everything that differs between serving hosts — port, device policy, memory
budget, allowed origins, tunnel settings — lives in one YAML file per host
under ``configs/serve/profiles/`` and is selected by the ``KINETIC_SERVE_PROFILE``
environment variable. Code never names a host. Switching from the RTX 5090 back
to the GB10 when it returns is a change of that one variable.

The device policy is the part that makes coexistence safe. A profile may pin
``cpu`` or ``cuda``, or say ``auto``: use the GPU unless a training job holds the
lock recorded in ``research/memory/state.json`` (``gpu_lock: true``), in which
case serve from the CPU. The 5090 carries the training queue, so its profile
says ``auto`` and a small residency budget; the GB10 has no training and 121 GB.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROFILE_ENV = "KINETIC_SERVE_PROFILE"
DEFAULT_PROFILE = "rtx5090"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROFILES_DIR = REPO_ROOT / "configs" / "serve" / "profiles"

#: Scalar fields an environment variable may override, as KINETIC_SERVE_<FIELD>.
_ENV_OVERRIDES: dict[str, type] = {"host": str, "port": int, "device": str, "results_dir": str}


@dataclass(frozen=True)
class ServeProfile:
    """One serving host, fully described."""

    name: str
    host: str
    port: int
    device: str  # "auto" | "cuda" | "cpu"
    dtype: str
    max_resident_gb: float
    allowed_origins: tuple[str, ...]
    results_dir: str = "./results"
    gpu_lock_file: Path | None = None
    tunnel_enabled: bool = False
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def list_profiles() -> list[str]:
    """Names of the profiles shipped in the repo."""
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def load_profile(name: str | None = None) -> ServeProfile:
    """Load a profile by name, or by ``KINETIC_SERVE_PROFILE``.

    Raises:
        KeyError: The named profile has no YAML file. Failing here is the
            point — falling back to a host that is not present is how a demo
            silently serves nothing.
    """
    name = name or os.environ.get(PROFILE_ENV, DEFAULT_PROFILE)
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise KeyError(f"no serving profile {name!r}; available: {list_profiles()}")
    raw = yaml.safe_load(path.read_text()) or {}

    for key, cast in _ENV_OVERRIDES.items():
        env = os.environ.get(f"KINETIC_SERVE_{key.upper()}")
        if env is not None:
            raw[key] = cast(env)

    lock = raw.get("gpu_lock_file")
    lock_path = (REPO_ROOT / lock) if lock and not Path(lock).is_absolute() else (Path(lock) if lock else None)
    known = {"host", "port", "device", "dtype", "max_resident_gb", "allowed_origins",
             "results_dir", "gpu_lock_file", "tunnel", "description", "name"}
    return ServeProfile(
        name=name,
        host=str(raw["host"]),
        port=int(raw["port"]),
        device=str(raw.get("device", "auto")),
        dtype=str(raw.get("dtype", "float32")),
        max_resident_gb=float(raw["max_resident_gb"]),
        allowed_origins=tuple(str(o) for o in raw.get("allowed_origins", []) if o != "*"),
        results_dir=str(raw.get("results_dir", "./results")),
        gpu_lock_file=lock_path,
        tunnel_enabled=bool((raw.get("tunnel") or {}).get("enabled", False)),
        description=str(raw.get("description", "")),
        extra={k: v for k, v in raw.items() if k not in known},
    )


def gpu_lock_held(lock_file: Path | None) -> bool:
    """Whether a training job holds the GPU per the recorded state.

    A missing or unreadable file counts as free: the lock is advisory state
    written by the research loop, and its absence means no loop is running.
    """
    if lock_file is None or not lock_file.exists():
        return False
    try:
        return bool(json.loads(lock_file.read_text()).get("gpu_lock", False))
    except (OSError, ValueError):
        return False


def resolve_device(profile: ServeProfile, cuda_available: bool | None = None) -> str:
    """Turn the profile's device policy into a concrete torch device string."""
    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()
    if profile.device == "cpu":
        return "cpu"
    if profile.device == "cuda":
        return "cuda" if cuda_available else "cpu"
    if not cuda_available or gpu_lock_held(profile.gpu_lock_file):
        return "cpu"
    return "cuda"


def _main() -> None:
    """``python -m kinetic_ai.serve.profile [FIELD]`` — print a field for shell scripts."""
    import sys

    p = load_profile()
    if len(sys.argv) == 1:
        print(json.dumps({**p.__dict__, "gpu_lock_file": str(p.gpu_lock_file),
                          "resolved_device": resolve_device(p)}, indent=2))
        return
    key = sys.argv[1]
    if key == "resolved_device":
        print(resolve_device(p))
    elif key == "allowed_origins":
        print(",".join(p.allowed_origins))
    else:
        print(getattr(p, key))


if __name__ == "__main__":
    _main()
