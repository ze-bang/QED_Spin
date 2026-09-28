"""The ``device=`` keyword of the sector-resolved verbs."""
from __future__ import annotations

from .. import _core

_NAMES = {"cpu": "Cpu", "gpu": "Gpu", "auto": "Auto"}


def resolve(device: str):
    key = str(device).lower()
    if key not in _NAMES:
        raise ValueError(f"device must be one of {sorted(_NAMES)}, got {device!r}")
    if key == "gpu" and not _core.has_cuda_build():
        raise ValueError("device='gpu' needs a build with CUDA")
    return getattr(_core.sectors.Device, _NAMES[key])
