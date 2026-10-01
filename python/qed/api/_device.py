"""The ``device=`` keyword of the sector-resolved verbs.

``"cpu"`` runs on the host and never initialises CUDA. ``"gpu"`` is strict: it needs a CUDA
build and a visible device (else :class:`qed.errors.DeviceUnavailable`), and every Krylov
solve then runs on the device -- a block without a device kernel raises
:class:`qed.errors.DeviceUnsupported`; small blocks may still be solved densely on the host
(reported in the result's ``placement``). ``"auto"`` chooses per block.
"""
from __future__ import annotations

from .. import _core
from ..errors import DeviceUnavailable, InvalidRequest

_NAMES = {"cpu": "Cpu", "gpu": "Gpu", "auto": "Auto"}


def resolve(device: str):
    key = str(device).lower()
    if key not in _NAMES:
        raise InvalidRequest(f"device must be one of {sorted(_NAMES)}, got {device!r}")
    if key == "gpu":
        if not _core.has_cuda_build():
            raise DeviceUnavailable("device='gpu' needs a build with CUDA")
        if _core.cuda_device_count() == 0:
            raise DeviceUnavailable("device='gpu', but no CUDA device is visible "
                                    "(CUDA_VISIBLE_DEVICES, or a node without a GPU)")
    return getattr(_core.sectors.Device, _NAMES[key])
