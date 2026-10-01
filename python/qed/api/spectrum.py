"""``qed.spectrum``: the complete spectrum of H, block by block."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import _core, _log
from . import _device
from .symmetry import Labelled, Symmetry


@dataclass
class SpectrumResult(Labelled):
    """``energies``: every eigenvalue with its multiplicity, ascending. ``levels``: one
    entry per block eigenvalue with the block's quantum numbers and multiplicity.
    ``device_blocks``: blocks diagonalised on a GPU. ``diagnostics``: (code, message) pairs
    for fallbacks the run took."""

    energies: np.ndarray
    levels: list
    device_blocks: int
    symmetry: Symmetry
    _spec: object = None
    _n_sites: int = 0
    diagnostics: list = field(default_factory=list)


@_log.replays
def spectrum(H, *, sym: Optional[Symmetry] = None, device: str = "cpu") -> SpectrumResult:
    """Every eigenvalue of ``H``: each symmetry block is diagonalised densely (batched onto
    the GPU with ``device='gpu'``)."""
    sym = Symmetry.auto() if sym is None else sym
    diagnostics: list = []
    spec = sym.resolve(H, diagnostics)
    raw = _core.sectors.spectrum(H, int(H.num_sites), spec,
                                 device=_device.resolve(device))
    return SpectrumResult(energies=np.asarray(raw.expanded()), levels=list(raw.levels),
                          device_blocks=int(raw.device_blocks), symmetry=sym,
                          _spec=spec, _n_sites=int(H.num_sites),
                          diagnostics=diagnostics + [tuple(x) for x in raw.diagnostics])
