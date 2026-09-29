"""``qed.spectrum``: the complete spectrum of H, block by block."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import _core
from . import _device
from .symmetry import Symmetry


@dataclass
class SpectrumResult:
    """``energies``: every eigenvalue with its multiplicity, ascending. ``levels``: one
    entry per block eigenvalue with the block's quantum numbers and multiplicity.
    ``device_blocks``: blocks diagonalised on a GPU."""

    energies: np.ndarray
    levels: list
    device_blocks: int
    symmetry: Symmetry


def spectrum(H, *, sym: Optional[Symmetry] = None, device: str = "cpu") -> SpectrumResult:
    """Every eigenvalue of ``H``: each symmetry block is diagonalised densely (batched onto
    the GPU with ``device='gpu'``)."""
    sym = Symmetry.auto() if sym is None else sym
    raw = _core.sectors.spectrum(H, int(H.num_sites), sym.resolve(H),
                                 device=_device.resolve(device))
    return SpectrumResult(energies=np.asarray(raw.expanded()), levels=list(raw.levels),
                          device_blocks=int(raw.device_blocks), symmetry=sym)
