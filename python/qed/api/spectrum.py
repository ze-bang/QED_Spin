"""``qed.spectrum``: the complete spectrum of H, block by block."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .. import _core
from .symmetry import Symmetry


@dataclass
class SpectrumResult:
    """``energies``: every eigenvalue with its multiplicity, ascending. ``levels``: one
    entry per block eigenvalue with the block's quantum numbers and multiplicity."""

    energies: np.ndarray
    levels: list
    symmetry: Symmetry


def spectrum(H, *, sym: Optional[Symmetry] = None) -> SpectrumResult:
    """Every eigenvalue of ``H``: each symmetry block is diagonalised densely."""
    sym = Symmetry.auto() if sym is None else sym
    raw = _core.sectors.spectrum(H, int(H.num_sites), sym.resolve(H))
    return SpectrumResult(energies=np.asarray(raw.expanded()), levels=list(raw.levels),
                          symmetry=sym)
