"""``qed.thermal``: finite-temperature thermodynamics over every symmetry sector of H."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import _core
from .symmetry import Symmetry

_METHODS = {"exact", "ftlm", "mtpq"}


@dataclass
class ThermalResult:
    """Thermodynamics per temperature. ``M`` and ``chi`` (magnetisation per system and
    susceptibility per site) are present when H conserves Sz."""

    T: np.ndarray
    E: np.ndarray
    C: np.ndarray
    S: np.ndarray
    F: np.ndarray
    lnZ: np.ndarray
    M: Optional[np.ndarray]
    chi: Optional[np.ndarray]
    method: str
    e0: float
    blocks: int
    symmetry: Symmetry = field(repr=False)


def thermal(H, T: Sequence[float], *, method: str = "ftlm", sym: Optional[Symmetry] = None,
            samples: int = 40, krylov: Optional[int] = None, exact_states: int = 0,
            seed: int = 0) -> ThermalResult:
    """Thermodynamics of ``H`` at the temperatures ``T``.

    ``method``: ``"exact"`` (every block's full spectrum), ``"ftlm"`` (finite-temperature
    Lanczos; ``exact_states > 0`` treats that many lowest states of each block exactly),
    or ``"mtpq"`` (microcanonical thermal pure quantum states). ``krylov`` is the FTLM
    Lanczos depth (default 100) or the mTPQ step count (default: automatic). ``samples``
    random vectors per block; ``seed`` 0 draws one.
    """
    key = str(method).lower()
    if key not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    sym = Symmetry.auto() if sym is None else sym
    t = _core.sectors.ThermalSpec()
    t.method = {"exact": _core.sectors.ThermalMethod.Exact, "ftlm": _core.sectors.ThermalMethod.FTLM,
                "mtpq": _core.sectors.ThermalMethod.mTPQ}[key]
    t.temperatures = [float(x) for x in T]
    t.samples = int(samples)
    t.krylov = int(krylov) if krylov is not None else (0 if key == "mtpq" else 100)
    t.exact_states = int(exact_states)
    t.seed = int(seed)
    r = _core.sectors.thermal(H, int(H.num_sites), sym.resolve(H), t)
    arr = lambda v: np.asarray(v, float)  # noqa: E731
    return ThermalResult(T=arr(r.T), E=arr(r.E), C=arr(r.C), S=arr(r.S), F=arr(r.F), lnZ=arr(r.lnZ),
                         M=arr(r.M) if len(r.M) else None, chi=arr(r.chi) if len(r.chi) else None,
                         method=key, e0=float(r.e0), blocks=int(r.blocks), symmetry=sym)
