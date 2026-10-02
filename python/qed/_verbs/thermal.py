"""``qed.thermal``: finite-temperature thermodynamics over every symmetry sector of H."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import _core, _log
from . import _device
from ..errors import InvalidRequest
from .symmetry import Symmetry

_METHODS = {"exact", "ftlm", "mtpq"}


@dataclass
class ThermalResult:
    """Thermodynamics per temperature: energy ``E``, heat capacity ``C``, ``entropy``, free
    energy ``F`` and ``lnZ``. ``M`` and ``chi`` (magnetisation per system and
    susceptibility per site) come from the Sz decomposition: present when H conserves Sz and
    the symmetry decomposes by it (None under ``Symmetry.none()`` or ``sz='off'``). A run
    restricted to part of the space (one Sz sector, a momentum or irrep selection, a total
    spin) is the canonical ensemble of that part -- its entropy tends to the log of the part's
    dimension -- and says so with a ``("restricted_ensemble", ...)`` entry in ``diagnostics``.
    ``O``: <O>(T) per requested observable, a complex array [len(observables), len(T)] (None
    without observables). ``diagnostics``: (code, message) pairs for fallbacks the run took."""

    T: np.ndarray
    E: np.ndarray
    C: np.ndarray
    entropy: np.ndarray
    F: np.ndarray
    lnZ: np.ndarray
    M: Optional[np.ndarray]
    chi: Optional[np.ndarray]
    O: Optional[np.ndarray]
    method: str
    e0: float
    blocks: int
    device_blocks: int
    symmetry: Symmetry = field(repr=False)
    diagnostics: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)


@_log.replays
def thermal(H, T: Sequence[float], *, method: str = "ftlm", sym: Optional[Symmetry] = None,
            samples: int = 40, krylov: Optional[int] = None, steps: Optional[int] = None,
            exact_states: int = 0, seed: int = 0, device: str = "cpu", observables: Optional[Sequence] = None,
            dense_max_dim: Optional[int] = None) -> ThermalResult:
    """Thermodynamics of ``H`` at the temperatures ``T``.

    ``method``: ``"exact"`` (every block's full spectrum), ``"ftlm"`` (finite-temperature
    Lanczos; ``exact_states > 0`` treats that many lowest states of each block exactly),
    or ``"mtpq"`` (microcanonical thermal pure quantum states). ``krylov`` is the FTLM
    Lanczos depth (default 100), ``steps`` the mTPQ steps per sample (default: enough for the
    coldest ``T``). ``samples`` random vectors per block; ``seed`` 0 draws one. ``dense_max_dim``: the sampled methods
    diagonalise blocks up to this dimension exactly instead (a sampled trace needs a dimension
    well above ``samples``); ``None`` is 512, 0 always samples.

    ``observables``: operators O whose thermal averages <O>(T) = Tr(e^{-H/T} O) / Z are
    returned in ``O`` (methods ``"exact"`` and ``"ftlm"`` without ``exact_states``). O may
    break the symmetries: each block uses O averaged over the symmetries it resolves, which
    has the same thermal average. Under ``total_spin`` every O must be SU(2) invariant.
    """
    key = str(method).lower()
    if key not in _METHODS:
        raise InvalidRequest(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    sym = Symmetry.auto() if sym is None else sym
    t = _core.sectors.ThermalSpec()
    t.method = {"exact": _core.sectors.ThermalMethod.Exact, "ftlm": _core.sectors.ThermalMethod.FTLM,
                "mtpq": _core.sectors.ThermalMethod.mTPQ}[key]
    t.temperatures = [float(x) for x in T]
    t.samples = int(samples)
    if key == "mtpq" and krylov is not None:
        raise InvalidRequest("method='mtpq' takes steps=, not krylov= (the Lanczos depth of FTLM)")
    if key != "mtpq" and steps is not None:
        raise InvalidRequest(f"steps= is the mTPQ step count; method={key!r} takes krylov=")
    if krylov is not None and int(krylov) < 1:
        raise InvalidRequest(f"krylov must be >= 1, got {krylov}")
    if steps is not None and int(steps) < 1:
        raise InvalidRequest(f"steps must be >= 1 (or None for automatic), got {steps}")
    t.krylov = 100 if krylov is None else int(krylov)
    t.steps = 0 if steps is None else int(steps)
    t.exact_states = int(exact_states)
    if dense_max_dim is not None:
        if int(dense_max_dim) < 0:
            raise InvalidRequest(f"dense_max_dim must be >= 0 or None, got {dense_max_dim}")
        t.dense_max_dim = int(dense_max_dim)
    t.seed = int(seed)
    t.device = _device.resolve(device)
    ops = [] if observables is None else list(observables)
    t.observables = ops
    diagnostics: list = []
    r = _core.sectors.thermal(H, sym.resolve(H, diagnostics), t)
    arr = lambda v: np.asarray(v, float)  # noqa: E731
    return ThermalResult(T=arr(r.T), E=arr(r.E), C=arr(r.C), entropy=arr(r.S), F=arr(r.F), lnZ=arr(r.lnZ),
                         M=arr(r.M) if len(r.M) else None, chi=arr(r.chi) if len(r.chi) else None,
                         O=np.asarray(r.O, complex) if ops else None,
                         method=key, e0=float(r.e0), blocks=int(r.blocks),
                         device_blocks=int(r.device_blocks), symmetry=sym,
                         diagnostics=diagnostics + [tuple(x) for x in r.diagnostics],
                         placement=dict(r.placement))
