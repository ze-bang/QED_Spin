"""``qed.dynamics``: dynamical correlations S(omega) of a probe O over the sectors of H."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import _core, _log
from . import _device
from ..errors import InvalidRequest
from .symmetry import Symmetry


@dataclass
class DynamicsResult:
    """``S[i]`` is S(omega) at temperature ``T[i]``; at T = 0 ``T`` is empty and ``S`` has one
    row. omega is measured from the ground-state energy at T = 0 and is the transferred
    energy at T > 0. ``diagnostics``: (code, message) pairs for fallbacks the run took."""

    omega: np.ndarray
    T: np.ndarray
    S: np.ndarray
    e0: float
    ground_manifold: int
    device_blocks: int
    symmetry: Symmetry = field(repr=False)
    diagnostics: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)


@_log.replays
def dynamics(H, O, omega: Sequence[float], *, eta: float = 0.05,
             T: Optional[Sequence[float]] = None, sym: Optional[Symmetry] = None,
             krylov: int = 200, samples: int = 40, seed: int = 0,
             degeneracy_tol: float = 1e-8, device: str = "cpu",
             dense_max_dim: Optional[int] = None) -> DynamicsResult:
    """S(omega) = sum_m p_m <m|O^dag delta(omega - H + E_m) O|m>, Lorentzian width ``eta``.

    ``T=None``: the ground state, averaged over a degenerate ground manifold.
    ``T=[...]``: finite-temperature Lanczos with ``samples`` random vectors per sector; a
    temperature listed twice gets the same row twice.
    ``O`` may change Sz (S+, S-) and need not share any symmetry of H, so dynamics works in
    momentum sectors. ``sym.select(sz=..., momentum=...)`` restricts the source states (the
    ground state of those sectors, or their restricted ensemble at T > 0, flagged in
    ``diagnostics``); ``k0``, ``irrep`` and ``irrep_character`` name point-group blocks and
    raise :class:`qed.errors.Unsupported`. ``spin_flip`` / ``time_reversal='require'`` check
    that H has the symmetry. ``dense_max_dim`` is the dense crossover of the ground-state
    eigensolve at T = 0, as in :func:`qed.eigs`.
    """
    if dense_max_dim is not None and int(dense_max_dim) < 0:
        raise InvalidRequest(f"dense_max_dim must be >= 0 or None, got {dense_max_dim}")
    sym = Symmetry.auto() if sym is None else sym
    d = _core.sectors.DynamicsSpec()
    d.omega = [float(w) for w in omega]
    d.eta = float(eta)
    temps = np.zeros(0) if T is None else np.atleast_1d(np.asarray(T, dtype=float))
    if not np.all(np.isfinite(temps)) or np.any(temps <= 0):
        raise InvalidRequest("temperatures must be finite and positive; use T=None for the ground state")
    unique, rows = np.unique(temps, return_inverse=True)
    d.temperatures = [float(t) for t in unique]
    d.krylov = int(krylov)
    d.samples = int(samples)
    d.seed = int(seed)
    d.degeneracy_tol = float(degeneracy_tol)
    d.dense_max_dim = -1 if dense_max_dim is None else int(dense_max_dim)
    d.device = _device.resolve(device)
    diagnostics: list = []
    r = _core.sectors.dynamics(H, sym.resolve(H, diagnostics), O, d)
    S = np.asarray(r.S)
    if len(temps):                       # the caller's temperatures, in the caller's order
        S = S[rows.reshape(-1)]
    return DynamicsResult(omega=np.asarray(r.omega), T=temps, S=S,
                          e0=float(r.e0), ground_manifold=int(r.ground_manifold),
                          device_blocks=int(r.device_blocks), symmetry=sym,
                          diagnostics=diagnostics + [tuple(x) for x in r.diagnostics],
                          placement=dict(r.placement))
