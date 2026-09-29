"""``qed.dynamics``: dynamical correlations S(omega) of a probe O over the sectors of H."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import _core
from . import _device
from .symmetry import Symmetry


@dataclass
class DynamicsResult:
    """``S[i]`` is S(omega) at temperature ``T[i]``; at T = 0 ``T`` is empty and ``S`` has one
    row. omega is measured from the ground-state energy at T = 0 and is the transferred
    energy at T > 0."""

    omega: np.ndarray
    T: np.ndarray
    S: np.ndarray
    e0: float
    ground_manifold: int
    device_blocks: int
    symmetry: Symmetry = field(repr=False)


def dynamics(H, O, omega: Sequence[float], *, eta: float = 0.05,
             T: Optional[Sequence[float]] = None, sym: Optional[Symmetry] = None,
             krylov: int = 200, samples: int = 30, seed: int = 0,
             degeneracy_tol: float = 1e-8, device: str = "cpu") -> DynamicsResult:
    """S(omega) = sum_m p_m <m|O^dag delta(omega - H + E_m) O|m>, Lorentzian width ``eta``.

    ``T=None``: the ground state, averaged over a degenerate ground manifold.
    ``T=[...]``: finite-temperature Lanczos with ``samples`` random vectors per sector.
    ``O`` may change Sz (S+, S-) and need not share any symmetry of H.
    """
    sym = Symmetry.auto() if sym is None else sym
    d = _core.sectors.DynamicsSpec()
    d.omega = [float(w) for w in omega]
    d.eta = float(eta)
    d.temperatures = [] if T is None else [float(t) for t in (np.atleast_1d(T))]
    if any(t <= 0 for t in d.temperatures):
        raise ValueError("temperatures must be positive; use T=None for the ground state")
    d.krylov = int(krylov)
    d.samples = int(samples)
    d.seed = int(seed)
    d.degeneracy_tol = float(degeneracy_tol)
    d.device = _device.resolve(device)
    r = _core.sectors.dynamics(H, int(H.num_sites), sym.resolve(H), O, d)
    return DynamicsResult(omega=np.asarray(r.omega), T=np.asarray(r.T), S=np.asarray(r.S),
                          e0=float(r.e0), ground_manifold=int(r.ground_manifold),
                          device_blocks=int(r.device_blocks), symmetry=sym)
