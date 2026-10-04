"""``qed.dynamics``: dynamical correlations S_AB(omega) over the sectors of H."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import _core, _log
from . import _device
from ..errors import InvalidRequest
from ..family import Family, MomentumFamily
from .symmetry import Symmetry


@dataclass
class DynamicsResult:
    """``S[..., i, :]`` is S(omega) at temperature ``T[i]``; at T = 0 ``T`` is empty and that axis has
    one row. The leading axes are the probes' (none for one ``O``): ``[len(O)]``, a family's shape
    (a :class:`qed.MomentumFamily`: ``(*lead, n_q)``, its momenta in ``q``), or those axes twice for
    ``B="all"``. ``index``: the family ``O`` was, if any. S is real for autocorrelations and complex once a probe
    pairs two operators. omega is measured from the ground-state energy at T = 0 and is the
    transferred energy at T > 0. ``diagnostics``: (code, message) pairs for fallbacks the run took."""

    omega: np.ndarray
    T: np.ndarray
    S: np.ndarray
    e0: float
    ground_manifold: int
    device_blocks: int
    symmetry: Symmetry = field(repr=False)
    diagnostics: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)
    q: Optional[np.ndarray] = None
    index: object = None


def _operators(x, what: str):
    """(operators, index shape) of a probe operand: an Operator (no axis), a sequence, a qed.Family
    (its shape) or a qed.MomentumFamily (its O_q, shape (*lead, n_q))."""
    if isinstance(x, MomentumFamily):
        return x.operators(), list(x.shape)
    if isinstance(x, Family):
        return list(x.ops), list(x.shape)
    if isinstance(x, _core.Operator):
        return [x], []
    try:
        ops = list(x)
    except TypeError:
        ops = []  # not an Operator and not iterable (a number, None)
    if not ops or not all(isinstance(o, _core.Operator) for o in ops):
        raise InvalidRequest(
            f"{what} must be a qed.Operator, a non-empty sequence of them, a qed.Family or a " "qed.MomentumFamily"
        )
    return ops, [len(ops)]


def _probes(O, B):
    """The (A, B) pairs of a call, the shape of their probe axes, and whether any is a cross pair."""
    ops, axes = _operators(O, "O")
    if B is None:
        return [(o, None) for o in ops], axes, False
    if isinstance(B, str):
        if B != "all":
            raise InvalidRequest(f"B must be None, a qed.Operator, a sequence of them or 'all', got {B!r}")
        return [(a, b) for a in ops for b in ops], axes + axes, True
    if isinstance(B, _core.Operator):
        return [(o, B) for o in ops], axes, True
    bs, _ = _operators(B, "B")
    if len(bs) != len(ops):
        raise InvalidRequest(f"B as a sequence pairs with O: {len(ops)} qed.Operator(s) expected, got {len(bs)}")
    return list(zip(ops, bs)), axes, True


@_log.replays
def dynamics(
    H,
    O,
    omega: Sequence[float],
    B=None,
    *,
    eta: float = 0.05,
    T: Optional[Sequence[float]] = None,
    sym: Optional[Symmetry] = None,
    krylov: int = 200,
    samples: int = 40,
    seed: int = 0,
    degeneracy_tol: float = 1e-8,
    device: str = "cpu",
    dense_max_dim: Optional[int] = None,
    prune: bool = True,
) -> DynamicsResult:
    """S_AB(omega) = sum_m p_m <m|A^dag delta(omega - H + E_m) B|m>, Lorentzian width ``eta``.

    ``O`` is the probe A, a sequence of them, a :class:`qed.Family` or a :class:`qed.MomentumFamily`
    (``qed.Family.spins(lattice, "z").fourier("cluster")`` gives S^zz(q, omega) at every cluster
    momentum). ``B``: ``None`` gives each O's autocorrelation (real); a qed.Operator, every
    <O_i^dag B>; a sequence or family as long as ``O``, the pairs <O_i^dag B_i>; ``"all"``, every
    <O_i^dag O_j> (a matrix of spectra; at T = 0 the pairs sharing a B share its Lanczos runs). One
    call shares the ground manifold (T = 0) and the source sectors (T > 0) among its probes.

    ``T=None``: the ground state, averaged over a degenerate ground manifold: every level within
    ``degeneracy_tol`` times the scale of H (the sum of |c| over its terms) of E0.
    ``T=[...]``: finite-temperature Lanczos with ``samples`` random vectors per sector; a
    temperature listed twice gets the same row twice.
    The probes may change Sz (S+, S-) and need not share any symmetry of H, so dynamics works in
    momentum sectors. ``sym.select(sz=..., momentum=...)`` restricts the source states (the
    ground state of those sectors, or their restricted ensemble at T > 0, flagged in
    ``diagnostics``); ``k0``, ``irrep`` and ``irrep_character`` name point-group blocks and
    raise :class:`qed.errors.Unsupported`. ``spin_flip`` / ``time_reversal='require'`` check
    that H has the symmetry. ``dense_max_dim`` is the dense crossover of the ground-state
    eigensolve at T = 0, as in :func:`qed.eigs`; ``prune=False`` solves every block there (no
    block is skipped on its 40-step estimate).
    """
    if dense_max_dim is not None and int(dense_max_dim) < 0:
        raise InvalidRequest(f"dense_max_dim must be >= 0 or None, got {dense_max_dim}")
    probes, axes, cross = _probes(O, B)
    sym = Symmetry.auto() if sym is None else sym
    d, temps, rows = _spec(omega, eta, T, krylov, samples, seed, degeneracy_tol, device, dense_max_dim, prune)
    diagnostics: list = []
    r = _core.sectors.dynamics(H, sym.resolve(H, diagnostics), probes, d)
    return _result(r, np.asarray(r.S, dtype=complex), O, axes, cross, temps, rows, sym, diagnostics)


def _spec(omega, eta, T, krylov, samples, seed, degeneracy_tol, device, dense_max_dim, prune):
    """(DynamicsSpec, the caller's temperatures, the row of each in the distinct ones)."""
    d = _core.sectors.DynamicsSpec()
    d.omega = [float(w) for w in omega]
    d.eta = float(eta)
    temps = np.zeros(0) if T is None else np.atleast_1d(np.asarray(T, dtype=float))
    if T is not None and temps.size == 0:
        raise InvalidRequest("T is empty; use T=None for the ground state")
    if not np.all(np.isfinite(temps)) or np.any(temps <= 0):
        raise InvalidRequest("temperatures must be finite and positive; use T=None for the ground state")
    unique, rows = np.unique(temps, return_inverse=True)
    d.temperatures = [float(t) for t in unique]
    d.krylov = int(krylov)
    d.samples = int(samples)
    d.seed = int(seed)
    d.degeneracy_tol = float(degeneracy_tol)
    d.dense_max_dim = -1 if dense_max_dim is None else int(dense_max_dim)
    d.prune = bool(prune)
    d.device = _device.resolve(device)
    return d, temps, rows.reshape(-1)


def _result(r, S, O, axes, cross, temps, rows, sym, diagnostics) -> DynamicsResult:
    """A DynamicsResult from the engine's curves and the [probe, row, omega] block of S that is O's."""
    if len(temps):  # the caller's temperatures, in the caller's order
        S = S[:, rows, :]
    if not cross:
        S = S.real
    S = S.reshape(tuple(axes) + S.shape[1:])
    return DynamicsResult(
        omega=np.asarray(r.omega),
        T=temps,
        S=S,
        e0=float(r.e0),
        ground_manifold=int(r.ground_manifold),
        device_blocks=int(r.device_blocks),
        symmetry=sym,
        diagnostics=diagnostics + [tuple(x) for x in r.diagnostics],
        placement=dict(r.placement),
        q=O.q if isinstance(O, MomentumFamily) else None,
        index=O if isinstance(O, (Family, MomentumFamily)) else None,
    )
