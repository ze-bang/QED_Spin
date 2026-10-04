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
    without observables). ``e0``: the lowest energy the method resolved -- exact, the ground
    state; ``ftlm``, the lowest weighted Ritz value (with ``exact_states``, the lowest certified
    eigenvalue); ``mtpq``, the spectral-bounds Lanczos estimate. The sampled values are upper
    bounds on E0, independent of the temperature grid. ``diagnostics``: (code, message) pairs for
    fallbacks the run took. ``measurements``: the answers to ``requests`` (rows: the temperatures)."""

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
    measurements: list = field(default_factory=list)


@_log.replays
def thermal(
    H,
    T: Sequence[float],
    *,
    method: str = "ftlm",
    sym: Optional[Symmetry] = None,
    samples: int = 40,
    krylov: Optional[int] = None,
    steps: Optional[int] = None,
    exact_states: int = 0,
    seed: int = 0,
    device: str = "cpu",
    observables: Optional[Sequence] = None,
    dense_max_dim: Optional[int] = None,
    requests: Optional[Sequence] = None,
) -> ThermalResult:
    """Thermodynamics of ``H`` at the temperatures ``T``.

    ``method``: ``"exact"`` (every block's full spectrum), ``"ftlm"`` (finite-temperature
    Lanczos; ``exact_states > 0`` treats that many lowest states of each block exactly, as
    eigenpairs the block eigensolver certified by their residuals -- a block that could not
    certify them all samples the rest and says so in an ``("oftlm_exact_states", ...)``
    diagnostic),
    or ``"mtpq"`` (microcanonical thermal pure quantum states); only ``"ftlm"`` reads
    ``exact_states``. ``krylov`` is the FTLM Lanczos depth (default 100, at least 2), ``steps``
    the mTPQ steps per sample (default: enough for the
    coldest ``T``). ``samples`` random vectors per block; ``seed`` 0 draws one. ``dense_max_dim``: the sampled methods
    diagonalise blocks up to this dimension exactly instead (a sampled trace needs a dimension
    well above ``samples``); ``None`` is 512, 0 always samples.

    ``observables``: operators O whose thermal averages <O>(T) = Tr(e^{-H/T} O) / Z are
    returned in ``O`` (every method, OFTLM included). O may
    break the symmetries: each block uses O averaged over the symmetries it resolves, which
    has the same thermal average. Under ``total_spin`` with an SU(2)-symmetric H an O that is
    not SU(2) invariant enters through its SU(2)-scalar part (its average over all spin
    rotations), which has the same thermal average; in a uniform field it enters as it is.
    Every observable of a block comes from one sweep over its vectors, so many cost little more
    than one: exact, the eigenvectors; FTLM, each sample's phi(T) = sum_j e^{-E_j / 2T} <psi_j|r> psi_j
    (beside OFTLM's exact states); mTPQ, each step's state and its successor, combined by the
    canonical series of Sugiura and Shimizu (exact for an O that commutes with H, the standard
    approximation otherwise). ``requests``: :class:`qed.Expect` / :class:`qed.Correlations` measured
    in the same pass, answered in ``measurements`` with one row per temperature (as
    :func:`qed.measure` with ``T=`` does).
    """
    from .measure import _answers, _plan

    singles, pairs, plan = _plan([] if requests is None else list(requests), thermal=True)
    ops = [] if observables is None else list(observables)
    r, raw = _thermal_run(
        H,
        T,
        method=method,
        sym=sym,
        samples=samples,
        krylov=krylov,
        steps=steps,
        exact_states=exact_states,
        seed=seed,
        device=device,
        dense_max_dim=dense_max_dim,
        singles=ops + singles,
        pairs=pairs,
    )
    r.O = raw[: len(ops)] if ops else None
    if plan:
        r.measurements = _answers(plan, raw[len(ops) :].T, singles, pairs, _thermal_rows(r))
    return r


def _thermal_rows(r: ThermalResult) -> dict:
    """The row description of answers at temperatures (measure._answers)."""
    return dict(
        energies=r.E,
        multiplicities=np.ones(len(r.T), int),
        levels=[],
        eigs=None,
        diagnostics=list(r.diagnostics),
        rows="T",
        T=r.T,
    )


def _thermal_run(
    H, T, *, method, sym, samples, krylov, steps, exact_states, seed, device, dense_max_dim, singles, pairs
):
    """One thermal pass: (ThermalResult without O, raw <X>(T) as [singles ++ pairs, len(T)])."""
    key = str(method).lower()
    if key not in _METHODS:
        raise InvalidRequest(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    sym = Symmetry.auto() if sym is None else sym
    t = _core.sectors.ThermalSpec()
    t.method = {
        "exact": _core.sectors.ThermalMethod.Exact,
        "ftlm": _core.sectors.ThermalMethod.FTLM,
        "mtpq": _core.sectors.ThermalMethod.mTPQ,
    }[key]
    t.temperatures = [float(x) for x in T]
    t.samples = int(samples)
    if key == "mtpq" and krylov is not None:
        raise InvalidRequest("method='mtpq' takes steps=, not krylov= (the Lanczos depth of FTLM)")
    if key != "mtpq" and steps is not None:
        raise InvalidRequest(f"steps= is the mTPQ step count; method={key!r} takes krylov=")
    if krylov is not None and int(krylov) < 2:
        raise InvalidRequest(f"krylov must be >= 2 (the Lanczos depth of FTLM), got {krylov}")
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
    t.observables = list(singles)
    t.observable_pairs = [(list(A), list(B)) for A, B in pairs]
    diagnostics: list = []
    r = _core.sectors.thermal(H, sym.resolve(H, diagnostics), t)
    arr = lambda v: np.asarray(v, float)  # noqa: E731
    n_x = len(singles) + sum(len(A) * len(B) for A, B in pairs)
    raw = np.asarray(r.O, complex).reshape(n_x, len(t.temperatures)) if n_x else np.zeros((0, len(t.temperatures)))
    return ThermalResult(
        T=arr(r.T),
        E=arr(r.E),
        C=arr(r.C),
        entropy=arr(r.S),
        F=arr(r.F),
        lnZ=arr(r.lnZ),
        M=arr(r.M) if len(r.M) else None,
        chi=arr(r.chi) if len(r.chi) else None,
        O=None,
        method=key,
        e0=float(r.e0),
        blocks=int(r.blocks),
        device_blocks=int(r.device_blocks),
        symmetry=sym,
        diagnostics=diagnostics + [tuple(x) for x in r.diagnostics],
        placement=dict(r.placement),
    ), raw
