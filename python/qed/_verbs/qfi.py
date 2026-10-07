"""Thermal quantum Fisher information from energy-resolved Lanczos poles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .. import _core, _log
from ..errors import InvalidRequest
from .dynamics import _operators, _spec
from .symmetry import Symmetry


@dataclass
class QFIResult:
    """QFI for each Hermitian generator, with temperature on the last axis.

    ``F``, ``F_positive`` and ``F_squared`` use respectively 4 tanh(w/2T),
    4 theta(w) tanh(w/2T) (1-exp(-w/T)), and 4 tanh(w/2T)^2. They agree for
    a converged canonical spectrum. ``balance_error`` is their largest absolute
    disagreement; it is a diagnostic, not a rigorous error bar. ``weight`` is
    the zeroth spectral moment <O^2>. Values are never clipped or symmetrised.
    ``E``, ``C`` and ``lnZ`` describe the whole system. No division by site count
    is implicit: normalise the generators explicitly to obtain a QFI density.
    """

    T: np.ndarray
    F: np.ndarray
    F_positive: np.ndarray
    F_squared: np.ndarray
    weight: np.ndarray
    balance_error: np.ndarray
    E: np.ndarray
    C: np.ndarray
    lnZ: np.ndarray
    e0: float
    samples: int
    krylov: int
    seed: int
    device_blocks: int
    symmetry: Symmetry = field(repr=False)
    diagnostics: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)


@_log.replays
def qfi(
    H,
    O,
    T: Sequence[float],
    *,
    sym: Optional[Symmetry] = None,
    krylov: int = 160,
    samples: int = 40,
    seed: int = 0,
    device: str = "cpu",
) -> QFIResult:
    """Thermal QFI of Hermitian generators using symmetry-resolved FTLM.

    The canonical spectrum is evaluated at each transition energy E_j-E_i,
    retaining the initial-state energies. All four pole moments are accumulated
    directly, with no Lorentzian broadening, frequency cutoff or integration grid.
    Both Krylov bases use full reorthogonalisation. Source runs also give E, C
    and lnZ. This is the Jaklic-Prelovsek finite-temperature Lanczos estimator;
    samples and Krylov depth must be converged, especially at low temperature.

    ``O`` accepts the same operator/family containers as ``dynamics``. Every
    generator must be Hermitian. For a complex Fourier mode use its Hermitian
    cosine and sine components and report their QFIs separately or their sum.
    The state is the full canonical ensemble; sector selections are refused.
    Degenerate states contribute with their thermal weights, not as a chosen
    pure ground state. Use independent nonzero seeds to estimate uncertainty.
    """
    ops, axes = _operators(O, "O")
    if any(not op.is_hermitian() for op in ops):
        raise InvalidRequest("qfi requires Hermitian generators; split complex modes into cosine and sine components")
    sym = Symmetry.auto() if sym is None else sym
    if (
        sym.sz not in (None, "auto", "off")
        or sym.total_spin is not None
        or len(sym.only_k0)
        or len(sym.only_irrep)
        or len(sym.only_momentum)
        or len(sym.only_irrep_character)
    ):
        raise InvalidRequest("qfi requires the full canonical ensemble; sector selections are not supported")
    if T is None:
        raise InvalidRequest("qfi requires positive temperatures; T=None is not a thermal state")
    d, temps, rows = _spec([0.0, 1.0, 2.0, 3.0], 1.0, T, krylov, samples, seed, 1e-8, device, None, True)
    d.qfi_moments = True
    d.thermodynamics = True
    diagnostics: list = []
    r = _core.sectors.dynamics(H, sym.resolve(H, diagnostics), [(o, None) for o in ops], d)
    raw = np.asarray(r.S, complex)[:, rows, :].reshape(tuple(axes) + (len(temps), 4))
    f, positive, squared, weight = (raw[..., i].real for i in range(4))
    return QFIResult(
        T=temps,
        F=f,
        F_positive=positive,
        F_squared=squared,
        weight=weight,
        balance_error=np.maximum(np.abs(f - positive), np.abs(f - squared)),
        E=np.asarray(r.E, float)[rows],
        C=np.asarray(r.V, float)[rows] / temps**2,
        lnZ=np.asarray(r.lnZ, float)[rows],
        e0=float(r.e0),
        samples=int(samples),
        krylov=int(krylov),
        seed=int(seed),
        device_blocks=int(r.device_blocks),
        symmetry=sym,
        diagnostics=diagnostics + [tuple(x) for x in r.diagnostics],
        placement=dict(r.placement),
    )
