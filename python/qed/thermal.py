"""qed.thermal -- one canonical call for finite-T diagonalization.

This module is the Python mirror of the C++ entry point
``ed::auto_pilot::thermal(...)`` declared in
``QED/include/ed/auto/thermal.h``. The motivation, as raised by the
matvec-unification audit, was: *"I just want one unified way that
automatically gives me the proper thermodynamics, fully optimised with
the Sz and symmetry sectors. Essentially the user just calls FTLM and
we get the results, but the internal steps are all automatically
optimised and using all possible symmetries."*

Usage::

    import qed
    res = qed.thermal(H, method="FTLM", T_min=0.05, T_max=5.0, num_T=64)
    res.thermo.energy        # full-Hilbert <H>(T)
    res.thermo.specific_heat # full-Hilbert C_v(T)
    res.per_sector           # one entry per Sz sector that was run

For a Hamiltonian on disk (with optional ``automorphism_results/`` for
spatial symmetry)::

    res = qed.thermal("ed_dir/", num_sites=12, method="FTLM",
                      sz_min=N//2 - 2, sz_max=N//2 + 2)

What the function does internally:

    1.  Detect Sz conservation by inspecting the operator (in-memory)
        or by loading the operator metadata (directory form).
    2.  Iterate ``n_up`` over the requested window (default: full
        ``[0, N]``).  Empty sectors and sectors outside ``[0, N]`` are
        silently skipped.
    3.  For each sector, dispatch through :func:`qed.diag` with
        ``sz=n_up`` and -- in the directory form -- ``symmetry=``
        auto-loaded from ``automorphism_results/``. Each per-sector call
        therefore exploits BOTH the Sz and the spatial-symmetry axis.
    4.  Z-recombine the per-sector :class:`ThermodynamicData` blocks
        using the same free-energy weighting as the C++
        ``ed::core::combine_sector_thermodynamics``.

The returned :class:`ThermalResult` exposes the recombined
thermodynamics plus a per-sector breakdown for diagnostics.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from . import _core
from ._core import (
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    Operator,
    exact_diagonalization_from_directory,
)
from .workflow import diag

__all__ = ["ThermalResult", "ThermalSectorEntry", "thermal"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class ThermalSectorEntry:
    """One entry of :attr:`ThermalResult.per_sector`."""

    n_up: Optional[int]
    sector_dim: int
    temperatures: np.ndarray
    energy: np.ndarray
    specific_heat: np.ndarray
    entropy: np.ndarray
    free_energy: np.ndarray


@dataclass
class ThermalResult:
    """Return type of :func:`qed.thermal`.

    The headline payload is the full-Hilbert thermodynamics in
    :attr:`temperatures`, :attr:`energy`, :attr:`specific_heat`,
    :attr:`entropy`, and :attr:`free_energy`. :attr:`per_sector`
    carries the per-Sz breakdown that produced it (in-memory case)
    or the per-Sz, irrep-already-recombined breakdown (directory case).
    """

    temperatures: np.ndarray
    energy: np.ndarray
    specific_heat: np.ndarray
    entropy: np.ndarray
    free_energy: np.ndarray
    method: str
    ground_state_energy: float
    used_sz_decomposition: bool
    used_symmetry_decomposition: bool
    per_sector: list[ThermalSectorEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_THERMAL_METHODS = {
    "FTLM": DiagonalizationMethod.FTLM,
    "LTLM": DiagonalizationMethod.LTLM,
    "KPM_DOS": DiagonalizationMethod.KPM_DOS,
    "MTPQ": DiagonalizationMethod.mTPQ,
    "CTPQ": DiagonalizationMethod.cTPQ,
}

# Methods that write per-sample HDF5 trajectories on disk and need a
# real (non-/dev/null) output directory so the recombination step can
# read them back. Mirrors the C++ ``detail::method_needs_output_dir``.
_TPQ_METHODS = {
    DiagonalizationMethod.mTPQ,
    DiagonalizationMethod.cTPQ,
}


def _allocate_tpq_workdir(root: str, method_tag: str, n_up: Optional[int]) -> str:
    """Allocate a unique scratch directory for a TPQ sector.

    Returns the directory path. Caller is responsible for cleanup.
    """
    base = root if root else os.path.join(tempfile.gettempdir(), "qed_thermal")
    os.makedirs(base, exist_ok=True)
    suffix = f"_n{n_up}" if n_up is not None else ""
    stamp = f"{time.time_ns()}"
    workdir = os.path.join(base, f"{method_tag}{suffix}_{stamp}")
    os.makedirs(workdir, exist_ok=True)
    return workdir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_method(method: Union[str, DiagonalizationMethod]) -> DiagonalizationMethod:
    if isinstance(method, DiagonalizationMethod):
        return method
    if isinstance(method, str):
        key = method.upper().replace("-", "_")
        if key in _THERMAL_METHODS:
            return _THERMAL_METHODS[key]
        # mTPQ / cTPQ are case-sensitive in the enum.
        if hasattr(DiagonalizationMethod, method):
            return getattr(DiagonalizationMethod, method)
        raise ValueError(
            f"qed.thermal: unknown thermal method '{method}'. "
            f"Supported: {sorted(_THERMAL_METHODS)}"
        )
    raise TypeError(
        f"qed.thermal: method must be str or DiagonalizationMethod, "
        f"got {type(method).__name__}"
    )


def _sector_thermo_arrays(res: EDResults) -> Optional[tuple[np.ndarray, ...]]:
    """Extract (temperatures, energy, C_v, entropy, F) from an EDResults.

    Returns ``None`` if the result carries no usable thermo block
    (e.g. KPM_DOS without the temperature post-processing, or a method
    that errored out).
    """
    t = res.thermo_data
    temps = np.asarray(t.temperatures, dtype=float)
    if temps.size == 0:
        return None
    energy = np.asarray(t.energy, dtype=float)
    cv = np.asarray(t.specific_heat, dtype=float)
    entropy = np.asarray(t.entropy, dtype=float)
    F = np.asarray(t.free_energy, dtype=float)
    if energy.size == 0 or cv.size == 0 or F.size == 0:
        return None
    return temps, energy, cv, entropy, F


def _combine_sector_thermodynamics(
    per_sector_thermo: list[tuple[np.ndarray, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recombine per-sector thermodynamics via free-energy Z-weighting.

    Mirror of the C++ ``ed::core::combine_sector_thermodynamics``. The
    math is the same:

        F_ref(beta)    = min_s F_s(beta)
        Z_s_shift      = exp(-beta (F_s - F_ref))
        Z_total        = sum_s Z_s_shift
        w_s            = Z_s_shift / Z_total
        <E>_total      = sum_s w_s * <E>_s
        <E2>_s         = C_s / beta^2 + <E>_s^2
        <E2>_total     = sum_s w_s * <E2>_s
        F_total        = F_ref - T * ln(Z_total)
        S_total        = beta * (<E>_total - F_total)
        C_total        = beta^2 * (<E2>_total - <E>_total^2)
    """
    assert per_sector_thermo, "need at least one sector"

    # Sanity: every sector must share the same T grid.
    temps0 = per_sector_thermo[0][0]
    n_T = temps0.size
    for s, (temps, *_rest) in enumerate(per_sector_thermo):
        if temps.size != n_T or not np.allclose(temps, temps0):
            raise ValueError(
                f"qed.thermal: sector {s} temperature grid disagrees "
                f"with the reference; cannot recombine."
            )

    # 1/T with the safe-zero handling at beta = +inf.
    safe_T = np.where(temps0 > 0.0, temps0, 1e-300)
    beta = 1.0 / safe_T

    # Stack per-sector arrays into shape (S, n_T).
    F_S = np.stack([s[4] for s in per_sector_thermo], axis=0)
    E_S = np.stack([s[1] for s in per_sector_thermo], axis=0)
    C_S = np.stack([s[2] for s in per_sector_thermo], axis=0)

    F_ref = F_S.min(axis=0)
    log_w = -beta[None, :] * (F_S - F_ref[None, :])
    w_max = log_w.max(axis=0)
    Z_shift = np.exp(log_w - w_max[None, :])
    Z_total = Z_shift.sum(axis=0)
    w = Z_shift / Z_total[None, :]

    # log Z_total (with the w_max offset) for the free energy.
    lnZ = np.log(Z_total) + w_max
    F_total = F_ref - safe_T * lnZ

    E_total = (w * E_S).sum(axis=0)
    # <E^2>_s reconstructed from C_v and <E>_s.
    E2_S = C_S / (beta[None, :] ** 2) + E_S ** 2
    E2_total = (w * E2_S).sum(axis=0)

    C_total = (beta ** 2) * (E2_total - E_total ** 2)
    S_total = beta * (E_total - F_total)

    return temps0, E_total, C_total, S_total, F_total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def thermal(
    H: Union[Operator, str],
    *,
    method: Union[str, DiagonalizationMethod] = "FTLM",
    T_min: float = 0.1,
    T_max: float = 10.0,
    num_T: int = 24,
    sz_min: Optional[int] = None,
    sz_max: Optional[int] = None,
    num_samples: int = 40,
    krylov_dim: Optional[int] = None,
    ftlm_krylov_dim: Optional[int] = None,
    ltlm_krylov_dim: Optional[int] = None,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
    random_seed: int = 0,
    use_symmetry_if_available: bool = True,
    use_sz_if_conserved: bool = True,
    auto_tune: bool = True,
    level: str = "balanced",
    output_dir: str = "",
    verbose: bool = True,
    # ---- KPM_DOS specific ------------------------------------------
    kpm_num_moments: int = 200,
    kpm_num_random_vectors: int = 16,
    # ---- TPQ specific (mTPQ + cTPQ) --------------------------------
    tpq_num_measure_points: int = 100,
    # ``None`` -> auto-derive from (T_min, T_max). For mTPQ/cTPQ we
    # need the measurement β grid to bracket every target β = 1/T
    # the recombiner asks for, so pin it to ``1/T_max`` / ``1/T_min``
    # by default and let the user override.
    tpq_measure_beta_min: Optional[float] = None,
    tpq_measure_beta_max: Optional[float] = None,
    tpq_delta_beta: float = 0.05,
    tpq_taylor_order: int = 8,
    tpq_measurement_interval: int = 1,
    # ``0.0`` -> auto-pick via a quick Lanczos spectral-bound estimate
    # inside ``exact_diagonalization_core`` (single source of truth).
    tpq_energy_shift: float = 0.0,
    # Directory-form-only knobs.
    num_sites: Optional[int] = None,
    spin: float = 0.5,
    extra_params: Optional[dict[str, Any]] = None,
) -> ThermalResult:
    """One canonical call for finite-T diagonalization.

    Auto-detects Sz conservation (and, for the directory form,
    spatial symmetry via ``automorphism_results/``). Iterates the
    Sz window, dispatches the chosen finite-T solver on each sector
    (with the spatial symmetry already applied), and Z-recombines
    the per-sector thermodynamics into a full-Hilbert
    :class:`ThermalResult`.

    Parameters
    ----------
    H : Operator or str
        In-memory :class:`Operator` instance OR path to a directory
        containing ``Trans.dat`` / ``InterAll.dat`` (and optionally
        ``automorphism_results/``). For the directory form
        ``num_sites`` is required.
    method : str or DiagonalizationMethod, optional
        Finite-T method: ``"FTLM"`` (default), ``"LTLM"``,
        ``"KPM_DOS"``, ``"mTPQ"``, ``"cTPQ"``.
    T_min, T_max, num_T : float / int, optional
        Temperature grid. Linear in T by convention; FTLM/LTLM use
        ``num_T`` evenly-spaced points in ``[T_min, T_max]``.
    sz_min, sz_max : int, optional
        Sz window in ``n_up`` convention (``Sz_total = n_up - N/2``).
        Defaults to the full ``[0, N]`` range when Sz is conserved.
        Use this to restrict to "adjacent Sz" bands that carry the
        partition-function weight at the temperatures of interest.
    num_samples, krylov_dim, tolerance, random_seed : optional
        Per-sector FTLM / LTLM solver knobs.
    use_symmetry_if_available, use_sz_if_conserved : bool, optional
        Force-off toggles for the auto-detection. Default both True.
    auto_tune, level : optional
        Phase-9.3 auto-tuner. ``level`` is one of ``"conservative"``,
        ``"balanced"``, ``"aggressive"``.
    output_dir : str, optional
        Where to write per-sector HDF5 sinks; default ``""`` means
        no I/O.
    verbose : bool, optional
        Print per-sector progress when True (default).
    num_sites, spin : optional
        Required for the directory form.
    extra_params : dict, optional
        Forwarded to :func:`qed.diag` per sector. Use for any niche
        ``EDParameters`` field this helper doesn't expose.

    Returns
    -------
    ThermalResult
        Recombined full-Hilbert thermodynamics plus per-sector
        breakdown.

    Examples
    --------

    .. code-block:: python

        import qed
        H = qed.build_heisenberg_chain(8, J=1.0, periodic=True)
        res = qed.thermal(H, T_min=0.05, T_max=5.0, num_T=64)
        # res.energy[t]        -- <H>(T_t) over the full 2^8 Hilbert space
        # res.specific_heat[t] -- C_v(T_t)
        # res.per_sector       -- per-n_up breakdown

    Sz-window restriction (skip sectors whose partition function is
    negligible at the T of interest):

    .. code-block:: python

        res = qed.thermal(H, T_min=0.05, T_max=2.0, num_T=64,
                          sz_min=N//2 - 1, sz_max=N//2 + 1)
    """
    method_enum = _coerce_method(method)
    is_directory = isinstance(H, (str, os.PathLike))

    if is_directory and num_sites is None:
        raise ValueError(
            "qed.thermal: pass num_sites=... when using the "
            "directory form."
        )

    # ------------------------------------------------------------------
    # Method-specific knob plumbing. Anything that touches an
    # EDParameters field but isn't a generic FTLM knob gets injected
    # here and merged with the user's own `extra_params`. The merge
    # order is: defaults < method-specific < explicit `extra_params`.
    # ------------------------------------------------------------------
    method_extra: dict[str, Any] = {
        "max_iterations": int(max_iterations),
    }
    if method_enum == DiagonalizationMethod.KPM_DOS:
        method_extra.update(
            kpm_num_moments=int(kpm_num_moments),
            kpm_num_random_vectors=int(kpm_num_random_vectors),
            kpm_seed=int(random_seed) if random_seed else 0,
        )
    elif method_enum in _TPQ_METHODS:
        # Auto-derive the TPQ measurement β grid from (T_min, T_max)
        # unless the user pinned them explicitly. We add a small
        # buffer on each end so the interpolation inside
        # ``compute_tpq_unified_thermo`` brackets the target T grid.
        auto_beta_min = 1.0 / max(T_max, 1e-300) * 0.5
        auto_beta_max = 1.0 / max(T_min, 1e-300) * 2.0
        beta_min_eff = float(tpq_measure_beta_min) if tpq_measure_beta_min is not None \
                       else auto_beta_min
        beta_max_eff = float(tpq_measure_beta_max) if tpq_measure_beta_max is not None \
                       else auto_beta_max
        method_extra.update(
            tpq_num_measure_points=int(tpq_num_measure_points),
            tpq_measure_beta_min=beta_min_eff,
            tpq_measure_beta_max=beta_max_eff,
            tpq_delta_beta=float(tpq_delta_beta),
            tpq_taylor_order=int(tpq_taylor_order),
            tpq_measurement_interval=int(tpq_measurement_interval),
            tpq_energy_shift=float(tpq_energy_shift),
        )

    merged_extra: dict[str, Any] = {**method_extra}
    if extra_params:
        merged_extra.update(extra_params)

    # TPQ needs a real (non-/dev/null) output dir per sector so its
    # per-sample HDF5 trajectories survive long enough for
    # ``compute_tpq_unified_thermo`` to read them back. We allocate
    # scratch dirs on-demand and clean them up when the call returns.
    tpq_scratch_dirs: list[str] = []
    needs_scratch = method_enum in _TPQ_METHODS and not output_dir

    def _alloc_scratch(n_up: Optional[int]) -> str:
        tag = "ctpq" if method_enum == DiagonalizationMethod.cTPQ else "mtpq"
        d = _allocate_tpq_workdir(root="", method_tag=tag, n_up=n_up)
        tpq_scratch_dirs.append(d)
        return d

    def _cleanup_scratch() -> None:
        for d in tpq_scratch_dirs:
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Load / inspect Hamiltonian to decide auto-axes.
    # ------------------------------------------------------------------
    if is_directory:
        H_op = Operator(num_sites=int(num_sites), spin=float(spin))
        directory = str(H)
        trans = os.path.join(directory, "Trans.dat")
        inter = os.path.join(directory, "InterAll.dat")
        if os.path.exists(trans):
            H_op.load_trans(trans)
        if os.path.exists(inter):
            H_op.load_inter_all(inter)
        # `automorphism_results/` triggers the streaming-symmetry route
        # via qed.diag(symmetry=...). When present we load the generator
        # set; qed.diag handles the per-irrep streaming internally.
        sym_dir = os.path.join(directory, "automorphism_results")
        has_sym = (
            use_symmetry_if_available
            and os.path.isdir(sym_dir)
            and any(
                os.path.exists(os.path.join(sym_dir, n))
                for n in (
                    "automorphisms.json", "max_clique.json",
                    "sector_metadata.json", "minimal_generators.json",
                    "sectors.json", "generators.json",
                )
            )
        )
    else:
        H_op = H
        directory = None
        has_sym = False
        # In-memory symmetry support would require an
        # explicit `symmetry=` arg; for now we route the
        # in-memory path through plain Sz iteration only.

    N = int(H_op.num_sites)
    sz_conserved = use_sz_if_conserved and bool(H_op.conserves_sz())

    if verbose:
        which = "directory" if is_directory else "in-memory"
        print(
            f"[qed.thermal] {which}  N={N}  "
            f"sz_conserved={sz_conserved}  symmetry={has_sym}  "
            f"method={method}"
        )

    # ------------------------------------------------------------------
    # 2. Resolve the Sz window early (used by both branches).
    # ------------------------------------------------------------------
    if sz_conserved:
        lo = int(sz_min) if sz_min is not None else 0
        hi = int(sz_max) if sz_max is not None else N
        if lo > hi:
            raise ValueError(
                f"qed.thermal: sz_min={lo} > sz_max={hi}."
            )
        if lo < 0 or hi > N:
            raise ValueError(
                f"qed.thermal: Sz window [{lo}, {hi}] must lie in "
                f"[0, {N}]."
            )

    # ------------------------------------------------------------------
    # 3a. Directory form -- per-Sz dispatch through the C++ dispatcher
    #     which handles spatial symmetry internally when
    #     `params.use_symmetry=True`.
    # ------------------------------------------------------------------
    if is_directory:
        per_sector_blocks: list[tuple[np.ndarray, ...]] = []
        per_sector_records: list[ThermalSectorEntry] = []
        gs_E = math.inf

        def _make_dir_params(n_up_val: Optional[int]) -> EDParameters:
            p = EDParameters()
            p.num_sites = N
            p.spin_length = float(spin)
            p.tolerance = tolerance
            p.num_samples = num_samples
            p.temp_min = T_min
            p.temp_max = T_max
            p.num_temp_bins = num_T
            # Decide output dir up-front. TPQ family always needs a real
            # scratch dir for the HDF5 round-trip; everything else
            # honours whatever the user passed (``""`` -> /dev/null).
            if needs_scratch:
                p.output_dir = _alloc_scratch(n_up_val)
            else:
                p.output_dir = output_dir
            if krylov_dim is not None:
                p.ftlm_krylov_dim = int(krylov_dim)
                p.ltlm_krylov_dim = int(krylov_dim)
            if ftlm_krylov_dim is not None:
                p.ftlm_krylov_dim = int(ftlm_krylov_dim)
            if ltlm_krylov_dim is not None:
                p.ltlm_krylov_dim = int(ltlm_krylov_dim)
                p.ltlm_ground_krylov = int(ltlm_krylov_dim)
            if random_seed:
                p.ftlm_seed = int(random_seed)
                p.ltlm_seed = int(random_seed)
            if n_up_val is not None:
                p.use_fixed_sz = True
                p.n_up = int(n_up_val)
            # TPQ doesn't factor cleanly through spatial irreps (single
            # random state), so we silently disable symmetry for TPQ
            # even when the directory has it. The Sz axis is still used.
            if method_enum in _TPQ_METHODS:
                p.use_symmetry = False
            else:
                p.use_symmetry = bool(has_sym)
            for k, v in merged_extra.items():
                setattr(p, k, v)
            return p

        if not sz_conserved:
            if verbose:
                print(
                    f"[qed.thermal] Sz not conserved; single full-Hilbert "
                    f"{method} call via the directory dispatcher."
                )
            try:
                res = exact_diagonalization_from_directory(
                    directory, method_enum, _make_dir_params(None))
                thermo = _sector_thermo_arrays(res)
                if thermo is None:
                    raise RuntimeError(
                        "qed.thermal: dispatcher returned no thermodynamic "
                        "data."
                    )
                temps, E, Cv, S, F = thermo
                gs_E = (float(res.eigenvalues[0])
                        if len(res.eigenvalues) > 0 else math.inf)
            finally:
                _cleanup_scratch()
            return ThermalResult(
                temperatures=temps, energy=E, specific_heat=Cv,
                entropy=S, free_energy=F, method=str(method),
                ground_state_energy=gs_E,
                used_sz_decomposition=False,
                # Spatial symmetry is suppressed for TPQ; report
                # accurately so callers can spot the silent fallback.
                used_symmetry_decomposition=(
                    has_sym and method_enum not in _TPQ_METHODS),
            )

        if verbose:
            print(
                f"[qed.thermal] iterating n_up in [{lo}, {hi}] of "
                f"[0, {N}]; per-sector symmetry={has_sym}"
            )

        try:
            for n_up in range(lo, hi + 1):
                sec_dim = math.comb(N, n_up)
                if sec_dim == 0:
                    continue
                if verbose:
                    print(f"  [n_up={n_up}] dim={sec_dim}")
                res = exact_diagonalization_from_directory(
                    directory, method_enum, _make_dir_params(n_up))
                thermo = _sector_thermo_arrays(res)
                if thermo is None:
                    if verbose:
                        print("    (no thermo data; skipping)")
                    continue
                temps, E, Cv, S, F = thermo
                per_sector_blocks.append((temps, E, Cv, S, F))
                per_sector_records.append(
                    ThermalSectorEntry(
                        n_up=n_up, sector_dim=sec_dim,
                        temperatures=temps, energy=E,
                        specific_heat=Cv, entropy=S, free_energy=F,
                    )
                )
                if len(res.eigenvalues) > 0:
                    gs_E = min(gs_E, float(res.eigenvalues[0]))

            if not per_sector_blocks:
                raise RuntimeError(
                    "qed.thermal: no Sz sector in the requested window "
                    "produced thermodynamic data."
                )
            temps, E, Cv, S, F = _combine_sector_thermodynamics(
                per_sector_blocks)
        finally:
            _cleanup_scratch()
        if verbose:
            sym_used = has_sym and method_enum not in _TPQ_METHODS
            print(
                f"[qed.thermal] Z-recombined {len(per_sector_blocks)} "
                f"Sz sectors"
                + (" (each sector irrep-recombined internally)"
                   if sym_used else "")
            )
        return ThermalResult(
            temperatures=temps, energy=E, specific_heat=Cv, entropy=S,
            free_energy=F, method=str(method),
            ground_state_energy=gs_E,
            used_sz_decomposition=True,
            used_symmetry_decomposition=(
                has_sym and method_enum not in _TPQ_METHODS),
            per_sector=per_sector_records,
        )

    # ------------------------------------------------------------------
    # 3b. In-memory form -- iterate via qed.diag(H, sz=n_up).
    # ------------------------------------------------------------------
    def _make_diag_kwargs(n_up_val: Optional[int]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "solver": method_enum,
            "num_eigenvalues": 1,
            "tolerance": tolerance,
            "num_samples": num_samples,
            "num_temp_points": num_T,
            "temp_min": T_min,
            "temp_max": T_max,
            "output_dir": _alloc_scratch(n_up_val) if needs_scratch else output_dir,
            "auto_tune": auto_tune,
            "level": level,
            "verbose": False,
            "plan": False,
        }
        # Build the per-sector extra_params bag.
        sector_extra: dict[str, Any] = dict(merged_extra)
        # The `krylov_dim` argument is a legacy alias: for FTLM it maps to
        # `ftlm_krylov_dim`, for LTLM to `ltlm_krylov_dim` / ground_krylov.
        if krylov_dim is not None:
            if method_enum == DiagonalizationMethod.FTLM:
                sector_extra.setdefault("ftlm_krylov_dim", int(krylov_dim))
            elif method_enum == DiagonalizationMethod.LTLM:
                sector_extra.setdefault("ltlm_krylov_dim", int(krylov_dim))
                sector_extra.setdefault("ltlm_ground_krylov", int(krylov_dim))
        if ftlm_krylov_dim is not None:
            sector_extra["ftlm_krylov_dim"] = int(ftlm_krylov_dim)
        if ltlm_krylov_dim is not None:
            sector_extra["ltlm_krylov_dim"] = int(ltlm_krylov_dim)
            sector_extra["ltlm_ground_krylov"] = int(ltlm_krylov_dim)
        if random_seed:
            sector_extra.setdefault("ftlm_seed", int(random_seed))
            sector_extra.setdefault("ltlm_seed", int(random_seed))
        kwargs["extra_params"] = sector_extra
        return kwargs

    if not sz_conserved:
        if verbose:
            print(
                f"[qed.thermal] Sz not conserved (or disabled); single "
                f"full-Hilbert {method} call."
            )
        try:
            res = diag(H_op, **_make_diag_kwargs(None))
            thermo = _sector_thermo_arrays(res)
            if thermo is None:
                raise RuntimeError(
                    "qed.thermal: solver returned no thermodynamic data."
                )
            temps, E, Cv, S, F = thermo
            gs_E = (float(res.eigenvalues[0])
                    if len(res.eigenvalues) > 0 else math.inf)
        finally:
            _cleanup_scratch()
        return ThermalResult(
            temperatures=temps, energy=E, specific_heat=Cv, entropy=S,
            free_energy=F, method=str(method),
            ground_state_energy=gs_E,
            used_sz_decomposition=False,
            used_symmetry_decomposition=False,
        )

    if verbose:
        print(
            f"[qed.thermal] iterating n_up in [{lo}, {hi}] of [0, {N}]"
        )

    per_sector_blocks = []
    per_sector_records = []
    gs_E = math.inf

    try:
        for n_up in range(lo, hi + 1):
            sec_dim = math.comb(N, n_up)
            if sec_dim == 0:
                continue
            if verbose:
                print(f"  [n_up={n_up}] dim={sec_dim}")

            res = diag(H_op, sz=n_up, **_make_diag_kwargs(n_up))
            thermo = _sector_thermo_arrays(res)
            if thermo is None:
                if verbose:
                    print("    (no thermo data; skipping)")
                continue
            temps, E, Cv, S, F = thermo
            per_sector_blocks.append((temps, E, Cv, S, F))
            per_sector_records.append(
                ThermalSectorEntry(
                    n_up=n_up, sector_dim=sec_dim,
                    temperatures=temps, energy=E,
                    specific_heat=Cv, entropy=S, free_energy=F,
                )
            )
            if len(res.eigenvalues) > 0:
                gs_E = min(gs_E, float(res.eigenvalues[0]))

        if not per_sector_blocks:
            raise RuntimeError(
                "qed.thermal: no Sz sector in the requested window "
                "produced thermodynamic data."
            )

        temps, E, Cv, S, F = _combine_sector_thermodynamics(per_sector_blocks)
    finally:
        _cleanup_scratch()
    if verbose:
        print(
            f"[qed.thermal] Z-recombined {len(per_sector_blocks)} "
            f"Sz sectors"
        )
    return ThermalResult(
        temperatures=temps, energy=E, specific_heat=Cv, entropy=S,
        free_energy=F, method=str(method),
        ground_state_energy=gs_E,
        used_sz_decomposition=True,
        used_symmetry_decomposition=False,
        per_sector=per_sector_records,
    )
