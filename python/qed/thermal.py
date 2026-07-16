"""qed.thermal -- one canonical call for finite-T diagonalization.

This module is the Python mirror of the C++ entry point
``ed::workflows::thermal(...)`` declared in
``QED/include/ed/orchestrator.h``. The motivation, as raised by the
matvec-unification audit, was: *"I just want one unified way that
automatically gives me the proper thermodynamics, fully optimised with
the Sz and symmetry sectors. Essentially the user just calls FTLM and
we get the results, but the internal steps are all automatically
optimised and using all possible symmetries."*

Usage::

    import qed
    res = qed.thermal(H, method="FTLM", T_min=0.05, T_max=5.0, num_T=64)
    res.energy               # full-Hilbert <H>(T)
    res.specific_heat        # full-Hilbert C_v(T)
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
    3.  For each sector, dispatch through :func:`qed.solve` with
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

import concurrent.futures
import math
import os
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Union

import numpy as np

from . import _core
from ._core import (
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    Operator,
    ThermodynamicData,
)
from .workflow import solve as _solve  # per-sector dispatcher
from .workflow import _resolve_device   # (use_gpu, use_mpi) device picker
from .workflow import (                 # in-memory symmetry -> temp directory
    _write_operator_directory as _write_operator_directory,
    _write_symmetry_directory as _write_symmetry_directory,
    _normalize_symmetry_info as _normalize_symmetry_info,
    resolve_auto_symmetry as _resolve_auto_symmetry,
    SymmetryArg as SymmetryArg,
)

__all__ = ["ThermalResult", "ThermalSectorEntry", "thermal"]




# Stage 11a: converters single-sourced in qed._params (this module's
# copy had silently diverged from workflow.py's -- see _params.py).
from ._params import (  # noqa: E402,F401
    THERMAL_METHOD_MAP as _THERMAL_METHOD_MAP,
    ed_params_to_thermal_options as _ed_params_to_thermal_options,
    ed_result_from_thermal_result as _ed_result_from_thermal_result,
)




def _sym_toggle_int(value, name: str, operator=None, verbose=True) -> int:
    """Map a spin_flip / time_reversal kwarg to the C++ toggle int:
    -1 auto / 0 off / 1 require, with 'on' = auto + detection report.
    Delegates to :func:`qed.workflow.resolve_discrete_toggle`."""
    from .workflow import resolve_discrete_toggle
    return resolve_discrete_toggle(operator, value, name, verbose=verbose)


def _thermal_via_workflows_all_sz_streaming_symmetry(
    directory: str,
    num_sites: int,
    spin_l: float,
    method: DiagonalizationMethod,
    params: EDParameters,
    n_up_min: int,
    n_up_max: int,
    *,
    use_gpu: bool = False,
    use_mpi: bool = False,
    spin_flip: int = -1,
    time_reversal: int = -1,
    star_maps: Optional[list] = None,
) -> "_core.ThermalResult":
    """Single C++ call covering ALL (n_up, irrep) sectors simultaneously.

    Calls ``workflows_thermal_all_sz_streaming_symmetry_directory``, which:
      1. Loads Hamiltonian + symmetry group info once.
      2. Runs ``enumerate_full_orbit_reps`` once (O(2^N × |G|)).
      3. Partitions reps by n_up and builds all (n_up, irrep) sectors.
      4. Runs a single flat OMP pool over all sectors
         (gate: ``ED_SYM_SECTOR_PARALLEL=1``).
      5. Returns a single ``ThermalResult`` with fully-combined thermo.

    Eliminates the N+1 cold-start overhead (JSON loads + orbit rep scans)
    from calling ``workflows_thermal_streaming_symmetry_directory`` once
    per n_up from a Python ThreadPoolExecutor.
    """
    opts = _ed_params_to_thermal_options(params, method)
    opts.backend.allow_gpu = bool(use_gpu)
    opts.backend.allow_mpi = bool(use_mpi)
    if use_gpu:
        opts.backend.gpu_dim_floor = 0
    opts.spin_flip     = int(spin_flip)      # Stage 8 composition toggles
    opts.time_reversal = int(time_reversal)
    if star_maps:
        opts.star_maps = [[int(x) for x in m] for m in star_maps]
    return _core.workflows_thermal_all_sz_streaming_symmetry_directory(
        directory, int(num_sites), float(spin_l), opts,
        int(n_up_min), int(n_up_max))


def _can_use_workflows_thermal(
    method: DiagonalizationMethod,
    has_symmetry: bool,
) -> bool:
    """True when `_core.workflows_thermal` (the single-operator
    orchestrator entry) covers the requested lane.

    Post-collapse SOTA upgrade (May 2026): when ``has_symmetry`` is
    True we now route through
    ``_core.workflows_thermal_streaming_symmetry_directory`` instead
    (which composes the orchestrator's single-operator entry with a
    per-irrep sector loop and recombines via
    ``ed::core::combine_sector_thermodynamics``). So
    ``_can_use_workflows_thermal`` returns True iff the method is
    bound, regardless of ``has_symmetry``; callers branch on
    ``has_symmetry`` to pick the right binding."""
    return method in _THERMAL_METHOD_MAP


def _thermal_via_workflows_thermal(
    H_op: Operator,
    method: DiagonalizationMethod,
    params: EDParameters,
    *,
    use_gpu: bool = False,
    use_mpi: bool = False,
) -> EDResults:
    """Route a (possibly fixed-Sz) operator through
    `_core.workflows_thermal` and reshape the result to `EDResults`.

    ``use_gpu`` / ``use_mpi`` flips the matching ``BackendConstraints``
    bits on the ``ThermalOptions`` so the C++ ``select_backend`` picks
    ``CudaBackend`` / ``MpiBackend`` for the matvec inside the thermal
    kernel."""
    if params.use_fixed_sz:
        op = H_op.make_fixed_sz(int(params.n_up))
    else:
        op = H_op
    opts = _ed_params_to_thermal_options(params, method)
    opts.backend.allow_gpu = bool(use_gpu)
    opts.backend.allow_mpi = bool(use_mpi)
    if use_gpu:
        opts.backend.gpu_dim_floor = 0
    tr = _core.workflows_thermal(op, opts)
    return _ed_result_from_thermal_result(tr)



def _sector_table_from_directory(directory: str) -> Optional[dict]:
    """The (sector_id, quantum_numbers) table a symmetry directory carries.

    ``sector_metadata.json`` is written by ``_write_symmetry_directory`` and is
    the same table the C++ side reads, so resolving a caller's quantum numbers
    against it is a lookup rather than a guess about index conventions.
    """
    import json as _json
    path = os.path.join(directory, "automorphism_results",
                        "sector_metadata.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return {"sectors": _json.load(f).get("sectors", [])}
    except Exception:
        return None

def _thermal_via_workflows_streaming_symmetry(
    directory: str,
    num_sites: int,
    spin_l: float,
    method: DiagonalizationMethod,
    params: EDParameters,
    *,
    use_gpu: bool = False,
    use_mpi: bool = False,
) -> EDResults:
    """Route a directory + ``automorphism_results/`` through the
    SOTA C++ streaming-symmetry thermal binding
    (``_core.workflows_thermal_streaming_symmetry_directory``).

    Mirrors :func:`_thermal_via_workflows_thermal` but lets the C++
    side own the per-irrep sector loop and the Z-recombination via
    ``ed::core::combine_sector_thermodynamics``. The returned
    ``EDResults.thermo_data`` is the full-Hilbert recombined thermo
    on the requested temperature grid.

    Phase C of the "Backend x Symmetries x Workflows" plan
    (May 2026): honours the caller's ``use_gpu`` flag by flipping
    ``opts.backend.allow_gpu`` on the streaming-symmetry binding
    too. The C++ side then routes the per-sector matvec through the
    lazy GPU mirror that Phase A wired up."""
    opts = _ed_params_to_thermal_options(params, method)
    opts.backend.allow_gpu = bool(use_gpu)
    opts.backend.allow_mpi = bool(use_mpi)
    if use_gpu:
        opts.backend.gpu_dim_floor = 0
    fixed_sz = int(params.n_up) if params.use_fixed_sz else None
    tr = _core.workflows_thermal_streaming_symmetry_directory(
        directory,
        int(num_sites),
        float(spin_l),
        opts,
        fixed_sz,
    )
    return _ed_result_from_thermal_result(tr)


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
    # U1b: how many times this block's spectrum enters the recombined Z
    # (star size x d_sigma x TR fold x Sz mirror). 1 on the abelian lanes,
    # whose loops enumerate every sector explicitly.
    weight: int = 1
    # Unified block labels (BlockTag work item, 2026-07-16): which block
    # this entry is, in the same vocabulary the solve verb's project lane
    # uses. k_raw is an ENGINE irrep index, NOT the momentum -- decode via
    # the abelian character table (chi_k of the translation generator),
    # exactly like decode_star_for_sector. -1 = axis absent / abelian lane
    # (whose entries predate the labels).
    k_raw: int = -1
    flip_parity: int = -1
    irrep: int = -1
    irrep_dim: int = 1
    star_size: int = 1


def _thermal_result_from_block_lane(out: dict, method) -> "ThermalResult":
    """Assemble a ThermalResult from _core.little_group_thermal output.
    Shared by the in-memory (U1b) and directory (U4a) block-lane routes.
    The curves are the COMBINED deliverable; per-block curves stay
    engine-side until a consumer needs them -- the per_sector entries
    carry the structural contract (n_up, block dim, recombination
    weight)."""
    per = [ThermalSectorEntry(
               n_up=int(out["block_n_up"][i]),
               sector_dim=int(out["block_dim"][i]),
               temperatures=np.asarray(out["temperatures"], dtype=float),
               energy=np.asarray([], dtype=float),
               specific_heat=np.asarray([], dtype=float),
               entropy=np.asarray([], dtype=float),
               free_energy=np.asarray([], dtype=float),
               weight=int(out["block_weight"][i]),
               k_raw=int(out["block_k_raw"][i]),
               flip_parity=int(out["block_flip_parity"][i]),
               irrep=int(out["block_irrep"][i]),
               irrep_dim=int(out["block_irrep_dim"][i]),
               star_size=int(out["block_star_size"][i]))
           for i in range(len(out["block_dim"]))]
    _E = np.asarray(out["energy"], dtype=float)
    return ThermalResult(
        temperatures=np.asarray(out["temperatures"], dtype=float),
        energy=_E,
        specific_heat=np.asarray(out["specific_heat"], dtype=float),
        entropy=np.asarray(out["entropy"], dtype=float),
        free_energy=np.asarray(out["free_energy"], dtype=float),
        method=str(method),
        ground_state_energy=float(out["ground_state_energy"]),
        used_sz_decomposition=True,
        used_symmetry_decomposition=True,
        per_sector=per,
    )


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
    # Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026): path to
    # the HDF5 file produced by the thermal orchestrator (when
    # ``output_dir`` was set). For mTPQ runs with ``probe_betas``
    # set, the file carries per-sample trajectories under
    # ``/tpq/samples/sample_<s>/thermodynamics`` and state vectors
    # under ``/tpq/samples/sample_<s>/states/beta_<b>``.
    #
    # Multi-sector mode (Save & DSSF follow-up, May 2026): when
    # ``qed.thermal`` iterates more than one Sz sector for a TPQ
    # method, each sector lands in its own subdirectory
    # ``<output_dir>/n_up_<n_up>/ed_results.h5`` to avoid the
    # ``/tpq/samples/sample_<s>/states/beta_<b>`` namespace collision
    # (the dataset names are NOT sector-tagged). In that case
    # :attr:`hdf5_path` points to the parent ``<output_dir>`` and
    # :attr:`sector_hdf5_paths` carries the per-sector ``(n_up, path)``
    # mapping so callers can reload state vectors per sector.
    hdf5_path: str = ""
    sector_hdf5_paths: dict[Optional[int], str] = field(default_factory=dict)
    # KPM_DOS raw density of states (Jul 2026): populated only by the
    # KPM_DOS method (full-Hilbert lane); empty for every other method.
    dos_energies: np.ndarray = field(default_factory=lambda: np.array([]))
    dos_values: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_THERMAL_METHODS = {
    "FTLM": DiagonalizationMethod.FTLM,
    "OFTLM": DiagonalizationMethod.OFTLM,
    "LTLM": DiagonalizationMethod.LTLM,
    "KPM_DOS": DiagonalizationMethod.KPM_DOS,
    "MTPQ": DiagonalizationMethod.mTPQ,
}

# Methods that write per-sample HDF5 trajectories on disk and need a
# real (non-/dev/null) output directory so the recombination step can
# read them back. Mirrors the C++ ``detail::method_needs_output_dir``.
_TPQ_METHODS = {
    DiagonalizationMethod.mTPQ,
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


def _persistent_sector_outdir(output_dir: str, n_up: Optional[int]) -> str:
    """Return a deterministic per-Sz-sector subdirectory under
    ``output_dir`` for TPQ runs.

    Save & DSSF Upgrades follow-up (May 2026): when the user passes an
    explicit ``output_dir`` AND ``qed.thermal`` iterates multiple Sz
    sectors, each sector's ``ed::workflows::thermal`` call would otherwise
    write to the SAME ``<output_dir>/ed_results.h5`` and overwrite the
    previous sector's TPQ samples / state vectors. The
    ``/tpq/samples/sample_<s>/states/beta_<b>`` namespace is keyed only
    by ``effective_beta`` -- there is no sector tag in the dataset name,
    so collisions silently destroy probe-beta snapshots. We route each
    sector to its own subdirectory; callers can rebuild the per-sector
    file path as ``<output_dir>/n_up_<n_up>/ed_results.h5``.
    """
    if not output_dir:
        return ""
    if n_up is None:
        return output_dir
    return os.path.join(output_dir, f"n_up_{int(n_up)}")


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
        # mTPQ is case-sensitive in the enum.
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
    sector: Optional[Sequence[int]] = None,
    symmetry: "SymmetryArg" = None,
    num_samples: int = 40,
    krylov_dim: Optional[int] = None,
    ftlm_krylov_dim: Optional[int] = None,
    ltlm_krylov_dim: Optional[int] = None,
    tolerance: float = 1e-10,
    max_iterations: Optional[int] = None,
    random_seed: int = 0,
    use_symmetry_if_available: bool = False,
    use_sz_if_conserved: bool = True,
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    point_group: Union[str, bool, None] = "auto",
    star_maps: Optional[list] = None,
    lattice: Optional[Any] = None,
    output_dir: str = "",
    verbose: bool = True,
    # Phase C of the "Backend x Symmetries x Workflows" plan
    # (May 2026): explicit device selector. ``None`` / ``"auto"`` picks
    # GPU when the Hilbert dim crosses the auto-tuner threshold;
    # ``"cpu"`` / ``"gpu"`` pin the choice. The selected backend flag
    # is threaded into every per-sector ``qed.solve`` (in-memory
    # branch) AND into the ``ThermalOptions.backend`` carried by the
    # directory / streaming-symmetry C++ binding.
    device: Optional[str] = None,
    # ---- KPM_DOS specific ------------------------------------------
    kpm_num_moments: int = 200,
    kpm_num_random_vectors: int = 16,
    # ---- TPQ specific (mTPQ) ---------------------------------------
    tpq_num_measure_points: int = 100,
    # ``None`` -> auto-derive from (T_min, T_max). For mTPQ we
    # need the measurement β grid to bracket every target β = 1/T
    # the recombiner asks for, so pin it to ``1/T_max`` / ``1/T_min``
    # by default and let the user override.
    tpq_measure_beta_min: Optional[float] = None,
    tpq_measure_beta_max: Optional[float] = None,
    tpq_delta_beta: float = 0.05,
    tpq_taylor_order: int = 8,
    tpq_measurement_interval: int = 1,
    # ``0.0`` -> auto-pick via a quick Lanczos spectral-bound estimate
    # inside the orchestrator's TPQ kernel (single source of truth).
    tpq_energy_shift: float = 0.0,
    # Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
    # user-supplied probe-betas at which the mTPQ kernel should
    # snapshot the running state vector and persist it (together with
    # the trajectory) inside ``<output_dir>/ed_results.h5``. Empty
    # (default) -> no state vectors saved. Combine with
    # ``output_dir="..."`` to land the snapshots on disk; reload via
    # h5py at ``/tpq/samples/sample_<s>/state_beta_<b>`` for the
    # TPQ-to-CF spectral pipeline.
    probe_betas: Optional[List[float]] = None,
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
        ``"KPM_DOS"``, ``"mTPQ"``.
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
    use_symmetry_if_available : bool, optional
        Opt-in spatial-symmetry detection for the directory form
        (reads ``automorphism_results/`` if present). **Default False**
        as of the May-2026 surface unification; pass
        ``use_symmetry_if_available=True`` to restore the old
        directory-form auto-detection.
    use_sz_if_conserved : bool, optional
        Force-off toggle for the Sz auto-iteration. Default True.
    _unused_doc : optional
        (auto-tuner removed; thermal knobs use struct defaults). ``level`` was one of ``"conservative"``,
        ``"balanced"``, ``"aggressive"``.
    output_dir : str, optional
        Where to write per-sector HDF5 sinks; default ``""`` means
        no I/O.
    verbose : bool, optional
        Print per-sector progress when True (default).
    num_sites, spin : optional
        Required for the directory form.
    extra_params : dict, optional
        Forwarded to :func:`qed.solve` per sector. Use for any niche
        ``EDParameters`` field this helper doesn't expose.
    probe_betas : list of float, optional
        Inverse temperatures at which the mTPQ kernel should
        snapshot (copy to host) the running TPQ state vector. Combined
        with ``output_dir``, this lands each snapshot in
        ``<output_dir>/ed_results.h5`` at
        ``/tpq/samples/sample_<s>/state_beta_<b>`` for downstream
        reload (e.g. by :func:`qed.spectral(method="GroundStateCF",
        initial_state=...)`). Empty / ``None`` (default) -> the
        kernel skips state-vector copies and only the trajectory is
        persisted. Ignored by FTLM / LTLM / KPM_DOS.

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
    _all_params = dict(locals())   # capture every parameter for the symmetry recursion
    is_directory = isinstance(H, (str, os.PathLike))

    # ------------------------------------------------------------------
    # method="exact": exact canonical thermodynamics from the block
    # engine's full per-block spectra. This is a METHOD, sitting beside
    # FTLM/LTLM/mTPQ -- point_group stays a pure symmetry-routing knob
    # ('auto' project-when-possible / 'full' require / 'off' abelian).
    # Historically this computation was reachable only through the
    # point_group='full' spelling, which conflated routing with solver
    # strategy; that spelling now warns (see below) and requires
    # projection without changing the method.
    # ------------------------------------------------------------------
    if isinstance(method, str) and method.upper() == "EXACT":
        if is_directory:
            # U4a pattern: the directory's own deck + automorphisms feed
            # the exact block engine. Same guards as the sampling route:
            # the Python loader reads only Trans/InterAll (refuse
            # ThreeBodyG.dat), and the group comes from
            # automorphisms.json (validated permutations).
            directory = str(H)
            if num_sites is None:
                raise ValueError(
                    "qed.thermal: pass num_sites= with the directory "
                    "form.")
            if os.path.exists(os.path.join(directory, "ThreeBodyG.dat")):
                raise NotImplementedError(
                    "qed.thermal(method='exact'): this directory carries "
                    "ThreeBodyG.dat, which the Python-side loader does "
                    "not read -- the exact lane would silently miss "
                    "terms. Use the sampling methods (exact below the "
                    "small-dim cutoff) or the in-memory form.")
            _autos_path = os.path.join(directory, "automorphism_results",
                                       "automorphisms.json")
            if not os.path.exists(_autos_path):
                raise ValueError(
                    "qed.thermal(method='exact'): the directory carries "
                    "no automorphism_results/automorphisms.json to build "
                    "the block engine from; pass the in-memory form with "
                    "symmetry= instead.")
            import json as _json
            with open(_autos_path) as f:
                _cand = _json.load(f)
            _N = int(num_sites)
            if not (isinstance(_cand, list) and _cand
                    and all(isinstance(p, list) and len(p) == _N
                            and sorted(p) == list(range(_N))
                            for p in _cand)):
                raise ValueError(
                    "qed.thermal(method='exact'): automorphisms.json is "
                    "not a list of site permutations.")
            H_ex = Operator(num_sites=_N, spin=float(spin))
            _trans = os.path.join(directory, "Trans.dat")
            _inter = os.path.join(directory, "InterAll.dat")
            if os.path.exists(_trans):
                H_ex.load_trans(_trans)
            if os.path.exists(_inter):
                H_ex.load_inter_all(_inter)
            H, symmetry, is_directory = H_ex, _cand, False
        if symmetry is None:
            raise NotImplementedError(
                "qed.thermal: method='exact' rides the little-group block "
                "engine and needs symmetry=. (Without symmetry, the "
                "sampling methods are already exact below dim 512 via the "
                "small-dim fallback.)")
        if sector is not None:
            raise NotImplementedError(
                "qed.thermal: sector= is not honoured on the exact lane "
                "yet -- refusing to silently ignore it.")
        from .point_group_routing import split_nonabelian, _close
        _split = split_nonabelian(symmetry)
        if isinstance(_split, str):
            # No non-abelian residue (or unsplittable): exact per plain
            # (n_up, k) block -- close the abelian generators, no
            # residues. Correct, merely less reduced.
            _gens = [list(g) for g in symmetry.generators] \
                if hasattr(symmetry, "generators") else \
                [list(g) for g in symmetry]
            _A = _close(_gens)
            if _A is None:
                raise ValueError(
                    f"qed.thermal(method='exact'): could not close the "
                    f"symmetry group ({_split})")
            _A, _res = [list(g) for g in _A], []
        else:
            _A, _res = _split
        temps = list(np.linspace(T_min, T_max, num_T))
        td = dict(_core.little_group_thermodynamics(
            H, _A, _res, temps, n_up=-1,
            use_gpu=(isinstance(device, str)
                     and device.lower() in ("gpu", "cuda")),
            spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
            time_reversal=_sym_toggle_int(time_reversal,
                                          "time_reversal")))
        if verbose:
            print(f"[qed.thermal] EXACT little-group lane: |A| = "
                  f"{len(_A)}, residues = {len(_res)}.")
        _E = np.asarray(td["energy"], dtype=float)
        return ThermalResult(
            temperatures=np.asarray(td["temperatures"], dtype=float),
            energy=_E,
            specific_heat=np.asarray(td["specific_heat"], dtype=float),
            entropy=np.asarray(td["entropy"], dtype=float),
            free_energy=np.asarray(td["free_energy"], dtype=float),
            method="exact",
            ground_state_energy=float(_E[0]) if len(_E) else 0.0,
            used_sz_decomposition=False,
            used_symmetry_decomposition=True,
        )

    method_enum = _coerce_method(method)

    # sector= names a SPATIAL-symmetry irrep, so it is meaningless without a
    # spatial group. Checked here, before any branch: the in-memory no-symmetry
    # path never reaches the directory branch that resolves sector=, so an
    # argument passed there would otherwise be silently ignored -- the same
    # failure mode this parameter exists to fix.
    if (sector is not None and symmetry is None and not is_directory):
        raise ValueError(
            "qed.thermal: sector= names a spatial-symmetry irrep, but no "
            "symmetry= was given (and an in-memory operator has no "
            "automorphism_results/ to auto-load one from). Pass symmetry= "
            "(e.g. qed.find_symmetries(H).full_set), or use sz_min/sz_max for "
            "magnetisation sectors.")

    # In-memory operator + explicit spatial `symmetry=`: materialise a temp
    # directory (operator + automorphism_results/) and re-dispatch as the
    # directory form, which runs the per-(Sz, irrep) stochastic lanes and
    # Z-recombines. This makes the spatial-symmetry sectors available to the
    # in-memory API. (A non-abelian generator set is reduced by its maximal
    # abelian subgroup here -- a complete, correct, coarser reduction; for the
    # FULL non-abelian reduction route through the little-group engine
    # (point_group="full").)
    if not is_directory:
        # symmetry='auto' -> maximal spatial generator set (or None);
        # 'on' toggles -> detection-checked ints (report + degrade).
        symmetry = _resolve_auto_symmetry(H, symmetry, verbose=verbose,
                                          lattice=lattice)
        spin_flip = _sym_toggle_int(spin_flip, "spin_flip", H, verbose)
        time_reversal = _sym_toggle_int(
            time_reversal, "time_reversal", H, verbose)
        # Stage 7a: star reduction -- non-abelian residue folds the
        # irrep sectors into isospectral orbits (solve one per star).
        _star = getattr(symmetry, "star_perms", None) or []
        if (star_maps is None and _star
                and point_group not in (False, 0, "off", "none")):
            _info_s = _normalize_symmetry_info(H, symmetry)
            if _info_s is not None:
                from .star_reduction import star_maps_from_info
                star_maps = star_maps_from_info(_info_s, _star) or None
                if star_maps and verbose:
                    print(f"[qed] point group: {len(star_maps)} residue "
                          "automorphisms fold the irrep sectors into "
                          "isospectral stars (solve one per star).")
    if (isinstance(point_group, str) and point_group.lower() == "full"):
        # SEMANTIC CLEANUP (U4, user-requested): point_group is a pure
        # symmetry-ROUTING knob. 'full' means REQUIRE projection (raise
        # on decline); it no longer implies exact per-block spectra --
        # that solver strategy is method="exact" now. Warn the old
        # spelling's users: with a sampling method their run is now a
        # sampled one inside the blocks (identical below the small-dim
        # exact cutoff, stochastic above it).
        import warnings as _warnings
        _warnings.warn(
            "qed.thermal(point_group='full') no longer implies exact "
            "per-block thermodynamics -- it now purely REQUIRES the "
            "little-group projection for whatever method= is set. For "
            "the old exact behaviour pass method='exact'.",
            FutureWarning, stacklevel=2)
    if (symmetry is not None and not is_directory
            and sector is None
            and isinstance(point_group, str)
            and point_group.lower() in ("auto", "full")):
        # U1b (lane unification): thermal 'auto' + a sampling method now
        # PROJECTS -- the run stays a sampling run, executed inside the
        # (n_up, k, +/-, sigma) little-group blocks via
        # _core.little_group_thermal (F-shift Z-recombination). The lane
        # resolver declines KPM_DOS (full-spectrum DOS) and honours
        # ED_SYM_LG_THERMAL=0; any decline falls through to the abelian
        # tempdir lane below unchanged. sector= keeps the abelian
        # filtering lane (the block engine has only_k0/only_irrep but the
        # QN decode for thermal is future work -- refusing to guess).
        from .point_group_routing import resolve_projection_lane
        lane = resolve_projection_lane(
            symmetry, point_group=point_group.lower(), consumer="thermal",
            eigenvalues_only=True, method=str(method),
            verbose=verbose)
        if lane.mode == "project":
            out = dict(_core.little_group_thermal(
                H, lane.A, lane.residues, method=str(method),
                t_min=float(T_min), t_max=float(T_max), num_t=int(num_T),
                num_samples=int(num_samples),
                krylov_dim=int(krylov_dim) if krylov_dim else 100,
                random_seed=int(random_seed) if random_seed else 0,
                use_gpu=(isinstance(device, str)
                         and device.lower() in ("gpu", "cuda")),
                spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
                time_reversal=_sym_toggle_int(time_reversal,
                                              "time_reversal")))
            if verbose:
                print(f"[qed.thermal] little-group SAMPLING lane "
                      f"(U1b): {len(out['block_dim'])} blocks, "
                      f"projected_any={out['projected_any']}, "
                      f"max block dim={max(out['block_dim'])}.")
            return _thermal_result_from_block_lane(out, method)
        elif verbose:
            print(f"[qed.thermal] projection declined ({lane.reason}); "
                  f"abelian sector lane.")
    if symmetry is not None and not is_directory:
        _N = int(H.num_sites)
        _tmp = tempfile.mkdtemp(prefix="qed_thermal_sym_")
        try:
            _write_operator_directory(H, _tmp)
            _info = _normalize_symmetry_info(H, symmetry)
            if _info is not None:
                _write_symmetry_directory(_tmp, _info)
            _fwd = {k: v for k, v in _all_params.items() if k != "H"}
            _fwd["num_sites"] = _N
            _fwd["use_symmetry_if_available"] = True
            _fwd["symmetry"] = None
            _fwd["spin_flip"] = spin_flip          # already resolved ints
            _fwd["time_reversal"] = time_reversal
            _fwd["star_maps"] = star_maps
            return thermal(_tmp, **_fwd)
        finally:
            shutil.rmtree(_tmp, ignore_errors=True)

    if is_directory and num_sites is None:
        raise ValueError(
            "qed.thermal: pass num_sites=... when using the "
            "directory form."
        )

    # Phase C of the "Backend x Symmetries x Workflows" plan
    # (May 2026): resolve the device once at the top. We pass the
    # whole-Hilbert dim as the "size hint" so ``device='auto'`` picks
    # the GPU only when the dim crosses the auto-tuner threshold.
    # Per-sector calls inherit the choice (the sector matvec is
    # bounded by the whole-Hilbert dim anyway, so a single decision
    # is the right grain).
    if is_directory:
        _dim_hint = 1 << int(num_sites)
    else:
        _dim_hint = 1 << int(H.num_sites)
    _use_gpu, _use_mpi = _resolve_device(device, _dim_hint)
    if _use_mpi:
        # qed.thermal does not have an MPI route today -- the C++
        # thermal binding is rank-local. We surface a clear error
        # rather than silently dropping the MPI request.
        raise NotImplementedError(
            "qed.thermal(device={!r}): MPI thermal is not wired yet "
            "(use the standalone ed_distributed_main binary). Pass "
            "device='cpu' or device='gpu'.".format(device)
        )

    # ------------------------------------------------------------------
    # Method-specific knob plumbing. Anything that touches an
    # EDParameters field but isn't a generic FTLM knob gets injected
    # here and merged with the user's own `extra_params`. The merge
    # order is: defaults < method-specific < explicit `extra_params`.
    # ------------------------------------------------------------------
    # ``max_iterations=None`` (default) is the clean "let the solver
    # decide" sentinel: for mTPQ the orchestrator auto-sizes the step
    # count from the spectral bounds so the requested coldest T is
    # reached. ``0`` is the wire encoding of that sentinel; any positive
    # value is respected exactly downstream.
    method_extra: dict[str, Any] = {
        "max_iterations": int(max_iterations) if max_iterations is not None else 0,
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
            # Mirror the iteration budget into ``tpq_max_steps`` so the
            # sentinel is a single source of truth across every dispatch
            # path (in-memory, per-sector qed.solve, streaming directory).
            # The EDParameters default (10000) would otherwise mask the
            # ``max_iterations=None`` auto request on paths that read
            # ``tpq_max_steps`` directly. ``0`` => auto-size (mTPQ).
            tpq_max_steps=int(max_iterations) if max_iterations is not None else 0,
        )
        # Pillar 1: probe-beta list for state-vector snapshots.
        # Empty/None => no snapshots; the orchestrator persists only
        # the trajectory in HDF5.
        if probe_betas:
            method_extra["tpq_probe_betas"] = [float(b) for b in probe_betas]

    merged_extra: dict[str, Any] = {**method_extra}
    if extra_params:
        merged_extra.update(extra_params)

    # Save & DSSF Upgrades follow-up (May 2026): when the user requested
    # state-vector snapshots via ``probe_betas`` but did NOT pass an
    # ``output_dir``, the helper used to allocate scratch directories
    # and then ``shutil.rmtree`` them in the ``finally`` block --
    # silently destroying every state vector the user asked us to save
    # AND surfacing a dangling ``hdf5_path``. Require an explicit
    # ``output_dir`` whenever ``probe_betas`` is non-empty so the
    # snapshots have a stable home on disk.
    if method_enum in _TPQ_METHODS \
            and probe_betas \
            and not output_dir:
        raise ValueError(
            "qed.thermal: probe_betas is set but output_dir is empty. "
            "State-vector snapshots would be written to a scratch "
            "directory and then deleted at the end of this call. "
            "Pass output_dir=... so the snapshots survive."
        )

    # TPQ needs a real (non-/dev/null) output dir per sector so its
    # per-sample HDF5 trajectories survive long enough for
    # ``compute_tpq_unified_thermo`` to read them back. We allocate
    # scratch dirs on-demand and clean them up when the call returns.
    # Optimization (Jul 2026): the unified TPQ kernel returns every
    # trajectory in-memory; the historical per-sector HDF5 scratch
    # round-trip is pure I/O overhead (2-7x wall time at small N).
    # No output_dir => writes disabled entirely. probe_betas snapshots
    # REQUIRE an explicit output_dir (warned below).
    tpq_scratch_dirs: list[str] = []
    needs_scratch = False
    if (method_enum in _TPQ_METHODS and not output_dir
            and probe_betas):
        import warnings as _warnings
        _warnings.warn(
            "probe_betas were requested without output_dir=; TPQ state "
            "snapshots are only persisted to disk, so pass output_dir= "
            "to keep them.", RuntimeWarning, stacklevel=2)

    def _alloc_scratch(n_up: Optional[int]) -> str:
        tag = "mtpq"
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
        # via qed.solve(symmetry=...). When present we load the generator
        # set; qed.solve handles the per-irrep streaming internally.
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
    # KPM_DOS produces a density of states -- a full-SPECTRUM quantity. Sz
    # decomposition would yield per-sector sub-DOS on different Chebyshev
    # grids that the thermodynamic recombination cannot merge into one DOS
    # (the raw density(E) then never reaches the caller). Run it on the full
    # space so ThermalResult.dos_* is the complete DOS; the derived
    # thermodynamics are identical (the DOS is Sz-summed either way).
    if method_enum == DiagonalizationMethod.KPM_DOS:
        sz_conserved = False

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
    # 2b. U4a: the directory form rides the SAME block lane as the
    #     in-memory form. The directory's own automorphisms.json is the
    #     full group; split_nonabelian carves it into (abelian core,
    #     residues) exactly like an explicit generator list. Guards keep
    #     every contract the block lane cannot yet honour on the abelian
    #     loop instead:
    #       * ThreeBodyG.dat: the Python-side loader above reads only
    #         Trans/InterAll, so H_op would be INCOMPLETE -- decline.
    #       * output_dir / probe_betas: per-sector files and TPQ
    #         snapshots are flat-pool deliverables -- decline.
    #       * sector= or a partial Sz window: the filtering loop serves
    #         those -- decline.
    # ------------------------------------------------------------------
    if (is_directory and has_sym and sector is None
            and isinstance(point_group, str)
            and point_group.lower() in ("auto", "full")
            and not output_dir and not probe_betas
            and str(method).upper() in ("FTLM", "LTLM", "MTPQ", "OFTLM")
            and not os.path.exists(os.path.join(directory, "ThreeBodyG.dat"))
            and (not sz_conserved or (lo == 0 and hi == N))):
        _autos_path = os.path.join(sym_dir, "automorphisms.json")
        _autos = None
        if os.path.exists(_autos_path):
            import json as _json
            try:
                with open(_autos_path) as f:
                    _cand = _json.load(f)
                if (isinstance(_cand, list) and _cand
                        and all(isinstance(p, list) and len(p) == N
                                and sorted(p) == list(range(N))
                                for p in _cand)):
                    _autos = _cand
            except Exception:
                _autos = None
        if _autos is not None:
            from .point_group_routing import resolve_projection_lane
            lane = resolve_projection_lane(
                _autos, point_group=point_group.lower(), consumer="thermal",
                eigenvalues_only=True, method=str(method), verbose=verbose)
            if lane.mode == "project":
                out = dict(_core.little_group_thermal(
                    H_op, lane.A, lane.residues, method=str(method),
                    t_min=float(T_min), t_max=float(T_max),
                    num_t=int(num_T), num_samples=int(num_samples),
                    krylov_dim=int(krylov_dim) if krylov_dim else 100,
                    random_seed=int(random_seed) if random_seed else 0,
                    use_gpu=_use_gpu,
                    spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
                    time_reversal=_sym_toggle_int(time_reversal,
                                                  "time_reversal")))
                if verbose:
                    print(f"[qed.thermal] directory -> little-group "
                          f"SAMPLING lane (U4a): "
                          f"{len(out['block_dim'])} blocks, "
                          f"projected_any={out['projected_any']}.")
                return _thermal_result_from_block_lane(out, method)
            elif verbose:
                print(f"[qed.thermal] directory projection declined "
                      f"({lane.reason}); flat-pool sector lane.")

    # ------------------------------------------------------------------
    # 3a. Directory form -- per-Sz dispatch through the C++ dispatcher
    #     which handles spatial symmetry internally when
    #     `params.use_symmetry=True`.
    # ------------------------------------------------------------------
    if is_directory:
        per_sector_blocks: list[tuple[np.ndarray, ...]] = []
        per_sector_records: list[ThermalSectorEntry] = []
        sector_hdf5_paths: dict[Optional[int], str] = {}
        gs_E = math.inf

        # ------------------------------------------------------------------
        # sector= : resolve the caller's QUANTUM NUMBERS to the raw sector
        # index the C++ filter selects on. Done once, up front, so a bad tuple
        # fails before any solving.
        # ------------------------------------------------------------------
        _sector_sid: Optional[int] = None
        if sector is not None:
            if not has_sym:
                raise ValueError(
                    "qed.thermal: sector= names a spatial-symmetry irrep, but "
                    "this run has no spatial symmetry (no automorphism_results/ "
                    "in the directory, or use_symmetry_if_available=False). "
                    "Use sz_min/sz_max for magnetisation sectors.")
            _tbl = _sector_table_from_directory(directory)
            if _tbl is None:
                raise RuntimeError(
                    "qed.thermal: sector= was given but the directory carries "
                    "no automorphism_results/sector_metadata.json to resolve "
                    "the quantum numbers against.")
            from .workflow import _resolve_sector_quantum_numbers
            _sector_sid = _resolve_sector_quantum_numbers(_tbl, sector)
            if verbose:
                print(f"[qed.thermal] sector={list(sector)} -> raw sector "
                      f"index {_sector_sid}")

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
            #
            # Multi-Sz overwrite fix (May 2026): when the user provided
            # an explicit ``output_dir`` AND we are iterating over
            # multiple Sz sectors for a TPQ method, each sector lands
            # in its own subdirectory ``<output_dir>/n_up_<n_up>/`` so
            # the per-sector ``ed_results.h5`` files do not overwrite
            # one another. For non-TPQ methods (FTLM / LTLM / KPM-DOS)
            # the file holds only aggregated curves which the
            # ``averaged/`` group can dedupe in-place; we still route
            # per-sector to be safe and consistent.
            if not output_dir:
                p.output_dir = "/dev/null"      # writes disabled
            elif output_dir and n_up_val is not None:
                p.output_dir = _persistent_sector_outdir(output_dir, n_up_val)
                if not p.output_dir.startswith("/dev/null"):
                    os.makedirs(p.output_dir, exist_ok=True)
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
            # Pipe ``max_iterations`` into ``tpq_max_steps`` for the
            # mTPQ lane (closes a gap where this helper only
            # set the FTLM/LTLM Krylov-dim and left the TPQ iteration
            # budget at the EDParameters default). ``_ed_params_to_thermal_options``
            # reads ``tpq_max_steps`` to populate ``opts.krylov_dim``
            # for mTPQ.
            if max_iterations is not None:
                p.max_iterations = int(max_iterations)
                p.tpq_max_steps  = int(max_iterations)
            else:
                # Auto sentinel: 0 tells the orchestrator's mTPQ lane to
                # size the iteration count from the spectral bounds. The
                # EDParameters default (10000) would otherwise pin a fixed
                # budget and defeat the auto-reach.
                p.tpq_max_steps  = 0
            if random_seed:
                p.ftlm_seed = int(random_seed)
                p.ltlm_seed = int(random_seed)
            if n_up_val is not None:
                p.use_fixed_sz = True
                p.n_up = int(n_up_val)
            # SOTA upgrade (May 2026): the per-irrep sector loop is
            # now wired for every thermal method (FTLM / LTLM / KPM /
            # mTPQ) via
            # ``_core.workflows_thermal_streaming_symmetry_directory``
            # + ``ed::core::combine_sector_thermodynamics``. We still
            # honour the user's ``use_symmetry_if_available`` toggle,
            # but no longer silently downgrade TPQ -- it now feeds
            # exactly the same streaming loop as FTLM/LTLM/KPM, with
            # the Z-weighted recombiner handling sector mixing.
            p.use_symmetry = bool(has_sym)
            # `sector=` names QUANTUM NUMBERS; selected_sectors takes raw
            # sector INDICES. Resolve against the directory's own
            # sector_metadata.json (the same table the C++ side reads) rather
            # than assuming the two axes coincide -- they only do for a
            # single-generator group.
            if _sector_sid is not None:
                # GAP 9: with the flip projection engaged the sector set
                # is EXTENDED (k and k + n_raw are the two parities of one
                # momentum). Select both; filter_sectors drops the partner
                # silently when the flip is off.
                _n_raw = len((_tbl or {}).get("sectors", []) or [])
                p.selected_sectors = [int(_sector_sid),
                                      int(_sector_sid) + _n_raw]
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
                p = _make_dir_params(None)
                if not _can_use_workflows_thermal(method_enum, has_sym):
                    raise NotImplementedError(
                        f"qed.thermal: method={method!r} has no "
                        f"orchestrator binding; supported methods are "
                        f"{sorted(_THERMAL_METHOD_MAP.keys())!r}."
                    )
                if has_sym:
                    res = _thermal_via_workflows_streaming_symmetry(
                        directory, N, float(spin),
                        method_enum, p,
                        use_gpu=_use_gpu, use_mpi=_use_mpi,
                    )
                else:
                    res = _thermal_via_workflows_thermal(
                        H_op, method_enum, p,
                        use_gpu=_use_gpu, use_mpi=_use_mpi,
                    )
                thermo = _sector_thermo_arrays(res)
                if thermo is None:
                    raise RuntimeError(
                        "qed.thermal: dispatcher returned no thermodynamic "
                        "data."
                    )
                temps, E, Cv, S, F = thermo
                gs_E = (float(res.eigenvalues[0])
                        if len(res.eigenvalues) > 0 else math.inf)
                h5_path = str(getattr(res, "eigenvectors_path", "") or "")
            finally:
                _cleanup_scratch()
            return ThermalResult(
                temperatures=temps, energy=E, specific_heat=Cv,
                entropy=S, free_energy=F, method=str(method),
                ground_state_energy=gs_E,
                used_sz_decomposition=False,
                # Post-SOTA upgrade (May 2026): the streaming-symmetry
                # path is wired for every thermal method, so the
                # symmetry-decomposition flag is now `has_sym`
                # unconditionally.
                used_symmetry_decomposition=bool(has_sym),
                hdf5_path=h5_path,
                dos_energies=np.asarray(
                    getattr(res, "dos_energies", []) or [], dtype=float),
                dos_values=np.asarray(
                    getattr(res, "dos_values", []) or [], dtype=float),
            )

        if verbose:
            print(
                f"[qed.thermal] iterating n_up in [{lo}, {hi}] of "
                f"[0, {N}]; per-sector symmetry={has_sym}"
            )

        try:
            # Validate method support once before spawning threads.
            if not _can_use_workflows_thermal(method_enum, has_sym):
                raise NotImplementedError(
                    f"qed.thermal: method={method!r} has no "
                    f"orchestrator binding; supported methods are "
                    f"{sorted(_THERMAL_METHOD_MAP.keys())!r}."
                )

            if has_sym:
                # Fast path (Jun 2026): single C++ call builds ALL
                # (n_up, irrep) sectors in one orbit-rep pass and runs
                # a flat OMP pool. Eliminates N+1 cold-start overhead
                # from the per-n_up ThreadPoolExecutor loop.
                # _make_dir_params(None) → output_dir is the parent dir;
                # C++ manages per-sector subdirs internally
                # (sz_<n_up>_sector_k_<k>/). For needs_scratch the
                # scratch dir has no n_up suffix.
                if _sector_sid is not None:
                    # The all-Sz binding builds its OWN sector set and does
                    # `topts.selected_sectors.clear()` per sector, so it does
                    # not filter its loop by the caller's selection -- passing
                    # sector= here would be SILENTLY IGNORED and the caller
                    # would get the fully recombined thermodynamics while
                    # believing they had one irrep.
                    #
                    # Refusing beats lying. Making this lane honour the filter
                    # is not a one-liner: its loop interacts with TR pairing
                    # (tr_plan.skip[i] copies from source[i], so a filter must
                    # keep the closure selected + their TR sources, or the
                    # selected sector silently copies an EMPTY result) and with
                    # the Stage-5 flip mirror that duplicates n_up < N/2 onto
                    # its partner. Both need a decided semantic, not a guess.
                    raise NotImplementedError(
                        "qed.thermal: sector= is not yet supported on the "
                        "all-Sz fast path (Sz-conserving H + spatial "
                        "symmetry): that C++ lane builds its own sector set "
                        "and does not honour a sector filter, so the argument "
                        "would be silently ignored. Workarounds: pass "
                        "use_sz_if_conserved=False to take the single-call "
                        "streaming-symmetry lane (which does filter), or use "
                        "qed.solve(sector=..., sz=...) for sector-resolved "
                        "eigenvalues.")
                p = _make_dir_params(None)
                tr = _thermal_via_workflows_all_sz_streaming_symmetry(
                    directory, N, float(spin), method_enum, p,
                    lo, hi, use_gpu=_use_gpu, use_mpi=_use_mpi,
                    spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
                    time_reversal=_sym_toggle_int(time_reversal,
                                                  "time_reversal"),
                    star_maps=star_maps)
                td = tr.thermo
                if not td or not td.temperatures:
                    raise RuntimeError(
                        "qed.thermal: all-Sz binding returned no "
                        "thermodynamic data.")
                temps = np.array(td.temperatures)
                E     = np.array(td.energy)
                Cv    = np.array(td.specific_heat)
                S     = np.array(td.entropy)
                F     = np.array(td.free_energy)
                gs_E  = float(tr.ground_state_energy)
                h5_path_multi = str(getattr(tr, "hdf5_path", "") or "")
                for s in (tr.per_sector or []):
                    if not (hasattr(s, "thermo") and s.thermo.temperatures):
                        continue
                    n_up_s = (int(s.sz_index)
                              if hasattr(s, "sz_index")
                              and s.sz_index is not None else None)
                    sec_dim_s = (int(s.tag.sector_dim)
                                 if hasattr(s, "tag") else 0)
                    per_sector_records.append(ThermalSectorEntry(
                        n_up=n_up_s,
                        sector_dim=sec_dim_s,
                        temperatures=np.array(s.thermo.temperatures),
                        energy=np.array(s.thermo.energy),
                        specific_heat=np.array(s.thermo.specific_heat),
                        entropy=np.array(s.thermo.entropy),
                        free_energy=np.array(s.thermo.free_energy),
                    ))
                if verbose:
                    print(
                        f"[qed.thermal] all-Sz flat-pool: "
                        f"{len(per_sector_records)} (n_up, irrep) sectors "
                        f"recombined in one C++ call")
            else:
                # Parallel Sz outer loop. The C++ bindings release the GIL
                # so Python threads run concurrently. Each n_up call is
                # fully independent: it reads from the same (read-only)
                # directory, builds its own SectorOperatorSet, and writes to
                # a dedicated per-n_up scratch/output subdirectory.
                # For maximum throughput on many-core machines combine with
                # ED_SYM_SECTOR_PARALLEL=1 in the C++ inner loop. The
                # recommended thread budget for that combination is:
                #   QED_SZ_WORKERS = ceil(N_cores / N_irreps)
                #   OMP_NUM_THREADS = N_cores / QED_SZ_WORKERS
                #   OPENBLAS_NUM_THREADS = same; OMP_MAX_ACTIVE_LEVELS=1
                # For the default (no QED_SZ_WORKERS set) we pick one worker
                # per Sz sector so all n_up values run simultaneously with
                # the remaining parallelism absorbed by the inner OMP loop.
                _n_up_values = [
                    n for n in range(lo, hi + 1)
                    if math.comb(N, n) > 0
                ]
                _sz_workers = len(_n_up_values)
                if (env_w := os.environ.get("QED_SZ_WORKERS")):
                    try:
                        _sz_workers = max(1, int(env_w))
                    except ValueError:
                        pass

                def _run_n_up(n_up: int):
                    sec_dim = math.comb(N, n_up)
                    p = _make_dir_params(n_up)
                    res = _thermal_via_workflows_thermal(
                        H_op, method_enum, p,
                        use_gpu=_use_gpu, use_mpi=_use_mpi,
                    )
                    return n_up, sec_dim, res

                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=_sz_workers) as _pool:
                    _futures = {
                        _pool.submit(_run_n_up, n_up): n_up
                        for n_up in _n_up_values
                    }
                    for _fut in concurrent.futures.as_completed(_futures):
                        n_up, sec_dim, res = _fut.result()
                        if verbose:
                            print(f"  [n_up={n_up}] dim={sec_dim} done")
                        thermo = _sector_thermo_arrays(res)
                        if thermo is None:
                            if verbose:
                                print(
                                    f"    (n_up={n_up}: no thermo data; "
                                    f"skipping)"
                                )
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
                        _sec_h5 = str(
                            getattr(res, "eigenvectors_path", "") or "")
                        if _sec_h5:
                            sector_hdf5_paths[n_up] = _sec_h5

                if not per_sector_blocks:
                    raise RuntimeError(
                        "qed.thermal: no Sz sector in the requested window "
                        "produced thermodynamic data."
                    )
                temps, E, Cv, S, F = _combine_sector_thermodynamics(
                    per_sector_blocks)
                h5_path_multi = output_dir if output_dir else ""
        finally:
            _cleanup_scratch()
        if verbose and not has_sym:
            print(
                f"[qed.thermal] Z-recombined {len(per_sector_blocks)} "
                f"Sz sectors"
            )
        return ThermalResult(
            temperatures=temps, energy=E, specific_heat=Cv, entropy=S,
            free_energy=F, method=str(method),
            ground_state_energy=gs_E,
            hdf5_path=h5_path_multi,
            sector_hdf5_paths=sector_hdf5_paths,
            used_sz_decomposition=True,
            # TPQ is NOT carved out here: since the May-2026 SOTA upgrade
            # every method (incl. mTPQ/cTPQ) feeds the same all-Sz
            # streaming-symmetry loop above, so the flag is `has_sym`
            # unconditionally -- matching the sz-not-conserved return site
            # and the comment at the `p.use_symmetry` assignment. (The old
            # `method_enum not in _TPQ_METHODS` carve-out was a stale
            # leftover of the pre-SOTA silent TPQ downgrade: the lane ran
            # all 66 (n_up, irrep) sectors and then reported False.)
            used_symmetry_decomposition=bool(has_sym),
            per_sector=per_sector_records,
        )

    # ------------------------------------------------------------------
    # 3b. In-memory form -- iterate via qed.solve(H, sz=n_up).
    # ------------------------------------------------------------------
    def _make_diag_kwargs(n_up_val: Optional[int]) -> dict[str, Any]:
        # Multi-Sz overwrite fix (May 2026): EVERY method's HDF5 schema
        # ( ``/tpq/samples/sample_<s>/...`` for TPQ,
        #   ``/ftlm/averaged/<curve>`` for FTLM,
        #   ``/ltlm/averaged/<curve>`` for LTLM,
        #   ``/kpm_dos/...`` for KPM_DOS )
        # is keyed by sample / beta / curve with **no sector tag**, so
        # if two Sz sectors share an ``output_dir`` the second sector's
        # ``ed_results.h5`` write silently overwrites the first. Route
        # every Sz sector to ``<output_dir>/n_up_<n_up>/ed_results.h5``
        # regardless of method. TPQ + multi-Sz was the user-visible
        # symptom because state vectors live ONLY on disk; for
        # FTLM / LTLM / KPM_DOS the recombined thermo in
        # ``ThermalResult.thermo`` would mask the corruption but the
        # HDF5 file (used for diagnostics, post-processing, audit
        # trails) would still be wrong.
        if not output_dir:
            _resolved_outdir = "/dev/null"      # writes disabled
        elif output_dir and n_up_val is not None:
            _resolved_outdir = _persistent_sector_outdir(output_dir, n_up_val)
            if not _resolved_outdir.startswith("/dev/null"):
                os.makedirs(_resolved_outdir, exist_ok=True)
        else:
            _resolved_outdir = output_dir

        kwargs: dict[str, Any] = {
            "solver": method_enum,
            "num_eigenvalues": 1,
            "tolerance": tolerance,
            "num_samples": num_samples,
            "num_temp_points": num_T,
            "temp_min": T_min,
            "temp_max": T_max,
            "output_dir": _resolved_outdir,
            "verbose": False,
            # Phase C of the "Backend x Symmetries x Workflows" plan
            # (May 2026): forward the device selector to every
            # per-sector solver call so the in-memory thermal path
            # reaches the GPU when requested.
            "device": device,
        }
        # Pipe ``max_iterations`` into the per-sector solver call. For
        # mTPQ this controls the iteration budget; for FTLM /
        # LTLM / KPM_DOS it's the Lanczos / KPM cap. Closes a gap where
        # this helper silently used the EDParameters default (1000)
        # regardless of what the user asked for.
        if max_iterations is not None:
            kwargs["max_iterations"] = int(max_iterations)
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
            res = _solve(H_op, auto_sz=False, **_make_diag_kwargs(None))
            thermo = _sector_thermo_arrays(res)
            if thermo is None:
                raise RuntimeError(
                    "qed.thermal: solver returned no thermodynamic data."
                )
            temps, E, Cv, S, F = thermo
            gs_E = (float(res.eigenvalues[0])
                    if len(res.eigenvalues) > 0 else math.inf)
            h5_path_solo = str(getattr(res, "eigenvectors_path", "") or "")
        finally:
            _cleanup_scratch()
        return ThermalResult(
            temperatures=temps, energy=E, specific_heat=Cv, entropy=S,
            free_energy=F, method=str(method),
            ground_state_energy=gs_E,
            used_sz_decomposition=False,
            used_symmetry_decomposition=False,
            hdf5_path=h5_path_solo,
            dos_energies=np.asarray(
                getattr(res, "dos_energies", []) or [], dtype=float),
            dos_values=np.asarray(
                getattr(res, "dos_values", []) or [], dtype=float),
        )

    if verbose:
        print(
            f"[qed.thermal] iterating n_up in [{lo}, {hi}] of [0, {N}]"
        )

    per_sector_blocks = []
    per_sector_records = []
    sector_hdf5_paths_imem: dict[Optional[int], str] = {}
    gs_E = math.inf

    try:
        for n_up in range(lo, hi + 1):
            sec_dim = math.comb(N, n_up)
            if sec_dim == 0:
                continue
            if verbose:
                print(f"  [n_up={n_up}] dim={sec_dim}")

            res = _solve(H_op, sz=n_up, **_make_diag_kwargs(n_up))
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
            # Save & DSSF Upgrades follow-up (May 2026): record this
            # sector's HDF5 file (now per-sector under
            # ``<output_dir>/n_up_<n_up>/``).
            _sec_h5_imem = str(getattr(res, "eigenvectors_path", "") or "")
            if _sec_h5_imem:
                sector_hdf5_paths_imem[n_up] = _sec_h5_imem

        if not per_sector_blocks:
            raise RuntimeError(
                "qed.thermal: no Sz sector in the requested window "
                "produced thermodynamic data."
            )

        temps, E, Cv, S, F = _combine_sector_thermodynamics(per_sector_blocks)
        # Save & DSSF Upgrades follow-up (May 2026): with per-sector
        # subdirectory layout the per-sector files live under
        # ``<output_dir>/n_up_<n_up>/``; surface the parent here and
        # expose the sector -> file map via ``sector_hdf5_paths``.
        h5_path_imem_multi = output_dir if output_dir else ""
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
        hdf5_path=h5_path_imem_multi,
        sector_hdf5_paths=sector_hdf5_paths_imem,
    )
