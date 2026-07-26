"""qed._params -- THE parameter/result conversion layer (Stage 11a).

One module owns every translation between the legacy ``EDParameters``
bag and the orchestrator's ``SolveOptions`` / ``ThermalOptions`` (and
back from the orchestrator result shapes into the ``EDResults``
envelope).

Why this exists: the thermal converter had silently FORKED --
``workflow.py`` and ``thermal.py`` each carried their own
``_ed_params_to_thermal_options`` which diverged by 75 lines. One copy
raised on a non-thermal method while the other silently defaulted it to
FTLM; one wired ``device=`` into the backend constraints while the
other dropped it; one forwarded the KPM moment/Hutchinson knobs while
the other silently dropped them (a ~250x cost amplifier when the
defaults kicked in). The merged converters below are the UNION of both
copies' fixes: strict method mapping, backend wiring, feasibility gate,
AND the KPM/OFTLM/mTPQ knob forwarding.

The direction of travel (Stage 11 program): the orchestrator Options
are the canonical parameter surface; ``EDParameters`` survives as the
kwargs staging bag whose ONLY consumers are the converters in this
file plus the CLI adapter (`ed_config_adapter.h`).

Stage 11a-tail: the SolveOptions converter is no longer duplicated
across the language boundary either -- ``ed_params_to_solve_options``
below delegates to the bound ``ed_adapter::toSolveOptions`` (the same
function the CLI calls), with the deliberate semantic differences
expressed as explicit flags instead of divergent copies. The
ThermalOptions converter has no C++ twin (the CLI drives thermal
through the legacy dispatcher), so it stays implemented here.
"""

from __future__ import annotations

import warnings

from . import _core as _core
from ._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDParameters,
    EDResults,
)

__all__ = [
    "THERMAL_METHOD_MAP",
    "ed_params_to_solve_options",
    "ed_params_to_thermal_options",
    "ed_result_from_gs_result",
    "ed_result_from_thermal_result",
]

# The one thermal-method map (enum -> enum). Strict: an unknown method
# raises in the converter -- the silent default-to-FTLM fork is dead.
THERMAL_METHOD_MAP = {
    DiagonalizationMethod.FTLM:    _core.ThermalMethod.FTLM,
    DiagonalizationMethod.OFTLM:   _core.ThermalMethod.OFTLM,
    DiagonalizationMethod.LTLM:    _core.ThermalMethod.LTLM,
    DiagonalizationMethod.mTPQ:    _core.ThermalMethod.mTPQ,
    DiagonalizationMethod.KPM_DOS: _core.ThermalMethod.KpmDos,
}


def ed_params_to_thermal_options(
    params: EDParameters,
    method: DiagonalizationMethod,
    allow_infeasible: bool = False,
) -> "_core.ThermalOptions":
    """Translate the ``EDParameters`` bag + a thermal method into a
    fresh ``_core.ThermalOptions`` for ``workflows_thermal``."""
    opts = _core.ThermalOptions()
    core_method = THERMAL_METHOD_MAP.get(method)
    if core_method is None:
        raise ValueError(
            f"ed_params_to_thermal_options: method {method!r} is not a "
            f"thermal lane. Use `ed_params_to_solve_options` for "
            f"ground-state methods.")
    opts.method      = core_method
    opts.num_samples = int(params.num_samples)
    # ``device=`` -> backend constraints: without this, the C++ default
    # (allow_gpu=true) silently routed device='cpu' thermal jobs through
    # the GPU promoter.
    opts.backend.allow_gpu = bool(getattr(params, "use_gpu", False))
    opts.backend.allow_mpi = bool(getattr(params, "use_mpi", False))
    if opts.backend.allow_gpu:
        # The Python surface vetted the GPU choice (explicit device= or
        # its own dim heuristic); disable the C++ auto-promotion floor.
        opts.backend.gpu_dim_floor = 0
    # ``krylov_dim`` is the orchestrator's per-method iteration budget:
    # FTLM/LTLM use it as the Krylov subspace dimension; mTPQ uses it as
    # the number of (L - H) iterations (0 = auto-size from the spectral
    # bounds to bracket beta_max = 1/T_min).
    if method in (DiagonalizationMethod.FTLM, DiagonalizationMethod.OFTLM):
        opts.krylov_dim = int(params.ftlm_krylov_dim or 100)
        if method == DiagonalizationMethod.OFTLM:
            # N_V (# exactly-treated low-lying states): honour an explicit
            # request, else keep the C++ ThermalOptions default (8).
            _nv = getattr(params, "oftlm_num_exact", None)
            if _nv is not None:
                opts.num_exact = int(_nv)
    elif method == DiagonalizationMethod.LTLM:
        opts.krylov_dim = int(params.ltlm_krylov_dim or 200)
    elif method == DiagonalizationMethod.mTPQ:
        steps = int(getattr(params, "tpq_max_steps", 0) or 0)
        if steps <= 0:
            mi = getattr(params, "max_iterations", 0)
            steps = int(mi) if mi else 0
        opts.krylov_dim = steps
    else:
        opts.krylov_dim = 100
    # KPM moment / Hutchinson knobs: forwarded so the streaming-symmetry
    # binding does not fall back to the KpmDosOptions defaults
    # (M=2048, R=20 -- a ~250x amplifier when unintended).
    if method == DiagonalizationMethod.KPM_DOS:
        kn_m = int(getattr(params, "kpm_num_moments", 0) or 0)
        kn_r = int(getattr(params, "kpm_num_random_vectors", 0) or 0)
        if kn_m > 0:
            opts.kpm_num_moments = kn_m
        if kn_r > 0:
            opts.kpm_num_random_vectors = kn_r
    opts.taylor_order  = int(params.tpq_taylor_order)
    opts.delta_beta    = float(params.tpq_delta_beta)
    opts.random_seed   = int(params.ftlm_seed or params.ltlm_seed or 0)
    opts.output_dir    = str(params.output_dir or "")
    opts.temp_min      = float(params.temp_min)
    opts.temp_max      = float(params.temp_max)
    opts.num_temp_bins = int(params.num_temp_bins)
    # mTPQ expert override of the (L*I - H) large value; 0.0 -> auto.
    if hasattr(opts, "energy_shift"):
        opts.energy_shift = float(
            getattr(params, "tpq_energy_shift", 0.0) or 0.0)
    # fp32 single-GPU mTPQ (memory-halving lane).
    if hasattr(opts, "mtpq_fp32"):
        opts.mtpq_fp32 = bool(getattr(params, "tpq_fp32", False))
    # Completion guarantee: the orchestrator refuses an infeasible plan
    # (clean throw before allocating) unless this is set.
    if hasattr(opts, "allow_infeasible"):
        opts.allow_infeasible = bool(allow_infeasible)
    # Probe-beta list for mTPQ state-vector snapshots.
    pb = list(getattr(params, "tpq_probe_betas", []) or [])
    if pb:
        opts.probe_betas = pb
    # Sector filter (raw sector INDICES -- see SolveOptions::selected_sectors;
    # qed.thermal resolves quantum numbers to indices before setting it). This
    # converter used to drop the field on the floor, so the thermal binding's
    # filter_sectors() call was unreachable from Python: the C++ honoured a
    # filter nothing could set. Same drift class the solve converter was
    # consolidated to kill (this module's header calls out selected_sectors by
    # name as one of the three fields its Python twin silently lost).
    sel = list(getattr(params, "selected_sectors", []) or [])
    if sel:
        opts.selected_sectors = [int(k) for k in sel]
    return opts


def ed_result_from_thermal_result(
    tr: "_core.ThermalResult",
) -> EDResults:
    """Wrap a ``_core.ThermalResult`` into the ``EDResults`` envelope
    (thermo curve + the persisted HDF5 path)."""
    out = EDResults()
    out.thermo_data = tr.thermo
    out.eigenvalues = ([float(tr.ground_state_energy)]
                       if tr.ground_state_energy != 0.0 else [])
    h5_path = str(getattr(tr, "hdf5_path", "") or "")
    out.eigenvectors_computed = bool(h5_path)
    out.eigenvectors_path     = h5_path
    # KPM-DOS raw density of states (Jul 2026): surface the density(E) the
    # method produces so callers can consume/integrate it (EDResults carries
    # py::dynamic_attr; empty lists for the non-DOS methods).
    out.dos_energies = list(getattr(tr, "dos_energies", []) or [])
    out.dos_values   = list(getattr(tr, "dos_values", []) or [])
    return out


def ed_params_to_solve_options(
    params: EDParameters,
    method: DiagonalizationMethod,
    auto_method: bool = False,
    allow_infeasible: bool = False,
) -> "_core.SolveOptions":
    """Translate ``EDParameters`` + ``DiagonalizationMethod`` into a
    ``_core.SolveOptions`` for the orchestrator.

    Delegates to THE one converter, ``ed_adapter::toSolveOptions``
    (ed_config_adapter.h) -- this module previously carried a Python
    twin of it that silently drifted in three fields (backend wiring,
    ``allow_infeasible``, ``selected_sectors``). ``wire_backend=True``
    is the Python-surface semantic: ``device='cpu'`` must PIN the CPU
    backend, where the CLI keeps its historical auto-promotion."""
    return _core.ed_params_to_solve_options(
        params=params,
        method=method,
        auto_method=bool(auto_method),
        wire_backend=True,
        allow_infeasible=bool(allow_infeasible),
    )


def ed_result_from_gs_result(
    gs_result: "_core.GroundStateResult",
    params: EDParameters,
) -> EDResults:
    """Wrap a ``_core.GroundStateResult`` into the ``EDResults``
    envelope (eigenvalues, per-sector diagnostics, backend lane,
    convergence bookkeeping + the warn-not-fail completion contract)."""
    out = EDResults()
    out.eigenvalues = list(gs_result.eigenvalues)
    out.eigenvectors_computed = bool(params.compute_eigenvectors)
    out.eigenvectors_path = str(getattr(gs_result, "hdf5_path", "") or "")
    _eps = getattr(gs_result, "eigenvalues_per_sector", None)
    if _eps:
        out.eigenvalues_per_sector = [list(s) for s in _eps]
    _tags = getattr(gs_result, "sector_tags", None)
    if _tags:
        out.sector_tags = list(_tags)
    # Stage 12 (SU(2) rollout): certified total-spin labels, parallel to
    # ``eigenvalues`` when the C++ side produced them. ``spin`` is S as a
    # float (0.0, 0.5, 1.0, ...) or None where certification failed.
    _twoS = list(getattr(gs_result, "two_S_of_eigenvalue", []) or [])
    if _twoS:
        out.spin = [(t / 2.0) if t >= 0 else None for t in _twoS]
        out.s2 = list(getattr(gs_result, "s2_of_eigenvalue", []) or [])
    _bk = getattr(gs_result, "backend", None)
    if _bk is not None:
        out.backend = _bk
    _kry = getattr(gs_result, "krylov", None)
    if _kry is not None:
        out.converged        = bool(getattr(_kry, "converged", False))
        out.iterations       = int(getattr(_kry, "iters_done", 0))
        out.residuals        = list(getattr(_kry, "ritz_residuals", []) or [])
        out.n_converged      = int(getattr(_kry, "n_converged", 0))
        out.residual_history = list(getattr(_kry, "resid_history", []) or [])
        if not out.converged:
            want = int(getattr(params, "num_eigenvalues", 1) or 1)
            if out.residuals:
                rmax = max(out.residuals)
                warnings.warn(
                    f"qed.solve: eigensolver did not fully converge "
                    f"({out.n_converged}/{want} eigenpairs below tolerance; "
                    f"max residual {rmax:.2e} after {out.iterations} "
                    f"iterations). Returning the best-effort result. Raise "
                    f"max_iterations / tolerance, or give the run more "
                    f"memory so the Krylov subspace need not be capped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                # Jul 2026: the old message claimed "0/k below tolerance;
                # max residual nan" whenever the backend simply did not
                # REPORT per-eigenpair residuals -- alarming and wrong
                # (caught by the e2e diagnostic on runs whose energies were
                # exact to 8+ digits). Say what is actually known.
                warnings.warn(
                    f"qed.solve: the backend reported no per-eigenpair "
                    f"residuals after {out.iterations} iterations, so "
                    f"convergence could not be verified (energies may "
                    f"still be fully converged). Cross-check E0 or rerun "
                    f"with a residual-reporting solver if it matters.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return out
