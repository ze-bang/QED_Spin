"""``qed.workflow``: implementation module for :func:`qed.solve`.

This is the internal home of the auto-pilot (method picker, device
picker, planner, auto-tuner, MPI launcher) that ``qed.solve`` exposes
as a single kwargs-only entry point. Callers should import the public
name :func:`qed.solve` instead of reaching in here directly.

The legacy alias :func:`qed.solve` has been removed; use
:func:`qed.solve` everywhere.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

from . import _core as _core
from .point_group_routing import resolve_projection_lane, split_nonabelian
from ._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    FixedSzOperator,
    Operator,
    ThermodynamicData,
    has_cuda_build,
    has_mpi_build,
)

__all__ = [
    "GeneratorSet",
    "SymmetryReport",
    "find_symmetries",
    "resolve_auto_symmetry",
    "solve",
    "full_spectrum",
    "list_diag_parameters",
    "solver_device_support",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Permutation = list[int]
SymmetryArg = Union["GeneratorSet", Sequence[Permutation], dict[str, Any], None]


# ---------------------------------------------------------------------------
# Surface unification (May 2026): the C++ pybind11 forwarders
# `_core.exact_diagonalization_*` have been deleted in lockstep with
# the Wave 2 / Wave 3 collapse. Every in-tree call site now routes
# through `_core.workflows_*` (ground-state + thermal) or
# `_core.workflows_solve_streaming_symmetry_directory` (streaming-
# symmetry). This shim is preserved as a no-op for any third-party
# code that imported it transitively; it will be deleted in the next
# cycle.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full Unified-Interface Collapse, Wave E1 (May 2026): helpers that bridge
# the legacy `EDParameters` + `DiagonalizationMethod` surface onto the new
# `_core.workflows_solve(op, SolveOptions)` orchestrator. Used by
# `_diag_via_workflows_solve` below to repoint the canonical
# CPU+no-symmetry ground-state path at the unified orchestrator while
# preserving the `EDResults` envelope every legacy caller expects.
# ---------------------------------------------------------------------------

# Ground-state methods the orchestrator's `workflows_solve` handles.
# Thermal methods (FTLM / LTLM / mTPQ / KPM_DOS) route through
# `_core.workflows_thermal` instead -- the dispatch is decided by the
# helper `_diag_via_workflows_solve` below.
_GROUND_STATE_METHODS = frozenset({
    DiagonalizationMethod.LANCZOS,
    DiagonalizationMethod.BLOCK_LANCZOS,
    DiagonalizationMethod.KRYLOV_SCHUR,
    DiagonalizationMethod.BLOCK_KRYLOV_SCHUR,
    DiagonalizationMethod.FULL,
})

# Stage 11a: the parameter/result converters live in qed._params (the
# thermal converter had FORKED between workflow.py and thermal.py).
from ._params import (  # noqa: E402,F401  (single conversion layer)
    THERMAL_METHOD_MAP as _THERMAL_METHOD_MAP,
    ed_params_to_solve_options as _ed_params_to_solve_options,
    ed_params_to_thermal_options as _ed_params_to_thermal_options,
    ed_result_from_gs_result as _ed_result_from_gs_result,
    ed_result_from_thermal_result as _ed_result_from_thermal_result,
)


def _is_ground_state_method(method: DiagonalizationMethod) -> bool:
    """True when `method` is one of the four ground-state lanes that
    `_core.workflows_solve` handles natively."""
    return method in _GROUND_STATE_METHODS






def _diag_via_workflows_solve(
    operator: Operator,
    method: DiagonalizationMethod,
    params: EDParameters,
    auto_method: bool = False,
    allow_infeasible: bool = False,
) -> EDResults:
    """Route an in-memory `Operator` through the unified orchestrator.

    Ground-state methods (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / FULL)
    go through ``_core.workflows_solve``. Thermal methods (FTLM / LTLM
    / mTPQ / KPM_DOS) route through ``_core.workflows_thermal``.
    No legacy fallback remains: the C++ ``exact_diagonalization_*``
    family was deleted in the surface-unification collapse and every
    Python-side call site now lands on ``_core.workflows_*``."""
    if _is_ground_state_method(method):
        opts = _ed_params_to_solve_options(params, method, auto_method, allow_infeasible)
        # The orchestrator's `workflows_solve` accepts an `Operator&`;
        # if the caller already projected to a fixed-Sz sector we hand
        # it the `FixedSzOperator` directly (it derives from `Operator`).
        gs   = _core.workflows_solve(operator, opts)
        return _ed_result_from_gs_result(gs, params)
    if method in _THERMAL_METHOD_MAP:
        opts = _ed_params_to_thermal_options(params, method, allow_infeasible)
        tr = _core.workflows_thermal(operator, opts)
        return _ed_result_from_thermal_result(tr)
    raise ValueError(
        f"_diag_via_workflows_solve: unsupported DiagonalizationMethod "
        f"{method!r}. Supported: ground-state "
        f"(LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / FULL) and thermal "
        f"(FTLM / LTLM / mTPQ / KPM_DOS)."
    )


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 10b: the L6 group-discovery layer lives in qed.discovery; every name
# is re-exported here for back-compat (qed.workflow.find_symmetries etc.).
# ---------------------------------------------------------------------------
from .discovery import (  # noqa: F401  (re-exports)
    GeneratorSet,
    SymmetryReport,
    find_symmetries,
    resolve_auto_symmetry,
    resolve_discrete_toggle,
    _FIND_SYM_MEMO,
    _find_symmetries_key,
    _find_symmetries_impl,
    _validate_explicit_generators,
    _full_group_generators,
    _operator_to_graph_records,
    _run_full_automorphism_pipeline,
    _translation_autos_from_lattice,
    _infer_cluster_dims,
    _make_generator_set_from_clique,
    _generators_equal,
)


def solve(
    H: Union[Operator, FixedSzOperator],
    *,
    num_eigenvalues: int = 1,
    tolerance: float = 1e-10,
    compute_eigenvectors: bool = False,
    solver: Optional[Union[str, DiagonalizationMethod]] = None,
    device: Optional[str] = None,
    symmetry: SymmetryArg = None,
    sector: Optional[Sequence[int]] = None,
    sz: Optional[int] = None,
    auto_sz: bool = True,
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    point_group: Union[str, bool, None] = "auto",
    lattice: Optional[Any] = None,
    output_dir: str = "",
    max_iterations: Optional[int] = None,
    block_size: Optional[int] = None,
    # Thermal-method first-class shortcuts (only consulted for
    # mTPQ / FTLM / LTLM; ignored for eigenvalue solvers, where
    # the relevant knob is num_eigenvalues + max_iterations).
    num_samples: Optional[int] = None,
    target_beta: Optional[float] = None,
    num_temp_points: Optional[int] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
    verbose: bool = True,
    full_spectrum: bool = False,
    extra_params: Optional[dict[str, Any]] = None,
) -> EDResults:
    """One-call exact diagonalization with smart defaults.

    Routes through the same C++ dispatcher as the legacy API, but
    handles parameter selection / device picking / sector selection /
    eigenvalue-vs-thermal classification automatically.

    Parameters
    ----------
    H : Operator or FixedSzOperator
        Spin Hamiltonian. Pass an :class:`Operator` for the full Hilbert
        space; pass a :class:`FixedSzOperator` (or use ``sz=``) to
        restrict to a fixed-Sz sector.
    num_eigenvalues : int, optional
        Number of eigenvalues to compute. Default 1.
    tolerance : float, optional
        Convergence tolerance the solver should meet on the requested
        eigenvalues. Default ``1e-10``.
    compute_eigenvectors : bool, optional
        If True, eigenvectors are computed and persisted to
        ``output_dir`` (HDF5). Default False.
    solver : str or DiagonalizationMethod, optional
        Backend to use. ``None`` (default) means ``"auto"``: the
        function picks LANCZOS / KRYLOV_SCHUR / FULL based on the
        Hilbert-space dimension and the requested ``num_eigenvalues``.
        Pass an explicit method name (string or enum) to override.
        String lookup is case-insensitive; ``"mtpq"`` /
        ``"mTPQ"`` / ``"MTPQ"`` all resolve to the same enum.
        Supported families:

        * eigenvalue (returns ``EDResults.eigenvalues``):
          ``LANCZOS``, ``BLOCK_LANCZOS``, ``KRYLOV_SCHUR``, ``FULL``.
        * thermal (returns ``EDResults`` with the imaginary-time
          trajectory in ``eigenvalues`` and the post-processed
          thermodynamic curve on disk in ``output_dir``):
          ``mTPQ``, ``FTLM``, ``LTLM``, ``KPM_DOS``.
    device : str, optional
        Backend device. One of ``"auto"`` / ``"cpu"`` / ``"gpu"`` /
        ``"mpi"`` / ``"mpi_gpu"``. ``None`` (default) means ``"auto"``.
        ``"mpi"`` / ``"mpi_gpu"`` are first-class: the workflow
        writes ``H`` (and the symmetry directory if ``symmetry=`` is
        set) to a temp directory, shells out to ``mpiexec
        ed_distributed_main`` (with ``--gpu`` for ``mpi_gpu``), and
        parses the HDF5 result back into an :class:`EDResults`. The
        binary itself, ``mpiexec``, and the MPI-rank count come from
        the ``mpi_*`` kwargs (or sensible defaults). Python never
        calls ``MPI_Init`` directly; the launcher does, in a
        separate process tree.
    symmetry : GeneratorSet, list[Permutation], or dict, optional
        If provided, the diagonalization runs in the symmetry-projected
        basis via the streaming symmetry kernel (per-sector matrix-free
        apply). Accepts a :class:`GeneratorSet`, a raw list of
        permutations, or the dict produced by
        :func:`qed.symmetry.group_from_generators`.

        **Not supported for TPQ methods**: TPQ relies on a single
        random vector spread across the whole Hilbert space, so the
        "diagonalise each sector independently" template doesn't
        apply. Combine TPQ with ``sz=`` instead, or pre-project to
        the relevant Sz block.
    sector : sequence[int], optional
        When ``symmetry`` is provided, restrict to the irrep with these
        quantum numbers (one per generator). When omitted, every irrep
        is diagonalised and the eigenvalues merged.
    sz : int, optional
        When ``H`` is an :class:`Operator` and the operator commutes
        with total Sz, restrict to the sector with this many up spins.
    point_group : str or bool, optional
        Stage 9c semantics. ``"auto"`` (default): eigenvalue-only calls
        PROJECT through the factorized little-group engine (translation
        x flip x little-co-group blocks); vector consumers
        (``compute_eigenvectors``, ``sector=``, sampling methods, an
        explicit GPU ``device=``) use the abelian rep lane with
        star/TR/flip folds and per-sector output. ``"full"``: REQUIRE
        projection -- raises with the decline reason instead of
        degrading. ``"off"``: abelian lane, star folds disabled.
        Output contract (Stage 10c): BOTH lanes set ``eigenvalues``.
        The projection lane additionally sets per-eigenvalue label
        arrays ``block_k_raw`` / ``block_flip_parity`` /
        ``block_irrep`` / ``block_irrep_dim`` / ``block_multiplicity``
        / ``block_subspace`` plus ``irrep_characters`` (decode the
        momentum as the phase of chi_k(T); the engine's irrep index
        order is NOT the directory sector order). The abelian lane
        sets ``sector_tags`` + ``eigenvalues_per_sector`` instead --
        pass ``point_group="off"`` when those per-sector arrays are
        required.
    spin_flip, time_reversal : str or bool, optional
        Per-symmetry toggles (``"auto"`` / ``"on"`` / ``"off"`` /
        ``"require"``): the flip projects at half filling and
        transports elsewhere; TR folds conjugate sectors. ``"require"``
        throws when H does not carry the symmetry. Applied identically
        on both lanes.
    lattice : qed.input.Lattice, optional
        Required by ``symmetry="translation"`` (identifies which
        automorphisms are pure translations; the point group rides as
        star residue).
    output_dir : str, optional
        Directory where the C++ engine should write eigenvectors /
        HDF5 artefacts and (for thermal methods) the imaginary-time
        trajectory + thermodynamic-curve text files. The default
        ``""`` disables disk writes for eigenvalue solvers (the
        ``isDisabledOutputPath`` shortcut); for **thermal** solvers
        an empty ``output_dir`` is auto-replaced with a fresh
        ``./qed_thermal_<timestamp>/`` directory whose path is
        printed when ``verbose=True``. Pass an explicit dir to keep
        the data alongside other artefacts.
    max_iterations, block_size : int, optional
        Manual overrides of the auto-tuned solver parameters. For
        eigenvalue Krylov solvers ``max_iterations`` is the Krylov
        subspace dimension; for thermal methods it is forwarded as
        ``tpq_max_steps`` (number of imaginary-time Taylor steps).
    num_samples : int, optional
        (Thermal only.) Number of random initial states to average
        over. Default 1; for production work 8-32 is typical.
    target_beta : float, optional
        (TPQ only.) Lowest temperature β = 1/T to reach. Default 20.
    num_temp_points : int, optional
        (Thermal only.) Number of measurement points along the
        imaginary-time trajectory. Default 20.
    temp_min, temp_max : float, optional
        (Thermal only.) Endpoints of the temperature grid the
        thermodynamic post-processing emits.
    verbose : bool, optional
        If True (default), prints what the auto-selector chose.
    extra_params : dict, optional
        Forwarded to :class:`EDParameters` as ``setattr`` calls.
        Useful for niche flags (``tpq_*``, ``ltlm_*``, ``kpm_*``,
        etc.) that the unified ``diag`` doesn't expose individually.
        Call :func:`list_diag_parameters` to see the full catalogue.

    Returns
    -------
    EDResults
        For eigenvalue solvers: ``.eigenvalues`` is the requested
        spectrum. For thermal solvers: ``.eigenvalues`` is the
        imaginary-time trajectory of energy expectations and
        ``.eigenvectors_path`` points at the unified
        thermodynamic-curve file written under ``output_dir``.

    Examples
    --------
    The one-liner:

    .. code-block:: python

        eigs = qed.solve(H).eigenvalues

    Bottom-of-spectrum, fixed Sz:

    .. code-block:: python

        eigs = qed.solve(H, num_eigenvalues=4, sz=N // 2).eigenvalues

    Symmetry projection:

    .. code-block:: python

        report = qed.find_symmetries(H, lattice=lat)
        eigs = qed.solve(H, symmetry=report.full_set,
                        sz=N // 2, num_eigenvalues=2).eigenvalues

    Thermal trajectory via mTPQ:

    .. code-block:: python

        res = qed.solve(H, solver="mTPQ",
                       sz=N // 2,         # OK
                       num_samples=4,
                       target_beta=20.0,
                       output_dir="ed_runs/thermal")
        # res.eigenvalues is the per-step E(β) trajectory
    """
    if not isinstance(H, Operator):
        raise TypeError(
            f"qed.solve(H, ...) expected Operator or FixedSzOperator, "
            f"got {type(H).__name__}"
        )

    # ------------------------------------------------------------------
    # Full-spectrum shortcut: compute EVERY eigenvalue decomposed by all
    # (Sz x spatial) symmetries via the memory-light representative SpMV.
    # Routes to the standalone :func:`full_spectrum` helper, which loops
    # the symmetry blocks and dense-diagonalises each. (Honours symmetry=
    # / sz auto-detection; the remaining solver/device/thermal kwargs do
    # not apply to a full-spectrum sweep.)
    # ------------------------------------------------------------------
    if full_spectrum:
        if isinstance(H, FixedSzOperator):
            raise ValueError(
                "full_spectrum=True spans every Sz sector; pass the full "
                "Operator (not a FixedSzOperator) so the sweep can loop "
                "magnetisation blocks.")
        spin_l = float(getattr(H, "spin", 0.5))
        _dev = device if isinstance(device, str) else "cpu"
        return full_spectrum_compute(
            H, symmetry=symmetry,
            sz_conserved=(None if auto_sz else False),
            spin_length=spin_l, device=_dev,
            spin_flip=spin_flip, time_reversal=time_reversal,
            point_group=point_group, lattice=lattice, verbose=verbose)

    fixed_sz_input = isinstance(H, FixedSzOperator)
    num_sites = int(H.num_sites)
    base_dim = int(H.dimension)  # full Hilbert dim, even for FixedSz

    # ------------------------------------------------------------------
    # 1. Resolve the fixed-Sz axis. sz="even"/"odd" selects the
    #    Sz-PARITY halves instead (the Z2 remnant when U(1) is broken;
    #    also valid for U(1)-conserving H) -- handled by the symmetry
    #    dispatch below, not the fixed-Sz machinery.
    # ------------------------------------------------------------------
    _sz_parity_str: Optional[int] = None
    if isinstance(sz, str):
        _key = sz.strip().lower()
        if _key not in ("even", "odd"):
            raise ValueError(
                f"sz={sz!r}: string forms are 'even'/'odd' (Sz-parity "
                "halves) or pass an integer n_up.")
        _sz_parity_str = 0 if _key == "even" else 1
        sz = None
    op_to_use: Operator = H
    if sz is not None:
        if fixed_sz_input:
            # Sanity-check that the provided FixedSzOperator matches.
            # FixedSzOperator does not currently expose ``n_up``, so we
            # only verify the dimension is consistent with C(N, sz).
            expected = math.comb(num_sites, int(sz))
            if int(H.dimension) != expected:
                raise ValueError(
                    f"sz={sz} implies dimension C({num_sites}, {sz})={expected} "
                    f"but the supplied FixedSzOperator has dimension {H.dimension}. "
                    "Pass `H` and `sz` as a matched pair, or pass an Operator "
                    "and let qed.solve construct the FixedSzOperator for you."
                )
            op_to_use = H
        else:
            if not H.conserves_sz():
                raise ValueError(
                    "sz=... was requested but the supplied Operator does not "
                    "commute with total Sz. Build the Hamiltonian without "
                    "Sz-breaking terms (no transverse field, no Jpmpm in "
                    "general orientation, etc.) or drop the sz= argument."
                )
            if not (0 <= int(sz) <= num_sites):
                raise ValueError(
                    f"sz={sz} out of range [0, num_sites={num_sites}]"
                )
            op_to_use = H.make_fixed_sz(int(sz))
            if verbose:
                d = op_to_use.dimension
                print(f"[qed.solve] Sz sector n_up={sz}: dim={d} "
                      f"(reduced from {base_dim}).")
    elif fixed_sz_input and verbose:
        print(f"[qed.solve] FixedSzOperator supplied: dim={H.dimension}.")
    elif (sz is None and not fixed_sz_input and auto_sz
          and symmetry is None
          and H.conserves_sz()
          and not (isinstance(device, str)
                   and device.lower() in ("mpi", "mpi_gpu"))):
        # Auto-Sz projection: the Hamiltonian respects total Sz but no
        # sector was named. Default to the half-filling sector
        # n_up = N//2 (where the ground state lives for Heisenberg-style
        # antiferromagnets) -- it is C(N, N//2) ~ 2^N / sqrt(pi N/2),
        # i.e. a sqrt(pi N/2)x speedup over the full 2^N Hilbert space
        # at no accuracy cost. Pass ``auto_sz=False`` to keep the full
        # Hilbert space, or ``sz=k`` to pick a different sector.
        #
        # IMPORTANT: skipped when ``symmetry=`` is explicitly given. Otherwise
        # "pure spatial" (symmetry, no sz) would silently become sz+spatial in
        # the n_up=N//2 sector -- an INCOMPLETE spectrum, and the WRONG ground
        # state for any model whose GS is not at half-filling (the "no accuracy
        # cost" claim holds only for the Heisenberg-AFM GS). With an explicit
        # spatial symmetry, that symmetry IS the reduction; add ``sz=`` if you
        # also want a magnetisation sector.
        #
        # MPI / MPI+GPU lanes are excluded from auto-Sz: the standalone
        # ed_distributed_main binary loads from full-Hilbert .dat files
        # only; the Sz projection happens inside the MPI driver via the
        # `sz=` argument that gets forwarded through the launcher.
        half = num_sites // 2
        op_to_use = H.make_fixed_sz(int(half))
        sz = half
        if verbose:
            try:
                sec = math.comb(num_sites, half)
            except Exception:  # pragma: no cover - defensive against odd N
                sec = op_to_use.dimension
            print(f"[qed.solve] auto-Sz: n_up={half}: dim={sec} "
                  f"(reduced from {base_dim}). Pass auto_sz=False to "
                  f"opt out, or sz=k for a different sector.")

    sector_dim = int(op_to_use.dimension)

    # ------------------------------------------------------------------
    # 2. Resolve solver + device + flags.
    # ------------------------------------------------------------------
    method = _resolve_solver(solver, num_eigenvalues, sector_dim)
    use_gpu, use_mpi = _resolve_device(device, sector_dim)
    is_thermal = _is_thermal_method(method)
    is_tpq = _is_tpq_method(method)

    # TPQ relies on a single random state evolving on the full Hilbert
    # space (or on a single fixed-Sz block); per-sector symmetry
    # projection breaks that -- reject explicitly.
    if symmetry is not None and is_tpq:
        raise ValueError(
            "qed.solve(H, solver='mTPQ', symmetry=...) is not supported: "
            "TPQ acts on a single random state across the whole sector "
            "and per-symmetry-block diagonalisation does not factor "
            "through the streaming kernel. Options: drop the symmetry= "
            "argument (TPQ + sz= is supported), or use a different "
            "thermal method (FTLM/LTLM combine across symmetry blocks "
            "correctly)."
        )

    # ------------------------------------------------------------------
    # 3. Output directory for thermal methods.
    #     Eigenvalue solvers are happy with output_dir="" (the
    #     `isDisabledOutputPath` shortcut). Thermal solvers MUST write
    #     SS_rand*.dat trajectories to disk, so an empty path causes
    #     silent data loss. Auto-create a fresh directory whose path
    #     we surface to the user.
    # ------------------------------------------------------------------
    effective_output = output_dir
    if is_thermal and not output_dir:
        # Optimization (Jul 2026): the unified thermal kernels return
        # trajectories + thermodynamics in memory; nothing needs the
        # historical auto-created qed_thermal_<ts>/ directory. Writes
        # happen only when the caller passes output_dir=.
        effective_output = "/dev/null"
    elif is_thermal and output_dir and not output_dir.startswith("/dev/null"):
        # Surface-unification follow-up (May 2026): the orchestrator's
        # `_core.workflows_thermal` does NOT mkdir its `output_dir`.
        # We mirror the historical mkdir-then-write behaviour here so
        # callers can pass a fresh path without manual mkdir.
        # Skip the mkdir for the /dev/null sentinel (benchmarking / testing).
        os.makedirs(output_dir, exist_ok=True)

    if verbose:
        method_name = method.name if hasattr(method, "name") else str(method)
        kind = "thermal" if is_thermal else "eigenvalue"
        print(f"[qed.solve] solver={method_name} ({kind})  "
              f"num_eigenvalues={num_eigenvalues}  "
              f"tolerance={tolerance:g}  use_gpu={use_gpu}  use_mpi={use_mpi}")

    # (Pre-flight planner removed: sensible defaults, no feasibility refusal.)

    # ------------------------------------------------------------------
    # 4. Build EDParameters with sensible default Krylov / thermal sizes.
    # ------------------------------------------------------------------
    params = _make_params(
        num_sites=num_sites,
        num_eigenvalues=num_eigenvalues,
        tolerance=tolerance,
        compute_eigenvectors=compute_eigenvectors,
        max_iterations=max_iterations,
        block_size=block_size,
        sector_dim=sector_dim,
        method=method,
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        sector=sector,
        sz=sz,
        output_dir=effective_output,
        num_samples=num_samples,
        target_beta=target_beta,
        num_temp_points=num_temp_points,
        temp_min=temp_min,
        temp_max=temp_max,
    )
    if extra_params:
        for key, value in extra_params.items():
            if not hasattr(params, key):
                raise AttributeError(
                    f"EDParameters has no field {key!r}; "
                    "call qed.list_diag_parameters() to see every "
                    "available knob (or filter by category, e.g. "
                    "qed.list_diag_parameters('tpq'))."
                )
            setattr(params, key, value)

    # ------------------------------------------------------------------
    # 4.5. Auto-tune family-specific knobs. Sentinel-based fill: only
    #     EDParameters fields still at their struct default get
    #     overwritten, so anything set above by the user (kwargs or
    #     ``extra_params``) passes through untouched. The C++ mirror
    #     of these heuristics lives in the kernel-specific options
    #     structs (e.g. ``FtlmKernelOptions``); the surface-unification
    #     collapse retired the cross-cutting ``ed/auto/diag_tune.h``.
    # (Auto-tuner removed: thermal/eigensolver knobs use the EDParameters
    # struct defaults; override per-call via kwargs or extra_params.)

    # ------------------------------------------------------------------
    # 5. Dispatch. Three branches:
    #     * symmetry path → orchestrator's streaming-symmetry kernel
    #       via ``_core.workflows_solve_streaming_symmetry_directory``
    #       (handles GPU per-sector).
    #     * GPU + no-symmetry → orchestrator with
    #       ``BackendConstraints::allow_gpu = true`` (the orchestrator
    #       builds the right GPUOperator under the hood).
    #     * CPU + no-symmetry → orchestrator with the CPU lane
    #       (the fastest path, no I/O).
    # ------------------------------------------------------------------
    symmetry = resolve_auto_symmetry(op_to_use, symmetry, verbose=verbose,
                                     lattice=lattice)
    if symmetry is not None:
        # Stage 9c: ONE routing decision. point_group='auto' (default)
        # PROJECTS through the factorized little-group engine whenever
        # the call is eigenvalue-only; 'full' requires projection
        # (raises on decline); any decline degrades to the abelian rep
        # lane with star/TR/flip folds. The monolithic SAB engine is a
        # test oracle -- no longer production-routed.
        lane = resolve_projection_lane(
            symmetry, point_group=point_group, consumer="solve",
            eigenvalues_only=(not compute_eigenvectors
                              and sector is None
                              and not is_thermal),
            # an EXPLICIT GPU request is served by the abelian rep lane
            # (the little-group lowest-k engine is CPU); auto-dispatch
            # to GPU must NOT veto the projection -- the projected
            # blocks are far smaller than what the GPU lane would chew.
            # 'full' still projects on CPU as before.
            prefer_abelian=(isinstance(device, str)
                            and device.lower() in ("gpu", "cuda")),
            verbose=verbose)
        if lane.mode == "project":
            _k = int(num_eigenvalues) if num_eigenvalues else 1
            _nu = int(sz) if isinstance(sz, int) else -1
            _sp = -1
            if _sz_parity_str is not None:
                _sp = int(_sz_parity_str)
            elif _nu < 0 and auto_sz:
                # Compose the diagonal axis automatically: fixed Sz for
                # U(1)-conserving H (GS at half filling), the parity
                # half otherwise when the Z2 remnant survives.
                if op_to_use.conserves_sz():
                    _nu = num_sites // 2
                else:
                    try:
                        if bool(_core.detect_hamiltonian_symmetries(
                                op_to_use)["sz_parity"]):
                            _sp = 2
                    except Exception:
                        _sp = -1
            _sf = resolve_discrete_toggle(op_to_use, spin_flip,
                                          "spin_flip", verbose=verbose)
            _tr = resolve_discrete_toggle(op_to_use, time_reversal,
                                          "time_reversal", verbose=verbose)
            _A, _res = lane.A, lane.residues
            try:
                def _lg_block(**kw):
                    # Stage 10c: both paths return LABELED rows
                    # (eigenvalue, k_raw, flip_parity, irrep, irrep_dim,
                    # multiplicity, subspace) so the project lane's
                    # EDResults carries the quantum numbers the engine
                    # always computed. solver='full' wants the exact
                    # spectrum: Lanczos without reorthogonalisation only
                    # converges the extreme pair, so big-k requests go
                    # through the dense per-block full-spectrum path.
                    sub = kw.get("n_up", kw.get("sz_parity", -1))
                    if method == DiagonalizationMethod.FULL:
                        d = dict(_core.little_group_full_spectrum(
                            op_to_use, _A, _res,
                            spin_flip=_sf, time_reversal=_tr, **kw))
                        rows = []
                        for e, m, kk, fp, ir, dd in zip(
                                d["block_values"], d["multiplicities"],
                                d["block_k_raw"], d["block_flip_parity"],
                                d["block_irrep"], d["block_irrep_dim"]):
                            rows.extend(
                                [(float(e), kk, fp, ir, dd, m, sub,
                                  True)] * m)   # full spectrum is exact
                        return rows, d["irrep_characters"]
                    d = dict(_core.little_group_lowest_eigenvalues_labeled(
                        op_to_use, _A, _res, k=_k,
                        spin_flip=_sf, time_reversal=_tr, **kw))
                    rows = [(float(e), kk, fp, ir, dd, m, sub, bool(cv))
                            for e, kk, fp, ir, dd, m, cv in zip(
                                d["eigenvalues"], d["k_raw"],
                                d["flip_parity"], d["irrep"],
                                d["irrep_dim"], d["multiplicity"],
                                d["converged"])]
                    return rows, d["irrep_characters"]

                if _sp == 2:
                    rows, chars = _lg_block(sz_parity=0)
                    rows2, _ = _lg_block(sz_parity=1)
                    rows += rows2
                elif _sp in (0, 1):
                    rows, chars = _lg_block(sz_parity=_sp)
                else:
                    rows, chars = _lg_block(n_up=_nu)
                rows.sort(key=lambda t: t[0])
                if method != DiagonalizationMethod.FULL:
                    rows = rows[:_k]
                out = EDResults()
                out.eigenvalues = [t[0] for t in rows]
                # Stage 10c contract: per-eigenvalue labels on the
                # PROJECT lane (engine irrep-index convention; decode the
                # momentum from irrep_characters via chi_k(T)). The
                # abelian lane provides sector_tags /
                # eigenvalues_per_sector instead -- see the point_group
                # docstring for the full table.
                out.block_k_raw        = [t[1] for t in rows]
                out.block_flip_parity  = [t[2] for t in rows]
                out.block_irrep        = [t[3] for t in rows]
                out.block_irrep_dim    = [t[4] for t in rows]
                out.block_multiplicity = [t[5] for t in rows]
                out.block_subspace     = [t[6] for t in rows]
                # 1b (Jul 2026): per-eigenvalue convergence flag -- a
                # budget-capped Lanczos block that could not deliver its
                # full window of distinct converged Ritz values marks its
                # rows False; merged campaigns must be able to tell.
                out.block_converged    = [t[7] for t in rows]
                out.irrep_characters   = chars
                if verbose:
                    print(f"[qed.solve] non-abelian LITTLE-GROUP lane "
                          f"(factorized): |A| = {len(_A)}, residues = "
                          f"{len(_res)}.")
                return out
            except Exception as exc:      # noqa: BLE001 -- graceful
                if isinstance(point_group, str) \
                        and point_group.lower() == "full":
                    raise
                if verbose:
                    print(f"[qed.solve] little-group lane declined "
                          f"({exc}); falling back to the abelian rep "
                          "lane.")
        return _diag_with_symmetry(
            op_to_use, symmetry, params, method,
            sz=sz if sz is not None else None,
            sz_parity=_sz_parity_str,
            auto_sz_axis=auto_sz,
            verbose=verbose,
            spin_flip=spin_flip,
            time_reversal=time_reversal,
            point_group=point_group,
        )


    if use_gpu:
        return _diag_via_directory(op_to_use, method, params, verbose=verbose)

    # solver=None => let the orchestrator pick the ground-state eigensolver
    # default (full diag for tiny dims, Lanczos otherwise).
    return _diag_via_workflows_solve(op_to_use, method, params,
                                     auto_method=(solver is None))


# ---------------------------------------------------------------------------
# list_diag_parameters
# ---------------------------------------------------------------------------


# Curated grouping of EDParameters fields so the introspection helper can
# print them organised by physical purpose rather than alphabetically.
# Anything in EDParameters that doesn't appear here lands under "other".
_PARAMETER_CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("general", "Eigenvalue / convergence basics", [
        "num_eigenvalues", "tolerance", "max_iterations",
        "compute_eigenvectors", "output_dir",
    ]),
    ("krylov", "Lanczos / Krylov-Schur subspace shape", [
        "block_size",
    ]),
    ("device", "Device & parallelism axes (orthogonal flags)", [
        "use_gpu", "use_mpi", "use_symmetry",
        "use_fixed_sz", "n_up", "translation_only", "full_sz_split",
    ]),
    ("ftlm", "Finite-Temperature Lanczos Method", [
        "ftlm_krylov_dim", "ftlm_full_reorth", "ftlm_reorth_freq",
        "ftlm_seed", "ftlm_store_samples", "ftlm_error_bars",
    ]),
    ("ltlm", "Low-Temperature Lanczos Method", [
        "ltlm_krylov_dim", "ltlm_ground_krylov", "ltlm_full_reorth",
        "ltlm_reorth_freq", "ltlm_seed", "ltlm_store_data",
    ]),
    ("tpq", "Thermal Pure Quantum / mTPQ imaginary-time evolution", [
        "tpq_max_steps", "tpq_measurement_interval",
        "tpq_energy_shift", "tpq_beta_max", "tpq_delta_beta",
        "tpq_taylor_order", "tpq_continue", "tpq_continue_sample",
        "tpq_continue_beta", "tpq_target_beta",
        "tpq_num_measure_points", "tpq_measure_beta_min",
        "tpq_measure_beta_max",
    ]),
    ("thermal", "Thermal post-processing grid (FTLM/LTLM/TPQ)", [
        "num_samples", "temp_min", "temp_max", "num_temp_bins",
        "save_thermal_states", "compute_spin_correlations",
    ]),
    ("observables", "Spectral / dynamical observables", [
        "omega_min", "omega_max", "num_points", "t_end", "dt",
        "observables", "observable_names",
    ]),
    ("lattice", "Lattice metadata (mostly informational)", [
        "num_sites", "spin_length", "sublattice_size",
        "selected_sectors",
    ]),
]


# ---------------------------------------------------------------------------
# Solver x Device compatibility introspection
#
# Static metadata (which solvers have which device kernels) plus a
# build-aware "is this actually reachable on the current build?" check.
# ---------------------------------------------------------------------------

# Per (solver_family, device) cell: True if the C++ side has a kernel
# wired for that combination, False if there's no such kernel.
# "device" axis values:
#   "cpu"     -> single-process CPU
#   "gpu"     -> single GPU (cuSPARSE / per-sector dispatch)
#   "mpi"     -> distributed CPU via ed_distributed_main
#   "mpi_gpu" -> distributed CPU + per-rank GPU (multi-GPU)
#
# Coverage current as of Phase 9: the table mirrors what
# ed/core/ed_method_traits.h + include/ed/distributed/ + the
# ed_distributed_main CLI actually expose.
_SOLVER_DEVICE_KERNELS: dict[str, dict[str, bool]] = {
    "LANCZOS":         {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "BLOCK_LANCZOS":   {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "KRYLOV_SCHUR":    {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "FULL":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "mTPQ":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "FTLM":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "OFTLM":           {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "LTLM":            {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    "KPM_DOS":         {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
}


def solver_device_support(
    *,
    solver: Optional[str] = None,
    return_dict: bool = False,
) -> Optional[dict[str, dict[str, dict[str, Any]]]]:
    """Inspect which (solver, device) cells are reachable on this build.

    The compatibility matrix has two layers:

    * **kernel** -- whether a C++ kernel exists for the cell at all
      (set at compile time of the C++ side).
    * **build** -- whether THIS python build can reach it
      (``WITH_CUDA`` / ``WITH_MPI`` / ``WITH_NCCL`` flags). For
      example, the LANCZOS-on-GPU cell exists in C++, but a build
      with ``WITH_CUDA=OFF`` cannot reach it.

    Without a ``solver`` argument the function prints a table; pass
    ``return_dict=True`` to get the matrix back as nested dicts of
    ``{solver: {device: {"kernel": bool, "available": bool, "note":
    str}}}``.

    Parameters
    ----------
    solver : str, optional
        Filter to one solver family (e.g. ``"LANCZOS"``,
        ``"KRYLOV_SCHUR"``, ``"mTPQ"``). Substring match is allowed.
    return_dict : bool, optional
        If True, suppress printing and return the structured matrix.

    Returns
    -------
    dict or None
        When ``return_dict=True``, the nested support matrix.

    Notes
    -----
    The MPI subprocess cells were retired in Stage 11d (Jul 2026);
    MPI runs go through the CLI under mpirun (SectorDistributor +
    MpiBackend).
    """
    cuda_ok = bool(has_cuda_build())
    mpi_ok = bool(has_mpi_build())

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for solver_name, devices in _SOLVER_DEVICE_KERNELS.items():
        if solver is not None and solver.upper() not in solver_name.upper():
            continue
        cells: dict[str, dict[str, Any]] = {}
        for device, has_kernel in devices.items():
            if not has_kernel:
                cells[device] = {
                    "kernel": False,
                    "available": False,
                    "note": "no C++ kernel for this combination",
                }
                continue
            if device == "cpu":
                cells[device] = {"kernel": True, "available": True, "note": ""}
            elif device == "gpu":
                cells[device] = {
                    "kernel": True,
                    "available": cuda_ok,
                    "note": ("" if cuda_ok
                             else "build has WITH_CUDA=OFF; rebuild with "
                                  "-DWITH_CUDA=ON"),
                }
            elif device in ("mpi", "mpi_gpu"):
                cells[device] = {
                    "kernel": False,
                    "available": False,
                    "note": ("retired (Stage 11d): run the CLI under "
                             "mpirun -- SectorDistributor + MpiBackend"),
                }
        matrix[solver_name] = cells

    if return_dict:
        return matrix

    print(f"Build flags: WITH_CUDA={'ON' if cuda_ok else 'OFF'}, "
          f"WITH_MPI={'ON' if mpi_ok else 'OFF'}")
    print()
    devices = ["cpu", "gpu", "mpi", "mpi_gpu"]
    header = f"{'solver':<22}" + "".join(f"{d:>11}" for d in devices)
    print(header)
    print("-" * len(header))
    for solver_name, cells in matrix.items():
        row = f"{solver_name:<22}"
        for d in devices:
            cell = cells[d]
            if not cell["kernel"]:
                row += f"{'-':>11}"
            elif cell["available"]:
                row += f"{'OK':>11}"
            else:
                row += f"{'(unbuilt)':>11}"
        print(row)
    print()
    print("Legend:  OK = wired and reachable on this build;")
    print("         (unbuilt) = C++ kernel exists but this build is missing")
    print("                     the WITH_CUDA / WITH_MPI flag;")
    print("         -  = no C++ kernel for this (solver, device) combination.")
    return None


def list_diag_parameters(
    category: Optional[str] = None,
    *,
    return_dict: bool = False,
) -> Optional[dict[str, list[tuple[str, Any]]]]:
    """Print (or return) every parameter accepted by :func:`solve` via
    ``extra_params=...``.

    Most users only need the keyword arguments :func:`solve` exposes
    directly (``num_eigenvalues``, ``tolerance``, ``solver``,
    ``device``, ``symmetry``, ``sz``, ``output_dir``,
    ``compute_eigenvectors``, ``max_iterations``, ``block_size``).
    Everything else lives on :class:`EDParameters` and is reachable
    via the ``extra_params`` dict; this helper lists those fields
    with their defaults, organised by physical purpose.

    Parameters
    ----------
    category : str, optional
        Filter to a single category. One of ``"general"``,
        ``"krylov"``, ``"device"``, ``"ftlm"``, ``"ltlm"``,
        ``"tpq"``, ``"thermal"``, ``"observables"``, ``"lattice"``,
        ``"other"``. Substring matches are accepted.
    return_dict : bool, optional
        If True, return the catalog as a dict instead of printing.
        Useful for programmatic discovery (e.g. autocomplete in a
        Jupyter notebook).

    Returns
    -------
    dict[str, list[tuple[str, Any]]] or None
        Mapping ``category -> [(field_name, default_value), ...]``
        when ``return_dict=True``, else ``None``.

    Examples
    --------
    Browse every knob:

    .. code-block:: python

        qed.list_diag_parameters()

    Just the FTLM section:

    .. code-block:: python

        qed.list_diag_parameters("ftlm")

    Use a niche knob via ``extra_params``:

    .. code-block:: python

        eigs = qed.solve(
            H,
            num_eigenvalues=6,
            solver="FTLM",
            extra_params={
                "ftlm_seed": 12345,        # only relevant if method=FTLM
            },
        ).eigenvalues
    """
    defaults = EDParameters()
    bound_fields = {
        name for name in dir(defaults)
        if not name.startswith("_")
        and not callable(getattr(defaults, name))
    }

    catalog: dict[str, list[tuple[str, Any]]] = {}
    seen: set[str] = set()
    for cat_name, cat_desc, fields in _PARAMETER_CATEGORIES:
        rows: list[tuple[str, Any]] = []
        for name in fields:
            if name in bound_fields:
                rows.append((name, getattr(defaults, name)))
                seen.add(name)
        if rows:
            catalog[cat_name] = rows

    leftovers = sorted(bound_fields - seen)
    if leftovers:
        catalog["other"] = [(n, getattr(defaults, n)) for n in leftovers]

    if category is not None:
        key = category.lower()
        matches = [k for k in catalog if key in k]
        if not matches:
            raise KeyError(
                f"No parameter category matching {category!r}. "
                f"Available: {sorted(catalog)}"
            )
        catalog = {k: catalog[k] for k in matches}

    if return_dict:
        return catalog

    descriptions = {name: desc for name, desc, _ in _PARAMETER_CATEGORIES}
    print(
        "EDParameters fields (pass any of these via "
        "qed.solve(..., extra_params={...})):"
    )
    for cat_name, rows in catalog.items():
        title = descriptions.get(cat_name, "")
        header = f"[{cat_name}]" + (f"  -- {title}" if title else "")
        print()
        print(header)
        for name, value in rows:
            print(f"  {name:<36s} = {value!r}")
    print()
    print(
        "Note: the most common knobs are first-class kwargs of "
        "qed.solve(...). Use extra_params={...} only for the niche "
        "fields above."
    )
    return None


# ===========================================================================
# Internal helpers
# ===========================================================================


# ---------------------------------------------------------------------------

def _thermal_method_names() -> set[str]:
    """Names of every TPQ / FTLM / LTLM / KPM_DOS variant.

    ``KPM_DOS`` belongs here too -- it produces a full
    ``ThermodynamicData`` block keyed off the same ``temp_min`` /
    ``temp_max`` / ``num_temp_bins`` grid as the random-vector
    methods.
    """
    return {
        "mTPQ",
        "FTLM",
        "LTLM",
        "KPM_DOS",
    }


def _is_thermal_method(method: DiagonalizationMethod) -> bool:
    return method.name in _thermal_method_names()


def _is_tpq_method(method: DiagonalizationMethod) -> bool:
    return method.name.lower().endswith("tpq") or "TPQ" in method.name


def _resolve_solver(
    solver: Optional[Union[str, DiagonalizationMethod]],
    num_eigenvalues: int,
    dim: int,
) -> DiagonalizationMethod:
    """Pick a default solver, or canonicalize a user-supplied one.

    String lookup is case-insensitive: ``"lanczos"`` / ``"LANCZOS"`` /
    ``"Lanczos"`` all resolve to ``DiagonalizationMethod.LANCZOS``. The
    TPQ enum names are mixed-case in C++ (``mTPQ``); we
    accept them in any case so users don't have to memorise the
    spelling.
    """
    if solver is not None:
        if isinstance(solver, DiagonalizationMethod):
            return solver
        if isinstance(solver, str):
            members = DiagonalizationMethod.__members__
            # Try the literal spelling first (handles mTPQ vs MTPQ).
            if solver in members:
                return members[solver]
            upper = solver.upper()
            if upper in members:
                return members[upper]
            # Build a case-folded lookup for the rest (handles "mtpq",
            # etc. against the mixed-case enum keys).
            folded = {name.casefold(): name for name in members}
            key = solver.casefold()
            if key in folded:
                return members[folded[key]]
            raise ValueError(
                f"Unknown solver name {solver!r}. "
                "Pass DiagonalizationMethod.<NAME> or one of "
                f"{sorted(members)}."
            )
        raise TypeError(f"solver must be str or DiagonalizationMethod, "
                        f"got {type(solver).__name__}")

    # solver is None: the ground-state eigensolver is chosen by the C++ dictator
    # (ed::planner); qed.solve passes SolveMethod::Auto (auto_method=True). This
    # value is only a ROUTING placeholder so the caller dispatches to the
    # ground-state lane (workflows_solve) rather than the thermal lane. The old
    # dim/num_eigs heuristic here DISAGREED with the planner (it never picked
    # BlockLanczos), so it was removed -- one dictator.
    return DiagonalizationMethod.LANCZOS


def _resolve_device(device: Optional[str], dim: int) -> tuple[bool, bool]:
    """Pick (use_gpu, use_mpi).

    Single-GPU is honoured for any solver the in-process build supports,
    via a temp-dir routing in :func:`solve` (see ``_diag_via_directory``).
    ``device='mpi'``/``'mpi_gpu'`` (the retired subprocess launcher)
    raise with guidance.

    Returns ``(use_gpu, use_mpi)``; ``use_mpi`` is always False now.
    """
    if device is None or device == "auto":
        use_gpu = bool(has_cuda_build()) and dim >= (1 << 14)
        return use_gpu, False
    device_lc = device.lower()
    if device_lc == "cpu":
        return False, False
    if device_lc == "gpu":
        if not has_cuda_build():
            raise RuntimeError(
                "device='gpu' requested but this build of qed._core "
                "does not have WITH_CUDA=ON. Rebuild with -DWITH_CUDA=ON or "
                "use device='cpu'."
            )
        return True, False
    if device_lc in ("mpi", "mpi_gpu"):
        raise RuntimeError(
            "device='mpi' / 'mpi_gpu' was retired (Stage 11d, Jul 2026): "
            "the subprocess launcher (ed_distributed_main + qed.mpi) and "
            "the distributed-operator family behind it were removed. "
            "For MPI runs, launch the CLI under mpirun -- across-sector "
            "distribution (SectorDistributor) engages automatically for "
            "symmetry workloads, and the in-process MpiBackend covers "
            "reduction parallelism. Single-node frontier runs use "
            "device='gpu' (fp32 mTPQ / rep-lane memory scaling)."
        )
    raise ValueError(
        f"device={device!r} not in "
        "{'auto', 'cpu', 'gpu', 'mpi', 'mpi_gpu'}."
    )


def _make_params(
    *,
    num_sites: int,
    num_eigenvalues: int,
    tolerance: float,
    compute_eigenvectors: bool,
    max_iterations: Optional[int],
    block_size: Optional[int],
    sector_dim: int,
    method: DiagonalizationMethod,
    use_gpu: bool,
    use_mpi: bool,
    sector: Optional[Sequence[int]],
    sz: Optional[int],
    output_dir: str,
    # Thermal-method first-class kwargs (only consulted when method is
    # thermal; ignored for eigenvalue solvers).
    num_samples: Optional[int] = None,
    target_beta: Optional[float] = None,
    num_temp_points: Optional[int] = None,
    temp_min: Optional[float] = None,
    temp_max: Optional[float] = None,
) -> EDParameters:
    """Compose an EDParameters with auto-tuned Krylov / thermal sizes."""
    p = EDParameters()
    p.num_sites = num_sites
    p.num_eigenvalues = max(1, int(num_eigenvalues))
    p.tolerance = float(tolerance)
    p.compute_eigenvectors = bool(compute_eigenvectors)
    p.output_dir = output_dir
    p.use_gpu = bool(use_gpu)
    p.use_mpi = bool(use_mpi)
    if sz is not None:
        p.use_fixed_sz = True
        p.n_up = int(sz)

    if _is_thermal_method(method):
        # ---- Thermal solvers (TPQ / FTLM / LTLM / KPM_DOS) ----
        # These don't extract eigenvalues from a Krylov subspace; they
        # build thermodynamic averages from random-state imaginary-time
        # trajectories (TPQ) or from Lanczos micro-bases (FTLM/LTLM).
        # The relevant knobs are different.
        if num_samples is not None:
            p.num_samples = int(num_samples)
        elif p.num_samples < 1:
            p.num_samples = 1
        if target_beta is not None:
            p.tpq_target_beta = float(target_beta)
        if num_temp_points is not None:
            p.tpq_num_measure_points = int(num_temp_points)
            p.num_temp_bins = int(num_temp_points)
        if temp_min is not None:
            p.temp_min = float(temp_min)
            p.tpq_measure_beta_max = 1.0 / float(temp_min) if temp_min > 0 \
                else p.tpq_measure_beta_max
        if temp_max is not None:
            p.temp_max = float(temp_max)
            p.tpq_measure_beta_min = 1.0 / float(temp_max) if temp_max > 0 \
                else p.tpq_measure_beta_min
        # TPQ imaginary-time step count: honour an explicit max_iterations /
        # tpq_max_steps, else a BOUNDED default. (The old code derived
        # target_beta/delta_beta, which exploded to ~1e5 steps for a low-T
        # grid and timed out at large dimension. Raise max_iterations for
        # deeper/finer cooling.)
        DEFAULT_TPQ_STEPS = 1000
        if max_iterations is not None:
            p.tpq_max_steps = int(max_iterations)
            p.max_iterations = int(max_iterations)
        else:
            steps = int(getattr(p, "tpq_max_steps", 0) or 0) or DEFAULT_TPQ_STEPS
            p.tpq_max_steps = steps
            p.max_iterations = steps
        if sector is not None:
            p.selected_sectors = list(int(q) for q in sector)
        return p

    # ---- Eigenvalue solvers ----
    # Auto-tuned Krylov sizes. Heuristic: enough headroom that the
    # requested num_eigenvalues converge to `tolerance` without the
    # caller having to think about it. ``max_iterations`` now doubles
    # as the Krylov subspace dimension (the legacy ``max_subspace``
    # was retired together with the ARPACK / Davidson / LOBPCG
    # solvers that used it).
    n_eigs = p.num_eigenvalues
    auto_iter = max(200, 8 * n_eigs + 80)
    if sector_dim > 1:
        auto_iter = min(auto_iter, sector_dim - 1)
    p.max_iterations = int(max_iterations) if max_iterations is not None \
        else auto_iter
    if block_size is not None:
        p.block_size = int(block_size)
    elif method == DiagonalizationMethod.BLOCK_LANCZOS:
        p.block_size = max(1, min(n_eigs, 4))

    if sector is not None:
        p.selected_sectors = list(int(q) for q in sector)
    return p


# ---------------------------------------------------------------------------
# Symmetry-path: write the operator + symmetry to a temp dir and call
# the streaming kernel.
# ---------------------------------------------------------------------------


def _diag_via_directory(
    operator: Operator,
    method: DiagonalizationMethod,
    params: EDParameters,
    *,
    verbose: bool,
) -> EDResults:
    """Route a GPU request for an in-memory Operator through the orchestrator.

    The unified C++ orchestrator (`ed::workflows::solve` /
    `ed::select_backend`) already knows how to pick the GPU lane when
    `BackendConstraints.allow_gpu = true`, so we no longer need the
    legacy temp-directory GPU dispatch path. We simply hand the
    operator to `_diag_via_workflows_solve`; the orchestrator's
    backend selector picks GPU when the build supports it and the
    operator's geometry asks for it.
    """
    if verbose:
        print(f"[qed.solve] GPU dispatch via _core.workflows_solve "
              f"(backend selection: allow_gpu=True)")
    return _diag_via_workflows_solve(operator, method, params)


# ---------------------------------------------------------------------------
# (Stage 11d, Jul 2026: the device='mpi' subprocess lane -- _diag_via_mpi,
# the ed_distributed_main launcher plumbing, the rank_<r>.h5 slab readers
# load_mpi_eigenvector(s), and the per-sector Z-weighted aggregation that
# only that lane produced -- was retired with the distributed-operator
# family. MPI runs go through the CLI under mpirun: SectorDistributor
# across sectors + MpiBackend in-process.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def _diag_with_symmetry(
    operator: Operator,
    symmetry: SymmetryArg,
    params: EDParameters,
    method: DiagonalizationMethod,
    *,
    sz: Optional[int],
    verbose: bool,
    spin_flip="auto",
    time_reversal="auto",
    point_group="auto",
    sz_parity: Optional[int] = None,
    auto_sz_axis: bool = True,
) -> EDResults:
    """Route a symmetry-projected diagonalisation through the C++
    streaming-symmetry pipeline.

    Conceptually this is the Python facade for the
    ``(Subspace, ProjectorChain)`` composition introduced in the
    "Orthogonal symmetry composition" wave (May 2026; see
    ``include/ed/symmetry/{subspace,projector,projector_chain}.h``).
    The four legacy "modes" map onto the new decomposition as:

        mode "none"      -> (FullSpaceSubspace, [])
        mode "Sz"        -> (FixedSzSubspace,   [])
        mode "Symm"      -> (FullSpaceSubspace, [SpatialProjector])
        mode "Sz+Symm"   -> (FixedSzSubspace,   [SpatialProjector])

    The kwargs to ``qed.solve`` are already orthogonal: ``sz=`` (or
    passing a ``FixedSzOperator``) selects the subspace; ``symmetry=``
    populates the chain with the spatial projector. Future axes (Z_2
    spin-flip, time reversal, SU(2) total-S) drop in by extending the
    chain or the subspace; no change to the Python signature is
    required when those axes land.
    """
    # ------------------------------------------------------------------
    # 1. Normalise the symmetry argument into (generators, info_dict).
    # ------------------------------------------------------------------
    from .symmetry import group_from_generators

    if isinstance(symmetry, GeneratorSet):
        gens = symmetry.generators
        if not gens:
            # Empty generators ⇔ trivial; no symmetry projection needed.
            # Full Unified-Interface Collapse, Wave E1 (May 2026): route
            # through the orchestrator (`_core.workflows_solve`) so the
            # trivial-symmetry path lands on the unified surface too.
            params.use_symmetry = False
            return _diag_via_workflows_solve(operator, method, params)
        info = group_from_generators(int(operator.num_sites), gens)
    elif isinstance(symmetry, dict):
        info = symmetry
    elif isinstance(symmetry, (list, tuple)):
        gens = [list(map(int, p)) for p in symmetry]
        if not gens:
            params.use_symmetry = False
            return _diag_via_workflows_solve(operator, method, params)
        info = group_from_generators(int(operator.num_sites), gens)
    else:
        raise TypeError(
            f"symmetry must be GeneratorSet, list[Permutation], or dict, "
            f"got {type(symmetry).__name__}"
        )

    params.use_symmetry = True

    # ------------------------------------------------------------------
    # 2. Materialise operator + symmetry into a temp directory the
    #    streaming kernel will read.
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="qed_diag_symm_")
    try:
        _write_operator_directory(operator, tmpdir)
        _write_symmetry_directory(tmpdir, info)

        if verbose:
            print(f"[qed.solve] symmetry projection: |G|="
                  f"{len(info.get('max_clique', []))}, "
                  f"sectors={len(info.get('sectors', []))}, "
                  f"tmpdir={tmpdir}")

        # Route through the unified orchestrator's streaming-symmetry
        # helper. It composes `ed::make_operator(streaming_symmetry=true)`
        # with a per-sector `ed::workflows::solve` loop -- the same
        # CLI path the C++ `run_streaming_symmetry_workflow` exercises.
        #
        # Phase B of the "Backend x Symmetries x Workflows" plan
        # (May 2026): thermal methods (FTLM / LTLM / mTPQ /
        # KPM_DOS) now route through the matching
        # ``workflows_thermal_streaming_symmetry_directory`` binding,
        # closing the "qed.solve(symmetry=..., solver='FTLM')" gap.
        fixed_sz_n_up = None
        # Sz-parity mode: explicit via sz="even"/"odd", or AUTO when the
        # Hamiltonian breaks U(1) but keeps the Z2 remnant (-1)^{n_up}
        # (all terms change n_up by even amounts): both halves in one
        # sector set.
        _parity_mode = sz_parity
        if (_parity_mode is None and sz is None and auto_sz_axis
                and not isinstance(operator, FixedSzOperator)
                and not operator.conserves_sz()):
            try:
                _det = _core.detect_hamiltonian_symmetries(operator)
                if bool(_det["sz_parity"]):
                    _parity_mode = 2          # both halves
                    if verbose:
                        print("[qed] Sz axis: U(1) broken but parity "
                              "(-1)^{n_up} conserved -> parity-half "
                              "sectors engage.")
            except Exception:
                _parity_mode = None
        if isinstance(operator, FixedSzOperator):
            if sz is None:
                if params.n_up < 0:
                    raise RuntimeError(
                        "internal: FixedSzOperator passed without n_up. "
                        "Use sz= in qed.solve(...) so the streaming kernel "
                        "knows the sector."
                    )
                sz = int(params.n_up)
            fixed_sz_n_up = int(sz)

        if _is_thermal_method(method):
            topts = _ed_params_to_thermal_options(params, method)
            # ``ThermalOptions`` carries no use_symmetry / use_fixed_sz
            # flags -- those live on the OperatorSpec the binding
            # builds internally (streaming_symmetry=true,
            # fixed_sz=fixed_sz_n_up). So we just hand it the temp
            # directory + sites + spin_l and the binding takes care of
            # composing the per-sector thermal lane.
            if _parity_mode is not None:
                topts.sz_parity = int(_parity_mode)
            topts.spin_flip = resolve_discrete_toggle(
                operator, spin_flip, "spin_flip", verbose=verbose)
            topts.time_reversal = resolve_discrete_toggle(
                operator, time_reversal, "time_reversal", verbose=verbose)
            tr = _core.workflows_thermal_streaming_symmetry_directory(
                tmpdir,
                int(operator.num_sites),
                float(params.spin_length),
                topts,
                fixed_sz_n_up,
            )
            return _ed_result_from_thermal_result(tr)

        # Ground-state lane (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR /
        # FULL) -- the original behaviour.
        opts = _ed_params_to_solve_options(params, method)
        opts.use_symmetry = True
        # Stage 8 composition toggles: -1 auto / 0 off / 1 require,
        # with 'on' = auto + detection report (warn-and-continue when
        # the Hamiltonian lacks the symmetry).
        opts.spin_flip = resolve_discrete_toggle(
            operator, spin_flip, "spin_flip", verbose=verbose)
        opts.time_reversal = resolve_discrete_toggle(
            operator, time_reversal, "time_reversal", verbose=verbose)
        if _parity_mode is not None:
            opts.sz_parity = int(_parity_mode)
        # Stage 7a: star reduction. The non-abelian residue of the
        # spatial group permutes the abelian irreps; related sectors
        # are isospectral, so the C++ plan solves one representative
        # per orbit and copies the spectrum to its partners.
        _star = getattr(symmetry, "star_perms", None) or []
        if _star and point_group not in (False, 0, "off", "none"):
            from .star_reduction import star_maps_from_info
            _maps = star_maps_from_info(info, _star)
            if _maps:
                opts.star_maps = _maps
                if verbose:
                    print(f"[qed] point group: {len(_maps)} residue "
                          "automorphisms fold the irrep sectors into "
                          "isospectral stars (solve one per star).")
        if fixed_sz_n_up is not None:
            opts.use_fixed_sz = True
            opts.n_up         = fixed_sz_n_up
        gs = _core.workflows_solve_streaming_symmetry_directory(
            tmpdir,
            int(operator.num_sites),
            float(params.spin_length),
            opts,
            fixed_sz_n_up,
        )
        return _ed_result_from_gs_result(gs, params)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _operator_conserves_sz(operator: Operator) -> bool:
    """True iff every Hamiltonian term commutes with total Sz.

    Each elementary operator changes total Sz by op_type 0 (S+) -> +1,
    1 (S-) -> -1, 2 (Sz) -> 0. A term conserves Sz iff the net change
    over its factors is zero. Used to decide whether
    :func:`full_spectrum` can decompose by magnetisation (loop n_up)
    in addition to the spatial irreps. Conservative: any parse failure
    -> ``False`` (treat as non-conserving, fall back to the full span).
    """
    # Prefer the operator's own predicate when present (authoritative).
    try:
        return bool(operator.conserves_sz())
    except Exception:
        pass
    _delta = {0: 1, 1: -1, 2: 0}
    try:
        for op_type, _site, _c in operator.iter_one_body_terms():
            if _delta.get(int(op_type), 1) != 0:
                return False
        for op1, _s1, op2, _s2, _c in operator.iter_two_body_terms():
            if _delta.get(int(op1), 1) + _delta.get(int(op2), 1) != 0:
                return False
        for op1, _s1, op2, _s2, op3, _s3, _c in operator.iter_three_body_terms():
            if (_delta.get(int(op1), 1) + _delta.get(int(op2), 1)
                    + _delta.get(int(op3), 1) != 0):
                return False
    except Exception:
        return False
    return True


def _normalize_symmetry_info(
    operator: Operator, symmetry: SymmetryArg
) -> Optional[dict[str, Any]]:
    """Normalise a symmetry argument into the group ``info`` dict the
    streaming kernel reads, or ``None`` for the trivial (no spatial
    symmetry) case."""
    from .symmetry import group_from_generators

    if symmetry is None:
        return None
    if isinstance(symmetry, GeneratorSet):
        gens = symmetry.generators
        return (group_from_generators(int(operator.num_sites), gens)
                if gens else None)
    if isinstance(symmetry, dict):
        return symmetry
    if isinstance(symmetry, (list, tuple)):
        gens = [list(map(int, p)) for p in symmetry]
        return (group_from_generators(int(operator.num_sites), gens)
                if gens else None)
    raise TypeError(
        f"symmetry must be GeneratorSet, list[Permutation], or dict, "
        f"got {type(symmetry).__name__}"
    )


def _raw_generators(symmetry: SymmetryArg) -> Optional[list[list[int]]]:
    """The UNRESTRICTED site-permutation generators of ``symmetry`` (before
    the abelian-subgroup guard in ``group_from_generators``). Used to decide
    whether the spatial group is non-abelian and to feed split_nonabelian
    (Stage 9c; previously the monolithic SAB engine)."""
    if symmetry is None:
        return None
    if isinstance(symmetry, GeneratorSet):
        gens = symmetry.generators
    elif isinstance(symmetry, (list, tuple)):
        gens = symmetry
    elif isinstance(symmetry, dict):
        gens = symmetry.get("generators")
    else:
        return None
    return [list(map(int, p)) for p in gens] if gens else None


def _generators_nonabelian(gens: list[list[int]]) -> bool:
    """True iff the group generated by ``gens`` is non-abelian. A group
    generated by pairwise-commuting elements is abelian, so checking the
    generators pairwise suffices (composition convention is immaterial for
    a commutativity test)."""
    if not gens or len(gens) < 2:
        return False
    def comp(a: list[int], b: list[int]) -> list[int]:
        return [a[b[i]] for i in range(len(b))]
    for i in range(len(gens)):
        for j in range(i + 1, len(gens)):
            if comp(gens[i], gens[j]) != comp(gens[j], gens[i]):
                return True
    return False


def full_spectrum(
    operator: Operator,
    *,
    symmetry: SymmetryArg = None,
    sz_conserved: Optional[bool] = None,
    spin_length: float = 0.5,
    device: str = "cpu",
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    point_group: Union[str, bool, None] = "auto",
    lattice: Optional[Any] = None,
    verbose: bool = False,
) -> EDResults:
    """Compute the COMPLETE eigenvalue spectrum of ``operator`` decomposed
    by all available ``(Sz x spatial)`` symmetries.

    Each ``(n_up, spatial irrep)`` block is dense-diagonalised through the
    memory-light on-the-fly representative SpMV
    (``CpuMatVecBackend<RepSymmetryBasisPolicy>`` on CPU, the GPU rep
    mirror on CUDA), and the full multiset of eigenvalues is collected and
    sorted. The result is mathematically identical to
    ``numpy.linalg.eigvalsh`` of the dense Hamiltonian, but never builds
    the full ``2^N x 2^N`` matrix.

    Parameters
    ----------
    operator : Operator
        The spin Hamiltonian (full Hilbert space).
    symmetry : GeneratorSet | list[Permutation] | dict | None
        Spatial symmetry generators. ``None`` => no spatial symmetry
        (still decomposes by Sz when conserved).
    sz_conserved : bool | None
        Whether the model conserves total Sz. ``None`` (default)
        auto-detects from the term list.
    spin_length : float
        Spin magnitude (0.5 for spin-1/2).

    Returns
    -------
    EDResults
        ``.eigenvalues`` is the complete sorted spectrum (every
        eigenvalue with its multiplicity).
    """
    import math

    N = int(operator.num_sites)
    symmetry = resolve_auto_symmetry(operator, symmetry, verbose=verbose,
                                     lattice=lattice)
    info = _normalize_symmetry_info(operator, symmetry)
    if sz_conserved is None:
        sz_conserved = _operator_conserves_sz(operator)
    use_gpu = isinstance(device, str) and device.lower() in ("gpu", "cuda")
    _sf = resolve_discrete_toggle(operator, spin_flip, "spin_flip",
                                  verbose=verbose)
    _tr = resolve_discrete_toggle(operator, time_reversal, "time_reversal",
                                  verbose=verbose)
    # Flip transport for the COMPLETE spectrum: [H, X] == 0 makes the
    # n_up and N - n_up magnetisation blocks isospectral, so the dense
    # sweep only diagonalises n_up <= N/2 and mirrors the spectra.
    _flip_transport = False
    if _sf != 0 and sz_conserved:
        try:
            _flip_transport = bool(
                _core.detect_hamiltonian_symmetries(operator)["spin_flip"])
        except Exception:
            _flip_transport = False

    # Spatial group: Stage 9c routes EVERY projectable call through the
    # factorized little-group engine (the streaming rep path below only
    # handles abelian 1-D irrep projection). Explicit NON-ABELIAN
    # generator input -- which previously had no route except the
    # monolithic SAB engine -- is split into (maximal abelian subgroup,
    # coset residues) by split_nonabelian. 'full' raises on decline;
    # 'auto' degrades to the abelian streaming path. device='gpu'
    # batches ALL block eigensolves through one cuSOLVER pool call.
    _gens = _raw_generators(symmetry)
    _star = list(getattr(symmetry, "star_perms", None) or []) \
        if symmetry is not None else []
    _sym_for_lane = symmetry
    if (_gens is not None and _generators_nonabelian(_gens) and not _star):
        _sym_for_lane = [list(g) for g in _gens]   # explicit raw-list input
    lane = resolve_projection_lane(
        _sym_for_lane, point_group=point_group, consumer="full_spectrum",
        eigenvalues_only=True, verbose=verbose)
    if lane.mode == "project":
        _A, _res = lane.A, lane.residues
        try:
            eigs = []
            if sz_conserved:
                top = N // 2 if _flip_transport else N
                for n_up in range(top + 1):
                    d = dict(_core.little_group_full_spectrum(
                        operator, _A, _res, n_up=int(n_up),
                        use_gpu=use_gpu, spin_flip=_sf, time_reversal=_tr))
                    block = [float(e) for e in d["eigenvalues"]]
                    eigs.extend(block)
                    if _flip_transport and n_up * 2 != N:
                        eigs.extend(block)   # isospectral mirror
            else:
                d = dict(_core.little_group_full_spectrum(
                    operator, _A, _res, use_gpu=use_gpu,
                    spin_flip=_sf, time_reversal=_tr))
                eigs = [float(e) for e in d["eigenvalues"]]
            if verbose:
                print("[qed.full_spectrum] spatial group -> "
                      "LITTLE-GROUP engine (factorized d_G reduction)"
                      + (", flip transport halves the Sz sweep"
                         if _flip_transport else ""))
            out = EDResults()
            out.eigenvalues = sorted(eigs)
            return out
        except Exception as exc:            # noqa: BLE001 -- graceful
            if isinstance(point_group, str) \
                    and point_group.lower() == "full":
                raise
            if verbose:
                print(f"[qed.full_spectrum] little-group lane declined "
                      f"({exc}); falling back to the abelian streaming "
                      "path.")

    # No spatial symmetry: a plain dense full diagonalisation already
    # returns every eigenvalue. (Sz-block looping without a spatial group
    # buys nothing for the spectrum multiset, so keep it simple.)
    if info is None:
        params = _bare_full_params(N, 1 << N, spin_length)
        params.use_gpu = use_gpu
        res = _diag_via_workflows_solve(
            operator, DiagonalizationMethod.FULL, params)
        res.eigenvalues = sorted(res.eigenvalues)
        return res

    tmpdir = tempfile.mkdtemp(prefix="qed_fullspec_")
    # full_spectrum is the many-small-sectors regime: turn on the C++ sector-
    # parallel FULL loop so independent (Sz, irrep) blocks are dense-diagonalised
    # across cores. Each sector's eigensolve runs single-threaded (see
    # full_diagonalization's omp_in_parallel guard) to avoid N_sectors x P
    # oversubscription. An explicit user ED_SYM_SECTOR_PARALLEL is honoured.
    _prev_sector_parallel = os.environ.get("ED_SYM_SECTOR_PARALLEL")
    if _prev_sector_parallel is None:
        os.environ["ED_SYM_SECTOR_PARALLEL"] = "1"
    try:
        _write_operator_directory(operator, tmpdir)
        _write_symmetry_directory(tmpdir, info)

        sz_values: list[Optional[int]] = (
            (list(range(N // 2 + 1)) if _flip_transport
             else list(range(N + 1)))
            if sz_conserved else [None])
        # Stage 7a star maps for the per-Sz sector loops.
        _star_maps = None
        if _star and point_group not in (False, 0, "off", "none") \
                and info is not None:
            from .star_reduction import star_maps_from_info
            _star_maps = star_maps_from_info(info, _star) or None
        if verbose:
            print(f"[qed.full_spectrum] N={N} |G|="
                  f"{len(info.get('max_clique', []))} "
                  f"sectors={len(info.get('sectors', []))} "
                  f"sz_conserved={sz_conserved} "
                  f"blocks={'Sz x irrep' if sz_conserved else 'irrep'}")

        eigs: list[float] = []
        eigs_per_sector: list[list[float]] = []
        sector_tags: list[Any] = []
        for n_up in sz_values:
            block_dim = (math.comb(N, n_up) if n_up is not None
                         else (1 << N))
            params = _bare_full_params(N, block_dim, spin_length)
            params.use_symmetry = True
            params.use_gpu = use_gpu
            if n_up is not None:
                params.use_fixed_sz = True
                params.n_up = int(n_up)
            opts = _ed_params_to_solve_options(
                params, DiagonalizationMethod.FULL)
            opts.use_symmetry = True
            opts.spin_flip = _sf
            opts.time_reversal = _tr
            if _star_maps:
                opts.star_maps = _star_maps
            if n_up is not None:
                opts.use_fixed_sz = True
                opts.n_up = int(n_up)
            gs = _core.workflows_solve_streaming_symmetry_directory(
                tmpdir, N, float(spin_length), opts, n_up)
            eigs.extend(gs.eigenvalues)
            if (_flip_transport and n_up is not None
                    and int(n_up) * 2 != N):
                # Isospectral N - n_up mirror of the whole Sz block.
                eigs.extend(gs.eigenvalues)
            _eps = getattr(gs, "eigenvalues_per_sector", None)
            if _eps:
                eigs_per_sector.extend([list(s) for s in _eps])
            _tags = getattr(gs, "sector_tags", None)
            if _tags:
                sector_tags.extend(
                    (int(n_up) if n_up is not None else None, t)
                    for t in _tags)
            if verbose:
                print(f"[qed.full_spectrum]   n_up={n_up} "
                      f"block_dim={block_dim} got={len(gs.eigenvalues)}")

        eigs.sort()
        out = EDResults()
        out.eigenvalues = eigs
        # Per-sector eigenvalues / sector tags are OPTIONAL diagnostics.
        # They rely on ``py::dynamic_attr`` being compiled into the
        # ``EDResults`` binding; on a stale ``_core`` (built before that
        # was enabled) the attribute set raises ``AttributeError``. The
        # complete spectrum in ``out.eigenvalues`` is the contract every
        # caller (incl. NLCE FULL_SYMMETRIZED) depends on, so never let a
        # missing diagnostic slot break it -- attach best-effort.
        if eigs_per_sector:
            try:
                out.eigenvalues_per_sector = eigs_per_sector
            except AttributeError:
                pass
        if sector_tags:
            try:
                out.sector_tags = sector_tags
            except AttributeError:
                pass
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if _prev_sector_parallel is None:
            os.environ.pop("ED_SYM_SECTOR_PARALLEL", None)
        else:
            os.environ["ED_SYM_SECTOR_PARALLEL"] = _prev_sector_parallel


# Internal alias so :func:`solve` can call the helper without colliding
# with its own ``full_spectrum`` (bool) keyword argument.
full_spectrum_compute = full_spectrum


def _bare_full_params(
    num_sites: int, num_eigenvalues: int, spin_length: float
) -> EDParameters:
    """Minimal EDParameters for a dense full-spectrum block (no
    auto-tune / thermal knobs)."""
    p = EDParameters()
    p.num_sites = int(num_sites)
    p.num_eigenvalues = int(num_eigenvalues)
    p.spin_length = float(spin_length)
    p.tolerance = 1e-12
    p.compute_eigenvectors = False
    return p


def _write_operator_directory(operator: Operator, directory: str) -> None:
    """Dump the operator's terms to ``Trans.dat`` / ``InterAll.dat`` /
    ``ThreeBodyG.dat`` in ``directory`` using the legacy mVMC header
    convention the C++ loader expects."""
    one_body = list(operator.iter_one_body_terms())
    two_body = list(operator.iter_two_body_terms())
    three_body = list(operator.iter_three_body_terms())

    _write_dat_file(
        os.path.join(directory, "Trans.dat"),
        rows=[
            (int(op_type), int(site), complex(coeff))
            for op_type, site, coeff in one_body
        ],
        formatter=_format_one_body_row,
    )
    _write_dat_file(
        os.path.join(directory, "InterAll.dat"),
        rows=[
            (int(op1), int(s1), int(op2), int(s2), complex(coeff))
            for op1, s1, op2, s2, coeff in two_body
        ],
        formatter=_format_two_body_row,
    )
    if three_body:
        _write_dat_file(
            os.path.join(directory, "ThreeBodyG.dat"),
            rows=[
                (int(op1), int(s1), int(op2), int(s2), int(op3), int(s3),
                 complex(coeff))
                for op1, s1, op2, s2, op3, s3, coeff in three_body
            ],
            formatter=_format_three_body_row,
        )


def _write_dat_file(path: str, rows: list[Any], formatter) -> None:
    """Write the standard 5-line header followed by formatted rows."""
    with open(path, "w") as f:
        f.write("===================\n")
        f.write(f"num {len(rows):>8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for row in rows:
            f.write(formatter(row))


def _format_one_body_row(row) -> str:
    op_type, site, coeff = row
    return f" {op_type:>8d}  {site:>8d}    {coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"


def _format_two_body_row(row) -> str:
    op1, s1, op2, s2, coeff = row
    return (
        f" {op1:>8d}  {s1:>8d}    {op2:>8d}    {s2:>8d}    "
        f"{coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"
    )


def _format_three_body_row(row) -> str:
    op1, s1, op2, s2, op3, s3, coeff = row
    return (
        f" {op1:>8d}  {s1:>8d}    {op2:>8d}    {s2:>8d}    "
        f"{op3:>8d}    {s3:>8d}    "
        f"{coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"
    )


def _write_symmetry_directory(directory: str, info: dict[str, Any]) -> None:
    """Write the four JSON files the C++ streaming-symmetry kernel needs.

    The C++ ``generate_automorphisms`` helper in
    ``ed/core/system_utils.h`` only re-runs the Python finder when
    ``automorphisms.json`` is missing, so we write all four files. That
    way the user-supplied ``GeneratorSet`` is honored verbatim instead
    of being silently overwritten by the full automorphism search.

    Files written into ``<directory>/automorphism_results/``:

    * ``automorphisms.json``      — flat array of permutations (gates the
      C++ regeneration check)
    * ``max_clique.json``         — flat array of permutations (the
      commuting group chosen by the user)
    * ``minimal_generators.json`` — ``{"generators":
      [{"permutation":..., "order":...}, ...]}``
    * ``sector_metadata.json``    — ``{"sectors": [{"sector_id":...,
      "quantum_numbers":..., "phase_factors": [{"real":..., "imag":...},
      ...]}, ...]}``
    """
    out_dir = os.path.join(directory, "automorphism_results")
    os.makedirs(out_dir, exist_ok=True)

    max_clique = info.get("max_clique", [])
    generators = info.get("generators", [])
    generator_orders = info.get("generator_orders", [])
    sectors = info.get("sectors", [])

    if not max_clique:
        # Fallback: reconstruct via the symmetry DSL when only generators
        # were provided (e.g. a raw dict from a third-party source).
        from .symmetry import group_from_generators  # noqa: WPS433
        info2 = group_from_generators(
            len(generators[0]) if generators else 0,
            [list(map(int, g)) for g in generators],
        )
        max_clique = info2["max_clique"]
        generator_orders = info2["generator_orders"]
        sectors = info2["sectors"]

    max_clique_int = [list(map(int, p)) for p in max_clique]

    # automorphisms.json: full nauty-style list. We give the C++ side
    # exactly the user's group; the downstream max-clique finder treats
    # it as already-commuting so the same permutations come back out.
    # Stale marker from a previous translation_only run would force a
    # regeneration -- delete it pre-emptively.
    marker_file = os.path.join(out_dir, ".translation_only")
    try:
        os.remove(marker_file)
    except FileNotFoundError:
        pass
    with open(os.path.join(out_dir, "automorphisms.json"), "w") as f:
        json.dump(max_clique_int, f, indent=2)

    with open(os.path.join(out_dir, "max_clique.json"), "w") as f:
        json.dump(max_clique_int, f, indent=2)

    gen_records = []
    for gen, order in zip(generators, generator_orders):
        gen_records.append({
            "permutation": list(map(int, gen)),
            "order": int(order),
        })
    with open(os.path.join(out_dir, "minimal_generators.json"), "w") as f:
        json.dump({"generators": gen_records}, f, indent=2)

    # ------------------------------------------------------------------
    # phase_factors: the on-disk schema is one entry PER GENERATOR (length
    # = num_generators), each = exp(2πi q_k / o_k). The C++ kernel then
    # composes the per-element character via
    #     χ_q(g) = ∏_k phase_factors[k]^{powers[k]}
    # where ``powers[k]`` comes from `power_representation[g]`.
    #
    # NOTE: this differs from the convention `group_from_generators`
    # uses in-memory (one full character per group element, length =
    # |max_clique|). We always recompute the per-generator form here so
    # the JSON path stays consistent regardless of how `info` was
    # produced.
    # ------------------------------------------------------------------
    import math as _math

    sector_records = []
    for s in sectors:
        sid = int(s.get("sector_id", 0))
        qn = list(map(int, s.get("quantum_numbers", [])))
        pf = []
        for k, q_k in enumerate(qn):
            o_k = int(generator_orders[k]) if k < len(generator_orders) else 1
            angle = 2.0 * _math.pi * float(q_k) / float(o_k) if o_k else 0.0
            pf.append({
                "real": float(_math.cos(angle)),
                "imag": float(_math.sin(angle)),
            })
        sector_records.append({
            "sector_id": sid,
            "quantum_numbers": qn,
            "phase_factors": pf,
        })
    with open(os.path.join(out_dir, "sector_metadata.json"), "w") as f:
        json.dump({"sectors": sector_records}, f, indent=2)
