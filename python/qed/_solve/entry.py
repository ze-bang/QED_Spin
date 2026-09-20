"""``qed._solve.entry``: :func:`qed.solve`.

Carved out of ``workflow.py`` (WP11). The kwargs-only auto-pilot: validate,
resolve the Sz axis (naming a block, a window, a parity half, or sweeping
and merging), then route to one of the lanes -- the full-spectrum sweep,
the little-group projection engine (eigenvalues or certified vectors), the
abelian streaming-symmetry lane, or the plain orchestrator call.

WP12 turned the routing inside out. The kwargs are frozen into one
:class:`~qed._solve.request.SolveRequest` near the top of :func:`solve`
and the lanes became small functions over that record, listed in the
ordered ``_LANES`` table that :func:`_dispatch` walks. The table IS the
routing contract: the first lane whose predicate fires and that does not
return ``DECLINED`` owns the call, so the row order reproduces, one for
one, the ``if`` chain the body used to be.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Optional, Sequence, Union

from .. import _core as _core
from .._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDResults,
    FixedSzOperator,
    Operator,
)
from ..discovery import resolve_auto_symmetry, resolve_discrete_toggle
from ..point_group_routing import (resolve_projection_lane,
                                   decode_star_for_sector,
                                   decode_irrep_for_character)
from .device import _resolve_device
from .methods import (
    _TOTAL_SPIN_CTX,
    _diag_via_directory,
    _diag_via_workflows_solve,
    _is_thermal_method,
    _is_tpq_method,
    _normalize_total_spin,
    _resolve_solver,
)
from .parameters import _make_params, normalize_sz
from .request import DECLINED, Lane, SolveRequest
from .results import _ProjectLaneBackend
from .spectrum import full_spectrum_compute
from .symmetry_input import SymmetryArg
from .symmetry_lane import _diag_with_symmetry


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
    irrep: Optional[dict] = None,
    flip: Optional[int] = None,
    sz: Union[int, str, tuple, None] = None,
    auto_sz: bool = True,
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    point_group: Union[str, bool, None] = "auto",
    total_spin: Union[int, float, str, None] = "auto",
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
        Backend device. One of ``"auto"`` / ``"cpu"`` / ``"gpu"``.
        ``None`` (default) means ``"auto"``. ``"mpi"`` / ``"mpi_gpu"``
        RAISE on this in-process surface (the ed_distributed_main
        launcher was retired in Stage 11d): distributed runs go through
        ``mpirun`` on the CLI ``ED`` binary, whose sector factory
        dim-balances (n_up, irrep) sectors across ranks and Allgathers
        the results (rank-local solves; see
        ``ed::make_sector_operators_tagged``).
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
        Magnetisation sector, as a SET-BIT count -- i.e. the number of
        **DOWN** spins, NOT up spins and NOT the value of total Sz.
        ``sz=0`` is the fully polarised-UP block (S_z = +N/2) and ``sz=N``
        the fully polarised-DOWN block; ``sz=N//2`` is half filling. The
        name is historical (the C++ side calls it ``n_up`` while counting
        set bits, and the set bit is the down spin -- hence ``S^-`` RAISES
        ``n_up``). Verified: for ``H = Heisenberg - 3*sum(Sz)`` on N=10,
        ``sz=0`` returns -12.5 (all up) and ``sz=10`` returns +17.5 (all
        down). Pinned in test_quantum_number_selection.py.

        Omit it to sweep EVERY magnetisation sector and merge (the true
        global ground state, whatever sector holds it). Naming it solves
        that one block and is ~sqrt(N)x cheaper.
    point_group : str or bool, optional
        Stage 9c semantics. ``"auto"`` (default): eigenvalue-only calls
        PROJECT through the factorized little-group engine (translation
        x flip x little-co-group blocks); vector consumers
        (``compute_eigenvectors``, sampling methods) use the abelian rep
        lane with star/TR/flip folds and per-sector output. ``sector=``
        no longer declines projection (Jul 2026): a named momentum
        resolves to its star(s) through the character table and the
        engine restricts its walk (``LittleGroupOptions::only_k0``) --
        naming a smaller block gets a smaller block. ``"full"``: REQUIRE
        projection -- raises with the decline reason instead of
        degrading. ``"off"``: abelian lane, star folds disabled.
    total_spin : int, float, str, or None, optional
        SU(2) total-spin axis (Stage 12). ``"auto"`` (default): when the
        Hamiltonian is SU(2)-invariant (per-bond isotropic exchange, no
        fields / DM / anisotropy), eigenstates returned with vectors are
        labeled with their certified total spin (``EDResults.spin`` /
        ``EDResults.s2``, parallel to ``eigenvalues``; ``None`` where an
        accidental degeneracy defeats certification). Numeric ``S``
        (integer or half-integer): TARGET the spin-S tower -- each block
        is solved through the Lowdin Casimir projector (seed projected,
        drift scrubbed per ``ED_SYM_SU2_REPROJECT_FREQ``), so the
        returned window is the lowest states OF THAT TOWER. Composes
        with ``sz=``, ``symmetry=``, ``spin_flip``, ``time_reversal``
        (rides the abelian rep lane; ``point_group='full'`` +
        targeting raises). ``"require"``: as auto but raise when H is
        not SU(2)-invariant. ``"off"``/``None``: axis disabled.
        Cost note: one projection costs ~(degree x N/2z) H-applies;
        targeting is opt-in precisely because of this.
    irrep : dict, optional
        Little-co-group irrep, named BY ITS CHARACTER as
        ``{residue_index: value}`` (``-1`` keys the identity) and
        resolved against the star's published table -- never by index
        (the engine's irrep index is per-star and internal, like
        ``k_raw``). Requires ``sector=``; with the spin flip engaged
        also requires ``flip=`` (the isotypic split is per
        ``(k, parity)``). Projection lane only -- raises on the abelian
        lane rather than being ignored.
    flip : int, optional
        Spin-flip parity half of a momentum star: ``0`` = (k,+),
        ``1`` = (k,-). Requires ``sector=`` and an engaged flip
        ([H, prod sigma^x] = 0 on a flip-invariant subspace). Omitted
        with the flip engaged, a named momentum returns BOTH parities.
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
    # Input validation (2026-09-11): these used to be clamped or passed through
    # silently (num_eigenvalues=0 -> 1, a negative tolerance -> the kernels).
    if num_eigenvalues is not None and int(num_eigenvalues) < 1:
        raise ValueError(f"qed.solve: num_eigenvalues must be >= 1, got {num_eigenvalues!r}")
    if not (float(tolerance) > 0.0):
        raise ValueError(f"qed.solve: tolerance must be > 0, got {tolerance!r}")
    if max_iterations is not None and int(max_iterations) < 1:
        raise ValueError(f"qed.solve: max_iterations must be >= 1, got {max_iterations!r}")
    if (solver is not None and str(getattr(solver, "name", solver)).upper() == "LANCZOS"
            and num_eigenvalues is not None and int(num_eigenvalues) > 1):
        warnings.warn(
            "qed.solve(solver='lanczos', num_eigenvalues>1): single-vector Lanczos "
            "reports each degenerate level ONCE, so a window can miss multiplet "
            "copies. Leave solver unset (windows default to Krylov-Schur) or use "
            "'krylov_schur' / 'block_lanczos' when levels may be degenerate.",
            RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # ONE Sz spelling (diction consolidation, Jul 2026): sz= accepts
    # int | (lo, hi) | "auto" | "off". auto_sz=False keeps working with
    # a FutureWarning when load-bearing.
    # ------------------------------------------------------------------
    if isinstance(sz, str) and sz.lower() in ("even", "odd"):
        # Pre-existing Sz-PARITY spelling (parsed further down into
        # _sz_parity_str) -- not part of the magnetisation axis the
        # normalizer owns. CI regression 2026-07-17: the first cut of
        # normalize_sz rejected it and broke the examples tour.
        _szmode = ("auto",)
    else:
        _szmode = normalize_sz(sz, verb="solve", auto_sz=auto_sz)
    _sz_window = None
    if _szmode[0] == "named":
        sz = _szmode[1]
    elif _szmode[0] == "off":
        sz, auto_sz = None, False
    elif _szmode[0] == "window":
        sz = None
        _sz_window = (int(_szmode[1] or 0), int(_szmode[2]
                      if _szmode[2] is not None else H.num_sites))
    elif not (isinstance(sz, str) and sz.lower() in ("even", "odd")):
        sz = None

    # ------------------------------------------------------------------
    # The kwargs are now settled. Freeze them into ONE record (WP12) and
    # let the lane table route: the loose locals this function used to
    # thread through a thousand lines are fields on `req`, and the two
    # normalisation steps the old body ran BETWEEN lanes ride the table
    # as the `prepare` of the first lane that needs them.
    # ------------------------------------------------------------------
    return _dispatch(SolveRequest(
        H=H,
        num_eigenvalues=num_eigenvalues,
        tolerance=tolerance,
        compute_eigenvectors=compute_eigenvectors,
        solver=solver,
        device=device,
        symmetry=symmetry,
        sector=sector,
        irrep=irrep,
        flip=flip,
        sz=sz,
        auto_sz=auto_sz,
        spin_flip=spin_flip,
        time_reversal=time_reversal,
        point_group=point_group,
        total_spin=total_spin,
        lattice=lattice,
        output_dir=output_dir,
        max_iterations=max_iterations,
        block_size=block_size,
        num_samples=num_samples,
        target_beta=target_beta,
        num_temp_points=num_temp_points,
        temp_min=temp_min,
        temp_max=temp_max,
        verbose=verbose,
        full_spectrum=full_spectrum,
        extra_params=extra_params,
        sz_window=_sz_window,
    ))


# ---------------------------------------------------------------------------
# Lane 1: the full-spectrum sweep.
# ---------------------------------------------------------------------------

def _wants_full_spectrum(req: SolveRequest) -> bool:
    return bool(req.full_spectrum)


def _lane_full_spectrum(req: SolveRequest) -> EDResults:
    """Every eigenvalue, decomposed by all (Sz x spatial) symmetries via
    the memory-light representative SpMV.

    Routes to the standalone :func:`qed.full_spectrum` helper, which loops
    the symmetry blocks and dense-diagonalises each. (Honours symmetry=
    / sz auto-detection; the remaining solver/device/thermal kwargs do
    not apply to a full-spectrum sweep.)
    """
    if isinstance(req.H, FixedSzOperator):
        raise ValueError(
            "full_spectrum=True spans every Sz sector; pass the full "
            "Operator (not a FixedSzOperator) so the sweep can loop "
            "magnetisation blocks.")
    # Audit fix (2026-07-30): this branch used to run BEFORE the
    # quantum-number guards below and silently discarded every named
    # selector -- qed.solve(H, full_spectrum=True, sz=6) returned the
    # complete spectrum of every magnetisation block, not block 6.
    # FORWARD the selectors qed.full_spectrum understands (it has its
    # own loud guards for unsupported compositions) and refuse the
    # two that are genuinely inapplicable to a dense sweep.
    _dropped = [name for name, val in (
        ("solver", req.solver), ("output_dir", req.output_dir or None),
    ) if val is not None]
    if _dropped:
        raise ValueError(
            f"qed.solve(full_spectrum=True): {_dropped} cannot be "
            f"honoured by the full-spectrum sweep (every block runs "
            f"the dense block engine; nothing is written). Drop "
            f"full_spectrum=True to solve with those knobs.")
    spin_l = float(getattr(req.H, "spin", 0.5))
    _dev = req.device if isinstance(req.device, str) else "cpu"
    return full_spectrum_compute(
        req.H, symmetry=req.symmetry,
        sz=(req.sz if isinstance(req.sz, int)
            else ("off" if not req.auto_sz else None)),
        sector=req.sector, irrep=req.irrep, flip=req.flip,
        spin_length=spin_l, device=_dev,
        spin_flip=req.spin_flip, time_reversal=req.time_reversal,
        point_group=req.point_group,
        total_spin=(req.total_spin if req.total_spin is not None else "auto"),
        lattice=req.lattice, verbose=req.verbose)


# ---------------------------------------------------------------------------
# Normalisation between lane 1 and lane 2: the quantum-number axes. The
# full-spectrum lane returns before any of this (it forwards the raw
# selectors to its own guards), which is why it is the sz-sweep lane's
# `prepare` and not part of the prologue.
# ---------------------------------------------------------------------------

def _resolve_axes(req: SolveRequest) -> SolveRequest:
    """Selector guards, the dimensions, the SU(2) axis and the Sz parity."""
    if req.flip is not None and req.sector is None:
        raise ValueError(
            "qed.solve: flip= selects the (k,+) or (k,-) half of a MOMENTUM "
            "star, so it needs sector= (the momentum) alongside it. Without a "
            "momentum there is no star to halve.")
    if req.irrep is not None and req.sector is None:
        raise ValueError(
            "qed.solve: irrep= names a LITTLE-CO-GROUP irrep, which is defined "
            "per momentum star -- decompose_irreps orders each star's "
            "decomposition independently, so 'irrep X' means different things "
            "in different stars. Pass sector= (the momentum) alongside it.")

    fixed_sz_input = isinstance(req.H, FixedSzOperator)
    num_sites = int(req.H.num_sites)
    base_dim = int(req.H.dimension)  # full Hilbert dim, even for FixedSz

    # ------------------------------------------------------------------
    # Stage 12 (SU(2) rollout): resolve the total-spin axis ONCE and
    # publish it through the context variable every SolveOptions build
    # reads (_apply_total_spin_opts). Set unconditionally so a stale
    # value from an interrupted earlier call can never leak in.
    # ------------------------------------------------------------------
    _ts2, _ts_label = _normalize_total_spin(req.total_spin, num_sites)
    _TOTAL_SPIN_CTX.set((_ts2, _ts_label))

    # ------------------------------------------------------------------
    # 1. Resolve the fixed-Sz axis. sz="even"/"odd" selects the
    #    Sz-PARITY halves instead (the Z2 remnant when U(1) is broken;
    #    also valid for U(1)-conserving H) -- handled by the symmetry
    #    dispatch below, not the fixed-Sz machinery.
    # ------------------------------------------------------------------
    sz = req.sz
    _sz_parity_str: Optional[int] = None
    if isinstance(sz, str):
        _key = sz.strip().lower()
        if _key not in ("even", "odd"):
            raise ValueError(
                f"sz={sz!r}: string forms are 'even'/'odd' (Sz-parity "
                "halves) or pass an integer n_up.")
        _sz_parity_str = 0 if _key == "even" else 1
        sz = None
        # Correctness (2026-09-11): a parity half is only a valid block when
        # (-1)^{n_down} is conserved. The TFIM (single S+/S- terms) is not,
        # and the parity lane returned wrong eigenvalues instead of refusing.
        try:
            _det_par = dict(_core.detect_hamiltonian_symmetries(req.H))
            _par_ok = bool(_det_par.get("sz_parity", False)) or bool(_det_par.get("u1", False))
        except Exception:  # noqa: BLE001
            _par_ok = True
        if not _par_ok:
            raise ValueError(
                f"qed.solve: sz={_key!r} names an Sz-parity half, but this operator "
                "does not conserve Sz parity (a term changes the number of down "
                "spins by an odd amount); use sz='off'.")

    return req.refine(
        sz=sz,
        fixed_sz_input=fixed_sz_input,
        num_sites=num_sites,
        base_dim=base_dim,
        two_s=_ts2,
        total_spin_label=_ts_label,
        sz_parity=_sz_parity_str,
    )


# ---------------------------------------------------------------------------
# Lane 2: Sz unnamed on the PLAIN lane -> sweep every magnetisation sector.
# ---------------------------------------------------------------------------

def _wants_sz_sweep(req: SolveRequest) -> bool:
    return (req.sz is None and req.sz_parity is None
            and not req.fixed_sz_input
            and req.auto_sz and req.symmetry is None
            and not req.full_spectrum
            and req.H.conserves_sz()
            and not (isinstance(req.device, str)
                     and req.device.lower() in ("mpi", "mpi_gpu")))


def _lane_sz_sweep(req: SolveRequest) -> EDResults:
    """Solve each U(1) block and merge.

    This used to pin n_up = N//2 (see the auto-Sz note below). Half filling
    is where the Heisenberg-AFM ground state lives; it is NOT where a
    ground state lives in general. Under a field, or for a ferro/doped
    model, the pin returned a silently WRONG E0 -- measured on an N=10 ring
    with -3*sum(Sz): -4.5154 against the true -12.5, no warning.

    "Unnamed" now means ALL: solve each U(1) block and merge. Still block
    diagonal (per-sector solves keep the U(1) reduction; this is NOT a
    fall back to the full 2^N space), and each sector is <= the old
    half-filling block, so the cost is ~sqrt(N)x -- paid only when the
    caller declined to name a sector. `sz=` still buys the single-sector
    solve, and `auto_sz=False` still forces the plain full-Hilbert lane.
    """
    _n = int(req.H.num_sites)
    _k_want = int(req.num_eigenvalues) if req.num_eigenvalues else 1
    if req.verbose:
        print(f"[qed.solve] Sz unnamed: sweeping n_up=0..{_n} and merging "
              f"(pass sz= to solve one sector).")
    _per: list[tuple[float, int]] = []
    _best: tuple[float, int, Any] = (float("inf"), -1, None)
    _swlo, _swhi = req.sz_window or (0, _n)
    for _nu_i in range(_swlo, _swhi + 1):
        _ri = solve(
            req.H, num_eigenvalues=req.num_eigenvalues, tolerance=req.tolerance,
            compute_eigenvectors=req.compute_eigenvectors, solver=req.solver,
            device=req.device, symmetry=None, sector=req.sector, irrep=req.irrep,
            flip=req.flip, sz=_nu_i, auto_sz=False, spin_flip=req.spin_flip,
            time_reversal=req.time_reversal, point_group=req.point_group,
            total_spin=req.total_spin,
            lattice=req.lattice, output_dir=req.output_dir,
            max_iterations=req.max_iterations, block_size=req.block_size,
            num_samples=req.num_samples, target_beta=req.target_beta,
            num_temp_points=req.num_temp_points, temp_min=req.temp_min,
            temp_max=req.temp_max, verbose=False, full_spectrum=False,
            extra_params=req.extra_params)
        _ev = list(getattr(_ri, "eigenvalues", []) or [])
        for _e in _ev:
            _per.append((float(_e), _nu_i))
        if _ev and float(_ev[0]) < _best[0]:
            _best = (float(_ev[0]), _nu_i, _ri)
    if not _per:
        raise RuntimeError(
            "qed.solve: the Sz sweep produced no eigenvalues.")
    _per.sort(key=lambda t: t[0])
    _merged = _per[:_k_want]
    # Eigenvectors live in their own sector's basis, so a merged window
    # spanning sectors has no single basis to return them in. Hand back
    # the winning sector's result (its vectors ARE the ground state) and
    # overwrite the eigenvalue window with the true merged one.
    _out = _best[2]
    _out.eigenvalues = [e for e, _ in _merged]
    try:
        _out.sz_sectors = [nu for _, nu in _merged]
    except Exception:
        pass
    if req.verbose:
        print(f"[qed.solve] Sz sweep: E0={_merged[0][0]:.10f} in "
              f"n_up={_merged[0][1]}")
    return _out


# ---------------------------------------------------------------------------
# Normalisation between lane 2 and lane 3: what actually runs. The Sz
# sweep returns before any of it (each of its recursive calls names a
# sector and resolves its own), which is why this is the first symmetry
# lane's `prepare` rather than part of the prologue.
# ---------------------------------------------------------------------------

def _resolve_execution(req: SolveRequest) -> SolveRequest:
    """The operator, the method/device decision, the output directory,
    the ``EDParameters`` bag and the resolved symmetry group."""
    op_to_use: Operator = req.H
    sz, num_sites, base_dim = req.sz, req.num_sites, req.base_dim
    if sz is not None:
        if req.fixed_sz_input:
            # Sanity-check that the provided FixedSzOperator matches.
            # FixedSzOperator does not currently expose ``n_up``, so we
            # only verify the dimension is consistent with C(N, sz).
            expected = math.comb(num_sites, int(sz))
            if int(req.H.dimension) != expected:
                raise ValueError(
                    f"sz={sz} implies dimension C({num_sites}, {sz})={expected} "
                    f"but the supplied FixedSzOperator has dimension {req.H.dimension}. "
                    "Pass `H` and `sz` as a matched pair, or pass an Operator "
                    "and let qed.solve construct the FixedSzOperator for you."
                )
            op_to_use = req.H
        else:
            if not req.H.conserves_sz():
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
            op_to_use = req.H.make_fixed_sz(int(sz))
            if req.verbose:
                d = op_to_use.dimension
                print(f"[qed.solve] Sz sector n_up={sz}: dim={d} "
                      f"(reduced from {base_dim}).")
    elif req.fixed_sz_input and req.verbose:
        print(f"[qed.solve] FixedSzOperator supplied: dim={req.H.dimension}.")
    # NOTE: the old "auto-Sz projection" branch lived here and pinned
    # n_up = N//2 when Sz was unnamed. It is gone: unnamed now means the
    # magnetisation SWEEP in lane 2 above, which is right for every model
    # rather than only for Heisenberg-style antiferromagnets. (MPI lanes were
    # excluded from auto-Sz because ed_distributed_main loads full-Hilbert
    # .dat files and does its own Sz projection via the launcher's sz=; the
    # sweep carries the same exclusion.)

    sector_dim = int(op_to_use.dimension)

    # ------------------------------------------------------------------
    # 2. Resolve solver + device + flags.
    # ------------------------------------------------------------------
    method = _resolve_solver(req.solver, req.num_eigenvalues, sector_dim)
    use_gpu, use_mpi = _resolve_device(req.device, sector_dim)
    is_thermal = _is_thermal_method(method)
    is_tpq = _is_tpq_method(method)

    # TPQ relies on a single random state evolving on the full Hilbert
    # space (or on a single fixed-Sz block); per-sector symmetry
    # projection breaks that -- reject explicitly.
    if req.symmetry is not None and is_tpq:
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
    effective_output = req.output_dir
    if is_thermal and not req.output_dir:
        # Optimization (Jul 2026): the unified thermal kernels return
        # trajectories + thermodynamics in memory; nothing needs the
        # historical auto-created qed_thermal_<ts>/ directory. Writes
        # happen only when the caller passes output_dir=.
        effective_output = "/dev/null"
    elif is_thermal and req.output_dir and not req.output_dir.startswith("/dev/null"):
        # Surface-unification follow-up (May 2026): the orchestrator's
        # `_core.workflows_thermal` does NOT mkdir its `output_dir`.
        # We mirror the historical mkdir-then-write behaviour here so
        # callers can pass a fresh path without manual mkdir.
        # Skip the mkdir for the /dev/null sentinel (benchmarking / testing).
        os.makedirs(req.output_dir, exist_ok=True)

    if req.verbose:
        method_name = method.name if hasattr(method, "name") else str(method)
        kind = "thermal" if is_thermal else "eigenvalue"
        # use_gpu/use_mpi are the REQUEST derived from device= and the
        # build, printed before lane dispatch -- say so (the old wording
        # read as "this run is on the GPU" even on CPU-only nodes).
        print(f"[qed.solve] solver={method_name} ({kind})  "
              f"num_eigenvalues={req.num_eigenvalues}  "
              f"tolerance={req.tolerance:g}  "
              f"gpu_requested={use_gpu}  mpi_requested={use_mpi} "
              f"(lane chosen at dispatch)")

    # (Pre-flight planner removed: sensible defaults, no feasibility refusal.)

    # ------------------------------------------------------------------
    # 4. Build EDParameters with sensible default Krylov / thermal sizes.
    # ------------------------------------------------------------------
    params = _make_params(
        num_sites=num_sites,
        num_eigenvalues=req.num_eigenvalues,
        tolerance=req.tolerance,
        compute_eigenvectors=req.compute_eigenvectors,
        max_iterations=req.max_iterations,
        block_size=req.block_size,
        sector_dim=sector_dim,
        method=method,
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        sector=req.sector,
        sz=sz,
        output_dir=effective_output,
        num_samples=req.num_samples,
        target_beta=req.target_beta,
        num_temp_points=req.num_temp_points,
        temp_min=req.temp_min,
        temp_max=req.temp_max,
    )
    if req.extra_params:
        for key, value in req.extra_params.items():
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

    symmetry = resolve_auto_symmetry(op_to_use, req.symmetry, verbose=req.verbose,
                                     lattice=req.lattice)
    if symmetry is None:
        # Audit fix (2026-07-30): with no spatial group resolved, the
        # discrete-symmetry toggles were accepted and never consulted --
        # including the 'require' / 'full' HARD contracts, which
        # became silent no-ops (the worst case: the caller believes the
        # composition is enforced). Refuse the hard spellings loudly;
        # the soft spellings ('auto'/'on'/'off') stay permissive by
        # design on this lane.
        _hard = [name for name, val, hard in (
            ("spin_flip", req.spin_flip, "require"),
            ("time_reversal", req.time_reversal, "require"),
            ("point_group", req.point_group, "full"),
        ) if isinstance(val, str) and val.lower() == hard]
        if _hard:
            raise ValueError(
                f"qed.solve: {_hard} demand their symmetry be CONSUMED, "
                f"but no spatial symmetry group resolved on this call "
                f"(symmetry=None/'off', or the model has only the trivial "
                f"automorphism), so there is no lane that can honour "
                f"them. Pass symmetry='auto' (or an explicit "
                f"GeneratorSet), or drop the 'require'/'full' spelling.")

    return req.refine(
        op=op_to_use,
        sector_dim=sector_dim,
        method=method,
        use_gpu=use_gpu,
        use_mpi=use_mpi,
        is_thermal=is_thermal,
        is_tpq=is_tpq,
        effective_output=effective_output,
        params=params,
        symmetry=symmetry,
    )


def _has_symmetry(req: SolveRequest) -> bool:
    """Stage 9c: ONE routing decision, taken over the resolved group.

    point_group='auto' (default) PROJECTS through the factorized
    little-group engine whenever the call is eigenvalue-only; 'full'
    requires projection (raises on decline); any decline degrades to the
    abelian rep lane with star/TR/flip folds. The monolithic SAB engine
    is a test oracle -- no longer production-routed.

    `sector=` no longer vetoes the projection (Jul 2026). Naming a
    momentum used to force the call onto the ABELIAN lane, i.e. asking
    for a smaller block silently bought a BIGGER one -- the (n_up, k)
    sector instead of (n_up, k, irrep). The engine can restrict its star
    walk (LittleGroupOptions::only_k0), so a named momentum now projects
    AND does only that star's work. Eigenvectors and sampling methods
    still decline: the projection lane genuinely does not produce them.
    """
    return req.symmetry is not None


# ---------------------------------------------------------------------------
# Lane 3: certified eigenvectors out of the little-group engine.
# ---------------------------------------------------------------------------

def _wants_little_group_vectors(req: SolveRequest) -> bool:
    return (req.symmetry is not None
            and req.compute_eigenvectors and not req.is_thermal
            and req.sector is None and req.irrep is None and req.flip is None
            and req.two_s < 0)  # Stage 12: targeting rides the abelian lane


def _lane_little_group_vectors(req: SolveRequest) -> Any:
    """r2b (lane unification): the projection lane now PRODUCES
    certified eigenvectors (little_group_lowest_vectors: per-block
    pairs, W_sigma lift re-certified against H_k0, flip-aware
    COMPUTATIONAL-basis expansion, persisted via the canonical
    /eigendata writer). Contract: a single NAMED subspace (sz=);
    the merged multi-subspace sweep keeps the abelian lane's
    winning-sector convention. A multiplicity-m row carries ONE
    representative vector (fold partners need U3 transport).
    """
    vlane = resolve_projection_lane(
        req.symmetry, point_group=req.point_group, consumer="solve",
        eigenvalues_only=True,   # the capability exists as of r2
        prefer_abelian=False, verbose=req.verbose)
    if vlane.mode == "project" and isinstance(req.sz, int):
        _sfv = resolve_discrete_toggle(
            req.op, req.spin_flip, "spin_flip", verbose=req.verbose)
        _trv = resolve_discrete_toggle(
            req.op, req.time_reversal, "time_reversal",
            verbose=req.verbose)
        d = dict(_core.little_group_lowest_vectors(
            req.op, vlane.A, vlane.residues,
            k=int(req.num_eigenvalues) if req.num_eigenvalues else 1,
            n_up=int(req.sz), spin_flip=_sfv, time_reversal=_trv,
            output_dir=str(req.output_dir) if req.output_dir else ""))
        out = EDResults()
        out.eigenvalues          = list(d["eigenvalues"])
        out.eigenvectors         = list(d["vectors"])
        out.eigenvectors_computed = True
        out.eigenvectors_path    = d["hdf5_path"]
        out.block_k_raw          = list(d["k_raw"])
        out.block_flip_parity    = list(d["flip_parity"])
        out.block_irrep          = list(d["irrep"])
        out.block_irrep_dim      = list(d["irrep_dim"])
        out.block_multiplicity   = list(d["multiplicity"])
        out.backend = _ProjectLaneBackend("cpu")
        if req.verbose:
            print(f"[qed.solve] little-group VECTOR lane (r2b): "
                  f"{len(out.eigenvalues)} certified pairs, "
                  f"computational basis.")
        return out
    elif req.point_group != "full" and not req.output_dir:
        # Correctness (2026-09-11): the abelian rep lane never returns
        # eigenvectors in memory (only per-sector HDF5 when output_dir
        # is set), so a vector request silently came back without
        # vectors. Use the plain fixed-Sz lane, which does. With an
        # output_dir the per-sector HDF5 contract stands, and
        # point_group='full' keeps raising on a decline (below).
        if req.verbose:
            print(f"[qed.solve] vector lane declined ({vlane.reason}); "
                  f"eigenvectors requested -> plain fixed-Sz lane "
                  f"(the abelian symmetry lane returns vectors only via output_dir).")
        return solve(
            req.H, num_eigenvalues=req.num_eigenvalues, tolerance=req.tolerance,
            compute_eigenvectors=True, solver=req.solver, device=req.device,
            symmetry=None, sz=req.sz, auto_sz=req.auto_sz, spin_flip="off",
            time_reversal="off", point_group="off", total_spin=req.total_spin,
            lattice=req.lattice, output_dir=req.output_dir,
            max_iterations=req.max_iterations, block_size=req.block_size,
            verbose=req.verbose, extra_params=req.extra_params)
    return DECLINED


# ---------------------------------------------------------------------------
# Lane 4: the factorized non-abelian little-group projection engine.
# ---------------------------------------------------------------------------

def _lane_little_group(req: SolveRequest) -> Any:
    """Eigenvalues (and their quantum-number labels) per projected block."""
    op_to_use, use_gpu, method = req.op, req.use_gpu, req.method
    sz, sector, irrep, flip = req.sz, req.sector, req.irrep, req.flip
    num_sites, point_group, verbose = req.num_sites, req.point_group, req.verbose
    symmetry, _ts2 = req.symmetry, req.two_s
    if _ts2 >= 0 and point_group == "full":
        raise ValueError(
            "qed.solve: total_spin targeting is not implemented inside "
            "the little-group projection engine yet -- it rides the "
            "abelian rep lane (which still exploits the translation "
            "clique). Drop point_group='full' or total_spin=.")
    lane = resolve_projection_lane(
        symmetry, point_group=point_group, consumer="solve",
        eigenvalues_only=(not req.compute_eigenvectors
                          and not req.is_thermal),
        # 2026-07-20: the project lane is genuinely GPU-capable again
        # -- the batched cuSOLVER block eigensolve (little_group_gpu.cu,
        # silently lost when Family 6 removed its SAB-owned kernel) is
        # restored and use_gpu= is threaded through _lg_block below, so
        # a device='gpu' request runs the non-abelian blocks ON the
        # device (plus the >= 2^20-rep gather for Lanczos-sized blocks)
        # and no longer vetoes projection.
        # Stage 12: total-spin TARGETING soft-vetoes projection (the
        # Lowdin wrapper lives on the abelian rep lane).
        prefer_abelian=(_ts2 >= 0),
        verbose=verbose)
    if irrep is not None and lane.mode != "project":
        # irrep= names a LITTLE-CO-GROUP irrep, which only the projection
        # lane decomposes -- the abelian lane's blocks are (n_up, k) and
        # have no isotypic split to name. Accepting it there returned the
        # whole star's ground state for EVERY character (measured: +1 and
        # -1 both -4.763032 on a 12-ring), i.e. silently ignoring the
        # argument. Refuse instead.
        raise ValueError(
            f"qed.solve: irrep= names a little-co-group irrep, which only "
            f"the projection lane computes -- this call resolved to the "
            f"abelian lane ({lane.reason or 'point_group is off'}), whose "
            f"blocks are (n_up, k) with no irrep axis to select on. Pass "
            f"point_group='full' to require projection (it raises with the "
            f"reason if it cannot), or drop irrep=.")
    if flip is not None and lane.mode != "project":
        # Same contract as irrep= above: flip= names a spin-flip parity,
        # which only the projection lane's star decoder consumes. The
        # abelian lane solves BOTH flip halves and merges, so accepting
        # flip= there silently returned the union (measured 2026-07-30:
        # flip=0 and flip=1 byte-identical on an 8-ring). Refuse instead.
        raise ValueError(
            f"qed.solve: flip= names a spin-flip parity, which only the "
            f"projection lane selects on -- this call resolved to the "
            f"abelian lane ({lane.reason or 'point_group is off'}), which "
            f"solves both flip halves and merges them. Pass "
            f"point_group='full' to require projection (it raises with "
            f"the reason if it cannot), or drop flip=.")
    if lane.mode == "project":
        _k = int(req.num_eigenvalues) if req.num_eigenvalues else 1
        _nu = int(sz) if isinstance(sz, int) else -1
        _nu_sweep = None       # magnetisation sectors to sweep + merge
        _only_k0: list[int] = []   # star restriction for a named momentum
        _only_irrep: list[int] = []  # isotypic restriction for a named irrep
        _sp = -1
        if req.sz_parity is not None:
            _sp = int(req.sz_parity)
        elif _nu < 0 and req.auto_sz:
            # Compose the diagonal axis automatically. For U(1)-conserving
            # H this used to PIN n_up = N//2 "(GS at half filling)". That is
            # a Heisenberg-AFM fact, not a theorem: in a field, or for a
            # ferro/doped model, the ground state moves off half filling and
            # the pin returned a SILENTLY WRONG E0 (measured: N=10 ring with
            # -3*sum(Sz) gave -4.5154 against the true -12.5). Worse, it
            # fired even when the caller passed symmetry= explicitly --
            # exactly what the sibling auto-Sz site refuses to do, for
            # exactly this reason (see its `symmetry is None` guard).
            #
            # "Unnamed" must mean ALL, not "the sector we bet on": sweep
            # every magnetisation sector and merge. This stays block
            # diagonal (each n_up is solved in its own U(1) block, so the
            # U(1) reduction is kept -- it is not a fall back to the full
            # 2^N space); it costs ~sum_n C(N,n) instead of one C(N,N/2)
            # solve, i.e. ~sqrt(N)x more work for an answer that is right
            # for every model. Name sz= to pay for one sector only.
            if op_to_use.conserves_sz():
                _nu_sweep = list(range(*( (req.sz_window[0],
                    req.sz_window[1] + 1) if req.sz_window else
                    (0, num_sites + 1) )))
            else:
                try:
                    if bool(_core.detect_hamiltonian_symmetries(
                            op_to_use)["sz_parity"]):
                        _sp = 2
                except Exception:
                    _sp = -1
        _sf = resolve_discrete_toggle(op_to_use, req.spin_flip,
                                      "spin_flip", verbose=verbose)
        _tr = resolve_discrete_toggle(op_to_use, req.time_reversal,
                                      "time_reversal", verbose=verbose)
        _A, _res = lane.A, lane.residues
        if sector is not None:
            # A NAMED momentum: decode it to a star and restrict the walk,
            # so the caller pays for that block alone instead of the whole
            # star sweep. k_raw is an engine-internal irrep index, so this
            # goes through the character table (see decode_star_for_sector)
            # -- never through an index convention.
            _plan = dict(_core.little_group_full_spectrum(
                op_to_use, _A, _res,
                n_up=(_nu if _nu >= 0 else (num_sites // 2
                                            if _nu_sweep else -1)),
                sz_parity=_sp if _sp in (0, 1) else -1,
                spin_flip=_sf, time_reversal=_tr, plan_only=True))
            _dec = decode_star_for_sector(
                _plan["stars"], _plan["irrep_characters"], _A,
                symmetry.generators, symmetry.orders, sector, flip=flip)
            if isinstance(_dec, str):
                raise RuntimeError(
                    f"qed.solve: sector={list(sector)} could not be "
                    f"resolved on the projection lane: {_dec}")
            _k0s, _kraw = _dec
            # Naming a momentum but not a flip means BOTH parities: the
            # decoder returns (k,+) and (k,-) so the caller gets the whole
            # momentum sector. Resolving only the raw index would have
            # matched the parity-0 star alone and silently halved the
            # spectrum.
            _only_k0 = list(_k0s)
            if verbose:
                print(f"[qed.solve] sector={list(sector)}"
                      f"{'' if flip is None else f' flip={flip}'} -> irrep "
                      f"k_raw={_kraw}, star(s) k0={_only_k0}: solving "
                      f"those only (projected).")
            if irrep is not None:
                # Name the little-co-group irrep BY ITS CHARACTER and
                # resolve it against THIS star's published table. The
                # index is per-star (decompose_irreps orders each star
                # independently), which is exactly why irrep= requires
                # sector= and why the caller never passes an index.
                if len(_only_k0) != 1:
                    raise ValueError(
                        "qed.solve: irrep= needs a single star, but "
                        f"sector={list(sector)} resolved to {len(_only_k0)} "
                        "(both flip parities). Pass flip=0 or flip=1 too -- "
                        "the isotypic decomposition is per (k, parity).")
                _k0 = _only_k0[0]
                _st = next((x for x in _plan["stars"]
                            if int(x["k0"]) == _k0), None)
                if _st is None:
                    raise RuntimeError(
                        f"qed.solve: star k0={_k0} vanished from the plan.")
                _idec = decode_irrep_for_character(_st, irrep)
                if isinstance(_idec, str):
                    raise ValueError(
                        f"qed.solve: irrep={irrep} could not be resolved "
                        f"on star k0={_k0}: {_idec}")
                _ii, _idim = _idec
                _only_irrep = [_ii]
                if verbose:
                    print(f"[qed.solve] irrep={irrep} -> index {_ii} "
                          f"(d={_idim}) on star k0={_k0}.")
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
                        use_gpu=use_gpu,
                        spin_flip=_sf, time_reversal=_tr,
                        only_k0=_only_k0, only_irrep=_only_irrep, **kw))
                    rows = []
                    for e, m, kk, fp, ir, dd in zip(
                            d["block_values"], d["multiplicities"],
                            d["block_k_raw"], d["block_flip_parity"],
                            d["block_irrep"], d["block_irrep_dim"]):
                        rows.extend(
                            [(float(e), kk, fp, ir, dd, m, sub,
                              True)] * m)   # full spectrum is exact
                    return rows, d["irrep_characters"], bool(
                        d.get("gpu_engaged", False))
                d = dict(_core.little_group_lowest_eigenvalues_labeled(
                    op_to_use, _A, _res, k=_k,
                    use_gpu=use_gpu,
                    spin_flip=_sf, time_reversal=_tr,
                    only_k0=_only_k0, only_irrep=_only_irrep, **kw))
                rows = [(float(e), kk, fp, ir, dd, m, sub, bool(cv))
                        for e, kk, fp, ir, dd, m, cv in zip(
                            d["eigenvalues"], d["k_raw"],
                            d["flip_parity"], d["irrep"],
                            d["irrep_dim"], d["multiplicity"],
                            d["converged"])]
                return rows, d["irrep_characters"], bool(
                    d.get("gpu_engaged", False))

            _gpu_used = False
            if _sp == 2:
                rows, chars, _g0 = _lg_block(sz_parity=0)
                rows2, _, _g1 = _lg_block(sz_parity=1)
                rows += rows2
                _gpu_used = _g0 or _g1
            elif _sp in (0, 1):
                rows, chars, _gpu_used = _lg_block(sz_parity=_sp)
            elif _nu_sweep is not None:
                # Sz unnamed -> every magnetisation sector, merged. Each
                # block_subspace row carries its own n_up, so the merged
                # rows stay attributable. irrep_characters are a property
                # of A, not of n_up, so the first block's table serves all
                # (same reasoning as the sz_parity==2 branch above).
                rows, chars, _gpu_used = _lg_block(n_up=_nu_sweep[0])
                for _nu_i in _nu_sweep[1:]:
                    _rows_i, _, _g_i = _lg_block(n_up=_nu_i)
                    rows += _rows_i
                    _gpu_used = _gpu_used or _g_i
            else:
                rows, chars, _gpu_used = _lg_block(n_up=_nu)
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
            # The project lane used to return an EDResults with NO backend
            # metadata at all, so `r.backend` raised AttributeError and any
            # GPU assertion against it was impossible. Report the lane the
            # ENGINE says it ran -- not the device= that was requested: the
            # little-group GPU gather only engages past 2^20 reps, so small
            # blocks legitimately stay on the CPU no matter what was asked.
            out.backend = _ProjectLaneBackend(
                "gpu" if _gpu_used else "cpu")
            if verbose:
                print(f"[qed.solve] non-abelian LITTLE-GROUP lane "
                      f"(factorized): |A| = {len(_A)}, residues = "
                      f"{len(_res)}.")
            return out
        except Exception as exc:      # noqa: BLE001 -- graceful
            if isinstance(point_group, str) \
                    and point_group.lower() == "full":
                raise
            if irrep is not None or flip is not None:
                # The abelian fallback cannot honour a named irrep or
                # flip parity (see the guards above, which ran BEFORE
                # lane resolution and cannot re-fire here). Falling
                # back would silently return the whole star's result
                # for every named character. Refuse instead.
                raise RuntimeError(
                    f"qed.solve: the little-group projection lane "
                    f"failed ({exc}) and the abelian fallback cannot "
                    f"honour irrep=/flip=. Drop those arguments to "
                    f"accept the abelian merge, or fix the projection "
                    f"failure.") from exc
            if verbose:
                print(f"[qed.solve] little-group lane declined "
                      f"({exc}); falling back to the abelian rep "
                      "lane.")
    return DECLINED


# ---------------------------------------------------------------------------
# Lane 5: the abelian streaming-symmetry rep lane.
# ---------------------------------------------------------------------------

def _lane_abelian_symmetry(req: SolveRequest) -> EDResults:
    """Per-sector matrix-free apply over the (n_up, k) blocks."""
    # Jul 2026 (#4): the abelian rep lane's per-sector Lanczos has no
    # distinct-value/residual-bound gate -- upper-window levels drift
    # ~1e-3 at 30 sites (measured CPU-vs-GPU by the e2e diagnostic).
    # Until the orchestrator sector solve adopts the project lane's
    # gate, say so instead of being quietly worse.
    if not req.is_thermal and req.sector is None \
            and int(req.num_eigenvalues or 1) > 1:
        warnings.warn(
            "qed.solve (abelian rep lane): eigenvalues beyond the "
            "lowest per sector are not convergence-guarded on this "
            "lane; for excited-window physics prefer the little-group "
            "project lane (point_group='auto' with retained residues), "
            "whose gate and filter guarantee distinct converged Ritz "
            "values.",
            RuntimeWarning, stacklevel=2)
    return _diag_with_symmetry(
        req.op, req.symmetry, req.params, req.method,
        sz=req.sz if req.sz is not None else None,
        sector=req.sector,
        sz_parity=req.sz_parity,
        auto_sz_axis=req.auto_sz,
        verbose=req.verbose,
        spin_flip=req.spin_flip,
        time_reversal=req.time_reversal,
        point_group=req.point_group,
        auto_method=req.auto_method,
    )


# ---------------------------------------------------------------------------
# Lane 6: GPU, no symmetry -- the orchestrator with allow_gpu.
# ---------------------------------------------------------------------------

def _wants_gpu_directory(req: SolveRequest) -> bool:
    return bool(req.use_gpu)


def _lane_gpu_directory(req: SolveRequest) -> EDResults:
    # 2026-09-11: honour "no solver named" as Auto here too (windows ->
    # Krylov-Schur); forcing Lanczos dropped degenerate copies on the GPU
    # lane while the CPU lane got them right.
    return _diag_via_directory(req.op, req.method, req.params,
                               verbose=req.verbose,
                               auto_method=req.auto_method)


# ---------------------------------------------------------------------------
# Lane 7: CPU, no symmetry -- the table's floor, the fastest path, no I/O.
# ---------------------------------------------------------------------------

def _lane_plain(req: SolveRequest) -> EDResults:
    # solver=None => let the orchestrator pick the ground-state eigensolver
    # default (full diag for tiny dims, Lanczos otherwise).
    return _diag_via_workflows_solve(req.op, req.method, req.params,
                                     auto_method=req.auto_method)


def _always(req: SolveRequest) -> bool:
    """The floor lane's predicate: some lane must own every call."""
    return True


# ---------------------------------------------------------------------------
# The lane table. ORDER IS THE CONTRACT -- it reproduces, row for row,
# the `if` chain solve() used to be:
#
#   * symmetry path -> orchestrator's streaming-symmetry kernel via
#     ``_core.workflows_solve_streaming_symmetry`` (handles GPU
#     per-sector), with the little-group projection engine taking the
#     eigenvalue-only and certified-vector calls off the front of it;
#   * GPU + no-symmetry -> orchestrator with
#     ``BackendConstraints::allow_gpu = true`` (the orchestrator builds
#     the right GPUOperator under the hood);
#   * CPU + no-symmetry -> orchestrator with the CPU lane (the fastest
#     path, no I/O).
#
# A row's `prepare` runs before its predicate is evaluated, whether or
# not the predicate then fires; that is where the normalisation the old
# body did BETWEEN two `if` blocks lives, and it is why the rows that
# short-circuit first never pay for it.
# ---------------------------------------------------------------------------

_LANES: tuple[Lane, ...] = (
    Lane("full_spectrum", _wants_full_spectrum, _lane_full_spectrum),
    Lane("sz_sweep", _wants_sz_sweep, _lane_sz_sweep,
         prepare=_resolve_axes),
    Lane("little_group_vectors", _wants_little_group_vectors,
         _lane_little_group_vectors, prepare=_resolve_execution),
    Lane("little_group", _has_symmetry, _lane_little_group),
    Lane("abelian_symmetry", _has_symmetry, _lane_abelian_symmetry),
    Lane("gpu_directory", _wants_gpu_directory, _lane_gpu_directory),
    Lane("plain", _always, _lane_plain),
)


def _dispatch(req: SolveRequest) -> EDResults:
    """Walk :data:`_LANES` in order; the first lane that fires and does
    not hand the request back owns the call."""
    for lane in _LANES:
        req = lane.prepare(req)
        if lane.predicate(req):
            result = lane.run(req)
            if result is not DECLINED:
                return result
    # Unreachable: the last row's predicate is _always.
    raise AssertionError("qed.solve: no lane accepted the request.")
