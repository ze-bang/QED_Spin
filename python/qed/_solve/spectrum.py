"""``qed._solve.spectrum``: :func:`qed.full_spectrum`.

Carved out of ``workflow.py`` (WP11). The COMPLETE eigenvalue spectrum,
decomposed by every available ``(Sz x spatial)`` symmetry: the factorized
little-group engine when the call projects, the streaming rep lane with a
trivial spatial group when it does not, and a plain dense solve only when
there is literally nothing to exploit.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Union

from .. import _core as _core
from .._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDResults,
    Operator,
)
from .._params import ed_params_to_solve_options as _ed_params_to_solve_options
from ..discovery import resolve_auto_symmetry, resolve_discrete_toggle
from ..point_group_routing import (resolve_projection_lane,
                                   decode_star_for_sector,
                                   decode_irrep_for_character)
from .methods import (
    _TOTAL_SPIN_CTX,
    _diag_via_workflows_solve,
    _normalize_total_spin,
)
from .parameters import _bare_full_params, normalize_sz
from .results import _attach_su2_full_spectrum_labels
from .symmetry_input import (
    SymmetryArg,
    _closed_symmetry_info,
    _generators_nonabelian,
    _normalize_symmetry_info,
    _operator_conserves_sz,
    _raw_generators,
)


def full_spectrum(
    operator: Operator,
    *,
    symmetry: SymmetryArg = None,
    sz: Union[int, str, tuple, None] = None,
    sector: Optional[Sequence[int]] = None,
    irrep: Optional[dict] = None,
    flip: Optional[int] = None,
    sz_conserved: Optional[bool] = None,
    spin_length: float = 0.5,
    device: str = "cpu",
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    point_group: Union[str, bool, None] = "auto",
    total_spin: Union[int, float, str, None] = "auto",
    lattice: Optional[Any] = None,
    verbose: bool = False,
) -> EDResults:
    """Compute the COMPLETE eigenvalue spectrum of ``operator`` decomposed
    by all available ``(Sz x spatial)`` symmetries.

    ``sz``: name a magnetisation block (a SET-BIT count -- the number of DOWN
    spins; ``sz=0`` is fully polarised UP -- see :func:`solve`) to get the
    complete spectrum OF THAT BLOCK. Omit it for every sector, merged (the
    default, and the true full spectrum). Naming a sector disables the
    flip-transport half-sweep: that optimisation mirrors a half sweep onto its
    partner sectors, which is meaningless -- and would silently double-count --
    when the caller asked for one specific block.

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
    # Stage 12 (SU(2) rollout): publish the total-spin axis for every
    # SolveOptions this verb builds. Targeting (numeric S) is not
    # meaningful for a COMPLETE spectrum -- refuse it; auto labeling
    # rides the dense S-resolution (Stage 12e) where implemented.
    _fs_ts2, _fs_label = _normalize_total_spin(
        total_spin, int(operator.num_sites))
    if _fs_ts2 >= 0:
        raise ValueError(
            "qed.full_spectrum: total_spin=<S> targeting selects one "
            "tower, which contradicts a full-spectrum request. Use "
            "qed.solve(total_spin=S) for tower minima, or "
            "total_spin='auto' here for labels.")
    _TOTAL_SPIN_CTX.set((_fs_ts2, _fs_label))
    # ONE Sz spelling (Jul 2026): int | "auto" | "off" ((lo, hi) windows
    # interact with the flip-transport mirror -- not supported here yet).
    _szmode = normalize_sz(sz, verb="full_spectrum",
                           sz_conserved=sz_conserved)
    if _szmode[0] == "named":
        sz = _szmode[1]
    elif _szmode[0] == "off":
        sz, sz_conserved = None, False
    elif _szmode[0] == "window":
        raise NotImplementedError(
            "qed.full_spectrum: sz=(lo, hi) windows are not supported on "
            "this verb (they interact with the flip-transport mirror); "
            "name a single block or take the full sweep.")
    else:
        sz = None
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
        eigenvalues_only=True,
        # 2026-07-20: the project lane's batched GPU block eigensolve is
        # restored (little_group_gpu.cu), and use_gpu= is already passed to
        # the full-spectrum binding -- no device-based veto needed.
        prefer_abelian=False,
        verbose=verbose)
    if lane.mode != "project" and (sector is not None or irrep is not None
                                   or flip is not None):
        # Same contract as qed.solve: these name little-group structure, which
        # only the projection lane has. Refusing beats accepting an argument
        # the lane cannot honour.
        raise ValueError(
            f"qed.full_spectrum: sector=/irrep=/flip= name projection-lane "
            f"structure, but this call resolved to the abelian lane "
            f"({lane.reason or 'point_group is off'}). Pass point_group='full' "
            f"to require projection, or drop them.")
    if lane.mode == "project":
        _A, _res = lane.A, lane.residues
        _fs_only_k0: list[int] = []
        _fs_only_irrep: list[int] = []
        if sector is not None or flip is not None:
            if sector is None:
                raise ValueError(
                    "qed.full_spectrum: flip= selects a half of a MOMENTUM "
                    "star, so it needs sector= alongside it.")
            _fs_plan = dict(_core.little_group_full_spectrum(
                operator, _A, _res,
                n_up=(int(sz) if sz is not None
                      else (N // 2 if sz_conserved else -1)),
                spin_flip=_sf, time_reversal=_tr, plan_only=True))
            _fgens = getattr(_sym_for_lane, "generators", None)
            _fords = getattr(_sym_for_lane, "orders", None)
            if _fgens is None or _fords is None:
                raise ValueError(
                    "qed.full_spectrum: sector= names quantum numbers per "
                    "GENERATOR, so it needs a GeneratorSet (e.g. from "
                    "qed.find_symmetries), not a raw permutation list -- a raw "
                    "list carries no generator/order basis to name them in.")
            _fd = decode_star_for_sector(
                _fs_plan["stars"], _fs_plan["irrep_characters"], _A,
                _fgens, _fords, sector, flip=flip)
            if isinstance(_fd, str):
                raise ValueError(
                    f"qed.full_spectrum: sector={list(sector)} could not be "
                    f"resolved: {_fd}")
            _fs_only_k0, _fkraw = list(_fd[0]), _fd[1]
            if irrep is not None:
                if len(_fs_only_k0) != 1:
                    raise ValueError(
                        "qed.full_spectrum: irrep= needs a single star, but "
                        f"sector={list(sector)} resolved to "
                        f"{len(_fs_only_k0)} (both flip parities). Pass "
                        "flip=0 or flip=1 too.")
                _fst = next((x for x in _fs_plan["stars"]
                             if int(x["k0"]) == _fs_only_k0[0]), None)
                _fid = decode_irrep_for_character(_fst, irrep)
                if isinstance(_fid, str):
                    raise ValueError(
                        f"qed.full_spectrum: irrep={irrep} could not be "
                        f"resolved on star k0={_fs_only_k0[0]}: {_fid}")
                _fs_only_irrep = [_fid[0]]
        elif irrep is not None:
            raise ValueError(
                "qed.full_spectrum: irrep= names a LITTLE-CO-GROUP irrep, "
                "which is defined per momentum star -- pass sector= too.")
        try:
            eigs = []
            _su2_blocks: list[tuple[Optional[int], list[float]]] = []
            if sz_conserved:
                if sz is not None:
                    # ONE named magnetisation block: its complete spectrum,
                    # no mirror (the caller asked for this sector, not its
                    # flip partner).
                    _sectors = [int(sz)]
                    _mirror = False
                else:
                    top = N // 2 if _flip_transport else N
                    _sectors = list(range(top + 1))
                    _mirror = _flip_transport
                for n_up in _sectors:
                    d = dict(_core.little_group_full_spectrum(
                        operator, _A, _res, n_up=int(n_up),
                        use_gpu=use_gpu, spin_flip=_sf, time_reversal=_tr,
                        only_k0=_fs_only_k0, only_irrep=_fs_only_irrep))
                    block = sorted(float(e) for e in d["eigenvalues"])
                    eigs.extend(block)
                    _su2_blocks.append((int(n_up), block))
                    if _mirror and n_up * 2 != N:
                        eigs.extend(block)   # isospectral mirror
                        _su2_blocks.append((N - int(n_up), block))
            else:
                d = dict(_core.little_group_full_spectrum(
                    operator, _A, _res, use_gpu=use_gpu,
                    spin_flip=_sf, time_reversal=_tr,
                    only_k0=_fs_only_k0, only_irrep=_fs_only_irrep))
                eigs = [float(e) for e in d["eigenvalues"]]
            if verbose:
                print("[qed.full_spectrum] spatial group -> "
                      "LITTLE-GROUP engine (factorized d_G reduction)"
                      + (", flip transport halves the Sz sweep"
                         if _flip_transport else ""))
            out = EDResults()
            out.eigenvalues = sorted(eigs)
            # Stage 12e: total-spin resolution by spectral differencing
            # (full sweep only; a restricted star walk breaks the tiling).
            _attach_su2_full_spectrum_labels(
                out, _su2_blocks, N, operator,
                enabled=(_fs_label != 0 and sz is None and sz_conserved
                         and not _fs_only_k0 and not _fs_only_irrep),
                require=(_fs_label == 1))
            return out
        except Exception as exc:            # noqa: BLE001 -- graceful
            if isinstance(point_group, str) \
                    and point_group.lower() == "full":
                raise
            if verbose:
                print(f"[qed.full_spectrum] little-group lane declined "
                      f"({exc}); falling back to the abelian streaming "
                      "path.")

    # No spatial symmetry. Sz / Sz-parity / flip-transport / time-reversal
    # still BLOCK the sweep, and eigh is O(d^3), so the blocking is a large
    # win even though it "buys nothing for the spectrum multiset" -- the old
    # single plain-dense solve here cost 32 s for a 13-site XXZ tree whose
    # Sz-blocked sweep is seconds (and its 2^N dense build was also the F6
    # first-commit race site). Route through the STREAMING sector path below
    # with the TRIVIAL spatial group (identity generator; one k=0 irrep per
    # sector): plain U(1) Sz blocks -- or the rep lane's native Sz-parity
    # half sectors for U(1)-broken-but-parity-graded models -- with the fast
    # rep-walk dense assembly and the sector-parallel FULL loop. (The
    # factorized little-group engine is the WRONG lane here: measured 16 s
    # per 2048-dim trivial-star block vs sub-second on the streaming path.)
    # Plain dense remains only when there is literally nothing to exploit.
    if info is None:
        _parity = False
        if not sz_conserved:
            try:
                _parity = bool(_core.detect_hamiltonian_symmetries(
                    operator)["sz_parity"])
            except Exception:
                _parity = False
        if sz_conserved or _parity:
            info = _normalize_symmetry_info(operator, [list(range(N))])
            if verbose and info is not None:
                print("[qed.full_spectrum] trivial spatial group -> "
                      + ("Sz-blocked" if sz_conserved
                         else "Sz-parity-blocked")
                      + " streaming sweep (no 2^N dense build)"
                      + (", flip transport halves the Sz sweep"
                         if _flip_transport else ""))
    if info is None:
        params = _bare_full_params(N, 1 << N, spin_length)
        params.use_gpu = use_gpu
        res = _diag_via_workflows_solve(
            operator, DiagonalizationMethod.FULL, params)
        res.eigenvalues = sorted(res.eigenvalues)
        return res

    # The in-memory binding receives this dict verbatim; close a raw
    # generators-only group first (same group the directory writer builds).
    info = _closed_symmetry_info(info)
    # full_spectrum is the many-small-sectors regime: turn on the C++ sector-
    # parallel FULL loop so independent (Sz, irrep) blocks are dense-diagonalised
    # across cores. Each sector's eigensolve runs single-threaded (see
    # full_diagonalization's omp_in_parallel guard) to avoid N_sectors x P
    # oversubscription. An explicit user ED_SYM_SECTOR_PARALLEL is honoured.
    _prev_sector_parallel = os.environ.get("ED_SYM_SECTOR_PARALLEL")
    if _prev_sector_parallel is None:
        os.environ["ED_SYM_SECTOR_PARALLEL"] = "1"
    try:
        if sz is not None and sz_conserved:
            # ONE named magnetisation block. Flip transport mirrors a HALF
            # sweep onto its partner; with a single named sector there is no
            # sweep to halve, so disable it rather than silently returning
            # the partner's copy too.
            sz_values: list[Optional[int]] = [int(sz)]
            _flip_transport = False
        else:
            sz_values = (
                (list(range(N // 2 + 1)) if _flip_transport
                 else list(range(N + 1)))
                if sz_conserved else [None])
        # Stage 7a star maps for the per-Sz sector loops.
        _star_maps = None
        if _star and point_group not in (False, 0, "off", "none") \
                and info is not None:
            from ..star_reduction import star_maps_from_info
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
        _su2_blocks: list[tuple[Optional[int], list[float]]] = []
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
            gs = _core.workflows_solve_streaming_symmetry(
                operator, info, N, float(spin_length), opts, n_up)
            eigs.extend(gs.eigenvalues)
            _su2_blocks.append(
                (int(n_up) if n_up is not None else None,
                 sorted(gs.eigenvalues)))
            if (_flip_transport and n_up is not None
                    and int(n_up) * 2 != N):
                # Isospectral N - n_up mirror of the whole Sz block.
                eigs.extend(gs.eigenvalues)
                _su2_blocks.append(
                    (N - int(n_up), sorted(gs.eigenvalues)))
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
        # Stage 12e: total-spin resolution by spectral differencing (needs
        # the complete Sz sweep -- a named sz block has no adjacent data).
        _attach_su2_full_spectrum_labels(
            out, _su2_blocks, N, operator,
            enabled=(_fs_label != 0 and sz is None and sz_conserved),
            require=(_fs_label == 1))
        return out
    finally:
        if _prev_sector_parallel is None:
            os.environ.pop("ED_SYM_SECTOR_PARALLEL", None)
        else:
            os.environ["ED_SYM_SECTOR_PARALLEL"] = _prev_sector_parallel


# Internal alias so :func:`solve` can call the helper without colliding
# with its own ``full_spectrum`` (bool) keyword argument.
full_spectrum_compute = full_spectrum
