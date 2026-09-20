"""``qed._solve.symmetry_lane``: the in-memory streaming-symmetry lane.

Carved out of ``workflow.py`` (WP11). One function: hand an operator plus
its (closed) group to ``_core.workflows_{solve,thermal}_streaming_symmetry``
with the composition toggles -- Sz / Sz-parity subspace, spin flip, time
reversal, star reduction -- resolved onto the options struct.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .. import _core as _core
from .._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    FixedSzOperator,
    Operator,
)
from .._params import (
    ed_params_to_solve_options as _ed_params_to_solve_options,
    ed_params_to_thermal_options as _ed_params_to_thermal_options,
    ed_result_from_gs_result as _ed_result_from_gs_result,
    ed_result_from_thermal_result as _ed_result_from_thermal_result,
)
from ..discovery import GeneratorSet, resolve_discrete_toggle
from .methods import (
    _apply_total_spin_opts,
    _diag_via_workflows_solve,
    _is_thermal_method,
)
from .symmetry_input import (
    SymmetryArg,
    _closed_symmetry_info,
    _resolve_sector_quantum_numbers,
)


def _diag_with_symmetry(
    operator: Operator,
    symmetry: SymmetryArg,
    params: EDParameters,
    method: DiagonalizationMethod,
    *,
    sz: Optional[int],
    verbose: bool,
    sector: Optional[Sequence[int]] = None,
    spin_flip="auto",
    time_reversal="auto",
    point_group="auto",
    sz_parity: Optional[int] = None,
    auto_sz_axis: bool = True,
    auto_method: bool = False,
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
    from ..symmetry import group_from_generators

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
    # Close a raw generators-only dict here, before anything reads the sector
    # table: the in-memory binding below receives exactly this dict, so the
    # sector= lookup and the C++ group share one table.
    info = _closed_symmetry_info(info)

    # `sector=` names QUANTUM NUMBERS; selected_sectors takes raw INDICES.
    # info['sectors'] is the table that maps between them (the same table the
    # C++ group is built from), so resolve here -- the first point where it
    # exists.
    if sector is not None:
        _sid = _resolve_sector_quantum_numbers(info, sector)
        # GAP 9 fix (extended 2026-07): the sector set may carry EXTENDED
        # indices in SLOTS of n_raw. Naming a momentum means every slot at
        # that momentum -- the same contract the project lane's decoder
        # honours. Slot layouts emitted by the factories:
        #   1 slot  -- no discrete extension
        #   2 slots -- flip halves (k,+)=k, (k,-)=k+n_raw  OR the two
        #              Sz-parity halves
        #   4 slots -- Sz-parity x flip: slot = 2*parity + sign, index =
        #              slot*n_raw + k
        # filter_sectors silently drops out-of-range ids, so selecting all
        # four candidates is a no-op for the smaller layouts. Selecting
        # only [k, k+n] silently dropped the odd-parity half in the 4-slot
        # layout (audit 2026-07-30: 120/256 states missing on a J++ ring).
        _n_raw = len(info.get("sectors", []) or [])
        params.selected_sectors = [_sid + _s * _n_raw for _s in range(4)]
        if verbose:
            print(f"[qed.solve] sector={list(sector)} -> raw sector index "
                  f"{_sid} (+ extended slot partners "
                  f"{[_sid + _s * _n_raw for _s in range(1, 4)]} when engaged)")

    # ------------------------------------------------------------------
    # 2. Hand operator + group to the streaming kernel in memory (the
    #    bindings copy the operator's terms and build the group from
    #    ``info`` with the directory writer's phase convention).
    # ------------------------------------------------------------------
    if verbose:
        print(f"[qed.solve] symmetry projection: |G|="
              f"{len(info.get('max_clique', []))}, "
              f"sectors={len(info.get('sectors', []))}")

    # Route through the unified orchestrator's streaming-symmetry
    # helper. It composes `ed::make_operator(streaming_symmetry=true)`
    # with a per-sector `ed::workflows::solve` loop -- the same
    # CLI path the C++ `run_streaming_symmetry_workflow` exercises.
    #
    # Phase B of the "Backend x Symmetries x Workflows" plan
    # (May 2026): thermal methods (FTLM / LTLM / mTPQ /
    # KPM_DOS) now route through the matching
    # ``workflows_thermal_streaming_symmetry`` binding,
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
        # fixed_sz=fixed_sz_n_up). So we just hand it the operator +
        # group + sites + spin_l and the binding takes care of
        # composing the per-sector thermal lane.
        if _parity_mode is not None:
            topts.sz_parity = int(_parity_mode)
        topts.spin_flip = resolve_discrete_toggle(
            operator, spin_flip, "spin_flip", verbose=verbose)
        topts.time_reversal = resolve_discrete_toggle(
            operator, time_reversal, "time_reversal", verbose=verbose)
        tr = _core.workflows_thermal_streaming_symmetry(
            operator,
            info,
            int(operator.num_sites),
            float(params.spin_length),
            topts,
            fixed_sz_n_up,
        )
        return _ed_result_from_thermal_result(tr)

    # Ground-state lane (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR /
    # FULL) -- the original behaviour.
    # 2026-09-11: honour "no solver named" as SolveMethod::Auto so per-sector
    # windows get Krylov-Schur (single-vector Lanczos drops degenerate
    # copies: measured 2e-2 on a staggered-field chain window).
    opts = _ed_params_to_solve_options(params, method, auto_method=auto_method)
    _apply_total_spin_opts(opts)
    opts.use_symmetry = True
    # Stage 8 composition toggles: -1 auto / 0 off / 1 require,
    # with 'on' = auto + detection report (warn-and-continue when
    # the Hamiltonian lacks the symmetry).
    opts.spin_flip = resolve_discrete_toggle(
        operator, spin_flip, "spin_flip", verbose=verbose)
    # Correctness (2026-09-11): the in-sector flip projection of the abelian
    # lane is eigenvalues-only (it has no orbit form to reconstruct vectors
    # from), so a vector request silently came back WITHOUT vectors. Keep
    # flip transport but disable the projection when vectors are wanted;
    # 'require' + vectors is a contradiction and raises.
    if bool(getattr(params, "compute_eigenvectors", False)) and opts.spin_flip != 0:
        if opts.spin_flip == 1:
            raise ValueError(
                "qed.solve: spin_flip='require' with compute_eigenvectors=True is not "
                "supported on the abelian symmetry lane (the flip projection is "
                "eigenvalues-only); use spin_flip='off' or point_group='full'.")
        if verbose:
            print("[qed.solve] compute_eigenvectors: in-sector spin-flip projection "
                  "disabled (eigenvalues-only lane); flip transport is kept.")
        opts.spin_flip = 0
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
        from ..star_reduction import star_maps_from_info
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
    gs = _core.workflows_solve_streaming_symmetry(
        operator,
        info,
        int(operator.num_sites),
        float(params.spin_length),
        opts,
        fixed_sz_n_up,
    )
    return _ed_result_from_gs_result(gs, params)
