"""``qed._solve.methods``: method taxonomy, solver picking, total-spin axis.

Carved out of ``workflow.py`` (WP11). Holds the ground-state / thermal
classification of ``DiagonalizationMethod``, the ``solver=``
canonicaliser, the SU(2) total-spin context variable every
``SolveOptions`` build reads, and the two helpers that hand an in-memory
``Operator`` to the C++ orchestrator.
"""

from __future__ import annotations

from typing import Optional, Union

from .. import _core as _core
from .._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    Operator,
)



# ---------------------------------------------------------------------------
# Surface unification (May 2026): the C++ pybind11 forwarders
# `_core.exact_diagonalization_*` have been deleted in lockstep with
# the Wave 2 / Wave 3 collapse. Every in-tree call site now routes
# through `_core.workflows_*` (ground-state + thermal) or
# `_core.workflows_solve_streaming_symmetry` (streaming-
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
from .._params import (  # noqa: E402,F401  (single conversion layer)
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



# ---------------------------------------------------------------------------
# Stage 12 (SU(2) rollout): total-spin axis plumbing.
#
# ``qed.solve(total_spin=...)`` resolves to a (two_total_spin,
# label_total_spin) pair carried through a context variable so every
# SolveOptions the dispatch tree builds (plain, fixed-Sz, streaming-
# symmetry) picks it up without threading a kwarg through each helper.
# The context is (re)set at every public entry, so a raised call cannot
# leak targeting into the next one.
# ---------------------------------------------------------------------------
import contextvars

_TOTAL_SPIN_CTX: "contextvars.ContextVar[tuple[int, int]]" = \
    contextvars.ContextVar("qed_total_spin", default=(-1, -1))


def _normalize_total_spin(total_spin, num_sites: int) -> tuple[int, int]:
    """Map the ``total_spin`` kwarg onto (two_total_spin, label_total_spin).

    ``"auto"`` -> (-1, -1): label when SU(2) holds and vectors exist.
    ``"off"`` / ``None`` / ``False`` -> (-1, 0): axis fully off.
    ``"require"`` -> (-1, 1): throw when H is not SU(2)-invariant.
    numeric S (int or half-integer) -> (2S, -1): target that tower.
    """
    if total_spin is None or total_spin is False:
        return (-1, 0)
    if isinstance(total_spin, str):
        key = total_spin.strip().lower()
        if key in ("auto", "on"):
            return (-1, -1)
        if key in ("off", "none", ""):
            return (-1, 0)
        if key == "require":
            return (-1, 1)
        raise ValueError(
            f"qed: total_spin={total_spin!r} not understood; expected "
            f"'auto', 'off', 'require', or a numeric S.")
    two_S = int(round(2.0 * float(total_spin)))
    if abs(2.0 * float(total_spin) - two_S) > 1e-9 or two_S < 0:
        raise ValueError(
            f"qed: total_spin={total_spin!r} must be a non-negative "
            f"integer or half-integer.")
    if two_S > num_sites or (two_S % 2) != (num_sites % 2):
        raise ValueError(
            f"qed: total_spin={total_spin!r} is not admissible for "
            f"{num_sites} spin-1/2 sites (needs S <= N/2 with "
            f"{'integer' if num_sites % 2 == 0 else 'half-integer'} S).")
    return (two_S, -1)


def _apply_total_spin_opts(opts) -> None:
    """Stamp the resolved total-spin pair onto a ``SolveOptions``."""
    two_S, label = _TOTAL_SPIN_CTX.get()
    opts.two_total_spin = two_S
    opts.label_total_spin = label


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
        _apply_total_spin_opts(opts)
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
        "OFTLM",
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
    auto_method: bool = False,
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
    return _diag_via_workflows_solve(operator, method, params,
                                     auto_method=auto_method)


# ---------------------------------------------------------------------------
# (Stage 11d, Jul 2026: the device='mpi' subprocess lane -- _diag_via_mpi,
# the ed_distributed_main launcher plumbing, the rank_<r>.h5 slab readers
# load_mpi_eigenvector(s), and the per-sector Z-weighted aggregation that
# only that lane produced -- was retired with the distributed-operator
# family. MPI runs go through the CLI under mpirun: SectorDistributor
# across sectors + MpiBackend in-process.)
# ---------------------------------------------------------------------------
