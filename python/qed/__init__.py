"""qed: Python interface to the C++ exact-diagonalization engine.

This package wraps the matrix-free C++ ``Operator`` / ``FixedSzOperator`` and
the family of solvers (full diagonalization, Lanczos, Block Lanczos,
Krylov-Schur, FTLM/LTLM, mTPQ/cTPQ, KPM_DOS, GPU per-sector dispatch under
streaming symmetry) through a thin pybind11 layer.

The package has **two stress-free entry points** plus three layers of
deeper access for advanced use:

* **One-call API** (the recommended path for routine work)

  - :func:`qed.diag(H, ...) <qed.diag>` — exact diagonalisation of any
    Hamiltonian. Auto-picks solver / device (CPU / GPU / MPI / MPI+GPU) /
    Sz sector / pre-flight planner.
  - :func:`qed.dssf.compute(directory, T=, omega=, ...) <qed.dssf.compute>`
    — every structure-factor / spectral-function computation, finite or
    zero temperature. Auto-picks the DSSF method **and** the internal knobs
    (η broadening, ω window, FTLM Krylov dim, # random vectors, KPM moments,
    device backend).

  See ``docs/guides/one_call_api.md`` for the canonical documentation of
  both entry points, including the auto-selection rules and per-knob
  overrides.

* **Low-level access** for niche / programmatic use:

  1. **Solver-level** ``qed.lanczos / full_diagonalization /
     finite_temperature_lanczos / low_temperature_lanczos /
     hybrid_thermal_method``. Stable, narrowly-typed wrappers; great for
     notebook prototyping.

  2. **Dispatcher-level** ``qed.exact_diagonalization_core(op, method,
     params)`` (and the directory + streaming-symmetry siblings). One Python
     function reaches every solver the ``./ED`` CLI knows about:
     LANCZOS, BLOCK_LANCZOS, KRYLOV_SCHUR, FULL, mTPQ, cTPQ, FTLM, LTLM,
     KPM_DOS. Set ``params.use_gpu = True`` on a CUDA-enabled build to
     route each sector / matrix-vector through the CUDA kernels (check
     :func:`has_cuda_build`).

  3. **Library-level submodules**: ``qed.input`` (lattice + Hamiltonian
     builders), ``qed.symmetry`` (programmatic permutation groups),
     ``qed.dssf`` (DSSF observable assembly + ``./ED dssf`` runner),
     ``qed.auto_tune`` (heuristic helpers for the DSSF auto-tuner),
     ``qed.bfg`` (BFG order-parameter kernels), ``qed.mpi``
     (helper for the standalone ``mpiexec ed_distributed_main`` binary).

Example -- one-call diagonalisation
-----------------------------------

    >>> import qed
    >>> N = 6
    >>> b = qed.input.HamiltonianBuilder(num_sites=N)
    >>> b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
    >>> H = b.to_operator()
    >>> sorted(qed.diag(H, num_eigenvalues=2).eigenvalues)[:2]   # doctest: +SKIP
    [-2.802..., -1.0]

Example -- one-call DSSF
------------------------

    >>> import numpy as np                                       # doctest: +SKIP
    >>> qed.dssf.compute("runs/heisenberg6",                     # doctest: +SKIP
    ...                  T=[0.1, 0.5],
    ...                  omega=np.linspace(-2, 2, 200))

Example -- low-level dispatcher (when you want full control)
------------------------------------------------------------

    >>> params = qed.EDParameters()
    >>> params.num_eigenvalues = 4
    >>> result = qed.exact_diagonalization_core(
    ...     H, qed.DiagonalizationMethod.LANCZOS, params)        # doctest: +SKIP

Example -- in-process symmetry projection on the same chain
-----------------------------------------------------------

    >>> g = qed.symmetry.translation(N, 1)
    >>> info = qed.symmetry.group_from_generators(N, [g])
    >>> H.set_symmetry_info_from_dict(info)
    >>> H.get_symmetry_info_as_dict()["num_generators"]
    1

See ``docs/guides/python_advanced.md`` for the full advanced-use catalogue.
"""

from __future__ import annotations

from typing import Final

from . import _core as _core
from ._core import (
    Operator,
    FixedSzOperator,
    OP_SPLUS,
    OP_SMINUS,
    OP_SZ,
    full_diagonalization,
    lanczos,
    compute_thermodynamics_from_spectrum,
    finite_temperature_lanczos,
    low_temperature_lanczos,
    FTLMParameters,
    LTLMParameters,
    # High-level dispatcher + symmetry setter + streaming/directory
    # dispatchers + build introspection.
    DiagonalizationMethod,
    HamiltonianFileFormat,
    EDParameters,
    EDResults,
    ThermodynamicData,
    exact_diagonalization_core,
    exact_diagonalization_from_directory,
    exact_diagonalization_streaming_symmetry,
    exact_diagonalization_streaming_symmetry_fixed_sz,
    has_cuda_build,
    has_mpi_build,
)

from . import dssf  # high-level DSSF observable-pair builder + ./ED dssf runner (P2.8)
from . import auto_tune  # Phase 9.2: heuristic helpers for the DSSF auto-tuner
from . import hamiltonian  # legacy Python-side fluent Hamiltonian DSL (P2.10)
from . import input  # standalone C++ ed_input library bindings (Phase 4)
from . import symmetry  # programmatic site-permutation symmetry DSL (P2.11)
from . import bfg  # BFG order-parameter library helpers (P2.1)
from . import mpi  # mpiexec ed_distributed_main runner helper (Phase 5)
from . import helpers  # re-export edlib utilities under qed.helpers
from . import workflow  # Phase 9: stress-free unified workflow API
from .workflow import (  # noqa: E402  (top-level re-exports)
    GeneratorSet,
    SymmetryReport,
    diag,
    find_symmetries,
    list_diag_parameters,
    load_mpi_eigenvector,
    load_mpi_eigenvectors,
    solver_device_support,
)
from . import thermal as _thermal_module  # matvec-unification audit: one canonical finite-T entry point
from .thermal import thermal, ThermalResult, ThermalSectorEntry  # noqa: E402

# ---------------------------------------------------------------------------
# Minimalist ED Collapse (May 2026) — canonical Python entry points.
# ---------------------------------------------------------------------------
# The collapse renames `qed.diag` -> `qed.solve` and adds `qed.spectral`
# alongside the existing `qed.thermal`. The legacy names remain available
# as thin deprecation aliases so existing notebooks continue to work.
#
#   * qed.solve(H, ...)               <- preferred (was qed.diag)
#   * qed.thermal(directory, ...)     <- unchanged (audit follow-up)
#   * qed.spectral(directory, ...)    <- preferred (was qed.dssf.compute)
#
# See docs/MIGRATION.md for the full porting guide.
# ---------------------------------------------------------------------------
import warnings as _warnings  # noqa: E402


def solve(*args, **kwargs):
    """Canonical ground-state / eigenvalue entry point.

    This is the Minimalist ED Collapse name for what used to be
    :func:`qed.diag`. Same signature, same return type; new code MUST
    use this name. ``qed.diag`` is preserved as a deprecation alias.

    For direct access to the unified C++ orchestrator (skipping the
    auto-pilot layer), use :func:`qed.workflows.solve` instead.
    """
    return diag(*args, **kwargs)


def spectral(*args, **kwargs):
    """Canonical spectral-function / DSSF entry point.

    Thin alias for :func:`qed.dssf.compute`. New code MUST use this name;
    ``qed.dssf.compute`` is preserved as a deprecation alias.

    For direct access to the unified C++ orchestrator, use
    :func:`qed.workflows.spectral` instead.
    """
    return dssf.compute(*args, **kwargs)


# Full Unified-Interface Collapse (May 2026): expose the structured
# ``workflows.*`` module that wraps the C++ orchestrator surface
# (`ed::workflows::solve / thermal / spectral`) directly.
from . import workflows  # noqa: E402, F401


def _deprecated_alias(new_name, fn):
    """Wrap ``fn`` so calling it emits a ``DeprecationWarning`` pointing
    at ``new_name``. Returns a function with the same signature."""

    def _wrapper(*args, **kwargs):
        _warnings.warn(
            f"{fn.__name__}() is deprecated; use qed.{new_name}() instead. "
            f"See docs/MIGRATION.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        return fn(*args, **kwargs)

    _wrapper.__name__ = fn.__name__
    _wrapper.__qualname__ = fn.__qualname__
    _wrapper.__doc__ = (
        f"Deprecated. Use :func:`qed.{new_name}` instead.\n\n"
        + (fn.__doc__ or "")
    )
    return _wrapper


# Deprecation aliases. These do NOT remove the underlying name; they only
# wrap the user-facing call site so notebooks get a one-time warning the
# first time they import-and-call. The underlying C++/Python implementation
# is unchanged.
#
# NOTE: we deliberately do not wrap `lanczos` / `finite_temperature_lanczos`
# at the package level because they are also used as method tags (e.g.
# `solver='lanczos'` strings) — the warnings would fire for completely
# benign re-imports inside `workflow.py`. The deprecation lives in the
# module docstring + MIGRATION.md instead.
#
# ED Cleanup Sweep Phase 3 (May 2026): wrap the legacy
# `exact_diagonalization_*` family with deprecation aliases pointing
# to the canonical `qed.solve` / `qed._core.workflows_*` surface.
# The legacy names continue to work; Phase 5 collapses
# `dispatcher_bindings.cpp` to alias-forwarders into `workflows_*`,
# at which point the deprecation message becomes load-bearing.
exact_diagonalization_core = _deprecated_alias(
    "solve", exact_diagonalization_core)
exact_diagonalization_from_directory = _deprecated_alias(
    "solve", exact_diagonalization_from_directory)
exact_diagonalization_streaming_symmetry = _deprecated_alias(
    "solve", exact_diagonalization_streaming_symmetry)
exact_diagonalization_streaming_symmetry_fixed_sz = _deprecated_alias(
    "solve", exact_diagonalization_streaming_symmetry_fixed_sz)
from . import feasibility  # Phase 9 / Layer 6: pre-flight planner
from .feasibility import (  # noqa: E402
    BasisChoice,
    FeasibilityReport,
    HostResources,
    OperatorMetadata,
    ResourceError,
    WorkflowCandidate,
    WorkflowSuggestion,
    estimate_resources,
    inspect_operator,
    probe_host,
    suggest_workflow,
)

__version__: Final[str] = "0.2.0"

__all__ = [
    "Operator",
    "FixedSzOperator",
    "OP_SPLUS",
    "OP_SMINUS",
    "OP_SZ",
    "full_diagonalization",
    "lanczos",
    "compute_thermodynamics_from_spectrum",
    "finite_temperature_lanczos",
    "low_temperature_lanczos",
    "FTLMParameters",
    "LTLMParameters",
    # Dispatcher surface
    "DiagonalizationMethod",
    "HamiltonianFileFormat",
    "EDParameters",
    "EDResults",
    "ThermodynamicData",
    "exact_diagonalization_core",
    "exact_diagonalization_from_directory",
    "exact_diagonalization_streaming_symmetry",
    "exact_diagonalization_streaming_symmetry_fixed_sz",
    "has_cuda_build",
    "has_mpi_build",
    # Audit follow-up: one canonical finite-T entry point
    "thermal",
    "ThermalResult",
    "ThermalSectorEntry",
    # Minimalist ED Collapse (May 2026) — canonical entry points
    "solve",
    "spectral",
    # Submodules
    "dssf",
    "auto_tune",
    "hamiltonian",
    "input",
    "symmetry",
    "bfg",
    "mpi",
    "helpers",
    "workflow",
    # Phase 9 unified workflow API
    "diag",
    "find_symmetries",
    "list_diag_parameters",
    "solver_device_support",
    "load_mpi_eigenvector",
    "load_mpi_eigenvectors",
    "GeneratorSet",
    "SymmetryReport",
    # Phase 9 / Layer 6: feasibility planner
    "feasibility",
    "estimate_resources",
    "suggest_workflow",
    "probe_host",
    "inspect_operator",
    "FeasibilityReport",
    "WorkflowSuggestion",
    "WorkflowCandidate",
    "ResourceError",
    "HostResources",
    "OperatorMetadata",
    "BasisChoice",
    "__version__",
]
