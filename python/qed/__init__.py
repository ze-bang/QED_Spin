"""qed: Python interface to the C++ exact-diagonalization engine.

This package wraps the matrix-free C++ ``Operator`` / ``FixedSzOperator`` and
the family of solvers (full diagonalization, every Lanczos / Krylov / Davidson
variant, ARPACK, FTLM/LTLM/Hybrid, mTPQ/cTPQ, GPU per-sector dispatch under
streaming symmetry, symmetrised block ED) through a thin pybind11 layer.

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
     function reaches every solver the ``./ED`` CLI knows about, including
     ARPACK, BLOCK_LANCZOS, KRYLOV_SCHUR, BLOCK_KRYLOV_SCHUR, DAVIDSON,
     LOBPCG, CHEBYSHEV_FILTERED, SHIFT_INVERT[_ROBUST], IRL/TRL, BICG,
     FULL/SCALAPACK, mTPQ/cTPQ, FTLM/LTLM/HYBRID. Pass a GPU method to the
     streaming or directory dispatchers and each sector / matrix-vector goes
     to a CUDA kernel (when the build was made with ``WITH_CUDA=ON``; check
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
    hybrid_thermal_method,
    FTLMParameters,
    LTLMParameters,
    HybridThermalParameters,
    # Phase 5 (Apr 2026): high-level dispatcher + symmetry setter +
    # streaming/directory dispatchers + build introspection.
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
    has_scalapack_build,
    canonicalize_method,
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
    "hybrid_thermal_method",
    "FTLMParameters",
    "LTLMParameters",
    "HybridThermalParameters",
    # Phase 5 dispatcher surface
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
    "has_scalapack_build",
    "canonicalize_method",
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
