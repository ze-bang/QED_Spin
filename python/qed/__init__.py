"""qed: Python interface to the C++ exact-diagonalization engine.

The public surface is intentionally small: **three verbs**:

* :func:`qed.solve(H, ...) <qed.solve>` -- ground-state /
  eigenvalue diagonalization. Picks the solver / device / Sz sector
  automatically; opt-out via the matching kwargs.

* :func:`qed.thermal(H, ...) <qed.thermal>` -- finite-temperature
  thermodynamics (FTLM / LTLM / mTPQ / KPM-DOS). Iterates the
  Sz axis automatically when Sz is conserved.

* :func:`qed.spectral(H, observables, ...) <qed.spectral>` --
  spectral / dynamical structure factors (ground-state continued
  fraction, FTLM dynamical). Accepts an in-memory ``H`` plus a list
  of observable operators, or a directory path for the ``./ED dssf``
  CLI workflow.

All three call into the unified C++ orchestrator
(``ed::workflows::{solve, thermal, spectral}`` in C++) and accept
plain keyword arguments -- there are no separate ``SolveOptions`` /
``ThermalOptions`` / ``SpectralOptions`` Python types.

Operators are built via :class:`qed.input.HamiltonianBuilder` (the
canonical fluent DSL) or directly via :class:`qed.Operator` /
:class:`qed.FixedSzOperator`. The C++ ``ed::make_operator(OperatorSpec)``
factory is the C++-side mirror; its Python binding lands in a follow-up
commit.

Quick start
-----------

    >>> import qed
    >>> N = 6
    >>> b = qed.input.HamiltonianBuilder(num_sites=N)
    >>> b.heisenberg(bonds=[(i, (i + 1) % N) for i in range(N)], J=1.0)
    >>> H = b.to_operator()
    >>> sorted(qed.solve(H, num_eigenvalues=2).eigenvalues)[:2]   # doctest: +SKIP
    [-2.802..., -1.0]

Submodules
----------

* :mod:`qed.input` -- lattice + Hamiltonian DSL.
* :mod:`qed.symmetry` -- programmatic permutation-group helpers.
* :mod:`qed.dssf` -- DSSF observable-pair builders (data helpers only;
  the actual workflow lives in :func:`qed.spectral`).
* :mod:`qed.bfg` -- BFG order-parameter helpers.
* :mod:`qed.mpi` -- helper for the standalone ``mpiexec
  ed_distributed_main`` binary.
* :mod:`qed.auto_tune` -- internal heuristic tuner (used by
  :func:`qed.spectral` when ``auto_tune=True``).
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
    DiagonalizationMethod,
    EDParameters,
    EDResults,
    ThermodynamicData,
    has_cuda_build,
    has_mpi_build,
)

from . import dssf  # DSSF observable-pair data helpers
from . import auto_tune  # heuristic helpers consumed internally by qed.spectral
from . import hamiltonian  # legacy Python-side fluent Hamiltonian DSL
from . import input  # standalone C++ ed_input library bindings
from . import symmetry  # programmatic site-permutation symmetry DSL
from . import bfg  # BFG order-parameter library helpers
from . import mpi  # mpiexec ed_distributed_main runner helper
from . import helpers  # re-export edlib utilities under qed.helpers
from . import workflow  # internal implementation module for qed.solve
from .workflow import (  # noqa: E402  (top-level re-exports)
    GeneratorSet,
    SymmetryReport,
    solve,
    full_spectrum,
    find_symmetries,
    list_diag_parameters,
    load_mpi_eigenvector,
    load_mpi_eigenvectors,
    solver_device_support,
)
from . import thermal as _thermal_module  # one canonical finite-T entry point
from .thermal import thermal, ThermalResult, ThermalSectorEntry  # noqa: E402
from . import spectral as _spectral_module  # one canonical spectral entry point
from .spectral import spectral  # noqa: E402

# (feasibility / pre-flight planner removed: sensible defaults instead.)

__version__: Final[str] = "0.3.0"

__all__ = [
    # Core operator types
    "Operator",
    "FixedSzOperator",
    "OP_SPLUS",
    "OP_SMINUS",
    "OP_SZ",
    # Low-level solver primitives (rarely needed; consider qed.solve instead)
    "full_diagonalization",
    "lanczos",
    "compute_thermodynamics_from_spectrum",
    "finite_temperature_lanczos",
    "low_temperature_lanczos",
    "FTLMParameters",
    "LTLMParameters",
    # Enums and parameter helpers
    "DiagonalizationMethod",
    "EDParameters",
    "EDResults",
    "ThermodynamicData",
    "has_cuda_build",
    "has_mpi_build",
    # The three canonical entry points
    "solve",
    "full_spectrum",
    "thermal",
    "spectral",
    # Result types
    "ThermalResult",
    "ThermalSectorEntry",
    # Symmetry helpers
    "GeneratorSet",
    "SymmetryReport",
    "find_symmetries",
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
    # Helpers
    "list_diag_parameters",
    "solver_device_support",
    "load_mpi_eigenvector",
    "load_mpi_eigenvectors",
    "__version__",
]
