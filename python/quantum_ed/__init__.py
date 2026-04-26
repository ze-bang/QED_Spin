"""quantum_ed: Python interface to the C++ exact-diagonalization engine.

This package wraps the matrix-free C++ ``Operator`` / ``FixedSzOperator`` and
the family of solvers (full diagonalization, every Lanczos / Krylov / Davidson
variant, ARPACK, FTLM/LTLM/Hybrid, mTPQ/cTPQ, GPU per-sector dispatch under
streaming symmetry, symmetrised block ED) through a thin pybind11 layer.

The package has three layers of access:

1. **Solver-level** ``quantum_ed.lanczos / full_diagonalization /
   finite_temperature_lanczos / low_temperature_lanczos /
   hybrid_thermal_method``. Stable, narrowly-typed wrappers; great for
   notebook prototyping.

2. **Dispatcher-level** ``quantum_ed.exact_diagonalization_core(op, method,
   params)`` (and the directory + streaming-symmetry siblings). One Python
   function reaches every solver the ``./ED`` CLI knows about, including
   ARPACK, BLOCK_LANCZOS, KRYLOV_SCHUR, BLOCK_KRYLOV_SCHUR, DAVIDSON,
   LOBPCG, CHEBYSHEV_FILTERED, SHIFT_INVERT[_ROBUST], IRL/TRL, BICG,
   FULL/SCALAPACK, mTPQ/cTPQ, FTLM/LTLM/HYBRID. Pass a GPU method to the
   streaming or directory dispatchers and each sector / matrix-vector goes
   to a CUDA kernel (when the build was made with ``WITH_CUDA=ON``; check
   :func:`has_cuda_build`).

3. **Library-level submodules**: ``quantum_ed.input`` (lattice + Hamiltonian
   builders), ``quantum_ed.symmetry`` (programmatic permutation groups),
   ``quantum_ed.dssf`` (DSSF observable assembly + ``./ED dssf`` runner),
   ``quantum_ed.bfg`` (BFG order-parameter kernels), ``quantum_ed.mpi``
   (helper for the standalone ``mpiexec ed_distributed_main`` binary).

Example -- end-to-end Heisenberg chain
--------------------------------------

    >>> import quantum_ed as qed
    >>> N = 6
    >>> # Build the Hamiltonian programmatically.
    >>> b = qed.input.HamiltonianBuilder(num_sites=N)
    >>> nn = [(i, (i + 1) % N) for i in range(N)]
    >>> b.heisenberg(bonds=nn, J=1.0)
    >>> op = b.to_operator()
    >>> # Drive the dispatcher.
    >>> params = qed.EDParameters()
    >>> params.num_eigenvalues = 4
    >>> result = qed.exact_diagonalization_core(
    ...     op, qed.DiagonalizationMethod.LANCZOS, params)
    >>> sorted(result.eigenvalues)[:2]
    [-2.802..., -1.0]

Example -- in-process symmetry projection on the same chain
-----------------------------------------------------------

    >>> g = qed.symmetry.translation(N, 1)
    >>> info = qed.symmetry.group_from_generators(N, [g])
    >>> # Either persist `info` to automorphism_results/ on disk and call
    >>> # exact_diagonalization_streaming_symmetry, or attach it directly
    >>> # to the operator for downstream workflows that introspect it:
    >>> op.set_symmetry_info_from_dict(info)
    >>> op.get_symmetry_info_as_dict()["num_generators"]
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
    exact_diagonalization_from_directory_symmetrized,
    exact_diagonalization_fixed_sz_symmetrized,
    exact_diagonalization_streaming_symmetry,
    exact_diagonalization_streaming_symmetry_fixed_sz,
    has_cuda_build,
    has_mpi_build,
    has_scalapack_build,
)

from . import dssf  # high-level DSSF observable-pair builder + ./ED dssf runner (P2.8)
from . import hamiltonian  # legacy Python-side fluent Hamiltonian DSL (P2.10)
from . import input  # standalone C++ ed_input library bindings (Phase 4)
from . import symmetry  # programmatic site-permutation symmetry DSL (P2.11)
from . import bfg  # BFG order-parameter library helpers (P2.1)
from . import mpi  # mpiexec ed_distributed_main runner helper (Phase 5)
from . import helpers  # re-export edlib utilities under quantum_ed.helpers

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
    "exact_diagonalization_from_directory_symmetrized",
    "exact_diagonalization_fixed_sz_symmetrized",
    "exact_diagonalization_streaming_symmetry",
    "exact_diagonalization_streaming_symmetry_fixed_sz",
    "has_cuda_build",
    "has_mpi_build",
    "has_scalapack_build",
    # Submodules
    "dssf",
    "hamiltonian",
    "input",
    "symmetry",
    "bfg",
    "mpi",
    "helpers",
    "__version__",
]
