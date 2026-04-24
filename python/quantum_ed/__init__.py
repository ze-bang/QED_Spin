"""quantum_ed: Python interface to the C++ exact-diagonalization engine.

This package wraps the matrix-free C++ ``Operator``/``FixedSzOperator`` and
the family of solvers (full diagonalization, Lanczos, FTLM/LTLM/Hybrid)
through a thin pybind11 layer. Vectors flow as NumPy ``complex128`` arrays.

Example
-------

    >>> import numpy as np
    >>> import quantum_ed as qed
    >>> N = 4
    >>> op = qed.Operator(num_sites=N)
    >>> for i in range(N - 1):
    ...     j = i + 1
    ...     op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    ...     op.add_two_body(qed.OP_SPLUS,  i, qed.OP_SMINUS, j, 0.5)
    ...     op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS,  j, 0.5)
    >>> eigs = qed.full_diagonalization(op)
    >>> float(eigs.min())   # ground state of 4-site Heisenberg OBC
    -1.6160254037844388

The :mod:`quantum_ed.helpers` sub-package re-exports the legacy ``edlib``
geometry helpers so existing notebooks keep working.
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
)

from . import helpers  # re-export edlib utilities under quantum_ed.helpers

__version__: Final[str] = "0.1.0"

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
    "helpers",
    "__version__",
]
