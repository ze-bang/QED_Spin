"""qed: exact diagonalization of spin-1/2 Hamiltonians over every symmetry they have.

Five verbs, one engine. Each resolves a :class:`Symmetry` against ``H`` (momenta,
little-group irreps, Sz or its parity, spin flip, time reversal, total spin) and runs
block by block on the CPU or a GPU:

* :func:`eigs` -- the lowest levels, with eigenvectors on demand;
* :func:`spectrum` -- every eigenvalue;
* :func:`thermal` -- thermodynamics: exact, FTLM (``exact_states`` for OFTLM) or mTPQ;
* :func:`dynamics` -- S(omega) at T = 0 or finite T;
* :func:`expect` -- expectation values in the lowest levels (and
  :meth:`EigResult.matrix_element` between them).

Operators come from :class:`qed.input.HamiltonianBuilder` or :class:`qed.Operator`.

    >>> import qed
    >>> b = qed.input.HamiltonianBuilder(6)
    >>> b.heisenberg(bonds=[(i, (i + 1) % 6) for i in range(6)], J=1.0)   # doctest: +SKIP
    >>> qed.eigs(b.to_operator(), 2).energies                              # doctest: +SKIP
"""

from __future__ import annotations

import os as _os
from typing import Final

# The compiled extension lives in a build directory, not in the source tree; see
# _locate_core for the QED_CORE_DIR contract. Must run before the first `_core` import.
from ._locate_core import extend_package_path as _extend_package_path

__path__ = _extend_package_path(__path__, _os.path.dirname(_os.path.abspath(__file__)))

from . import _core as _core
from ._core import OP_SMINUS, OP_SPLUS, OP_SZ, Operator, has_cuda_build

from . import dssf  # observable builders for dynamics
from . import input  # lattice + Hamiltonian DSL
from . import lattice  # lattice geometries: space group + physical labels
from . import symmetry  # programmatic site-permutation helpers
from .discovery import GeneratorSet, SymmetryReport, find_symmetries


def debug_env(prefix: str = "") -> str:
    """Every registered ``ED_*`` / ``QED_*`` environment variable whose name starts
    with ``prefix``, with its live value, default and one-line meaning. The table is
    ``include/ed/config/env_registry.h`` -- the only place a variable is declared.
    Print this into bug reports: machine-to-machine behaviour differences become one
    diff instead of a grep of the tree."""
    return _core.env_dump(prefix)


def env_snapshot() -> dict:
    """``{name: value}`` of the registered environment variables that are set: the
    environment-dependent inputs of this run. Store it next to results."""
    return dict(_core.env_snapshot())


def _check_environment() -> None:
    """A misspelt ``ED_*`` variable is read by nothing and used to fail silently.
    Unknown names warn once at import; ``ED_ENV_STRICT=1`` turns the warning into an
    error (job scripts that must not run with a typo)."""
    import difflib
    import warnings

    unknown = list(_core.env_unknown())
    if not unknown:
        return
    names = list(_core.env_names())
    parts = []
    for n in sorted(unknown):
        near = difflib.get_close_matches(n, names, n=1, cutoff=0.75)
        parts.append(f"{n} (did you mean {near[0]}?)" if near else n)
    msg = ("qed: environment variable(s) not read by anything: " + ", ".join(parts)
           + ". See qed.debug_env() for the variables that exist.")
    if _os.environ.get("ED_ENV_STRICT", "") not in ("", "0"):
        raise RuntimeError(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


_check_environment()

from .api import (DynamicsResult, EigResult, ExpectResult, SpectrumResult, Symmetry,  # noqa: E402
                  ThermalResult, dynamics, eigs, expect, spectrum, thermal)

__version__: Final[str] = "0.4.0"

__all__ = [
    "Operator", "OP_SPLUS", "OP_SMINUS", "OP_SZ",
    "Symmetry", "eigs", "EigResult", "spectrum", "SpectrumResult", "thermal", "ThermalResult",
    "dynamics", "DynamicsResult", "expect", "ExpectResult",
    "find_symmetries", "GeneratorSet", "SymmetryReport",
    "has_cuda_build", "debug_env", "env_snapshot",
    "dssf", "input", "lattice", "symmetry",
    "__version__",
]
