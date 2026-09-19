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
* :mod:`qed.auto_tune` -- internal heuristic tuner (used by
  :func:`qed.spectral` when ``auto_tune=True``).
"""

from __future__ import annotations

import os as _os
from typing import Final

# The compiled extension lives in a build directory, not in the source tree; see
# _locate_core for the QED_CORE_DIR contract. Must run before the first `_core` import.
from ._locate_core import extend_package_path as _extend_package_path

__path__ = _extend_package_path(__path__, _os.path.dirname(_os.path.abspath(__file__)))

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
from . import helpers  # re-export edlib utilities under qed.helpers
from . import lattice  # lattice geometries: space group + physical labels
from . import little_group  # labelled block spectra and <n|O|n>
from . import workflow  # internal implementation module for qed.solve
from .workflow import (  # noqa: E402  (top-level re-exports)
    GeneratorSet,
    SymmetryReport,
    solve,
    full_spectrum,
    find_symmetries,
    list_diag_parameters,
    solver_device_support,
)


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
    "helpers",
    "lattice",
    "little_group",
    "workflow",
    # Helpers
    "list_diag_parameters",
    "solver_device_support",
    "__version__",
]
