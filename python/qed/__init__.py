"""qed: exact diagonalization of spin-1/2 Hamiltonians over every symmetry they have.

Five verbs, one engine. Each resolves a :class:`Symmetry` against ``H`` (momenta,
little-group irreps, Sz or its parity, spin flip, time reversal, total spin) and runs
block by block on the CPU or a GPU:

* :func:`eigs` -- the lowest levels, with eigenvectors on demand;
* :func:`spectrum` -- every eigenvalue;
* :func:`thermal` -- thermodynamics: exact, FTLM (``exact_states`` for OFTLM) or mTPQ;
* :func:`dynamics` -- S(omega) at T = 0 or finite T;
* :func:`measure` -- equal-time quantities of the lowest levels in one pass: one-point values
  (:func:`expect`) and pair correlations (:func:`correlations`) of operators or
  :class:`Family` index axes, with :meth:`EigResult.matrix_element` between levels.

Operators come from :class:`qed.input.HamiltonianBuilder` or :class:`qed.Operator`.

A run that goes well prints nothing; warnings arrive as :class:`qed.errors.QEDWarning`.
:func:`set_log_level` (or ``QED_LOG_LEVEL``) routes the engine's progress into
``logging.getLogger("qed")`` or straight to a stream. Refusals raise the classes of
:mod:`qed.errors`, each also the builtin it refines (``ValueError``, ...); an argument of
the wrong type raises ``TypeError`` and an index out of range ``IndexError``.

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

from . import errors  # qed.errors: QEDError and the classes the engine raises
from .family import Family, MomentumFamily
from ._log import configure_from_env as _configure_log_from_env
from ._log import get_log_level, set_log_level

_configure_log_from_env()

from . import dssf  # observable builders for dynamics
from . import input  # lattice + Hamiltonian DSL
from . import symmetry  # programmatic site-permutation helpers
from .discovery import Symmetries, find_symmetries


def debug_env(prefix: str = "") -> str:
    """Every registered ``ED_*`` / ``QED_*`` environment variable whose name starts
    with ``prefix``, with its live value, default and one-line meaning. The table is
    ``include/ed/core/config.h`` -- the only place a variable is declared.
    Print this into bug reports: machine-to-machine behaviour differences become one
    diff instead of a grep of the tree."""
    return _core.env_dump(prefix)


def env_snapshot() -> dict:
    """``{name: value}`` of the registered environment variables that are set: the
    environment-dependent inputs of this run. Store it next to results."""
    return dict(_core.env_snapshot())


# Variables that were removed: what replaces each (named in the import warning).
_REMOVED_ENV = {
    "ED_SYM_LG_DENSE_FLOOR": "pass qed.eigs(..., dense_max_dim=...)",
    "ED_THERMAL_EXACT_SMALL": "pass qed.thermal(..., dense_max_dim=0) to always sample",
    "ED_SYM_CACHE": "the orbit-table disk cache is gone",
    "ED_SYM_CACHE_DIR": "the orbit-table disk cache is gone",
}


def _check_environment() -> None:
    """A misspelt ``ED_*`` variable is read by nothing and would fail silently, and a
    malformed value of a registered one makes every verb refuse. Both warn once at import;
    ``ED_ENV_STRICT`` set to a true flag word (``1``, ``true``, ...; read by the engine's
    parser, ``_core.env_flag``) turns the warnings into :class:`qed.errors.InvalidRequest` (job scripts that
    must not run with a typo)."""
    import difflib
    import warnings

    malformed = list(_core.env_malformed())
    if malformed:
        msg = (
            "qed: environment variable(s) whose value does not parse: "
            + ", ".join(malformed)
            + ". Every verb refuses to run until they are fixed; see qed.debug_env()."
        )
        if _core.env_flag("ED_ENV_STRICT", False):
            raise errors.InvalidRequest(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    unknown = list(_core.env_unknown())
    if not unknown:
        return
    names = list(_core.env_names())
    parts = []
    for n in sorted(unknown):
        near = difflib.get_close_matches(n, names, n=1, cutoff=0.75)
        if n in _REMOVED_ENV:
            parts.append(f"{n} (removed: {_REMOVED_ENV[n]})")
        else:
            parts.append(f"{n} (did you mean {near[0]}?)" if near else n)
    msg = (
        "qed: environment variable(s) not read by anything: "
        + ", ".join(parts)
        + ". See qed.debug_env() for the variables that exist."
    )
    if _core.env_flag("ED_ENV_STRICT", False):
        raise errors.InvalidRequest(msg)
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


_check_environment()

from ._verbs import (  # noqa: E402
    CorrelationResult,
    Correlations,
    DynamicsResult,
    EigResult,
    Expect,
    ExpectResult,
    MeasureResult,
    SpectrumResult,
    StructureFactor,
    Symmetry,
    ThermalResult,
    correlations,
    dynamics,
    eigs,
    expect,
    load_eigs,
    measure,
    spectrum,
    thermal,
)

__version__: Final[str] = _core.__version__  # pyproject.toml's version, compiled into _core

__all__ = [
    "Operator",
    "Family",
    "MomentumFamily",
    "OP_SPLUS",
    "OP_SMINUS",
    "OP_SZ",
    "Symmetry",
    "eigs",
    "EigResult",
    "load_eigs",
    "spectrum",
    "SpectrumResult",
    "thermal",
    "ThermalResult",
    "dynamics",
    "DynamicsResult",
    "measure",
    "MeasureResult",
    "Expect",
    "Correlations",
    "expect",
    "ExpectResult",
    "correlations",
    "CorrelationResult",
    "StructureFactor",
    "find_symmetries",
    "Symmetries",
    "has_cuda_build",
    "debug_env",
    "env_snapshot",
    "set_log_level",
    "get_log_level",
    "errors",
    "dssf",
    "input",
    "symmetry",
    "__version__",
]
