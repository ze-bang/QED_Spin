"""High-level Python wrapper around ``ed::dssf`` (P2.8 / DSSF PR-G).

The C++ ``ED dssf`` subcommand calls
``ed::dssf::build_observable_pairs`` to assemble the list of observable
pairs ``(O_1, O_2, name)`` that get fed into the dynamical / static
structure-factor evaluator. This module re-exports the *same* builder so
Python notebooks and downstream scripts produce **byte-identical** observable
names and ordering, instead of hand-rolling
``Sum``/``Transverse``/``Sublattice``/``Experimental`` operator
constructors and risking drift.

Quick start
-----------

.. code-block:: python

    import quantum_ed as qed

    spec = qed.dssf.OperatorSpec()
    spec.operator_type     = "transverse"
    spec.basis             = "xyz"
    spec.spin_combinations = [("x", "x"), ("y", "y")]
    spec.momentum_points   = [[0.0, 0.0, 0.0], [3.14159, 0.0, 0.0]]
    spec.polarization      = [0.0, 0.0, 1.0]
    spec.unit_cell_size    = 4
    spec.num_sites         = 4
    spec.spin_length       = 0.5
    spec.positions_file    = "/abs/path/to/positions.dat"

    pairs = qed.dssf.build_observable_pairs(spec)
    for name in pairs.names:
        print(name)

The returned :class:`ObservablePairs` carries three parallel lists --
``obs_1``, ``obs_2``, ``names`` -- of equal length. The ``Operator`` objects
inside are the *same* C++ ``Operator`` / ``FixedSzOperator`` exposed via
:mod:`quantum_ed`, so they can be plugged directly into Lanczos / FTLM /
LTLM / Hybrid solvers without conversion.

See also
--------
:func:`compute_transverse_bases` -- helper that returns the ``(e1, e2)``
basis ``ed::dssf`` uses internally for transverse-component operators.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable, Optional, Sequence

from ._core.dssf import (  # type: ignore[attr-defined]
    ObservablePairs,
    OperatorSpec,
    build_observable_pairs,
    compute_transverse_bases,
)


# ----------------------------------------------------------------------------
# Phase 5 (Apr 2026): ``ED dssf <method>`` runner. The full
# ``ed::dssf::run(...)`` C++ entry point consumes an ``EDConfig`` (the
# hierarchical config the CLI uses) plus an ``OperatorSpec``; rather than
# bind every nested ``EDConfig`` field to Python (which would be a large
# ABI surface to maintain), we provide a thin runner that shells out to the
# canonical ``./ED dssf`` subcommand. The CLI parses a single
# ``parameters.def``-style input file, so this helper lets callers stay in
# Python while delegating the heavy lifting to the C++ workflow.
#
# Use ``ed::dssf::build_observable_pairs`` (re-exported above as
# :func:`build_observable_pairs`) when you want to stay fully in-process and
# reuse the resulting ``Operator`` objects with the in-process solvers
# (lanczos, FTLM, mTPQ); reach for :func:`run_from_directory` when you want
# the full HDF5 ``/dssf/...`` deck on disk plus the C++ continued-fraction
# accumulator.
# ----------------------------------------------------------------------------


def _resolve_ed_binary(ed_binary: Optional[str]) -> str:
    """Locate the ``ED`` executable, preferring an explicit override."""
    if ed_binary:
        if not os.path.isfile(ed_binary):
            raise FileNotFoundError(
                f"ed_binary={ed_binary!r} does not exist; pass an absolute path"
            )
        return ed_binary
    on_path = shutil.which("ED")
    if on_path is not None:
        return on_path
    raise FileNotFoundError(
        "Could not find the `ED` binary. Either build it (cmake --build "
        "<build> --target ED), put the build directory on $PATH, or pass "
        "ed_binary=/abs/path/to/ED to run_from_directory(...)."
    )


def run_from_directory(
    directory: str,
    method: str,
    *,
    ed_binary: Optional[str] = None,
    extra_args: Sequence[str] = (),
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``./ED dssf <method>`` against a directory of Hamiltonian files.

    This is the canonical Python entry point for the full DSSF / SSSF /
    static-response pipeline (the same path the C++ CLI takes). ``./ED``
    parses ``directory/parameters.def`` plus the standard ``InterAll.dat``
    / ``Trans.dat`` deck, calls
    :func:`build_observable_pairs` internally to assemble observables, and
    invokes ``ed::dssf::run(...)`` which dispatches into the
    ``compute_*_workflow`` kernels in ``src/cli/workflows.cpp``.

    Parameters
    ----------
    directory : str
        Directory containing ``parameters.def`` plus the Hamiltonian dat
        files (and ``automorphism_results/`` when symmetry-projected).
    method : str
        One of ``"dynamical_thermal"``, ``"static_thermal"``,
        ``"ground_state_dssf"``, ``"single_expectation"``. Mirrors the
        ``ed::dssf::DSSFMethod`` enum tokens.
    ed_binary : str, optional
        Absolute path to the ``ED`` binary. Defaults to ``shutil.which("ED")``.
    extra_args : sequence of str, optional
        Extra CLI flags forwarded after ``dssf <method> <directory>``;
        e.g. ``("--frequency-window", "-1.0,1.0,200")``.
    env : dict, optional
        Environment overrides for the subprocess.
    check : bool, optional
        If True (default), raise ``CalledProcessError`` on non-zero exit.
    capture_output : bool, optional
        If True, capture stdout/stderr in the returned object.

    Returns
    -------
    subprocess.CompletedProcess
        The ``./ED`` invocation result.

    See also
    --------
    build_observable_pairs : in-process observable assembly (no C++ binary).
    """
    binary = _resolve_ed_binary(ed_binary)
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"directory={directory!r} does not exist or is not a directory"
        )
    cmd = [binary, "dssf", method, directory, *extra_args]
    return subprocess.run(
        cmd,
        check=check,
        env=env,
        capture_output=capture_output,
        text=True,
    )


__all__ = [
    "ObservablePairs",
    "OperatorSpec",
    "build_observable_pairs",
    "compute_transverse_bases",
    "run_from_directory",
]
