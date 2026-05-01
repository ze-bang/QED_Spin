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
    "pick_method",
    "compute",
]


# ----------------------------------------------------------------------------
# Phase 9 (Apr 2026): stress-free auto-pilot mirroring `qed.diag`.
#
# `qed.dssf.compute(directory, T=..., omega=..., ...)` picks the right
# `DSSFMethod` from the (T, omega) tuple, then delegates to the canonical
# `./ED dssf <method> <directory>` runner. The selection rule mirrors the
# C++ `ed::dssf::DSSFMethod` enum (see include/ed/dssf/dssf_engine.h):
#
#   T is None,    omega is None     -> "single_expectation"   (zero-T <O>)
#   T is None,    omega is not None -> "ground_state_dssf"    (T=0 S(Q,ω))
#   T is not None, omega is None    -> "static_thermal"       (S(Q,T))
#   T is not None, omega is not None-> "dynamical_thermal"    (S(Q,ω,T))
#
# This is the "I just want the answer" entry point. Users who need
# fine-grained control over individual workflow knobs (frequency window,
# Krylov dim, num_random_states, ...) keep using `run_from_directory`
# directly with `extra_args=`.
# ----------------------------------------------------------------------------


_VALID_METHODS: tuple[str, ...] = (
    "dynamical_thermal",
    "static_thermal",
    "ground_state_dssf",
    "single_expectation",
)


def pick_method(*, T: Optional[float | Iterable[float]] = None,
                omega: Optional[Iterable[float]] = None) -> str:
    """Return the DSSF method token for a (T, omega) tuple.

    Parameters
    ----------
    T : float, sequence of floats, or None
        Temperature axis. ``None`` means the user wants T=0 (ground-state
        path). A scalar or any non-empty iterable selects a thermal
        kernel.
    omega : sequence of floats or None
        Frequency axis. ``None`` means the user wants no ω-resolved
        spectrum (static / single-expectation path).

    Returns
    -------
    str
        One of ``"dynamical_thermal"``, ``"static_thermal"``,
        ``"ground_state_dssf"``, ``"single_expectation"`` -- the same
        tokens accepted by :func:`run_from_directory`.

    Examples
    --------
    >>> pick_method(T=None, omega=None)
    'single_expectation'
    >>> pick_method(T=0.5, omega=None)
    'static_thermal'
    >>> pick_method(T=None, omega=[0.0, 0.1, 0.2])
    'ground_state_dssf'
    >>> pick_method(T=[0.1, 1.0], omega=[0.0, 0.5])
    'dynamical_thermal'
    """
    has_T = T is not None
    has_w = omega is not None
    if has_T and has_w:
        return "dynamical_thermal"
    if has_T:
        return "static_thermal"
    if has_w:
        return "ground_state_dssf"
    return "single_expectation"


def compute(
    directory: str,
    *,
    T: Optional[float | Iterable[float]] = None,
    omega: Optional[Iterable[float]] = None,
    method: Optional[str] = None,
    ed_binary: Optional[str] = None,
    extra_args: Sequence[str] = (),
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
    verbose: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Auto-pilot DSSF runner -- one call, no method picking required.

    Mirrors :func:`quantum_ed.diag` for spectral / structure-factor
    computations. The DSSF method is auto-selected from whether ``T``
    and/or ``omega`` are supplied (see :func:`pick_method`); pass
    ``method=`` to override the auto-rule. Everything else is delegated
    verbatim to :func:`run_from_directory`.

    Parameters
    ----------
    directory : str
        Directory containing ``parameters.def`` plus the Hamiltonian
        deck (the same layout :func:`run_from_directory` consumes).
    T : float, sequence of floats, or None, optional
        Temperature(s). ``None`` (default) → T=0 path.
    omega : sequence of floats or None, optional
        Frequency grid. ``None`` (default) → no ω-resolved output.
    method : str, optional
        Explicit override for the auto-selected method token.
    ed_binary, extra_args, env, check, capture_output : see
        :func:`run_from_directory`.
    verbose : bool, optional
        If True (default), print one ``[qed.dssf.compute]`` line
        announcing the auto-selected method.

    Returns
    -------
    subprocess.CompletedProcess
        The ``./ED dssf`` invocation result.

    Examples
    --------
    Static structure factor at one temperature:

    .. code-block:: python

        qed.dssf.compute("runs/heisenberg6", T=0.5)

    T=0 dynamical S(Q, ω):

    .. code-block:: python

        qed.dssf.compute("runs/heisenberg6", omega=np.linspace(-2, 2, 200))

    Full S(Q, ω, T):

    .. code-block:: python

        qed.dssf.compute("runs/heisenberg6",
                         T=[0.1, 0.3, 1.0],
                         omega=np.linspace(-2, 2, 200))
    """
    if method is None:
        chosen = pick_method(T=T, omega=omega)
    else:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method={method!r} is not a recognised DSSF method token. "
                f"Valid tokens: {_VALID_METHODS}."
            )
        chosen = method
    if verbose:
        has_T = T is not None
        has_w = omega is not None
        print(f"[qed.dssf.compute] method={chosen!r} "
              f"(T given: {has_T}, omega given: {has_w})")
    return run_from_directory(
        directory,
        chosen,
        ed_binary=ed_binary,
        extra_args=tuple(extra_args),
        env=env,
        check=check,
        capture_output=capture_output,
    )
