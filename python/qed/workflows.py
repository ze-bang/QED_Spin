"""qed.workflows --- the canonical Python entry points for the unified
exact-diagonalization interface.

The Full Unified-Interface Collapse (May 2026) introduces three
top-level workflow functions that supersede the historical
``qed.exact_diagonalization_*`` family:

* :func:`qed.workflows.solve(op, opts)`     -- ground-state eigenproblem
* :func:`qed.workflows.thermal(op, opts)`   -- finite-temperature
  workflows (FTLM / LTLM / mTPQ / cTPQ / KPM-DOS).
* :func:`qed.workflows.spectral(op, obs, opts)` -- dynamical
  correlators (ground-state continued fraction or FTLM dynamical).

Each function is a thin Python wrapper over the corresponding pybind11
binding (``_core.workflows_solve / workflows_thermal /
workflows_spectral``) which in turn invokes the C++ orchestrator
defined in ``include/ed/orchestrator.h``. Backend (CPU / GPU /
MPI / MPI+GPU) is auto-selected internally via
``ed::select_backend``.

The high-level :func:`qed.diag` / :func:`qed.dssf.compute` /
:func:`qed.thermal` entry points still exist for the smart defaults
they provide (auto-Sz, auto-method, auto-tune); ``qed.workflows.*`` is
the recommended path for callers who want explicit, structured access
to the unified C++ orchestrator surface without the auto-pilot layer.

Example
-------

>>> import qed
>>> from qed import workflows
>>> # Build the Hamiltonian programmatically.
>>> b = qed.input.HamiltonianBuilder(num_sites=6)
>>> b.heisenberg(bonds=[(i, (i + 1) % 6) for i in range(6)], J=1.0)
>>> H = b.to_operator()
>>>
>>> # Ground-state solve.
>>> opts = workflows.SolveOptions()
>>> opts.num_eigs = 2
>>> opts.tolerance = 1e-12
>>> opts.method = workflows.SolveMethod.Lanczos
>>> result = workflows.solve(H, opts)
>>> print(result.eigenvalues[:2])
[-2.802..., -1.0]
"""

from __future__ import annotations

from typing import Sequence, Union

from . import _core

# Re-export the pybind11-defined option / result / method-tag classes
# so callers can write `qed.workflows.SolveOptions(...)` etc. without
# reaching into the internal ``_core`` module.
SolveOptions = _core.SolveOptions
ThermalOptions = _core.ThermalOptions
SpectralOptions = _core.SpectralOptions

SolveMethod = _core.SolveMethod
ThermalMethod = _core.ThermalMethod
SpectralMethod = _core.SpectralMethod

GroundStateResult = _core.GroundStateResult
ThermalResult = _core.ThermalResult
SpectralResult = _core.SpectralResult

BackendConstraints = _core.BackendConstraints
BackendMetadata = _core.BackendMetadata
KrylovDiagnostics = _core.KrylovDiagnostics


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def solve(op, opts: SolveOptions = None) -> GroundStateResult:
    """Run the unified ground-state eigenproblem workflow on ``op``.

    Parameters
    ----------
    op
        A :class:`qed.Operator` or :class:`qed.FixedSzOperator`. Both
        are :class:`ed::LinearOperator`-derived on the C++ side; the
        backend is auto-selected via :func:`ed::select_backend`.
    opts
        :class:`SolveOptions`. Defaults to a freshly-constructed
        instance (1 eigenvalue, Auto method, CPU lane).

    Returns
    -------
    :class:`GroundStateResult`
        Eigenvalues (sorted ascending), Krylov diagnostics, backend
        metadata, and optional HDF5 path for eigenvectors.
    """
    if opts is None:
        opts = SolveOptions()
    return _core.workflows_solve(op, opts)


def thermal(op, opts: ThermalOptions = None) -> ThermalResult:
    """Run the unified finite-temperature workflow on ``op``.

    Parameters
    ----------
    op
        A :class:`qed.Operator` or :class:`qed.FixedSzOperator`.
    opts
        :class:`ThermalOptions`. Defaults to FTLM with 30 samples and
        a Krylov dim of 100.

    Returns
    -------
    :class:`ThermalResult`
        Thermodynamic vectors (energy / specific heat / entropy /
        free energy), backend metadata, and the ground-state energy
        estimate.

    Notes
    -----
    Today the FTLM / LTLM / KPM-DOS lanes are CPU-only (the kernel
    facades static_assert on Backend type). The mTPQ / cTPQ lanes are
    fully backend-templated. See ``ed/thermal/ftlm_kernel.h`` for the
    delegation-inversion roadmap.
    """
    if opts is None:
        opts = ThermalOptions()
    return _core.workflows_thermal(op, opts)


def spectral(op, observables, opts: SpectralOptions = None) -> SpectralResult:
    """Run the unified dynamical-correlator workflow on ``op``.

    Parameters
    ----------
    op
        A :class:`qed.Operator` or :class:`qed.FixedSzOperator`.
    observables
        Sequence of one or two :class:`qed.Operator` instances; the
        auto-correlator case takes a single operator and the kernel
        treats it as the self-correlator.
    opts
        :class:`SpectralOptions`. Defaults to the ground-state CF
        spectrum with broadening ``0.05`` on the ``[-10, +10]``
        ``omega`` window.

    Returns
    -------
    :class:`SpectralResult`
        ``omega``, ``S_real``, ``S_imag``, optional error bars,
        Krylov diagnostics, and backend metadata.
    """
    if opts is None:
        opts = SpectralOptions()
    return _core.workflows_spectral(op, list(observables), opts)
