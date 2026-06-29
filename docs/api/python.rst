Python API reference (``qed``)
==============================

.. default-domain:: py

The Python side lives under :py:mod:`qed`. It is a thin ``pybind11``
layer over the C++ ``ed_solvers_*`` static libraries, wired up via
``scikit-build-core``. The canonical user-facing surface is three
verbs that mirror the C++ orchestrator: :func:`qed.solve`,
:func:`qed.thermal`, :func:`qed.spectral`. All three take plain
keyword arguments — no ``SolveOptions`` / ``ThermalOptions`` /
``SpectralOptions`` objects in user code (those are constructed
internally from your kwargs).

.. contents::
   :local:
   :depth: 2

Top-level facade (``qed``)
--------------------------

``import qed`` exposes everything you need to drive a workflow
end-to-end. The most common entry points are:

* :func:`qed.solve` — ground state, eigenvalues, low-lying spectrum.
* :func:`qed.thermal` — finite-temperature thermodynamics.
* :func:`qed.spectral` — static and dynamical structure factors.
* :func:`qed.full_diagonalization`, :func:`qed.lanczos` — explicit
  low-level access to the dense LAPACK and Lanczos drivers.
* :func:`qed.find_symmetries` — symmetry discovery returning
  :class:`qed.SymmetryReport` + :class:`qed.GeneratorSet` candidates.
* :func:`qed.has_cuda_build`, :func:`qed.has_mpi_build`,
  :func:`qed.has_nccl_build`, :func:`qed.has_scalapack_build` — build
  introspection.

The helper classes (:class:`qed.Operator`, :class:`qed.FixedSzOperator`,
:class:`qed.GeneratorSet`,
:class:`qed.SymmetryReport`, …) are re-exported from their canonical
sub-modules; see the per-submodule sections below for full member
listings.

.. automodule:: qed
   :no-members:

Ground-state entry point (``qed.workflow``)
-------------------------------------------

:func:`qed.solve` and its plumbing live in :mod:`qed.workflow`.

.. automodule:: qed.workflow
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: GeneratorSet, SymmetryReport

Thermal entry point (``qed.thermal``)
-------------------------------------

.. automodule:: qed.thermal
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: thermal

.. autofunction:: qed.thermal.thermal
   :no-index:

Spectral / structure-factor entry point (``qed.spectral``)
----------------------------------------------------------

.. automodule:: qed.spectral
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: spectral

.. autofunction:: qed.spectral.spectral
   :no-index:

DSSF helpers (``qed.dssf``)
---------------------------

Lower-level helpers consumed by :func:`qed.spectral` (operator-spec
builders, broadening utilities). New code should call
:func:`qed.spectral` directly.

.. automodule:: qed.dssf
   :members:
   :undoc-members:
   :show-inheritance:

Hamiltonian builder DSL (``qed.input``)
---------------------------------------

The Python mirror of the ``ed::input`` C++ library. Builds lattices
and Hamiltonians fluently and emits either an in-memory ``Operator``
or the ``InterAll.dat`` / ``Trans.dat`` / ``positions.dat`` directory
that ``./ED`` consumes.

.. automodule:: qed.input
   :members:
   :undoc-members:
   :show-inheritance:

Programmatic symmetries (``qed.symmetry``)
------------------------------------------

Generator-set construction (translations, reflections, custom
permutations) consumed via the ``symmetry=`` kwarg of
:func:`qed.solve` / :func:`qed.thermal` / :func:`qed.spectral`.

.. automodule:: qed.symmetry
   :members:
   :undoc-members:
   :show-inheritance:

Auto-tuner (``qed.auto_tune``)
------------------------------

Pure-Python heuristics for spectral broadening (η), the ω-grid, random-vector
count, and Krylov dimension. Consumed by :func:`qed.spectral` when
``auto_tune=True`` (see :func:`qed.auto_tune.tune_dssf`); exposed standalone for
inspection. (There is no pre-flight planner / feasibility module — it was
removed in favour of sensible defaults plus a runtime memory guard.)

.. automodule:: qed.auto_tune
   :members:
   :undoc-members:
   :show-inheritance:

MPI launcher helper (``qed.mpi``)
---------------------------------

Helpers that build the right ``mpiexec`` invocation for
``ed_distributed_main`` so multi-rank runs can be driven from a single
Python process.

.. automodule:: qed.mpi
   :members:
   :undoc-members:
   :show-inheritance:

Hamiltonian builder (``qed.hamiltonian``)
-----------------------------------------

Lower-level ``Operator`` / ``FixedSzOperator`` construction helpers
shared between :mod:`qed.input` and direct callers.

.. automodule:: qed.hamiltonian
   :members:
   :undoc-members:
   :show-inheritance:

BFG post-processing (``qed.bfg``)
---------------------------------

Bond-bilinear, structure-factor, and order-parameter post-processing
on eigenstates.

.. automodule:: qed.bfg
   :members:
   :undoc-members:
   :show-inheritance:
