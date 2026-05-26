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

Top-level facade
----------------

.. automodule:: qed
   :members:
   :undoc-members:
   :show-inheritance:

Thermal entry point (``qed.thermal``)
-------------------------------------

.. automodule:: qed.thermal
   :members:
   :undoc-members:
   :show-inheritance:

Spectral / structure-factor entry point (``qed.spectral``)
----------------------------------------------------------

.. automodule:: qed.spectral
   :members:
   :undoc-members:
   :show-inheritance:

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

Pure-Python heuristics for solver / device / Krylov-dim / broadening
selection. Consulted automatically by the three orchestrator verbs;
exposed standalone for inspection and integration into custom
schedulers.

.. automodule:: qed.auto_tune
   :members:
   :undoc-members:
   :show-inheritance:

Pre-flight planner (``qed.feasibility``)
----------------------------------------

Resource estimation and ranked workflow suggestions
(:func:`qed.estimate_resources`, :func:`qed.suggest_workflow`,
``ResourceError``).

.. automodule:: qed.feasibility
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

BFG post-processing (``qed.bfg``)
---------------------------------

Bond-bilinear, structure-factor, and order-parameter post-processing
on eigenstates.

.. automodule:: qed.bfg
   :members:
   :undoc-members:
   :show-inheritance:
