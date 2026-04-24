C++ API reference
=================

This page is auto-generated from the headers under ``include/ed/`` via
Doxygen + Breathe. Every public symbol in the listed namespaces is
documented; if a symbol is missing, please add a Doxygen comment to the
declaration in the corresponding header.

.. contents::
   :local:
   :depth: 2

ed::dssf -- Dynamical / static structure factor
-----------------------------------------------

.. doxygenstruct:: ed::dssf::OperatorSpec
   :members:

.. doxygenstruct:: ed::dssf::ObservablePairs
   :members:

.. doxygenfunction:: ed::dssf::build_observable_pairs

.. doxygenfunction:: ed::dssf::compute_transverse_bases

ed::core -- Hamiltonian construction & basis
--------------------------------------------

.. doxygenclass:: Operator
   :members:
   :no-link:

.. doxygenclass:: FixedSzOperator
   :members:
   :no-link:

ed::core -- Diagonalization types
---------------------------------

.. doxygenenum:: ed::DiagonalizationMethod

ed::core -- HDF5 I/O
--------------------

.. doxygenclass:: HDF5IO
   :members:
   :no-link:

ed::solvers -- CPU solvers
--------------------------

The CPU solver entry points live in the following headers. Each one is
indexed and cross-linked from the auto-generated symbol index; see the
header source for full signatures and per-parameter documentation.

- ``ed/solvers/lanczos.h`` -- Lanczos diagonalization (in-memory and
  disk-streamed) plus the basis-vector helpers used by FTLM/LTLM.
- ``ed/solvers/ftlm.h`` -- Finite-Temperature Lanczos Method
  (thermodynamics + dynamical / static response).
- ``ed/solvers/ltlm.h`` -- Low-Temperature Lanczos Method (energy /
  observables in the low-T regime).
- ``ed/solvers/observables.h`` -- spin / energy observables on
  eigenstates and TPQ snapshots.

ed::solvers -- GPU solvers (built only with WITH_CUDA=ON)
---------------------------------------------------------

GPU solvers mirror their CPU counterparts under ``include/ed/gpu/``:
``gpu_ed_wrapper.h``, ``gpu_lanczos.cuh``, ``gpu_ftlm.cuh``,
``gpu_tpq.cuh``, ``gpu_dynamics.cuh``, ``gpu_cg.cuh``,
``gpu_operator.cuh``. They are only compiled when ``WITH_CUDA=ON``.
