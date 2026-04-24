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

.. doxygenfile:: solvers/diagonalization.h
   :no-link:

.. doxygenfile:: solvers/lanczos.h
   :no-link:

.. doxygenfile:: solvers/ftlm.h
   :no-link:

.. doxygenfile:: solvers/ltlm.h
   :no-link:

.. doxygenfile:: solvers/observables.h
   :no-link:

ed::solvers -- GPU solvers (built only with WITH_CUDA=ON)
---------------------------------------------------------

.. doxygenfile:: gpu/gpu_ed_wrapper.h
   :no-link:
