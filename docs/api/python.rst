Python API reference (``qed``)
=====================================

.. default-domain:: py

The Python side lives under :py:mod:`qed`. It is a thin pybind11 layer
over the C++ ``ed_solvers_cpu`` library, wired up via ``scikit-build-core``.

.. contents::
   :local:
   :depth: 2

Top-level facade
----------------

.. automodule:: qed
   :members:
   :undoc-members:
   :show-inheritance:

Dynamical / static structure factor (``qed.dssf``)
---------------------------------------------------------

.. automodule:: qed.dssf
   :members:
   :undoc-members:
   :show-inheritance:

Hamiltonian builder DSL (``qed.hamiltonian``)
----------------------------------------------------

.. automodule:: qed.hamiltonian
   :members:
   :undoc-members:
   :show-inheritance:

Helpers (legacy ``edlib`` re-exports)
-------------------------------------

.. automodule:: qed.helpers
   :members:
   :undoc-members:
