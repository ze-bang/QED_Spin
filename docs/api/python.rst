Python API reference (``qed``)
==============================

.. default-domain:: py

``import qed`` gives five verbs over one symmetry description, plus the Hamiltonian
builders and symmetry discovery. Every verb takes ``sym=`` (a :class:`qed.Symmetry`,
default :meth:`qed.Symmetry.auto`) and ``device="cpu" | "gpu" | "auto"``.

.. contents::
   :local:
   :depth: 1

Symmetry
--------

.. autoclass:: qed.Symmetry
   :members: auto, none, select, groups, resolve

.. autofunction:: qed.api.symmetry.momentum_of
.. autofunction:: qed.api.symmetry.irrep_characters_of

Levels and vectors
------------------

.. autofunction:: qed.eigs
.. autoclass:: qed.EigResult
   :members: vectors, expect, matrix_element, save, momentum, irrep_characters
.. autofunction:: qed.load_eigs

.. autofunction:: qed.spectrum
.. autoclass:: qed.SpectrumResult
   :members: momentum, irrep_characters

.. autofunction:: qed.expect
.. autoclass:: qed.ExpectResult

Thermodynamics
--------------

.. autofunction:: qed.thermal
.. autoclass:: qed.ThermalResult

Dynamics
--------

.. autofunction:: qed.dynamics
.. autoclass:: qed.DynamicsResult

Hamiltonians and operators
--------------------------

``qed.Operator`` holds the terms (one-, two- and three-body products of S+, S-, Sz);
``qed.input.HamiltonianBuilder`` builds it from bonds, and ``qed.dssf`` builds the
momentum-resolved probe operators used with :func:`qed.dynamics`.

.. automodule:: qed.input
   :members:

.. automodule:: qed.dssf
   :members:

Symmetry discovery
------------------

.. autofunction:: qed.find_symmetries
.. autoclass:: qed.Symmetries
   :members: describe

Build introspection
-------------------

* :func:`qed.has_cuda_build` -- whether the extension was built with CUDA.
* :func:`qed.env_snapshot`, :func:`qed.debug_env` -- the registered environment
  variables and their current values.
