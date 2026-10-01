C++ API reference
=================

Generated from the headers under ``include/ed/`` by Doxygen and Breathe. The Python
verbs are thin bindings over the ``ed::sectors`` layer; the kernels below it are
templated on a backend (``CpuBackend`` or ``CudaBackend``). :doc:`../architecture`
describes how the layers fit together.

.. contents::
   :local:
   :depth: 1

Sectors: ``ed::sectors``
------------------------

The symmetry description (``Spec``), the subspaces and blocks it resolves H into, and one
entry point per task.

.. doxygenfile:: ed/sectors/sectors.h
.. doxygenfile:: ed/sectors/thermal.h
.. doxygenfile:: ed/sectors/dynamics.h
.. doxygenfile:: ed/sectors/expect.h

Operators
---------

.. doxygenfile:: ed/core/operator.h
.. doxygenfile:: ed/core/linear_operator.h

Krylov kernels
--------------

.. doxygenfile:: ed/krylov/lanczos_kernel.h
.. doxygenfile:: ed/krylov/krylov_schur_kernel.h

Thermodynamics kernels
----------------------

.. doxygenfile:: ed/thermal/ftlm_kernel.h
.. doxygenfile:: ed/thermal/mtpq_kernel.h
.. doxygenfile:: ed/thermal/tpq_kernel.h

Dynamics kernels
----------------

.. doxygenfile:: ed/observables/cf_spectral_kernel.h
.. doxygenfile:: ed/observables/ftlm_dynamics_kernel.h
.. doxygenfile:: ed/observables/ftlm_cross_irrep_kernel.h

Backends and batching
---------------------

.. doxygenfile:: ed/matvec/backend.h
.. doxygenfile:: ed/matvec/matvec_batcher.h

Orchestrator
------------

``ed::workflows::solve`` and ``ed::workflows::thermal`` run one operator (a block) on the
backend ``select_backend`` picks; the sector layer calls them per block.

.. doxygenfile:: ed/orchestrator.h
