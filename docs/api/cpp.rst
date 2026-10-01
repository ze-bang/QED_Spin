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

.. doxygenfile:: ed/ops/operator.h
.. doxygenfile:: ed/matvec/linear_operator.h

Krylov kernels
--------------

.. doxygenfile:: ed/krylov/lanczos.h
.. doxygenfile:: ed/krylov/krylov_schur.h
.. doxygenfile:: ed/krylov/tridiag.h

Thermodynamics kernels
----------------------

.. doxygenfile:: ed/thermal/ftlm.h
.. doxygenfile:: ed/thermal/mtpq.h
.. doxygenfile:: ed/thermal/sample_seed.h

Dynamics kernels
----------------

.. doxygenfile:: ed/dynamics/cf.h
.. doxygenfile:: ed/dynamics/ftlm_dynamics.h

Backends and batching
---------------------

.. doxygenfile:: ed/matvec/backend.h
.. doxygenfile:: ed/matvec/batcher.h

Placement
---------

``ed::place`` decides where each block of every verb runs (host or device, dense or Krylov),
from the 'auto' table in ``device.h``; ``ed::with_backend`` runs a lane on a fresh backend.
Eigs blocks run the Backend-templated block lanes; sampled thermal blocks call the thermal
kernels (``ftlm_kernel``, ``mtpq``, ``oftlm_cpu``).

.. doxygenfile:: ed/core/device.h
.. doxygenfile:: ed/core/select_backend.h
