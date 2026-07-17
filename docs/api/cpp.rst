C++ API reference
=================

This page is auto-generated from the headers under ``include/ed/`` via
Doxygen + Breathe. Every public symbol in the listed namespaces is
documented; if a symbol is missing, please add a Doxygen comment to the
declaration in the corresponding header.

For the post-collapse architectural picture, read
``docs/architecture/ARCHITECTURE.md`` first; for the symmetry math,
``docs/architecture/SYMMETRY.md``.

.. contents::
   :local:
   :depth: 2

The orchestrator: ``ed::workflows``
-----------------------------------

The canonical user-facing surface is three orchestrator verbs declared
in ``include/ed/orchestrator.h``. Every workflow goes through one of
them.

.. doxygenfunction:: ed::workflows::solve

.. doxygenfunction:: ed::workflows::thermal

.. doxygenfunction:: ed::workflows::spectral

.. doxygenstruct:: ed::SolveOptions
   :members:

.. doxygenstruct:: ed::ThermalOptions
   :members:

.. doxygenstruct:: ed::SpectralOptions
   :members:

The operator factory: ``ed::make_operator``
--------------------------------------------

Operators are constructed by a single factory that consumes
``OperatorSpec`` (a tagged union over ``InMemoryOperator`` /
``FilePaths`` / ``DirectoryPath`` sources) and returns a
``std::unique_ptr<LinearOperator>``.

.. doxygenfunction:: ed::make_operator

.. doxygenstruct:: ed::OperatorSpec
   :members:

.. doxygenclass:: ed::LinearOperator
   :members:

ed::core -- Hamiltonian + basis
-------------------------------

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

ed::symmetry -- Subspace × ProjectorChain
------------------------------------------

The orthogonal symmetry composition introduced in May 2026. A sector
is described by one :cpp:class:`ed::symmetry::Subspace` and one
ordered :cpp:class:`ed::symmetry::ProjectorChain`. Future axes
(spin-flip Z2, time reversal, SU(2) total-S) plug into the same
duck-types without touching the operator hierarchy.

.. doxygenclass:: ed::symmetry::FullSpaceSubspace
   :members:

.. doxygenclass:: ed::symmetry::FixedSzSubspace
   :members:

.. doxygenclass:: ed::symmetry::SpatialProjector
   :members:

.. doxygenclass:: ed::symmetry::InternalZ2Projector
   :members:

.. doxygenclass:: ed::symmetry::AntiunitaryProjector
   :members:

.. doxygenclass:: ed::symmetry::ProjectorChain
   :members:

.. doxygenfunction:: ed::symmetry::compute_orbit_for_state

ed::krylov / ed::thermal / ed::observables -- kernel headers
-------------------------------------------------------------

The CPU kernel entry points live in the following template headers.
Each one is indexed and cross-linked from the auto-generated symbol
index; see the header source for full signatures and per-parameter
documentation.

- ``include/ed/krylov/lanczos_kernel.h`` -- single-vector Lanczos
  (CPU / MPI specialisations of the ``Backend`` template).
- ``include/ed/krylov/block_lanczos_kernel.h`` -- block Lanczos.
- ``include/ed/krylov/krylov_schur_kernel.h`` -- thick-restart
  Krylov-Schur.
- ``include/ed/thermal/{ftlm,ltlm,mtpq,ctpq,kpm_dos}_kernel.h`` --
  finite-temperature kernel facades.
- ``include/ed/observables/{expectation,static_correlator,cf_dynamical,kpm_dynamical,time_evolution}.h``
  -- correlator primitives (cf-spectral, KPM, time evolution).

ed::dssf -- Dynamical / static structure factor
-----------------------------------------------

.. doxygenstruct:: ed::dssf::OperatorSpec
   :members:

.. doxygenstruct:: ed::dssf::ObservablePairs
   :members:

.. doxygenfunction:: ed::dssf::build_observable_pairs

.. doxygenfunction:: ed::dssf::compute_transverse_bases

ed::input -- Lattice + Hamiltonian builder
-------------------------------------------

The standalone ``ed_input`` library replaces the legacy ``edlib``
helpers. It emits either an in-memory ``Operator`` or the
``InterAll.dat`` / ``Trans.dat`` / ``positions.dat`` directory that
``./ED`` consumes.

.. doxygenclass:: ed::input::HamiltonianBuilder
   :members:

.. doxygennamespace:: ed::input::lattice

MPI lane (built only with WITH_MPI=ON)
---------------------------------------

Production MPI is the in-process ``ed::matvec::MpiBackend``
(``include/ed/matvec/backends/mpi_backend.h``, selected by
``select_backend`` when the process runs under ``mpirun``) plus the
across-sector SectorDistributor in the CLI sector loop. The NCCL
multi-GPU communicator lives in ``include/ed/parallel/multi_gpu.h``.
(The ``ed::distributed`` operator family was retired in Stage 11d,
Jul 2026.)

ed::gpu -- CUDA lane (built only with WITH_CUDA=ON)
----------------------------------------------------

GPU operators and solvers live under ``include/ed/gpu/`` and are only
compiled when ``WITH_CUDA=ON``: ``gpu_operator.cuh``, ``gpu_ftlm.cuh``,
``gpu_solvers.h``, ``gpu_ed_wrapper.h``, ``gpu_mixed_precision.h``,
``kpm_dos_gpu.cuh``, ``kernel_config.h``, ``bit_operations.cuh``,
``combinadic.cuh``.
