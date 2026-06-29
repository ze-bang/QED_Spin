# =============================================================================
# cmake/EDLibraries.cmake
#
# Defines the project's first-class static libraries:
#
#   ed_io           Pure I/O helpers (basis vector storage, lanczos basis
#                   buffer). No solver dependencies.
#   ed_core         Core types/config (ed_config.cpp). Depends on ed_io.
#   ed_solvers_cpu  CPU eigensolvers + thermal methods (Lanczos, block
#                   Lanczos, Krylov-Schur, full diagonalization, TPQ, FTLM,
#                   LTLM, KPM_DOS, observables, dynamics). Depends on
#                   ed_core and ed_io.
#   ed_solvers_gpu  All GPU/CUDA solvers (only built when WITH_CUDA). Depends
#                   on ed_core, ed_io, and the CUDA runtime/cuBLAS/cuSPARSE/
#                   cuRAND/cuSOLVER imported targets.
#
# Each library exposes its include directories and link dependencies via
# PUBLIC properties, so executables (ED, compute_bfg_order_parameters, the
# test binaries) only need to write `target_link_libraries(<exe> PRIVATE
# ed_solvers_cpu)` -- the include path and BLAS/LAPACK/HDF5/OpenMP/MPI/CUDA
# link stack propagate automatically.
#
# This module is a pure structural refactor: every TU that the previous
# inline source-list approach compiled is still compiled here, with the same
# CPU_OPT_FLAGS, the same -DWITH_* compile definitions, and the same link
# stack. Build artifacts are bit-identical modulo file paths.
#
# P1.2 / audit Q5.
# =============================================================================

# -----------------------------------------------------------------------------
# Build the linkage stack ED_COMMON_LINK_LIBS used by the libraries below.
# Order matters: ScaLAPACK must come before BLAS, because some BLAS profiles
# (notably AOCL with libflame) carry an incompatible ScaLAPACK in the same
# directory and we want the explicit one to win in RPATH search order.
# -----------------------------------------------------------------------------
set(ED_COMMON_LINK_LIBS "")
list(APPEND ED_COMMON_LINK_LIBS
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${EXTRA_LINALG_LIBRARIES}
    ${HDF5_LIBRARIES}
)
if(WITH_MPI)
    list(APPEND ED_COMMON_LINK_LIBS ${MPI_CXX_LIBRARIES})
endif()

# OpenMP must already have been found by the parent CMakeLists.txt (we put
# the find_package(OpenMP) call before include(EDLibraries) for exactly this
# reason). If found, propagate it to all libraries as PUBLIC.
if(OpenMP_CXX_FOUND)
    list(APPEND ED_COMMON_LINK_LIBS OpenMP::OpenMP_CXX)
endif()

# Helper: every public include directory is wrapped in BUILD_INTERFACE so the
# install(EXPORT) step doesn't try to bake source-tree paths into the
# exported targets. Installed consumers find headers via
# INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}.
set(_ED_PUBLIC_INCLUDES
    "$<BUILD_INTERFACE:${INCLUDE_DIR}>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/core>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/solvers>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/io>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/bfg>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/cli>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/symmetry>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/parallel>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/distributed>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/matvec>"  # Matvec-unification revamp
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# -----------------------------------------------------------------------------
# ed_parallel: NUMA-aware first-touch + thread-pinning hooks (Phase 3a #4).
#
# Tiny utility library: a single TU that exposes
# `ed::parallel::first_touch_complex` / `first_touch_bytes` /
# `pin_omp_threads_once` / `describe_numa_state`. Both knobs are
# DEFAULT-OFF (`ED_NUMA_FIRST_TOUCH`, `ED_NUMA_PIN_THREADS`); turning them
# on never changes numerical results, only page placement and thread
# affinity. Linked PUBLIC into ed_io and ed_solvers_cpu so the in-memory
# basis tile loader (`lanczos_reorth.cpp`) and the Lanczos / FTLM /
# selective-reorth entry points can call into the helpers without
# pulling in an extra optional dep.
#
# Intentionally NO libnuma dependency: the first cut works on any Linux +
# glibc + OpenMP system. Explicit `numa_alloc_onnode` / `mbind` placement
# can be added later as a follow-up if profiling justifies it (would need
# a `find_library(NUMA numa)` and a WITH_LIBNUMA option).
# -----------------------------------------------------------------------------
add_library(ed_parallel STATIC
    ${PARALLEL_DIR}/numa.cpp
    ${PARALLEL_DIR}/thread_budget.cpp
)
target_include_directories(ed_parallel PUBLIC ${_ED_PUBLIC_INCLUDES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(ed_parallel PUBLIC OpenMP::OpenMP_CXX)
endif()
# pthread for pthread_setaffinity_np on Linux. Empty no-op on platforms
# where pthreads isn't a separate library (most modern glibc setups link
# it transitively, but be explicit so this archive is self-contained).
find_package(Threads QUIET)
if(Threads_FOUND)
    target_link_libraries(ed_parallel PUBLIC Threads::Threads)
endif()
target_compile_options(ed_parallel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_parallel PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_io: I/O helpers (basis vector / lanczos basis buffer)
# -----------------------------------------------------------------------------
add_library(ed_io STATIC
    ${IO_DIR}/basis_vector_storage.cpp
    ${IO_DIR}/lanczos_basis_buffer.cpp
    ${IO_DIR}/lanczos_checkpoint.cpp
    ${IO_DIR}/lanczos_reorth.cpp
)
target_include_directories(ed_io PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_io PUBLIC ed_parallel ${ED_COMMON_LINK_LIBS})
target_link_libraries(ed_io PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_io PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
# -fPIC so this archive can be linked into the pybind11 shared module
# (`_core.so`). Harmless overhead for the C++-only executables.
set_target_properties(ed_io PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_core: ed_config + (eventually) split-out construct_ham / hdf5_io / etc.
# -----------------------------------------------------------------------------
add_library(ed_core STATIC
    ${CORE_DIR}/ed_config.cpp
    # Phase A of the "Backend x Symmetries x Workflows" plan (May 2026)
    # -- CPU-only stub for the lazy GPU sector mirror entry point.
    # When WITH_CUDA is OFF this TU provides the throwing stub that
    # makes SectorView::bind_cuda() fail loudly. When WITH_CUDA is ON
    # the file is an empty translation unit and the strong definitions
    # come from ${SRC_DIR}/symmetry/streaming_symmetry_gpu_mirror.cu
    # (added to ed_solvers_gpu below). Splitting the TU avoids
    # contaminating ed_core with a CUDA include path and avoids the
    # multiple-definition / undefined-reference traps of a single
    # source compiled into both libraries.
    ${SRC_DIR}/symmetry/streaming_symmetry_gpu_mirror.cpp
    # Phase A operator-collapse GPU parity (Jun 2026) -- CPU-only stub
    # for ed::symmetry::SectorOperator::bind_cuda(). Empty TU under
    # WITH_CUDA (strong def comes from sector_operator_gpu.cu in
    # ed_solvers_gpu below); throwing stub when WITH_CUDA is OFF.
    ${SRC_DIR}/symmetry/sector_operator_gpu.cpp
    # Phase 2a operator-collapse GPU parity (Jun 2026) -- WEAK ed_core
    # fallbacks for the NON-VIRTUAL GPU mirror hooks on Operator /
    # FixedSzOperator (cuda_mirror_available_ + bind_cuda_*_impl_). Under
    # WITH_CUDA these report "no device mirror" and are overridden by the
    # strong defs in operator_gpu.cu (ed_solvers_gpu) for any binary that
    # links the GPU archive; CPU-only binaries keep the weak fallback so
    # they still link. bind_cuda()/geometry() stay inline in the headers,
    # so Operator's vtable has no key function pinned to ed_solvers_gpu.
    # Empty TU when WITH_CUDA is OFF (the hooks are never referenced).
    ${CORE_DIR}/operator_gpu.cpp
)
target_include_directories(ed_core PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_core PUBLIC ed_io ${ED_COMMON_LINK_LIBS})
target_link_libraries(ed_core PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_matvec: unified matrix-vector multiplication layer (Phase 1 of the
# matvec-unification revamp). Provides:
#
#   * MatVecOperator       polymorphic base for any operator that acts on
#                          a vector (any backend, any basis, MPI or not)
#   * Backend              host/cuda/mpi backends for the surrounding
#                          level-1 BLAS (axpy/dot/norm/scale)
#   * basis::*Policy       compile-time basis descriptions used by the
#                          shared term kernel
#   * kernel::apply_terms  the *single* matrix-free term-evaluation
#                          implementation, parameterised on basis policy
#                          and scalar type
#   * OperatorAdapter /    legacy Operator / FixedSzOperator wrapped as
#     FixedSzOperatorAdapter MatVecOperator instances (Phase 2 collapses
#                          these so the Operator implementations call
#                          directly into the shared kernel)
#
# Layered above ed_core (which owns Operator / FixedSzOperator term
# storage); consumed by ed_solvers_cpu (and later ed_solvers_gpu /
# ed_distributed).
# -----------------------------------------------------------------------------
add_library(ed_matvec STATIC
    ${MATVEC_DIR}/sanity_check.cpp
    # P6: explicit instantiation of the three host CpuMatVecBackend cells
    # (Full / FixedSz / Symmetry) over the canonical term-view shape.
    ${MATVEC_DIR}/cpu_backend_instantiations.cpp
)
target_include_directories(ed_matvec PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_matvec PUBLIC ed_core ${ED_COMMON_LINK_LIBS})
target_link_libraries(ed_matvec PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_matvec PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_matvec PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_solvers_cpu: CPU eigensolvers + thermal methods.
# -----------------------------------------------------------------------------
set(ED_SOLVERS_CPU_SOURCES
    ${SOLVERS_CPU_DIR}/observables.cpp
    ${SOLVERS_CPU_DIR}/lanczos.cpp
    ${SOLVERS_CPU_DIR}/dynamics.cpp
    ${SOLVERS_CPU_DIR}/TPQ.cpp
    ${SOLVERS_CPU_DIR}/tpq_io.cpp
    ${SOLVERS_CPU_DIR}/ftlm.cpp
    ${SOLVERS_CPU_DIR}/ftlm_dynamical.cpp
    ${SOLVERS_CPU_DIR}/ftlm_kpm.cpp
    ${SOLVERS_CPU_DIR}/kpm_dos.cpp
    ${SOLVERS_CPU_DIR}/ltlm.cpp
    ${SOLVERS_CPU_DIR}/symmetry_adapted_solve.cpp
    ${SRC_DIR}/observables/ftlm_cross_irrep_kernel.cpp
    ${SRC_DIR}/orchestrator.cpp
    # Phase A of the "mirror examples" plan (May 2026): Python-named
    # kwargs facade + small helpers (build introspection, find_symmetries,
    # estimate_resources, suggest_workflow, thermal_auto). Implementation
    # is header-light, lives alongside orchestrator.cpp.
    ${SRC_DIR}/api/api_facade.cpp
    ${SRC_DIR}/api/build_introspection.cpp
    ${SRC_DIR}/api/symmetry_helpers.cpp
    # (execution-planner "dictator" + feasibility advisor removed: sensible
    #  defaults + env-override leaf policy hooks instead.)
)

add_library(ed_solvers_cpu STATIC ${ED_SOLVERS_CPU_SOURCES})
target_include_directories(ed_solvers_cpu PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_solvers_cpu PUBLIC ed_matvec ed_core ed_io ed_parallel ${ED_COMMON_LINK_LIBS})

# Phase A of the "mirror examples" plan (May 2026): the Python-mirror
# facade `src/api/symmetry_helpers.cpp` calls `ed::sym::translation_group_1d`
# / `group_from_generators`, which live in `ed_symmetry`. Pull the
# library into ed_solvers_cpu's PUBLIC link surface so downstream
# consumers (tests, examples, new SDK callers) do not have to remember
# the dep. ed_distributed already PUBLIC-ly links ed_solvers_cpu, so
# the inverse link would create a cycle; downstream targets that need
# the `WITH_MPI` distributed lane of `ed::make_operator` continue to
# link `ed_distributed` explicitly via the `ed_add_test` helper.
target_link_libraries(ed_solvers_cpu PUBLIC ed_symmetry)
# Phase 5.3 of the Krylov-unification gap-fill (May 2026 day 12+): suppress
# the `[[deprecated]]` warning on `build_lanczos_tridiagonal_with_basis`
# for our own legacy CPU callsites. The attribute remains active for every
# external consumer of `<ed/solvers/lanczos.h>`.
target_compile_definitions(ed_solvers_cpu PRIVATE ED_BUILDING_INTERNAL=1)
target_link_libraries(ed_solvers_cpu PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_solvers_cpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_solvers_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_distributed: Phase 3b -- distributed-memory matrix-free SpMV
# (DistributedOperator) and distributed Lanczos / FTLM hooks built on top.
#
# Only built when WITH_MPI is ON; the library is the canonical home for
# every TU that depends symbolically on `<mpi.h>` for the SOTA "honest 40"
# scaling story. Depends on ed_core (for the `Operator` term storage we
# read), ed_solvers_cpu (for re-orth helpers reused by DistributedLanczos),
# and ed_parallel (for NUMA hooks shared with the single-rank Lanczos).
#
# The three Phase-3b TUs are intentionally CPU-only and OpenMP-friendly:
# they layer cleanly on top of the rank-local SpMV without re-implementing
# any of the matrix-free term iteration in `Operator::apply_real`.
#
# Phase 3b items:
#   #1 distributed_operator.cpp     -- gather-form SpMV + halo exchange
#   #2 distributed_lanczos.cpp      -- Lanczos w/ MPI_Allreduce dot/norm
#                                      and rank-local Krylov basis
#                                      (#6: + eigenvector reconstruction)
#   #3 distributed_ftlm.cpp         -- multi-sample FTLM w/ MPI-over-samples
#                                      composed on top of distributed Lanczos
#                                      (#5: + observable expectation values)
#   #8 distributed_tpq.cpp          -- canonical TPQ via Taylor-truncated
#                                      e^{-(delta/2) H} on rank-local |psi>
# -----------------------------------------------------------------------------
if(WITH_MPI)
    add_library(ed_distributed STATIC
        ${DISTRIBUTED_DIR}/distributed_operator.cpp
        # Wave 2 ("Unify all 16 matvec cells", May 2026) -- cell 2C
        # (Distributed Fixed-Sz) as a native LinearOperator instead of
        # the Phase G symmetry-with-trivial-group workaround.
        ${DISTRIBUTED_DIR}/distributed_fixed_sz_operator.cpp
        ${DISTRIBUTED_DIR}/distributed_lanczos.cpp
        ${DISTRIBUTED_DIR}/distributed_ftlm.cpp
        ${DISTRIBUTED_DIR}/distributed_tpq.cpp
        # Phase 9 / Layer 3: distributed thick-restart Lanczos (Krylov-Schur
        # for Hermitian operators); see distributed_krylov_schur.h.
        ${DISTRIBUTED_DIR}/distributed_krylov_schur.cpp
        # Phase 3b #7 stage 1: orbit-respecting partition primitive.
        ${DISTRIBUTED_DIR}/orbit_partition.cpp
        # Phase 3b #7 stage 2 prep: orbit-aware MPI_Alltoallv halo plan
        # (consumed by DistributedSymmetryOperator).
        ${DISTRIBUTED_DIR}/orbit_halo_plan.cpp
        # Phase 3b #7 stage 2: symmetry-projected distributed SpMV
        # (orbit-partitioned rows + orbit-aware halo + sparse local SpMV).
        ${DISTRIBUTED_DIR}/distributed_symmetry_operator.cpp
    )
    target_include_directories(ed_distributed PUBLIC ${_ED_PUBLIC_INCLUDES})
    target_link_libraries(ed_distributed PUBLIC
        ed_core ed_io ed_solvers_cpu ed_parallel ${ED_COMMON_LINK_LIBS}
    )
    target_compile_options(ed_distributed PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
    set_target_properties(ed_distributed PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# -----------------------------------------------------------------------------
# ed_distributed_gpu: Phase 3c -- multi-GPU collectives over NCCL.
#
# Built only when WITH_MPI && WITH_CUDA && NCCL_FOUND. Holds the .cu TU
# that #includes <nccl.h> and <cuda_runtime.h>; keeping it OUT of
# ed_distributed lets the CPU-only path stay a pure CXX library and avoids
# pulling the CUDA toolchain into builds that don't ask for it.
#
# Phase 3c items:
#   #1 multi_gpu.cu  -- MultiGpuCommunicator (RAII over ncclComm_t built
#                       from MPI_Comm) + sum/broadcast wrappers used by
#                       the future distributed_lanczos_gpu (#2). Stage 1
#                       implementation; the SpMV halo continues to ride
#                       MPI in this stage and only the dot/norm
#                       reductions are NCCL-aware.
#
# Wired into ed_distributed_main and the multi-GPU tests via
# `if(TARGET ed_distributed_gpu)` checks downstream so the rest of the
# tree is honest about the optional dependency.
#
# NCCL_FOUND / NCCL_INCLUDE_DIRS / NCCL_LIBRARIES are populated by the
# Phase 3c block in the top-level CMakeLists.txt (Phase 3c detection).
# -----------------------------------------------------------------------------
if(WITH_MPI AND WITH_CUDA AND NCCL_FOUND)
    add_library(ed_distributed_gpu STATIC
        ${DISTRIBUTED_DIR}/multi_gpu.cu
        # Phase 3c stage 2: GPU-resident distributed Lanczos with
        # ncclAllReduce dot/norm + host-staged SpMV via the existing
        # CPU DistributedOperator.
        ${DISTRIBUTED_DIR}/distributed_lanczos_gpu.cu
        # Phase 3c stage 3: fully GPU-resident DistributedGPUOperator
        # (NCCL pairwise SendRecv halo + CUDA SpMV kernel).
        ${DISTRIBUTED_DIR}/distributed_gpu_operator.cu
        # Phase 9 / Layer 2: multi-GPU canonical TPQ. Device-resident
        # |psi> + DistributedGPUOperator SpMV + cuBLAS axpys/dotcs +
        # NCCL allreduces for normalisation/observables. MPI-over-
        # samples mirrors the CPU distributed_tpq.
        ${DISTRIBUTED_DIR}/distributed_tpq_gpu.cu
        # Phase A (device matrix MPI+GPU): on-device FTLM. Per-sample
        # Lanczos with full re-orth runs entirely on the GPU (basis
        # held in one device slab, cublasZdotc/cublasZaxpy/NCCL
        # allreduce for the inner products, DistributedGPUOperator for
        # the SpMV, host-side tridiag eigensolve), J&P observable
        # contraction reuses the same on-device basis. MPI-over-
        # samples mirrors the CPU distributed_ftlm.
        ${DISTRIBUTED_DIR}/distributed_ftlm_gpu.cu
        # Phase B (device matrix MPI+GPU): on-device Krylov-Schur
        # (thick-restart Lanczos with Ritz-pair locking). In-cycle
        # basis + locked Ritz vectors held in two device slabs;
        # twice-CGS reorth against both is coalesced into single NCCL
        # allreduces; DistributedGPUOperator for the SpMV; host-side
        # (m x m) Eigen tridiag eigensolve per cycle.
        ${DISTRIBUTED_DIR}/distributed_krylov_schur_gpu.cu
        # Phase C (device matrix MPI+GPU): on-device symmetry-projected
        # SpMV. Wraps a CPU `DistributedSymmetryOperator` (orbit basis
        # + LPT partition + OrbitHaloPlan + projected CSR row slab),
        # uploads the per-row CSR (col_idx, is_local mask, complex
        # coefficients) and the halo-plan send_local_idx to device,
        # halo runs through NCCL pairwise SendRecv, SpMV is one
        # CUDA kernel per local row.
        ${DISTRIBUTED_DIR}/distributed_symmetry_operator_gpu.cu
    )
    target_include_directories(ed_distributed_gpu PUBLIC ${_ED_PUBLIC_INCLUDES})
    target_include_directories(ed_distributed_gpu PRIVATE ${NCCL_INCLUDE_DIRS})
    target_link_libraries(ed_distributed_gpu PUBLIC
        ed_distributed
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        ${NCCL_LIBRARIES}
        ${ED_COMMON_LINK_LIBS}
    )
    target_compile_definitions(ed_distributed_gpu PUBLIC ED_HAVE_NCCL=1)
    set_target_properties(ed_distributed_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(ed_distributed_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endif()

# -----------------------------------------------------------------------------
# ed_dssf: pure-data DSSF/SSSF observable assembly (operator_spec etc.).
#
# This library is the canonical home for the (operator_type x basis x momentum
# x spin-combo x fixed-Sz) cross-product that every dynamical/static
# structure-factor workflow needs. The ED CLI consumes it via
# `compute_*_workflow` (in ed_cli) and the canonical `ed::dssf::run` engine
# seam (P1.10 / DSSF PR-A; P2.5 / DSSF PR-F; P2.14 deletion of TPQ_DSSF).
#
# Depends only on ed_core for the `Operator` definitions.
# -----------------------------------------------------------------------------
add_library(ed_dssf STATIC
    ${DSSF_DIR}/operator_spec.cpp
    ${DSSF_DIR}/dssf_method.cpp
    ${DSSF_DIR}/dssf_io.cpp
    ${DSSF_DIR}/cross_sector_observable.cpp
    ${DSSF_DIR}/cross_sector_orbit_observable.cpp
)
target_include_directories(ed_dssf PUBLIC ${_ED_PUBLIC_INCLUDES})
target_include_directories(ed_dssf PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(ed_dssf PUBLIC ed_core ${ED_COMMON_LINK_LIBS})
target_link_libraries(ed_dssf PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_dssf PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_dssf PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_symmetry: programmatic site-permutation symmetry DSL (P2.11 / audit
# §3.10). Builds `SymmetryGroupInfo` from a list of permutation
# generators without going through the JSON detour (`automorphism_finder.py`
# + automorphism_results/*.json + SymmetryGroupInfo::loadFromDirectory).
#
# Depends on ed_core because `SymmetryGroupInfo` is declared inside
# `ed/core/construct_ham.h` (alongside the `Operator` definition that
# consumes it).
# -----------------------------------------------------------------------------
add_library(ed_symmetry STATIC
    ${SYMMETRY_DIR}/group.cpp
    ${SYMMETRY_DIR}/irreps.cpp
    ${SYMMETRY_DIR}/symmetry_adapted.cpp
)
target_include_directories(ed_symmetry PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_symmetry PUBLIC ed_core)
target_compile_options(ed_symmetry PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_symmetry PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_input: standalone C++ lattice + Hamiltonian builder (replaces the
# `python/edlib/helper_*.py` family).
#
# Three TUs:
#   * lattice.cpp                -- 1D / 2D / 3D lattice generators (chain,
#                                   square, triangular, honeycomb, kagome,
#                                   pyrochlore, custom-from-edges,
#                                   cluster.txt).
#   * hamiltonian_builder.cpp    -- fluent term accumulator with shortcuts
#                                   for Heisenberg / XXZ / XYZ / Ising /
#                                   Kitaev / DM / Zeeman / pyrochlore
#                                   non-Kramers + emit_into(Operator&) /
#                                   write_directory(...) outputs.
#   * file_io.cpp                -- low-level Trans.dat / InterAll.dat /
#                                   ThreeBodyG.dat / positions.dat /
#                                   one_body_correlations*.dat /
#                                   two_body_correlations**.dat writers.
#
# `ed_input` PUBLIC-links `ed_core` because `HamiltonianBuilder::emit_into`
# touches `Operator::transform_data_` / `three_body_data_` directly (matching
# the way the existing `addOneBody` / `addTwoBody` shortcuts in
# construct_ham.h push records into those vectors). The only consumers of
# `ed_input` are (i) the new examples under examples/, (ii) the pybind11
# bindings under python/qed/_input.cpp, and (iii) the unit tests in
# tests/unit/test_input_*.cpp -- the production `./ED <dir>` driver does
# not depend on it.
# -----------------------------------------------------------------------------
add_library(ed_input STATIC
    ${SRC_DIR}/input/lattice.cpp
    ${SRC_DIR}/input/hamiltonian_builder.cpp
    ${SRC_DIR}/input/file_io.cpp
)
target_include_directories(ed_input PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_input PUBLIC ed_core)
target_compile_options(ed_input PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_input PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_bfg: BFG order-parameter library (P2.1).
#
# Houses the kagome / pyrochlore-superlattice geometry loader plus the
# pure-physics building blocks for the order-parameter pipeline:
#
#   * cluster.cpp           -- `Cluster` struct + `load_cluster(...)`.
#   * topology.cpp          -- `find_triangles` / `find_bowties`
#                              (combinatorial helpers, no wavefunctions).
#   * correlations.cpp      -- two-body spin correlations + bond expectations
#                              (`compute_smsp_correlations`,
#                               `compute_szsz_correlations`, the four
#                               `compute_*_bond_expectations` overloads).
#   * structure_factor.cpp  -- bond-bilinear structure factors and
#                              Fourier-applied dimer operators
#                              (`compute_dimer_sf_direct`,
#                               `compute_heisenberg_sf_direct`,
#                               `apply_dimer_fourier`,
#                               `apply_heisenberg_dimer_fourier`,
#                               `compute_dimer_dimer_correlation`,
#                               `compute_heisenberg_dimer_dimer_correlation`,
#                               `set_memory_efficient_mode`).
#   * ring_observables.cpp  -- bowtie ring-flip / triangle ring-exchange
#                              kernels: `apply_bowtie_fourier`,
#                              `compute_bowtie_resonance`,
#                              `compute_triangle_chiral`. Reuses the
#                              memory-efficient flag from structure_factor.
#   * spin_structure_factor.cpp -- site-resolved spin structure factor
#                              `compute_spin_structure_factor` over
#                              precomputed two-body correlation tables;
#                              produces `StructureFactorResult`.
#   * wavefunction_io.cpp   -- HDF5 wavefunction + TPQ-state loaders shared
#                              by the CPU and GPU drivers and by the Python
#                              bindings (`load_wavefunction`,
#                              `load_all_tpq_states`, `load_tpq_state`,
#                              `TPQState`).
#   * results_io.cpp        -- HDF5 results writers + scalar-summary structs
#                              for the order-parameter pipeline
#                              (`save_results`, `save_temperature_scan_results`,
#                              `save_scan_results`, `NematicResult`,
#                              `VBSResult`, `PlaquetteResult`, `Sq2DGridResult`,
#                              `OrderParameterResults`).
#   * order_parameters.cpp  -- physics kernels building the result aggregates
#                              above: `compute_nematic_order(_real)`,
#                              `compute_vbs_order`, `compute_plaquette_order`,
#                              `compute_sq_2d_grid`,
#                              `compute_all_order_parameters`.
#   * cli.cpp               -- BFG order-parameter CLI orchestration helpers
#                              (`process_all_temperatures`,
#                              `scan_jpm_directories`, `run_single_file`,
#                              `run_scan`, `print_usage`) consumed by
#                              `compute_bfg_order_parameters`. Keeps the
#                              driver itself a thin argv shell. P2.1 (9th
#                              slice).
#
# The CPU driver (`compute_bfg_order_parameters`), the GPU driver
# (`compute_bfg_order_parameters_gpu`), and the future Python bindings all
# call into this library so they share one authoritative implementation
# instead of copy-pasting the kernels three times.
#
# Link-deps: OpenMP for the correlation kernels (PUBLIC so executables
# inherit it). HDF5 is now PUBLIC because `wavefunction_io.cpp` exposes
# H5::Exception in its declared signatures (the headers themselves only
# include <complex>, but the implementation pulls in libhdf5_cpp). MPI
# integration is intentionally compile-time-optional inside
# `cli.cpp` -- the calling executable owns whether to define USE_MPI and
# link MPI; `ed_bfg` itself never pulls MPI into its PUBLIC link line so
# the library remains usable from non-MPI executables (the GPU driver,
# the Python `_core.so` extension, the unit-test binaries) without any
# MPI dependency leaking through.
# -----------------------------------------------------------------------------
add_library(ed_bfg STATIC
    ${BFG_DIR}/cluster.cpp
    ${BFG_DIR}/topology.cpp
    ${BFG_DIR}/correlations.cpp
    ${BFG_DIR}/structure_factor.cpp
    ${BFG_DIR}/ring_observables.cpp
    ${BFG_DIR}/spin_structure_factor.cpp
    ${BFG_DIR}/wavefunction_io.cpp
    ${BFG_DIR}/results_io.cpp
    ${BFG_DIR}/order_parameters.cpp
    ${BFG_DIR}/cli.cpp
)
target_include_directories(ed_bfg PUBLIC ${_ED_PUBLIC_INCLUDES})
target_include_directories(ed_bfg PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(ed_bfg PUBLIC ${HDF5_LIBRARIES})
if(OpenMP_CXX_FOUND)
    target_link_libraries(ed_bfg PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(ed_bfg PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_bfg PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_cli: CLI workflow entry points (run_*_workflow / compute_*_workflow /
# parse_* / construct_operators_from_config / print_eigenvalue_summary /
# compute_thermodynamics).
#
# Extracted from src/apps/ed_main.cpp in P1.11 (DSSF PR-B / audit §3.10) so
# the same workflow functions can be reused by future entry points
# (`ED dssf` subcommand in P2.4) and by pybind11 bindings without also
# pulling in the legacy `int main()` machinery.
#
# Depends on ed_solvers_cpu (Lanczos / FTLM / observables), ed_dssf
# (build_observable_pairs), ed_io (HDF5IO via construct_ham), and
# transitively on ed_core. When WITH_CUDA, also depends on ed_solvers_gpu
# for the GPU dynamical/static-response code paths under the WITH_CUDA
# guards.
# -----------------------------------------------------------------------------
add_library(ed_cli STATIC
    ${CLI_DIR}/workflows.cpp
    ${CLI_DIR}/dssf_engine.cpp
)
target_include_directories(ed_cli PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_cli PUBLIC ed_solvers_cpu ed_dssf ed_io ed_core)
if(WITH_CUDA)
    target_link_libraries(ed_cli PUBLIC ed_solvers_gpu)
endif()
# Wave C2 (Full Unified-Interface Collapse, May 2026): the CLI now
# builds operators via the inline `ed::make_operator` factory, whose
# body references `DistributedOperator` /
# `DistributedSymmetryOperator` constructors under `WITH_MPI`. The
# distributed lanes are never selected from the CLI (it always sets
# `spec.distributed = false`), but the constructors still appear as
# weak references in the emitted object file and need to resolve
# transitively for any binary that links `ed_cli`.
if(WITH_MPI AND TARGET ed_distributed)
    target_link_libraries(ed_cli PUBLIC ed_distributed)
endif()
target_link_libraries(ed_cli PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_cli PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_cli PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Phase 5.3 (Krylov-unification gap-fill): suppress the legacy
# `build_lanczos_tridiagonal_with_basis` deprecation warning inside the
# CLI workflows -- they still call the legacy function through
# `<ed/core/ed_wrapper.h>`.
target_compile_definitions(ed_cli PRIVATE ED_BUILDING_INTERNAL=1)

# -----------------------------------------------------------------------------
# ed_solvers_gpu: CUDA-only library; depends on the CUDA imported targets.
# Always defined so callers can write `if(TARGET ed_solvers_gpu)` without
# having to also re-check WITH_CUDA themselves.
# -----------------------------------------------------------------------------
if(WITH_CUDA)
    set(ED_SOLVERS_GPU_SOURCES
        ${SOLVERS_GPU_DIR}/gpu_operator.cu
        ${SOLVERS_GPU_DIR}/gpu_operator_conversion.cpp
        ${SOLVERS_GPU_DIR}/gpu_kernels.cu
        # Operator-collapse Phase 2b (Jun 2026): gpu_fixed_sz_operator.cu,
        # gpu_full_diag.cu, gpu_block_lanczos.cu, gpu_krylov_schur.cu, and
        # gpu_tpq.cu were deleted along with the dead GPUEDWrapper forwarders
        # that were their only callers. The production GPU paths now run off
        # the unified host operators' bind_cuda() device matvec (CudaBackend /
        # CudaMatVecBackend) and a directly-constructed GPUFTLMSolver.
        # gpu_lanczos.cu (the Gen-1 hand-rolled GPULanczos class) was retired:
        # runGPULanczos routes entirely through gpu_lanczos_kernel_facade.cu
        # (lanczos_kernel<CudaBackend>). The one capability it uniquely held --
        # on-disk basis spill for oversized-basis eigenvector runs -- is now a
        # clear facade error rather than a separate hand-rolled solver.
        ${SOLVERS_GPU_DIR}/gpu_ed_wrapper.cu
        ${SOLVERS_GPU_DIR}/gpu_lanczos_kernel_facade.cu
        ${SOLVERS_GPU_DIR}/kpm_dos_gpu.cu
        ${SOLVERS_GPU_DIR}/gpu_ftlm.cu
        ${SOLVERS_GPU_DIR}/gpu_mixed_precision.cu
        ${SOLVERS_GPU_DIR}/symmetry_adapted_gpu.cu
        # Phase A of the "Backend x Symmetries x Workflows" plan
        # (May 2026) -- real lazy GPU sector mirror for
        # StreamingSymmetryOperator + FixedSz variant. Lives here (and
        # not in ed_core) because it pulls in <cuda_runtime.h> +
        # thrust + the device basis policy headers. The ed_core .cpp
        # twin is an empty TU when WITH_CUDA is ON, so there is no
        # multiple-definition risk.
        ${SRC_DIR}/symmetry/streaming_symmetry_gpu_mirror.cu
        # Phase A operator-collapse GPU parity (Jun 2026) -- strong def of
        # ed::symmetry::SectorOperator::bind_cuda(), built once into
        # ed_solvers_gpu (reuses make_sector_matvec_gpu from the mirror TU
        # above). The ed_core .cpp twin is an empty TU under WITH_CUDA.
        ${SRC_DIR}/symmetry/sector_operator_gpu.cu
        # Phase 2a operator-collapse GPU parity (Jun 2026) -- STRONG defs of
        # the Operator / FixedSzOperator GPU mirror hooks
        # (cuda_mirror_available_ + bind_cuda_*_impl_), routing the
        # full-Hilbert / fixed-Sz GPU matvec through the SOTA no-atomic
        # CudaMatVecBackend. Override the weak ed_core fallbacks
        # (operator_gpu.cpp) wherever this archive is linked.
        ${CORE_DIR}/operator_gpu.cu
    )

    add_library(ed_solvers_gpu STATIC ${ED_SOLVERS_GPU_SOURCES})
    target_include_directories(ed_solvers_gpu PUBLIC
        ${_ED_PUBLIC_INCLUDES}
        "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/gpu>"
    )
    # The src/solvers/gpu directory holds private *.cuh helpers shared between
    # the .cu TUs in this library; keep that BUILD-only and PRIVATE so it
    # doesn't leak into INTERFACE_INCLUDE_DIRECTORIES at install time.
    target_include_directories(ed_solvers_gpu PRIVATE
        "$<BUILD_INTERFACE:${SOLVERS_GPU_DIR}>"
    )
    # ed_solvers_gpu calls into the CPU-side helpers (e.g. save_ftlm_results,
    # average_ftlm_samples in ftlm.cpp), so it has a hard dependency on
    # ed_solvers_cpu. Declaring it PUBLIC means CMake will list the archives
    # in the correct order on the executable link line and downstream callers
    # get the CPU symbols transitively.
    target_link_libraries(ed_solvers_gpu PUBLIC
        ed_solvers_cpu
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
        CUDA::cusolver
        ${ED_COMMON_LINK_LIBS}
    )
    target_link_libraries(ed_solvers_gpu PUBLIC
        "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
    )
    set_target_properties(ed_solvers_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(ed_solvers_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    # Phase 5.3 (Krylov-unification gap-fill): suppress the legacy
    # `build_lanczos_tridiagonal_with_basis` deprecation warning -- the GPU
    # ed_wrapper still calls the CPU legacy function for the eigenvector
    # reconstruction path.
    target_compile_definitions(ed_solvers_gpu PRIVATE ED_BUILDING_INTERNAL=1)
endif()
