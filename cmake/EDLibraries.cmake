# =============================================================================
# cmake/EDLibraries.cmake
#
# Defines the project's first-class static libraries:
#
#   ed_io           Pure I/O helpers (basis vector storage, lanczos basis
#                   buffer). No solver dependencies.
#   ed_core         Core types/config (ed_config.cpp). Depends on ed_io.
#   ed_solvers_cpu  All CPU eigensolvers + thermal methods (Lanczos, ARPACK,
#                   CG, dynamics, TPQ, FTLM, LTLM, hybrid, observables, and
#                   optionally ScaLAPACK distributed diag). Depends on ed_core
#                   and ed_io.
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
if(WITH_SCALAPACK AND SCALAPACK_LIBRARIES)
    list(APPEND ED_COMMON_LINK_LIBS ${SCALAPACK_LIBRARIES})
endif()
list(APPEND ED_COMMON_LINK_LIBS
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${EXTRA_LINALG_LIBRARIES}
    ${ARPACK_LIBRARY}
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
# ed_solvers_cpu: CPU eigensolvers + thermal methods.
# -----------------------------------------------------------------------------
set(ED_SOLVERS_CPU_SOURCES
    ${SOLVERS_CPU_DIR}/arpack.cpp
    ${SOLVERS_CPU_DIR}/observables.cpp
    ${SOLVERS_CPU_DIR}/lanczos.cpp
    ${SOLVERS_CPU_DIR}/CG.cpp
    ${SOLVERS_CPU_DIR}/dynamics.cpp
    ${SOLVERS_CPU_DIR}/TPQ.cpp
    ${SOLVERS_CPU_DIR}/ftlm.cpp
    ${SOLVERS_CPU_DIR}/ltlm.cpp
    ${SOLVERS_CPU_DIR}/hybrid_thermal.cpp
)
if(WITH_SCALAPACK AND SCALAPACK_LIBRARIES)
    list(APPEND ED_SOLVERS_CPU_SOURCES ${SOLVERS_CPU_DIR}/scalapack_diag.cpp)
endif()

add_library(ed_solvers_cpu STATIC ${ED_SOLVERS_CPU_SOURCES})
target_include_directories(ed_solvers_cpu PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_solvers_cpu PUBLIC ed_core ed_io ed_parallel ${ED_COMMON_LINK_LIBS})
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
#   #3 distributed_ftlm.cpp         -- multi-sample FTLM w/ MPI-over-samples
#                                      composed on top of distributed Lanczos
# -----------------------------------------------------------------------------
if(WITH_MPI)
    add_library(ed_distributed STATIC
        ${DISTRIBUTED_DIR}/distributed_operator.cpp
        ${DISTRIBUTED_DIR}/distributed_lanczos.cpp
        ${DISTRIBUTED_DIR}/distributed_ftlm.cpp
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
)
target_include_directories(ed_symmetry PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_symmetry PUBLIC ed_core)
target_compile_options(ed_symmetry PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_symmetry PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
# Depends on ed_solvers_cpu (Lanczos / FTLM / observables / ARPACK), ed_dssf
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
target_link_libraries(ed_cli PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_cli PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_cli PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
        ${SOLVERS_GPU_DIR}/gpu_fixed_sz_operator.cu
        ${SOLVERS_GPU_DIR}/gpu_symmetrized_operator.cu
        ${SOLVERS_GPU_DIR}/gpu_full_diag.cu
        ${SOLVERS_GPU_DIR}/gpu_lanczos.cu
        ${SOLVERS_GPU_DIR}/gpu_block_lanczos.cu
        ${SOLVERS_GPU_DIR}/gpu_krylov_schur.cu
        ${SOLVERS_GPU_DIR}/gpu_block_krylov_schur.cu
        ${SOLVERS_GPU_DIR}/gpu_ed_wrapper.cu
        ${SOLVERS_GPU_DIR}/gpu_tpq.cu
        ${SOLVERS_GPU_DIR}/gpu_cg.cu
        ${SOLVERS_GPU_DIR}/lobpcg_eigen_solve.cpp
        ${SOLVERS_GPU_DIR}/gpu_dynamics.cu
        ${SOLVERS_GPU_DIR}/gpu_ftlm.cu
        ${SOLVERS_GPU_DIR}/gpu_mixed_precision.cu
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
endif()
