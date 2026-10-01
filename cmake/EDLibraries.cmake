# =============================================================================
# cmake/EDLibraries.cmake
#
# Defines the project's first-class static libraries:
#
#   ed_parallel     Thread budget + OpenMP thread pinning.
#   ed_core         Core types. Depends on ed_parallel.
#   ed_solvers_cpu  CPU eigensolvers + thermal methods (Lanczos,
#                   Krylov-Schur, full diagonalization, TPQ, FTLM, OFTLM,
#                   observables, dynamics). Depends on ed_core.
#   ed_solvers_gpu  All GPU/CUDA solvers (only built when WITH_CUDA). Depends
#                   on ed_core and the CUDA runtime/cuBLAS/cuSPARSE/
#                   cuRAND/cuSOLVER imported targets.
#
# Each library exposes its include directories and link dependencies via
# PUBLIC properties, so executables (the
# test binaries) only need to write `target_link_libraries(<exe> PRIVATE
# ed_solvers_cpu)` -- the include path and BLAS/LAPACK/OpenMP/CUDA
# link stack propagate automatically.
# =============================================================================

# -----------------------------------------------------------------------------
# Build the linkage stack ED_COMMON_LINK_LIBS used by the libraries below.
# -----------------------------------------------------------------------------
set(ED_COMMON_LINK_LIBS "")
list(APPEND ED_COMMON_LINK_LIBS
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${EXTRA_LINALG_LIBRARIES}
)

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
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/symmetry>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/parallel>"
    "$<BUILD_INTERFACE:${INCLUDE_DIR}/ed/matvec>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# -----------------------------------------------------------------------------
# ed_parallel: thread-budget + OpenMP thread-pinning hooks.
#
# Tiny utility library: `ed::parallel::ThreadBudgetScope` /
# `auto_threads_for_dim` and `pin_omp_threads_once` (knob
# `ED_NUMA_PIN_THREADS`, default off; it never changes numerical results,
# only thread affinity). Linked PUBLIC into ed_core and ed_solvers_cpu.
# Intentionally NO libnuma dependency.
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
# ed_core: core types and the CPU stub of the lazy GPU sector mirror.
# -----------------------------------------------------------------------------
add_library(ed_core STATIC
    # CPU-only stub for the GPU sector-matvec factories
    # (ed/symmetry/sector_gpu_mirror.h). When WITH_CUDA is OFF this TU
    # provides throwing stubs so a misrouted GPU request fails loudly.
    # When WITH_CUDA is ON
    # the file is an empty translation unit and the strong definitions
    # come from ${SRC_DIR}/symmetry/streaming_symmetry_gpu_mirror.cu
    # (added to ed_solvers_gpu below). Splitting the TU avoids
    # contaminating ed_core with a CUDA include path and avoids the
    # multiple-definition / undefined-reference traps of a single
    # source compiled into both libraries.
    ${SRC_DIR}/symmetry/streaming_symmetry_gpu_mirror.cpp
)
target_include_directories(ed_core PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_core PUBLIC ed_parallel ${ED_COMMON_LINK_LIBS})
if(WITH_CUDA)
    # Host-compiled TUs of ed_core include <cuComplex.h>/<cublas_v2.h>
    # transitively (linear_operator.h -> cuda_backend.cuh under WITH_CUDA).
    # On NVIDIA-repo toolkit installs the headers live under
    # /usr/local/cuda-*/include, NOT the default compiler search path
    # (Ubuntu's nvidia-cuda-toolkit package masks this locally by dropping
    # them into /usr/include). Surface them explicitly; PUBLIC so every
    # downstream host target inheriting these headers compiles too.
    target_include_directories(ed_core PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_compile_options(ed_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_matvec: unified matrix-vector multiplication layer. Provides:
#
#   * LinearOperator       polymorphic base for any operator that acts on
#                          a vector (any backend, any basis)
#   * Backend              host/cuda backends for the surrounding
#                          level-1 BLAS (axpy/dot/norm/scale)
#   * basis::*Policy       compile-time basis descriptions used by the
#                          shared term kernel
#   * kernel::apply_terms  the *single* matrix-free term-evaluation
#                          implementation, parameterised on basis policy
#                          and scalar type
#
# Layered above ed_core (which owns the Operator term
# storage); consumed by ed_solvers_cpu and ed_solvers_gpu.
# -----------------------------------------------------------------------------
add_library(ed_matvec STATIC
    # Explicit instantiation of the host CpuMatVecBackend cells
    # (Full / RepSymmetry) over the canonical term-view shape.
    ${MATVEC_DIR}/cpu_backend_instantiations.cpp
)
target_include_directories(ed_matvec PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_matvec PUBLIC ed_core ${ED_COMMON_LINK_LIBS})
target_compile_options(ed_matvec PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_matvec PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_solvers_cpu: CPU eigensolvers + thermal methods.
# -----------------------------------------------------------------------------
set(ED_SOLVERS_CPU_SOURCES
    ${SOLVERS_CPU_DIR}/lanczos.cpp
    ${SOLVERS_CPU_DIR}/ftlm.cpp
    ${SOLVERS_CPU_DIR}/oftlm.cpp
    ${SRC_DIR}/solvers/little_group/lg_engine.cpp
    ${SRC_DIR}/solvers/little_group/lg_block_solve.cpp
    ${SRC_DIR}/solvers/little_group/lg_stars.cpp
    ${SRC_DIR}/solvers/little_group/lg_sectors.cpp
    ${SRC_DIR}/solvers/little_group/lg_sectors_thermal.cpp
    ${SRC_DIR}/solvers/little_group/lg_sectors_dynamics.cpp
    ${SRC_DIR}/solvers/little_group/lg_sectors_expect.cpp
    ${SRC_DIR}/solvers/little_group/lg_blocks.cpp
    ${SRC_DIR}/solvers/little_group/lg_group_sector.cpp
    ${SRC_DIR}/solvers/little_group/lg_ground_state.cpp
    ${SRC_DIR}/observables/ftlm_cross_irrep_kernel.cpp
)

add_library(ed_solvers_cpu STATIC ${ED_SOLVERS_CPU_SOURCES})
target_include_directories(ed_solvers_cpu PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_solvers_cpu PUBLIC ed_matvec ed_core ed_parallel ${ED_COMMON_LINK_LIBS})

# ed_symmetry (the permutation DSL, group closure) and ed_dssf (observable
# assembly) are part of ed_solvers_cpu's public link surface.
target_link_libraries(ed_solvers_cpu PUBLIC ed_symmetry ed_dssf)
target_compile_options(ed_solvers_cpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_solvers_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_dssf: pure-data DSSF/SSSF observable assembly (operator_spec etc.).
#
# This library is the canonical home for the (operator_type x basis x momentum
# x spin-combo x fixed-Sz) cross-product that every dynamical/static
# structure-factor workflow needs, plus the cross-irrep orbit observable the bindings apply.
#
# Depends only on ed_core for the `Operator` definitions.
# -----------------------------------------------------------------------------
add_library(ed_dssf STATIC
    ${DSSF_DIR}/operator_spec.cpp
    ${DSSF_DIR}/cross_sector_orbit_observable.cpp
)
target_include_directories(ed_dssf PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_dssf PUBLIC ed_core ${ED_COMMON_LINK_LIBS})
target_compile_options(ed_dssf PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_dssf PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_symmetry: programmatic site-permutation DSL --
# permutation algebra + generate_group -- and the numerical irrep
# decomposition (irreps.cpp).
# -----------------------------------------------------------------------------
add_library(ed_symmetry STATIC
    ${SYMMETRY_DIR}/group.cpp
    ${SYMMETRY_DIR}/irreps.cpp
)
target_include_directories(ed_symmetry PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_symmetry PUBLIC ed_core)
target_compile_options(ed_symmetry PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_symmetry PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# ed_input: standalone C++ lattice + Hamiltonian builder.
#
# Three TUs:
#   * lattice.cpp                -- 1D / 2D / 3D lattice generators (chain,
#                                   square, triangular, honeycomb, kagome,
#                                   pyrochlore, custom-from-edges,
#                                   cluster.txt).
#   * hamiltonian_builder.cpp    -- fluent term accumulator with shortcuts
#                                   for Heisenberg / XXZ / XYZ / Ising /
#                                   Kitaev / DM / Zeeman / pyrochlore
#                                   non-Kramers + emit_into(Operator&).
#
# `ed_input` PUBLIC-links `ed_core` because `HamiltonianBuilder::emit_into`
# touches `Operator::transform_data_` / `three_body_data_` directly (matching
# the way the `addOneBody` / `addTwoBody` shortcuts in construct_ham.h push
# records into those vectors). Its consumers are the pybind11 bindings
# (python/qed/_bindings/input_bindings.cpp) and the unit tests in
# tests/unit/test_input_*.cpp.
# -----------------------------------------------------------------------------
add_library(ed_input STATIC
    ${SRC_DIR}/input/lattice.cpp
    ${SRC_DIR}/input/hamiltonian_builder.cpp
)
target_include_directories(ed_input PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_input PUBLIC ed_core)
target_compile_options(ed_input PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_input PROPERTIES POSITION_INDEPENDENT_CODE ON)


# -----------------------------------------------------------------------------
# ed_solvers_gpu: CUDA-only library; depends on the CUDA imported targets.
# Always defined so callers can write `if(TARGET ed_solvers_gpu)` without
# having to also re-check WITH_CUDA themselves.
# -----------------------------------------------------------------------------
if(WITH_CUDA)
    set(ED_SOLVERS_GPU_SOURCES
        # The GPU paths run off the rep-sector device mirror below and the
        # little-group block kernels; combinadic.cu defines the shared
        # constant-memory
        # Pascal table the device basis policies rank fixed-Sz states with.
        ${SOLVERS_GPU_DIR}/combinadic.cu
        ${SOLVERS_GPU_DIR}/little_group_gpu.cu
        # On-the-fly representative GPU sector matvec for the rep
        # sectors (ed/symmetry/sector_gpu_mirror.h). Lives here (and
        # not in ed_core) because it pulls in <cuda_runtime.h> +
        # thrust + the device basis policy headers. The ed_core .cpp
        # twin is an empty TU when WITH_CUDA is ON, so there is no
        # multiple-definition risk.
        ${SRC_DIR}/symmetry/streaming_symmetry_gpu_mirror.cu
        # device CSR of the cross-sector probe (finite-T dynamics on GPU)
        ${SRC_DIR}/matvec/device_csr.cu
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
        CUDA::curand
        CUDA::cusolver
        ${ED_COMMON_LINK_LIBS}
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
