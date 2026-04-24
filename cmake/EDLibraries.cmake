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
# PUBLIC properties, so executables (ED, TPQ_DSSF, compute_bfg_order_parameters,
# the test binaries) only need to write `target_link_libraries(<exe> PRIVATE
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
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# -----------------------------------------------------------------------------
# ed_io: I/O helpers (basis vector / lanczos basis buffer)
# -----------------------------------------------------------------------------
add_library(ed_io STATIC
    ${IO_DIR}/basis_vector_storage.cpp
    ${IO_DIR}/lanczos_basis_buffer.cpp
)
target_include_directories(ed_io PUBLIC ${_ED_PUBLIC_INCLUDES})
target_link_libraries(ed_io PUBLIC ${ED_COMMON_LINK_LIBS})
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
target_link_libraries(ed_solvers_cpu PUBLIC ed_core ed_io ${ED_COMMON_LINK_LIBS})
target_link_libraries(ed_solvers_cpu PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
)
target_compile_options(ed_solvers_cpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
set_target_properties(ed_solvers_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
