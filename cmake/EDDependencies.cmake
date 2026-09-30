# =============================================================================
# cmake/EDDependencies.cmake
#
# Third-party dependencies (other than the BLAS/LAPACK profile, which lives
# in cmake/EDBlasProfile.cmake):
#
#   * Eigen3              REQUIRED   linear algebra (header-only)
#   * CUDAToolkit         optional   gated by WITH_CUDA
#
# P1.1 / audit Q5.
# =============================================================================

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# CUDA setup
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DWITH_CUDA)
    add_definitions(-DTPQ_HAVE_CUDA)
    add_definitions(-DENABLE_GPU)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit include directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# ARPACK is no longer a dependency: the ARPACK_SM / ARPACK_LM /
# ARPACK_SHIFT_INVERT / ARPACK_ADVANCED solvers were retired in the
# minimalist-architecture rev (May 2026). All Krylov needs are served
# by the in-tree LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR kernels.
