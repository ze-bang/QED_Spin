# =============================================================================
# cmake/EDDependencies.cmake -- third-party dependencies other than BLAS (cmake/EDBlas.cmake):
# Eigen3 (header-only, required) and, under WITH_CUDA, the CUDA toolkit.
# =============================================================================

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION}: ${CUDAToolkit_INCLUDE_DIRS}")
endif()
