# =============================================================================
# cmake/EDDependencies.cmake
#
# Third-party dependencies (other than the BLAS/LAPACK profile, which lives
# in cmake/EDBlasProfile.cmake, and MPI/ScaLAPACK, which lives in
# cmake/EDMpiScalapack.cmake):
#
#   * Eigen3              REQUIRED   linear algebra (header-only)
#   * nlohmann/json       REQUIRED   JSON parsing (find_package() with
#                                    FetchContent fallback to v3.11.3)
#   * HDF5 (CXX)          REQUIRED   binary I/O for diagonalization output
#   * CUDAToolkit         optional   gated by WITH_CUDA
#   * ARPACK              REQUIRED   sparse eigensolver
#
# Note: nlohmann_json is exposed via link_libraries() so its INTERFACE
# include directories propagate to every subsequently-defined target.
# It is header-only, so this carries no link-time symbols.
#
# P1.1 / audit Q5.
# =============================================================================

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# -----------------------------------------------------------------------------
# nlohmann/json (P0.15 / audit Q8).
# Replaces three brittle hand-rolled JSON parsers in
# include/ed/core/construct_ham.h that walked strings with .find('[') / .substr().
# We prefer a system-installed nlohmann_json (most distros ship it as
# nlohmann-json3-dev), and fall back to FetchContent for portability on
# clusters where it is not installed.
# -----------------------------------------------------------------------------
find_package(nlohmann_json 3.7.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found system-wide; fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install    OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)
else()
    message(STATUS "nlohmann_json found: ${nlohmann_json_VERSION}")
endif()

# nlohmann_json is header-only. We deliberately do NOT use a global
# `link_libraries(nlohmann_json::nlohmann_json)` here, because that would
# bake the imported target into INTERFACE_LINK_LIBRARIES of our exported
# static libraries -- and a fetched target cannot be exported via
# install(EXPORT). EDLibraries.cmake instead wires nlohmann_json into each
# library via `$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>`. Downstream
# consumers refind it via QEDConfig.cmake's find_dependency(nlohmann_json).

# Find HDF5 (with C++ bindings)
# Prefer HDF5_DIR from environment (set by module system)
if(DEFINED ENV{HDF5_DIR} AND NOT HDF5_ROOT)
    set(HDF5_ROOT $ENV{HDF5_DIR})
    message(STATUS "Using HDF5_ROOT from environment: ${HDF5_ROOT}")
endif()
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
message(STATUS "HDF5 found: ${HDF5_VERSION}")
message(STATUS "HDF5 include directories: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")
add_definitions(${HDF5_DEFINITIONS})

# Extract HDF5 include directories from imported target if variable is empty
# (needed for modern HDF5 config mode with CUDA)
if(TARGET hdf5_cpp-shared)
    get_target_property(HDF5_CPP_INCLUDE_DIRS hdf5_cpp-shared INTERFACE_INCLUDE_DIRECTORIES)
    if(HDF5_CPP_INCLUDE_DIRS)
        list(APPEND HDF5_INCLUDE_DIRS ${HDF5_CPP_INCLUDE_DIRS})
        list(REMOVE_DUPLICATES HDF5_INCLUDE_DIRS)
        message(STATUS "HDF5 include directories (from target): ${HDF5_INCLUDE_DIRS}")
    endif()
endif()

# CUDA setup
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DWITH_CUDA)
    add_definitions(-DTPQ_HAVE_CUDA)
    add_definitions(-DENABLE_GPU)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit include directories: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# Find ARPACK
find_library(ARPACK_LIBRARY NAMES arpack arpack-ng)
if(NOT ARPACK_LIBRARY)
    message(FATAL_ERROR "ARPACK library not found")
endif()
