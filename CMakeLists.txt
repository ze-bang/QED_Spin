cmake_minimum_required(VERSION 3.18)

# =============================================================================
# BLAS/LAPACK BACKEND CONFIGURATION
# =============================================================================
# This project supports multiple BLAS/LAPACK backends. To avoid ABI conflicts,
# we define coherent "library profiles" that ensure all components are compatible.
#
# Available profiles (set via -DBLAS_PROFILE=<profile>):
#   AUTO      - Auto-detect best available (default)
#   FLEXIBLAS - FlexiBLAS wrapper (recommended for HPC clusters)
#   MKL       - Intel MKL (full stack including ScaLAPACK)
#   AOCL      - AMD AOCL (BLIS + libflame)
#   OPENBLAS  - OpenBLAS
#   GENERIC   - System BLAS/LAPACK (reference implementation)
#
# ScaLAPACK compatibility matrix:
#   MKL profile      -> MKL ScaLAPACK (requires Intel MPI or compatible OpenMPI)
#   FLEXIBLAS        -> System ScaLAPACK (FlexiBLAS backend at runtime)
#   OPENBLAS/GENERIC -> System ScaLAPACK (built against same LAPACK)
#   AOCL             -> ScaLAPACK disabled (ABI conflicts with system ScaLAPACK)
# =============================================================================

set(BLAS_PROFILE "AUTO" CACHE STRING "BLAS/LAPACK library profile (AUTO, FLEXIBLAS, MKL, AOCL, OPENBLAS, GENERIC)")
set_property(CACHE BLAS_PROFILE PROPERTY STRINGS AUTO FLEXIBLAS MKL AOCL OPENBLAS GENERIC)

# Optional custom ScaLAPACK root (for AOCL or custom builds)
set(SCALAPACK_ROOT "" CACHE PATH "Root path to a custom ScaLAPACK installation")
# Optional custom LAPACKE root (to avoid system BLAS/LAPACK alternatives)
set(LAPACKE_ROOT "" CACHE PATH "Root path to a custom LAPACKE installation")

# -----------------------------------------------------------------------------
# Local-developer cache variables (P0.10).
# Replaces the previously hardcoded /home/pc_linux/exact_diagonalization_clean/...
# paths. Set these via -D on the command line or in a local.cmake file:
#
#   cmake --preset default -DED_LAPACKE_ROOT=/path/to/lapacke_ref/install \
#                          -DED_BLAS_SHIM_DIR=/path/to/blas_shim
#
# Both default to empty; when empty, the system search paths are used.
# -----------------------------------------------------------------------------
set(ED_LAPACKE_ROOT "" CACHE PATH
    "Optional override: path to a local LAPACKE install (used when LAPACKE_ROOT is unset). \
P0.10 / audit Q7.")
set(ED_BLAS_SHIM_DIR "" CACHE PATH
    "Optional override: directory containing libblas.so.3 to PREPEND to RPATH \
(used by the AOCL profile to avoid MKL alternatives via liblapacke). \
P0.10 / audit Q7.")

# Make CMake search cmake/ for our project-local modules.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Host platform detection (CPU vendor + AVX2/FMA reporting).
include(EDPlatform)

# Legacy option compatibility - map old options to BLAS_PROFILE
if(DEFINED USE_AOCL_BLIS AND USE_AOCL_BLIS AND BLAS_PROFILE STREQUAL "AUTO")
    set(BLAS_PROFILE "AOCL")
endif()
if(DEFINED WITH_MKL AND WITH_MKL AND BLAS_PROFILE STREQUAL "AUTO")
    set(BLAS_PROFILE "MKL")
endif()
if(DEFINED USE_FLEXIBLAS AND USE_FLEXIBLAS AND BLAS_PROFILE STREQUAL "AUTO")
    set(BLAS_PROFILE "FLEXIBLAS")
endif()

set(DEFAULT_WITH_CUDA ON)
set(DEFAULT_WITH_MPI ON)

# Set defaults based on CPU vendor (only if AUTO)
if(BLAS_PROFILE STREQUAL "AUTO")
    if(CPU_VENDOR STREQUAL "AMD")
        # Check if AOCL is installed
        if(EXISTS "/opt/AMD/aocl" OR DEFINED ENV{AOCL_ROOT} OR DEFINED ENV{EBROOTAOCLMINBLAS})
            set(BLAS_PROFILE "AOCL")
        elseif(DEFINED ENV{EBROOTFLEXIBLAS} OR DEFINED ENV{FLEXIBLAS_ROOT})
            set(BLAS_PROFILE "FLEXIBLAS")
        else()
            set(BLAS_PROFILE "OPENBLAS")
        endif()
    elseif(CPU_VENDOR STREQUAL "Intel")
        if(EXISTS "/opt/intel/oneapi/mkl" OR DEFINED ENV{MKLROOT})
            set(BLAS_PROFILE "MKL")
        elseif(DEFINED ENV{EBROOTFLEXIBLAS})
            set(BLAS_PROFILE "FLEXIBLAS")
        else()
            set(BLAS_PROFILE "OPENBLAS")
        endif()
    else()
        # Unknown vendor - try FlexiBLAS, then OpenBLAS
        if(DEFINED ENV{EBROOTFLEXIBLAS} OR DEFINED ENV{FLEXIBLAS_ROOT})
            set(BLAS_PROFILE "FLEXIBLAS")
        else()
            set(BLAS_PROFILE "OPENBLAS")
        endif()
    endif()
endif()

message(STATUS "BLAS/LAPACK profile: ${BLAS_PROFILE}")

# Derive legacy flags from BLAS_PROFILE for compatibility
set(WITH_MKL OFF)
set(USE_ONEMKL OFF)
set(USE_AOCL_BLIS OFF)
set(USE_FLEXIBLAS OFF)

if(BLAS_PROFILE STREQUAL "MKL")
    set(WITH_MKL ON)
    set(USE_ONEMKL ON)
elseif(BLAS_PROFILE STREQUAL "AOCL")
    set(USE_AOCL_BLIS ON)
elseif(BLAS_PROFILE STREQUAL "FLEXIBLAS")
    set(USE_FLEXIBLAS ON)
endif()

# Options (define before project)
option(WITH_CUDA "Build with CUDA support" ${DEFAULT_WITH_CUDA})
option(WITH_MPI "Build with MPI support" ${DEFAULT_WITH_MPI})

# Set languages based on options. The version is mirrored in CITATION.cff
# and CHANGELOG.md; bump in lockstep when releasing.
if(WITH_CUDA)
    project(ExactDiagonalization VERSION 0.1.0 LANGUAGES CXX CUDA)
    enable_language(CUDA)
else()
    project(ExactDiagonalization VERSION 0.1.0 LANGUAGES CXX)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard if CUDA is enabled
if(WITH_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA architectures - use native if GPU available, otherwise use common modern architectures
    # Compute capability 6.0+ required for native double atomicAdd
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Try native first, but provide fallback for headless compilation
        execute_process(
            COMMAND ${CMAKE_CUDA_COMPILER} --run --gpu-architecture=native -x cu -c /dev/null
            RESULT_VARIABLE CUDA_NATIVE_CHECK_RESULT
            OUTPUT_QUIET ERROR_QUIET
        )
        if(CUDA_NATIVE_CHECK_RESULT EQUAL 0)
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            # Fallback: compile for common modern architectures (V100, A100, etc.)
            set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")
            message(STATUS "No GPU detected for native arch, using fallback: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")
    endif()
    
    # Add CUDA compiler flags to handle C++ standard library compatibility
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
    
    # Separate host and device compilation to avoid C++17 issues
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Source directories - new organized layout
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CORE_DIR ${SRC_DIR}/core)
set(APPS_DIR ${SRC_DIR}/apps)
set(SOLVERS_CPU_DIR ${SRC_DIR}/solvers/cpu)
set(SOLVERS_GPU_DIR ${SRC_DIR}/solvers/gpu)
set(IO_DIR ${SRC_DIR}/io)
set(DSSF_DIR ${SRC_DIR}/dssf)
set(BFG_DIR ${SRC_DIR}/bfg)
set(CLI_DIR ${SRC_DIR}/cli)
set(SYMMETRY_DIR ${SRC_DIR}/symmetry)

# Include directories - public headers in include/, private in src/
include_directories(${INCLUDE_DIR})
include_directories(${INCLUDE_DIR}/ed/core)
include_directories(${INCLUDE_DIR}/ed/solvers)
include_directories(${INCLUDE_DIR}/ed/io)
include_directories(${INCLUDE_DIR}/ed/gpu)
include_directories(${INCLUDE_DIR}/ed/bfg)
include_directories(${SRC_DIR})
include_directories(${CORE_DIR})
include_directories(${SOLVERS_CPU_DIR})
include_directories(${SOLVERS_GPU_DIR})
include_directories(${IO_DIR})

# =============================================================================
# BLAS / LAPACK / LAPACKE BACKEND
# =============================================================================
# Profile-based selection (FLEXIBLAS / MKL / AOCL / AOCL_BLIS / OPENBLAS /
# GENERIC). Sets BLAS_LIBRARIES, LAPACK_LIBRARIES, LAPACKE_LIBRARIES,
# EXTRA_LINALG_LIBRARIES, BLAS_VENDOR, SCALAPACK_COMPATIBLE in this scope.
include(EDBlasProfile)
# =============================================================================
# OTHER DEPENDENCIES
# =============================================================================
# Eigen3, nlohmann/json, HDF5 (CXX), optional CUDAToolkit (gated by
# WITH_CUDA), and ARPACK. Each is independent of the BLAS profile above.
include(EDDependencies)

# =============================================================================
# MPI / ScaLAPACK
# =============================================================================
# MPI is gated by WITH_MPI; if found, also tries to enable distributed
# diagonalization via ScaLAPACK (compatibility constrained by BLAS_PROFILE).
include(EDMpiScalapack)

# CPU optimization flags must be defined before EDLibraries so the static
# libraries pick them up via target_compile_options(... PRIVATE CPU_OPT_FLAGS).
include(EDOptimizationFlags)

# Find OpenMP BEFORE EDLibraries so the static libraries can link
# OpenMP::OpenMP_CXX as PUBLIC, propagating it to every executable.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP Support: YES")
else()
    message(STATUS "  OpenMP Support: NO")
endif()

# =============================================================================
# STATIC LIBRARIES
# =============================================================================
# Defines ed_io / ed_core / ed_solvers_cpu / ed_solvers_gpu (the last only
# when WITH_CUDA). Each library exposes its include directories and its link
# stack via PUBLIC properties so the executables below need only
# `target_link_libraries(<exe> PRIVATE ed_solvers_cpu ...)`.
include(EDLibraries)

# =============================================================================
# EXECUTABLES
# =============================================================================

# Main `ED` driver. The library targets carry every other source file.
# When CUDA is on, ed_solvers_gpu PUBLIC-links ed_solvers_cpu so we only need
# to mention the topmost library.
add_executable(ED ${APPS_DIR}/ed_main.cpp)
if(WITH_CUDA)
    target_link_libraries(ED PRIVATE ed_cli ed_solvers_gpu ed_dssf)
else()
    target_link_libraries(ED PRIVATE ed_cli ed_solvers_cpu ed_dssf)
endif()

# TPQ_DSSF is a legacy DSSF/SSSF wrapper around the same CPU+GPU stack as ED,
# plus parallel HDF5 for MPI-safe output. It will be replaced by `ED dssf` in
# Phase 2 (DSSF PR-E/F); for now it just links the same libraries as ED.
if(WITH_MPI)
    set(HDF5_PREFER_PARALLEL TRUE)
    find_path(HDF5_PARALLEL_INCLUDE_DIR H5Cpp.h
        HINTS /usr/include/hdf5/openmpi /usr/include/hdf5/mpi
        NO_DEFAULT_PATH
    )
    find_library(HDF5_PARALLEL_C_LIBRARY NAMES hdf5_openmpi hdf5
        HINTS /usr/lib/x86_64-linux-gnu/hdf5/openmpi
        NO_DEFAULT_PATH
    )
    find_library(HDF5_PARALLEL_CXX_LIBRARY NAMES hdf5_openmpi_cpp hdf5_cpp
        HINTS /usr/lib/x86_64-linux-gnu/hdf5/openmpi
        NO_DEFAULT_PATH
    )
    find_library(HDF5_PARALLEL_HL_LIBRARY NAMES hdf5_openmpi_hl hdf5_hl
        HINTS /usr/lib/x86_64-linux-gnu/hdf5/openmpi
        NO_DEFAULT_PATH
    )

    if(HDF5_PARALLEL_INCLUDE_DIR AND HDF5_PARALLEL_C_LIBRARY AND HDF5_PARALLEL_CXX_LIBRARY)
        message(STATUS "Found parallel HDF5 for TPQ_DSSF:")
        message(STATUS "  Include: ${HDF5_PARALLEL_INCLUDE_DIR}")
        message(STATUS "  C lib: ${HDF5_PARALLEL_C_LIBRARY}")
        message(STATUS "  C++ lib: ${HDF5_PARALLEL_CXX_LIBRARY}")
        set(HDF5_PARALLEL_LIBRARIES ${HDF5_PARALLEL_CXX_LIBRARY} ${HDF5_PARALLEL_C_LIBRARY})
        if(HDF5_PARALLEL_HL_LIBRARY)
            list(APPEND HDF5_PARALLEL_LIBRARIES ${HDF5_PARALLEL_HL_LIBRARY})
        endif()
        set(HAS_PARALLEL_HDF5 TRUE)
    else()
        message(STATUS "Parallel HDF5 not found, TPQ_DSSF will use serial HDF5 with file locking workaround")
        set(HAS_PARALLEL_HDF5 FALSE)
    endif()

    add_executable(TPQ_DSSF ${APPS_DIR}/TPQ_DSSF.cpp)
    if(WITH_CUDA)
        target_link_libraries(TPQ_DSSF PRIVATE ed_solvers_gpu ed_dssf)
    else()
        target_link_libraries(TPQ_DSSF PRIVATE ed_solvers_cpu ed_dssf)
    endif()
endif()

# BFG Order Parameter computation (lightweight, HDF5 + OpenMP only)
add_executable(compute_bfg_order_parameters ${APPS_DIR}/compute_bfg_order_parameters.cpp)
target_link_libraries(compute_bfg_order_parameters ed_bfg ${HDF5_LIBRARIES})
target_include_directories(compute_bfg_order_parameters PRIVATE ${HDF5_INCLUDE_DIRS})
if(OpenMP_CXX_FOUND)
    target_link_libraries(compute_bfg_order_parameters OpenMP::OpenMP_CXX)
    target_compile_definitions(compute_bfg_order_parameters PRIVATE _OPENMP)
endif()
target_compile_options(compute_bfg_order_parameters PRIVATE -O3 -march=native -fopenmp)

# BFG Order Parameter GPU version (CUDA accelerated)
if(WITH_CUDA)
    add_executable(compute_bfg_order_parameters_gpu ${APPS_DIR}/compute_bfg_order_parameters_gpu.cu)
    # Link the BFG library so the GPU driver shares the same authoritative
    # `load_wavefunction` / `Cluster` infrastructure as the CPU driver
    # (P2.1, third slice). `ed_bfg` brings the include path for
    # `ed/bfg/wavefunction_io.h` and the HDF5 link stack PUBLIC-ly.
    target_link_libraries(compute_bfg_order_parameters_gpu
        CUDA::cudart
        ed_bfg
        ${HDF5_LIBRARIES}
    )
    # Explicitly add HDF5 include directories for CUDA compilation
    # Use both target_include_directories and compile options to ensure nvcc gets them
    target_include_directories(compute_bfg_order_parameters_gpu SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS})
    foreach(_hdf5_inc_dir ${HDF5_INCLUDE_DIRS})
        target_compile_options(compute_bfg_order_parameters_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${_hdf5_inc_dir}>)
    endforeach()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(compute_bfg_order_parameters_gpu OpenMP::OpenMP_CXX)
        target_compile_definitions(compute_bfg_order_parameters_gpu PRIVATE _OPENMP)
    endif()
    set_target_properties(compute_bfg_order_parameters_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()


# ED and TPQ_DSSF inherit BLAS/LAPACK/LAPACKE/HDF5/ARPACK/MPI/CUDA/OpenMP
# transitively from ed_solvers_cpu (and ed_solvers_gpu when WITH_CUDA), since
# those libraries declare their dependencies as PUBLIC.

# TPQ_DSSF still needs its parallel-HDF5 override (different libs, extra
# compile definition).
if(TARGET TPQ_DSSF AND HAS_PARALLEL_HDF5)
    target_include_directories(TPQ_DSSF PRIVATE ${HDF5_PARALLEL_INCLUDE_DIR})
    target_compile_definitions(TPQ_DSSF PRIVATE HDF5_PARALLEL)
    target_link_libraries(TPQ_DSSF PRIVATE
        ${HDF5_PARALLEL_LIBRARIES}
        z sz crypto curl dl m
    )
endif()

# Link libraries for test_dynamical_correlation
if(TARGET test_dynamical_correlation)
    target_link_libraries(test_dynamical_correlation
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
    )
    
    # Add CPU optimization flags
    target_compile_options(test_dynamical_correlation PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
    
    # Add OpenMP if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_dynamical_correlation OpenMP::OpenMP_CXX)
    endif()
endif()


# Link libraries for GPU Lanczos example
if(TARGET GPU_LANCZOS_EXAMPLE)
    target_link_libraries(GPU_LANCZOS_EXAMPLE
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
    )
endif()


# Link libraries for CUDA example if it exists
if(TARGET CUDA_EXAMPLE)
    target_link_libraries(CUDA_EXAMPLE
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()

# Link libraries for CUDA test if it exists
if(TARGET CUDA_TEST)
    target_link_libraries(CUDA_TEST
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
        ${ARPACK_LIBRARY}
    )
endif()


# If using oneMKL via its modern CMake target, propagate it to the libraries.
if(WITH_MKL AND USE_ONEMKL AND MKL_FOUND)
    target_link_libraries(ed_solvers_cpu PUBLIC MKL::MKL)
    if(TARGET ed_solvers_gpu)
        target_link_libraries(ed_solvers_gpu PUBLIC MKL::MKL)
    endif()
endif()

# =============================================================================
# INSTALL RULES + EDConfig.cmake export (P1.7 / audit Q5)
#
# Allows the package to be installed into a prefix and consumed by downstream
# CMake projects via:
#
#   find_package(ED CONFIG REQUIRED)
#   target_link_libraries(myapp PRIVATE ED::ed_solvers_cpu)
#
# Layout under <prefix>:
#   bin/                        ED, TPQ_DSSF, compute_bfg_order_parameters
#   include/ed/                 public headers (kept hierarchical)
#   lib/                        libed_io.a, libed_core.a, libed_solvers_cpu.a
#                               libed_solvers_gpu.a (when WITH_CUDA)
#   lib/cmake/ED/               EDConfig.cmake, EDConfigVersion.cmake,
#                               EDTargets.cmake
# =============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_ED_INSTALL_TARGETS ed_io ed_core ed_dssf ed_bfg ed_symmetry ed_solvers_cpu ed_cli)
if(TARGET ed_solvers_gpu)
    list(APPEND _ED_INSTALL_TARGETS ed_solvers_gpu)
endif()

install(TARGETS ${_ED_INSTALL_TARGETS} ED
    EXPORT EDTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET TPQ_DSSF)
    install(TARGETS TPQ_DSSF RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(TARGET compute_bfg_order_parameters)
    install(TARGETS compute_bfg_order_parameters
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Public headers (preserve subdirectory layout).
install(DIRECTORY ${INCLUDE_DIR}/ed
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

install(EXPORT EDTargets
        FILE EDTargets.cmake
        NAMESPACE ED::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ED
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/EDConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ED
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/EDConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/EDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/EDConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ED
)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
if(WITH_CUDA)
    message(STATUS "  CUDA Support: YES")
    message(STATUS "  CUDA Version: ${CUDAToolkit_VERSION}")
    if(CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
else()
    message(STATUS "  CUDA Support: NO")
endif()
message(STATUS "  MPI Support: ${WITH_MPI}")

# Build configuration summary and performance options
message(STATUS "")
message(STATUS "=== Performance Configuration Summary ===")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  BLAS Profile: ${BLAS_PROFILE}")
if(BLAS_LIBRARIES)
    message(STATUS "  BLAS Libraries: ${BLAS_LIBRARIES}")
endif()
if(LAPACK_LIBRARIES)
    message(STATUS "  LAPACK Libraries: ${LAPACK_LIBRARIES}")
endif()
if(WITH_SCALAPACK AND SCALAPACK_LIBRARIES)
    message(STATUS "  ScaLAPACK Libraries: ${SCALAPACK_LIBRARIES}")
    message(STATUS "  ScaLAPACK Compatible: ${SCALAPACK_COMPATIBLE}")
elseif(WITH_MPI)
    message(STATUS "  ScaLAPACK: DISABLED")
endif()
message(STATUS "  Aggressive Optimizations: ${ENABLE_AGGRESSIVE_OPTS}")
message(STATUS "  Profile-Guided Optimization: ${ENABLE_PROFILE_GUIDED_OPTS}")
message(STATUS "  Native Architecture: ${ENABLE_NATIVE_ARCH}")
message(STATUS "  CUDA Support: ${WITH_CUDA}")
message(STATUS "  MPI Support: ${WITH_MPI}")
message(STATUS "  OpenMP Support: ${OpenMP_CXX_FOUND}")

# CPU information (informational; uses CPU_VENDOR set by EDPlatform).
_ed_report_cpu_features()

# Apply CPU optimization flags to driver TUs (the bulk of the codebase
# already inherits them from the static libraries' PRIVATE compile options).
target_compile_options(ED PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)
if(TARGET TPQ_DSSF)
    target_compile_options(TPQ_DSSF PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
endif()

# =============================================================================
# Optional Python bindings (P2.7 / audit "modern python interface").
#
# When enabled, builds the `quantum_ed._core` pybind11 extension using the
# project's static-library stack. Two ways to enable:
#
#   * Top-level CMake: `cmake -B build -DED_BUILD_PYTHON=ON ...` and then
#     run `import quantum_ed` from `python/`.
#   * `pip install .` from the project root, which uses scikit-build-core
#     (configured in pyproject.toml) and forces ED_BUILD_PYTHON=ON behind
#     the scenes.
#
# The default is OFF so the C++-only build stays lean for HPC clusters
# without a Python toolchain.
# =============================================================================
option(ED_BUILD_PYTHON "Build the quantum_ed pybind11 Python bindings" OFF)
if(SKBUILD)
    set(ED_BUILD_PYTHON ON CACHE BOOL "" FORCE)
endif()

if(ED_BUILD_PYTHON)
    add_subdirectory(python/quantum_ed)
endif()

# =============================================================================
# Documentation site (Doxygen + Sphinx + Breathe -> GitHub Pages).
#
# Opt-in via `-DED_BUILD_DOCS=ON`. The required Python packages
# (sphinx, breathe, myst-parser, furo) are listed in docs/requirements.txt;
# install them inside your virtualenv before configuring with this flag.
# Doxygen itself must be available on the host (`apt install doxygen`).
#
# Targets (only created when ED_BUILD_DOCS is on):
#   * `doxygen`  -- regenerates the C++ API XML
#   * `sphinx`   -- builds the public HTML site (depends on `doxygen`)
#
# CI builds the site on every push to main and publishes the result to
# GitHub Pages; see .github/workflows/docs.yml for the deployment pipeline.
# =============================================================================
option(ED_BUILD_DOCS "Build the Doxygen + Sphinx documentation site" OFF)
if(ED_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# =============================================================================
# DMRG: removed in P0.9 (audit Q13).
# The DMRG scaffold (include/ed/dmrg/, tests/test_dmrg.cpp) was an explicit
# "implement the TODOs" placeholder with no working iDMRG backend. Rather
# than carry dead code, it has been deleted entirely. If DMRG is needed in
# the future, prefer integrating an established library (ITensor) over
# rebuilding the scaffold from scratch.
# =============================================================================

# =============================================================================
# ED test suite (CPU).
#
# A small, fast set of regression tests that exercise the matrix-free Operator
# path, the fixed-Sz operator, full diagonalization, every Lanczos-family
# solver, the in-memory basis buffer, and the Trans.dat/InterAll.dat loaders.
# Enable with `-DBUILD_ED_TESTS=ON` (defaults to ON) and run with `ctest`.
# =============================================================================
option(BUILD_ED_TESTS "Build the ED regression test suite" ON)

# -----------------------------------------------------------------------------
# Google-Benchmark micro-benchmarks (P2.13). Off by default to keep the
# default build short; opt in with `-DED_BUILD_BENCHMARKS=ON`. The CI nightly
# job (.github/workflows/benchmarks.yml) flips this to ON.
# -----------------------------------------------------------------------------
option(ED_BUILD_BENCHMARKS "Build the Google-Benchmark micro-benchmarks" OFF)
if(ED_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_ED_TESTS)
    enable_testing()

    # Catch2 v3 (P1.8 / audit Q12). Brought in via FetchContent with a
    # system-wide find_package fallback. Defines Catch2::Catch2WithMain.
    include(EDCatch2)

    # Helper to declare a test executable. Each test gets its own binary so
    # CTest can run them in parallel and report failures independently.
    # Tests link against ed_solvers_cpu (which transitively pulls in ed_core,
    # ed_io, BLAS/LAPACK/HDF5/ARPACK/MPI/OpenMP), and against ed_solvers_gpu
    # when CUDA is enabled. This guarantees tests are compiled against the
    # exact same TUs as the production binaries.
    #
    # Tests that opt into the new Catch2 v3 harness pass USE_CATCH2 as a
    # third argument, in which case the helper links Catch2::Catch2WithMain
    # and registers the binary with `catch_discover_tests` so CTest sees
    # each TEST_CASE individually.
    function(ed_add_test test_name test_source)
        add_executable(${test_name} ${test_source})
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        if(WITH_CUDA)
            target_link_libraries(${test_name} PRIVATE ed_solvers_gpu)
        else()
            target_link_libraries(${test_name} PRIVATE ed_solvers_cpu)
        endif()
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
        )

        set(_use_catch2 FALSE)
        if(${ARGC} GREATER 2 AND "${ARGV2}" STREQUAL "USE_CATCH2")
            set(_use_catch2 TRUE)
        endif()

        if(_use_catch2)
            target_link_libraries(${test_name} PRIVATE Catch2::Catch2WithMain)
            catch_discover_tests(${test_name}
                PROPERTIES
                    ENVIRONMENT "ED_TEST_TMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_scratch"
                    TIMEOUT 180
            )
        else()
            add_test(NAME ${test_name} COMMAND ${test_name})
            set_tests_properties(${test_name} PROPERTIES
                ENVIRONMENT "ED_TEST_TMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_scratch"
                TIMEOUT 180
            )
        endif()
    endfunction()

    ed_add_test(test_operator_apply             tests/unit/test_operator_apply.cpp USE_CATCH2)
    ed_add_test(test_fixed_sz_operator          tests/unit/test_fixed_sz_operator.cpp USE_CATCH2)
    ed_add_test(test_full_diagonalization       tests/unit/test_full_diagonalization.cpp USE_CATCH2)
    ed_add_test(test_lanczos_variants           tests/unit/test_lanczos_variants.cpp USE_CATCH2)
    ed_add_test(test_lanczos_basis_buffer       tests/unit/test_lanczos_basis_buffer.cpp USE_CATCH2)
    ed_add_test(test_trans_interall_loading     tests/unit/test_trans_interall_loading.cpp USE_CATCH2)
    ed_add_test(test_observables                tests/unit/test_observables.cpp USE_CATCH2)
    ed_add_test(test_thermal_methods            tests/unit/test_thermal_methods.cpp USE_CATCH2)
    ed_add_test(test_hdf5_io                    tests/unit/test_hdf5_io.cpp USE_CATCH2)
    ed_add_test(test_symmetry                   tests/unit/test_symmetry.cpp USE_CATCH2)
    ed_add_test(test_dssf_operator_spec         tests/unit/test_dssf_operator_spec.cpp USE_CATCH2)
    target_link_libraries(test_dssf_operator_spec PRIVATE ed_dssf)
    target_compile_definitions(test_dssf_operator_spec PRIVATE
        ED_TEST_POSITIONS_4SITE="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/positions_4site.dat"
    )

    # P2.2 (DSSF PR-C): lockdown for the new ed::dssf::run dispatch + DSSFMethod
    # round-trip tokens. Links against ed_cli because the implementation lives
    # there (it needs to reach the workflow functions).
    ed_add_test(test_dssf_engine                tests/unit/test_dssf_engine.cpp USE_CATCH2)
    target_link_libraries(test_dssf_engine PRIVATE ed_cli ed_dssf)

    # P2.3 (DSSF PR-D): bit-for-bit lockdown of the legacy /dynamical/ and
    # /correlations/ HDF5 schemas, so the unified /dssf/ writer cannot
    # accidentally regress them. Reaches into ed_solvers_cpu for HDF5IO.
    ed_add_test(test_dssf_legacy_schema         tests/unit/test_dssf_legacy_schema.cpp USE_CATCH2)
    target_include_directories(test_dssf_legacy_schema PRIVATE ${HDF5_INCLUDE_DIRS})

    # P2.3 (DSSF PR-D): round-trip lockdown for the new unified /dssf/ schema
    # (ensure_metadata + write_record + read_record). Lives in ed_dssf.
    ed_add_test(test_dssf_io                    tests/unit/test_dssf_io.cpp USE_CATCH2)
    target_link_libraries(test_dssf_io PRIVATE ed_dssf)
    target_include_directories(test_dssf_io PRIVATE ${HDF5_INCLUDE_DIRS})

    # P2.11: lock down the programmatic symmetry DSL (ed::sym::Permutation
    # algebra + group_from_generators + Bloch-momentum convention +
    # phantom-irrep filter). Lives in ed_symmetry.
    ed_add_test(test_symmetry_dsl               tests/unit/test_symmetry_dsl.cpp USE_CATCH2)
    target_link_libraries(test_symmetry_dsl PRIVATE ed_symmetry)

    ed_add_test(test_bfg_cluster                tests/unit/test_bfg_cluster.cpp USE_CATCH2)
    target_link_libraries(test_bfg_cluster PRIVATE ed_bfg)
    target_compile_definitions(test_bfg_cluster PRIVATE
        ED_TEST_BFG_KAGOME_1X1_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/bfg_kagome_1x1"
    )

    # P2.1: lockdown for the topology / correlation kernels lifted out of
    # `compute_bfg_order_parameters.cpp` into `ed_bfg`.
    ed_add_test(test_bfg_topology               tests/unit/test_bfg_topology.cpp USE_CATCH2)
    target_link_libraries(test_bfg_topology PRIVATE ed_bfg)
    target_compile_definitions(test_bfg_topology PRIVATE
        ED_TEST_BFG_KAGOME_1X1_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/bfg_kagome_1x1"
    )

    ed_add_test(test_bfg_correlations           tests/unit/test_bfg_correlations.cpp USE_CATCH2)
    target_link_libraries(test_bfg_correlations PRIVATE ed_bfg)

    # P2.1 (4th slice): bond-bilinear structure factors / Fourier-applied
    # dimer kernels shared by the CPU driver, the GPU driver, and the
    # forthcoming Python bindings.
    ed_add_test(test_bfg_structure_factor       tests/unit/test_bfg_structure_factor.cpp USE_CATCH2)
    target_link_libraries(test_bfg_structure_factor PRIVATE ed_bfg)

    # P2.1 (5th slice): bowtie + triangle ring kernels promoted into
    # `ed_bfg::ring_observables`. Same call-site contract as the dimer
    # structure factor tests above, plus toggles for the memory-efficient
    # global flag.
    ed_add_test(test_bfg_ring_observables       tests/unit/test_bfg_ring_observables.cpp USE_CATCH2)
    target_link_libraries(test_bfg_ring_observables PRIVATE ed_bfg)

    # P2.1 (third slice): HDF5 wavefunction loaders shared by the CPU + GPU
    # drivers and by the Python bindings.
    ed_add_test(test_bfg_wavefunction_io        tests/unit/test_bfg_wavefunction_io.cpp USE_CATCH2)
    target_link_libraries(test_bfg_wavefunction_io PRIVATE ed_bfg)
    target_include_directories(test_bfg_wavefunction_io PRIVATE ${HDF5_INCLUDE_DIRS})

    # -------------------------------------------------------------------------
    # CPU/GPU equivalence (P1.9). Only added when WITH_CUDA is ON so non-GPU
    # builds skip compilation entirely. The TEST_CASEs themselves SKIP at
    # runtime when no CUDA device is detected (cudaGetDeviceCount), which
    # keeps the build-only `linux-cuda-build` CI lane green even though it
    # has no NVIDIA hardware attached.
    # -------------------------------------------------------------------------
    if(WITH_CUDA)
        ed_add_test(test_cpu_gpu_equivalence    tests/unit/test_cpu_gpu_equivalence.cpp USE_CATCH2)
    endif()

    # Regression test of the on-disk fallback path for the Lanczos basis
    # buffer. `force_disk_storage()` is cached per-process, so we reuse the
    # variants binary in a separate subprocess with ED_LANCZOS_DISK=1.
    add_test(NAME test_lanczos_variants_disk COMMAND test_lanczos_variants)
    set_tests_properties(test_lanczos_variants_disk PROPERTIES
        ENVIRONMENT "ED_LANCZOS_DISK=1;ED_TEST_TMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_scratch_disk"
        TIMEOUT 240
    )

    # -------------------------------------------------------------------------
    # End-to-end smoke test: run the actual `ED` binary on a 4-site Heisenberg
    # ring fixture and verify the ground-state energy is -2 exactly.
    # -------------------------------------------------------------------------
    set(TEST_CHAIN4_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chain4)
    set(TEST_CHAIN4_OUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/test_scratch/chain4_out)
    set(TEST_CHAIN4_CFG
        ${CMAKE_CURRENT_BINARY_DIR}/test_scratch/chain4.cfg)
    set(ED_BINARY_FALLBACK
        ${CMAKE_CURRENT_BINARY_DIR}/ED)

    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/chain4.cfg.in
        ${TEST_CHAIN4_CFG}
        @ONLY
    )
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_ed_smoke.sh.in
        ${CMAKE_CURRENT_BINARY_DIR}/run_ed_smoke.sh
        @ONLY
    )

    add_test(NAME test_ed_smoke
             COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/run_ed_smoke.sh)
    set_tests_properties(test_ed_smoke PROPERTIES
        ENVIRONMENT "ED_BINARY=$<TARGET_FILE:ED>"
        DEPENDS "ED"
        TIMEOUT 120
    )
endif()
