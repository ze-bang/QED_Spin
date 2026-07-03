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
    project(QED VERSION 0.1.0 LANGUAGES CXX CUDA)
    enable_language(CUDA)
else()
    project(QED VERSION 0.1.0 LANGUAGES CXX)
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
set(PARALLEL_DIR ${SRC_DIR}/parallel)
set(DISTRIBUTED_DIR ${SRC_DIR}/distributed)
set(MATVEC_DIR ${SRC_DIR}/matvec)  # Matvec-unification revamp (Phase 1)

# Include directories - public headers in include/, private in src/
include_directories(${INCLUDE_DIR})
include_directories(${INCLUDE_DIR}/ed/core)
include_directories(${INCLUDE_DIR}/ed/solvers)
include_directories(${INCLUDE_DIR}/ed/io)
include_directories(${INCLUDE_DIR}/ed/gpu)
include_directories(${INCLUDE_DIR}/ed/bfg)
include_directories(${INCLUDE_DIR}/ed/matvec)
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

# -----------------------------------------------------------------------------
# Phase 3c: NCCL detection (used by ed_distributed_gpu).
#
# Multi-GPU dot/norm reductions for ed::distributed::distributed_lanczos
# ride NCCL (Phase 3c stage 1). This block is run BEFORE EDLibraries so
# that `ed_distributed_gpu` can be added conditionally inside that file.
#
# Sets:
#   * NCCL_FOUND          (TRUE / FALSE)
#   * NCCL_INCLUDE_DIRS
#   * NCCL_LIBRARIES
#
# Disabling: set `-DWITH_NCCL=OFF` to forcibly turn the discovery off
# even on hosts where it would otherwise be found (useful for CPU-only
# regression rebuilds on GPU nodes).
# -----------------------------------------------------------------------------
option(WITH_NCCL "Discover NCCL for ed_distributed_gpu (Phase 3c)" ON)
if(WITH_CUDA AND WITH_MPI AND WITH_NCCL)
    find_path(NCCL_INCLUDE_DIRS NAMES nccl.h
              HINTS $ENV{NCCL_ROOT}/include $ENV{NCCL_HOME}/include
                    /usr/include /usr/local/include /opt/nccl/include)
    find_library(NCCL_LIBRARIES NAMES nccl
                 HINTS $ENV{NCCL_ROOT}/lib $ENV{NCCL_HOME}/lib
                       /usr/lib/x86_64-linux-gnu /usr/local/lib /opt/nccl/lib)
    if(NCCL_INCLUDE_DIRS AND NCCL_LIBRARIES)
        set(NCCL_FOUND TRUE)
        message(STATUS "  NCCL Support: DETECTED (Phase 3c multi-GPU enabled)")
        message(STATUS "    NCCL_INCLUDE_DIRS: ${NCCL_INCLUDE_DIRS}")
        message(STATUS "    NCCL_LIBRARIES:    ${NCCL_LIBRARIES}")
    else()
        set(NCCL_FOUND FALSE)
        message(STATUS "  NCCL Support: NOT FOUND "
                       "(Phase 3c multi-GPU disabled; pass NCCL_ROOT to override)")
    endif()
else()
    set(NCCL_FOUND FALSE)
    if(NOT WITH_NCCL)
        message(STATUS "  NCCL Support: DISABLED via WITH_NCCL=OFF")
    endif()
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
# Wave C2 (Full Unified-Interface Collapse, May 2026): `ed_cli` already
# pulls in `ed_distributed` PUBLIC-ly so binaries linking `ed_cli`
# resolve `DistributedOperator` / `DistributedSymmetryOperator`
# constructors that the inline `ed::make_operator` factory references.

# NOTE: The historical `TPQ_DSSF` standalone binary (~4.2 kLOC of duplicated
# kernels + a positional CLI + parallel HDF5 plumbing) was deleted in P2.14.
# Every feature it offered is now reachable through `ED dssf <method>`, which
# routes the same `compute_*_workflow` kernels through `ed::dssf::run(...)`.
# See `docs/history/MODERNIZATION_AUDIT.md` for migration notes and the historical context.

# -----------------------------------------------------------------------------
# Phase 3b #4: ed_distributed_main (MPI cluster launcher)
#
# Tiny standalone driver that exercises ed::distributed::distributed_lanczos
# and ed::distributed::distributed_ftlm on an unsymmetrised Heisenberg chain.
# Designed to validate a build/queue/interconnect triple end-to-end before
# scientists try the full DSSF/FTLM workflows on top of it. See the file
# header for the CLI surface and `scripts/distributed/run_dist.sh` for the
# launcher template.
# -----------------------------------------------------------------------------
if(WITH_MPI AND TARGET ed_distributed)
    add_executable(ed_distributed_main src/cli/ed_distributed_main.cpp)
    target_link_libraries(ed_distributed_main PRIVATE ed_distributed)
    target_compile_options(ed_distributed_main PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
    # Phase 3c: when NCCL is discovered, also link the multi-GPU library so
    # the launcher can print runtime status (`nccl_status_string()`) and so
    # the future GPU-backed distributed_lanczos can resolve its symbols.
    # ed_distributed_gpu propagates ED_HAVE_NCCL=1 to consumers via PUBLIC
    # compile_definitions, so the CLI TU automatically picks the
    # NCCL-aware code branch when the library is in scope.
    if(TARGET ed_distributed_gpu)
        target_link_libraries(ed_distributed_main PRIVATE ed_distributed_gpu)
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


# ED inherits BLAS/LAPACK/LAPACKE/HDF5/ARPACK/MPI/CUDA/OpenMP transitively
# from ed_solvers_cpu (and ed_solvers_gpu when WITH_CUDA), since those
# libraries declare their dependencies as PUBLIC.

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
    )
endif()

# Link libraries for CUDA test if it exists
if(TARGET CUDA_TEST)
    target_link_libraries(CUDA_TEST
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${LAPACKE_LIBRARIES}
        ${EXTRA_LINALG_LIBRARIES}
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
# INSTALL RULES + QEDConfig.cmake export (P1.7 / audit Q5)
#
# Allows the package to be installed into a prefix and consumed by downstream
# CMake projects via:
#
#   find_package(QED CONFIG REQUIRED)
#   target_link_libraries(myapp PRIVATE QED::ed_solvers_cpu)
#
# Layout under <prefix>:
#   bin/                        ED, compute_bfg_order_parameters
#   include/ed/                 public headers (kept hierarchical)
#   lib/                        libed_io.a, libed_core.a, libed_solvers_cpu.a
#                               libed_solvers_gpu.a (when WITH_CUDA)
#   lib/cmake/QED/              QEDConfig.cmake, QEDConfigVersion.cmake,
#                               QEDTargets.cmake
# =============================================================================
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(_ED_INSTALL_TARGETS ed_parallel ed_io ed_core ed_matvec ed_dssf ed_bfg ed_symmetry ed_input ed_solvers_cpu ed_cli)
if(WITH_MPI AND TARGET ed_distributed)
    list(APPEND _ED_INSTALL_TARGETS ed_distributed)
endif()
if(TARGET ed_solvers_gpu)
    list(APPEND _ED_INSTALL_TARGETS ed_solvers_gpu)
endif()

install(TARGETS ${_ED_INSTALL_TARGETS} ED
    EXPORT QEDTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET compute_bfg_order_parameters)
    install(TARGETS compute_bfg_order_parameters
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Phase 3b #4: install the distributed cluster launcher alongside ED.
if(TARGET ed_distributed_main)
    install(TARGETS ed_distributed_main
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

install(EXPORT QEDTargets
        FILE QEDTargets.cmake
        NAMESPACE QED::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QED
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/QEDConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QED
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/QEDConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QED
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

# =============================================================================
# Optional Python bindings (P2.7 / audit "modern python interface").
#
# When enabled, builds the `qed._core` pybind11 extension using the
# project's static-library stack. Two ways to enable:
#
#   * Top-level CMake: `cmake -B build -DED_BUILD_PYTHON=ON ...` and then
#     run `import qed` from `python/`.
#   * `pip install .` from the project root, which uses scikit-build-core
#     (configured in pyproject.toml) and forces ED_BUILD_PYTHON=ON behind
#     the scenes.
#
# The default is OFF so the C++-only build stays lean for HPC clusters
# without a Python toolchain.
# =============================================================================
option(ED_BUILD_PYTHON "Build the qed pybind11 Python bindings" OFF)
if(SKBUILD)
    set(ED_BUILD_PYTHON ON CACHE BOOL "" FORCE)
endif()

if(ED_BUILD_PYTHON)
    add_subdirectory(python/qed)
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

# -----------------------------------------------------------------------------
# Runnable use-case examples under examples/. Off by default; opt in via
# `-DED_BUILD_EXAMPLES=ON`. See examples/README.md for the full index.
# -----------------------------------------------------------------------------
option(ED_BUILD_EXAMPLES "Build the runnable C++ examples in examples/" OFF)
if(ED_BUILD_EXAMPLES)
    add_subdirectory(examples)
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
        # Wave A (Full unified-interface collapse, May 2026): `make_operator.h`
        # references `DistributedOperator` / `DistributedSymmetryOperator`
        # constructors in inline factory bodies under `WITH_MPI`. The compiler
        # emits the inline bodies for every TU that includes the header even
        # when the runtime branch is unreachable, so the linker needs the
        # distributed library when MPI is on. Link transitively here so test
        # authors do not have to remember per-target.
        if(WITH_MPI AND TARGET ed_distributed)
            target_link_libraries(${test_name} PRIVATE ed_distributed)
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
            # SKIP_RETURN_CODE 4: when a Catch2 v3 TEST_CASE explicitly
            # calls SKIP(...) and is the only case matched by the
            # `--test-name-pattern` ctest spec, Catch2 exits with code 4
            # (== "no tests ran"). Without this property CTest reports the
            # otherwise correct skip as a Failed test in the CI lane.
            catch_discover_tests(${test_name}
                PROPERTIES
                    ENVIRONMENT "ED_TEST_TMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_scratch"
                    TIMEOUT 180
                    SKIP_RETURN_CODE 4
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
    # Krylov-kernel unification Phase A: one Lanczos algorithm shared by
    # CPU / GPU / CPU+MPI / GPU+MPI via the abstract Backend interface.
    # This test pins agreement with the legacy sequential-MGS body and
    # orthogonality of the CGS2 basis.
    ed_add_test(test_lanczos_kernel             tests/unit/test_lanczos_kernel.cpp USE_CATCH2)
    # test_square_operator was retired in the minimalist-architecture rev
    # (May 2026) together with `ed/core/square_operator.h`,
    # `ed/core/square_operator_factories.h`,
    # `ed/core/rectangular_operator.h`, and
    # `ed/core/basis_policy_runtime.h`. Those types had zero production
    # consumers -- the only call sites were this test and the headers
    # mutually including each other. Solvers consume the unified
    # `ed::matvec::MatVecOperator` (which `Operator` /
    # `FixedSzOperator` / `GPUOperator` / `DistributedOperator` /
    # `StreamingSymmetryOperator` all derive from) directly. If a
    # future workstream needs a runtime BasisPolicy + wrapper, build
    # it fresh against an actual consumer.
    ed_add_test(test_kernel_facades             tests/unit/test_kernel_facades.cpp USE_CATCH2)
    ed_add_test(test_orchestrator               tests/unit/test_orchestrator.cpp USE_CATCH2)
    # Operator-collapse Phase 2a (Jun 2026): orchestrator-level GPU parity
    # for the plain Operator / FixedSzOperator device-mirror lane
    # (supports_device_matvec + bind_cuda -> CudaMatVecBackend). SKIPs at
    # runtime without a CUDA device; no-op TU when WITH_CUDA is OFF.
    ed_add_test(test_operator_gpu_parity        tests/unit/test_operator_gpu_parity.cpp USE_CATCH2)
    # Phase A (PR-1 step 6) of the "mirror examples" plan (May 2026):
    # pins byte-equality between the Python-named `ed::api::*` kwargs
    # facade and the legacy `ed::workflows::*` orchestrator across a
    # representative cell from each method.
    ed_add_test(test_api_mirror                 tests/unit/test_api_mirror.cpp USE_CATCH2)
    # Eigenvector-save contract (May 2026): every CPU-single-rank solver
    # method (Lanczos / BlockLanczos / KrylovSchur / FullDiag) MUST honor
    # `compute_vectors=true` + `output_dir=...` by writing
    # `<output_dir>/ed_results.h5` and reporting the path via
    # `GroundStateResult::hdf5_path` (which `EDResults.eigenvectors_path`
    # mirrors on the Python side).
    ed_add_test(test_eigenvector_save           tests/unit/test_eigenvector_save.cpp USE_CATCH2)
    # Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
    # mirror-image of `test_eigenvector_save` but for
    # `ed::workflows::thermal` -- pins the uniform `ed_results.h5`
    # contract for FTLM / LTLM / KPM_DOS (thermo curves) and
    # mTPQ / cTPQ (trajectory + probe-beta state vectors).
    ed_add_test(test_thermal_save               tests/unit/test_thermal_save.cpp USE_CATCH2)
    # Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026): pins the
    # ``SpectralOptions::initial_state`` contract on the GroundStateCF
    # lane (auto-GS, user-supplied GS, and TPQ-style warm seeds).
    ed_add_test(test_spectral_initial_state     tests/unit/test_spectral_initial_state.cpp USE_CATCH2)
    # Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026): pins the
    # ``SpectralOptions::Method::KpmDynamical`` lane against the dense
    # Lehmann reference (smoothed L2 < 5%) on the 4-site Heisenberg ring.
    ed_add_test(test_kpm_dynamical_spectral     tests/unit/test_kpm_dynamical_spectral.cpp USE_CATCH2)
    ed_add_test(test_minimalist_collapse        tests/unit/test_minimalist_collapse.cpp USE_CATCH2)
    # Full Unified-Interface Collapse, end-to-end acceptance (May 2026):
    # walks every (source x axis x workflow x method) cell of the
    # unified interface on the same 6-site Heisenberg ring and asserts
    # numerical agreement against the Bethe-ansatz reference E_0.
    ed_add_test(test_unified_interface_e2e      tests/integration/test_unified_interface_e2e.cpp USE_CATCH2)
    # Operator-collapse option-1 acceptance (Jun 2026): the
    # `ed::make_sector_operators(OperatorSpec)` factory routes a
    # directory-loaded Hamiltonian through the P5 sector-set enumerator
    # and solves each sector via ed::workflows::solve, cross-checking the
    # full union spectrum against the legacy streaming-symmetry path.
    ed_add_test(test_make_sector_operators_e2e  tests/integration/test_make_sector_operators_e2e.cpp USE_CATCH2)
    # Wave A acceptance (Full unified-interface collapse, May 2026):
    # the `ed::make_operator(OperatorSpec)` factory + the new
    # `bind_<Backend>` override matrix.
    ed_add_test(test_make_operator              tests/unit/test_make_operator.cpp USE_CATCH2)
    # test_method_canonicalize retired in the minimalist-architecture rev
    # (May 2026): all the deprecated `_GPU`/`_FIXED_SZ`/`_MPI`/`_CUDA`
    # enum aliases and ARPACK/SCALAPACK/DAVIDSON/LOBPCG variants this
    # test was pinning have been removed; the canonicalizer is no longer
    # required because GPU/MPI/fixed-Sz are now first-class EDParameters
    # flags rather than embedded in the DiagonalizationMethod enum.
    ed_add_test(test_full_diagonalization       tests/unit/test_full_diagonalization.cpp USE_CATCH2)
    # Smoke-test the unified orchestrator
    # `ed::workflows::solve(LinearOperator&, SolveOptions)` — Sz guard,
    # auto fixed-Sz projection, default-method picker, GPU fallback.
    ed_add_test(test_auto_solve                 tests/unit/test_auto_solve.cpp USE_CATCH2)
    # End-to-end correctness of every ground-state algorithm reachable
    # through `ed::workflows::solve` (LANCZOS, BLOCK_LANCZOS,
    # KRYLOV_SCHUR, FULL) with auto-Sz projection on. Validates the
    # matvec wiring through the virtual `LinearOperator::apply` and
    # the orchestrator's solver-family switch in one place.
    ed_add_test(test_ed_solver_matrix_e2e       tests/unit/test_ed_solver_matrix_e2e.cpp USE_CATCH2)
    # Matvec-unification audit follow-up: cross-sector Z-recombination of
    # finite-T ThermodynamicData. The streaming-symmetry kernel relies on
    # `ed::core::combine_sector_thermodynamics` to merge per-sector
    # ThermodynamicData blocks back into the full-Hilbert thermo.
    # Reference is the dense `calculate_thermodynamics_from_spectrum` on
    # the union of all sector spectra.
    ed_add_test(test_sector_thermo              tests/unit/test_sector_thermo.cpp USE_CATCH2)
    # Full Unified-Interface Collapse, Wave F-partial (May 2026):
    # `tests/unit/test_dispatch_streaming_thermo.cpp` and its target
    # were removed together with `include/ed/core/dispatch.h`. The
    # streaming-symmetry + FTLM end-to-end behaviour is now covered by
    # `tests/unit/test_auto_thermal.cpp` (which uses the unified entry
    # `ed::workflows::thermal(...)`).
    # One canonical finite-T entry point: `ed::workflows::thermal(...)`.
    # Exercises auto-Sz + auto-symmetry + Z-recombination via both the
    # in-memory and the directory overloads.
    ed_add_test(test_auto_thermal               tests/unit/test_auto_thermal.cpp USE_CATCH2)
    target_link_libraries(test_auto_thermal PRIVATE ed_input)
    # Numerical accuracy: all finite-T methods vs full dense-diag reference,
    # across all backends (CPU, GPU), all symmetry modes (none, Sz, spatial,
    # Sz+spatial), multiple random seeds.
    ed_add_test(test_thermal_dense_ref          tests/unit/test_thermal_dense_ref.cpp USE_CATCH2)
    # Smoke-test the unified spectral entry point
    # `ed::workflows::spectral(DSSFRequest, SpectralOptions)`:
    # method-selection rule + QEDConfig-null guard.
    ed_add_test(test_auto_dssf                  tests/unit/test_auto_dssf.cpp USE_CATCH2)
    target_link_libraries(test_auto_dssf PRIVATE ed_cli ed_dssf ed_input)
    # ED Cleanup Sweep Phase 5 (May 2026): test_dssf_tune / test_diag_tune
    # locked down the cross-cutting DSSF / diag tune heuristics that
    # were deleted along with `include/ed/auto/{diag,dssf}_tune.h`.
    # The per-kernel knobs are now exercised by the kernel-specific
    # tests below.
    ed_add_test(test_lanczos_variants           tests/unit/test_lanczos_variants.cpp USE_CATCH2)
    ed_add_test(test_lanczos_basis_buffer       tests/unit/test_lanczos_basis_buffer.cpp USE_CATCH2)
    # Phase 3a #1: Krylov-state checkpoint / restart for the default Lanczos
    # solver (round-trip + resume-equivalence + validation-error coverage).
    ed_add_test(test_lanczos_checkpoint         tests/unit/test_lanczos_checkpoint.cpp USE_CATCH2)
    # Phase 3a #2: Out-of-core blocked-tile reorthogonalization (CGS correctness,
    # threshold + skip filters, in-memory + on-disk tile loading, end-to-end
    # tile-size invariance for lanczos_selective_reorth).
    ed_add_test(test_lanczos_reorth             tests/unit/test_lanczos_reorth.cpp USE_CATCH2)
    # Phase 3a #4: NUMA first-touch + thread-pinning hooks (env-knob defaults +
    # parsing, threshold + counter contract, idempotent pinning, end-to-end
    # Lanczos eigenvalue invariance with knobs on vs off).
    ed_add_test(test_numa                       tests/unit/test_numa.cpp USE_CATCH2)
    # Phase 3a #5: SortedUint64Index — drop-in replacement for
    # `unordered_map<uint64_t, size_t>` in the streaming-symmetry hot path.
    # Coverage: empty / build / duplicate-key / clear / sort-invariant +
    # 1M-entry stress vs unordered_map oracle (memory + correctness).
    ed_add_test(test_sorted_uint64_index        tests/unit/test_sorted_uint64_index.cpp USE_CATCH2)

    # -------------------------------------------------------------------------
    # Phase 3b: distributed-memory tests. Always built when WITH_MPI is ON
    # (regardless of MPIEXEC discovery); each test is registered N times --
    # once per `np` count in {1, 2, 4} -- and launched via `mpiexec -np N`.
    # The test binary uses Catch2 with a manual MPI_Init/Finalize main, and
    # rank 0 owns stdout while non-zero ranks pipe to /dev/null. A failed
    # REQUIRE() on any rank propagates via MPI_Allreduce(MPI_MAX) to the
    # CTest exit code.
    # -------------------------------------------------------------------------
    if(WITH_MPI AND TARGET ed_distributed AND MPIEXEC_EXECUTABLE)
        function(ed_add_mpi_test test_name test_source)
            add_executable(${test_name} ${test_source})
            target_include_directories(${test_name} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/tests
            )
            target_link_libraries(${test_name} PRIVATE
                ed_distributed Catch2::Catch2)
            target_compile_options(${test_name} PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
            )
            foreach(np IN ITEMS 1 2 4)
                set(_tname "${test_name}_np${np}")
                add_test(NAME ${_tname}
                         COMMAND ${MPIEXEC_EXECUTABLE}
                                 ${MPIEXEC_NUMPROC_FLAG} ${np}
                                 ${MPIEXEC_PREFLAGS}
                                 $<TARGET_FILE:${test_name}>
                                 ${MPIEXEC_POSTFLAGS})
                set_tests_properties(${_tname} PROPERTIES
                    ENVIRONMENT
                        "ED_TEST_TMP_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_scratch_${_tname}"
                    TIMEOUT 240
                    LABELS "mpi;phase3b"
                )
            endforeach()
        endfunction()

        # Phase 3b #1: distributed matrix-free SpMV (DistributedOperator)
        ed_add_mpi_test(test_distributed_operator
                        tests/unit/test_distributed_operator.cpp)

        # P4: policy-agnostic CommPlan halo-exchange schedule (the reusable
        # extraction of DistributedOperator's build_comm_pattern_ for the
        # unified MemSpace=DistributedHost matvec lane). Self-contained; no
        # Operator dependency. Validates geometry, send/recv conservation,
        # and end-to-end halo correctness (complex + real lanes) at np 1/2/4.
        ed_add_mpi_test(test_comm_plan
                        tests/unit/test_comm_plan.cpp)

        # P4: MpiMatVecImpl -- the MemSpace=DistributedHost matrix-free SpMV
        # built on CommPlan + the GATHER kernel (the per-apply body lifted
        # out of DistributedOperator::apply_local_). Validates the slab SpMV
        # (complex + real lanes) against the INDEPENDENT serial SCATTER
        # kernel over the full Hilbert space at np 1/2/4.
        ed_add_mpi_test(test_mpi_matvec_impl
                        tests/unit/test_mpi_matvec_impl.cpp)

        # Phase 3b #2: distributed Lanczos (rank-local Krylov + MPI_Allreduce)
        ed_add_mpi_test(test_distributed_lanczos
                        tests/unit/test_distributed_lanczos.cpp)

        # "Universal save contract" follow-up (May 2026): rank-0
        # emission of <output_dir>/ed_results.h5 from
        # ed::workflows::solve in the MPI lane (aggregate fields only;
        # slab-distributed eigenvectors stay in per-rank rank_<r>.h5
        # files). Pins the contract at np ∈ {1, 2, 4}.
        ed_add_mpi_test(test_distributed_workflows_save
                        tests/unit/test_distributed_workflows_save.cpp)
        # The orchestrator template-instantiates ``solve_on<CudaBackend>``
        # under WITH_CUDA so the linker needs the GPU solver libs even
        # when the test exercises only the CPU/MPI lane.
        if(WITH_CUDA AND TARGET ed_solvers_gpu)
            target_link_libraries(test_distributed_workflows_save PRIVATE
                ed_solvers_gpu)
        endif()
        target_link_libraries(test_distributed_workflows_save PRIVATE
            ${HDF5_LIBRARIES})
        target_include_directories(test_distributed_workflows_save PRIVATE
            ${HDF5_INCLUDE_DIRS})

        # Phase 3b #3: distributed FTLM (J&P trace estimator with Ritz weights,
        # MPI parallelism over samples)
        ed_add_mpi_test(test_distributed_ftlm
                        tests/unit/test_distributed_ftlm.cpp)

        # Phase 3b #6: distributed Lanczos eigenvector reconstruction
        # (rank-local V_local @ U[:,k] contraction; assemble via Allgatherv;
        # ||H psi - E psi|| < 1e-8 lockdown)
        ed_add_mpi_test(test_distributed_eigenvectors
                        tests/unit/test_distributed_eigenvectors.cpp)

        # Phase 3b #8: distributed canonical TPQ (Taylor-truncated
        # e^{-(delta/2) H} on rank-local |psi>; cross-check vs exact
        # thermal energy at np=1,2,4)
        ed_add_mpi_test(test_distributed_tpq
                        tests/unit/test_distributed_tpq.cpp)

        # Phase 3b #7 stage 2 prep: orbit-aware MPI_Alltoallv halo plan
        # consumed by the DistributedSymmetryOperator. Synthetic
        # 1-D ring lockdown at np=1,2,4 (no GPU dependency).
        ed_add_mpi_test(test_orbit_halo_plan
                        tests/unit/test_orbit_halo_plan.cpp)

        # Phase 3b #7 stage 2: full DistributedSymmetryOperator end-to-end
        # (orbit-aware partition + halo + sparse symmetry-projected SpMV).
        # Cross-checks distributed apply vs an independently-built dense
        # reference of H_q on N=4/N=6 Heisenberg chains with translation
        # symmetry, every momentum sector, np ∈ {1, 2, 4}.
        ed_add_mpi_test(test_distributed_symmetry_operator
                        tests/unit/test_distributed_symmetry_operator.cpp)
        target_link_libraries(test_distributed_symmetry_operator PRIVATE
            ed_symmetry)

        # Phase 3b #7 stage 3: distributed Lanczos rebased onto
        # DistributedSymmetryOperator. Cross-checks ground-state energies
        # vs a dense Eigen::SelfAdjointEigenSolver reference of H_q on
        # N=4/N=6 Heisenberg chains with translation symmetry across np ∈ {1,2,4}.
        ed_add_mpi_test(test_distributed_lanczos_symmetry
                        tests/unit/test_distributed_lanczos_symmetry.cpp)
        target_link_libraries(test_distributed_lanczos_symmetry PRIVATE
            ed_symmetry)

        # Phase D step 2: distributed Krylov-Schur rebased onto
        # DistributedSymmetryOperator (CPU-only, no NCCL). Cross-checks
        # the leading min(k_target, ks_locked) eigenvalues against
        # `distributed_lanczos_symmetry` at the same seed/max_iter on
        # N=4/N=6 Heisenberg chains across all momentum sectors at
        # np ∈ {1, 2, 4}.
        ed_add_mpi_test(test_distributed_krylov_schur_symmetry
                        tests/unit/test_distributed_krylov_schur_symmetry.cpp)
        target_link_libraries(test_distributed_krylov_schur_symmetry PRIVATE
            ed_symmetry)

        # Phase D step 4: distributed FTLM rebased onto
        # DistributedSymmetryOperator (CPU-only, no NCCL). Verifies that
        # summing per-sector Z(beta) over every momentum sector of an
        # N=4 PBC Heisenberg chain reproduces the full-space partition
        # function within the J&P trace-estimator noise floor at np
        # {1, 2, 4}.
        ed_add_mpi_test(test_distributed_ftlm_symmetry
                        tests/unit/test_distributed_ftlm_symmetry.cpp)
        target_link_libraries(test_distributed_ftlm_symmetry PRIVATE
            ed_symmetry)

        # Phase E step 1: per-sector canonical TPQ rebased onto
        # DistributedSymmetryOperator (CPU-only). Verifies that the
        # sample-averaged <H>(beta) measured inside one momentum sector
        # of an N=4 PBC Heisenberg chain converges to the dense-projected
        # thermal energy of that sector at np {1, 2, 4}.
        ed_add_mpi_test(test_distributed_tpq_symmetry
                        tests/unit/test_distributed_tpq_symmetry.cpp)
        target_link_libraries(test_distributed_tpq_symmetry PRIVATE
            ed_symmetry)

        # Phase 3c #1: NCCL multi-GPU collectives. Built only when
        # ed_distributed_gpu is alive (WITH_CUDA + NCCL_FOUND); registered
        # at np {1,2,4} same as the CPU MPI tests. The tests SKIP when no
        # CUDA device is visible at runtime (e.g. CI on a CPU box) so the
        # CTest invocation stays green even without GPUs.
        if(TARGET ed_distributed_gpu)
            # Helper to register a Phase 3c MPI+GPU test at np {1,2,4}.
            function(ed_add_phase3c_test test_name test_source)
                add_executable(${test_name} ${test_source})
                target_include_directories(${test_name} PRIVATE
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests
                )
                target_link_libraries(${test_name} PRIVATE
                    ed_distributed_gpu Catch2::Catch2)
                target_compile_options(${test_name} PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
                )
                foreach(np IN ITEMS 1 2 4)
                    set(_tname "${test_name}_np${np}")
                    add_test(NAME ${_tname}
                             COMMAND ${MPIEXEC_EXECUTABLE}
                                     ${MPIEXEC_NUMPROC_FLAG} ${np}
                                     ${MPIEXEC_PREFLAGS}
                                     $<TARGET_FILE:${test_name}>
                                     ${MPIEXEC_POSTFLAGS})
                    set_tests_properties(${_tname} PROPERTIES
                        TIMEOUT 240
                        LABELS "mpi;phase3c;gpu"
                    )
                endforeach()
            endfunction()

            # Phase 3c #1: NCCL collective wrappers (lockdown vs MPI ref).
            ed_add_phase3c_test(test_multi_gpu_nccl
                                tests/unit/test_multi_gpu_nccl.cpp)

            # Phase 3.5 (Krylov-unification gap-fill): direct lockdown for
            # `ed::matvec::MpiCudaBackend` overrides (dot, nrm2, dot_many,
            # axpy_many, all_reduce_sum). Mirrors `test_cuda_backend.cpp`
            # for the distributed lane. NCCL-gated.
            ed_add_phase3c_test(test_mpi_cuda_backend
                                tests/unit/test_mpi_cuda_backend.cpp)
            target_link_libraries(test_mpi_cuda_backend PRIVATE
                ed_distributed)

            # Phase 3c #2 (stage 2): GPU-resident distributed Lanczos with
            # ncclAllReduce dot/norm + host-staged SpMV. Cross-checks the
            # CPU distributed_lanczos on small Heisenberg chains; SKIPs
            # gracefully when CUDA / NCCL / device count is insufficient.
            ed_add_phase3c_test(test_distributed_lanczos_gpu
                                tests/unit/test_distributed_lanczos_gpu.cpp)
            target_link_libraries(test_distributed_lanczos_gpu PRIVATE
                ed_distributed)

            # Phase 3c #2 (stage 3): fully GPU-resident DistributedGPUOperator
            # (NCCL pairwise SendRecv halo + CUDA SpMV kernel). Cross-checks
            # the CPU DistributedOperator::apply on small Heisenberg chains;
            # SKIPs gracefully when CUDA / NCCL / device count is insufficient.
            ed_add_phase3c_test(test_distributed_gpu_operator
                                tests/unit/test_distributed_gpu_operator.cpp)
            target_link_libraries(test_distributed_gpu_operator PRIVATE
                ed_distributed)

            # Phase A (device matrix MPI+GPU): on-device FTLM. Cross-checks
            # the CPU `distributed_ftlm` on the same MPI_COMM_WORLD with the
            # same operator/seeds/betas, with and without an observable.
            # SKIPs gracefully when CUDA / NCCL / device count is
            # insufficient (the test is build-only on CI's CUDA-build-only
            # lane and runtime-tested on dev hosts with GPUs).
            ed_add_phase3c_test(test_distributed_ftlm_gpu
                                tests/unit/test_distributed_ftlm_gpu.cpp)
            target_link_libraries(test_distributed_ftlm_gpu PRIVATE
                ed_distributed)

            # Phase B (device matrix MPI+GPU): on-device Krylov-Schur
            # (thick-restart Lanczos with locking). Cross-checks the
            # CPU `distributed_krylov_schur` on the lowest k_target
            # eigenvalues at the same tol. SKIPs gracefully when CUDA /
            # NCCL / device count is insufficient.
            ed_add_phase3c_test(test_distributed_krylov_schur_gpu
                                tests/unit/test_distributed_krylov_schur_gpu.cpp)
            target_link_libraries(test_distributed_krylov_schur_gpu PRIVATE
                ed_distributed)

            # Phase C (device matrix MPI+GPU): on-device symmetry-projected
            # SpMV. Cross-checks `DistributedSymmetryOperatorGPU::apply`
            # against the CPU `DistributedSymmetryOperator::apply` on
            # Heisenberg chains with translation symmetry across all
            # momentum sectors at np in {1, 2, 4}.
            ed_add_phase3c_test(test_distributed_symmetry_operator_gpu
                                tests/unit/test_distributed_symmetry_operator_gpu.cpp)
            target_link_libraries(test_distributed_symmetry_operator_gpu PRIVATE
                ed_distributed ed_symmetry)

            # Phase D step 1 (path matrix symm × mpi+gpu): on-device
            # Lanczos consuming a `DistributedSymmetryOperatorGPU`. Cross-
            # checks the GPU symm Lanczos eigenvalues against the CPU
            # `distributed_lanczos_symmetry` at the same seed/max_iter.
            ed_add_phase3c_test(test_distributed_lanczos_gpu_symmetry
                                tests/unit/test_distributed_lanczos_gpu_symmetry.cpp)
            target_link_libraries(test_distributed_lanczos_gpu_symmetry PRIVATE
                ed_distributed ed_symmetry)

            # Phase D step 3 (path matrix symm × mpi+gpu): on-device
            # thick-restart Lanczos consuming a `DistributedSymmetryOperatorGPU`.
            # Cross-checks the GPU symm KS ground-state eigenvalue
            # against the CPU `distributed_krylov_schur_symmetry` at
            # the same seed/max_iter.
            ed_add_phase3c_test(test_distributed_krylov_schur_gpu_symmetry
                                tests/unit/test_distributed_krylov_schur_gpu_symmetry.cpp)
            target_link_libraries(test_distributed_krylov_schur_gpu_symmetry PRIVATE
                ed_distributed ed_symmetry)

            # Phase D step 5 (path matrix symm × mpi+gpu): on-device FTLM
            # consuming a `DistributedSymmetryOperatorGPU`. Cross-checks
            # the GPU symm FTLM Z(beta) against the CPU
            # `distributed_ftlm_symmetry` at the same seed/max_iter on
            # an N=4 PBC Heisenberg chain (single sector).
            ed_add_phase3c_test(test_distributed_ftlm_gpu_symmetry
                                tests/unit/test_distributed_ftlm_gpu_symmetry.cpp)
            target_link_libraries(test_distributed_ftlm_gpu_symmetry PRIVATE
                ed_distributed ed_symmetry)

            # Phase E step 2 (path matrix symm × mpi+gpu for TPQ):
            # on-device canonical TPQ consuming a
            # `DistributedSymmetryOperatorGPU`. Cross-checks the GPU
            # symm TPQ energy(beta) against the CPU
            # `distributed_tpq_symmetry` at the same seed / delta_beta
            # / taylor_order on an N=4 PBC Heisenberg chain (every
            # momentum sector).
            ed_add_phase3c_test(test_distributed_tpq_gpu_symmetry
                                tests/unit/test_distributed_tpq_gpu_symmetry.cpp)
            target_link_libraries(test_distributed_tpq_gpu_symmetry PRIVATE
                ed_distributed ed_symmetry)
        endif()
    endif()

    # Phase 3b #7 stage 1: orbit-respecting partition primitive (no MPI;
    # the LPT greedy is a deterministic function of (orbit_weights,
    # n_ranks) and runs in serial).
    if(WITH_MPI AND TARGET ed_distributed)
        ed_add_test(test_orbit_partition
                    tests/unit/test_orbit_partition.cpp USE_CATCH2)
        target_link_libraries(test_orbit_partition PRIVATE ed_distributed)
    endif()
    ed_add_test(test_trans_interall_loading     tests/unit/test_trans_interall_loading.cpp USE_CATCH2)
    # ed_input library lockdown (lattice generators + HamiltonianBuilder +
    # write_directory roundtrip vs Operator::loadFromDirectory).
    ed_add_test(test_input_library              tests/unit/test_input_library.cpp USE_CATCH2)
    target_link_libraries(test_input_library PRIVATE ed_input)
    ed_add_test(test_observables                tests/unit/test_observables.cpp USE_CATCH2)
    # Retired in the minimalist-architecture rev (May 2026):
    #   * test_thermal_methods   (Hybrid thermal + GPU/MPI alias paths gone)
    #   * test_ftlm_jp           (FTLM double-Lanczos dynamical kernel gone)
    #   * test_ftlm_ltlm_dyn     (LTLM dynamical kernel gone)
    #   * test_ftlm_sssf         (FTLM SSSF static-structure-factor kernel gone)
    #   * test_ftlm_kpm          (KPM dynamical / FTLM-KPM kernel gone -- KPM_DOS
    #                              stays; covered by test_kpm_dos)
    # The dynamical-correlator coverage moves to Phase 6 primitives.
    ed_add_test(test_ltlm_static_connected_qh   tests/unit/test_ltlm_static_connected_qh.cpp USE_CATCH2)
    # (execution-planner test removed with the planner)
    # Streaming/tableless basis construction (Gosper enum + membership subspace)
    ed_add_test(test_streaming_enum             tests/unit/test_streaming_enum.cpp USE_CATCH2)
    # Closed-form orbit-projected norm primitive (shared by every sector builder)
    ed_add_test(test_rep_projection             tests/unit/test_rep_projection.cpp USE_CATCH2)
    # Stage-1 SymmetryEngine v2: compiled byte-LUT group action bit-identity
    ed_add_test(test_compiled_group             tests/unit/test_compiled_group.cpp USE_CATCH2)
    # Stage-2 SymmetryEngine v2: fused OrbitTable + prefiltered parallel sector build
    ed_add_test(test_orbit_table                tests/unit/test_orbit_table.cpp USE_CATCH2)
    # Stage-3 SymmetryEngine v2: content-addressed OrbitTable cache (registry + disk)
    ed_add_test(test_symmetry_cache             tests/unit/test_symmetry_cache.cpp USE_CATCH2)
    # Stage-5 SymmetryEngine v2: spin-flip commutation check (SectorTransporter gate)
    ed_add_test(test_spin_flip                  tests/unit/test_spin_flip.cpp USE_CATCH2)
    # Stage-5b SymmetryEngine v2: in-sector flip projection ((k,+/-) spectrum union)
    ed_add_test(test_flip_projection            tests/unit/test_flip_projection.cpp USE_CATCH2)
    # Stage-6 SymmetryEngine v2: time-reversal conjugate-sector pairing map
    ed_add_test(test_time_reversal              tests/unit/test_time_reversal.cpp USE_CATCH2)
    # Numerical irrep decomposition (non-abelian symmetry engine)
    ed_add_test(test_irreps                     tests/unit/test_irreps.cpp USE_CATCH2)
    # End-to-end non-abelian symmetry-adapted ED (C4v Heisenberg vs brute force)
    ed_add_test(test_symmetry_adapted           tests/unit/test_symmetry_adapted.cpp USE_CATCH2)
    # Phase D2 (KPM density-of-states + thermodynamics kernel)
    ed_add_test(test_kpm_dos                    tests/unit/test_kpm_dos.cpp USE_CATCH2)
    # Audit item #1 (full): cross-sector rectangular observable + DSSF kernel
    ed_add_test(test_cross_sector_dssf          tests/unit/test_cross_sector_dssf.cpp USE_CATCH2)
    target_link_libraries(test_cross_sector_dssf PRIVATE ed_dssf)
    # Phase E (TPQ dynamical correlator) -- retired in the
    # minimalist-architecture rev (May 2026). TPQ dynamical correlators
    # become workflow-level compositions of mtpq_kernel +
    # time_evolution_correlator (Phase 6 / Phase 7).
    ed_add_test(test_hdf5_io                    tests/unit/test_hdf5_io.cpp USE_CATCH2)
    ed_add_test(test_symmetry                   tests/unit/test_symmetry.cpp USE_CATCH2)
    # Phase R5 of the "Orthogonal symmetry composition: Subspace x
    # ProjectorChain" plan (May 2026): byte-equality + extensibility
    # pinning for the new (Subspace, ProjectorChain) decomposition.
    ed_add_test(test_projector_chain            tests/unit/test_projector_chain.cpp USE_CATCH2)
    ed_add_test(test_chain_extensibility        tests/unit/test_chain_extensibility.cpp USE_CATCH2)
    # Operator-collapse refactor (Jun 2026): owning ``SectorBasis`` that
    # completes the basis triplet (FullSpaceSubspace / FixedSzSubspace /
    # SectorBasis) and yields the POD SymmetryBasisPolicy. Pins build()
    # / adopt() lookup round-trips + norm parity against
    # compute_orbit_for_state for the Heisenberg Z_N ring (N=6).
    ed_add_test(test_sector_basis               tests/unit/test_sector_basis.cpp USE_CATCH2)
    # P2: the unified CpuMatVecBackend<SymmetryBasisPolicy> + make_cpu_symmetry_backend
    # factory reproduce the legacy StreamingSymmetryOperator::applySymmetrized matvec
    # byte-for-byte in every momentum sector (Heisenberg Z_N ring, N=6).
    ed_add_test(test_symmetry_backend           tests/unit/test_symmetry_backend.cpp USE_CATCH2)
    # P2 (fixed-Sz symmetry lane): the SAME generic backend reproduces the
    # legacy FixedSzStreamingSymmetryOperator::applySymmetrizedFixedSz matvec
    # byte-for-byte in every momentum sector (half-filled Heisenberg Z_N ring,
    # N=6, n_up=3) -- orbits built over a FixedSzSubspace instead of full space.
    ed_add_test(test_symmetry_backend_fixedsz   tests/unit/test_symmetry_backend_fixedsz.cpp USE_CATCH2)
    # "Optimized symmetry ED + NLCE" (Jun 2026), Phase 1 gate: the CPU
    # on-the-fly representative SpMV (make_cpu_rep_symmetry_backend +
    # CpuMatVecBackend<RepSymmetryBasisPolicy> + apply_terms_rep_symmetry)
    # reproduces the orbit-CSR reference matvec in every momentum sector
    # WITHOUT materialising the per-sector orbit CSR (rep data from the
    # CSR-free getRepSectorData). Heisenberg Z_N ring, N=6/8.
    ed_add_test(test_rep_symmetry_backend       tests/unit/test_rep_symmetry_backend.cpp USE_CATCH2)
    # P2c: the standalone per-sector ed::symmetry::SectorOperator (owning
    # SectorBasis + symmetry backend, no parent StreamingSymmetryOperator)
    # matches the legacy applySymmetrized matvec in every sector and its
    # dense k=0 spectrum yields the Bethe-ansatz GS (Heisenberg Z_N, N=6).
    ed_add_test(test_sector_operator            tests/unit/test_sector_operator.cpp USE_CATCH2)
    # P5: build_full_sector_operators / build_fixed_sz_sector_operators emit a flat
    # vector of standalone SectorOperators (one per surviving sector) whose dims tile
    # the Hilbert space (full) / fixed-Sz subspace and whose min ground state is the
    # Bethe-ansatz GS (Heisenberg Z_N, N=6) -- the SectorView fan-out replacement.
    ed_add_test(test_sector_set                 tests/unit/test_sector_set.cpp USE_CATCH2)
    # Non-abelian safety guard (restrict to maximal abelian subgroup)
    ed_add_test(test_nonabelian_guard           tests/unit/test_nonabelian_guard.cpp USE_CATCH2)
    # Operator-collapse refactor (Phase C, Jun 2026): the production cutover --
    # StreamingSymmetryHandle::sector(k) routes through the standalone
    # SectorOperator (collapse target); the legacy SectorView and the
    # ED_SYMMETRY_SECTOR_OPERATOR gate were removed. Verifies the handle
    # returns a SectorOperator and the min sector GS is the Bethe GS (N=6).
    ed_add_test(test_sector_loop_gate           tests/unit/test_sector_loop_gate.cpp USE_CATCH2)
    # Cutover-safety invariant: the legacy StreamingSymmetryOperator and the new
    # build_full_sector_operators path emit the SAME per-sector basis ORDERING
    # (orbit_rep + orbit_elements sequence), not just matching dims/eigenvalues
    # (which are basis-order-invariant). Guards the sector index -> basis index
    # mapping that keys HDF5 orbit caches + cross-sector observables (N=6 Z_N).
    ed_add_test(test_symmetry_ordering_parity   tests/unit/test_symmetry_ordering_parity.cpp USE_CATCH2)
    # P3a: CudaMatVecBackend<DeviceFullBasisPolicy> (GPU twin of CpuMatVecBackend)
    # matvec matches the host CpuMatVecBackend to <1e-9 across every term bin and
    # both scalar lanes (complex apply + real apply_real) on the full Hilbert
    # space. CUDA-only; plain main() harness (not Catch2). Needs a GPU at runtime
    # (skips cleanly with exit 0 when no device is present).
    if(WITH_CUDA)
        ed_add_test(test_cuda_matvec_backend    tests/unit/test_cuda_matvec_backend.cu)
        # P3c: CudaMatVecBackend<DeviceSymmetryBasisPolicy> (GPU symmetry lane)
        # -- DeviceSymmetryBasisPolicyHolder uploads the orbit CSR + pre-baked
        # projection hash; the GPU symmetry matvec matches the host
        # CpuMatVecBackend<Symmetry> to <1e-9 across every momentum sector of
        # the Heisenberg Z_N ring (N=6). CUDA-only; plain main() harness.
        ed_add_test(test_cuda_symmetry_matvec_backend tests/unit/test_cuda_symmetry_matvec_backend.cu)
        # Phase A operator-collapse GPU parity (Jun 2026): a standalone
        # ed::symmetry::SectorOperator (from make_sector_operators -> the P5
        # sector-set enumerator, no StreamingSymmetryOperator) is backend-
        # complete on the GPU -- its bind_cuda() matvec matches its own CPU
        # apply to <1e-9 across every full-Hilbert and fixed-Sz (n_up=3)
        # sector of the Heisenberg Z_N ring (N=6). CUDA-only; plain main().
        ed_add_test(test_sector_operator_gpu tests/unit/test_sector_operator_gpu.cu)
        # "On-the-fly representative SpMV for streaming symmetry" plan
        # (Jun 2026): with ED_GPU_SYMMETRY_REP=1 a fixed-Sz SectorOperator's
        # bind_cuda() takes the resident on-the-fly representative path
        # (apply_terms_rep_symmetry_scatter -- NO orbit CSR / no
        # O(full-Sz-dim) projection table). The rep matvec matches the CPU
        # applySymmetrized reference to <1e-10 across every momentum sector
        # of the Heisenberg Z_N ring at N=6 (n_up=3) and N=8 (n_up=4, |G|=8,
        # non-trivial complex characters). CUDA-only; plain main().
        ed_add_test(test_rep_symmetry_gpu tests/unit/test_rep_symmetry_gpu.cu)

        # CSR-free lazy sector loop ("scan other region", Jun 2026): drives
        # the fixed-Sz streaming operator THROUGH StreamingSymmetryHandle and
        # pins that (a) the handle hands out a rep-lazy SectorOperator whose
        # dim is known WITHOUT materialising the ~24 GiB/sector host orbit CSR,
        # (b) bind_cuda's GPU rep path never materialises it either, (c) the GPU
        # matvec matches the CPU apply reference to <1e-10, and (d) the CPU
        # apply fallback lazily materialises the CSR. CUDA-only; plain main().
        ed_add_test(test_rep_lazy_sector_loop tests/unit/test_rep_lazy_sector_loop.cu)
    endif()
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

    # P2.1 (6th slice): site-resolved spin structure factor kernel
    # (`compute_spin_structure_factor`) on synthesised two-body
    # correlation tables. Tests on a 2-site Neel state with hand-checked
    # Re/Im parts.
    ed_add_test(test_bfg_spin_structure_factor  tests/unit/test_bfg_spin_structure_factor.cpp USE_CATCH2)
    target_link_libraries(test_bfg_spin_structure_factor PRIVATE ed_bfg)

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
        # Phase 3a #3: FP32 cuSPARSE SpMV + FP64 dot/norm/axpy mixed-precision path.
        ed_add_test(test_gpu_mixed_precision_spmv tests/unit/test_gpu_mixed_precision_spmv.cpp USE_CATCH2)
        # Phase 2 of the matvec-unification (May 2026): CudaBackend BLAS-1
        # round-trips + agreement of `lanczos_kernel<Backend&>` between
        # CpuBackend and CudaBackend on a 6-site Heisenberg chain. First
        # production consumer of `ed/matvec/backends/cuda_backend.cuh`.
        ed_add_test(test_cuda_backend           tests/unit/test_cuda_backend.cpp USE_CATCH2)
        # Phase A.1 of the "Kill the GPU State-Lookup Hash" plan
        # (May 2026): pin the colex convention between
        # ``unrank_combination_dev`` (already in the codebase) and the
        # new ``rank_combination_dev`` that replaces the 8-32 GiB
        # ``d_state_hash_`` device table. Failure here means the
        # fixed-Sz matvec kernels would silently miss states once we
        # flip the dispatch in Phase A.2.
        ed_add_test(test_gpu_fixed_sz_rank      tests/unit/test_gpu_fixed_sz_rank.cpp USE_CATCH2)
    endif()

    # Phase 1 of the Minimalist ED Collapse (May 2026): Backend BLAS-3
    # surface (gemm / gemv / trsm / qr_thin) lockdown across CpuBackend
    # (+ CudaBackend when WITH_CUDA). MpiBackend / MpiCudaBackend qr_thin
    # CholeskyQR2 coverage lives in test_backend_blas3_mpi.cpp under the
    # WITH_MPI guard below (when added).
    ed_add_test(test_backend_blas3              tests/unit/test_backend_blas3.cpp USE_CATCH2)

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
