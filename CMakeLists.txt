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

# Detect host CPU vendor to pick optimized BLAS/LAPACK defaults
set(CPU_VENDOR "Unknown")
if(EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" CPUINFO_RAW)
    if(CPUINFO_RAW MATCHES "AuthenticAMD")
        set(CPU_VENDOR "AMD")
    elseif(CPUINFO_RAW MATCHES "GenuineIntel")
        set(CPU_VENDOR "Intel")
    endif()
endif()
message(STATUS "Detected CPU vendor: ${CPU_VENDOR}")

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

# Set languages based on options
if(WITH_CUDA)
    project(ExactDiagonalization LANGUAGES CXX CUDA)
    enable_language(CUDA)
else()
    project(ExactDiagonalization LANGUAGES CXX)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED OFF)
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

# Include directories - public headers in include/, private in src/
include_directories(${INCLUDE_DIR})
include_directories(${INCLUDE_DIR}/ed/core)
include_directories(${INCLUDE_DIR}/ed/solvers)
include_directories(${INCLUDE_DIR}/ed/io)
include_directories(${INCLUDE_DIR}/ed/gpu)
include_directories(${SRC_DIR})
include_directories(${CORE_DIR})
include_directories(${SOLVERS_CPU_DIR})
include_directories(${SOLVERS_GPU_DIR})
include_directories(${IO_DIR})

# =============================================================================
# BLAS/LAPACK LIBRARY CONFIGURATION
# =============================================================================
# Configure based on selected BLAS_PROFILE

set(BLAS_LIBRARIES "")
set(LAPACK_LIBRARIES "")
set(LAPACKE_LIBRARIES "")
set(EXTRA_LINALG_LIBRARIES "")
set(BLAS_VENDOR "Unknown")
set(SCALAPACK_COMPATIBLE ON)  # Whether ScaLAPACK can be safely used

# -----------------------------------------------------------------------------
# Profile: FLEXIBLAS - Runtime-switchable BLAS (ideal for HPC clusters)
# FlexiBLAS provides BLAS/LAPACK but LAPACKE needs separate library
# On clusters with AOCL modules, libflame provides LAPACKE
# -----------------------------------------------------------------------------
if(BLAS_PROFILE STREQUAL "FLEXIBLAS")
    message(STATUS "Configuring FlexiBLAS (runtime-switchable BLAS)")
    
    set(_FLEXIBLAS_HINT_DIRS)
    if(DEFINED ENV{EBROOTFLEXIBLAS})
        list(APPEND _FLEXIBLAS_HINT_DIRS $ENV{EBROOTFLEXIBLAS})
    endif()
    if(DEFINED ENV{FLEXIBLAS_ROOT})
        list(APPEND _FLEXIBLAS_HINT_DIRS $ENV{FLEXIBLAS_ROOT})
    endif()
    list(APPEND _FLEXIBLAS_HINT_DIRS /usr /usr/local)
    
    find_library(FLEXIBLAS_LIBRARY NAMES flexiblas
        HINTS ${_FLEXIBLAS_HINT_DIRS}
        PATH_SUFFIXES lib lib64
    )
    
    find_path(FLEXIBLAS_INCLUDE_DIR flexiblas/cblas.h cblas.h
        HINTS ${_FLEXIBLAS_HINT_DIRS}
        PATH_SUFFIXES include include/flexiblas
    )
    
    if(FLEXIBLAS_LIBRARY)
        set(BLAS_LIBRARIES ${FLEXIBLAS_LIBRARY})
        set(LAPACK_LIBRARIES ${FLEXIBLAS_LIBRARY})
        
        # FlexiBLAS doesn't directly provide LAPACKE - need separate library
        # Check for AOCL libflame (provides LAPACKE on clusters with AOCL modules)
        set(_LAPACKE_HINT_DIRS)
        if(DEFINED ENV{EBROOTAOCLMINLAPACK})
            list(APPEND _LAPACKE_HINT_DIRS $ENV{EBROOTAOCLMINLAPACK})
        endif()
        if(DEFINED ENV{AOCL_ROOT})
            list(APPEND _LAPACKE_HINT_DIRS $ENV{AOCL_ROOT})
        endif()
        list(APPEND _LAPACKE_HINT_DIRS ${_FLEXIBLAS_HINT_DIRS} /usr /usr/local)
        
        # Try to find AOCL libflame first (has LAPACKE)
        find_library(AOCL_FLAME_LIBRARY NAMES flame
            HINTS ${_LAPACKE_HINT_DIRS}
            PATH_SUFFIXES lib lib64
        )
        
        # Also try standalone lapacke
        find_library(LAPACKE_STANDALONE_LIBRARY NAMES lapacke
            HINTS ${_LAPACKE_HINT_DIRS}
            PATH_SUFFIXES lib lib64
        )
        
        if(AOCL_FLAME_LIBRARY)
            set(LAPACKE_LIBRARIES ${AOCL_FLAME_LIBRARY})
            message(STATUS "  LAPACKE from AOCL libflame: ${AOCL_FLAME_LIBRARY}")
            
            # Add AOCL LAPACK include for lapacke.h
            if(DEFINED ENV{EBROOTAOCLMINLAPACK})
                set(AOCL_LAPACK_INCLUDE $ENV{EBROOTAOCLMINLAPACK}/include)
                if(EXISTS ${AOCL_LAPACK_INCLUDE}/lapacke.h)
                    include_directories(SYSTEM ${AOCL_LAPACK_INCLUDE})
                    message(STATUS "  LAPACKE include: ${AOCL_LAPACK_INCLUDE}")
                endif()
            endif()
            
            # Set RPATH for libflame
            get_filename_component(FLAME_LIB_DIR ${AOCL_FLAME_LIBRARY} DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH ${FLAME_LIB_DIR})
            list(APPEND CMAKE_INSTALL_RPATH ${FLAME_LIB_DIR})
        elseif(LAPACKE_STANDALONE_LIBRARY)
            set(LAPACKE_LIBRARIES ${LAPACKE_STANDALONE_LIBRARY})
            message(STATUS "  LAPACKE library: ${LAPACKE_STANDALONE_LIBRARY}")
        else()
            # Fall back to FlexiBLAS (may not have all LAPACKE functions)
            set(LAPACKE_LIBRARIES ${FLEXIBLAS_LIBRARY})
            message(STATUS "  LAPACKE: using FlexiBLAS (limited LAPACKE support)")
        endif()
        
        if(FLEXIBLAS_INCLUDE_DIR)
            include_directories(SYSTEM ${FLEXIBLAS_INCLUDE_DIR})
        endif()
        add_definitions(-DUSE_FLEXIBLAS)
        set(BLAS_VENDOR "FlexiBLAS")
        message(STATUS "  FlexiBLAS library: ${FLEXIBLAS_LIBRARY}")
        message(STATUS "  Runtime backend selection: Set FLEXIBLAS=<backend> environment variable")
        message(STATUS "    Available backends: OpenBLAS, BLIS, MKL, ATLAS, NETLIB, AOCL")
    else()
        message(WARNING "FlexiBLAS not found, falling back to OPENBLAS profile")
        set(BLAS_PROFILE "OPENBLAS")
    endif()
endif()

# -----------------------------------------------------------------------------
# Profile: MKL - Intel Math Kernel Library (full stack)
# -----------------------------------------------------------------------------
if(BLAS_PROFILE STREQUAL "MKL")
    message(STATUS "Configuring Intel MKL")
    
    # Find MKL
    set(_MKL_HINT_DIRS
        /opt/intel/oneapi/mkl/latest
        /opt/intel/oneapi/mkl/2025.1
        /opt/intel/mkl
        $ENV{MKLROOT}
    )
    
    find_path(MKL_INCLUDE_DIR mkl.h
        HINTS ${_MKL_HINT_DIRS}
        PATH_SUFFIXES include
    )
    
    find_library(MKL_CORE NAMES mkl_core
        HINTS ${_MKL_HINT_DIRS}
        PATH_SUFFIXES lib lib/intel64
    )
    
    find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64
        HINTS ${_MKL_HINT_DIRS}
        PATH_SUFFIXES lib lib/intel64
    )
    
    find_library(MKL_INTEL_THREAD NAMES mkl_intel_thread
        HINTS ${_MKL_HINT_DIRS}
        PATH_SUFFIXES lib lib/intel64
    )
    
    find_library(IOMP5_LIBRARY NAMES iomp5
        HINTS ${_MKL_HINT_DIRS} /opt/intel/oneapi/compiler/latest/lib
        PATH_SUFFIXES lib lib/intel64 ../compiler/lib/intel64
    )
    
    if(MKL_INCLUDE_DIR AND MKL_CORE AND MKL_INTEL_LP64 AND MKL_INTEL_THREAD)
        include_directories(SYSTEM ${MKL_INCLUDE_DIR})
        set(BLAS_LIBRARIES ${MKL_INTEL_LP64} ${MKL_INTEL_THREAD} ${MKL_CORE})
        set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
        set(LAPACKE_LIBRARIES ${BLAS_LIBRARIES})
        if(IOMP5_LIBRARY)
            list(APPEND EXTRA_LINALG_LIBRARIES ${IOMP5_LIBRARY})
        endif()
        list(APPEND EXTRA_LINALG_LIBRARIES pthread m dl)
        add_definitions(-DWITH_MKL)
        set(BLAS_VENDOR "Intel MKL")
        set(WITH_MKL ON)
        message(STATUS "  MKL include: ${MKL_INCLUDE_DIR}")
        message(STATUS "  MKL libraries: ${BLAS_LIBRARIES}")
    else()
        message(WARNING "MKL not found, falling back to OPENBLAS profile")
        set(BLAS_PROFILE "OPENBLAS")
        set(WITH_MKL OFF)
    endif()
endif()

# -----------------------------------------------------------------------------
# Profile: AOCL - Full AMD AOCL stack (BLIS + libflame + AOCL-ScaLAPACK)
# Uses AOCC Fortran runtime for full compatibility with AOCL ScaLAPACK
# Requires: AOCC compiler installed (/opt/AMD/aocc-compiler-*)
# -----------------------------------------------------------------------------
if(BLAS_PROFILE STREQUAL "AOCL")
    message(STATUS "Configuring full AMD AOCL stack (BLIS + libflame + ScaLAPACK)")
    
    set(_AOCL_HINT_DIRS)
    if(DEFINED ENV{AOCL_ROOT})
        list(APPEND _AOCL_HINT_DIRS $ENV{AOCL_ROOT})
    endif()
    list(APPEND _AOCL_HINT_DIRS
        /opt/AMD/aocl/aocl-linux-aocc-5.0.0/aocc
        /opt/AMD/aocl/aocl-linux-x86_64
        /opt/AMD/aocl
        /opt/aocl
    )
    
    # Find AOCC compiler runtime (required for AOCL ScaLAPACK)
    set(_AOCC_HINT_DIRS
        /opt/AMD/aocc-compiler-5.0.0
        /opt/AMD/aocc-compiler-4.2.0
        /opt/AMD/aocc-compiler-4.1.0
        /opt/AMD/aocc-compiler-4.0.0
    )
    find_path(AOCC_LIB_DIR libflang.so
        HINTS ${_AOCC_HINT_DIRS}
        PATH_SUFFIXES lib lib64
    )
    
    find_path(AOCL_INCLUDE_DIR cblas.h
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES include include/blis
    )
    
    find_library(AOCL_BLIS_LIBRARY NAMES blis-mt blis
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64
    )
    
    find_library(AOCL_LAPACK_LIBRARY NAMES flame
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64
    )
    
    find_library(AOCL_SCALAPACK_LIBRARY NAMES scalapack
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64
    )
    
    if(AOCL_INCLUDE_DIR AND AOCL_BLIS_LIBRARY AND AOCL_LAPACK_LIBRARY AND AOCL_SCALAPACK_LIBRARY AND AOCC_LIB_DIR)
        include_directories(SYSTEM ${AOCL_INCLUDE_DIR})
        set(BLAS_LIBRARIES ${AOCL_BLIS_LIBRARY})
        set(LAPACK_LIBRARIES ${AOCL_LAPACK_LIBRARY})
        set(LAPACKE_LIBRARIES ${AOCL_LAPACK_LIBRARY})  # libflame includes LAPACKE
        set(SCALAPACK_LIBRARIES ${AOCL_SCALAPACK_LIBRARY})
        
        # Get AOCL lib directory
        get_filename_component(AOCL_LIB_DIR ${AOCL_BLIS_LIBRARY} DIRECTORY)
        
        # RPATH setup: AOCC runtime first (for libflang, libflangrti, libpgmath)
        # then AOCL libs, then libomp
        list(APPEND CMAKE_BUILD_RPATH ${AOCC_LIB_DIR})
        list(APPEND CMAKE_BUILD_RPATH ${AOCL_LIB_DIR})
        list(APPEND CMAKE_INSTALL_RPATH ${AOCC_LIB_DIR})
        list(APPEND CMAKE_INSTALL_RPATH ${AOCL_LIB_DIR})
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        
        # Use DT_RPATH for indirect dependency resolution
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
        
        # Find libomp for AOCL
        if(EXISTS "/usr/lib/llvm-18/lib/libomp.so")
            list(APPEND CMAKE_BUILD_RPATH /usr/lib/llvm-18/lib)
            list(APPEND CMAKE_INSTALL_RPATH /usr/lib/llvm-18/lib)
        endif()
        
        list(APPEND EXTRA_LINALG_LIBRARIES pthread m)
        
        add_definitions(-DUSE_AOCL_BLIS)
        add_definitions(-DWITH_SCALAPACK)
        set(USE_AOCL_BLIS ON)
        set(BLAS_VENDOR "AMD AOCL (full stack)")
        set(SCALAPACK_COMPATIBLE ON)
        set(WITH_SCALAPACK ON)
        
        message(STATUS "  AOCL include: ${AOCL_INCLUDE_DIR}")
        message(STATUS "  AOCL BLIS: ${AOCL_BLIS_LIBRARY}")
        message(STATUS "  AOCL libflame: ${AOCL_LAPACK_LIBRARY}")
        message(STATUS "  AOCL ScaLAPACK: ${AOCL_SCALAPACK_LIBRARY}")
        message(STATUS "  AOCC runtime: ${AOCC_LIB_DIR}")
        message(STATUS "  RPATH: ${AOCC_LIB_DIR}:${AOCL_LIB_DIR}")
    elseif(AOCL_INCLUDE_DIR AND AOCL_BLIS_LIBRARY AND AOCL_LAPACK_LIBRARY AND NOT AOCC_LIB_DIR)
        message(WARNING "AOCC compiler not found - AOCL ScaLAPACK requires AOCC runtime")
        message(WARNING "Install AOCC: https://www.amd.com/en/developer/aocc.html")
        message(WARNING "Falling back to AOCL without ScaLAPACK")
        
        include_directories(SYSTEM ${AOCL_INCLUDE_DIR})
        set(BLAS_LIBRARIES ${AOCL_BLIS_LIBRARY})
        set(LAPACK_LIBRARIES ${AOCL_LAPACK_LIBRARY})
        set(LAPACKE_LIBRARIES ${AOCL_LAPACK_LIBRARY})
        
        get_filename_component(AOCL_LIB_DIR ${AOCL_BLIS_LIBRARY} DIRECTORY)
        list(APPEND CMAKE_BUILD_RPATH ${AOCL_LIB_DIR})
        list(APPEND CMAKE_INSTALL_RPATH ${AOCL_LIB_DIR})
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
        
        if(EXISTS "/usr/lib/llvm-18/lib/libomp.so")
            list(APPEND CMAKE_BUILD_RPATH /usr/lib/llvm-18/lib)
            list(APPEND CMAKE_INSTALL_RPATH /usr/lib/llvm-18/lib)
        endif()
        
        list(APPEND EXTRA_LINALG_LIBRARIES pthread m)
        add_definitions(-DUSE_AOCL_BLIS)
        set(USE_AOCL_BLIS ON)
        set(BLAS_VENDOR "AMD AOCL (no ScaLAPACK)")
        set(SCALAPACK_COMPATIBLE OFF)
    else()
        message(WARNING "AOCL not found, falling back to OPENBLAS profile")
        set(BLAS_PROFILE "OPENBLAS")
        set(USE_AOCL_BLIS OFF)
    endif()
endif()

# -----------------------------------------------------------------------------
# Profile: AOCL_BLIS - AMD BLIS for BLAS + Reference LAPACK
# This profile is ScaLAPACK-compatible (all gfortran ABI)
# Use with custom ScaLAPACK built against BLIS + reference LAPACK
# -----------------------------------------------------------------------------
if(BLAS_PROFILE STREQUAL "AOCL_BLIS")
    message(STATUS "Configuring AMD BLIS + Reference LAPACK (ScaLAPACK-compatible)")
    
    set(_AOCL_HINT_DIRS)
    if(DEFINED ENV{AOCL_ROOT})
        list(APPEND _AOCL_HINT_DIRS $ENV{AOCL_ROOT})
    endif()
    list(APPEND _AOCL_HINT_DIRS
        /opt/AMD/aocl/aocl-linux-aocc-5.0.0/aocc
        /opt/AMD/aocl/aocl-linux-x86_64
        /opt/AMD/aocl
        /opt/aocl
    )
    
    find_path(AOCL_INCLUDE_DIR cblas.h
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES include include/blis
    )
    
    find_library(AOCL_BLIS_LIBRARY NAMES blis-mt blis
        HINTS ${_AOCL_HINT_DIRS}
        PATH_SUFFIXES lib lib64
    )
    
    # Find reference LAPACK (gfortran-compiled, no Fortran runtime conflicts)
    find_library(REFLAPACK_LIBRARY NAMES lapack
        PATHS 
            /usr/lib/x86_64-linux-gnu/lapack
            /usr/lib/lapack
        NO_DEFAULT_PATH
    )

    # Default LAPACKE_ROOT to local reference build if present
    if(NOT LAPACKE_ROOT AND EXISTS "/home/pc_linux/exact_diagonalization_clean/lapacke_ref/install")
        set(LAPACKE_ROOT "/home/pc_linux/exact_diagonalization_clean/lapacke_ref/install")
    endif()
    
    if(AOCL_INCLUDE_DIR AND AOCL_BLIS_LIBRARY AND REFLAPACK_LIBRARY)
        include_directories(SYSTEM ${AOCL_INCLUDE_DIR})
        set(BLAS_LIBRARIES ${AOCL_BLIS_LIBRARY})
        set(LAPACK_LIBRARIES ${REFLAPACK_LIBRARY})
        
        # Reference LAPACK may depend on BLAS - ensure linking order
        list(APPEND LAPACK_LIBRARIES ${AOCL_BLIS_LIBRARY})
        
        # Set RPATH to ensure BLIS is found at runtime
        get_filename_component(AOCL_LIB_DIR ${AOCL_BLIS_LIBRARY} DIRECTORY)
        list(APPEND CMAKE_BUILD_RPATH ${AOCL_LIB_DIR})
        list(APPEND CMAKE_INSTALL_RPATH ${AOCL_LIB_DIR})
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

        # BLAS shim to override system libblas (avoids MKL alternatives via liblapacke)
        set(BLAS_SHIM_DIR "/home/pc_linux/exact_diagonalization_clean/blas_shim")
        if(EXISTS "${BLAS_SHIM_DIR}/libblas.so.3")
            list(PREPEND CMAKE_BUILD_RPATH ${BLAS_SHIM_DIR})
            list(PREPEND CMAKE_INSTALL_RPATH ${BLAS_SHIM_DIR})
            message(STATUS "  Added BLAS shim RPATH (prepend): ${BLAS_SHIM_DIR}")
        endif()

        # AOCL BLIS depends on libomp
        if(EXISTS "/usr/lib/llvm-18/lib/libomp.so")
            list(APPEND CMAKE_BUILD_RPATH /usr/lib/llvm-18/lib)
            list(APPEND CMAKE_INSTALL_RPATH /usr/lib/llvm-18/lib)
            message(STATUS "  Added libomp RPATH: /usr/lib/llvm-18/lib")
        endif()

        # Use DT_RPATH instead of DT_RUNPATH so indirect deps (liblapacke -> libblas)
        # honor the shim path without requiring LD_LIBRARY_PATH
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
        
        # LAPACKE from custom root if provided (avoids system BLAS/LAPACK alternatives)
        if(LAPACKE_ROOT)
            find_library(LAPACKE_LIBRARY NAMES lapacke
                HINTS ${LAPACKE_ROOT}/lib ${LAPACKE_ROOT}/lib64
                NO_DEFAULT_PATH
            )
            if(LAPACKE_LIBRARY)
                set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
                get_filename_component(LAPACKE_LIB_DIR ${LAPACKE_LIBRARY} DIRECTORY)
                list(PREPEND CMAKE_BUILD_RPATH ${LAPACKE_LIB_DIR})
                list(PREPEND CMAKE_INSTALL_RPATH ${LAPACKE_LIB_DIR})
                set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
                message(STATUS "  LAPACKE from: ${LAPACKE_LIBRARY}")
                message(STATUS "  Added RPATH (prepend): ${LAPACKE_LIB_DIR}")
            else()
                message(WARNING "LAPACKE_ROOT is set but liblapacke not found; falling back to system LAPACKE")
            endif()
        endif()
        if(NOT LAPACKE_LIBRARIES)
            find_library(LAPACKE_LIBRARY NAMES lapacke)
            if(LAPACKE_LIBRARY)
                set(LAPACKE_LIBRARIES ${LAPACKE_LIBRARY})
            else()
                set(LAPACKE_LIBRARIES ${REFLAPACK_LIBRARY})
            endif()
        endif()
        
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            list(APPEND EXTRA_LINALG_LIBRARIES OpenMP::OpenMP_CXX)
        endif()
        list(APPEND EXTRA_LINALG_LIBRARIES pthread m gfortran)
        
        add_definitions(-DUSE_AOCL_BLIS)
        set(USE_AOCL_BLIS ON)
        set(BLAS_VENDOR "AMD BLIS + Reference LAPACK")
        set(SCALAPACK_COMPATIBLE ON)  # Compatible with gfortran-compiled ScaLAPACK
        
        message(STATUS "  AOCL BLIS: ${AOCL_BLIS_LIBRARY}")
        message(STATUS "  Reference LAPACK: ${REFLAPACK_LIBRARY}")
        message(STATUS "  Added RPATH: ${AOCL_LIB_DIR}")
        message(STATUS "  ScaLAPACK: Compatible (use -DSCALAPACK_ROOT=... for custom build)")
    else()
        if(NOT AOCL_BLIS_LIBRARY)
            message(WARNING "AOCL BLIS not found")
        endif()
        if(NOT REFLAPACK_LIBRARY)
            message(WARNING "Reference LAPACK not found at /usr/lib/x86_64-linux-gnu/lapack")
        endif()
        message(WARNING "Falling back to OPENBLAS profile")
        set(BLAS_PROFILE "OPENBLAS")
        set(USE_AOCL_BLIS OFF)
    endif()
endif()

# -----------------------------------------------------------------------------
# Profile: OPENBLAS - OpenBLAS (good default for most systems)
# -----------------------------------------------------------------------------
if(BLAS_PROFILE STREQUAL "OPENBLAS")
    message(STATUS "Configuring OpenBLAS")
    
    # Try to find OpenBLAS specifically to avoid MKL masquerading as system BLAS
    find_library(OPENBLAS_LIBRARY NAMES openblas openblasp
        PATHS 
            /usr/lib/x86_64-linux-gnu/openblas-pthread
            /usr/lib/x86_64-linux-gnu/openblas-openmp
            /usr/lib/x86_64-linux-gnu/openblas-serial
            /usr/lib/openblas
        NO_DEFAULT_PATH
    )
    
    if(NOT OPENBLAS_LIBRARY)
        # Fall back to CMake's FindBLAS but check it's not MKL
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            # Check if this is actually MKL in disguise
            string(FIND "${BLAS_LIBRARIES}" "mkl" _MKL_CHECK)
            if(_MKL_CHECK GREATER -1)
                message(STATUS "  System BLAS is MKL (via alternatives), looking for explicit OpenBLAS...")
                find_library(OPENBLAS_LIBRARY NAMES openblas)
            else()
                set(OPENBLAS_LIBRARY ${BLAS_LIBRARIES})
            endif()
        endif()
    endif()
    
    if(OPENBLAS_LIBRARY)
        set(BLAS_LIBRARIES ${OPENBLAS_LIBRARY})
        # OpenBLAS includes LAPACK
        set(LAPACK_LIBRARIES ${OPENBLAS_LIBRARY})
        
        # Find LAPACKE (might be separate or integrated)
        find_library(LAPACKE_LIB NAMES lapacke
            PATHS /usr/lib/x86_64-linux-gnu
        )
        if(LAPACKE_LIB)
            set(LAPACKE_LIBRARIES ${LAPACKE_LIB})
        else()
            set(LAPACKE_LIBRARIES ${OPENBLAS_LIBRARY})
        endif()
        
        set(BLAS_VENDOR "OpenBLAS")
        message(STATUS "  OpenBLAS library: ${OPENBLAS_LIBRARY}")
    else()
        message(STATUS "  OpenBLAS not found, falling back to GENERIC")
        set(BLAS_PROFILE "GENERIC")
    endif()
endif()

# -----------------------------------------------------------------------------
# Profile: GENERIC - System BLAS/LAPACK (reference or whatever is default)
# -----------------------------------------------------------------------------
if(BLAS_PROFILE STREQUAL "GENERIC")
    message(STATUS "Configuring generic system BLAS/LAPACK")
    
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES})
    set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES})
    
    # Try to find LAPACKE
    find_library(LAPACKE_LIB NAMES lapacke)
    if(LAPACKE_LIB)
        set(LAPACKE_LIBRARIES ${LAPACKE_LIB})
    else()
        set(LAPACKE_LIBRARIES ${LAPACK_LIBRARIES})
    endif()
    
    set(BLAS_VENDOR "Generic")
    message(STATUS "  BLAS: ${BLAS_LIBRARIES}")
    message(STATUS "  LAPACK: ${LAPACK_LIBRARIES}")
    
    # Check if system LAPACK is actually MKL (affects ScaLAPACK compatibility)
    string(FIND "${LAPACK_LIBRARIES}" "mkl" _MKL_CHECK)
    if(_MKL_CHECK GREATER -1)
        message(STATUS "  Note: System LAPACK points to MKL")
        message(STATUS "        This may cause issues with system ScaLAPACK")
        message(STATUS "        Consider: sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu")
    endif()
endif()

message(STATUS "BLAS/LAPACK vendor: ${BLAS_VENDOR}")
message(STATUS "BLAS library: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK library: ${LAPACK_LIBRARIES}")

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

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

# MPI setup
if(WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)
    
    # Check MPI implementation for ScaLAPACK compatibility
    if(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Intel" OR MPI_C_COMPILER MATCHES "mpiicpc")
        set(MPI_IS_INTEL TRUE)
        message(STATUS "Detected Intel MPI")
    elseif(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI" OR MPI_C_COMPILER MATCHES "mpicc")
        set(MPI_IS_OPENMPI TRUE)
        message(STATUS "Detected OpenMPI")
    elseif(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "MPICH")
        set(MPI_IS_MPICH TRUE)
        message(STATUS "Detected MPICH")
    endif()
    
    # ScaLAPACK for distributed diagonalization
    option(WITH_SCALAPACK "Build with ScaLAPACK support for distributed diagonalization" ON)
    if(WITH_SCALAPACK)
        # Custom ScaLAPACK override (e.g., AOCL-linked ScaLAPACK)
        if(SCALAPACK_ROOT)
            find_library(SCALAPACK_LIBRARY NAMES scalapack scalapack-openmpi
                HINTS ${SCALAPACK_ROOT}/lib ${SCALAPACK_ROOT}/lib64
                NO_DEFAULT_PATH  # Don't fall back to system ScaLAPACK
            )
            find_library(BLACS_LIBRARY NAMES blacs blacs-openmpi blacsF77init-openmpi blacsCinit-openmpi
                HINTS ${SCALAPACK_ROOT}/lib ${SCALAPACK_ROOT}/lib64
                NO_DEFAULT_PATH
            )
            if(SCALAPACK_LIBRARY)
                set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
                if(BLACS_LIBRARY)
                    list(APPEND SCALAPACK_LIBRARIES ${BLACS_LIBRARY})
                endif()
                set(SCALAPACK_COMPATIBLE ON)
                message(STATUS "Using custom ScaLAPACK from SCALAPACK_ROOT: ${SCALAPACK_LIBRARY}")
                
                # Set RPATH to ensure custom ScaLAPACK is found at runtime
                # PREPEND to ensure it's searched BEFORE BLAS directory (which may have incompatible ScaLAPACK)
                get_filename_component(SCALAPACK_LIB_DIR ${SCALAPACK_LIBRARY} DIRECTORY)
                list(PREPEND CMAKE_BUILD_RPATH ${SCALAPACK_LIB_DIR})
                list(PREPEND CMAKE_INSTALL_RPATH ${SCALAPACK_LIB_DIR})
                set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
                message(STATUS "  Added RPATH (prepend): ${SCALAPACK_LIB_DIR}")
            else()
                message(WARNING "SCALAPACK_ROOT is set but no ScaLAPACK library was found in ${SCALAPACK_ROOT}")
            endif()
        endif()

        # Check profile-based ScaLAPACK compatibility
        if(NOT SCALAPACK_COMPATIBLE AND NOT SCALAPACK_LIBRARIES)
            message(WARNING "ScaLAPACK disabled: ${BLAS_PROFILE} profile indicates incompatible configuration")
            message(WARNING "Common causes:")
            message(WARNING "  - AOCL with system ScaLAPACK linked to MKL")
            message(WARNING "  - MKL ScaLAPACK with OpenMPI (ABI mismatch)")
            message(WARNING "Solutions: Use OPENBLAS profile, or Intel MPI with MKL profile")
            set(WITH_SCALAPACK OFF)
        endif()
    endif()
    
    if(WITH_SCALAPACK)
        if(NOT SCALAPACK_LIBRARIES)
            set(SCALAPACK_LIBRARIES "")
        endif()
        
        # Profile-specific ScaLAPACK detection
        if(NOT SCALAPACK_LIBRARIES AND BLAS_PROFILE STREQUAL "MKL")
            # MKL ScaLAPACK - prefer Intel MPI BLACS, warn about OpenMPI
            find_library(MKL_SCALAPACK_LIBRARY NAMES mkl_scalapack_lp64
                HINTS $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib
                /opt/intel/oneapi/mkl/latest/lib/intel64
                /opt/intel/oneapi/mkl/latest/lib
            )
            
            # Select BLACS based on MPI implementation
            if(MPI_IS_INTEL)
                find_library(MKL_BLACS_LIBRARY NAMES mkl_blacs_intelmpi_lp64
                    HINTS $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib
                    /opt/intel/oneapi/mkl/latest/lib/intel64
                )
                if(MKL_SCALAPACK_LIBRARY AND MKL_BLACS_LIBRARY)
                    set(SCALAPACK_LIBRARIES ${MKL_SCALAPACK_LIBRARY} ${MKL_BLACS_LIBRARY})
                    message(STATUS "MKL ScaLAPACK with Intel MPI BLACS: ${MKL_SCALAPACK_LIBRARY}")
                endif()
            elseif(MPI_IS_OPENMPI)
                # OpenMPI with MKL ScaLAPACK has known ABI issues
                message(WARNING "MKL ScaLAPACK with OpenMPI has known compatibility issues.")
                message(WARNING "Runtime crashes in BLACS grid setup are common.")
                message(WARNING "Recommendation: Use Intel MPI, or OPENBLAS profile with system ScaLAPACK.")
                # Still try to find it, but warn user
                find_library(MKL_BLACS_LIBRARY NAMES mkl_blacs_openmpi_lp64
                    HINTS $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib
                    /opt/intel/oneapi/mkl/latest/lib/intel64
                )
                if(MKL_SCALAPACK_LIBRARY AND MKL_BLACS_LIBRARY)
                    set(SCALAPACK_LIBRARIES ${MKL_SCALAPACK_LIBRARY} ${MKL_BLACS_LIBRARY})
                    message(STATUS "MKL ScaLAPACK (OpenMPI - may be unstable): ${MKL_SCALAPACK_LIBRARY}")
                endif()
            endif()
        elseif(NOT SCALAPACK_LIBRARIES)
            # Generic/OpenBLAS/FlexiBLAS - use system ScaLAPACK
            find_library(SCALAPACK_LIBRARY NAMES scalapack scalapack-openmpi)
            find_library(BLACS_LIBRARY NAMES blacs blacs-openmpi blacsF77init-openmpi blacsCinit-openmpi)
            
            if(SCALAPACK_LIBRARY)
                set(SCALAPACK_LIBRARIES ${SCALAPACK_LIBRARY})
                if(BLACS_LIBRARY)
                    list(APPEND SCALAPACK_LIBRARIES ${BLACS_LIBRARY})
                endif()
                message(STATUS "Using system ScaLAPACK: ${SCALAPACK_LIBRARY}")
            endif()
        endif()
        
        if(NOT SCALAPACK_LIBRARIES)
            message(STATUS "ScaLAPACK not found - distributed diagonalization will not be available")
            set(WITH_SCALAPACK OFF)
        else()
            add_definitions(-DWITH_SCALAPACK)
        endif()
    endif()
endif()

# New elegant main sources - organized layout
set(SOURCES
    ${APPS_DIR}/ed_main.cpp
    ${CORE_DIR}/ed_config.cpp
    ${SOLVERS_CPU_DIR}/arpack.cpp
    ${SOLVERS_CPU_DIR}/observables.cpp
    ${SOLVERS_CPU_DIR}/lanczos.cpp
    ${SOLVERS_CPU_DIR}/CG.cpp
    ${SOLVERS_CPU_DIR}/dynamics.cpp
    ${SOLVERS_CPU_DIR}/TPQ.cpp
    ${SOLVERS_CPU_DIR}/ftlm.cpp
    ${SOLVERS_CPU_DIR}/ltlm.cpp
    ${SOLVERS_CPU_DIR}/hybrid_thermal.cpp
    ${IO_DIR}/basis_vector_storage.cpp
)

# Add ScaLAPACK sources if available
if(WITH_SCALAPACK AND SCALAPACK_LIBRARIES)
    list(APPEND SOURCES ${SOLVERS_CPU_DIR}/scalapack_diag.cpp)
endif()

# Header-only sources that should be included in compilation
set(HEADER_SOURCES
    ${SRC_DIR}/*.h
)

# GPU sources for CUDA-enabled builds - organized layout
if(WITH_CUDA)
    set(GPU_SOURCES
        ${SOLVERS_GPU_DIR}/gpu_operator.cu
        ${SOLVERS_GPU_DIR}/gpu_operator_conversion.cpp
        ${SOLVERS_GPU_DIR}/gpu_kernels.cu
        ${SOLVERS_GPU_DIR}/gpu_fixed_sz_operator.cu
        ${SOLVERS_GPU_DIR}/gpu_lanczos.cu
        ${SOLVERS_GPU_DIR}/gpu_block_lanczos.cu
        ${SOLVERS_GPU_DIR}/gpu_krylov_schur.cu
        ${SOLVERS_GPU_DIR}/gpu_block_krylov_schur.cu
        ${SOLVERS_GPU_DIR}/gpu_ed_wrapper.cu
        ${SOLVERS_GPU_DIR}/gpu_tpq.cu
        ${SOLVERS_GPU_DIR}/gpu_cg.cu
        ${SOLVERS_GPU_DIR}/gpu_dynamics.cu
        ${SOLVERS_GPU_DIR}/gpu_ftlm.cu
    )
    
    set(GPU_HEADERS
        ${INCLUDE_DIR}/ed/gpu/kernel_config.h
        ${INCLUDE_DIR}/ed/gpu/bit_operations.cuh
        ${INCLUDE_DIR}/ed/gpu/gpu_operator.cuh
        ${INCLUDE_DIR}/ed/gpu/gpu_lanczos.cuh
        ${INCLUDE_DIR}/ed/gpu/gpu_ed_wrapper.h
        ${INCLUDE_DIR}/ed/gpu/gpu_tpq.cuh
        ${INCLUDE_DIR}/ed/gpu/gpu_cg.cuh
        ${INCLUDE_DIR}/ed/gpu/gpu_dynamics.cuh
        ${INCLUDE_DIR}/ed/gpu/gpu_ftlm.cuh
    )
endif()

# Define executables

# New elegant main (add GPU sources if CUDA is enabled)
if(WITH_CUDA)
    add_executable(ED ${SOURCES} ${GPU_SOURCES})
    target_include_directories(ED PRIVATE ${SOLVERS_GPU_DIR})
else()
    add_executable(ED ${SOURCES})
endif()

# Add TPQ_DSSF executable (only if MPI is enabled since it uses MPI)
# TPQ_DSSF uses parallel HDF5 for MPI-safe file access
if(WITH_MPI)
    # Find parallel HDF5 for TPQ_DSSF (OpenMPI version)
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

    add_executable(TPQ_DSSF ${APPS_DIR}/TPQ_DSSF.cpp ${SOLVERS_CPU_DIR}/dynamics.cpp ${SOLVERS_CPU_DIR}/TPQ.cpp ${SOLVERS_CPU_DIR}/ftlm.cpp ${SOLVERS_CPU_DIR}/lanczos.cpp ${IO_DIR}/basis_vector_storage.cpp)
endif()

# BFG Order Parameter computation (lightweight, HDF5 + OpenMP only)
add_executable(compute_bfg_order_parameters ${APPS_DIR}/compute_bfg_order_parameters.cpp)
target_link_libraries(compute_bfg_order_parameters ${HDF5_LIBRARIES})
target_include_directories(compute_bfg_order_parameters PRIVATE ${HDF5_INCLUDE_DIRS})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(compute_bfg_order_parameters OpenMP::OpenMP_CXX)
    target_compile_definitions(compute_bfg_order_parameters PRIVATE _OPENMP)
endif()
target_compile_options(compute_bfg_order_parameters PRIVATE -O3 -march=native -fopenmp)

# BFG Order Parameter GPU version (CUDA accelerated)
if(WITH_CUDA)
    add_executable(compute_bfg_order_parameters_gpu ${APPS_DIR}/compute_bfg_order_parameters_gpu.cu)
    # Link HDF5 libraries - use the same libraries as the CPU version
    target_link_libraries(compute_bfg_order_parameters_gpu 
        CUDA::cudart
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


# Link libraries for new main
# ScaLAPACK must come FIRST so its RPATH is searched before BLAS's directory
# (which may contain an incompatible ScaLAPACK)
if(WITH_SCALAPACK AND SCALAPACK_LIBRARIES)
    target_link_libraries(ED ${SCALAPACK_LIBRARIES})
endif()

target_link_libraries(ED
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${EXTRA_LINALG_LIBRARIES}
    ${ARPACK_LIBRARY}
    ${HDF5_LIBRARIES}
)

# Add CUDA libraries to ED if CUDA is enabled
if(WITH_CUDA)
    target_link_libraries(ED
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
        CUDA::cusolver
    )
    
endif()

# Link libraries for TPQ_DSSF executable (only if it exists)
if(TARGET TPQ_DSSF)
    # Use parallel HDF5 if available, otherwise serial HDF5
    if(HAS_PARALLEL_HDF5)
        target_include_directories(TPQ_DSSF PRIVATE ${HDF5_PARALLEL_INCLUDE_DIR})
        target_compile_definitions(TPQ_DSSF PRIVATE HDF5_PARALLEL)
        target_link_libraries(TPQ_DSSF
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
            ${LAPACKE_LIBRARIES}
            ${EXTRA_LINALG_LIBRARIES}
            ${ARPACK_LIBRARY}
            ${HDF5_PARALLEL_LIBRARIES}
            z sz crypto curl dl m
        )
    else()
        target_link_libraries(TPQ_DSSF
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
            ${LAPACKE_LIBRARIES}
            ${EXTRA_LINALG_LIBRARIES}
            ${ARPACK_LIBRARY}
            ${HDF5_LIBRARIES}
        )
    endif()
    
    # Add CUDA support for TPQ_DSSF if enabled
    if(WITH_CUDA)
        # Add GPU source files to TPQ_DSSF
        target_sources(TPQ_DSSF PRIVATE
            ${SOLVERS_GPU_DIR}/gpu_ed_wrapper.cu
            ${SOLVERS_GPU_DIR}/gpu_operator.cu
            ${SOLVERS_GPU_DIR}/gpu_operator_conversion.cpp
            ${SOLVERS_GPU_DIR}/gpu_kernels.cu
            ${SOLVERS_GPU_DIR}/gpu_fixed_sz_operator.cu
            ${SOLVERS_GPU_DIR}/gpu_lanczos.cu
            ${SOLVERS_GPU_DIR}/gpu_block_lanczos.cu
            ${SOLVERS_GPU_DIR}/gpu_krylov_schur.cu
            ${SOLVERS_GPU_DIR}/gpu_block_krylov_schur.cu
            ${SOLVERS_GPU_DIR}/gpu_tpq.cu
            ${SOLVERS_GPU_DIR}/gpu_cg.cu
            ${SOLVERS_GPU_DIR}/gpu_ftlm.cu
        )
        
        target_link_libraries(TPQ_DSSF
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
            CUDA::cusolver
        )
    endif()
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


if(WITH_MKL)
    if(USE_ONEMKL AND MKL_FOUND)
        # Link oneMKL using its modern CMake targets
        target_link_libraries(ED MKL::MKL)
        
        if(TARGET TPQ_DSSF)
            target_link_libraries(TPQ_DSSF MKL::MKL)
        endif()
        
        # Link MKL to CUDA example if it exists
        if(TARGET CUDA_EXAMPLE)
            target_link_libraries(CUDA_EXAMPLE MKL::MKL)
        endif()
        
        # Link MKL to CUDA test if it exists
        if(TARGET CUDA_TEST)
            target_link_libraries(CUDA_TEST MKL::MKL)
        endif()
        
    else()
        # Link traditional MKL (already linked via BLAS_LIBRARIES)
        # No additional linking needed since MKL is detected as BLAS/LAPACK
    endif()
endif()

# Add CUDA libraries if enabled
if(WITH_CUDA)
    target_link_libraries(ED
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
        CUDA::cusolver
    )
    
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
            CUDA::curand
            CUDA::cusolver
        )
    endif()
    
endif()

# Add MPI if enabled
if(WITH_MPI)
    target_link_libraries(ED ${MPI_CXX_LIBRARIES})
    
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF ${MPI_CXX_LIBRARIES})
    endif()
endif()

# Install targets
install(TARGETS ED DESTINATION bin)

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

# CPU information
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    execute_process(COMMAND grep -c "^processor" /proc/cpuinfo OUTPUT_VARIABLE CPU_COUNT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND grep "model name" /proc/cpuinfo COMMAND head -1 COMMAND sed "s/.*: //" OUTPUT_VARIABLE CPU_MODEL OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "  CPU: ${CPU_MODEL} (${CPU_COUNT} cores)")
    
    # Check for specific CPU features
    execute_process(COMMAND grep -q "avx2" /proc/cpuinfo RESULT_VARIABLE AVX2_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(AVX2_RESULT EQUAL 0)
        message(STATUS "  AVX2: Available")
    endif()
    
    execute_process(COMMAND grep -q "fma" /proc/cpuinfo RESULT_VARIABLE FMA_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(FMA_RESULT EQUAL 0)
        message(STATUS "  FMA: Available")
    endif()
endif()

# CPU optimization flags
set(CPU_OPT_FLAGS "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND CPU_OPT_FLAGS
        -O3                          # Maximum optimization
        -march=native                # Optimize for the build machine's CPU
        -mtune=native                # Tune for the build machine's CPU
        -ffast-math                  # Aggressive floating-point optimizations
        -funroll-loops               # Unroll loops for better performance
        -ftree-vectorize             # Enable auto-vectorization
        -fomit-frame-pointer         # Remove frame pointer for extra register
    )
    
    # Additional flags for GCC
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        list(APPEND CPU_OPT_FLAGS
            -fprefetch-loop-arrays   # Enable prefetching
            -fipa-pta                # Inter-procedural pointer analysis
        )
    endif()
    
    # Additional flags for Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND CPU_OPT_FLAGS
            -fvectorize              # Explicit vectorization flag
        )
    endif()
    
    message(STATUS "  CPU Optimization Flags: ${CPU_OPT_FLAGS}")
endif()

# Apply CPU optimization flags to all targets
target_compile_options(ED PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)

if(TARGET TPQ_DSSF)
    target_compile_options(TPQ_DSSF PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
    )
endif()

# =============================================================================
# DMRG Test executable
# =============================================================================
add_executable(test_dmrg_vs_ed tests/test_dmrg.cpp)
target_include_directories(test_dmrg_vs_ed PRIVATE ${INCLUDE_DIR})
target_link_libraries(test_dmrg_vs_ed
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    ${EXTRA_LINALG_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_dmrg_vs_ed OpenMP::OpenMP_CXX)
endif()
if(WITH_MPI)
    target_link_libraries(test_dmrg_vs_ed ${MPI_CXX_LIBRARIES})
endif()
target_compile_options(test_dmrg_vs_ed PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CPU_OPT_FLAGS}>
)

# CUDA-specific compile options
if(WITH_CUDA)

    set_property(TARGET ED PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(ED PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    
    if(TARGET TPQ_DSSF)
        set_property(TARGET TPQ_DSSF PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        target_compile_options(TPQ_DSSF PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        )
    endif()
    
endif()

# Add OpenMP support if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ED OpenMP::OpenMP_CXX)
    if(TARGET TPQ_DSSF)
        target_link_libraries(TPQ_DSSF OpenMP::OpenMP_CXX)
    endif()
    message(STATUS "  OpenMP Support: YES")
else()
    message(STATUS "  OpenMP Support: NO")
endif()
