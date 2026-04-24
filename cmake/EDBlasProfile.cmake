# =============================================================================
# cmake/EDBlasProfile.cmake
#
# BLAS / LAPACK / LAPACKE backend selection (the "profile" system).
#
# Profiles supported by BLAS_PROFILE (set by the user, validated/auto-resolved
# in the parent CMakeLists.txt):
#   FLEXIBLAS  - runtime-switchable BLAS via libflexiblas (HPC clusters)
#   MKL        - Intel oneMKL full stack (LP64)
#   AOCL       - full AMD AOCL stack (BLIS + libflame + AOCL ScaLAPACK)
#   AOCL_BLIS  - AMD BLIS only + reference LAPACK (ScaLAPACK-compatible)
#   OPENBLAS   - OpenBLAS (preferred fallback)
#   GENERIC    - system reference BLAS / LAPACK
#
# This module only handles the SERIAL linear algebra stack. ScaLAPACK is
# selected separately in cmake/EDMpiScalapack.cmake (so MPI / WITH_SCALAPACK
# decisions stay together with MPI detection).
#
# Variables produced (parent / directory scope):
#   BLAS_LIBRARIES, LAPACK_LIBRARIES, LAPACKE_LIBRARIES,
#   EXTRA_LINALG_LIBRARIES, BLAS_VENDOR, SCALAPACK_COMPATIBLE,
#   plus profile-specific compile definitions (-DUSE_AOCL_BLIS,
#   -DUSE_FLEXIBLAS, -DWITH_MKL, ...).
#
# P1.1 / audit Q5.
# =============================================================================

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

    # Default LAPACKE_ROOT to a user-provided ED_LAPACKE_ROOT if set; otherwise
    # leave empty and let the search below fall back to system paths.
    # (The previous version hard-coded a per-user absolute path; that has been
    # replaced by the ED_LAPACKE_ROOT cache variable as part of P0.10.)
    if(NOT LAPACKE_ROOT AND ED_LAPACKE_ROOT AND EXISTS "${ED_LAPACKE_ROOT}")
        set(LAPACKE_ROOT "${ED_LAPACKE_ROOT}")
        message(STATUS "  LAPACKE_ROOT (from ED_LAPACKE_ROOT): ${LAPACKE_ROOT}")
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

        # BLAS shim to override system libblas (avoids MKL alternatives via liblapacke).
        # Set ED_BLAS_SHIM_DIR via -DED_BLAS_SHIM_DIR=... or in local.cmake to point
        # at a directory containing libblas.so.3. Hard-coded path removed in P0.10.
        if(ED_BLAS_SHIM_DIR AND EXISTS "${ED_BLAS_SHIM_DIR}/libblas.so.3")
            list(PREPEND CMAKE_BUILD_RPATH ${ED_BLAS_SHIM_DIR})
            list(PREPEND CMAKE_INSTALL_RPATH ${ED_BLAS_SHIM_DIR})
            message(STATUS "  Added BLAS shim RPATH (prepend): ${ED_BLAS_SHIM_DIR}")
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
message(STATUS "BLAS library:      ${BLAS_LIBRARIES}")
message(STATUS "LAPACK library:    ${LAPACK_LIBRARIES}")
