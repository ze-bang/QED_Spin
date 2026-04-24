# =============================================================================
# cmake/EDMpiScalapack.cmake
#
# MPI detection (gated by WITH_MPI) and ScaLAPACK detection (gated by
# WITH_SCALAPACK, which is itself gated by WITH_MPI).
#
# This module assumes that BLAS_PROFILE / SCALAPACK_COMPATIBLE / SCALAPACK_ROOT
# have already been set by cmake/EDBlasProfile.cmake (or by the user via
# -DSCALAPACK_ROOT=...), and that nothing else above touched MPI_*.
#
# Sets:
#   MPI_IS_INTEL / MPI_IS_OPENMPI / MPI_IS_MPICH   informational flags
#   WITH_SCALAPACK            possibly demoted to OFF if libs not found
#   SCALAPACK_LIBRARIES       list of -lscalapack -lblacs ...
#   -DWITH_MPI / -DWITH_SCALAPACK    compile definitions
#
# P1.1 / audit Q5.
# =============================================================================

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
