# =============================================================================
# cmake/EDMpiScalapack.cmake
#
# MPI detection (gated by WITH_MPI).
#
# Sets:
#   MPI_IS_INTEL / MPI_IS_OPENMPI / MPI_IS_MPICH   informational flags
#   -DWITH_MPI    compile definition
#
# ScaLAPACK support was retired together with the SCALAPACK /
# SCALAPACK_MIXED diagonalization methods in the minimalist-architecture
# rev (May 2026); the distributed dense path is no longer needed because
# the Krylov family (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR) has fully
# functional MPI back-ends for any sector dim where dense diag is feasible.
# =============================================================================

if(WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    add_definitions(-DWITH_MPI)

    # Informational flags on the detected MPI implementation.
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
endif()
