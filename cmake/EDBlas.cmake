# =============================================================================
# cmake/EDBlas.cmake -- BLAS, LAPACK and LAPACKE (CBLAS + LAPACKE headers).
#
# BLAS_PROFILE:
#   FLEXIBLAS  the Alliance clusters (scripts/clusters/alliance.env): libflexiblas for BLAS and
#              LAPACK, AMD libflame for LAPACKE
#   OPENBLAS   OpenBLAS + liblapacke (CI, workstations)
#   MKL        oneMKL, LP64, threaded; defines WITH_MKL (lapack.h includes mkl.h)
#   AUTO       whatever FindBLAS / FindLAPACK find, + liblapacke
# Sets ED_LINALG_LIBRARIES in link order. The include directories are directory-wide SYSTEM.
# =============================================================================

set(BLAS_PROFILE "AUTO" CACHE STRING "BLAS/LAPACK provider: AUTO, FLEXIBLAS, OPENBLAS or MKL")
set_property(CACHE BLAS_PROFILE PROPERTY STRINGS AUTO FLEXIBLAS OPENBLAS MKL)

if(BLAS_PROFILE STREQUAL "FLEXIBLAS")
    set(_hints $ENV{EBROOTFLEXIBLAS} $ENV{FLEXIBLAS_ROOT} /usr /usr/local)
    find_library(FLEXIBLAS_LIBRARY NAMES flexiblas HINTS ${_hints} PATH_SUFFIXES lib lib64 REQUIRED)
    find_path(FLEXIBLAS_INCLUDE_DIR flexiblas/cblas.h cblas.h HINTS ${_hints} PATH_SUFFIXES include include/flexiblas)
    find_library(LAPACKE_LIBRARY NAMES flame HINTS $ENV{EBROOTAOCLMINLAPACK} $ENV{AOCL_ROOT} ${_hints}
                 PATH_SUFFIXES lib lib64 REQUIRED)
    # libflexiblas first: both libraries export the LAPACK symbols, and the first in load order serves them.
    set(ED_LINALG_LIBRARIES ${FLEXIBLAS_LIBRARY} ${LAPACKE_LIBRARY})
    # <cblas.h> and <lapacke.h> must be FlexiBLAS's, which CPATH lists after AOCL's include directory:
    # naming AOCL's directory -isystem moves it behind every CPATH entry.
    if(EXISTS "$ENV{EBROOTAOCLMINLAPACK}/include/lapacke.h")
        include_directories(SYSTEM $ENV{EBROOTAOCLMINLAPACK}/include)
    endif()
    if(FLEXIBLAS_INCLUDE_DIR)
        include_directories(SYSTEM ${FLEXIBLAS_INCLUDE_DIR})
    endif()
    get_filename_component(_flame_dir ${LAPACKE_LIBRARY} DIRECTORY)
    list(APPEND CMAKE_BUILD_RPATH ${_flame_dir})
    list(APPEND CMAKE_INSTALL_RPATH ${_flame_dir})
elseif(BLAS_PROFILE STREQUAL "MKL")
    set(BLA_VENDOR Intel10_64lp)
    find_package(LAPACK REQUIRED)                      # brings BLAS
    find_path(MKL_INCLUDE_DIR mkl.h HINTS $ENV{MKLROOT} PATH_SUFFIXES include REQUIRED)
    include_directories(SYSTEM ${MKL_INCLUDE_DIR})
    add_compile_definitions(WITH_MKL)
    set(ED_LINALG_LIBRARIES ${LAPACK_LIBRARIES})       # LAPACKE is part of MKL
elseif(BLAS_PROFILE STREQUAL "OPENBLAS" OR BLAS_PROFILE STREQUAL "AUTO")
    if(BLAS_PROFILE STREQUAL "OPENBLAS")
        set(BLA_VENDOR OpenBLAS)
    endif()
    find_package(LAPACK REQUIRED)                      # brings BLAS
    find_library(LAPACKE_LIBRARY NAMES lapacke REQUIRED)
    set(ED_LINALG_LIBRARIES ${LAPACKE_LIBRARY} ${LAPACK_LIBRARIES})
else()
    message(FATAL_ERROR "BLAS_PROFILE must be AUTO, FLEXIBLAS, OPENBLAS or MKL, not ${BLAS_PROFILE}")
endif()
message(STATUS "BLAS/LAPACK (${BLAS_PROFILE}): ${ED_LINALG_LIBRARIES}")
