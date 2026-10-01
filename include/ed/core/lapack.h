// lapack.h - Unified BLAS/LAPACK interface
// Provides a single include that maps to the selected vendor backend

#pragma once

// MKL ships its own umbrella header (BLAS_PROFILE=MKL defines WITH_MKL); every other provider
// gives the standard CBLAS and LAPACKE headers (cmake/EDBlas.cmake).
#if defined(WITH_MKL)
    #include <mkl.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

// Ensure LAPACK_COMPLEX_CPP is defined for C++ std::complex interoperability.
#ifndef LAPACK_COMPLEX_CPP
    #define LAPACK_COMPLEX_CPP
#endif
