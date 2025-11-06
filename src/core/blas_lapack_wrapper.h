// blas_lapack_wrapper.h - Unified BLAS/LAPACK interface
// Provides a single include that maps to the selected vendor backend

#ifndef BLAS_LAPACK_WRAPPER_H
#define BLAS_LAPACK_WRAPPER_H

// Always prefer vendor-specific umbrella headers when available.
#if defined(WITH_MKL)
    #include <mkl.h>
    #define BLAS_LAPACK_BACKEND "Intel MKL"
#elif defined(USE_AOCL_BLIS)
    #include <cblas.h>
    #include <lapacke.h>
    #define BLAS_LAPACK_BACKEND "AMD AOCL-BLIS"
#else
    #include <cblas.h>
    #include <lapacke.h>
    #define BLAS_LAPACK_BACKEND "Generic BLAS/LAPACK"
#endif

// Ensure LAPACK_COMPLEX_CPP is defined for C++ std::complex interoperability.
#ifndef LAPACK_COMPLEX_CPP
    #define LAPACK_COMPLEX_CPP
#endif

#ifdef DEBUG_BLAS_BACKEND
    #include <iostream>
    namespace {
        struct BlasBackendReporter {
            BlasBackendReporter() {
                std::cout << "Using BLAS/LAPACK backend: " << BLAS_LAPACK_BACKEND << std::endl;
            }
        };
        static BlasBackendReporter g_blas_backend_reporter;
    }
#endif

#endif // BLAS_LAPACK_WRAPPER_H
