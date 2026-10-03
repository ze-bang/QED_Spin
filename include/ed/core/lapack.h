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

#include <cstdint>

namespace ed::core {

/// The largest n whose n x n matrix the linked LAPACK can address. With a 32-bit lapack_int (LP64)
/// the element offsets of an n x n column-major matrix overflow past n = 46340 (46340^2 < 2^31 <
/// 46341^2): the dense solvers refuse larger blocks instead of returning wrong spectra, and the
/// dense crossovers stay below it. A 64-bit lapack_int (ILP64) has no practical limit.
[[nodiscard]] constexpr std::uint64_t lapack_max_dense_n() noexcept {
    return sizeof(lapack_int) >= 8 ? (std::uint64_t{1} << 31) : std::uint64_t{46340};
}

}  // namespace ed::core
