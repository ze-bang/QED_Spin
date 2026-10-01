#pragma once
// =============================================================================
// include/ed/krylov/tridiag.h -- the eigensolve of a Lanczos tridiagonal, the one every Krylov-family
// kernel uses (the block lanes, Krylov-Schur, FTLM, OFTLM, mTPQ bounds, the continued fraction,
// dynamics). Host-only, no CUDA dependency.
// =============================================================================

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/errors.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace ed::krylov {

/// Eigenvalues (ascending) and, when asked, eigenvectors of an m x m symmetric tridiagonal.
struct TridiagEig {
    std::size_t m = 0;
    std::vector<double> values;
    std::vector<double> vectors;   ///< column-major m x m: column j is the eigenvector of values[j]

    /// Component `row` of the eigenvector of values[col].
    [[nodiscard]] double z(std::size_t row, std::size_t col) const { return vectors[col * m + row]; }

    /// The squared first components (the Lanczos start vector's weight on each Ritz vector).
    [[nodiscard]] std::vector<double> weights() const {
        std::vector<double> w(m);
        for (std::size_t j = 0; j < m; ++j) w[j] = vectors[j * m] * vectors[j * m];
        return w;
    }
};

/// The leading m x m block of the tridiagonal with diagonal alpha[0..m) and off-diagonal beta[1..m)
/// (lanczos_kernel's convention: beta[i] couples steps i-1 and i, beta[0] is unused). LAPACK dstevd:
/// values only through dsterf, with vectors through divide and conquer. Throws ed::ConvergenceError
/// on a non-finite entry (an H that produced one) and when LAPACK fails.
[[nodiscard]] inline TridiagEig tridiag_eig(const std::vector<double>& alpha, const std::vector<double>& beta,
                                            std::size_t m, bool vectors) {
    TridiagEig t;
    t.m = m;
    if (m == 0) return t;
    t.values.assign(alpha.begin(), alpha.begin() + static_cast<std::ptrdiff_t>(m));
    std::vector<double> off(m > 1 ? m - 1 : 1, 0.0);
    for (std::size_t i = 0; i + 1 < m; ++i) off[i] = beta[i + 1];
    for (std::size_t i = 0; i < m; ++i)
        if (!std::isfinite(t.values[i]) || (i + 1 < m && !std::isfinite(off[i])))
            throw ed::ConvergenceError("tridiagonal eigensolve: non-finite entry at step "
                                       + std::to_string(i) + " of " + std::to_string(m));
    if (vectors) t.vectors.assign(m * m, 0.0);
    const lapack_int n = static_cast<lapack_int>(m);
    const lapack_int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, vectors ? 'V' : 'N', n, t.values.data(),
                                           off.data(), vectors ? t.vectors.data() : nullptr, vectors ? n : 1);
    if (info != 0)
        throw ed::ConvergenceError("tridiagonal eigensolve failed (dstevd info " + std::to_string(info)
                                   + ", m = " + std::to_string(m) + ")");
    return t;
}

}  // namespace ed::krylov
