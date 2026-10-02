#pragma once
// =============================================================================
// include/ed/krylov/tridiag.h -- the eigensolve of a Lanczos tridiagonal, the one every Krylov-family
// kernel uses (the block lanes, Krylov-Schur, FTLM, OFTLM, mTPQ bounds, the continued fraction,
// dynamics), and its O(m)-per-value form for the convergence gates. Host-only, no CUDA dependency.
// =============================================================================

#include <ed/core/lapack.h>
#include <ed/core/errors.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
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

/// The lowest eigenvalues of a Lanczos tridiagonal, each one's unit eigenvector, and the largest
/// eigenvalue: what the convergence gates read (the Paige bound of Ritz value j is
/// |beta_m z_{m-1}(j)|).
struct TridiagEnds {
    std::size_t         m = 0;
    std::vector<double> values;    ///< the `count` lowest, ascending
    std::vector<double> vectors;   ///< column-major m x count: column j is the eigenvector of values[j]
    double              top = 0.0; ///< the largest eigenvalue

    /// Component `row` of the eigenvector of values[col].
    [[nodiscard]] double z(std::size_t row, std::size_t col) const { return vectors[col * m + row]; }
};

/// tridiag_ends of the leading m x m block (tridiag_eig's convention): the `count` lowest eigenpairs by
/// bisection and inverse iteration and the largest eigenvalue by bisection (LAPACK dstevx), O(m) per
/// value where tridiag_eig with vectors costs O(m^2) or more; a convergence gate can afford it every
/// step. Throws like tridiag_eig.
[[nodiscard]] inline TridiagEnds tridiag_ends(const std::vector<double>& alpha, const std::vector<double>& beta,
                                              std::size_t m, std::size_t count) {
    TridiagEnds t;
    t.m = m;
    if (m == 0) return t;
    count = std::max<std::size_t>(1, std::min(count, m));
    const auto entries = [&] {
        std::pair<std::vector<double>, std::vector<double>> de(
            std::vector<double>(alpha.begin(), alpha.begin() + static_cast<std::ptrdiff_t>(m)),
            std::vector<double>(m > 1 ? m - 1 : 1, 0.0));
        for (std::size_t i = 0; i + 1 < m; ++i) de.second[i] = beta[i + 1];
        return de;
    };
    auto [d, e] = entries();
    for (std::size_t i = 0; i < m; ++i)
        if (!std::isfinite(d[i]) || (i + 1 < m && !std::isfinite(e[i])))
            throw ed::ConvergenceError("tridiagonal eigensolve: non-finite entry at step "
                                       + std::to_string(i) + " of " + std::to_string(m));
    const lapack_int n = static_cast<lapack_int>(m), want = static_cast<lapack_int>(count);
    const double abstol = 2.0 * std::numeric_limits<double>::min();   // bisection to full accuracy (dlamch(S))
    std::vector<double> w(m);
    std::vector<lapack_int> ifail(m);
    lapack_int found = 0;
    t.vectors.assign(m * count, 0.0);
    lapack_int info = LAPACKE_dstevx(LAPACK_COL_MAJOR, 'V', 'I', n, d.data(), e.data(), 0.0, 0.0, 1, want,
                                     abstol, &found, w.data(), t.vectors.data(), n, ifail.data());
    if (info != 0 || found != want)
        throw ed::ConvergenceError("tridiagonal eigensolve failed (dstevx info " + std::to_string(info)
                                   + ", m = " + std::to_string(m) + ")");
    t.values.assign(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(count));
    if (count == m) {
        t.top = t.values.back();
        return t;
    }
    auto [d2, e2] = entries();   // dstevx may rescale its inputs
    info = LAPACKE_dstevx(LAPACK_COL_MAJOR, 'N', 'I', n, d2.data(), e2.data(), 0.0, 0.0, n, n, abstol, &found,
                          w.data(), nullptr, 1, ifail.data());
    if (info != 0 || found != 1)
        throw ed::ConvergenceError("tridiagonal eigensolve failed (dstevx info " + std::to_string(info)
                                   + ", m = " + std::to_string(m) + ")");
    t.top = w[0];
    return t;
}

/// Eigenvalues (ascending) and eigenvectors of a dense m x m symmetric matrix `a` (column-major,
/// overwritten): Krylov-Schur's projected matrix after a thick restart, an arrowhead plus a
/// tridiagonal. LAPACK dsyevd; throws ed::ConvergenceError on a non-finite entry or when it fails.
[[nodiscard]] inline TridiagEig symmetric_eig(std::vector<double> a, std::size_t m) {
    TridiagEig t;
    t.m = m;
    if (m == 0) return t;
    for (double x : a)
        if (!std::isfinite(x)) throw ed::ConvergenceError("projected eigensolve: non-finite entry");
    t.values.assign(m, 0.0);
    const lapack_int n = static_cast<lapack_int>(m);
    const lapack_int info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', n, a.data(), n, t.values.data());
    if (info != 0)
        throw ed::ConvergenceError("projected eigensolve failed (dsyevd info " + std::to_string(info)
                                   + ", m = " + std::to_string(m) + ")");
    t.vectors = std::move(a);
    return t;
}

}  // namespace ed::krylov
