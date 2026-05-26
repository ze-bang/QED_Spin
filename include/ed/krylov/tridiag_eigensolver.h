#pragma once
// =============================================================================
// include/ed/krylov/tridiag_eigensolver.h
//
// Pure-host helpers for solving the small projected tridiagonal eigen-
// problem that every Lanczos-family kernel (single-vector Lanczos,
// thick-restart Krylov-Schur, block Lanczos, FTLM Jacobi-and-Pratt
// trace estimator) needs after the m-step Krylov factorisation.
//
// These helpers are pure host code: they consume two `std::vector<double>`
// (the alpha / beta tridiagonal entries) and return / fill another set of
// `std::vector<double>` (eigenvalues, optional first-component weights,
// optional column-major eigenvectors). They depend only on Eigen and on
// the C++ standard library; in particular, they have **no** MPI or CUDA
// dependency, so they are safe to include from any Krylov kernel
// regardless of `WITH_MPI` / `WITH_CUDA`.
//
// Provenance: extracted unchanged from
// `include/ed/distributed/distributed_lanczos_kernel.h` (where they lived
// inside an `#ifdef WITH_MPI` block and were therefore unavailable to the
// CPU-only Krylov-Schur kernel in `include/ed/krylov/krylov_schur_kernel.h`).
// The distributed kernel header now `#include`s this header and re-exports
// the three functions via a `using` alias, so callers that previously
// reached for `ed::distributed::kernel::solve_tridiag_with_eigenvectors`
// continue to work unchanged.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <vector>

#include <Eigen/Dense>

namespace ed::krylov::detail {

/// Eigenvalues only. Returns a sorted `std::vector<double>` of length `m`.
inline std::vector<double> solve_tridiag(const std::vector<double>& alpha,
                                         const std::vector<double>& beta,
                                         std::size_t m) {
    if (m == 0) return {};
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(T, Eigen::EigenvaluesOnly);
    std::vector<double> evals(m);
    for (std::size_t i = 0; i < m; ++i) evals[i] = es.eigenvalues()(i);
    std::sort(evals.begin(), evals.end());
    return evals;
}

/// Eigenvalues + first-component weights `|<e_0, y_k>|^2` (the FTLM / TPQ
/// Jacobi-and-Pratt trace-estimator coefficient).
inline void solve_tridiag_with_weights(const std::vector<double>& alpha,
                                       const std::vector<double>& beta,
                                       std::size_t m,
                                       std::vector<double>& evals,
                                       std::vector<double>& weights) {
    evals.clear(); weights.clear();
    if (m == 0) return;
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    evals.resize(m); weights.resize(m);
    const auto& V = es.eigenvectors();
    for (std::size_t k = 0; k < m; ++k) {
        evals[k] = es.eigenvalues()(k);
        const double v0k = V(0, k);
        weights[k] = v0k * v0k;
    }
}

/// Eigenvalues + first-component weights + full column-major eigenvector
/// matrix (column k starts at `evecs_cm[k * m]`). Used by the Krylov-Schur
/// kernel to extract Ritz vectors for restart.
inline void solve_tridiag_with_eigenvectors(const std::vector<double>& alpha,
                                            const std::vector<double>& beta,
                                            std::size_t m,
                                            std::vector<double>& evals,
                                            std::vector<double>& weights,
                                            std::vector<double>& evecs_cm) {
    evals.clear(); weights.clear(); evecs_cm.clear();
    if (m == 0) return;
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        T(i, i) = alpha[i];
        if (i + 1 < m) {
            T(i, i + 1) = beta[i + 1];
            T(i + 1, i) = beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    evals.resize(m); weights.resize(m); evecs_cm.resize(m * m);
    const auto& V = es.eigenvectors();
    for (std::size_t k = 0; k < m; ++k) {
        evals[k] = es.eigenvalues()(k);
        const double v0k = V(0, k);
        weights[k] = v0k * v0k;
        for (std::size_t j = 0; j < m; ++j) {
            evecs_cm[k * m + j] = V(j, k);
        }
    }
}

}  // namespace ed::krylov::detail
