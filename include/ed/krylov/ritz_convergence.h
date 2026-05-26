#pragma once
// =============================================================================
// include/ed/krylov/ritz_convergence.h
//
// Stateful Ritz-convergence predicate factory for the
// `LanczosKernelOptions::convergence_check` callback. The kernel header
// itself stays Eigen-free: callers who want the standard relative-Δλ
// early-exit pull in this header, hand the resulting `std::function` to
// the kernel, and the predicate carries its own per-call state.
//
// Standard policy:
//
//     |smallest_now - smallest_prev| / max(|smallest_now|, 1e-300) < tol
//
// This is the shared convention of every legacy ED Lanczos in the repo
// (`src/solvers/cpu/lanczos.cpp`,
// `src/distributed/distributed_lanczos.cpp` /
// `include/ed/distributed/distributed_lanczos_kernel.h`,
// `src/solvers/gpu/gpu_lanczos.cu`), and the well-trodden ARPACK / SLEPc
// / Anasazi default for one-eigenvalue convergence.
//
// May 2026 -- shipped alongside the `distributed_lanczos` migration onto
// `lanczos_kernel<MpiBackend>`.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace ed::krylov {

/// Build a stateful Ritz-convergence predicate suitable for
/// `LanczosKernelOptions::convergence_check`.
///
/// The returned `std::function` is invoked by the kernel every
/// `convergence_check_interval` iterations with the running `alpha`
/// and the running `beta` (beta layout follows the kernel's legacy
/// convention: `beta[0] == 0.0` is the sentinel, `beta[j+1]` is the
/// sub-diagonal entry `T(j+1, j)`). On every call the predicate solves
/// the `m x m` real-symmetric tridiagonal `T(alpha, beta)`, takes its
/// smallest eigenvalue, and returns `true` when the relative change
/// against the previous call drops below `tol`.
///
/// `min_iters` (default `exct + 1`) gates the very first check: until
/// the tridiagonal has at least that many rows the smallest Ritz value
/// hasn't seen a comparable measurement yet, and the predicate
/// unconditionally returns `false`.
///
/// Thread-safety: the returned function carries its own state inside a
/// `std::shared_ptr`. Each call to this factory produces an independent
/// predicate; do NOT share one predicate across concurrent
/// `lanczos_kernel` runs (e.g. if you parallelise multiple Lanczos
/// solves over an OMP region, build one predicate per region task).
inline std::function<bool(const std::vector<double>&,
                          const std::vector<double>&)>
make_smallest_ritz_convergence(std::size_t exct = 1,
                               double tol = 1e-12,
                               std::size_t min_iters = 0)
{
    struct State {
        double prev = std::numeric_limits<double>::infinity();
    };
    auto st = std::make_shared<State>();

    const std::size_t gate = (min_iters == 0) ? (exct + 1) : min_iters;

    return [st, gate, tol]
           (const std::vector<double>& alpha,
            const std::vector<double>& beta) -> bool
    {
        const int m = static_cast<int>(alpha.size());
        if (static_cast<std::size_t>(m) < gate) return false;

        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i) T(i, i) = alpha[i];
        for (int i = 1; i < m; ++i) {
            const double b = beta[i];  // kernel layout: beta[i] = T(i, i-1)
            T(i, i - 1) = b;
            T(i - 1, i) = b;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.compute(T, Eigen::EigenvaluesOnly);
        if (es.info() != Eigen::Success) return false;

        const double smallest = es.eigenvalues()(0);
        const double denom    = std::max(std::abs(smallest), 1e-300);
        const double delta    = std::abs(smallest - st->prev) / denom;
        st->prev = smallest;
        return delta < tol;
    };
}

}  // namespace ed::krylov
