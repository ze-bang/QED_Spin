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

    return [st, gate, tol, exct]
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
        // GAP-10 v3 (2026-07-17, the ROOT fix): despite taking `exct`,
        // this predicate historically certified only THE SMALLEST Ritz
        // value's stall -- values 2..exct of a requested window were
        // whatever they happened to be when E0 plateaued, which is
        // exactly how stalled interior garbage reached merged spectra
        // (GAP 10) and why window contents differed across BLAS
        // backends. Now: the stall check stays on E0 (cheap early
        // exit), and when exct > 1 the WINDOW must also certify via
        // the textbook per-Ritz residual bound |beta_m| * |z_{m-1,i}|
        // for the exct lowest values. Converged values agree on every
        // backend, so window counts are environment-stable by
        // construction; the downstream merge filter becomes a
        // tripwire instead of a load-bearing patch.
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
        es.compute(T, exct > 1 ? Eigen::ComputeEigenvectors
                               : Eigen::EigenvaluesOnly);
        if (es.info() != Eigen::Success) return false;

        const double smallest = es.eigenvalues()(0);
        const double denom    = std::max(std::abs(smallest), 1e-300);
        const double delta    = std::abs(smallest - st->prev) / denom;
        st->prev = smallest;
        if (delta >= tol) return false;
        if (exct <= 1) return true;
        const double beta_m =
            (beta.size() > static_cast<std::size_t>(m))
                ? std::abs(beta[static_cast<std::size_t>(m)]) : 0.0;
        if (beta_m == 0.0) return true;   // exhausted the space: exact
        const double btol = std::max(tol * 10.0, 1e-8);
        const int want = std::min<int>(static_cast<int>(exct), m);
        for (int i2 = 0; i2 < want; ++i2) {
            const double bound =
                beta_m * std::abs(es.eigenvectors()(m - 1, i2));
            if (bound > btol) return false;
        }
        return true;
    };
}

}  // namespace ed::krylov
