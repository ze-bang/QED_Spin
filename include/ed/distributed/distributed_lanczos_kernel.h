// =============================================================================
// include/ed/distributed/distributed_lanczos_kernel.h    (Phase 3b #7, stage 3)
//
// Header-only templated Lanczos kernel that BOTH `DistributedOperator`
// (1D row-slab decomposition over the unsymmetrised basis) AND
// `DistributedSymmetryOperator` (LPT-balanced orbit slabs over the
// symmetry-projected basis) can use, without code duplication.
//
// The template is gated on the duck-typed "DistributedOperator-shaped"
// interface that the kernel actually consumes:
//
//   void   op.apply(const Complex* x_local, Complex* y_local) const;
//   int    op.rank()         const;
//   int    op.comm_size()    const;
//   MPI_Comm op.comm()       const;
//   uint64 op.global_dim()   const;   // total global rows
//   uint64 op.local_size()   const;   // rank-local rows
//
// Both operators above implement this surface. The kernel performs:
//
//   * Three-term Lanczos recurrence on rank-local slabs.
//   * `MPI_Allreduce(MPI_SUM)` for `dot` (alpha) and `norm` (beta).
//   * Optional full-MGS re-orthogonalisation against every prior basis
//     vector (also `MPI_Allreduce` per pair).
//   * Tridiagonal eigenproblem solved redundantly on every rank.
//
// Eigenvalue/eigenvector/weight extraction is identical to the
// `DistributedOperator` kernel that ships in
// `src/distributed/distributed_lanczos.cpp`. To keep changes minimal we
// stay value-compatible with `DistributedLanczosResult` /
// `DistributedLanczosOptions` so callers can swap the entry point
// without changing call sites.
//
// Initial vector handling differs between operators because the
// row-slab geometry is different (`DistributedOperator::balanced_slab`
// vs `OrbitPartition::rank_orbits[r]`). The kernel takes an already-
// scattered initial vector to keep itself geometry-agnostic.
// =============================================================================

#pragma once

#ifdef WITH_MPI

#include <ed/distributed/distributed_lanczos.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/ritz_convergence.h>
#include <ed/krylov/tridiag_eigensolver.h>
#include <ed/matvec/backends/mpi_backend.h>
#include <ed/parallel/thread_budget.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>     // Wave 4.5: ED_DIST_LANCZOS_LOCAL_DGKS env
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <mpi.h>

namespace ed::distributed::kernel {

using Complex = std::complex<double>;

// Not constexpr: OpenMPI predefined handles cast through (void*) at runtime.
inline MPI_Datatype mpi_complex_datatype() { return MPI_C_DOUBLE_COMPLEX; }

// ----- rank-local helpers ----------------------------------------------------
inline double local_norm_sq(const Complex* x, std::uint64_t n) {
    double s = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) s += std::norm(x[i]);
    return s;
}

inline Complex local_zdotc(const Complex* x, const Complex* y,
                           std::uint64_t n) {
    Complex s(0.0, 0.0);
    for (std::uint64_t i = 0; i < n; ++i) s += std::conj(x[i]) * y[i];
    return s;
}

inline void local_axpy(Complex a, const Complex* x, Complex* y,
                       std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) y[i] += a * x[i];
}

inline void local_scal(double a, Complex* x, std::uint64_t n) {
    for (std::uint64_t i = 0; i < n; ++i) x[i] *= a;
}

inline double dist_norm(const Complex* x_local, std::uint64_t n_local,
                        MPI_Comm comm) {
    double local = local_norm_sq(x_local, n_local);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return std::sqrt(global);
}

inline Complex dist_zdotc(const Complex* x_local, const Complex* y_local,
                          std::uint64_t n_local, MPI_Comm comm) {
    Complex local = local_zdotc(x_local, y_local, n_local);
    double buf_in[2]  = {local.real(), local.imag()};
    double buf_out[2] = {0.0, 0.0};
    MPI_Allreduce(buf_in, buf_out, 2, MPI_DOUBLE, MPI_SUM, comm);
    return Complex(buf_out[0], buf_out[1]);
}

// Phase 8 #6: batched <basis[k]|w> for all k via a single MPI_Allreduce.
// See the matching docstring on
// ``ed::distributed::dist_zdotc_batched`` in distributed_lanczos.cpp.
inline void dist_zdotc_batched(
    const std::vector<std::vector<Complex>>& basis,
    const Complex* w_local,
    std::uint64_t n_local,
    MPI_Comm comm,
    std::vector<Complex>& c_out) {

    const std::size_t m = basis.size();
    c_out.assign(m, Complex(0.0, 0.0));
    if (m == 0) return;

    std::vector<double> local_buf(2 * m, 0.0);
    for (std::size_t k = 0; k < m; ++k) {
        const Complex c = local_zdotc(basis[k].data(), w_local, n_local);
        local_buf[2 * k]     = c.real();
        local_buf[2 * k + 1] = c.imag();
    }
    std::vector<double> global_buf(2 * m, 0.0);
    MPI_Allreduce(local_buf.data(), global_buf.data(),
                  static_cast<int>(2 * m),
                  MPI_DOUBLE, MPI_SUM, comm);
    for (std::size_t k = 0; k < m; ++k) {
        c_out[k] = Complex(global_buf[2 * k], global_buf[2 * k + 1]);
    }
}

// ----- tridiagonal solvers ---------------------------------------------------
// Re-exported from `ed::krylov::detail::solve_tridiag*` so the same Eigen-only
// helpers are visible in both the MPI and the CPU-only Krylov-Schur kernels.
// The CPU-only kernel cannot reach into this `#ifdef WITH_MPI` block, so the
// canonical definitions now live in `ed/krylov/tridiag_eigensolver.h`.
using ed::krylov::detail::solve_tridiag;
using ed::krylov::detail::solve_tridiag_with_weights;
using ed::krylov::detail::solve_tridiag_with_eigenvectors;

// ----- the templated Lanczos kernel -----------------------------------------
//
// Pre-scattered `v0_local` of length `op.local_size()` is the initial Krylov
// vector slab on this rank. Caller is responsible for making it
// L2-normalised globally; the kernel re-normalises defensively to absorb
// scatter noise.
//
// May 2026 (Phase D of the Krylov-kernel unification): this body now
// delegates to `ed::krylov::lanczos_kernel<MpiBackend>`. The previous
// ~140-line inline three-term recurrence (matvec, dot/Allreduce, axpy,
// batched CGS2 reorth, beta-breakdown, relative-Δλ early-exit) was
// near-identical to the unified kernel in `ed/krylov/lanczos_kernel.h`;
// this function now just supplies the `MpiBackend(op.comm())`, the
// matvec callback, the early-exit predicate
// (`make_smallest_ritz_convergence(exct, tol)`), and the
// `DistributedLanczosResult`-flavoured post-processing.
template <typename OpT>
DistributedLanczosResult distributed_lanczos_kernel(
    const OpT& op,
    std::vector<Complex>&& v0_local,
    const DistributedLanczosOptions& options) {

    const int rank = op.rank();
    const std::uint64_t local_n  = op.local_size();
    const std::uint64_t max_iter = options.max_iter;
    const std::uint64_t exct = std::max<std::uint64_t>(1, options.exct);
    const double tol = options.tol;

    if (max_iter == 0) {
        throw std::invalid_argument("distributed_lanczos_kernel: max_iter == 0");
    }
    if (v0_local.size() != local_n) {
        throw std::invalid_argument(
            "distributed_lanczos_kernel: initial vector slab size ("
            + std::to_string(v0_local.size())
            + ") != op.local_size() (" + std::to_string(local_n) + ")");
    }

    // Phase 8 #3: dim-aware OMP+BLAS thread cap on the rank-local slab.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(local_n));

    const bool keep_basis =
        options.full_reorth || options.compute_eigenvectors;

    // Defensive re-normalise -- `lanczos_kernel` does this internally
    // too, but doing it here means the initial vector slab the caller
    // hands us is consistent with the kernel's first basis vector
    // (otherwise different fp roundoff regimes produce subtly different
    // V_0 across MPI runs and rank-local eigenvectors disagree on the
    // last bit).
    {
        const double n2 = dist_norm(v0_local.data(), local_n, op.comm());
        if (n2 > 0.0) local_scal(1.0 / n2, v0_local.data(), local_n);
    }

    ed::matvec::MpiBackend mpi(op.comm());

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter      = static_cast<std::size_t>(max_iter);

    // Wave 4.5 of the SOTA Performance rollout (May 2026): on a
    // distributed Lanczos with ``keep_basis == false`` (the
    // eigenvalues-only path) the K=1 LocalDGKS3 reorth needs only
    // ``v_curr`` / ``v_prev`` (already maintained), saving the
    // m-vector ``dot_many``/``axpy_many`` batched Allreduce per
    // iter that FullCGS2 demands. Opt-in via
    // ``ED_DIST_LANCZOS_LOCAL_DGKS=1`` -- the default keeps the
    // pre-Wave behaviour (None for eigvals-only, FullCGS2 with
    // basis) so existing accuracy guarantees on near-degenerate
    // spectra are preserved. Production users on latency-bound
    // multi-node runs should benchmark both and adopt the env.
    const bool use_local_dgks = []() {
        const char* env = std::getenv("ED_DIST_LANCZOS_LOCAL_DGKS");
        return env && env[0] == '1';
    }();
    if (keep_basis) {
        kopts.reorth = ed::krylov::ReorthPolicy::FullCGS2;
    } else if (use_local_dgks) {
        kopts.reorth         = ed::krylov::ReorthPolicy::LocalDGKS3;
        kopts.local_ring_size = 1;
    } else {
        kopts.reorth = ed::krylov::ReorthPolicy::None;
    }
    kopts.keep_basis    = keep_basis;
    // Critical for distributed runs: the kernel's default
    // `cap = min(max_iter, local_n)` is computed from the RANK-LOCAL
    // slab, which is too small for small problems split over many
    // ranks (e.g. global dim 6 / np=4 gives local_n in {1, 2} and the
    // kernel would stop after a single iteration). Override the cap
    // with the GLOBAL problem dimension to match the legacy
    // `distributed_lanczos` body, which only bounded by `max_iter`.
    kopts.dim_cap       = static_cast<std::size_t>(op.global_dim());
    // Matches the legacy `b < 1e-14` breakdown threshold in this kernel's
    // previous inline body. Distinct from the kernel's default
    // `breakdown_tol = 1e-300` (which is the "exact zero" detection
    // that the LTLM tests rely on); the historic distributed-MPI body
    // had a much looser breakdown bar.
    kopts.breakdown_tol = 1e-14;
    // Relative-Δλ early-exit every 5 iterations, matching the legacy
    // cadence and Ritz tolerance. The predicate captures its own state
    // so we hand off a fresh closure per call.
    kopts.convergence_check_interval = 5;
    kopts.convergence_check =
        ed::krylov::make_smallest_ritz_convergence(
            static_cast<std::size_t>(exct), tol);

    auto matvec = [&op](const Complex* in, Complex* out, std::size_t /*n*/) {
        op.apply(in, out);
    };

    auto kres = ed::krylov::lanczos_kernel(
        mpi, matvec, local_n, v0_local.data(), kopts);

    if (options.verbose && rank == 0) {
        std::cout << "  [dist-lanczos-kernel] completed "
                  << kres.iters_done << " iters (max_iter=" << max_iter
                  << ")\n";
    }

    const std::size_t m = kres.alpha.size();
    DistributedLanczosResult result;
    result.iterations = static_cast<int>(kres.iters_done);

    if (options.compute_eigenvectors) {
        std::vector<double> evals_unsorted, weights_unsorted, evecs_cm;
        solve_tridiag_with_eigenvectors(kres.alpha, kres.beta, m,
                                        evals_unsorted, weights_unsorted,
                                        evecs_cm);
        result.tridiag_eigenvalues  = std::move(evals_unsorted);
        result.tridiag_weights      = std::move(weights_unsorted);
        result.tridiag_eigenvectors = std::move(evecs_cm);

        // Copy the kernel's backend-owned basis (Backend::UniqueVec
        // entries; host RAM for MpiBackend) into the
        // `DistributedLanczosResult::krylov_basis_local` shape that the
        // legacy ABI promises. `kres.basis.size() == kres.iters_done`
        // already (the kernel does not push the trailing V_m on the
        // final iteration), so no resize is needed.
        result.krylov_basis_local.assign(kres.basis.size(),
                                         std::vector<Complex>{});
        for (std::size_t k = 0; k < kres.basis.size(); ++k) {
            const Complex* src = kres.basis[k].get();
            result.krylov_basis_local[k].assign(src, src + local_n);
        }

        std::vector<double> evals_sorted = result.tridiag_eigenvalues;
        std::sort(evals_sorted.begin(), evals_sorted.end());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  evals_sorted.size());
        result.eigenvalues.assign(evals_sorted.begin(),
                                  evals_sorted.begin() + n_keep);
    } else if (options.compute_weights) {
        std::vector<double> evals_unsorted, weights_unsorted;
        solve_tridiag_with_weights(kres.alpha, kres.beta, m,
                                   evals_unsorted, weights_unsorted);
        result.tridiag_eigenvalues = std::move(evals_unsorted);
        result.tridiag_weights     = std::move(weights_unsorted);
        std::vector<double> evals_sorted = result.tridiag_eigenvalues;
        std::sort(evals_sorted.begin(), evals_sorted.end());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  evals_sorted.size());
        result.eigenvalues.assign(evals_sorted.begin(),
                                  evals_sorted.begin() + n_keep);
    } else {
        std::vector<double> all_ev = solve_tridiag(kres.alpha, kres.beta, m);
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  all_ev.size());
        result.eigenvalues.assign(all_ev.begin(),
                                  all_ev.begin() + n_keep);
    }
    return result;
}

}  // namespace ed::distributed::kernel

#endif  // WITH_MPI
