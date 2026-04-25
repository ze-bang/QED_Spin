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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
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

// ----- tridiagonal solvers ---------------------------------------------------
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

// ----- the templated Lanczos kernel -----------------------------------------
//
// Pre-scattered `v0_local` of length `op.local_size()` is the initial Krylov
// vector slab on this rank. Caller is responsible for making it
// L2-normalised globally; the kernel re-normalises defensively to absorb
// scatter noise.
template <typename OpT>
DistributedLanczosResult distributed_lanczos_kernel(
    const OpT& op,
    std::vector<Complex>&& v0_local,
    const DistributedLanczosOptions& options) {

    const int rank = op.rank();
    const std::uint64_t local_n = op.local_size();
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

    const bool keep_basis =
        options.full_reorth || options.compute_eigenvectors;

    std::vector<Complex> v_curr = std::move(v0_local);

    // Defensive re-normalisation.
    {
        const double n2 = dist_norm(v_curr.data(), local_n, op.comm());
        if (n2 > 0.0) local_scal(1.0 / n2, v_curr.data(), local_n);
    }

    std::vector<std::vector<Complex>> basis;
    if (keep_basis) {
        basis.reserve(max_iter);
        basis.push_back(v_curr);
    }

    std::vector<Complex> v_prev(local_n, Complex(0.0, 0.0));
    std::vector<Complex> w(local_n, Complex(0.0, 0.0));

    std::vector<double> alpha; alpha.reserve(max_iter);
    std::vector<double> beta;  beta.reserve(max_iter + 1);
    beta.push_back(0.0);

    double prev_smallest = std::numeric_limits<double>::infinity();
    int iters_done = 0;

    for (std::uint64_t j = 0; j < max_iter; ++j) {
        op.apply(v_curr.data(), w.data());

        Complex alpha_c =
            dist_zdotc(v_curr.data(), w.data(), local_n, op.comm());
        alpha.push_back(alpha_c.real());

        local_axpy(Complex(-alpha.back(), 0.0), v_curr.data(),
                   w.data(), local_n);
        if (j > 0) {
            local_axpy(Complex(-beta.back(), 0.0), v_prev.data(),
                       w.data(), local_n);
        }

        if (keep_basis && !basis.empty()) {
            for (std::size_t k = 0; k < basis.size(); ++k) {
                Complex c = dist_zdotc(basis[k].data(), w.data(),
                                       local_n, op.comm());
                local_axpy(-c, basis[k].data(), w.data(), local_n);
            }
        }

        const double b = dist_norm(w.data(), local_n, op.comm());
        beta.push_back(b);
        ++iters_done;

        if (options.verbose && rank == 0) {
            std::cout << "  [dist-lanczos-kernel] j=" << j
                      << " alpha=" << alpha.back()
                      << " beta_{j+1}=" << b << std::endl;
        }

        if (b < 1e-14) break;

        if ((j + 1) % 5 == 0 || j + 1 == max_iter) {
            std::vector<double> ev = solve_tridiag(alpha, beta, alpha.size());
            const double smallest = ev.front();
            if (alpha.size() >= exct + 1) {
                if (std::abs(smallest - prev_smallest) < tol) {
                    iters_done = static_cast<int>(alpha.size());
                    if (options.verbose && rank == 0) {
                        std::cout << "  [dist-lanczos-kernel] converged at "
                                  << "iter " << alpha.size()
                                  << " (smallest=" << smallest << ")\n";
                    }
                    break;
                }
            }
            prev_smallest = smallest;
        }

        v_prev.swap(v_curr);
        local_scal(1.0 / b, w.data(), local_n);
        v_curr.swap(w);

        if (keep_basis) basis.push_back(v_curr);
    }

    DistributedLanczosResult result;
    result.iterations = iters_done;

    if (options.compute_eigenvectors) {
        std::vector<double> evals_unsorted, weights_unsorted, evecs_cm;
        solve_tridiag_with_eigenvectors(alpha, beta, alpha.size(),
                                        evals_unsorted, weights_unsorted,
                                        evecs_cm);
        result.tridiag_eigenvalues  = evals_unsorted;
        result.tridiag_weights      = weights_unsorted;
        result.tridiag_eigenvectors = std::move(evecs_cm);
        if (basis.size() > static_cast<std::size_t>(iters_done)) {
            basis.resize(static_cast<std::size_t>(iters_done));
        }
        result.krylov_basis_local = std::move(basis);
        std::vector<double> evals_sorted = evals_unsorted;
        std::sort(evals_sorted.begin(), evals_sorted.end());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  evals_sorted.size());
        result.eigenvalues.assign(evals_sorted.begin(),
                                  evals_sorted.begin() + n_keep);
    } else if (options.compute_weights) {
        std::vector<double> evals_unsorted, weights_unsorted;
        solve_tridiag_with_weights(alpha, beta, alpha.size(),
                                   evals_unsorted, weights_unsorted);
        result.tridiag_eigenvalues = evals_unsorted;
        result.tridiag_weights     = weights_unsorted;
        std::vector<double> evals_sorted = evals_unsorted;
        std::sort(evals_sorted.begin(), evals_sorted.end());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  evals_sorted.size());
        result.eigenvalues.assign(evals_sorted.begin(),
                                  evals_sorted.begin() + n_keep);
    } else {
        std::vector<double> all_ev =
            solve_tridiag(alpha, beta, alpha.size());
        const std::size_t n_keep =
            std::min<std::size_t>(static_cast<std::size_t>(exct),
                                  all_ev.size());
        result.eigenvalues.assign(all_ev.begin(), all_ev.begin() + n_keep);
    }
    return result;
}

}  // namespace ed::distributed::kernel

#endif  // WITH_MPI
