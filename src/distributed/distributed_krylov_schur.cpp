// =============================================================================
// src/distributed/distributed_krylov_schur.cpp
//
// Phase 9 / Layer 3: distributed thick-restart Lanczos.
// See header for the algorithmic and design notes.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_krylov_schur.h>
#include <ed/distributed/distributed_lanczos_kernel.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/parallel/thread_budget.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <mpi.h>

namespace ed::distributed {

using Complex = std::complex<double>;
using kernel::dist_norm;
using kernel::dist_zdotc;
using kernel::local_axpy;
using kernel::local_scal;
using kernel::solve_tridiag_with_eigenvectors;

namespace {

// Generate the global initial vector on rank 0 (deterministic from `seed`),
// scatter slabs to every rank, defensively renormalise. Mirrors the helper
// in distributed_lanczos.cpp.
void scatter_initial_vector(const DistributedOperator& op,
                            unsigned long seed,
                            std::vector<Complex>& v_local) {
    const int rank = op.rank();
    const int size = op.comm_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t local_n    = op.local_size();

    v_local.assign(local_n, Complex(0.0, 0.0));

    std::vector<int> sendcounts(size), displs(size);
    int run = 0;
    for (int r = 0; r < size; ++r) {
        std::uint64_t off, n;
        DistributedOperator::balanced_slab(global_dim, r, size, off, n);
        sendcounts[r] = static_cast<int>(n);
        displs[r]     = run;
        run          += sendcounts[r];
    }

    if (rank == 0) {
        std::vector<Complex> v_global(static_cast<std::size_t>(global_dim));
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_global[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_global) z *= inv;

        MPI_Scatterv(v_global.data(), sendcounts.data(), displs.data(),
                     MPI_C_DOUBLE_COMPLEX,
                     v_local.data(), sendcounts[rank], MPI_C_DOUBLE_COMPLEX,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     MPI_C_DOUBLE_COMPLEX,
                     v_local.data(), sendcounts[rank], MPI_C_DOUBLE_COMPLEX,
                     0, op.comm());
    }

    const double n2 = dist_norm(v_local.data(), local_n, op.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, v_local.data(), local_n);
}

// Symmetry-aware initial vector scatter (Phase D step 2). Mirrors the
// scatter inside `distributed_lanczos_symmetry`: rank 0 builds a
// deterministic global random vector in NATURAL orbit ordering, permutes
// it into rank-major + LPT-orbit-scrambled order, and MPI_Scatterv's
// each rank's slab. Slot k of `v_local` holds the amplitude of orbit
// `partition.rank_orbits[rank][k]` -- which is exactly what
// DistributedSymmetryOperator::apply expects.
void scatter_initial_vector(const DistributedSymmetryOperator& op,
                            unsigned long seed,
                            std::vector<Complex>& v_local) {
    const int rank = op.rank();
    const int size = op.comm_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t local_n    = op.local_size();

    v_local.assign(local_n, Complex(0.0, 0.0));

    const auto& partition = op.partition();
    std::vector<int> sendcounts(size, 0), displs(size, 0);
    {
        int run = 0;
        for (int r = 0; r < size; ++r) {
            sendcounts[r] = static_cast<int>(partition.rank_orbits[r].size());
            displs[r] = run;
            run += sendcounts[r];
        }
    }

    if (rank == 0) {
        std::vector<Complex> v_natural(static_cast<std::size_t>(global_dim));
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_natural[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_natural) z *= inv;

        std::vector<Complex> v_rankmajor(
            static_cast<std::size_t>(global_dim));
        for (int r = 0; r < size; ++r) {
            for (std::size_t k = 0; k < partition.rank_orbits[r].size(); ++k) {
                const std::size_t orbit_id = partition.rank_orbits[r][k];
                const std::size_t global_pos = partition.rank_offsets[r] + k;
                v_rankmajor[global_pos] = v_natural[orbit_id];
            }
        }
        MPI_Scatterv(v_rankmajor.data(), sendcounts.data(), displs.data(),
                     MPI_C_DOUBLE_COMPLEX,
                     v_local.data(), sendcounts[rank], MPI_C_DOUBLE_COMPLEX,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     MPI_C_DOUBLE_COMPLEX,
                     v_local.data(), sendcounts[rank], MPI_C_DOUBLE_COMPLEX,
                     0, op.comm());
    }

    const double n2 = dist_norm(v_local.data(), local_n, op.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, v_local.data(), local_n);
}

// Reorthogonalise w against the locked Ritz vectors and the in-cycle Krylov
// basis. Two CGS passes ("twice-is-enough") for robustness; both passes
// happen with the same `aux` set on every rank, so the cost per pass is
// (|aux| + |basis|) Allreduces.
void reorth_against(const std::vector<std::vector<Complex>>& aux,
                    const std::vector<std::vector<Complex>>& basis,
                    Complex* w,
                    std::uint64_t local_n,
                    MPI_Comm comm) {
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& v : aux) {
            const Complex c = dist_zdotc(v.data(), w, local_n, comm);
            local_axpy(-c, v.data(), w, local_n);
        }
        for (const auto& v : basis) {
            const Complex c = dist_zdotc(v.data(), w, local_n, comm);
            local_axpy(-c, v.data(), w, local_n);
        }
    }
}

// Reconstruct phi_local = sum_j evec_j * basis[j] (purely local op).
void reconstruct_local(const std::vector<std::vector<Complex>>& basis,
                       const double* evec_col,
                       std::uint64_t local_n,
                       std::vector<Complex>& phi) {
    phi.assign(local_n, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < basis.size(); ++j) {
        local_axpy(Complex(evec_col[j], 0.0), basis[j].data(),
                   phi.data(), local_n);
    }
}

}  // namespace

namespace {

// Templated KS body. `OpT` must satisfy the same duck-typed surface that
// `kernel::distributed_lanczos_kernel` consumes (apply, rank, comm_size,
// comm, global_dim, local_size). Both `DistributedOperator` and
// `DistributedSymmetryOperator` qualify; the only operator-specific
// hook is `scatter_initial_vector(op, seed, v_local)`, which is
// overload-resolved on the operator type.
//
// The `lanczos_fallback` callable handles the tiny-problem fallback to
// the matching baseline kernel without forcing this template to know
// anything about `distributed_lanczos` vs `distributed_lanczos_symmetry`.
template <typename OpT, typename LanczosFallback>
DistributedLanczosResult distributed_krylov_schur_impl(
    const OpT& op,
    const DistributedLanczosOptions& options,
    LanczosFallback&& lanczos_fallback) {

    const int rank = op.rank();
    const std::uint64_t local_n = op.local_size();
    const std::uint64_t global_dim = op.global_dim();
    const std::uint64_t k_target = std::max<std::uint64_t>(1, options.exct);
    const std::uint64_t m_max =
        std::min<std::uint64_t>(options.max_iter,
                                 std::max<std::uint64_t>(global_dim, 1));
    const double tol = options.tol;

    if (m_max == 0) {
        throw std::invalid_argument(
            "distributed_krylov_schur: max_iter == 0");
    }

    // For tiny problems where we'd "convergence" trivially, defer to the
    // baseline kernel (avoids an awkward edge case where the basis is
    // smaller than the requested k_target).
    if (k_target * 2 + 4 > m_max || m_max < 8) {
        return lanczos_fallback(op, options);
    }

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(local_n));

    // Initial seed: random, scattered, normalised.
    std::vector<Complex> v_seed_local;
    scatter_initial_vector(op, options.seed, v_seed_local);

    std::vector<std::vector<Complex>> locked_vecs;
    std::vector<double> locked_evals;
    locked_vecs.reserve(k_target);
    locked_evals.reserve(k_target);

    constexpr int kMaxRestarts = 30;
    int total_iters = 0;

    for (int restart = 0; restart < kMaxRestarts; ++restart) {
        // Re-orthogonalise the seed against locked Ritz vectors.
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& lv : locked_vecs) {
                const Complex c = dist_zdotc(lv.data(), v_seed_local.data(),
                                              local_n, op.comm());
                local_axpy(-c, lv.data(), v_seed_local.data(), local_n);
            }
        }
        const double seed_norm = dist_norm(v_seed_local.data(), local_n,
                                            op.comm());
        if (seed_norm < 1e-13) {
            scatter_initial_vector(op,
                                    options.seed + 1u + 7919u * restart,
                                    v_seed_local);
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& lv : locked_vecs) {
                    const Complex c = dist_zdotc(lv.data(),
                                                  v_seed_local.data(),
                                                  local_n, op.comm());
                    local_axpy(-c, lv.data(), v_seed_local.data(), local_n);
                }
            }
            const double n2 = dist_norm(v_seed_local.data(), local_n,
                                         op.comm());
            if (n2 < 1e-13) break;
            local_scal(1.0 / n2, v_seed_local.data(), local_n);
        } else {
            local_scal(1.0 / seed_norm, v_seed_local.data(), local_n);
        }

        std::vector<std::vector<Complex>> basis;
        basis.reserve(m_max);
        basis.push_back(v_seed_local);

        std::vector<Complex> v_curr = v_seed_local;
        std::vector<Complex> v_prev(local_n, Complex(0.0, 0.0));
        std::vector<Complex> w(local_n, Complex(0.0, 0.0));

        std::vector<double> alpha; alpha.reserve(m_max);
        std::vector<double> beta;  beta.reserve(m_max + 1);
        beta.push_back(0.0);

        std::uint64_t iters_done_cycle = 0;
        for (std::uint64_t j = 0; j < m_max; ++j) {
            op.apply(v_curr.data(), w.data());
            ++total_iters;

            const Complex alpha_c =
                dist_zdotc(v_curr.data(), w.data(), local_n, op.comm());
            alpha.push_back(alpha_c.real());

            local_axpy(Complex(-alpha.back(), 0.0), v_curr.data(),
                       w.data(), local_n);
            if (j > 0) {
                local_axpy(Complex(-beta.back(), 0.0), v_prev.data(),
                           w.data(), local_n);
            }

            reorth_against(locked_vecs, basis, w.data(), local_n, op.comm());

            const double b = dist_norm(w.data(), local_n, op.comm());
            beta.push_back(b);
            ++iters_done_cycle;

            if (options.verbose && rank == 0) {
                std::cout << "  [dist-ks] cycle=" << restart
                          << " j=" << j
                          << " alpha=" << alpha.back()
                          << " beta_{j+1}=" << b
                          << " locked=" << locked_evals.size()
                          << std::endl;
            }

            if (b < 1e-13) break;

            v_prev.swap(v_curr);
            local_scal(1.0 / b, w.data(), local_n);
            v_curr.swap(w);
            if (j + 1 < m_max) basis.push_back(v_curr);
        }

        if (alpha.empty()) break;

        std::vector<double> evals, weights, evecs_cm;
        solve_tridiag_with_eigenvectors(alpha, beta, alpha.size(),
                                        evals, weights, evecs_cm);

        const std::size_t m_eff = alpha.size();
        const double beta_last = beta.back();

        std::vector<std::size_t> idx(m_eff);
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(),
                  [&](std::size_t a, std::size_t b) {
                      return evals[a] < evals[b];
                  });

        std::size_t newly_locked = 0;
        const std::uint64_t need = (k_target > locked_evals.size())
                                       ? k_target - locked_evals.size()
                                       : 0;
        for (std::size_t r = 0; r < std::min<std::size_t>(need, m_eff); ++r) {
            const std::size_t i = idx[r];
            const double residual =
                beta_last
                * std::abs(evecs_cm[i * m_eff + (m_eff - 1)]);
            if (residual < tol) {
                std::vector<Complex> phi;
                reconstruct_local(basis, &evecs_cm[i * m_eff],
                                  local_n, phi);
                for (int pass = 0; pass < 2; ++pass) {
                    for (const auto& lv : locked_vecs) {
                        const Complex c = dist_zdotc(lv.data(), phi.data(),
                                                      local_n, op.comm());
                        local_axpy(-c, lv.data(), phi.data(), local_n);
                    }
                }
                const double pn = dist_norm(phi.data(), local_n, op.comm());
                if (pn > 1e-14) {
                    local_scal(1.0 / pn, phi.data(), local_n);
                    locked_evals.push_back(evals[i]);
                    locked_vecs.push_back(std::move(phi));
                    ++newly_locked;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if (locked_evals.size() >= k_target) break;

        const std::size_t seed_rank =
            std::min<std::size_t>(newly_locked, m_eff - 1);
        const std::size_t i_seed = idx[seed_rank];
        std::vector<Complex> phi;
        reconstruct_local(basis, &evecs_cm[i_seed * m_eff], local_n, phi);
        v_seed_local = std::move(phi);

        if (iters_done_cycle == 0) break;
    }

    DistributedLanczosResult result;
    result.iterations = total_iters;

    std::vector<std::size_t> ord(locked_evals.size());
    std::iota(ord.begin(), ord.end(), std::size_t{0});
    std::sort(ord.begin(), ord.end(),
              [&](std::size_t a, std::size_t b) {
                  return locked_evals[a] < locked_evals[b];
              });
    result.eigenvalues.reserve(locked_evals.size());
    for (std::size_t i : ord) result.eigenvalues.push_back(locked_evals[i]);

    if (options.compute_eigenvectors) {
        const std::size_t n_eig = ord.size();
        result.krylov_basis_local.resize(n_eig);
        for (std::size_t k = 0; k < n_eig; ++k) {
            result.krylov_basis_local[k] = std::move(locked_vecs[ord[k]]);
        }
        result.tridiag_eigenvalues = result.eigenvalues;
        result.tridiag_weights.assign(n_eig, 1.0);
        result.tridiag_eigenvectors.assign(n_eig * n_eig, 0.0);
        for (std::size_t k = 0; k < n_eig; ++k) {
            result.tridiag_eigenvectors[k * n_eig + k] = 1.0;
        }
    }

    return result;
}

}  // namespace

DistributedLanczosResult distributed_krylov_schur(
    const DistributedOperator& op,
    const DistributedLanczosOptions& options) {
    return distributed_krylov_schur_impl(
        op, options,
        [](const DistributedOperator& o,
           const DistributedLanczosOptions& opts) {
            return distributed_lanczos(o, opts);
        });
}

DistributedLanczosResult distributed_krylov_schur_symmetry(
    const DistributedSymmetryOperator& op,
    const DistributedLanczosOptions& options) {
    return distributed_krylov_schur_impl(
        op, options,
        [](const DistributedSymmetryOperator& o,
           const DistributedLanczosOptions& opts) {
            return distributed_lanczos_symmetry(o, opts);
        });
}

}  // namespace ed::distributed

#endif  // WITH_MPI
