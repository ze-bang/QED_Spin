// =============================================================================
// src/distributed/distributed_ftlm.cpp
//
// Phase 3b #3 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/distributed_symmetry_operator.h>

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/construct_ham.h>
#include <ed/parallel/thread_budget.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::distributed {

namespace {

// Not constexpr: OpenMPI predefined handles cast through (void*) at runtime.
const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

// Per-sample state cached during the Lanczos pass and consumed by the
// post-pass thermo accumulation. Holding this in memory across all
// owned samples lets us (a) determine the global E_shift via MPI
// Allreduce *before* evaluating any exp(-beta E_k) so the exponentials
// are O(1), and (b) replace the old O(m^2 * B) triple loop with one
// dgemm per sample.
//
// Memory budget per sample: m doubles for E, m for w, m*m for U, plus
// (if observable_op) 2m doubles for q. Even at m=400 and 32 samples
// this is ~10 MB per group rank.
struct SampleCache {
    int                 sample_id   = -1;
    std::size_t         m           = 0;
    std::vector<double> E;                  // tridiag eigenvalues  (m)
    std::vector<double> w;                  // tridiag weights      (m)
    std::vector<double> U;                  // eigvecs, k-major     (m*m)
    std::vector<double> q_re;               // <V_j | O r>, real    (m)
};

// Templated FTLM body. `Op` must satisfy the duck-typed surface
// `{apply(const Complex*, Complex*), local_size(), global_dim(),
// rank(), comm_size(), comm()}` -- both `DistributedOperator` and
// `DistributedSymmetryOperator` qualify. The `lanczos_call` callable
// is invoked once per sample and routes to the matching baseline
// (`distributed_lanczos` vs `distributed_lanczos_symmetry`).
template <typename Op, typename LanczosCall>
DistributedFtlmResult ftlm_impl(
    Op& dop,
    Op* dop_O,
    int world_rank,
    int my_group,
    int n_groups,
    int ranks_per_group,
    MPI_Comm world_comm,
    MPI_Comm group_comm,
    const DistributedFtlmOptions& options,
    LanczosCall&& lanczos_call) {

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(dop.local_size()));

    const bool compute_obs = (dop_O != nullptr);

    const int n_samples = std::max(1, options.n_samples);
    std::vector<int> my_samples;
    my_samples.reserve(n_samples / n_groups + 1);
    for (int s = 0; s < n_samples; ++s) {
        if ((s % n_groups) == my_group) my_samples.push_back(s);
    }

    std::vector<double> betas = options.betas;
    if (betas.empty()) betas.push_back(1.0);
    const std::size_t B = betas.size();

    DistributedLanczosOptions lopts;
    lopts.max_iter        = options.lanczos_max_iter;
    lopts.exct            = options.lanczos_max_iter;
    lopts.full_reorth     = true;
    lopts.compute_weights = true;
    if (compute_obs) {
        lopts.compute_eigenvectors = true;
    }

    const double D = static_cast<double>(dop.global_dim());
    const std::size_t local_n = static_cast<std::size_t>(dop.local_size());

    // -------- Pass 1: per-sample Lanczos + cache, track local min(E) --------
    std::vector<SampleCache> cache;
    cache.reserve(my_samples.size());
    double local_min_E = std::numeric_limits<double>::infinity();

    std::vector<Complex> u_local;

    for (int s : my_samples) {
        lopts.seed = options.seed_offset + static_cast<unsigned long>(s);

        DistributedLanczosResult lres = lanczos_call(dop, lopts);
        if (lres.tridiag_eigenvalues.empty()) continue;

        SampleCache c;
        c.sample_id = s;
        c.m         = lres.tridiag_eigenvalues.size();
        c.E         = std::move(lres.tridiag_eigenvalues);
        c.w         = std::move(lres.tridiag_weights);

        if (!c.E.empty()) {
            local_min_E = std::min(local_min_E, c.E.front());
        }

        if (compute_obs) {
            // Need eigenvectors and basis vectors for the observable path.
            if (lres.krylov_basis_local.size() != c.m ||
                lres.tridiag_eigenvectors.size() != c.m * c.m) {
                continue;
            }
            c.U = std::move(lres.tridiag_eigenvectors);

            // u_local = O |V_0> (the seed vector).
            u_local.assign(local_n, Complex(0.0, 0.0));
            dop_O->apply(lres.krylov_basis_local[0].data(), u_local.data());

            // q_j = <V_j | u> = sum_i conj(V_j[i]) * u[i].
            // FTLM for Hermitian H has real Krylov coordinates; the
            // imaginary part of q is statistical noise and was already
            // dropped downstream via q[j].real() in the legacy code.
            // We still Allreduce both parts so the algebra is identical
            // to the previous implementation.
            std::vector<double> q_buf(2 * c.m, 0.0);
            for (std::size_t j = 0; j < c.m; ++j) {
                Complex local(0.0, 0.0);
                const auto& Vj = lres.krylov_basis_local[j];
                for (std::size_t i = 0; i < local_n; ++i) {
                    local += std::conj(Vj[i]) * u_local[i];
                }
                q_buf[2 * j]     = local.real();
                q_buf[2 * j + 1] = local.imag();
            }
            std::vector<double> q_red(2 * c.m, 0.0);
            MPI_Allreduce(q_buf.data(), q_red.data(),
                          static_cast<int>(2 * c.m),
                          MPI_DOUBLE, MPI_SUM, group_comm);
            c.q_re.assign(c.m, 0.0);
            for (std::size_t j = 0; j < c.m; ++j) c.q_re[j] = q_red[2 * j];
        }

        if (options.verbose && world_rank == 0) {
            std::cout << "  [dist-ftlm] sample s=" << s
                      << " group=" << my_group
                      << " E0="    << c.E.front()
                      << " m="     << c.m
                      << (compute_obs ? "  (with O)" : "")
                      << std::endl;
        }
        cache.push_back(std::move(c));
    }

    // -------- Global E_shift via Allreduce(MIN) on world_comm --------------
    double E_shift = 0.0;
    {
        double in = local_min_E;
        if (!std::isfinite(in)) in = std::numeric_limits<double>::infinity();
        MPI_Allreduce(&in, &E_shift, 1, MPI_DOUBLE, MPI_MIN, world_comm);
        // If every sample failed the Lanczos guard (no caches anywhere),
        // E_shift is +inf; fall back to 0.0 so subsequent exp() are sane.
        if (!std::isfinite(E_shift)) E_shift = 0.0;
    }

    // -------- Pass 2: shifted-exponent Z and observable accumulation -------
    std::vector<double> N_Z_local(B, 0.0);
    std::vector<double> N_O_local(B, 0.0);

    // Scratch reused across samples.
    std::vector<double> G;     // (m x B) row-major:  G[k*B + b] = U[0,k] * exp(-beta_b * (E_k - E_shift))
    std::vector<double> F;     // (m x B) row-major:  F = U^T * G

    for (const auto& c : cache) {
        const std::size_t m = c.m;
        // --- Z contribution: zk(s, b) = sum_k w_k * exp(-beta(E_k - E_shift))
        for (std::size_t b = 0; b < B; ++b) {
            const double beta = betas[b];
            double zk = 0.0;
            for (std::size_t k = 0; k < m; ++k) {
                zk += c.w[k] * std::exp(-beta * (c.E[k] - E_shift));
            }
            N_Z_local[b] += zk;
        }

        if (!compute_obs || c.U.empty() || c.q_re.empty()) continue;

        // --- Observable contribution: contrib(s, b) = q^T * (U^T * G_b)
        // where G[k,b] = U[k * m + 0] * exp(-beta(E_k - E_shift)) (legacy
        // index U[k * m + j] is row-major eigenvector-major).
        G.assign(m * B, 0.0);
        for (std::size_t k = 0; k < m; ++k) {
            const double Uk0 = c.U[k * m + 0];
            const double Ek_shift = c.E[k] - E_shift;
            for (std::size_t b = 0; b < B; ++b) {
                G[k * B + b] = Uk0 * std::exp(-betas[b] * Ek_shift);
            }
        }

        // F = U^T * G   ((m x m)^T * (m x B) -> (m x B))
        F.assign(m * B, 0.0);
        if (m > 0 && B > 0) {
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        static_cast<int>(m), static_cast<int>(B), static_cast<int>(m),
                        1.0,
                        c.U.data(), static_cast<int>(m),
                        G.data(),    static_cast<int>(B),
                        0.0,
                        F.data(),    static_cast<int>(B));
        }

        // contrib_b = q^T * F[:, b]  (== dgemv on F with q)
        for (std::size_t b = 0; b < B; ++b) {
            double s = 0.0;
            for (std::size_t j = 0; j < m; ++j) {
                s += c.q_re[j] * F[j * B + b];
            }
            N_O_local[b] += s;
        }
    }

    // Only one rank per group contributes (groups are replicated across
    // ranks_per_group ranks via the slab-decomposed inner Lanczos); zero
    // out the others before the world reduction.
    if (world_rank % ranks_per_group != 0) {
        std::fill(N_Z_local.begin(), N_Z_local.end(), 0.0);
        std::fill(N_O_local.begin(), N_O_local.end(), 0.0);
    }
    std::vector<double> N_Z(B, 0.0);
    std::vector<double> N_O(B, 0.0);
    MPI_Allreduce(N_Z_local.data(), N_Z.data(),
                  static_cast<int>(B), MPI_DOUBLE, MPI_SUM, world_comm);
    if (compute_obs) {
        MPI_Allreduce(N_O_local.data(), N_O.data(),
                      static_cast<int>(B), MPI_DOUBLE, MPI_SUM, world_comm);
    }

    const double DoverR = D / static_cast<double>(n_samples);
    std::vector<double> Z(B, 0.0);
    std::vector<double> lnZ(B, -std::numeric_limits<double>::infinity());
    for (std::size_t b = 0; b < B; ++b) {
        // Z(beta) = (D/R) * exp(-beta * E_shift) * N_Z[b]
        Z[b] = DoverR * std::exp(-betas[b] * E_shift) * N_Z[b];
        if (N_Z[b] > 0.0) {
            lnZ[b] = std::log(DoverR) - betas[b] * E_shift + std::log(N_Z[b]);
        }
    }

    std::vector<double> O_expectation;
    if (compute_obs) {
        O_expectation.assign(B, 0.0);
        for (std::size_t b = 0; b < B; ++b) {
            // The exp(-beta E_shift) factor cancels in N_O / N_Z because
            // both numerator and denominator carry it; we never multiplied
            // it back in either accumulator above, so the ratio is direct.
            O_expectation[b] = (N_Z[b] > 0.0) ? (N_O[b] / N_Z[b]) : 0.0;
        }
    }

    DistributedFtlmResult result;
    result.Z             = std::move(Z);
    result.lnZ           = std::move(lnZ);
    result.E_shift       = E_shift;
    result.O_expectation = std::move(O_expectation);
    result.samples_used  = n_samples;
    return result;
}

}  // namespace

DistributedFtlmResult distributed_ftlm(
    std::shared_ptr<class ::Operator> op,
    const DistributedFtlmOptions& options,
    MPI_Comm world_comm) {

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) {
        n_groups = world_size;
    }
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_ftlm: n_groups (" + std::to_string(n_groups)
            + ") must divide world_size (" + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    DistributedOperator dop(op, group_comm);
    std::unique_ptr<DistributedOperator> dop_O;
    if (options.observable_op) {
        dop_O = std::make_unique<DistributedOperator>(
            options.observable_op, group_comm);
    }

    DistributedFtlmResult result = ftlm_impl(
        dop, dop_O.get(),
        world_rank, my_group, n_groups, ranks_per_group,
        world_comm, group_comm, options,
        [](DistributedOperator& d, const DistributedLanczosOptions& lo) {
            return distributed_lanczos(d, lo);
        });

    MPI_Comm_free(&group_comm);
    return result;
}

DistributedFtlmResult distributed_ftlm_symmetry(
    std::shared_ptr<class ::Operator> op,
    std::size_t sector_idx,
    const DistributedFtlmOptions& options,
    MPI_Comm world_comm) {

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) {
        n_groups = world_size;
    }
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_ftlm_symmetry: n_groups (" + std::to_string(n_groups)
            + ") must divide world_size (" + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    DistributedSymmetryOperator dop(op, sector_idx, group_comm);
    std::unique_ptr<DistributedSymmetryOperator> dop_O;
    if (options.observable_op) {
        dop_O = std::make_unique<DistributedSymmetryOperator>(
            options.observable_op, sector_idx, group_comm);
    }

    DistributedFtlmResult result = ftlm_impl(
        dop, dop_O.get(),
        world_rank, my_group, n_groups, ranks_per_group,
        world_comm, group_comm, options,
        [](DistributedSymmetryOperator& d,
           const DistributedLanczosOptions& lo) {
            return distributed_lanczos_symmetry(d, lo);
        });

    MPI_Comm_free(&group_comm);
    return result;
}

}  // namespace ed::distributed

#endif  // WITH_MPI
