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

#include <ed/core/construct_ham.h>
#include <ed/parallel/thread_budget.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::distributed {

namespace {

// Not constexpr: OpenMPI predefined handles cast through (void*) at runtime.
const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

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

    std::vector<double> N_Z_local(betas.size(), 0.0);
    std::vector<double> N_O_local(betas.size(), 0.0);

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

    std::vector<Complex> u_local;
    std::vector<Complex> q;
    std::vector<double>  g_b;
    std::vector<double>  f_b;

    for (int s : my_samples) {
        lopts.seed = options.seed_offset + static_cast<unsigned long>(s);

        DistributedLanczosResult lres = lanczos_call(dop, lopts);

        if (lres.tridiag_eigenvalues.empty()) continue;

        const std::size_t m = lres.tridiag_eigenvalues.size();

        for (std::size_t b = 0; b < betas.size(); ++b) {
            double zk = 0.0;
            const double beta = betas[b];
            for (std::size_t k = 0; k < m; ++k) {
                zk += lres.tridiag_weights[k]
                      * std::exp(-beta * lres.tridiag_eigenvalues[k]);
            }
            N_Z_local[b] += zk;
        }

        if (compute_obs) {
            if (lres.krylov_basis_local.size() != m ||
                lres.tridiag_eigenvectors.size() != m * m) {
                continue;
            }

            u_local.assign(local_n, Complex(0.0, 0.0));
            dop_O->apply(lres.krylov_basis_local[0].data(), u_local.data());

            q.assign(m, Complex(0.0, 0.0));
            std::vector<double> q_re(m, 0.0), q_im(m, 0.0);
            for (std::size_t j = 0; j < m; ++j) {
                Complex local(0.0, 0.0);
                const auto& Vj = lres.krylov_basis_local[j];
                for (std::size_t i = 0; i < local_n; ++i) {
                    local += std::conj(Vj[i]) * u_local[i];
                }
                q_re[j] = local.real();
                q_im[j] = local.imag();
            }
            std::vector<double> q_buf(2 * m, 0.0);
            for (std::size_t j = 0; j < m; ++j) {
                q_buf[2 * j]     = q_re[j];
                q_buf[2 * j + 1] = q_im[j];
            }
            std::vector<double> q_red(2 * m, 0.0);
            MPI_Allreduce(q_buf.data(), q_red.data(), static_cast<int>(2 * m),
                          MPI_DOUBLE, MPI_SUM, group_comm);
            for (std::size_t j = 0; j < m; ++j) {
                q[j] = Complex(q_red[2 * j], q_red[2 * j + 1]);
            }

            const double* U = lres.tridiag_eigenvectors.data();
            const double* E = lres.tridiag_eigenvalues.data();
            g_b.assign(m, 0.0);
            f_b.assign(m, 0.0);
            for (std::size_t b = 0; b < betas.size(); ++b) {
                const double beta = betas[b];
                for (std::size_t k = 0; k < m; ++k) {
                    g_b[k] = U[k * m + 0] * std::exp(-beta * E[k]);
                }
                for (std::size_t j = 0; j < m; ++j) {
                    double f = 0.0;
                    for (std::size_t k = 0; k < m; ++k) {
                        f += U[k * m + j] * g_b[k];
                    }
                    f_b[j] = f;
                }
                double contrib = 0.0;
                for (std::size_t j = 0; j < m; ++j) {
                    contrib += q[j].real() * f_b[j];
                }
                N_O_local[b] += contrib;
            }
        }

        if (options.verbose && world_rank == 0) {
            std::cout << "  [dist-ftlm] sample s=" << s
                      << " group=" << my_group
                      << " E0=" << lres.eigenvalues.front()
                      << " iters=" << lres.iterations
                      << (compute_obs ? "  (with O)" : "")
                      << std::endl;
        }
    }

    if (world_rank % ranks_per_group != 0) {
        std::fill(N_Z_local.begin(), N_Z_local.end(), 0.0);
        std::fill(N_O_local.begin(), N_O_local.end(), 0.0);
    }
    std::vector<double> N_Z(betas.size(), 0.0);
    std::vector<double> N_O(betas.size(), 0.0);
    MPI_Allreduce(N_Z_local.data(), N_Z.data(),
                  static_cast<int>(betas.size()),
                  MPI_DOUBLE, MPI_SUM, world_comm);
    if (compute_obs) {
        MPI_Allreduce(N_O_local.data(), N_O.data(),
                      static_cast<int>(betas.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
    }

    const double DoverR = D / static_cast<double>(n_samples);
    std::vector<double> Z(betas.size(), 0.0);
    for (std::size_t b = 0; b < betas.size(); ++b) {
        Z[b] = DoverR * N_Z[b];
    }

    std::vector<double> O_expectation;
    if (compute_obs) {
        O_expectation.assign(betas.size(), 0.0);
        for (std::size_t b = 0; b < betas.size(); ++b) {
            O_expectation[b] = (N_Z[b] > 0.0) ? (N_O[b] / N_Z[b]) : 0.0;
        }
    }

    DistributedFtlmResult result;
    result.Z = std::move(Z);
    result.O_expectation = std::move(O_expectation);
    result.samples_used = n_samples;
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
