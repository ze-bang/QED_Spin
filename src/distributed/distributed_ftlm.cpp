// =============================================================================
// src/distributed/distributed_ftlm.cpp
//
// Phase 3b #3 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>

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

    // Split world communicator into per-group subcomms. Each subcomm holds
    // `ranks_per_group` ranks (a slice of the row decomposition for that
    // group's DistributedOperator).
    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    DistributedOperator dop(op, group_comm);

    // Phase 8 #3: dim-aware OMP+BLAS thread cap. distributed_lanczos
    // (called per sample below) installs its own ThreadBudgetScope, but we
    // also cover the FTLM-level outer loop here -- the per-sample
    // observable contraction (length-m complex zdotc and length-m real
    // dot products in the f_b accumulation) and the dop_O.apply matvec
    // are bottlenecked on the same OpenBLAS pthread pool. Nesting two
    // scopes is fine; the inner restore on its destructor returns the
    // counts to whatever the outer scope set.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(dop.local_size()));

    const bool compute_obs = static_cast<bool>(options.observable_op);
    std::unique_ptr<DistributedOperator> dop_O;
    if (compute_obs) {
        dop_O = std::make_unique<DistributedOperator>(
            options.observable_op, group_comm);
    }

    // Distribute samples round-robin across groups: sample s belongs to
    // group s % n_groups.
    const int n_samples = std::max(1, options.n_samples);
    std::vector<int> my_samples;
    my_samples.reserve(n_samples / n_groups + 1);
    for (int s = 0; s < n_samples; ++s) {
        if ((s % n_groups) == my_group) my_samples.push_back(s);
    }

    std::vector<double> betas = options.betas;
    if (betas.empty()) betas.push_back(1.0);

    // Per-group accumulators (un-normalised; the (D/R) factor is applied
    // at the end). The Z denominator and the O numerator share the same
    // (D/R) scaling, which cancels in the ratio O_expectation = N_O / N_Z
    // -- but we keep the symmetric scaling so result.Z still matches the
    // canonical J&P partition function on its own.
    std::vector<double> N_Z_local(betas.size(), 0.0);  // sum_s sum_k w_k e^{-bE_k}
    std::vector<double> N_O_local(betas.size(), 0.0);  // sum_s Re(sum_j q_j^* f_j)

    DistributedLanczosOptions lopts;
    lopts.max_iter        = options.lanczos_max_iter;
    lopts.exct            = options.lanczos_max_iter;  // keep all Ritz values
    lopts.full_reorth     = true;   // FTLM weights need clean orthogonality
    lopts.compute_weights = true;
    if (compute_obs) {
        // Need the rank-local Krylov basis (V_local) and the m x m
        // tridiagonal eigenvector matrix (U) so we can form
        //   q_j = <V_s[j] | u>  where u = O r_s
        //   f_j(beta) = sum_k U[j,k] U[0,k] e^{-beta E_k}
        lopts.compute_eigenvectors = true;
    }

    // FTLM trace-estimator scaling: the standard J&P normalisation is
    //   Tr(A) ~ (D / R) * sum_s <r_s | A | r_s>
    // where D = global Hilbert-space dimension and R = number of random
    // samples. Our v0_s is L2-normalised to 1, so each <r_s|A|r_s>
    // expectation is Tr(A)/D in expectation -- restore the D/R prefactor
    // at the end.
    const double D = static_cast<double>(dop.global_dim());
    const std::size_t local_n = static_cast<std::size_t>(dop.local_size());

    // Per-sample workspace (allocated once; resized lazily inside the loop).
    std::vector<Complex> u_local;
    std::vector<Complex> q;             // length m, complex
    std::vector<double>  g_b;           // length m, real, recycled per beta
    std::vector<double>  f_b;           // length m, real, recycled per beta

    for (int s : my_samples) {
        lopts.seed = options.seed_offset + static_cast<unsigned long>(s);

        DistributedLanczosResult lres = distributed_lanczos(dop, lopts);

        if (lres.tridiag_eigenvalues.empty()) {
            // Recurrence broke down too early -- skip sample.
            continue;
        }

        const std::size_t m = lres.tridiag_eigenvalues.size();

        // ------------------------------------------------------------------
        // Z numerator: sum_k w_k(s) e^{-beta E_k(s)} per beta. (Same as
        // Phase 3b #3 modulo the (D/R) factor that we now apply outside.)
        // ------------------------------------------------------------------
        for (std::size_t b = 0; b < betas.size(); ++b) {
            double zk = 0.0;
            const double beta = betas[b];
            for (std::size_t k = 0; k < m; ++k) {
                zk += lres.tridiag_weights[k]
                      * std::exp(-beta * lres.tridiag_eigenvalues[k]);
            }
            N_Z_local[b] += zk;
        }

        // ------------------------------------------------------------------
        // O numerator: <r_s | O e^{-beta H} | r_s> via the J&P formula
        //   contribution(beta) = Re( sum_j q_j(s)^* f_j(s, beta) )
        //   q_j(s)        = <V_s[j] | u>,   u = O r_s = O V_s[0]
        //   f_j(s, beta)  = sum_k U[j,k] U[0,k] exp(-beta E_k)
        // ------------------------------------------------------------------
        if (compute_obs) {
            if (lres.krylov_basis_local.size() != m ||
                lres.tridiag_eigenvectors.size() != m * m) {
                // Defensive: should not happen because compute_eigenvectors
                // was set above; skip the observable update for this
                // sample if it ever does.
                continue;
            }

            // u_local = O * r_s_local = O * V_s[0]_local.
            u_local.assign(local_n, Complex(0.0, 0.0));
            dop_O->apply(lres.krylov_basis_local[0].data(), u_local.data());

            // q_j = <V_s[j] | u>, j = 0..m-1
            //   q_j_local = sum_i conj(V_s[j]_i) * u_i
            //   q_j       = MPI_Allreduce(q_j_local) over group_comm
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
            // Coalesce into a single Allreduce (2*m doubles, group_comm).
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

            // For each beta:
            //   g_k = U[0,k] * exp(-beta * E_k)   (length m, real)
            //   f_j = sum_k U[j,k] * g_k          (length m, real)
            //   contrib = Re( sum_j q_j^* f_j ) = sum_j Re(q_j) f_j
            //                                       (because f_j is real)
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

    // Reduce across groups via the world communicator.
    // Each rank in a group holds the SAME N_*_local (DistributedLanczos and
    // the per-sample dist-zdotc Allreduce both produce replicated values
    // within the group). To avoid double-counting, only rank 0 of each
    // group contributes to the world-level reduction.
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
            // <O>(beta) = N_O / N_Z. The (D/R) prefactor cancels in the
            // ratio, so we can use the un-normalised numerators directly.
            O_expectation[b] = (N_Z[b] > 0.0) ? (N_O[b] / N_Z[b]) : 0.0;
        }
    }

    MPI_Comm_free(&group_comm);

    DistributedFtlmResult result;
    result.Z = std::move(Z);
    result.O_expectation = std::move(O_expectation);
    result.samples_used = n_samples;
    return result;
}

}  // namespace ed::distributed

#endif  // WITH_MPI
