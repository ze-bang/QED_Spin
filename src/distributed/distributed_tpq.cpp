// =============================================================================
// src/distributed/distributed_tpq.cpp
//
// Phase 3b #8 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_tpq.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/distributed_symmetry_operator.h>

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

using Complex = std::complex<double>;

// Not constexpr: OpenMPI predefined handles cast through (void*) at runtime.
const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

// ----- rank-local BLAS-level helpers (mirror of distributed_lanczos.cpp) ----
inline double local_norm_sq(const Complex* x, std::uint64_t n) {
    double s = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) s += std::norm(x[i]);
    return s;
}

inline Complex local_zdotc(const Complex* x, const Complex* y, std::uint64_t n) {
    Complex s(0.0, 0.0);
    for (std::uint64_t i = 0; i < n; ++i) s += std::conj(x[i]) * y[i];
    return s;
}

inline void local_axpy(Complex a, const Complex* x, Complex* y, std::uint64_t n) {
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
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    }

    const double n2 = dist_norm(v_local.data(), local_n, op.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, v_local.data(), local_n);
}

// Symmetry-projected scatter (Phase E). Same algorithm as the orbit
// scatter helpers in distributed_lanczos_gpu.cu / distributed_ftlm_gpu.cu:
// generate a unit Gaussian vector in natural-orbit indexing on rank 0,
// permute into rank-major packed layout matching the partition, then
// MPI_Scatterv on op.comm(). Final renorm via dist_norm absorbs scatter
// rounding.
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
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    } else {
        MPI_Scatterv(nullptr, sendcounts.data(), displs.data(),
                     kComplexDatatype,
                     v_local.data(), sendcounts[rank], kComplexDatatype,
                     0, op.comm());
    }

    const double n2 = dist_norm(v_local.data(), local_n, op.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, v_local.data(), local_n);
}

// Apply e^{-(delta/2) H} to psi_local via Taylor truncation. Templated
// on the operator type: both `DistributedOperator` and
// `DistributedSymmetryOperator` expose
// `{apply(const Complex*, Complex*), local_size(), comm()}`.
//
// Returns ||result||  (i.e. the L2 norm of e^{-(delta/2)H}|psi> assuming
// |psi> entered the function with unit norm). The caller can use this to
// accumulate log w_r(beta) = log <r|e^{-beta H}|r> for the canonical-TPQ
// J&P estimator of Z(beta).
template <typename Op>
double taylor_step(const Op& dop,
                   std::vector<Complex>& psi_local,
                   double delta,
                   std::uint64_t taylor_order) {
    const std::uint64_t local_n = dop.local_size();
    std::vector<Complex> term(psi_local);
    std::vector<Complex> Hterm(local_n, Complex(0.0, 0.0));
    std::vector<Complex> result(psi_local);

    double coef = 1.0;
    for (std::uint64_t order = 1; order <= taylor_order; ++order) {
        dop.apply(term.data(), Hterm.data());
        std::swap(term, Hterm);

        coef *= -(delta / 2.0) / static_cast<double>(order);
        local_axpy(Complex(coef, 0.0), term.data(), result.data(), local_n);

        if (std::abs(coef) < 1e-30) break;
    }

    psi_local.swap(result);

    const double n2 = dist_norm(psi_local.data(), local_n, dop.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, psi_local.data(), local_n);
    return n2;
}

// Templated per-sample TPQ driver shared by the dense and symm paths.
// `Op` provides {apply(const Complex*, Complex*), local_size(),
// global_dim(), rank(), comm_size(), comm()}; `ScatterFn` is a callable
// `void(const Op&, unsigned long seed, std::vector<Complex>&)`.
template <typename Op, typename ScatterFn>
DistributedTpqResult tpq_impl(
    const Op& dop,
    int world_rank,
    int my_group,
    int n_groups,
    int ranks_per_group,
    MPI_Comm world_comm,
    MPI_Comm group_comm,
    const DistributedTpqOptions& options,
    ScatterFn&& scatter_fn) {

    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(dop.local_size()));

    const int n_samples = std::max(1, options.n_samples);
    std::vector<int> my_samples;
    my_samples.reserve(n_samples / n_groups + 1);
    for (int s = 0; s < n_samples; ++s) {
        if ((s % n_groups) == my_group) my_samples.push_back(s);
    }

    std::vector<double> betas = options.betas;
    if (betas.empty()) betas.push_back(1.0);
    for (std::size_t i = 1; i < betas.size(); ++i) {
        if (betas[i] <= betas[i - 1]) {
            throw std::invalid_argument(
                "distributed_tpq: options.betas must be strictly ascending");
        }
    }
    if (betas.front() < 0.0) {
        throw std::invalid_argument(
            "distributed_tpq: options.betas must be non-negative");
    }

    const double delta_beta = std::max(1e-12, options.delta_beta);
    const std::uint64_t taylor_order = std::max<std::uint64_t>(
        1, options.taylor_order);

    const std::size_t local_n = static_cast<std::size_t>(dop.local_size());
    const std::size_t B = betas.size();

    // Per-sample, per-beta storage on each rank. Only entries that this
    // rank's group owns are populated; everything else stays at 0 and the
    // world-level Allreduce consolidates them.
    //
    //   logw_per_sample[s*B + b] = log <r_s | e^{-beta_b H} | r_s>
    //   E_per_sample   [s*B + b] = <psi_s(beta_b) | H | psi_s(beta_b)>   (normalised)
    //   E2_per_sample  [s*B + b] = <psi_s(beta_b) | H^2 | psi_s(beta_b)> (normalised)
    //
    // For samples not owned by this group, all three remain zero. We then
    // mark non-owned (s,b) entries by carrying a separate "owned" mask
    // that we Allreduce as well; this lets us subtract the unowned zeros
    // from the logsumexp sentinel below.
    std::vector<double> logw_per_sample(static_cast<std::size_t>(n_samples) * B, 0.0);
    std::vector<double> E_per_sample   (static_cast<std::size_t>(n_samples) * B, 0.0);
    std::vector<double> E2_per_sample  (static_cast<std::size_t>(n_samples) * B, 0.0);
    std::vector<int>    owned_mask     (static_cast<std::size_t>(n_samples), 0);

    std::vector<Complex> psi_local;
    std::vector<Complex> Hpsi_local(local_n, Complex(0.0, 0.0));

    for (int s : my_samples) {
        scatter_fn(dop,
                   options.seed_offset + static_cast<unsigned long>(s),
                   psi_local);

        double cur_beta = 0.0;
        // log w_r(beta) accumulated as the sum of 2*log||result|| across
        // every Taylor step, since the state is unit-normalised entering
        // each step. Equivalently log <r|e^{-beta H}|r>.
        double log_w = 0.0;

        for (std::size_t b = 0; b < B; ++b) {
            const double tgt = betas[b];

            while (cur_beta + 0.5 * delta_beta < tgt) {
                const double remain = tgt - cur_beta;
                const double step   = std::min(delta_beta, remain);
                const double n2 = taylor_step(dop, psi_local, step, taylor_order);
                if (n2 > 0.0) log_w += 2.0 * std::log(n2);
                cur_beta += step;
            }
            if (std::abs(cur_beta - tgt) > 1e-12) {
                const double n2 = taylor_step(
                    dop, psi_local, tgt - cur_beta, taylor_order);
                if (n2 > 0.0) log_w += 2.0 * std::log(n2);
                cur_beta = tgt;
            }

            dop.apply(psi_local.data(), Hpsi_local.data());
            Complex Ec = dist_zdotc(psi_local.data(), Hpsi_local.data(),
                                    local_n, group_comm);
            const double E_b = Ec.real();

            const std::size_t idx =
                static_cast<std::size_t>(s) * B + b;
            logw_per_sample[idx] = log_w;
            E_per_sample[idx]    = E_b;

            if (options.compute_variance) {
                const double Hpsi_n = dist_norm(
                    Hpsi_local.data(), local_n, group_comm);
                E2_per_sample[idx] = Hpsi_n * Hpsi_n;
            }

            if (options.verbose && world_rank == 0) {
                std::cout << "  [dist-tpq] sample s=" << s
                          << " group=" << my_group
                          << " beta=" << tgt
                          << " E=" << E_b
                          << " logw=" << log_w << std::endl;
            }
        }
        owned_mask[s] = 1;
    }

    // Within a group only rank 0 contributes; null out the rest so the
    // world-level SUM Allreduce doesn't double-count.
    if (world_rank % ranks_per_group != 0) {
        std::fill(logw_per_sample.begin(), logw_per_sample.end(), 0.0);
        std::fill(E_per_sample.begin(),    E_per_sample.end(),    0.0);
        std::fill(E2_per_sample.begin(),   E2_per_sample.end(),   0.0);
        std::fill(owned_mask.begin(),      owned_mask.end(),      0);
    }
    {
        std::vector<double> tmp(logw_per_sample.size(), 0.0);
        MPI_Allreduce(logw_per_sample.data(), tmp.data(),
                      static_cast<int>(tmp.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
        logw_per_sample.swap(tmp);
    }
    {
        std::vector<double> tmp(E_per_sample.size(), 0.0);
        MPI_Allreduce(E_per_sample.data(), tmp.data(),
                      static_cast<int>(tmp.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
        E_per_sample.swap(tmp);
    }
    if (options.compute_variance) {
        std::vector<double> tmp(E2_per_sample.size(), 0.0);
        MPI_Allreduce(E2_per_sample.data(), tmp.data(),
                      static_cast<int>(tmp.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
        E2_per_sample.swap(tmp);
    }
    {
        std::vector<int> tmp(owned_mask.size(), 0);
        MPI_Allreduce(owned_mask.data(), tmp.data(),
                      static_cast<int>(tmp.size()),
                      MPI_INT, MPI_SUM, world_comm);
        owned_mask.swap(tmp);
    }

    const double D = static_cast<double>(dop.global_dim());
    const double R = static_cast<double>(n_samples);

    DistributedTpqResult result;
    result.energy.assign(B, 0.0);
    result.Z.assign(B, 0.0);
    result.lnZ.assign(B, 0.0);
    if (options.compute_variance) result.variance.assign(B, 0.0);

    for (std::size_t b = 0; b < B; ++b) {
        // logsumexp over owned samples.
        double max_lw = -std::numeric_limits<double>::infinity();
        for (int s = 0; s < n_samples; ++s) {
            if (!owned_mask[s]) continue;
            const double lw = logw_per_sample[static_cast<std::size_t>(s) * B + b];
            if (lw > max_lw) max_lw = lw;
        }
        if (!std::isfinite(max_lw)) {
            // No samples owned anywhere: degenerate. Leave zeros.
            continue;
        }
        double sum_w     = 0.0;
        double sum_w_E   = 0.0;
        double sum_w_E2  = 0.0;
        for (int s = 0; s < n_samples; ++s) {
            if (!owned_mask[s]) continue;
            const std::size_t idx =
                static_cast<std::size_t>(s) * B + b;
            const double w = std::exp(logw_per_sample[idx] - max_lw);
            sum_w    += w;
            sum_w_E  += w * E_per_sample[idx];
            if (options.compute_variance) {
                sum_w_E2 += w * E2_per_sample[idx];
            }
        }
        if (sum_w > 0.0) {
            result.energy[b] = sum_w_E / sum_w;
            if (options.compute_variance) {
                const double E_b  = result.energy[b];
                const double E2_b = sum_w_E2 / sum_w;
                result.variance[b] = E2_b - E_b * E_b;
            }
        }
        // Z(beta) = (D/R) * sum_r exp(log w_r) = (D/R) * exp(max_lw) * sum_w
        result.lnZ[b]  = std::log(D / R) + max_lw + std::log(sum_w);
        result.Z[b]    = std::exp(result.lnZ[b]);
    }
    result.samples_used = n_samples;
    return result;
}

}  // namespace

DistributedTpqResult distributed_tpq(
    std::shared_ptr<class ::Operator> op,
    const DistributedTpqOptions& options,
    MPI_Comm world_comm) {

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) n_groups = world_size;
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_tpq: n_groups (" + std::to_string(n_groups)
            + ") must divide world_size (" + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    DistributedOperator dop(op, group_comm);

    DistributedTpqResult result = tpq_impl(
        dop, world_rank, my_group, n_groups, ranks_per_group,
        world_comm, group_comm, options,
        [](const DistributedOperator& d, unsigned long seed,
           std::vector<Complex>& v_local) {
            scatter_initial_vector(d, seed, v_local);
        });

    MPI_Comm_free(&group_comm);
    return result;
}

DistributedTpqResult distributed_tpq_symmetry(
    std::shared_ptr<class ::Operator> op,
    std::size_t sector_idx,
    const DistributedTpqOptions& options,
    MPI_Comm world_comm) {

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) n_groups = world_size;
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_tpq_symmetry: n_groups ("
            + std::to_string(n_groups)
            + ") must divide world_size ("
            + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    DistributedSymmetryOperator dop(op, sector_idx, group_comm);

    DistributedTpqResult result = tpq_impl(
        dop, world_rank, my_group, n_groups, ranks_per_group,
        world_comm, group_comm, options,
        [](const DistributedSymmetryOperator& d, unsigned long seed,
           std::vector<Complex>& v_local) {
            scatter_initial_vector(d, seed, v_local);
        });

    MPI_Comm_free(&group_comm);
    return result;
}

}  // namespace ed::distributed

#endif  // WITH_MPI
