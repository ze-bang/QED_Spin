// =============================================================================
// src/distributed/distributed_tpq.cpp
//
// Phase 3b #8 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_tpq.h>
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

// Apply e^{-(delta/2) H} to psi_local via Taylor truncation:
//   result = sum_{n=0}^{taylor_order} (-(delta/2))^n H^n psi / n!
// then renormalise.  All matvecs are DistributedOperator.apply (one
// MPI_Alltoallv per matvec) and all axpys are rank-local.
void taylor_step(const DistributedOperator& dop,
                 std::vector<Complex>& psi_local,
                 double delta,
                 std::uint64_t taylor_order) {
    const std::uint64_t local_n = dop.local_size();
    std::vector<Complex> term(psi_local);   // term_0 = psi
    std::vector<Complex> Hterm(local_n, Complex(0.0, 0.0));
    std::vector<Complex> result(psi_local); // result accumulator

    double coef = 1.0;
    for (std::uint64_t order = 1; order <= taylor_order; ++order) {
        // term <- H * term  (one Alltoallv + local matvec)
        dop.apply(term.data(), Hterm.data());
        std::swap(term, Hterm);

        coef *= -(delta / 2.0) / static_cast<double>(order);
        local_axpy(Complex(coef, 0.0), term.data(), result.data(), local_n);

        // Cheap stability bail-out: if |coef| * ||term|| underflows below
        // 1e-30, further terms are noise. Skip the work for the next
        // iteration; we use a lazy estimate that ||term|| <= ||H||^order
        // (bounded by ||H|| ~ O(N) for spin Hamiltonians at our test
        // sizes), so we just check the coefficient size.
        if (std::abs(coef) < 1e-30) break;
    }

    psi_local.swap(result);

    // Renormalise on the slab (Allreduce).
    const double n2 = dist_norm(psi_local.data(), local_n, dop.comm());
    if (n2 > 0.0) local_scal(1.0 / n2, psi_local.data(), local_n);
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

    // Phase 8 #3: dim-aware OMP+BLAS thread cap, sized against the
    // rank-local slab. Same rationale as distributed_lanczos -- the
    // taylor_step inner loops (local_axpy, local_norm_sq, local_scal)
    // and the per-sample matvec packing are memory-bound and pay heavy
    // OpenBLAS / OMP setup cost on small-to-mid N. ED_AUTO_THREADS=0
    // disables the cap for users running their own pinning.
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
    // Defensive: enforce strict ascending order.
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

    std::vector<double> E_local(betas.size(), 0.0);
    std::vector<double> E2_local(betas.size(), 0.0);

    std::vector<Complex> psi_local;
    std::vector<Complex> Hpsi_local(local_n, Complex(0.0, 0.0));

    for (int s : my_samples) {
        scatter_initial_vector(dop,
                               options.seed_offset
                                   + static_cast<unsigned long>(s),
                               psi_local);

        double cur_beta = 0.0;

        // Special case: if betas[0] == 0, measure the initial state
        // directly without any propagation.
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double tgt = betas[b];

            // Advance from cur_beta to tgt in substeps of delta_beta,
            // ending with a (possibly shorter) final substep so we land
            // exactly on tgt.
            while (cur_beta + 0.5 * delta_beta < tgt) {
                const double remain = tgt - cur_beta;
                const double step   = std::min(delta_beta, remain);
                taylor_step(dop, psi_local, step, taylor_order);
                cur_beta += step;
            }
            // Snap-to (tiny adjustment if any rounding leftover) -- only
            // do it if the gap is non-negligible relative to delta_beta.
            if (std::abs(cur_beta - tgt) > 1e-12) {
                taylor_step(dop, psi_local, tgt - cur_beta, taylor_order);
                cur_beta = tgt;
            }

            // Measure E = <psi | H psi>.
            dop.apply(psi_local.data(), Hpsi_local.data());
            Complex Ec = dist_zdotc(psi_local.data(), Hpsi_local.data(),
                                    local_n, group_comm);
            const double E_b = Ec.real();
            E_local[b] += E_b;

            if (options.compute_variance) {
                // E2 = <psi | H^2 psi> = || H psi ||_2^2.
                const double Hpsi_n2 =
                    dist_norm(Hpsi_local.data(), local_n, group_comm);
                E2_local[b] += Hpsi_n2 * Hpsi_n2;
            }

            if (options.verbose && world_rank == 0) {
                std::cout << "  [dist-tpq] sample s=" << s
                          << " group=" << my_group
                          << " beta=" << tgt
                          << " E=" << E_b << std::endl;
            }
        }
    }

    // World-level reduce: only group rank 0 contributes to avoid double
    // counting (each rank in a group holds the same per-sample E because
    // dist_zdotc + dist_norm are collective on group_comm).
    if (world_rank % ranks_per_group != 0) {
        std::fill(E_local.begin(),  E_local.end(),  0.0);
        std::fill(E2_local.begin(), E2_local.end(), 0.0);
    }
    std::vector<double> E_global(betas.size(),  0.0);
    std::vector<double> E2_global(betas.size(), 0.0);
    MPI_Allreduce(E_local.data(),  E_global.data(),
                  static_cast<int>(betas.size()),
                  MPI_DOUBLE, MPI_SUM, world_comm);
    if (options.compute_variance) {
        MPI_Allreduce(E2_local.data(), E2_global.data(),
                      static_cast<int>(betas.size()),
                      MPI_DOUBLE, MPI_SUM, world_comm);
    }

    const double inv = 1.0 / static_cast<double>(n_samples);

    DistributedTpqResult result;
    result.energy.assign(betas.size(), 0.0);
    for (std::size_t b = 0; b < betas.size(); ++b) {
        result.energy[b] = E_global[b] * inv;
    }
    if (options.compute_variance) {
        result.variance.assign(betas.size(), 0.0);
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double E_b  = result.energy[b];
            const double E2_b = E2_global[b] * inv;
            result.variance[b] = E2_b - E_b * E_b;
        }
    }
    result.samples_used = n_samples;

    MPI_Comm_free(&group_comm);
    return result;
}

}  // namespace ed::distributed

#endif  // WITH_MPI
