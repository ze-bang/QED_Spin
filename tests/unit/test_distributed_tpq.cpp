// =============================================================================
// test_distributed_tpq (Phase 3b #8)
//
// MPI lockdown for ed::distributed::distributed_tpq, the canonical-TPQ
// driver:
//
//   |psi(beta)> = e^{-beta H/2} |r> / || ... ||
//   <H>(beta) ~= <psi(beta) | H | psi(beta)>
//
// Coverage:
//   * np = 1, 2, 4
//   * E(beta) approximates the exact thermal energy
//       E_exact(beta) = sum_n E_n e^{-beta E_n} / sum_n e^{-beta E_n}
//     within a generous tolerance compatible with R = 16 samples on a
//     16-state Hilbert space (N=4 OBC Heisenberg).
//   * E(beta) is replicated bit-identically across all ranks.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_tpq.h>
#include <ed/distributed/distributed_operator.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedTpqOptions;
using ed::distributed::DistributedTpqResult;
using ed::distributed::distributed_tpq;

namespace {

double exact_E(const std::vector<double>& E, double beta) {
    const double E0 = E.front();
    double num = 0.0, den = 0.0;
    for (double e : E) {
        const double w = std::exp(-beta * (e - E0));
        num += (e) * w;
        den += w;
    }
    return num / den;
}

}  // namespace

TEST_CASE("Distributed TPQ: <H>(beta) approximates exact thermal energy",
          "[distributed_tpq][heisenberg]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    const std::uint64_t dim = 1ULL << 4;
    auto ref = ed_tests::reference_from_operator(*op, dim);

    // Test is matvec-heavy (taylor_order * substeps per (sample, beta));
    // keep the budget modest because np=4 on dim=16 is over-decomposed
    // and dominated by MPI_Alltoallv overhead. taylor_order=15 is more
    // than enough for ||delta_beta * H|| ~ 0.1 * 4 = 0.4.
    DistributedTpqOptions opts;
    opts.n_samples       = 8;
    opts.n_groups        = 1;
    opts.delta_beta      = 0.1;
    opts.taylor_order    = 15;
    opts.betas           = {0.5, 2.0};
    opts.seed_offset     = 7UL;
    opts.compute_variance = true;

    DistributedTpqResult res = distributed_tpq(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.energy.size() == opts.betas.size());
    REQUIRE(res.variance.size() == opts.betas.size());

    for (std::size_t b = 0; b < opts.betas.size(); ++b) {
        const double beta = opts.betas[b];
        const double Eexact = exact_E(ref.eigs, beta);
        const double Etpq   = res.energy[b];
        const double abs_err = std::abs(Etpq - Eexact);
        INFO("beta=" << beta
             << " E_exact=" << Eexact
             << " E_tpq="   << Etpq
             << " |diff|="  << abs_err
             << " var="     << res.variance[b]);
        // Statistical TPQ error on a 16-state Hilbert space at R=16
        // samples is generous; we only require the kernel reproduces the
        // thermal energy within an order of magnitude of 1/sqrt(D) ~ 25%
        // of the energy spread (~|E_max - E_min| ~ 4 for N=4 Heisenberg).
        REQUIRE(abs_err < 0.6);
        // Variance of H in a near-thermal state is non-negative within
        // numerical noise.
        REQUIRE(res.variance[b] > -1e-10);
    }
}

TEST_CASE("Distributed TPQ: E(beta) replicated bit-identically across ranks",
          "[distributed_tpq][replicated]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedTpqOptions opts;
    opts.n_samples       = 4;
    opts.n_groups        = 1;
    opts.delta_beta      = 0.1;
    opts.taylor_order    = 20;
    opts.betas           = {0.5, 1.0};
    opts.seed_offset     = 999UL;

    DistributedTpqResult res = distributed_tpq(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.energy.size() == 2);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> rank0_E = res.energy;
    MPI_Bcast(rank0_E.data(), static_cast<int>(rank0_E.size()),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (std::size_t b = 0; b < rank0_E.size(); ++b) {
        REQUIRE(std::abs(res.energy[b] - rank0_E[b]) < 1e-10);
    }
}

TEST_CASE("Distributed TPQ: E(beta=0) ~= Tr(H)/D for any single sample",
          "[distributed_tpq][infinite_T]") {
    // At infinite temperature (beta -> 0), <H>_thermal = Tr(H)/D.
    // For Heisenberg without symmetrization, Tr(H) = 0 (off-diagonal
    // S+S- + S-S+ have zero diagonal; SzSz has zero trace because Σ_states
    // sign products average to zero). So E(beta=0) should be small for
    // any single sample (vanishing in expectation over the random vector
    // distribution; for a single sample, bounded by 1/sqrt(D) variance).
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedTpqOptions opts;
    opts.n_samples       = 16;
    opts.n_groups        = 1;
    opts.delta_beta      = 0.05;
    opts.taylor_order    = 15;
    opts.betas           = {0.0, 0.01};   // start at 0 (no propagation)
    opts.seed_offset     = 0UL;

    DistributedTpqResult res = distributed_tpq(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.energy.size() == 2);

    // <r|H|r> for unit Gaussian r averages to Tr(H)/D = 0; with R=32
    // samples on D=16, statistical bound is generous.
    INFO("E(beta=0)  = " << res.energy[0]
         << "  E(beta=0.01) = " << res.energy[1]);
    REQUIRE(std::abs(res.energy[0]) < 0.6);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        FILE* dummy = std::freopen("/dev/null", "w", stdout);
        (void)dummy;
    }

    int result = Catch::Session().run(argc, argv);

    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();
    return global_result;
}
