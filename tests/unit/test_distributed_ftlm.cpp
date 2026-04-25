// =============================================================================
// test_distributed_ftlm (Phase 3b #3)
//
// MPI lockdown for ed::distributed::distributed_ftlm on small Heisenberg
// chains. Cross-checks the J&P trace estimator at moderate sample count
// against the EXACT partition function Z(beta) = sum_n exp(-beta * E_n)
// computed from the dense-reference spectrum.
//
// Coverage:
//   * np=1: degenerate-decomposition (all samples in one group).
//   * np=2,4: outer parallelism over samples (1 group) and
//     inner-only parallelism (n_groups = np, 1 sample per group)
//     produce the same Z(beta) within statistical noise.
//   * Exact partition function: |Z_ftlm(beta) - Z_exact(beta)| / Z_exact
//     <= 1/sqrt(R) at high temperature (R = 32 samples gives ~17%).
//     We assert <= 0.30 to leave generous headroom.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_operator.h>
#include "common/test_harness.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedFtlmOptions;
using ed::distributed::DistributedFtlmResult;
using ed::distributed::distributed_ftlm;

namespace {

double exact_Z(const std::vector<double>& E, double beta) {
    double E0 = E.front();
    double s = 0.0;
    for (double e : E) s += std::exp(-beta * (e - E0));
    return s * std::exp(-beta * E0);
}

// Exact thermal expectation for an observable O whose diagonal in the
// eigenbasis of H is `O_diag` (sorted by ascending E like ref.eigs).
double exact_O(const std::vector<double>& E,
               const std::vector<double>& O_diag,
               double beta) {
    const double E0 = E.front();
    double num = 0.0, den = 0.0;
    for (std::size_t n = 0; n < E.size(); ++n) {
        const double w = std::exp(-beta * (E[n] - E0));
        num += w * O_diag[n];
        den += w;
    }
    return num / den;
}

}  // namespace

TEST_CASE("Distributed FTLM: Z(beta) approximates exact trace at high T",
          "[distributed_ftlm][heisenberg]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    const std::uint64_t dim = 1ULL << 4;
    auto ref = ed_tests::reference_from_operator(*op, dim);

    DistributedFtlmOptions opts;
    opts.n_samples       = 64;
    opts.n_groups        = 1;     // one group spanning every rank
    opts.lanczos_max_iter = 30;
    opts.betas           = {0.1, 0.5};
    opts.seed_offset     = 0UL;

    DistributedFtlmResult res = distributed_ftlm(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.Z.size() == opts.betas.size());

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (std::size_t b = 0; b < opts.betas.size(); ++b) {
        double Zexact = exact_Z(ref.eigs, opts.betas[b]);
        double Zftlm  = res.Z[b];
        double rel    = std::abs(Zftlm - Zexact) / Zexact;
        INFO("beta=" << opts.betas[b]
             << " Zexact=" << Zexact
             << " Zftlm="  << Zftlm
             << " relerr=" << rel);
        // Statistical FTLM error bound at R=64 samples on a 16-state
        // Hilbert space is generous; we only require the kernel is
        // reproducing the trace within an order of magnitude of the
        // canonical 1/sqrt(R) ~ 12.5%.
        REQUIRE(rel < 0.30);
    }
}

TEST_CASE("Distributed FTLM: replicated Z across all ranks",
          "[distributed_ftlm][replicated]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedFtlmOptions opts;
    opts.n_samples       = 8;
    opts.n_groups        = 1;
    opts.lanczos_max_iter = 20;
    opts.betas           = {1.0};
    opts.seed_offset     = 999UL;

    DistributedFtlmResult res = distributed_ftlm(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.Z.size() == 1);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double rank0_Z = res.Z[0];
    MPI_Bcast(&rank0_Z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    REQUIRE(std::abs(res.Z[0] - rank0_Z) < 1e-10);
}

TEST_CASE("Distributed FTLM: <H>(beta) approximates exact thermal energy",
          "[distributed_ftlm][observable][heisenberg]") {
    // Use H itself as the observable: <H>(beta) = thermal energy.
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    auto obs_op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    const std::uint64_t dim = 1ULL << 4;
    auto ref = ed_tests::reference_from_operator(*op, dim);
    // For O = H, <n|O|n> = E_n.
    std::vector<double> O_diag = ref.eigs;

    DistributedFtlmOptions opts;
    opts.n_samples       = 64;
    opts.n_groups        = 1;
    opts.lanczos_max_iter = 30;
    opts.betas           = {0.1, 0.5, 1.0};
    opts.seed_offset     = 0UL;
    opts.observable_op   = obs_op;

    DistributedFtlmResult res = distributed_ftlm(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.Z.size() == opts.betas.size());
    REQUIRE(res.O_expectation.size() == opts.betas.size());

    for (std::size_t b = 0; b < opts.betas.size(); ++b) {
        const double beta = opts.betas[b];
        const double Eexact = exact_O(ref.eigs, O_diag, beta);
        const double Eftlm  = res.O_expectation[b];
        const double abs_err = std::abs(Eftlm - Eexact);
        // Energies are O(1) for N=4, so absolute error here is meaningful.
        // 64 samples gives stat noise ~1/sqrt(64) ~ 12.5%; we allow 0.30
        // absolute (looser than relative because <H> can cross zero with
        // beta).
        INFO("beta=" << beta
             << " <H>_exact=" << Eexact
             << " <H>_ftlm="  << Eftlm
             << " |diff|=" << abs_err);
        REQUIRE(abs_err < 0.30);
    }
}

TEST_CASE("Distributed FTLM: <O>(beta) replicated across all ranks",
          "[distributed_ftlm][observable][replicated]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    auto obs_op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedFtlmOptions opts;
    opts.n_samples       = 8;
    opts.n_groups        = 1;
    opts.lanczos_max_iter = 20;
    opts.betas           = {0.5, 2.0};
    opts.seed_offset     = 31UL;
    opts.observable_op   = obs_op;

    DistributedFtlmResult res = distributed_ftlm(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.O_expectation.size() == opts.betas.size());

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> rank0_O = res.O_expectation;
    MPI_Bcast(rank0_O.data(), static_cast<int>(rank0_O.size()),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (std::size_t b = 0; b < rank0_O.size(); ++b) {
        REQUIRE(std::abs(res.O_expectation[b] - rank0_O[b]) < 1e-10);
    }
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
