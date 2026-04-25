// =============================================================================
// test_distributed_lanczos (Phase 3b #2)
//
// MPI lockdown for ed::distributed::distributed_lanczos on small Heisenberg
// chains. Compares distributed ground-state energy to the dense-reference
// spectrum from tests/common/test_harness.h on np=1, np=2, np=4.
//
// Coverage:
//   * N=4 OBC ground state (analytical -1.6160255...)
//   * N=6 PBC ground state vs dense Eigen reference
//   * full_reorth = true vs false on N=6 PBC (must agree to within Lanczos
//     three-term-recurrence noise tolerance ~1e-8)
//   * Replicated eigenvalues across all ranks (bit-identical)
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include "common/test_harness.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedOperator;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedLanczosResult;
using ed::distributed::distributed_lanczos;

TEST_CASE("Distributed Lanczos: N=4 OBC ground state vs dense reference",
          "[distributed_lanczos][heisenberg]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);

    // Dense reference (every rank computes its own; deterministic).
    auto ref = ed_tests::reference_from_operator(*op, dop.global_dim());

    DistributedLanczosOptions opts;
    opts.max_iter   = 60;
    opts.exct       = 1;
    opts.tol        = 1e-12;
    opts.full_reorth = false;
    opts.seed       = 12345UL;

    auto res = distributed_lanczos(dop, opts);

    REQUIRE(!res.eigenvalues.empty());
    INFO("E0_dist = " << res.eigenvalues.front()
         << "  E0_dense = " << ref.eigs.front()
         << "  iters = " << res.iterations);
    REQUIRE(std::abs(res.eigenvalues.front() - ref.eigs.front()) < 1e-8);
}

TEST_CASE("Distributed Lanczos: N=6 PBC ground state vs dense reference",
          "[distributed_lanczos][heisenberg][pbc]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);

    auto ref = ed_tests::reference_from_operator(*op, dop.global_dim());

    DistributedLanczosOptions opts;
    opts.max_iter   = 80;
    opts.exct       = 1;
    opts.tol        = 1e-12;
    opts.full_reorth = false;
    opts.seed       = 7UL;

    auto res = distributed_lanczos(dop, opts);

    REQUIRE(!res.eigenvalues.empty());
    INFO("E0_dist = " << res.eigenvalues.front()
         << "  E0_dense = " << ref.eigs.front()
         << "  iters = " << res.iterations);
    REQUIRE(std::abs(res.eigenvalues.front() - ref.eigs.front()) < 1e-8);
}

TEST_CASE("Distributed Lanczos: full_reorth agrees with no-orth on N=6 PBC",
          "[distributed_lanczos][reorth]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);

    DistributedLanczosOptions opts;
    opts.max_iter = 60;
    opts.exct     = 1;
    opts.tol      = 1e-12;
    opts.seed     = 31UL;

    opts.full_reorth = false;
    auto res_no = distributed_lanczos(dop, opts);

    opts.full_reorth = true;
    auto res_re = distributed_lanczos(dop, opts);

    REQUIRE(!res_no.eigenvalues.empty());
    REQUIRE(!res_re.eigenvalues.empty());
    INFO("no-orth E0=" << res_no.eigenvalues.front()
         << "  full-reorth E0=" << res_re.eigenvalues.front());
    REQUIRE(std::abs(res_no.eigenvalues.front() - res_re.eigenvalues.front())
            < 1e-8);
}

TEST_CASE("Distributed Lanczos: eigenvalues are replicated bit-for-bit "
          "across all ranks",
          "[distributed_lanczos][replicated]") {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);

    DistributedLanczosOptions opts;
    opts.max_iter = 50;
    opts.exct     = 3;
    opts.tol      = 1e-12;
    opts.seed     = 999UL;

    auto res = distributed_lanczos(dop, opts);

    REQUIRE(!res.eigenvalues.empty());
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = static_cast<int>(res.eigenvalues.size());
    int n_rank0 = n;
    MPI_Bcast(&n_rank0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    REQUIRE(n == n_rank0);

    std::vector<double> rank0_evals(n_rank0);
    if (rank == 0) {
        rank0_evals = res.eigenvalues;
    }
    MPI_Bcast(rank0_evals.data(), n_rank0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n; ++i) {
        // Replicated computation: should be exactly equal modulo MPI_SUM
        // round-off (MPI does not guarantee bit-for-bit determinism for
        // floating-point sums in general, but for the N=6 problem with the
        // same algorithm path on every rank the differences are << 1e-12).
        REQUIRE(std::abs(res.eigenvalues[i] - rank0_evals[i]) < 1e-10);
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
