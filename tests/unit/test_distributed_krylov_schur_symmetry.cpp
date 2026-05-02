// =============================================================================
// test_distributed_krylov_schur_symmetry    (Phase D step 2)
//
// MPI lockdown for `ed::distributed::distributed_krylov_schur_symmetry`.
// Cross-checks the symmetry-projected thick-restart Lanczos against the
// CPU `distributed_lanczos_symmetry` running on the SAME MPI_COMM_WORLD,
// SAME operator + sector, SAME seed. Both are CPU-only (no NCCL gate).
//
// Eigenvalue agreement is checked at `1e-9` (KS does locking and uses
// twice-CGS reorth, so this is well within the noise floor at
// max_iter=200 on these dim<= a few hundred problems).
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_krylov_schur.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::distributed_krylov_schur_symmetry;
using ed::distributed::distributed_lanczos_symmetry;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedSymmetryOperator;

namespace {

std::shared_ptr<Operator>
make_heisenberg_translation_op(int N, double J, bool periodic) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    op->symmetry_info = ed::sym::translation_group_1d(N);
    return op;
}

void check_sector(int N, double J, bool periodic,
                  std::size_t sector_idx, std::uint64_t exct,
                  unsigned long seed) {
    auto op = make_heisenberg_translation_op(N, J, periodic);
    DistributedSymmetryOperator dop(op, sector_idx, MPI_COMM_WORLD);

    DistributedLanczosOptions lopts;
    lopts.max_iter = 200;
    lopts.exct     = exct;
    lopts.seed     = seed;
    auto baseline = distributed_lanczos_symmetry(dop, lopts);

    auto ks = distributed_krylov_schur_symmetry(dop, lopts);

    // Ground-state lockdown: KS may not converge every requested
    // eigenvalue when the sector is small, but the leading
    // eigenvalue should always agree with the dense / Lanczos
    // reference. (Higher-k Ritz values are checked against the
    // dense reference in `test_distributed_lanczos_symmetry`; here
    // we are only validating that the templated KS body produces
    // the same leading eigenvalue when fed a `DistributedSymmetryOperator`.)
    REQUIRE(!ks.eigenvalues.empty());
    REQUIRE(!baseline.eigenvalues.empty());
    const double diff = std::abs(ks.eigenvalues[0] - baseline.eigenvalues[0]);
    INFO("N=" << N << " periodic=" << periodic
         << " sector=" << sector_idx
         << " ks_E0=" << ks.eigenvalues[0]
         << " lan_E0=" << baseline.eigenvalues[0]
         << " diff=" << diff);
    REQUIRE(diff <= 1e-9);
}

}  // namespace

TEST_CASE("distributed_krylov_schur_symmetry: N=4 PBC vs lanczos_symmetry",
          "[distributed_krylov_schur_symmetry][heisenberg][n4][pbc]") {
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                     /*sector=*/s, /*exct=*/2, /*seed=*/1234UL + s);
    }
}

TEST_CASE("distributed_krylov_schur_symmetry: N=6 PBC vs lanczos_symmetry",
          "[distributed_krylov_schur_symmetry][heisenberg][n6][pbc]") {
    auto info = ed::sym::translation_group_1d(6);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/6, /*J=*/-1.5, /*periodic=*/true,
                     /*sector=*/s, /*exct=*/3, /*seed=*/4242UL + s);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        FILE* dummy = std::freopen("/dev/null", "w", stdout);
        (void)dummy;
    }

    int local_result = Catch::Session().run(argc, argv);
    int global_result = 0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Finalize();
    return global_result;
}
