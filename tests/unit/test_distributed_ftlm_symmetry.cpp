// =============================================================================
// test_distributed_ftlm_symmetry    (Phase D step 4)
//
// MPI lockdown for `ed::distributed::distributed_ftlm_symmetry`. Confirms
// that the per-sector FTLM kernel (a) returns replicated values across
// ranks within a group, and (b) when summed over every momentum sector
// of a translation-invariant Heisenberg chain reproduces the full-space
// partition function `Z(beta) = Tr e^{-beta H}` to within the J&P
// trace-estimator noise floor at moderate sample counts.
//
// Reference comes from a dense-eigendecomposition of the full operator
// (no symmetry projection on the reference path), so the test also
// exercises that the per-sector dimensions consistently sum to the
// global Hilbert-space dimension.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_ftlm.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedFtlmOptions;
using ed::distributed::DistributedFtlmResult;
using ed::distributed::distributed_ftlm_symmetry;

namespace {

double exact_Z(const std::vector<double>& E, double beta) {
    if (E.empty()) return 0.0;
    const double E0 = E.front();
    double s = 0.0;
    for (double e : E) s += std::exp(-beta * (e - E0));
    return s * std::exp(-beta * E0);
}

std::shared_ptr<Operator>
make_heisenberg_translation_op(int N, double J, bool periodic) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    op->symmetry_info = ed::sym::translation_group_1d(N);
    return op;
}

}  // namespace

TEST_CASE("distributed_ftlm_symmetry: sector Z sums to exact full-space Z",
          "[distributed_ftlm_symmetry][heisenberg][n4][pbc]") {
    const int N = 4;
    const double J = 1.0;
    auto op = make_heisenberg_translation_op(N, J, /*periodic=*/true);

    // Dense reference (no symmetry projection): full 2^N spectrum.
    auto op_dense = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         /*periodic=*/true).release());
    const std::uint64_t dim = 1ULL << N;
    auto ref = ed_tests::reference_from_operator(*op_dense, dim);

    const std::vector<double> betas = {0.1, 0.5};

    DistributedFtlmOptions opts;
    opts.n_samples       = 32;
    opts.n_groups        = 1;
    opts.lanczos_max_iter = 30;
    opts.betas           = betas;
    opts.seed_offset     = 0UL;

    const std::size_t n_sectors = op->symmetry_info.sectors.size();
    REQUIRE(n_sectors > 0);

    std::vector<double> Z_total(betas.size(), 0.0);
    for (std::size_t s = 0; s < n_sectors; ++s) {
        opts.seed_offset = 1000UL * static_cast<unsigned long>(s + 1);
        DistributedFtlmResult res =
            distributed_ftlm_symmetry(op, s, opts, MPI_COMM_WORLD);
        REQUIRE(res.Z.size() == betas.size());
        for (std::size_t b = 0; b < betas.size(); ++b) {
            Z_total[b] += res.Z[b];
        }
    }

    for (std::size_t b = 0; b < betas.size(); ++b) {
        const double Zexact = exact_Z(ref.eigs, betas[b]);
        const double rel    = std::abs(Z_total[b] - Zexact) / Zexact;
        INFO("beta=" << betas[b]
             << " Zexact=" << Zexact
             << " Zftlm="  << Z_total[b]
             << " relerr=" << rel);
        // Per-sector dim is small (16/4 = 4), Lanczos with full reorth
        // and max_iter >= dim is essentially exact -- the only remaining
        // noise is the initial random vector overlap with the leading
        // eigenmode of each sector. With 32 samples per sector at
        // moderate beta this is a few percent; allow 0.30 to leave
        // headroom for run-to-run variability in CI.
        REQUIRE(rel < 0.30);
    }
}

TEST_CASE("distributed_ftlm_symmetry: replicated Z across ranks within group",
          "[distributed_ftlm_symmetry][replicated]") {
    const int N = 4;
    auto op = make_heisenberg_translation_op(N, /*J=*/1.0, /*periodic=*/true);

    DistributedFtlmOptions opts;
    opts.n_samples       = 8;
    opts.n_groups        = 1;
    opts.lanczos_max_iter = 20;
    opts.betas           = {0.5, 1.0};
    opts.seed_offset     = 7UL;

    DistributedFtlmResult res =
        distributed_ftlm_symmetry(op, /*sector_idx=*/0, opts, MPI_COMM_WORLD);
    REQUIRE(res.Z.size() == opts.betas.size());

    std::vector<double> rank0_Z = res.Z;
    MPI_Bcast(rank0_Z.data(), static_cast<int>(rank0_Z.size()),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (std::size_t b = 0; b < rank0_Z.size(); ++b) {
        REQUIRE(std::abs(res.Z[b] - rank0_Z[b]) < 1e-10);
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
