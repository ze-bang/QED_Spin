// =============================================================================
// test_distributed_tpq_symmetry    (Phase E step 1)
//
// MPI lockdown for `ed::distributed::distributed_tpq_symmetry`. Confirms
// that the per-sector canonical-TPQ kernel returns:
//   (a) replicated `energy[b]` across ranks within a group, and
//   (b) when run in a single symmetry sector with many samples, the
//       sample-averaged <H>(beta) converges to the exact thermal
//       energy of that sector,
//         <H>_s(beta) = sum_{k in sector s} E_k * exp(-beta * E_k)
//                       / sum_{k in sector s} exp(-beta * E_k).
//
// Reference comes from a dense projection: build the dense Hamiltonian
// matrix, project onto each momentum sector via the Operator's
// SymmetryInfo, diagonalise, compute the per-sector thermal average.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_tpq.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedTpqOptions;
using ed::distributed::DistributedTpqResult;
using ed::distributed::distributed_tpq_symmetry;
using ed::distributed::DistributedSymmetryOperator;

namespace {

double exact_E_in_sector(const std::vector<double>& E, double beta) {
    if (E.empty()) return 0.0;
    const double E0 = E.front();
    double Z = 0.0, num = 0.0;
    for (double e : E) {
        const double w = std::exp(-beta * (e - E0));
        Z += w;
        num += e * w;
    }
    return num / Z;
}

// Diagonalise the dense per-sector projected Hamiltonian by exploiting
// the same DistributedSymmetryOperator (np=1 ==> local slab is the
// whole sector) and reading column-by-column.
std::vector<double> sector_eigenvalues(std::shared_ptr<Operator> op,
                                       std::size_t sector_idx) {
    DistributedSymmetryOperator dsop(op, sector_idx, MPI_COMM_SELF);
    const std::uint64_t d = dsop.global_dim();
    REQUIRE(dsop.local_size() == d);
    std::vector<std::complex<double>> col(d), out(d);
    Eigen::MatrixXcd Hmat(static_cast<int>(d), static_cast<int>(d));
    for (std::uint64_t j = 0; j < d; ++j) {
        std::fill(col.begin(), col.end(), std::complex<double>(0.0, 0.0));
        col[j] = std::complex<double>(1.0, 0.0);
        std::fill(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
        dsop.apply(col.data(), out.data());
        for (std::uint64_t i = 0; i < d; ++i) {
            Hmat(static_cast<int>(i), static_cast<int>(j)) = out[i];
        }
    }
    return ed_tests::dense_eigenvalues(Hmat);
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

TEST_CASE("distributed_tpq_symmetry: per-sector <H>(beta) matches exact",
          "[distributed_tpq_symmetry][heisenberg][n4][pbc]") {
    const int N = 4;
    const double J = 1.0;
    auto op = make_heisenberg_translation_op(N, J, /*periodic=*/true);

    const std::vector<double> betas = {0.5, 1.5};

    DistributedTpqOptions opts;
    opts.n_samples        = 64;
    opts.n_groups         = 1;
    opts.delta_beta       = 0.05;
    opts.taylor_order     = 12;
    opts.betas            = betas;
    opts.seed_offset      = 0UL;
    opts.compute_variance = false;

    const std::size_t n_sectors = op->symmetry_info.sectors.size();
    REQUIRE(n_sectors > 0);

    // Pick a sector with non-trivial dimension > 1 so the average
    // is meaningful (the trivially 1D sectors give exact <H>=E
    // regardless of sample count and don't exercise the random
    // averaging).
    std::size_t test_sector = 0;
    for (std::size_t s = 0; s < n_sectors; ++s) {
        DistributedSymmetryOperator probe(op, s, MPI_COMM_SELF);
        if (probe.global_dim() >= 2) { test_sector = s; break; }
    }

    auto eigs = sector_eigenvalues(op, test_sector);
    REQUIRE(eigs.size() >= 2);

    opts.seed_offset = 1234UL * static_cast<unsigned long>(test_sector + 1);
    DistributedTpqResult res =
        distributed_tpq_symmetry(op, test_sector, opts, MPI_COMM_WORLD);
    REQUIRE(res.energy.size() == betas.size());

    for (std::size_t b = 0; b < betas.size(); ++b) {
        const double Eexact = exact_E_in_sector(eigs, betas[b]);
        const double Etpq   = res.energy[b];
        const double E_scale = std::max(1.0, std::abs(Eexact));
        const double rel = std::abs(Etpq - Eexact) / E_scale;
        INFO("sector=" << test_sector
             << " beta=" << betas[b]
             << " Eexact=" << Eexact
             << " Etpq="   << Etpq
             << " relerr=" << rel);
        // Canonical-TPQ noise on a small sector with 64 samples: the
        // main random source is the initial-state overlap distribution.
        // Per-sector dim is small here (4-6 for N=4 PBC), so a few
        // percent agreement is realistic. Allow 0.20 of E-scale.
        REQUIRE(rel < 0.20);
    }
}

TEST_CASE("distributed_tpq_symmetry: replicated energy across ranks within group",
          "[distributed_tpq_symmetry][replicated]") {
    const int N = 4;
    auto op = make_heisenberg_translation_op(N, /*J=*/1.0, /*periodic=*/true);

    DistributedTpqOptions opts;
    opts.n_samples        = 8;
    opts.n_groups         = 1;
    opts.delta_beta       = 0.1;
    opts.taylor_order     = 10;
    opts.betas            = {0.5, 1.0};
    opts.seed_offset      = 7UL;
    opts.compute_variance = true;

    DistributedTpqResult res =
        distributed_tpq_symmetry(op, /*sector_idx=*/0, opts, MPI_COMM_WORLD);
    REQUIRE(res.energy.size() == opts.betas.size());

    std::vector<double> rank0_E = res.energy;
    MPI_Bcast(rank0_E.data(), static_cast<int>(rank0_E.size()),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (std::size_t b = 0; b < rank0_E.size(); ++b) {
        REQUIRE(std::abs(res.energy[b] - rank0_E[b]) < 1e-10);
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
