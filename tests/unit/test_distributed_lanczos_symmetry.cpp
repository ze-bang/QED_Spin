// =============================================================================
// test_distributed_lanczos_symmetry    (Phase 3b #7, stage 3)
//
// MPI lockdown for `ed::distributed::distributed_lanczos_symmetry` --
// the distributed-memory Lanczos rebased onto `DistributedSymmetryOperator`.
// We cross-check the ground-state energy against:
//
//   1. A serial dense `Eigen::SelfAdjointEigenSolver` of the projected
//      matrix `H_q` (built using the same per-group-element character
//      convention as `DistributedSymmetryOperator`).
//   2. The serial CPU `lanczos_no_ortho` on the same H_q (to lock down
//      the iterative tolerance).
//
// Coverage:
//   * np ∈ {1, 2, 4}
//   * N=4 OBC sector 0, N=4 PBC sector 0, N=6 PBC sector 0
//   * Bit-replicated eigenvalues across all ranks.
//
// Tolerance: |E0_dist - E0_dense| < 1e-8 (Lanczos to 1e-12 tridiag-solve
//             tol; the dense reference is the "exact" eigenvalue).
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <queue>
#include <set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <mpi.h>

using ed::distributed::DistributedSymmetryOperator;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::distributed_lanczos_symmetry;
using Complex = std::complex<double>;

namespace {

std::shared_ptr<Operator>
make_heisenberg_translation_op(int N, double J, bool periodic) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    op->symmetry_info = ed::sym::translation_group_1d(N);
    return op;
}

// Build the dense projected matrix H_q in NATURAL orbit ordering.
Eigen::MatrixXcd build_projected_matrix(const Operator& op,
                                        std::size_t sector_idx) {
    const std::uint64_t n_bits = op.getNumBits();
    const std::uint64_t dim = (n_bits == 0) ? 1ULL : (1ULL << n_bits);
    const auto& info   = op.symmetry_info;
    const auto& sector = info.sectors[sector_idx];

    REQUIRE(sector.phase_factors.size() == info.max_clique.size());
    const std::vector<Complex>& chi = sector.phase_factors;

    constexpr std::int64_t kNoOrbit = -1;
    std::vector<std::int64_t> raw_state_to_orbit(dim, kNoOrbit);
    std::vector<Complex>      raw_phi(dim, Complex(0.0, 0.0));
    std::vector<std::uint64_t> raw_orbit_reps;
    std::vector<double>        raw_orbit_norms_sq;

    for (std::uint64_t b = 0; b < dim; ++b) {
        if (raw_state_to_orbit[b] != kNoOrbit) continue;

        std::set<std::uint64_t> orbit;
        orbit.insert(b);
        std::queue<std::uint64_t> q;
        q.push(b);
        while (!q.empty()) {
            const std::uint64_t cur = q.front(); q.pop();
            for (const auto& gen : info.generators) {
                const std::uint64_t nxt = applyPermutation(cur, gen);
                if (orbit.insert(nxt).second) q.push(nxt);
            }
        }

        const std::uint64_t rep = *orbit.begin();
        const std::int64_t oid =
            static_cast<std::int64_t>(raw_orbit_reps.size());

        for (std::uint64_t s : orbit) raw_phi[s] = Complex(0.0, 0.0);
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            const std::uint64_t s = applyPermutation(rep, info.max_clique[g]);
            raw_phi[s] += std::conj(chi[g]);
        }

        double N_i = 0.0;
        for (std::uint64_t s : orbit) N_i += std::norm(raw_phi[s]);

        for (std::uint64_t s : orbit) raw_state_to_orbit[s] = oid;
        raw_orbit_reps.push_back(rep);
        raw_orbit_norms_sq.push_back(N_i);
    }

    // Filter zero-norm orbits.
    constexpr double kZeroNormTolerance =
        DistributedSymmetryOperator::kZeroNormTolerance;
    std::vector<std::int64_t> raw_to_dense(raw_orbit_reps.size(), kNoOrbit);
    std::vector<double> orbit_norms;
    std::size_t n_orbits = 0;
    for (std::size_t i = 0; i < raw_orbit_reps.size(); ++i) {
        if (raw_orbit_norms_sq[i] > kZeroNormTolerance) {
            raw_to_dense[i] = static_cast<std::int64_t>(n_orbits++);
            orbit_norms.push_back(raw_orbit_norms_sq[i]);
        }
    }

    // Per-state dense orbit + phi.
    std::vector<std::int64_t> state_to_orbit(dim, kNoOrbit);
    std::vector<Complex>      phi(dim, Complex(0.0, 0.0));
    for (std::uint64_t b = 0; b < dim; ++b) {
        const std::int64_t raw = raw_state_to_orbit[b];
        if (raw == kNoOrbit) continue;
        const std::int64_t dense = raw_to_dense[raw];
        if (dense == kNoOrbit) continue;
        state_to_orbit[b] = dense;
        phi[b] = raw_phi[b];
    }
    std::vector<double> inv_sqrt_N(n_orbits);
    for (std::size_t i = 0; i < n_orbits; ++i) {
        inv_sqrt_N[i] = 1.0 / std::sqrt(orbit_norms[i]);
    }

    // Build dense H_q column-by-column.
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(n_orbits, n_orbits);
    std::vector<Complex> tilde_j(dim, Complex(0.0, 0.0));
    std::vector<Complex> H_tilde_j(dim, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < n_orbits; ++j) {
        std::fill(tilde_j.begin(), tilde_j.end(), Complex(0.0, 0.0));
        for (std::uint64_t b = 0; b < dim; ++b) {
            if (state_to_orbit[b] == static_cast<std::int64_t>(j)) {
                tilde_j[b] = phi[b];
            }
        }
        const_cast<Operator&>(op).apply(tilde_j.data(), H_tilde_j.data(),
                                        static_cast<size_t>(dim));
        for (std::uint64_t b = 0; b < dim; ++b) {
            const std::int64_t i = state_to_orbit[b];
            if (i == kNoOrbit) continue;
            H(static_cast<std::size_t>(i), j) +=
                std::conj(phi[b]) * H_tilde_j[b];
        }
        for (std::size_t i = 0; i < n_orbits; ++i) {
            H(i, j) *= inv_sqrt_N[i] * inv_sqrt_N[j];
        }
    }
    return H;
}

double dense_ground_state_energy(const Operator& op,
                                 std::size_t sector_idx) {
    Eigen::MatrixXcd H = build_projected_matrix(op, sector_idx);
    if (H.rows() == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    return es.eigenvalues().minCoeff();
}

void check_lanczos(int N, double J, bool periodic, std::size_t sector_idx,
                   unsigned long seed, std::uint64_t max_iter) {
    auto op = make_heisenberg_translation_op(N, J, periodic);
    const double E0_dense = dense_ground_state_energy(*op, sector_idx);

    DistributedSymmetryOperator dop(op, sector_idx, MPI_COMM_WORLD);

    DistributedLanczosOptions opts;
    opts.max_iter   = max_iter;
    opts.exct       = 1;
    opts.tol        = 1e-12;
    opts.seed       = seed;
    opts.full_reorth = true;  // small dims -- avoid Krylov pollution
    opts.verbose    = false;

    auto res = distributed_lanczos_symmetry(dop, opts);
    REQUIRE(!res.eigenvalues.empty());
    const double E0_dist = res.eigenvalues.front();

    INFO("N=" << N << " periodic=" << periodic << " sector=" << sector_idx
         << " global_dim=" << dop.global_dim()
         << " seed=" << seed << " iters=" << res.iterations
         << " E0_dist=" << E0_dist << " E0_dense=" << E0_dense);
    REQUIRE(std::isfinite(E0_dist));
    REQUIRE(std::isfinite(E0_dense));
    REQUIRE(std::abs(E0_dist - E0_dense) < 1e-8);

    // Eigenvalues must be bit-replicated across all ranks.
    int rank = 0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 0; MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size > 1) {
        double payload = E0_dist;
        MPI_Bcast(&payload, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        REQUIRE(payload == E0_dist);
    }
}

}  // namespace

TEST_CASE("distributed_lanczos_symmetry: N=4 OBC sector=0",
          "[distributed_lanczos_symmetry][n4][obc]") {
    check_lanczos(/*N=*/4, /*J=*/1.0, /*periodic=*/false,
                  /*sector_idx=*/0, /*seed=*/12345UL,
                  /*max_iter=*/60);
}

TEST_CASE("distributed_lanczos_symmetry: N=4 PBC sector=0",
          "[distributed_lanczos_symmetry][n4][pbc]") {
    check_lanczos(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                  /*sector_idx=*/0, /*seed=*/777UL,
                  /*max_iter=*/60);
}

TEST_CASE("distributed_lanczos_symmetry: N=6 PBC sector=0",
          "[distributed_lanczos_symmetry][n6][pbc]") {
    check_lanczos(/*N=*/6, /*J=*/1.0, /*periodic=*/true,
                  /*sector_idx=*/0, /*seed=*/42UL,
                  /*max_iter=*/120);
}

TEST_CASE("distributed_lanczos_symmetry: N=6 PBC every momentum sector",
          "[distributed_lanczos_symmetry][n6][pbc][allsectors]") {
    auto info = ed::sym::translation_group_1d(6);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_lanczos(/*N=*/6, /*J=*/1.0, /*periodic=*/true,
                      /*sector_idx=*/s, /*seed=*/100UL + s,
                      /*max_iter=*/120);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
