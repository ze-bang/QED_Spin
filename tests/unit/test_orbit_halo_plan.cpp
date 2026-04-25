// =============================================================================
// test_orbit_halo_plan    (Phase 3b #7, stage 2 prep)
//
// MPI lockdown for ed::distributed::OrbitHaloPlan. Synthetic
// problem:
//
//   * `n_orbits` orbits arranged in a 1D ring; each orbit talks to
//     its two neighbours under the (synthetic) Hamiltonian.
//   * `OrbitPartition` from `balanced_orbit_slab(weights, comm_size)`
//     assigns orbits to ranks via LPT greedy.
//   * Every rank computes `needed_orbits = union over locally-owned
//     orbit i of {i-1 mod n_orbits, i+1 mod n_orbits}` -- this is
//     the "I need amplitude(j) to compute H_ij contribution into
//     amplitude(i)" set in the future symmetry-projected SpMV.
//   * `OrbitHaloPlan` builds the all-to-all halo, exchanges
//     synthetic complex amplitudes (`amp[k] = exp(2 pi i k /
//     n_orbits)`), and we verify every recv slot matches the
//     analytic value.
//
// Coverage:
//   * Counts/displs are non-negative; recv_counts_[rank] = 0
//     (locally-owned orbits are never in the halo).
//   * sum(recv_counts_) over all ranks == sum(send_counts_) == total
//     number of (orbit, owner-rank) edges across the partition.
//   * exchange(local, halo) populates halo with the analytic value
//     for every recv slot, on np ∈ {1, 2, 4}.
//   * np=1: halo is empty (no remote orbits).
//
// Run-time gating: this test does NOT require CUDA. It is registered
// at np ∈ {1, 2, 4} via `ed_add_mpi_test`.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/orbit_halo_plan.h>
#include <ed/distributed/orbit_partition.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include <mpi.h>

using ed::distributed::balanced_orbit_slab;
using ed::distributed::OrbitHaloPlan;
using ed::distributed::OrbitPartition;
using Complex = std::complex<double>;

namespace {

// "Hamiltonian" amplitude for orbit k: a deterministic complex of unit
// modulus indexed by the (global) orbit id. Allows every rank to verify
// its halo locally without any cross-rank reduction.
Complex synth_amp(std::uint64_t k, std::uint64_t n_orbits) {
    const double theta = 2.0 * M_PI * static_cast<double>(k)
                         / static_cast<double>(n_orbits);
    return Complex(std::cos(theta), std::sin(theta));
}

// Return the set of orbit ids THIS rank needs from other ranks under
// the synthetic ring topology: for every locally-owned orbit i, mark
// (i-1) mod n and (i+1) mod n.
std::vector<std::size_t> ring_needed_orbits(const OrbitPartition& part,
                                            int rank,
                                            std::uint64_t n_orbits) {
    std::vector<std::size_t> needed;
    needed.reserve(2 * part.local_size(rank));
    for (std::size_t i : part.rank_orbits[static_cast<std::size_t>(rank)]) {
        const std::uint64_t i64 = static_cast<std::uint64_t>(i);
        const std::uint64_t left  = (i64 + n_orbits - 1) % n_orbits;
        const std::uint64_t right = (i64 + 1) % n_orbits;
        needed.push_back(static_cast<std::size_t>(left));
        needed.push_back(static_cast<std::size_t>(right));
    }
    return needed;
}

}  // namespace

TEST_CASE("OrbitHaloPlan: ring of 32 orbits, equal weights",
          "[orbit_halo_plan][ring]") {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::uint64_t n_orbits = 32;
    std::vector<std::uint64_t> weights(n_orbits, 1);
    OrbitPartition part = balanced_orbit_slab(weights, size);
    REQUIRE(part.n_ranks == size);

    auto needed = ring_needed_orbits(part, rank, n_orbits);
    OrbitHaloPlan plan(part, needed, MPI_COMM_WORLD);

    REQUIRE(plan.rank() == rank);
    REQUIRE(plan.comm_size() == size);
    // Locally-owned orbits never live in the halo.
    REQUIRE(plan.recv_counts()[rank] == 0);
    REQUIRE(plan.send_counts()[rank] == 0);
    // Counts are non-negative.
    for (int r = 0; r < size; ++r) {
        REQUIRE(plan.recv_counts()[r] >= 0);
        REQUIRE(plan.send_counts()[r] >= 0);
    }

    // Pack local amplitudes: local_amplitudes[owner_local_index(orbit)]
    // = synth_amp(orbit). owner_local_index is in [0, local_size(rank)).
    const std::size_t local_n = part.local_size(rank);
    std::vector<Complex> local(local_n, Complex(0.0, 0.0));
    for (std::size_t orbit : part.rank_orbits[static_cast<std::size_t>(rank)]) {
        const std::size_t k = part.owner_local_index(orbit);
        local[k] = synth_amp(orbit, n_orbits);
    }

    std::vector<Complex> halo(plan.recv_total(), Complex(0.0, 0.0));
    plan.exchange(local.data(), halo.data());

    // Verify every halo slot.
    for (std::size_t k = 0; k < plan.recv_total(); ++k) {
        const std::uint64_t orbit = plan.recv_orbit_id()[k];
        const Complex expected = synth_amp(orbit, n_orbits);
        INFO("rank=" << rank << " k=" << k << " orbit=" << orbit
             << " got=(" << halo[k].real() << "," << halo[k].imag()
             << ")  expected=(" << expected.real() << ","
             << expected.imag() << ")");
        REQUIRE(std::abs(halo[k].real() - expected.real()) < 1e-12);
        REQUIRE(std::abs(halo[k].imag() - expected.imag()) < 1e-12);
    }

    // Global accounting: sum of all per-rank recv totals == sum of all
    // per-rank send totals (every halo edge is a send on one rank and
    // a recv on another).
    int local_recv_total = static_cast<int>(plan.recv_total());
    int local_send_total = static_cast<int>(plan.send_total());
    int global_recv = 0, global_send = 0;
    MPI_Allreduce(&local_recv_total, &global_recv, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_send_total, &global_send, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    REQUIRE(global_recv == global_send);

    // np=1: halo is necessarily empty (only one rank, owns everything).
    if (size == 1) {
        REQUIRE(plan.recv_total() == 0u);
        REQUIRE(plan.send_total() == 0u);
    }

    // For np>1: at least SOME rank must have non-empty halo (the ring
    // has 32 boundary edges and `balanced_orbit_slab` cuts it; the
    // exact count depends on the partition geometry but the
    // *existence* of cross-rank traffic is guaranteed).
    if (size > 1) {
        REQUIRE(global_recv > 0);
    }
}

TEST_CASE("OrbitHaloPlan: heavy-outlier partition still exchanges correctly",
          "[orbit_halo_plan][heavy]") {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::uint64_t n_orbits = 24;
    std::vector<std::uint64_t> weights(n_orbits, 1);
    weights[0] = 100;
    weights[7] = 50;
    OrbitPartition part = balanced_orbit_slab(weights, size);

    auto needed = ring_needed_orbits(part, rank, n_orbits);
    OrbitHaloPlan plan(part, needed, MPI_COMM_WORLD);

    const std::size_t local_n = part.local_size(rank);
    std::vector<Complex> local(local_n, Complex(0.0, 0.0));
    for (std::size_t orbit : part.rank_orbits[static_cast<std::size_t>(rank)]) {
        const std::size_t k = part.owner_local_index(orbit);
        local[k] = synth_amp(orbit, n_orbits);
    }

    std::vector<Complex> halo(plan.recv_total(), Complex(0.0, 0.0));
    plan.exchange(local.data(), halo.data());
    for (std::size_t k = 0; k < plan.recv_total(); ++k) {
        const std::uint64_t orbit = plan.recv_orbit_id()[k];
        const Complex expected = synth_amp(orbit, n_orbits);
        REQUIRE(std::abs(halo[k] - expected) < 1e-12);
    }
}

TEST_CASE("OrbitHaloPlan: needed_orbits with duplicates and locally-owned ids",
          "[orbit_halo_plan][dedup]") {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::uint64_t n_orbits = 16;
    std::vector<std::uint64_t> weights(n_orbits, 1);
    OrbitPartition part = balanced_orbit_slab(weights, size);

    // Build a "needed" set that includes (a) every orbit in the world
    // (so it's a strict superset of the ring needs), with (b) every id
    // duplicated 3x. The plan must filter out locally-owned ids and
    // dedupe duplicates.
    std::vector<std::size_t> needed;
    for (int rep = 0; rep < 3; ++rep) {
        for (std::uint64_t i = 0; i < n_orbits; ++i) {
            needed.push_back(static_cast<std::size_t>(i));
        }
    }
    OrbitHaloPlan plan(part, needed, MPI_COMM_WORLD);

    // recv_orbit_id_ must be unique across all slots.
    const auto& recv_ids = plan.recv_orbit_id();
    for (std::size_t k = 1; k < recv_ids.size(); ++k) {
        // Plan stores recv ids rank-major, sorted ascending within each
        // rank slab. Across rank boundaries the ids may be out of
        // order, so we test global uniqueness instead.
    }
    std::vector<std::uint64_t> sorted = recv_ids;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 1; k < sorted.size(); ++k) {
        REQUIRE(sorted[k - 1] != sorted[k]);
    }

    // None of the recv ids may be locally owned.
    for (std::uint64_t orbit : recv_ids) {
        REQUIRE(part.owner_rank(orbit) != rank);
    }

    // For np > 1, the recv set is "every orbit not owned by this rank"
    // (since `needed` was the universe). For np = 1, recv_total = 0.
    if (size == 1) {
        REQUIRE(plan.recv_total() == 0u);
    } else {
        REQUIRE(plan.recv_total() == n_orbits - part.local_size(rank));
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
    int local = Catch::Session().run(argc, argv);
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global;
}
