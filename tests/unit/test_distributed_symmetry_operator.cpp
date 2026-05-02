// =============================================================================
// test_distributed_symmetry_operator    (Phase 3b #7, stage 2)
//
// MPI lockdown for `ed::distributed::DistributedSymmetryOperator` on small
// Heisenberg chains with 1D translation symmetry. The test:
//
//   1. Builds a Heisenberg chain Operator + populates `symmetry_info`
//      via `ed::sym::translation_group_1d(N)`.
//   2. Builds an INDEPENDENT serial reference of the symmetry-projected
//      matrix `H_q` in the orbit basis (same projection formulas as
//      `DistributedSymmetryOperator`'s ctor, but written from scratch so a
//      bug in the class shows up as a test failure).
//   3. Constructs `DistributedSymmetryOperator(op, sector_idx,
//      MPI_COMM_WORLD)`, applies it to a deterministic random vector,
//      `MPI_Allgatherv`s the local result, and compares to the dense
//      reference apply across every momentum sector.
//
// Coverage:
//   * np ∈ {1, 2, 4}
//   * N=4 OBC and PBC (Z_4 translation -- 4 momentum sectors)
//   * N=6 PBC (Z_6 translation -- 6 momentum sectors)
//   * Slab geometry: sum(local_size) == global_dim
//   * Halo plan exists; recv_total >= 0; sends/recvs sane
//
// Tolerance: 1e-10 on the per-component apply comparison. The matrix
// construction is dense projection over O(|G|) terms with no extra
// flops beyond `op->apply` itself, so machine-precision agreement
// between reference and distributed paths is the expected baseline.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <queue>
#include <random>
#include <set>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedSymmetryOperator;
using Complex = std::complex<double>;

namespace {

// Build a Heisenberg chain on N sites + translation symmetry (Z_N).
std::shared_ptr<Operator>
make_heisenberg_translation_op(int N, double J, bool periodic) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    op->symmetry_info = ed::sym::translation_group_1d(N);
    return op;
}

// Reference: independently build the symmetry-projected dense matrix
// `H_q` for sector `s` of the operator `op`, using the same projection
// formulas as DistributedSymmetryOperator but written from scratch so
// that a regression in the class definitively fails the test.
struct ProjectedReference {
    std::vector<std::uint64_t> orbit_reps;     // length n_orbits
    std::vector<double>        orbit_norms_sq; // length n_orbits
    std::vector<std::vector<Complex>> H;       // n_orbits x n_orbits
};

ProjectedReference build_reference(const Operator& op, std::size_t sector_idx) {
    const std::uint64_t n_bits = op.getNumBits();
    const std::uint64_t dim    = (n_bits == 0) ? 1ULL : (1ULL << n_bits);
    const auto& info   = op.symmetry_info;
    const auto& sector = info.sectors[sector_idx];

    const std::size_t n_group = info.max_clique.size();
    REQUIRE(sector.phase_factors.size() == n_group);
    // Per-group-element character convention -- matches
    // ed::sym::group_from_generators (src/symmetry/group.cpp L221-230)
    // and tests/unit/test_symmetry_dsl.cpp L127-139.
    const std::vector<Complex>& chi = sector.phase_factors;

    // BFS-enumerate orbits.
    std::vector<std::int64_t> raw_to_dense;
    std::vector<std::int64_t> state_to_orbit(dim, -1);
    std::vector<Complex>      phi(dim, Complex(0.0, 0.0));

    ProjectedReference ref;
    std::vector<std::int64_t> raw_state_to_orbit(dim, -1);
    std::vector<Complex>      raw_phi(dim, Complex(0.0, 0.0));
    std::vector<std::uint64_t> raw_orbit_reps;
    std::vector<double>        raw_orbit_norms_sq;

    for (std::uint64_t b = 0; b < dim; ++b) {
        if (raw_state_to_orbit[b] != -1) continue;
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
        const std::int64_t  oid =
            static_cast<std::int64_t>(raw_orbit_reps.size());
        for (std::uint64_t s : orbit) raw_phi[s] = Complex(0.0, 0.0);
        for (std::size_t g = 0; g < n_group; ++g) {
            const std::uint64_t s =
                applyPermutation(rep, info.max_clique[g]);
            raw_phi[s] += std::conj(chi[g]);
        }
        double N_i = 0.0;
        for (std::uint64_t s : orbit) {
            N_i += std::norm(raw_phi[s]);
            raw_state_to_orbit[s] = oid;
        }
        raw_orbit_reps.push_back(rep);
        raw_orbit_norms_sq.push_back(N_i);
    }

    raw_to_dense.assign(raw_orbit_reps.size(), -1);
    for (std::size_t i = 0; i < raw_orbit_reps.size(); ++i) {
        if (raw_orbit_norms_sq[i] >
            DistributedSymmetryOperator::kZeroNormTolerance) {
            raw_to_dense[i] =
                static_cast<std::int64_t>(ref.orbit_reps.size());
            ref.orbit_reps.push_back(raw_orbit_reps[i]);
            ref.orbit_norms_sq.push_back(raw_orbit_norms_sq[i]);
        }
    }
    for (std::uint64_t b = 0; b < dim; ++b) {
        const std::int64_t raw = raw_state_to_orbit[b];
        if (raw == -1) continue;
        const std::int64_t dense = raw_to_dense[raw];
        if (dense == -1) continue;
        state_to_orbit[b] = dense;
        phi[b]            = raw_phi[b];
    }

    const std::size_t n_orbits = ref.orbit_reps.size();
    ref.H.assign(n_orbits, std::vector<Complex>(n_orbits, Complex(0.0, 0.0)));

    std::vector<double> inv_sqrt_N(n_orbits);
    for (std::size_t i = 0; i < n_orbits; ++i) {
        inv_sqrt_N[i] = 1.0 / std::sqrt(ref.orbit_norms_sq[i]);
    }

    std::vector<Complex> tilde_j(dim, Complex(0.0, 0.0));
    std::vector<Complex> H_tilde_j(dim, Complex(0.0, 0.0));
    for (std::size_t j = 0; j < n_orbits; ++j) {
        std::fill(tilde_j.begin(), tilde_j.end(), Complex(0.0, 0.0));
        for (std::uint64_t b = 0; b < dim; ++b) {
            if (state_to_orbit[b] == static_cast<std::int64_t>(j)) {
                tilde_j[b] = phi[b];
            }
        }
        op.apply(tilde_j.data(), H_tilde_j.data(), static_cast<size_t>(dim));
        for (std::uint64_t b = 0; b < dim; ++b) {
            const std::int64_t i = state_to_orbit[b];
            if (i == -1) continue;
            ref.H[static_cast<std::size_t>(i)][j] +=
                std::conj(phi[b]) * H_tilde_j[b];
        }
        for (std::size_t i = 0; i < n_orbits; ++i) {
            ref.H[i][j] *= (inv_sqrt_N[i] * inv_sqrt_N[j]);
        }
    }
    return ref;
}

// Apply the dense reference matrix to a vector.
std::vector<Complex> apply_reference(const ProjectedReference& ref,
                                     const std::vector<Complex>& x_global) {
    const std::size_t n = ref.orbit_reps.size();
    std::vector<Complex> y(n, Complex(0.0, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        Complex acc(0.0, 0.0);
        for (std::size_t j = 0; j < n; ++j) acc += ref.H[i][j] * x_global[j];
        y[i] = acc;
    }
    return y;
}

// Allgatherv every rank's slab into a rank-major global vector. The
// distributed symmetry operator stores its k-th local row at global
// position `local_offset + k`, so the simple Allgatherv over slabs is
// the right inverse.
std::vector<Complex>
allgather_slabs(const Complex* local, std::uint64_t local_n,
                std::uint64_t global_n, MPI_Comm comm) {
    int size = 0; MPI_Comm_size(comm, &size);
    std::vector<int> counts(size), displs(size);
    int local_n_int = static_cast<int>(local_n);
    MPI_Allgather(&local_n_int, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int run = 0;
    for (int r = 0; r < size; ++r) { displs[r] = run; run += counts[r]; }
    std::vector<Complex> global(static_cast<std::size_t>(global_n),
                                Complex(0.0, 0.0));
    MPI_Allgatherv(local, local_n_int, MPI_C_DOUBLE_COMPLEX,
                   global.data(), counts.data(), displs.data(),
                   MPI_C_DOUBLE_COMPLEX, comm);
    return global;
}

std::vector<Complex>
deterministic_global_vector(std::size_t n, unsigned long seed) {
    std::vector<Complex> v(n);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& z : v) z = Complex(nd(gen), nd(gen));
    return v;
}

double max_abs_diff(const std::vector<Complex>& a,
                    const std::vector<Complex>& b) {
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

// Drive the full distributed-vs-reference comparison for a single
// (operator, sector_index, seeds).
void check_sector(int N, double J, bool periodic, std::size_t sector_idx,
                  std::initializer_list<unsigned long> seeds) {
    auto op  = make_heisenberg_translation_op(N, J, periodic);
    auto ref = build_reference(*op, sector_idx);

    DistributedSymmetryOperator dop(op, sector_idx, MPI_COMM_WORLD);

    REQUIRE(dop.global_dim() == ref.orbit_reps.size());

    // Slab self-consistency.
    std::uint64_t total = 0;
    std::uint64_t my_n  = dop.local_size();
    MPI_Allreduce(&my_n, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    REQUIRE(total == dop.global_dim());

    // Reconstruct the rank-major orbit ordering -- LPT scrambles orbit ids
    // across ranks, so x_local[k] corresponds to orbit
    //   partition.rank_orbits[my_rank][k]
    // not orbit (local_offset + k) of the natural orbit ordering.
    const auto& partition = dop.partition();
    const int my_rank = dop.rank();
    const int n_ranks = dop.comm_size();

    // Build the global rank-major -> orbit_id permutation (replicated).
    std::vector<std::size_t> rank_major_to_orbit(dop.global_dim(), 0);
    for (int r = 0; r < n_ranks; ++r) {
        for (std::size_t k = 0; k < partition.rank_orbits[r].size(); ++k) {
            const std::size_t global_pos = partition.rank_offsets[r] + k;
            rank_major_to_orbit[global_pos] = partition.rank_orbits[r][k];
        }
    }

    for (unsigned long seed : seeds) {
        // Generate the input in NATURAL orbit ordering, then permute into
        // rank-major slabs to feed the operator.
        auto x_natural = deterministic_global_vector(
            static_cast<std::size_t>(dop.global_dim()), seed);

        std::vector<Complex> x_local(dop.local_size());
        for (std::uint64_t k = 0; k < dop.local_size(); ++k) {
            const std::size_t orbit_id = partition.rank_orbits[my_rank][k];
            x_local[k] = x_natural[orbit_id];
        }

        std::vector<Complex> y_local(dop.local_size(),
                                     Complex(0.0, 0.0));
        dop.apply(x_local.data(), y_local.data());

        // Gather as rank-major and unscramble back to natural orbit order.
        auto y_rankmajor = allgather_slabs(y_local.data(), dop.local_size(),
                                           dop.global_dim(), MPI_COMM_WORLD);
        std::vector<Complex> y_dist(dop.global_dim(), Complex(0.0, 0.0));
        for (std::size_t g = 0; g < y_rankmajor.size(); ++g) {
            y_dist[rank_major_to_orbit[g]] = y_rankmajor[g];
        }

        auto y_ref = apply_reference(ref, x_natural);

        const double err = max_abs_diff(y_dist, y_ref);
        INFO("N=" << N << " periodic=" << periodic
             << " sector=" << sector_idx
             << " global_dim=" << dop.global_dim()
             << " seed=" << seed
             << " err=" << err);
        REQUIRE(err <= 1e-10);
    }
}

}  // namespace

// -----------------------------------------------------------------------------
// N=4 OBC, Z_4 translation symmetry. OBC translation orbits are still
// well-defined (sites cyclically permute regardless of bond connectivity);
// some sectors will be zero-norm because the bonds break translation
// covariance, which is what we want to lock down: filtering and apply both
// agree with the dense reference.
// -----------------------------------------------------------------------------
TEST_CASE("DistributedSymmetryOperator: N=4 OBC, all Z_4 sectors",
          "[distributed_symmetry_operator][heisenberg][n4][obc]") {
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/4, /*J=*/1.0, /*periodic=*/false,
                     /*sector_idx=*/s, {1UL, 17UL, 12345UL});
    }
}

// -----------------------------------------------------------------------------
// N=4 PBC, all 4 momentum sectors. PBC is the actual home of translation
// symmetry; orbits map cleanly to momentum sectors.
// -----------------------------------------------------------------------------
TEST_CASE("DistributedSymmetryOperator: N=4 PBC, all Z_4 sectors",
          "[distributed_symmetry_operator][heisenberg][n4][pbc]") {
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                     /*sector_idx=*/s, {2UL, 31415UL});
    }
}

// -----------------------------------------------------------------------------
// N=6 PBC, all 6 momentum sectors. Larger orbit count exercises the LPT
// load-balancing on np>=2.
// -----------------------------------------------------------------------------
TEST_CASE("DistributedSymmetryOperator: N=6 PBC, all Z_6 sectors",
          "[distributed_symmetry_operator][heisenberg][n6][pbc]") {
    auto info = ed::sym::translation_group_1d(6);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/6, /*J=*/-1.5, /*periodic=*/true,
                     /*sector_idx=*/s, {99UL, 1729UL});
    }
}

// -----------------------------------------------------------------------------
// Halo plan + partition diagnostics on N=6 PBC, k=0 sector. We don't pin
// exact halo counts (they depend on the LPT load balance + np), but we
// require the plan exists and has self-consistent send/recv totals.
// -----------------------------------------------------------------------------
TEST_CASE("DistributedSymmetryOperator: halo plan diagnostics N=6 PBC",
          "[distributed_symmetry_operator][diagnostics]") {
    auto op = make_heisenberg_translation_op(/*N=*/6, /*J=*/1.0,
                                             /*periodic=*/true);
    DistributedSymmetryOperator dop(op, /*sector_idx=*/0, MPI_COMM_WORLD);

    int size = 0; MPI_Comm_size(MPI_COMM_WORLD, &size);

    REQUIRE(dop.partition().n_ranks == size);

    const auto* plan = dop.halo_plan();
    REQUIRE(plan != nullptr);
    REQUIRE(plan->rank() == dop.rank());
    REQUIRE(plan->comm_size() == dop.comm_size());

    // No rank should claim to send to / receive from itself.
    REQUIRE(plan->recv_counts()[dop.rank()] == 0);
    REQUIRE(plan->send_counts()[dop.rank()] == 0);

    // Sum of recv totals across ranks == sum of send totals
    // (every "recv" has a matching "send" across the wire).
    std::uint64_t my_recv = plan->recv_total();
    std::uint64_t my_send = plan->send_total();
    std::uint64_t total_recv = 0, total_send = 0;
    MPI_Allreduce(&my_recv, &total_recv, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&my_send, &total_send, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    REQUIRE(total_recv == total_send);

    // local_nnz across ranks sums to a reasonable count (>= global_dim
    // since the diagonal of H_q is generally nonzero).
    std::uint64_t my_nnz = dop.local_nnz();
    std::uint64_t total_nnz = 0;
    MPI_Allreduce(&my_nnz, &total_nnz, 1, MPI_UINT64_T, MPI_SUM,
                  MPI_COMM_WORLD);
    REQUIRE(total_nnz >= dop.global_dim());
}

// -----------------------------------------------------------------------------
// Phase F: optional Sz (popcount) filter on the orbit basis.
//
// Site permutations preserve popcount, so restricting the symm-projected
// basis to "orbits whose representative has popcount == n_up" yields a
// closed sub-block. We verify:
//
//   1. Setting `info.n_up = k` reduces global_dim() to exactly the count
//      of unfiltered orbits whose representative has popcount k.
//   2. The Sz-filtered apply matches the unfiltered apply restricted
//      to the popcount-k subspace (componentwise, modulo orbit
//      reindexing).
//   3. Sum over k=0..N of Sz-filtered global_dim == unfiltered global_dim
//      (partition of unity across Sz blocks).
//   4. n_up == 0 (or N) gives global_dim == 1 (the unique all-down /
//      all-up state, which is its own translation orbit).
// -----------------------------------------------------------------------------

// Build a fresh op with the Sz filter applied to symmetry_info.
std::shared_ptr<Operator>
make_heisenberg_translation_op_sz(int N, double J, bool periodic, int n_up) {
    auto op = make_heisenberg_translation_op(N, J, periodic);
    op->symmetry_info.n_up = n_up;
    return op;
}

void check_sz_filter(int N, double J, bool periodic,
                     std::size_t sector_idx,
                     std::initializer_list<unsigned long> seeds) {
    // Unfiltered reference (full symm-projected basis).
    auto op_full = make_heisenberg_translation_op(N, J, periodic);
    DistributedSymmetryOperator dop_full(op_full, sector_idx,
                                          MPI_COMM_WORLD);
    const auto& reps_full = dop_full.orbit_reps();
    const std::uint64_t full_dim = dop_full.global_dim();

    // Per-Sz expected dim count + Sz partition of unity.
    std::vector<std::uint64_t> per_sz_count(N + 1, 0);
    for (auto rep : reps_full) {
        per_sz_count[__builtin_popcountll(rep)]++;
    }
    std::uint64_t sum_sz = 0;
    for (auto c : per_sz_count) sum_sz += c;
    REQUIRE(sum_sz == full_dim);

    // Map from (popcount-filtered orbit index in the Sz-restricted dop)
    // to its position in the unfiltered orbit_reps array. Build it
    // for each Sz value and verify dims + apply.
    for (int n_up = 0; n_up <= N; ++n_up) {
        auto op_sz = make_heisenberg_translation_op_sz(N, J, periodic, n_up);
        DistributedSymmetryOperator dop_sz(op_sz, sector_idx, MPI_COMM_WORLD);

        INFO("N=" << N << " periodic=" << periodic
             << " sector=" << sector_idx
             << " n_up=" << n_up
             << " filtered_dim=" << dop_sz.global_dim()
             << " expected=" << per_sz_count[n_up]);
        REQUIRE(dop_sz.global_dim() == per_sz_count[n_up]);

        if (per_sz_count[n_up] == 0) continue;

        // Reps of the Sz-filtered dop must be a popcount-k subset of
        // reps_full, in the SAME order (BFS visits states in increasing
        // index order in both ctors).
        const auto& reps_sz = dop_sz.orbit_reps();
        std::vector<std::size_t> sz_to_full;
        sz_to_full.reserve(reps_sz.size());
        std::size_t cursor = 0;
        for (auto r : reps_sz) {
            REQUIRE(static_cast<int>(__builtin_popcountll(r)) == n_up);
            while (cursor < reps_full.size() && reps_full[cursor] != r) {
                ++cursor;
            }
            REQUIRE(cursor < reps_full.size());
            sz_to_full.push_back(cursor++);
        }
        REQUIRE(sz_to_full.size() == reps_sz.size());

        // Apply check: drive the Sz-filtered dop with a deterministic
        // vector and compare against the unfiltered apply restricted
        // to popcount-k orbit indices.
        const auto& part_sz   = dop_sz.partition();
        const auto& part_full = dop_full.partition();
        const int my_rank = dop_sz.rank();
        const int n_ranks = dop_sz.comm_size();

        // rank-major -> orbit_id permutations for both ops.
        std::vector<std::size_t> rm_to_orbit_sz(dop_sz.global_dim(), 0);
        for (int r = 0; r < n_ranks; ++r) {
            for (std::size_t k = 0; k < part_sz.rank_orbits[r].size(); ++k) {
                rm_to_orbit_sz[part_sz.rank_offsets[r] + k] =
                    part_sz.rank_orbits[r][k];
            }
        }
        std::vector<std::size_t> rm_to_orbit_full(full_dim, 0);
        for (int r = 0; r < n_ranks; ++r) {
            for (std::size_t k = 0; k < part_full.rank_orbits[r].size(); ++k) {
                rm_to_orbit_full[part_full.rank_offsets[r] + k] =
                    part_full.rank_orbits[r][k];
            }
        }

        for (unsigned long seed : seeds) {
            // Generate a vector in NATURAL Sz-orbit ordering.
            auto x_sz_natural = deterministic_global_vector(
                static_cast<std::size_t>(dop_sz.global_dim()), seed);

            // Lift to the unfiltered basis: zero everywhere, copy the
            // popcount-k entries from x_sz_natural at positions
            // sz_to_full[i].
            std::vector<Complex> x_full_natural(full_dim,
                                                Complex(0.0, 0.0));
            for (std::size_t i = 0; i < x_sz_natural.size(); ++i) {
                x_full_natural[sz_to_full[i]] = x_sz_natural[i];
            }

            // Drive Sz-filtered dop.
            std::vector<Complex> x_sz_local(dop_sz.local_size());
            for (std::uint64_t k = 0; k < dop_sz.local_size(); ++k) {
                x_sz_local[k] = x_sz_natural[part_sz.rank_orbits[my_rank][k]];
            }
            std::vector<Complex> y_sz_local(dop_sz.local_size(),
                                             Complex(0.0, 0.0));
            dop_sz.apply(x_sz_local.data(), y_sz_local.data());
            auto y_sz_rm = allgather_slabs(
                y_sz_local.data(), dop_sz.local_size(),
                dop_sz.global_dim(), MPI_COMM_WORLD);
            std::vector<Complex> y_sz_natural(dop_sz.global_dim(),
                                               Complex(0.0, 0.0));
            for (std::size_t g = 0; g < y_sz_rm.size(); ++g) {
                y_sz_natural[rm_to_orbit_sz[g]] = y_sz_rm[g];
            }

            // Drive unfiltered dop with the lifted vector.
            std::vector<Complex> x_full_local(dop_full.local_size());
            for (std::uint64_t k = 0; k < dop_full.local_size(); ++k) {
                x_full_local[k] =
                    x_full_natural[part_full.rank_orbits[my_rank][k]];
            }
            std::vector<Complex> y_full_local(dop_full.local_size(),
                                                Complex(0.0, 0.0));
            dop_full.apply(x_full_local.data(), y_full_local.data());
            auto y_full_rm = allgather_slabs(
                y_full_local.data(), dop_full.local_size(),
                dop_full.global_dim(), MPI_COMM_WORLD);
            std::vector<Complex> y_full_natural(full_dim,
                                                  Complex(0.0, 0.0));
            for (std::size_t g = 0; g < y_full_rm.size(); ++g) {
                y_full_natural[rm_to_orbit_full[g]] = y_full_rm[g];
            }

            // Compare: y_full_natural projected onto the popcount-k
            // orbit indices must equal y_sz_natural componentwise.
            // (Because [H, popcount] = 0, the unfiltered apply leaves
            // the popcount-k subspace invariant.)
            double err = 0.0;
            for (std::size_t i = 0; i < y_sz_natural.size(); ++i) {
                err = std::max(
                    err,
                    std::abs(y_sz_natural[i] - y_full_natural[sz_to_full[i]]));
            }
            // Also: the orthogonal complement (popcount != k) of the
            // unfiltered output, when driven by the lifted vector,
            // must be ZERO -- this is the Sz-conservation check.
            std::set<std::size_t> popcount_k_positions(sz_to_full.begin(),
                                                        sz_to_full.end());
            double leak = 0.0;
            for (std::size_t i = 0; i < full_dim; ++i) {
                if (popcount_k_positions.count(i) == 0) {
                    leak = std::max(leak, std::abs(y_full_natural[i]));
                }
            }
            INFO("N=" << N << " periodic=" << periodic
                 << " sector=" << sector_idx << " n_up=" << n_up
                 << " seed=" << seed
                 << " err=" << err << " leak=" << leak);
            REQUIRE(err <= 1e-10);
            REQUIRE(leak <= 1e-10);
        }
    }
}

TEST_CASE("DistributedSymmetryOperator: Sz filter, N=4 PBC, k=0 sector",
          "[distributed_symmetry_operator][sz][n4][pbc]") {
    check_sz_filter(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                    /*sector_idx=*/0, {7UL, 31415UL});
}

TEST_CASE("DistributedSymmetryOperator: Sz filter, N=4 PBC, all sectors",
          "[distributed_symmetry_operator][sz][n4][pbc]") {
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sz_filter(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                        /*sector_idx=*/s, {1UL});
    }
}

TEST_CASE("DistributedSymmetryOperator: Sz filter, N=6 PBC, k=0 sector",
          "[distributed_symmetry_operator][sz][n6][pbc]") {
    check_sz_filter(/*N=*/6, /*J=*/-1.5, /*periodic=*/true,
                    /*sector_idx=*/0, {99UL});
}

TEST_CASE("DistributedSymmetryOperator: Sz filter, all-up / all-down "
          "give global_dim==1",
          "[distributed_symmetry_operator][sz][edge]") {
    for (int N : {4, 6}) {
        for (int n_up : {0, N}) {
            auto op = make_heisenberg_translation_op_sz(
                N, /*J=*/1.0, /*periodic=*/true, n_up);
            DistributedSymmetryOperator dop(op, /*sector_idx=*/0,
                                              MPI_COMM_WORLD);
            INFO("N=" << N << " n_up=" << n_up
                 << " global_dim=" << dop.global_dim());
            REQUIRE(dop.global_dim() == 1);
        }
    }
}

// -----------------------------------------------------------------------------
// Phase G: bare FixedSz route via the trivial 1-element symmetry group.
//
// The `qed.diag(H, device='mpi', sz=k)` path (without symmetry=) routes
// through DistributedSymmetryOperator with a TRIVIAL one-element group
// (identity only) + the Phase F popcount filter. With |G|=1 every orbit
// is a singleton, so the popcount-filtered orbit basis IS exactly the
// C(N, k) binomial basis (the same sub-block FixedSzOperator carries
// in-process). We verify:
//
//   1. global_dim() == binomial(N, k) for every k in [0, N].
//   2. orbit_reps() are exactly the popcount-k states in lex order.
//   3. The apply matches the unfiltered (full Hilbert) DistributedOperator
//      apply restricted to the popcount-k subspace -- i.e., Sz is
//      conserved by H on the bare FixedSz path.
// -----------------------------------------------------------------------------

std::shared_ptr<Operator>
make_heisenberg_trivial_op_sz(int N, double J, bool periodic, int n_up) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    // Trivial group: identity permutation only.
    std::vector<int> identity(N);
    for (int i = 0; i < N; ++i) identity[i] = i;
    op->symmetry_info = ed::sym::group_from_generators(
        N, {identity}, /*sector_quantum_numbers=*/{});
    op->symmetry_info.n_up = n_up;
    return op;
}

std::uint64_t binomial(int n, int k) {
    if (k < 0 || k > n) return 0;
    std::uint64_t r = 1;
    for (int i = 0; i < k; ++i) {
        r = r * static_cast<std::uint64_t>(n - i) /
            static_cast<std::uint64_t>(i + 1);
    }
    return r;
}

TEST_CASE("DistributedSymmetryOperator: Phase G (trivial group + sz) "
          "is the binomial basis",
          "[distributed_symmetry_operator][trivial_group][sz][phaseG]") {
    for (int N : {4, 6}) {
        // Reference: the unfiltered apply on the full Hilbert space
        // (no symmetry, no Sz). We reconstruct it densely on rank 0
        // for the comparison since DistributedOperator's apply is
        // tested elsewhere and we just need a serial reference here.
        auto op_full_serial = std::shared_ptr<Operator>(
            ed_tests::build_heisenberg_chain(
                static_cast<uint64_t>(N), 1.0, /*periodic=*/true).release());

        const std::uint64_t dim = 1ULL << N;
        // Build the dense Hamiltonian once; the apply we want is
        // y_full = H * x_full via op_full_serial->apply (per column).
        // Cheaper: just call op->apply directly on the lifted vectors.

        for (int n_up = 0; n_up <= N; ++n_up) {
            auto op_sz = make_heisenberg_trivial_op_sz(
                N, /*J=*/1.0, /*periodic=*/true, n_up);
            DistributedSymmetryOperator dop(op_sz, /*sector_idx=*/0,
                                              MPI_COMM_WORLD);

            const std::uint64_t expected_dim = binomial(N, n_up);
            INFO("N=" << N << " n_up=" << n_up
                 << " global_dim=" << dop.global_dim()
                 << " expected=" << expected_dim);
            REQUIRE(dop.global_dim() == expected_dim);

            // Reps must be exactly the popcount-n_up states in lex order.
            const auto& reps = dop.orbit_reps();
            std::vector<std::uint64_t> expected_reps;
            for (std::uint64_t b = 0; b < dim; ++b) {
                if (static_cast<int>(__builtin_popcountll(b)) == n_up) {
                    expected_reps.push_back(b);
                }
            }
            REQUIRE(reps.size() == expected_reps.size());
            for (std::size_t i = 0; i < reps.size(); ++i) {
                REQUIRE(reps[i] == expected_reps[i]);
            }

            // Orbit sizes must all be 1 (singleton orbits) and norms
            // must all be 1 (trivial projection: chi_q(e) = 1).
            for (auto sz : dop.orbit_sizes()) REQUIRE(sz == 1);
            for (auto N_i : dop.orbit_norms_sq()) {
                REQUIRE(std::abs(N_i - 1.0) <= 1e-12);
            }

            if (expected_dim == 0) continue;

            // Apply check: drive the trivial-group dop with a
            // deterministic vector and compare against the full-Hilbert
            // serial apply restricted to popcount-n_up positions.
            const auto& part = dop.partition();
            const int my_rank = dop.rank();

            auto x_natural = deterministic_global_vector(
                static_cast<std::size_t>(dop.global_dim()), /*seed=*/N * 100 + n_up);

            // Lift to full Hilbert vector.
            std::vector<Complex> x_full(dim, Complex(0.0, 0.0));
            for (std::size_t i = 0; i < x_natural.size(); ++i) {
                x_full[expected_reps[i]] = x_natural[i];
            }

            // Distributed apply.
            std::vector<Complex> x_local(dop.local_size());
            for (std::uint64_t k = 0; k < dop.local_size(); ++k) {
                x_local[k] = x_natural[part.rank_orbits[my_rank][k]];
            }
            std::vector<Complex> y_local(dop.local_size(), Complex(0.0, 0.0));
            dop.apply(x_local.data(), y_local.data());
            auto y_rm = allgather_slabs(y_local.data(), dop.local_size(),
                                         dop.global_dim(), MPI_COMM_WORLD);
            std::vector<Complex> y_dist(dop.global_dim(), Complex(0.0, 0.0));
            for (std::size_t g = 0; g < y_rm.size(); ++g) {
                std::size_t orb_id = 0;
                for (int r = 0; r < dop.comm_size(); ++r) {
                    if (g >= part.rank_offsets[r] &&
                        g <  part.rank_offsets[r] + part.rank_orbits[r].size()) {
                        orb_id = part.rank_orbits[r][g - part.rank_offsets[r]];
                        break;
                    }
                }
                y_dist[orb_id] = y_rm[g];
            }

            // Serial reference: y_full = H * x_full via op->apply.
            std::vector<Complex> y_full(dim, Complex(0.0, 0.0));
            op_full_serial->apply(x_full.data(), y_full.data(), dim);

            // Compare popcount-n_up positions.
            double err = 0.0;
            for (std::size_t i = 0; i < expected_reps.size(); ++i) {
                err = std::max(err,
                               std::abs(y_dist[i] - y_full[expected_reps[i]]));
            }
            // Sz conservation: leak into popcount != n_up positions
            // of y_full must be zero.
            double leak = 0.0;
            for (std::uint64_t b = 0; b < dim; ++b) {
                if (static_cast<int>(__builtin_popcountll(b)) != n_up) {
                    leak = std::max(leak, std::abs(y_full[b]));
                }
            }
            INFO("N=" << N << " n_up=" << n_up
                 << " err=" << err << " leak=" << leak);
            REQUIRE(err <= 1e-10);
            REQUIRE(leak <= 1e-10);
        }
    }
}

// -----------------------------------------------------------------------------
// Custom main: MPI_Init + Catch2 + MPI_Finalize.
// Same pattern as the other Phase 3b MPI tests.
// -----------------------------------------------------------------------------
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
