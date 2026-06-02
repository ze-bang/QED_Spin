// =============================================================================
// test_comm_plan (P4)
//
// MPI lockdown for ed::matvec::dist::CommPlan -- the policy-agnostic
// halo-exchange schedule extracted from DistributedOperator's
// build_comm_pattern_ for the unified MemSpace=DistributedHost matvec lane.
//
// The test is self-contained (no Operator / construct_ham dependency): it
// builds a Heisenberg N=8 PBC TermStorage by hand, extracts the flip
// patterns via flip_patterns_from_terms, builds the CommPlan, and validates
// the halo exchange END-TO-END at np in {1, 2, 4}:
//
//   * Geometry: sum of per-rank local_n equals global_dim; balanced_slab
//     agrees with the plan's local_offset/local_n.
//   * Conservation: sum of total_send across ranks equals sum of total_recv
//     (every requested column is served by exactly one owner).
//   * Correctness (complex + real lanes): for a deterministic global vector
//     known to every rank, ``exchange()`` fills recv_buf such that for every
//     (local row XOR flip pattern) that lands off-rank,
//     recv_buf[recv_index_of(col)] equals the true global amplitude at col.
//     This is the exact invariant the GATHER-form SpMV inner loop relies on.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/matvec/comm_plan.h>
#include <ed/matvec/term_storage.h>
#include "common/test_harness.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <mpi.h>

using ed::matvec::dist::CommPlan;
using ed::matvec::dist::flip_patterns_from_terms;
using ed::matvec::TermStorage;
using Complex = std::complex<double>;

namespace {

// Heisenberg chain term storage: Sz Sz (diagonal, p=0) + 0.5(S+S- + S-S+)
// (off-diagonal, p = (1<<i)^(1<<j)) on every bond.
TermStorage heisenberg_terms(int N, bool pbc) {
    TermStorage ts;
    const int last = pbc ? N : N - 1;
    for (int i = 0; i < last; ++i) {
        const int j = (i + 1) % N;
        ts.add_diag_two_body(i, j, Complex(1.0, 0.0));
        ts.add_offdiag_two_body(i, j, /*S+*/ 0, /*S-*/ 1, Complex(0.5, 0.0));
        ts.add_offdiag_two_body(i, j, /*S-*/ 1, /*S+*/ 0, Complex(0.5, 0.0));
    }
    return ts;
}

// Deterministic global amplitude at index g -- identical on every rank, so
// each rank can fill its slab and independently know the true off-rank
// values the halo should deliver.
Complex amp(std::uint64_t g) {
    const double re = static_cast<double>((g * 2654435761ULL) % 1000) - 500.0;
    const double im = static_cast<double>((g * 40503ULL + 7) % 1000) - 500.0;
    return Complex(re * 1e-3, im * 1e-3);
}
double amp_real(std::uint64_t g) { return amp(g).real(); }

}  // namespace

// -----------------------------------------------------------------------------
TEST_CASE("CommPlan: geometry + conservation invariants", "[comm_plan][mpi]") {
    MPI_Comm comm = MPI_COMM_WORLD;
    int size = 0, rank = 0;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    for (int N : {4, 6, 8}) {
        const std::uint64_t global_dim = (1ULL << N);
        const auto patterns = flip_patterns_from_terms(heisenberg_terms(N, true));

        CommPlan plan;
        plan.build(comm, global_dim, patterns);

        // local_n partitions the row space exactly.
        std::uint64_t sum_local = 0;
        std::uint64_t my_local  = plan.local_n();
        MPI_Allreduce(&my_local, &sum_local, 1, MPI_UINT64_T, MPI_SUM, comm);
        REQUIRE(sum_local == global_dim);

        // plan geometry matches the static balanced_slab.
        std::uint64_t off = 0, n = 0;
        CommPlan::balanced_slab(global_dim, rank, size, off, n);
        REQUIRE(plan.local_offset() == off);
        REQUIRE(plan.local_n() == n);

        // Every requested column is served by exactly one owner:
        // sum(total_send) == sum(total_recv).
        long long my_send = plan.total_send();
        long long my_recv = plan.total_recv();
        long long sum_send = 0, sum_recv = 0;
        MPI_Allreduce(&my_send, &sum_send, 1, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&my_recv, &sum_recv, 1, MPI_LONG_LONG, MPI_SUM, comm);
        REQUIRE(sum_send == sum_recv);
    }
}

// -----------------------------------------------------------------------------
TEST_CASE("CommPlan: halo exchange delivers correct off-rank amplitudes",
          "[comm_plan][mpi]") {
    MPI_Comm comm = MPI_COMM_WORLD;

    for (int N : {4, 6, 8}) {
        const std::uint64_t global_dim = (1ULL << N);
        const auto terms    = heisenberg_terms(N, true);
        const auto patterns = flip_patterns_from_terms(terms);

        CommPlan plan;
        plan.build(comm, global_dim, patterns);

        const std::uint64_t off = plan.local_offset();
        const std::uint64_t loc = plan.local_n();

        // ---- complex lane ----------------------------------------------
        std::vector<Complex> v_local(loc);
        for (std::uint64_t i = 0; i < loc; ++i) v_local[i] = amp(off + i);

        std::vector<Complex> recv_buf;
        plan.exchange<Complex>(v_local.data(), recv_buf);

        for (std::uint64_t r_local = 0; r_local < loc; ++r_local) {
            const std::uint64_t r = off + r_local;
            for (std::uint64_t p : patterns) {
                const std::uint64_t c = r ^ p;
                if (c >= global_dim) continue;
                if (plan.is_local(c)) continue;  // read from local slab
                const std::size_t k = plan.recv_index_of(c);
                REQUIRE(k != ed::core::SortedUint64Index::kNotFound);
                REQUIRE(std::abs(recv_buf[k] - amp(c)) < 1e-15);
            }
        }

        // ---- real lane (MPI_DOUBLE halo) -------------------------------
        std::vector<double> vr_local(loc);
        for (std::uint64_t i = 0; i < loc; ++i) vr_local[i] = amp_real(off + i);

        std::vector<double> recv_real;
        plan.exchange<double>(vr_local.data(), recv_real);

        for (std::uint64_t r_local = 0; r_local < loc; ++r_local) {
            const std::uint64_t r = off + r_local;
            for (std::uint64_t p : patterns) {
                const std::uint64_t c = r ^ p;
                if (c >= global_dim) continue;
                if (plan.is_local(c)) continue;
                const std::size_t k = plan.recv_index_of(c);
                REQUIRE(k != ed::core::SortedUint64Index::kNotFound);
                REQUIRE(std::abs(recv_real[k] - amp_real(c)) < 1e-15);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Catch2 + MPI runner: rank 0 owns stdout; a failure on any rank propagates
// to the CTest exit code via MPI_Allreduce(MAX).
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Suppress non-zero ranks' Catch2 output to keep the CTest log tidy.
    if (rank != 0) {
        FILE* dummy = std::freopen("/dev/null", "w", stdout);
        (void)dummy;
    }

    int local_rc = Catch::Session().run(argc, argv);

    int global_rc = 0;
    MPI_Allreduce(&local_rc, &global_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_rc;
}
