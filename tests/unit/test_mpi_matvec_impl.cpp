// =============================================================================
// test_mpi_matvec_impl (P4)
//
// MPI lockdown for ed::matvec::dist::MpiMatVecImpl -- the
// MemSpace=DistributedHost matrix-free SpMV built on the reusable CommPlan +
// the GATHER kernel. Validates the distributed slab SpMV END-TO-END against
// an INDEPENDENT serial reference (the SCATTER kernel apply_terms over the
// full Hilbert space) at np in {1, 2, 4}.
//
// Self-contained: builds a Heisenberg N=4/6/8 PBC TermStorage by hand (no
// Operator / construct_ham dependency). The serial reference uses
// ed::matvec::kernel::apply_terms (SCATTER direction) so the test genuinely
// cross-checks the GATHER-form distributed apply against the orthogonal
// SCATTER kernel -- not just "the same kernel twice".
//
// Coverage:
//   * complex apply: distributed y slabs, MPI_Allgathered, vs serial
//     scatter apply on the full vector -- bit-for-bit to 1e-12.
//   * real apply: real-Hermitian fast path (MPI_DOUBLE halo) vs the real
//     part of the serial scatter apply.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/matvec/mpi_matvec_impl.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/term_storage.h>
#include "common/test_harness.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <mpi.h>

using ed::matvec::dist::MpiMatVecImpl;
using ed::matvec::TermStorage;
using Complex = std::complex<double>;

namespace {

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

// Deterministic global complex vector (identical on every rank).
std::vector<Complex> global_vector(std::uint64_t dim, unsigned long seed) {
    std::vector<Complex> v(dim);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& z : v) z = Complex(nd(gen), nd(gen));
    return v;
}

// Independent serial reference: y = H * v via the SCATTER kernel over the
// full Hilbert space (FullBasisPolicy). This is the orthogonal direction to
// the GATHER kernel the distributed apply uses.
std::vector<Complex> serial_scatter_apply(const TermStorage& ts, int N,
                                           const std::vector<Complex>& v,
                                           double spin_l) {
    ed::matvec::basis::FullBasisPolicy basis{static_cast<std::uint64_t>(N)};
    std::vector<Complex> y(v.size(), Complex(0.0, 0.0));
    ed::matvec::kernel::apply_terms(
        basis, spin_l,
        ts.diag_one_body, ts.offdiag_one_body, ts.diag_two_body,
        ts.mixed_two_body, ts.offdiag_two_body, ts.three_body,
        v.data(), y.data());
    return y;
}

std::vector<Complex> allgatherv_slabs(const Complex* local,
                                      std::uint64_t local_n,
                                      std::uint64_t global_dim,
                                      MPI_Comm comm) {
    int size = 0;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(size), displs(size);
    int local_n_int = static_cast<int>(local_n);
    MPI_Allgather(&local_n_int, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int run = 0;
    for (int r = 0; r < size; ++r) { displs[r] = run; run += counts[r]; }
    std::vector<Complex> global(static_cast<std::size_t>(global_dim));
    MPI_Allgatherv(local, local_n_int, MPI_C_DOUBLE_COMPLEX,
                   global.data(), counts.data(), displs.data(),
                   MPI_C_DOUBLE_COMPLEX, comm);
    return global;
}

std::vector<double> allgatherv_slabs_real(const double* local,
                                          std::uint64_t local_n,
                                          std::uint64_t global_dim,
                                          MPI_Comm comm) {
    int size = 0;
    MPI_Comm_size(comm, &size);
    std::vector<int> counts(size), displs(size);
    int local_n_int = static_cast<int>(local_n);
    MPI_Allgather(&local_n_int, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int run = 0;
    for (int r = 0; r < size; ++r) { displs[r] = run; run += counts[r]; }
    std::vector<double> global(static_cast<std::size_t>(global_dim));
    MPI_Allgatherv(local, local_n_int, MPI_DOUBLE,
                   global.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, comm);
    return global;
}

double max_abs_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

}  // namespace

// -----------------------------------------------------------------------------
TEST_CASE("MpiMatVecImpl: distributed complex SpMV matches serial scatter",
          "[mpi_matvec][mpi]") {
    MPI_Comm comm = MPI_COMM_WORLD;
    const double spin_l = 0.5;

    for (int N : {4, 6, 8}) {
        const std::uint64_t global_dim = (1ULL << N);
        const auto terms = heisenberg_terms(N, true);

        MpiMatVecImpl mv;
        mv.build(comm, static_cast<std::uint64_t>(N), terms, spin_l);
        REQUIRE(mv.global_dim() == global_dim);

        const auto v_global = global_vector(global_dim, 1234u + N);

        std::vector<Complex> v_local(mv.local_n());
        for (std::uint64_t i = 0; i < mv.local_n(); ++i) {
            v_local[i] = v_global[mv.local_offset() + i];
        }
        std::vector<Complex> y_local(mv.local_n());
        mv.apply_complex(v_local.data(), y_local.data());

        auto y_dist = allgatherv_slabs(y_local.data(), mv.local_n(),
                                       global_dim, comm);
        auto y_ref = serial_scatter_apply(terms, N, v_global, spin_l);
        REQUIRE(max_abs_diff(y_dist, y_ref) < 1e-12);
    }
}

// -----------------------------------------------------------------------------
TEST_CASE("MpiMatVecImpl: distributed real SpMV matches serial scatter (real part)",
          "[mpi_matvec][mpi]") {
    MPI_Comm comm = MPI_COMM_WORLD;
    const double spin_l = 0.5;

    for (int N : {4, 6, 8}) {
        const std::uint64_t global_dim = (1ULL << N);
        const auto terms = heisenberg_terms(N, true);

        MpiMatVecImpl mv;
        mv.build(comm, static_cast<std::uint64_t>(N), terms, spin_l);

        const auto v_global = global_vector(global_dim, 9876u + N);

        std::vector<double> vr_local(mv.local_n());
        for (std::uint64_t i = 0; i < mv.local_n(); ++i) {
            vr_local[i] = v_global[mv.local_offset() + i].real();
        }
        std::vector<double> yr_local(mv.local_n());
        mv.apply_real(vr_local.data(), yr_local.data());

        auto yr_dist = allgatherv_slabs_real(yr_local.data(), mv.local_n(),
                                             global_dim, comm);

        // Serial reference: scatter apply on the real-only global vector,
        // take real part (Heisenberg is real-Hermitian).
        std::vector<Complex> v_real(global_dim);
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            v_real[i] = Complex(v_global[i].real(), 0.0);
        }
        auto y_ref = serial_scatter_apply(terms, N, v_real, spin_l);

        double m = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            m = std::max(m, std::abs(yr_dist[i] - y_ref[i].real()));
        }
        REQUIRE(m < 1e-12);
    }
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
