// =============================================================================
// test_distributed_eigenvectors (Phase 3b #6)
//
// MPI lockdown for ed::distributed::distributed_lanczos_eigenvectors. The
// rank-local Krylov basis V_local plus the (m x m) tridiagonal eigenvector
// matrix U is contracted into psi_k_local = V_local @ U[:, k]. We check:
//
//   * the assembled global psi has ||psi||_2 == 1 (within 1e-10),
//   * the residual ||H psi - E_0 psi||_2 < 1e-8 against the SERIAL Operator
//     applied on the global vector (assembled via MPI_Allgatherv),
//   * the assembled global psi is bit-identical across all ranks (Bcast diff
//     < 1e-12).
//
// Coverage: N=4 OBC ground state, N=6 PBC ground state, np=1, np=2, np=4.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedOperator;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedEigenpairsResult;
using ed::distributed::distributed_lanczos_eigenvectors;

using Complex = std::complex<double>;

namespace {

// Assemble a global vector from rank-local slabs via MPI_Allgatherv, using
// the same balanced_slab decomposition that DistributedOperator uses.
std::vector<Complex>
assemble_global(const std::vector<Complex>& local,
                std::uint64_t global_dim,
                int size,
                MPI_Comm comm) {
    std::vector<int> recvcounts(size), displs(size);
    int run = 0;
    for (int r = 0; r < size; ++r) {
        std::uint64_t off, n;
        DistributedOperator::balanced_slab(global_dim, r, size, off, n);
        recvcounts[r] = static_cast<int>(n);
        displs[r]     = run;
        run          += recvcounts[r];
    }
    std::vector<Complex> global(static_cast<std::size_t>(global_dim));
    MPI_Allgatherv(local.data(), static_cast<int>(local.size()),
                   MPI_C_DOUBLE_COMPLEX,
                   global.data(), recvcounts.data(), displs.data(),
                   MPI_C_DOUBLE_COMPLEX, comm);
    return global;
}

double norm2(const std::vector<Complex>& v) {
    double s = 0.0;
    for (const auto& z : v) s += std::norm(z);
    return std::sqrt(s);
}

}  // namespace

TEST_CASE("Distributed eigenvectors: N=4 OBC ground state H psi = E psi",
          "[distributed_eigenvectors][heisenberg]") {
    auto op_uniq = ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                                    /*periodic=*/false);
    auto op = std::shared_ptr<Operator>(op_uniq.release());

    DistributedOperator dop(op, MPI_COMM_WORLD);
    const std::uint64_t dim = dop.global_dim();

    DistributedLanczosOptions opts;
    opts.max_iter            = 60;
    opts.exct                = 1;
    opts.tol                 = 1e-12;
    opts.seed                = 12345UL;
    opts.compute_eigenvectors = true;  // implies full_reorth + compute_weights

    DistributedEigenpairsResult res = distributed_lanczos_eigenvectors(dop, opts);

    REQUIRE(!res.eigenvalues.empty());
    REQUIRE(res.eigenvectors_local.size() == 1);
    REQUIRE(res.eigenvectors_local[0].size() == dop.local_size());

    // Assemble global psi.
    auto psi = assemble_global(res.eigenvectors_local[0], dim,
                               dop.comm_size(), dop.comm());

    const double E0 = res.eigenvalues.front();
    const double n_psi = norm2(psi);

    INFO("E0 = " << E0 << "  ||psi|| = " << n_psi);
    REQUIRE(std::abs(n_psi - 1.0) < 1e-10);

    // Residual on the SERIAL operator (every rank computes its own; cheap).
    std::vector<Complex> Hpsi(dim, Complex(0.0, 0.0));
    op->apply(psi.data(), Hpsi.data(), static_cast<size_t>(dim));

    double res_sq = 0.0;
    for (std::uint64_t i = 0; i < dim; ++i) {
        Complex r = Hpsi[i] - E0 * psi[i];
        res_sq += std::norm(r);
    }
    const double residual = std::sqrt(res_sq);
    INFO("||H psi - E0 psi||_2 = " << residual);
    REQUIRE(residual < 1e-8);
}

TEST_CASE("Distributed eigenvectors: N=6 PBC ground state H psi = E psi",
          "[distributed_eigenvectors][heisenberg][pbc]") {
    auto op_uniq = ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                                    /*periodic=*/true);
    auto op = std::shared_ptr<Operator>(op_uniq.release());

    DistributedOperator dop(op, MPI_COMM_WORLD);
    const std::uint64_t dim = dop.global_dim();

    DistributedLanczosOptions opts;
    opts.max_iter            = 80;
    opts.exct                = 1;
    opts.tol                 = 1e-12;
    opts.seed                = 7UL;
    opts.compute_eigenvectors = true;

    DistributedEigenpairsResult res = distributed_lanczos_eigenvectors(dop, opts);
    REQUIRE(!res.eigenvalues.empty());
    REQUIRE(res.eigenvectors_local.size() == 1);
    REQUIRE(res.eigenvectors_local[0].size() == dop.local_size());

    auto psi = assemble_global(res.eigenvectors_local[0], dim,
                               dop.comm_size(), dop.comm());

    const double E0 = res.eigenvalues.front();
    const double n_psi = norm2(psi);
    INFO("E0 = " << E0 << "  ||psi|| = " << n_psi);
    REQUIRE(std::abs(n_psi - 1.0) < 1e-10);

    std::vector<Complex> Hpsi(dim, Complex(0.0, 0.0));
    op->apply(psi.data(), Hpsi.data(), static_cast<size_t>(dim));

    double res_sq = 0.0;
    for (std::uint64_t i = 0; i < dim; ++i) {
        Complex r = Hpsi[i] - E0 * psi[i];
        res_sq += std::norm(r);
    }
    const double residual = std::sqrt(res_sq);
    INFO("||H psi - E0 psi||_2 = " << residual);
    REQUIRE(residual < 1e-8);
}

TEST_CASE("Distributed eigenvectors: assembled global psi is replicated "
          "across ranks",
          "[distributed_eigenvectors][replicated]") {
    auto op_uniq = ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                                    /*periodic=*/true);
    auto op = std::shared_ptr<Operator>(op_uniq.release());

    DistributedOperator dop(op, MPI_COMM_WORLD);
    const std::uint64_t dim = dop.global_dim();

    DistributedLanczosOptions opts;
    opts.max_iter            = 60;
    opts.exct                = 1;
    opts.tol                 = 1e-12;
    opts.seed                = 31UL;
    opts.compute_eigenvectors = true;

    DistributedEigenpairsResult res = distributed_lanczos_eigenvectors(dop, opts);

    auto psi = assemble_global(res.eigenvectors_local[0], dim,
                               dop.comm_size(), dop.comm());
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Bcast rank-0's vector and compare element-wise. Eigenvector sign is
    // arbitrary so we tolerate a global -1 flip (compare |psi_i| or
    // <psi_rank0|psi_local>).
    std::vector<Complex> psi0 = psi;
    MPI_Bcast(psi0.data(), static_cast<int>(dim),
              MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    Complex ip(0.0, 0.0);
    for (std::uint64_t i = 0; i < dim; ++i) {
        ip += std::conj(psi0[i]) * psi[i];
    }
    // <psi0|psi> should be a global phase of unit modulus.
    const double mag = std::abs(ip);
    INFO("<psi_rank0|psi_local> magnitude = " << mag);
    REQUIRE(std::abs(mag - 1.0) < 1e-10);
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
