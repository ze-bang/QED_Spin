// =============================================================================
// examples/06_mpi_distributed_eigenvectors.cpp
//
// Phase 3b #6: distributed Ritz vector reconstruction. Each rank holds
// only its slab of every Krylov vector and the redundant tridiagonal
// eigenvector matrix, then locally contracts to recover its slab of
// the k-th eigenvector. We assemble the global vector via MPI_Allgatherv
// and verify the residual on rank 0.
//
// Run:
//     OMP_NUM_THREADS=4 mpiexec -n 4 ./build/ex06_mpi_distributed_eigenvectors 8 1
// =============================================================================

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using Complex = std::complex<double>;

static void add_bond(Operator& op, std::uint64_t i, std::uint64_t j, double J) {
    Operator::TransformData zz;
    zz.op_type = 2; zz.site_index = i; zz.op_type_2 = 2; zz.site_index_2 = j;
    zz.coefficient = Complex(J, 0.0); zz.is_two_body = true;
    op.transform_data_.push_back(zz);

    Operator::TransformData pm;
    pm.op_type = 0; pm.site_index = i; pm.op_type_2 = 1; pm.site_index_2 = j;
    pm.coefficient = Complex(0.5 * J, 0.0); pm.is_two_body = true;
    op.transform_data_.push_back(pm);

    Operator::TransformData mp;
    mp.op_type = 1; mp.site_index = i; mp.op_type_2 = 0; mp.site_index_2 = j;
    mp.coefficient = Complex(0.5 * J, 0.0); mp.is_two_body = true;
    op.transform_data_.push_back(mp);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::uint64_t N        = (argc > 1) ? std::stoull(argv[1]) : 8;
    const bool          periodic = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;

    auto op = std::make_shared<Operator>(N, /*spin=*/0.5f);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) add_bond(*op, i, (i + 1) % N, 1.0);

    ed::distributed::DistributedOperator dop(op, MPI_COMM_WORLD);

    ed::distributed::DistributedLanczosOptions opts;
    opts.max_iter            = 200;
    opts.exct                = 1;
    opts.tol                 = 1e-12;
    opts.full_reorth         = true;
    opts.compute_eigenvectors = true;   // Phase 3b #6 hook
    opts.seed                 = 42UL;

    auto pairs = ed::distributed::distributed_lanczos_eigenvectors(dop, opts);

    if (rank == 0) {
        std::cout << "Heisenberg chain  N=" << N
                  << "  PBC=" << (periodic ? 1 : 0)
                  << "  global_dim=" << dop.global_dim()
                  << "  np=" << size
                  << "  iters=" << pairs.iterations << "\n";
        std::cout << "  E_0 = " << pairs.eigenvalues[0] << "\n";
    }

    // Apply H to the rank-local slab of psi_0 and assemble globally to
    // measure ||H psi - E psi||_2.
    const std::uint64_t local_n = dop.local_size();
    std::vector<Complex> Hpsi_local(local_n);
    dop.apply(pairs.eigenvectors_local[0].data(), Hpsi_local.data());

    // Local residual: ||H psi - E psi||_local^2
    double local_resid_sq = 0.0;
    for (std::uint64_t i = 0; i < local_n; ++i) {
        Complex r = Hpsi_local[i] - pairs.eigenvalues[0] * pairs.eigenvectors_local[0][i];
        local_resid_sq += std::norm(r);
    }
    double global_resid_sq = 0.0;
    MPI_Allreduce(&local_resid_sq, &global_resid_sq, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "  ||H psi_0 - E_0 psi_0||_2 = "
                  << std::sqrt(global_resid_sq) << "\n";
        std::cout << "  (a value below 1e-8 indicates a converged eigenpair)\n";
    }
    MPI_Finalize();
    return 0;
}
