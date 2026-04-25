// =============================================================================
// examples/05_mpi_distributed_lanczos.cpp
//
// Rank-distributed ground-state Lanczos via the Phase 3b
// `ed::distributed` API. Same Heisenberg chain as ex01, but the Hilbert
// space is split into 1D row-slabs across MPI ranks.
//
// Build requires MPI (-DWITH_MPI=ON in the parent build).
//
// Run (single node, four ranks):
//
//     OMP_NUM_THREADS=4 mpiexec -n 4 ./build/ex05_mpi_distributed_lanczos 14 1
// =============================================================================

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>

#include <mpi.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

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

    const std::uint64_t N        = (argc > 1) ? std::stoull(argv[1]) : 14;
    const bool          periodic = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;

    auto op = std::make_shared<Operator>(N, /*spin=*/0.5f);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) add_bond(*op, i, (i + 1) % N, 1.0);

    ed::distributed::DistributedOperator dop(op, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Heisenberg chain  N=" << N
                  << "  PBC=" << (periodic ? 1 : 0)
                  << "  global_dim=" << dop.global_dim()
                  << "  np=" << size << "\n";
    }
    std::cout << "  rank=" << rank
              << "  local_offset=" << dop.local_offset()
              << "  local_n="      << dop.local_size()
              << "  plan_bytes="   << dop.plan_bytes() << "\n";

    ed::distributed::DistributedLanczosOptions opts;
    opts.max_iter    = 200;
    opts.exct        = 3;
    opts.tol         = 1e-12;
    opts.full_reorth = true;
    opts.seed        = 42UL;
    opts.verbose     = false;

    auto result = ed::distributed::distributed_lanczos(dop, opts);

    if (rank == 0) {
        std::cout << "Iterations: " << result.iterations << "\n";
        std::cout << "Lowest 3 eigenvalues (replicated on every rank):\n";
        for (std::size_t k = 0; k < result.eigenvalues.size() && k < 3; ++k) {
            std::cout << "  E[" << k << "] = " << result.eigenvalues[k] << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}
