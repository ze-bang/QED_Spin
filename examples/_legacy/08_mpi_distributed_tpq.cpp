// =============================================================================
// examples/08_mpi_distributed_tpq.cpp
//
// Phase 3b #8: distributed canonical Thermal Pure Quantum (cTPQ) state
// evolution. Each rank evolves its slab of |psi(beta)> in small
// `delta_beta` substeps using a Taylor-expanded e^{-(delta_beta/2) H},
// and we sample-average <H>(beta) at a list of target inverse
// temperatures.
//
// Run:
//     OMP_NUM_THREADS=2 mpiexec -n 4 ./build/ex08_mpi_distributed_tpq 6 1 4 8
// =============================================================================

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_tpq.h>

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

    const std::uint64_t N         = (argc > 1) ? std::stoull(argv[1]) : 6;
    const bool          periodic  = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;
    const int           n_groups  = (argc > 3) ? std::atoi(argv[3]) : 1;
    const int           n_samples = (argc > 4) ? std::atoi(argv[4]) : 8;

    auto op = std::make_shared<Operator>(N, /*spin=*/0.5f);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) add_bond(*op, i, (i + 1) % N, 1.0);

    ed::distributed::DistributedTpqOptions opts;
    opts.n_samples       = n_samples;
    opts.n_groups        = n_groups;
    opts.delta_beta      = 0.05;
    opts.taylor_order    = 30;
    opts.betas           = {0.5, 1.0, 2.0};
    opts.compute_variance = true;
    opts.seed_offset     = 12345UL;

    auto result = ed::distributed::distributed_tpq(op, opts, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Distributed TPQ  N=" << N
                  << "  PBC=" << (periodic ? 1 : 0)
                  << "  np=" << size
                  << "  groups=" << n_groups
                  << "  samples=" << n_samples
                  << "  samples_used=" << result.samples_used
                  << "  delta_beta=" << opts.delta_beta
                  << "  taylor_order=" << opts.taylor_order << "\n";
        std::cout << "  beta        E(beta)            var(beta) = <H^2> - <H>^2\n";
        for (std::size_t b = 0; b < opts.betas.size(); ++b) {
            std::cout << "  " << opts.betas[b]
                      << "    " << result.energy[b]
                      << "    " << (result.variance.empty() ? 0.0 : result.variance[b])
                      << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}
