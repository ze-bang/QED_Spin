// =============================================================================
// examples/07_mpi_distributed_ftlm.cpp
//
// Phase 3b #3 + #5: distributed FTLM with two-level parallelism
// (groups across samples; ranks across rows). Computes Z(beta) and the
// thermal energy <H>(beta) using H itself as the observable, then
// compares against the exact thermal energy on a small enough N.
//
// Run:
//     OMP_NUM_THREADS=2 mpiexec -n 4 ./build/ex07_mpi_distributed_ftlm 8 1 4
//   (here n_groups=4 -> one group per sample; if you pass n_groups=1
//    the same sample is replicated across every rank.)
// =============================================================================

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_ftlm.h>

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

    const std::uint64_t N         = (argc > 1) ? std::stoull(argv[1]) : 8;
    const bool          periodic  = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;
    const int           n_groups  = (argc > 3) ? std::atoi(argv[3]) : 1;
    const int           n_samples = (argc > 4) ? std::atoi(argv[4]) : 16;

    if (size % n_groups != 0) {
        if (rank == 0)
            std::cerr << "world size (" << size << ") must be a multiple of n_groups ("
                      << n_groups << ")\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto op = std::make_shared<Operator>(N, /*spin=*/0.5f);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) add_bond(*op, i, (i + 1) % N, 1.0);

    ed::distributed::DistributedFtlmOptions opts;
    opts.n_samples         = n_samples;
    opts.n_groups          = n_groups;
    opts.lanczos_max_iter  = 100;
    opts.betas             = {0.1, 0.5, 1.0, 2.0, 5.0};
    opts.observable_op     = op;          // <H>(beta): use H itself
    opts.seed_offset       = 12345UL;
    opts.verbose           = false;

    auto result = ed::distributed::distributed_ftlm(op, opts, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Distributed FTLM  N=" << N
                  << "  PBC=" << (periodic ? 1 : 0)
                  << "  np=" << size
                  << "  groups=" << n_groups
                  << "  samples=" << n_samples
                  << "  samples_used=" << result.samples_used << "\n";
        std::cout << "  beta        Z(beta)        <H>(beta) = N_O / N_Z\n";
        for (std::size_t b = 0; b < opts.betas.size(); ++b) {
            std::cout << "  " << opts.betas[b]
                      << "    " << result.Z[b]
                      << "    " << (result.O_expectation.empty() ? 0.0 : result.O_expectation[b])
                      << "\n";
        }
    }
    MPI_Finalize();
    return 0;
}
