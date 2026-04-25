// Phase 3b bootstrap: MPI-parallel matrix-free H·v vs serial Operator::apply.
// Run under MPI (e.g. `mpirun -n 4 ./test_distributed_matrix_free`); CTest
// registers an np=4 launch when WITH_MPI=ON.

#if defined(WITH_MPI) && WITH_MPI

#include "common/test_harness.h"

#include <ed/distributed/distributed_operator.h>

#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <vector>

using ed_tests::build_heisenberg_chain;
using ed_tests::Complex;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    constexpr uint64_t N = 8;
    constexpr uint64_t dim = 1ULL << N;
    auto op = build_heisenberg_chain(N, 1.0, true);

    ed::distributed::DistributedMatrixFreeOperator dist(*op, MPI_COMM_WORLD);

    if (dist.global_dim() != dim) {
        if (rank == 0) std::fprintf(stderr, "global_dim mismatch\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<Complex> v_full(dim);
    for (uint64_t i = 0; i < dim; ++i) {
        v_full[i] = Complex(0.01 * static_cast<double>(i % 17),
                            0.02 * static_cast<double>(i % 13));
    }

    const uint64_t loff = dist.local_row_offset();
    const uint64_t lcnt = dist.local_row_count();
    std::vector<Complex> v_local(lcnt);
    for (uint64_t k = 0; k < lcnt; ++k) {
        v_local[k] = v_full[loff + k];
    }

    std::vector<Complex> workspace;
    std::vector<Complex> y_local(lcnt);
    dist.apply(v_local.data(), workspace, y_local.data());

    std::vector<Complex> y_serial(dim);
    op->apply(v_full.data(), y_serial.data(), dim);

    int local_ok = 1;
    double max_err = 0.0;
    for (uint64_t k = 0; k < lcnt; ++k) {
        const Complex a = y_local[k];
        const Complex b = y_serial[loff + k];
        const double d =
            std::abs(a.real() - b.real()) + std::abs(a.imag() - b.imag());
        max_err = std::max(max_err, d);
    }
    constexpr double tol = 1e-11;
    if (max_err > tol) {
        local_ok = 0;
        if (rank == 0) {
            std::fprintf(stderr, "rank %d row-slice max err %g (tol %g)\n", rank,
                         max_err, tol);
        }
    }

    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);

    if (rank == 0 && !global_ok) {
        std::fprintf(stderr,
                     "test_distributed_matrix_free: slice mismatch on at least one rank\n");
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}

#endif
