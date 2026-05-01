// =============================================================================
// test_distributed_krylov_schur_gpu  (Phase B: device matrix MPI+GPU KS)
//
// MPI lockdown for `ed::distributed::distributed_krylov_schur_gpu`. Cross
// -checks the GPU Krylov-Schur result against the CPU reference produced
// by `distributed_krylov_schur` on the SAME inputs (operator, seeds,
// max_iter, tol) -- the GPU path is architecturally a different
// implementation but mathematically identical, so per-eigenvalue
// agreement to ~1e-8 is the realistic envelope (the locked Ritz pairs
// are computed against the same convergence tolerance, so the dominant
// noise source is the sub-tolerance residual rather than reduction
// rounding).
//
// Coverage:
//   * N=4 OBC: lowest 3 eigenvalues vs CPU KS.
//   * N=6 PBC: lowest 4 eigenvalues vs CPU KS.
//   * Replicated eigenvalues across all ranks.
//
// Run-time gating mirrors test_distributed_lanczos_gpu /
// test_distributed_ftlm_gpu.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_krylov_schur.h>
#include <ed/distributed/distributed_krylov_schur_gpu.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/multi_gpu.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#endif

using ed::distributed::distributed_krylov_schur;
using ed::distributed::distributed_krylov_schur_gpu;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedLanczosResult;
using ed::distributed::DistributedOperator;

namespace {

bool runtime_supports_gpu_ks() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_krylov_schur_gpu");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP distributed_krylov_schur_gpu");
        return false;
    }
    if (world_size > n_devices) {
        SUCCEED("world_size > visible device count; SKIP "
                "(multi-rank-on-same-device not exercised here)");
        return false;
    }
    return true;
#else
    return false;
#endif
}

}  // namespace

TEST_CASE("distributed_krylov_schur_gpu: N=4 OBC lowest 3 match CPU KS",
          "[distributed_krylov_schur_gpu][heisenberg]") {
    if (!runtime_supports_gpu_ks()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedLanczosOptions opts;
    opts.max_iter = 32;
    opts.exct     = 3;
    opts.tol      = 1e-10;
    opts.seed     = 13UL;

    DistributedOperator cpu_dop(op, MPI_COMM_WORLD);
    auto cpu = distributed_krylov_schur(cpu_dop, opts);
    auto gpu = distributed_krylov_schur_gpu(op, opts, MPI_COMM_WORLD);

    REQUIRE(cpu.eigenvalues.size() >= 3);
    REQUIRE(gpu.eigenvalues.size() >= 3);

    for (std::size_t k = 0; k < 3; ++k) {
        const double diff =
            std::abs(gpu.eigenvalues[k] - cpu.eigenvalues[k]);
        INFO("k=" << k << " E_cpu=" << cpu.eigenvalues[k]
             << " E_gpu=" << gpu.eigenvalues[k]
             << " diff=" << diff);
        // Both kernels lock against the same `tol = 1e-10`, so the
        // returned Ritz pairs are accurate to that tolerance plus a
        // small reduction-noise envelope.
        REQUIRE(diff < 1e-8);
    }
}

TEST_CASE("distributed_krylov_schur_gpu: N=6 PBC lowest 4 match CPU KS",
          "[distributed_krylov_schur_gpu][heisenberg][pbc]") {
    if (!runtime_supports_gpu_ks()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());

    DistributedLanczosOptions opts;
    opts.max_iter = 60;
    opts.exct     = 4;
    opts.tol      = 1e-10;
    opts.seed     = 71UL;

    DistributedOperator cpu_dop(op, MPI_COMM_WORLD);
    auto cpu = distributed_krylov_schur(cpu_dop, opts);
    auto gpu = distributed_krylov_schur_gpu(op, opts, MPI_COMM_WORLD);

    REQUIRE(cpu.eigenvalues.size() >= 4);
    REQUIRE(gpu.eigenvalues.size() >= 4);

    for (std::size_t k = 0; k < 4; ++k) {
        const double diff =
            std::abs(gpu.eigenvalues[k] - cpu.eigenvalues[k]);
        INFO("k=" << k << " E_cpu=" << cpu.eigenvalues[k]
             << " E_gpu=" << gpu.eigenvalues[k]
             << " diff=" << diff);
        REQUIRE(diff < 1e-8);
    }
}

TEST_CASE("distributed_krylov_schur_gpu: replicated eigenvalues across "
          "world ranks",
          "[distributed_krylov_schur_gpu][replicated]") {
    if (!runtime_supports_gpu_ks()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedLanczosOptions opts;
    opts.max_iter = 32;
    opts.exct     = 2;
    opts.tol      = 1e-10;
    opts.seed     = 99UL;
    auto res = distributed_krylov_schur_gpu(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.eigenvalues.size() >= 2);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> rank0(res.eigenvalues.size(), 0.0);
    if (rank == 0) rank0 = res.eigenvalues;
    MPI_Bcast(rank0.data(), static_cast<int>(rank0.size()),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (std::size_t k = 0; k < res.eigenvalues.size(); ++k) {
        REQUIRE(std::abs(res.eigenvalues[k] - rank0[k]) < 1e-12);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rc = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return rc;
}
