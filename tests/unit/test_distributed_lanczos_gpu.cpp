// =============================================================================
// test_distributed_lanczos_gpu    (Phase 3c stage 2)
//
// MPI lockdown for `ed::distributed::distributed_lanczos_gpu`. The GPU
// path is functionally equivalent to `distributed_lanczos` on the same
// inputs (same initial vector, same recurrence, same eigenvalue), so
// every test case here cross-checks the GPU result against the CPU
// reference produced by `distributed_lanczos` running the SAME
// MPI_COMM_WORLD on the SAME problem.
//
// Coverage:
//   * N=4 OBC ground state vs CPU `distributed_lanczos` (and dense).
//   * N=6 PBC ground state vs CPU `distributed_lanczos`.
//   * Replicated eigenvalues across all ranks (bit-for-bit modulo
//     allreduce noise).
//
// Run-time gating:
//   * SKIP gracefully when the build does not have NCCL
//     (`multi_gpu::nccl_compiled_in() == false`).
//   * SKIP gracefully when no CUDA device is visible (e.g. CI on a
//     CPU node).
//   * SKIP gracefully when world_size > visible CUDA device count
//     (multi-rank-on-same-device is allowed by NCCL but tickles
//     known issues on MIG slices; not what we want to lockdown here).
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_lanczos_gpu.h>
#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/multi_gpu.h>
#include "common/test_harness.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#endif

using ed::distributed::distributed_lanczos;
using ed::distributed::distributed_lanczos_gpu;
using ed::distributed::DistributedLanczosGPUOptions;
using ed::distributed::DistributedLanczosGPUResult;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedOperator;

namespace {

// Returns true if the runtime has at least one CUDA device per MPI rank
// and the build was configured with NCCL. Otherwise prints a SUCCEED()
// message explaining the SKIP and returns false.
bool runtime_supports_gpu_lanczos() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_lanczos_gpu");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP distributed_lanczos_gpu");
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

TEST_CASE("distributed_lanczos_gpu: N=4 OBC ground state vs CPU + dense",
          "[distributed_lanczos_gpu][heisenberg]") {
    if (!runtime_supports_gpu_lanczos()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);
    auto ref = ed_tests::reference_from_operator(*op, dop.global_dim());

    // CPU reference (no full re-orth, same seed).
    DistributedLanczosOptions cpu_opts;
    cpu_opts.max_iter = 60;
    cpu_opts.exct     = 1;
    cpu_opts.tol      = 1e-12;
    cpu_opts.seed     = 12345UL;
    auto cpu = distributed_lanczos(dop, cpu_opts);
    REQUIRE(!cpu.eigenvalues.empty());

    // GPU run (same seed -> same initial vector -> same Krylov tridiag).
    DistributedLanczosGPUOptions gpu_opts;
    gpu_opts.max_iter = 60;
    gpu_opts.exct     = 1;
    gpu_opts.tol      = 1e-12;
    gpu_opts.seed     = 12345UL;
    auto gpu = distributed_lanczos_gpu(dop, gpu_opts);
    REQUIRE(!gpu.eigenvalues.empty());

    INFO("E0_gpu = " << gpu.eigenvalues.front()
         << "  E0_cpu = " << cpu.eigenvalues.front()
         << "  E0_dense = " << ref.eigs.front()
         << "  iters_gpu = " << gpu.iterations
         << "  iters_cpu = " << cpu.iterations);
    REQUIRE(std::abs(gpu.eigenvalues.front() - ref.eigs.front()) < 1e-8);
    // GPU must agree with CPU within the same numerical noise envelope.
    REQUIRE(std::abs(gpu.eigenvalues.front() - cpu.eigenvalues.front()) < 1e-10);
}

TEST_CASE("distributed_lanczos_gpu: N=6 PBC ground state vs CPU + dense",
          "[distributed_lanczos_gpu][heisenberg][pbc]") {
    if (!runtime_supports_gpu_lanczos()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);
    auto ref = ed_tests::reference_from_operator(*op, dop.global_dim());

    DistributedLanczosOptions cpu_opts;
    cpu_opts.max_iter = 80;
    cpu_opts.exct     = 1;
    cpu_opts.tol      = 1e-12;
    cpu_opts.seed     = 7UL;
    auto cpu = distributed_lanczos(dop, cpu_opts);
    REQUIRE(!cpu.eigenvalues.empty());

    DistributedLanczosGPUOptions gpu_opts;
    gpu_opts.max_iter = 80;
    gpu_opts.exct     = 1;
    gpu_opts.tol      = 1e-12;
    gpu_opts.seed     = 7UL;
    auto gpu = distributed_lanczos_gpu(dop, gpu_opts);
    REQUIRE(!gpu.eigenvalues.empty());

    INFO("E0_gpu = " << gpu.eigenvalues.front()
         << "  E0_cpu = " << cpu.eigenvalues.front()
         << "  E0_dense = " << ref.eigs.front());
    REQUIRE(std::abs(gpu.eigenvalues.front() - ref.eigs.front()) < 1e-8);
    REQUIRE(std::abs(gpu.eigenvalues.front() - cpu.eigenvalues.front()) < 1e-10);
}

TEST_CASE("distributed_lanczos_gpu: replicated eigenvalues across ranks",
          "[distributed_lanczos_gpu][replicated]") {
    if (!runtime_supports_gpu_lanczos()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);

    DistributedLanczosGPUOptions opts;
    opts.max_iter = 50;
    opts.exct     = 3;
    opts.tol      = 1e-12;
    opts.seed     = 999UL;
    auto res = distributed_lanczos_gpu(dop, opts);
    REQUIRE(!res.eigenvalues.empty());

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int n = static_cast<int>(res.eigenvalues.size());
    int n_rank0 = n;
    MPI_Bcast(&n_rank0, 1, MPI_INT, 0, MPI_COMM_WORLD);
    REQUIRE(n == n_rank0);

    std::vector<double> rank0_evals(n_rank0, 0.0);
    if (rank == 0) rank0_evals = res.eigenvalues;
    MPI_Bcast(rank0_evals.data(), n_rank0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < n; ++i) {
        REQUIRE(std::abs(res.eigenvalues[i] - rank0_evals[i]) < 1e-10);
    }
}

// =============================================================================
// Stage 4 lockdown: gpu_resident_spmv = true must agree with the
// host-staged GPU path (and therefore with the CPU `distributed_lanczos`)
// to within 1e-10 on the smallest eigenvalue. This validates the
// `DistributedGPUOperator` (NCCL pairwise SendRecv halo + GPU SpMV
// kernel) inside the same Lanczos loop that the stage 2 path runs in,
// which catches integration regressions that the standalone
// `test_distributed_gpu_operator` cannot (e.g. cuBLAS / NCCL stream
// ordering interleaved with the device SpMV).
// =============================================================================

TEST_CASE("distributed_lanczos_gpu: stage4 gpu_resident_spmv == stage2 host-staged "
          "(N=4 OBC + N=6 PBC)",
          "[distributed_lanczos_gpu][stage4][gpu_resident_spmv]") {
    if (!runtime_supports_gpu_lanczos()) return;

    struct Case { int N; bool periodic; unsigned long seed; std::uint64_t max_iter; };
    const std::vector<Case> cases{
        {4, false, 12345UL, 60},
        {6, true,     7UL, 80},
    };

    for (const auto& tc : cases) {
        auto op = std::shared_ptr<Operator>(
            ed_tests::build_heisenberg_chain(tc.N, /*J=*/1.0, tc.periodic)
                .release());
        DistributedOperator dop(op, MPI_COMM_WORLD);

        DistributedLanczosGPUOptions stage2;
        stage2.max_iter = tc.max_iter;
        stage2.exct     = 1;
        stage2.tol      = 1e-12;
        stage2.seed     = tc.seed;
        stage2.gpu_resident_spmv = false;
        auto r2 = distributed_lanczos_gpu(dop, stage2);
        REQUIRE(!r2.eigenvalues.empty());

        DistributedLanczosGPUOptions stage4 = stage2;
        stage4.gpu_resident_spmv = true;
        auto r4 = distributed_lanczos_gpu(dop, stage4);
        REQUIRE(!r4.eigenvalues.empty());

        INFO("N=" << tc.N << " pbc=" << tc.periodic
             << "  E0_stage2=" << r2.eigenvalues.front()
             << "  E0_stage4=" << r4.eigenvalues.front()
             << "  iters2=" << r2.iterations
             << "  iters4=" << r4.iterations);
        REQUIRE(std::abs(r4.eigenvalues.front() - r2.eigenvalues.front())
                < 1e-10);
    }
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        FILE* dummy = std::freopen("/dev/null", "w", stdout);
        (void)dummy;
    }
    int local = Catch::Session().run(argc, argv);
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global;
}
