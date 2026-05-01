// =============================================================================
// test_distributed_ftlm_gpu    (Phase A: device matrix MPI+GPU FTLM)
//
// MPI lockdown for `ed::distributed::distributed_ftlm_gpu`. Every test
// case here cross-checks the GPU result against the CPU reference
// produced by `distributed_ftlm` on the SAME inputs (same operator,
// same seeds, same betas, same MPI_COMM_WORLD) -- the GPU path is
// architecturally a different implementation but mathematically
// identical, so per-sample bit-stability is too strong (NCCL allreduce
// ordering vs MPI_Allreduce on doubles will differ at the 1e-13 level
// after thousands of accumulations) but per-sample-AVERAGED Z(beta)
// agreement to 1e-6 is realistic and what we lockdown.
//
// Coverage:
//   * N=4 OBC: Z(beta) at small set of betas, with and without an
//     observable. Cross-check vs CPU FTLM.
//   * N=6 PBC: same, larger Krylov dim.
//   * Replicated Z across all ranks (FTLM result struct is
//     replicated by construction; a regression here would mean we
//     accidentally double-counted in the world-level reduce).
//
// Run-time gating mirrors test_distributed_lanczos_gpu:
//   * SKIP gracefully when build does not have NCCL.
//   * SKIP gracefully when no CUDA device is visible.
//   * SKIP gracefully when world_size > visible CUDA device count.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_ftlm_gpu.h>
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

using ed::distributed::distributed_ftlm;
using ed::distributed::distributed_ftlm_gpu;
using ed::distributed::DistributedFtlmGPUOptions;
using ed::distributed::DistributedFtlmOptions;
using ed::distributed::DistributedFtlmResult;
using ed::distributed::DistributedOperator;

namespace {

bool runtime_supports_gpu_ftlm() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_ftlm_gpu");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP distributed_ftlm_gpu");
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

TEST_CASE("distributed_ftlm_gpu: N=4 OBC Z(beta) matches CPU FTLM",
          "[distributed_ftlm_gpu][heisenberg]") {
    if (!runtime_supports_gpu_ftlm()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    const std::vector<double> betas = {0.1, 0.5, 1.0, 2.0};

    DistributedFtlmOptions cpu_opts;
    cpu_opts.n_samples        = 16;
    cpu_opts.n_groups         = 1;
    cpu_opts.lanczos_max_iter = 30;
    cpu_opts.betas            = betas;
    cpu_opts.seed_offset      = 42UL;
    auto cpu = distributed_ftlm(op, cpu_opts, MPI_COMM_WORLD);

    DistributedFtlmGPUOptions gpu_opts;
    gpu_opts.n_samples        = 16;
    gpu_opts.n_groups         = 1;
    gpu_opts.lanczos_max_iter = 30;
    gpu_opts.betas            = betas;
    gpu_opts.seed_offset      = 42UL;
    auto gpu = distributed_ftlm_gpu(op, gpu_opts, MPI_COMM_WORLD);

    REQUIRE(cpu.Z.size() == betas.size());
    REQUIRE(gpu.Z.size() == betas.size());

    for (std::size_t b = 0; b < betas.size(); ++b) {
        const double rel = std::abs(gpu.Z[b] - cpu.Z[b])
                          / std::max(1e-30, std::abs(cpu.Z[b]));
        INFO("b=" << b << " beta=" << betas[b]
             << " Z_cpu=" << cpu.Z[b] << " Z_gpu=" << gpu.Z[b]
             << " rel=" << rel);
        // CPU and GPU FTLM use different inner-product reduction
        // implementations (MPI_Allreduce vs NCCL allreduce) so per-
        // accumulation rounding noise differs. 1e-6 relative is the
        // empirical envelope we have observed on small N.
        REQUIRE(rel < 1e-6);
    }
}

TEST_CASE("distributed_ftlm_gpu: N=6 PBC Z(beta) matches CPU FTLM",
          "[distributed_ftlm_gpu][heisenberg][pbc]") {
    if (!runtime_supports_gpu_ftlm()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());

    const std::vector<double> betas = {0.2, 1.0, 3.0};

    DistributedFtlmOptions cpu_opts;
    cpu_opts.n_samples        = 12;
    cpu_opts.n_groups         = 1;
    cpu_opts.lanczos_max_iter = 50;
    cpu_opts.betas            = betas;
    cpu_opts.seed_offset      = 7UL;
    auto cpu = distributed_ftlm(op, cpu_opts, MPI_COMM_WORLD);

    DistributedFtlmGPUOptions gpu_opts;
    gpu_opts.n_samples        = 12;
    gpu_opts.n_groups         = 1;
    gpu_opts.lanczos_max_iter = 50;
    gpu_opts.betas            = betas;
    gpu_opts.seed_offset      = 7UL;
    auto gpu = distributed_ftlm_gpu(op, gpu_opts, MPI_COMM_WORLD);

    for (std::size_t b = 0; b < betas.size(); ++b) {
        const double rel = std::abs(gpu.Z[b] - cpu.Z[b])
                          / std::max(1e-30, std::abs(cpu.Z[b]));
        INFO("b=" << b << " beta=" << betas[b]
             << " Z_cpu=" << cpu.Z[b] << " Z_gpu=" << gpu.Z[b]
             << " rel=" << rel);
        REQUIRE(rel < 1e-6);
    }
}

TEST_CASE("distributed_ftlm_gpu: N=4 OBC <O>(beta) matches CPU FTLM",
          "[distributed_ftlm_gpu][heisenberg][observable]") {
    if (!runtime_supports_gpu_ftlm()) return;

    // Same Hamiltonian as the first test; observable = H itself, so
    // <O>(beta) = <H>(beta) = E(beta), the thermal energy. Both CPU and
    // GPU FTLM should reproduce the same FTLM estimate.
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    // "Observable" Operator is the same Hamiltonian (deep copy; share is
    // fine because Operator::apply is read-only).
    auto obs = op;

    const std::vector<double> betas = {0.5, 1.0, 2.0};

    DistributedFtlmOptions cpu_opts;
    cpu_opts.n_samples        = 24;
    cpu_opts.n_groups         = 1;
    cpu_opts.lanczos_max_iter = 30;
    cpu_opts.betas            = betas;
    cpu_opts.seed_offset      = 11UL;
    cpu_opts.observable_op    = obs;
    auto cpu = distributed_ftlm(op, cpu_opts, MPI_COMM_WORLD);

    DistributedFtlmGPUOptions gpu_opts;
    gpu_opts.n_samples        = 24;
    gpu_opts.n_groups         = 1;
    gpu_opts.lanczos_max_iter = 30;
    gpu_opts.betas            = betas;
    gpu_opts.seed_offset      = 11UL;
    gpu_opts.observable_op    = obs;
    auto gpu = distributed_ftlm_gpu(op, gpu_opts, MPI_COMM_WORLD);

    REQUIRE(cpu.O_expectation.size() == betas.size());
    REQUIRE(gpu.O_expectation.size() == betas.size());

    for (std::size_t b = 0; b < betas.size(); ++b) {
        // <O> = N_O / N_Z. Both are O(1), so absolute tolerance is the
        // honest yardstick here.
        const double diff =
            std::abs(gpu.O_expectation[b] - cpu.O_expectation[b]);
        INFO("b=" << b << " beta=" << betas[b]
             << " <O>_cpu=" << cpu.O_expectation[b]
             << " <O>_gpu=" << gpu.O_expectation[b]
             << " diff=" << diff);
        REQUIRE(diff < 1e-6);
    }
}

TEST_CASE("distributed_ftlm_gpu: replicated Z across world ranks",
          "[distributed_ftlm_gpu][replicated]") {
    if (!runtime_supports_gpu_ftlm()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());

    DistributedFtlmGPUOptions opts;
    opts.n_samples        = 8;
    opts.n_groups         = 1;
    opts.lanczos_max_iter = 25;
    opts.betas            = {1.0};
    opts.seed_offset      = 99UL;
    auto res = distributed_ftlm_gpu(op, opts, MPI_COMM_WORLD);
    REQUIRE(res.Z.size() == 1);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double rank0_Z = (rank == 0) ? res.Z[0] : 0.0;
    MPI_Bcast(&rank0_Z, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    REQUIRE(std::abs(res.Z[0] - rank0_Z) < 1e-12);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rc = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return rc;
}
