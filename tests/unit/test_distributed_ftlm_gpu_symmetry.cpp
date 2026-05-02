// =============================================================================
// test_distributed_ftlm_gpu_symmetry    (Phase D step 5)
//
// MPI lockdown for `ed::distributed::distributed_ftlm_gpu_symmetry`.
// Cross-checks the GPU symmetry-projected FTLM Z(beta) against the CPU
// `distributed_ftlm_symmetry` running on the SAME MPI_COMM_WORLD,
// SAME operator + sector, SAME seed_offset.
//
// Build-only on CI's CUDA build-only lane; runtime-tested via the
// `runtime_supports_gpu_ftlm_sym()` SKIP gate.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_ftlm_gpu.h>
#include <ed/distributed/multi_gpu.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#endif

using ed::distributed::DistributedFtlmOptions;
using ed::distributed::DistributedFtlmGPUOptions;
using ed::distributed::DistributedFtlmResult;
using ed::distributed::distributed_ftlm_symmetry;
using ed::distributed::distributed_ftlm_gpu_symmetry;

namespace {

bool runtime_supports_gpu_ftlm_sym() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_ftlm_gpu_symmetry");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP");
        return false;
    }
    if (world_size > n_devices) {
        SUCCEED("world_size > visible device count; SKIP");
        return false;
    }
    return true;
#else
    return false;
#endif
}

std::shared_ptr<Operator>
make_heisenberg_translation_op(int N, double J, bool periodic) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    op->symmetry_info = ed::sym::translation_group_1d(N);
    return op;
}

}  // namespace

TEST_CASE("distributed_ftlm_gpu_symmetry: Z(beta) matches CPU FTLM symm",
          "[distributed_ftlm_gpu_symmetry][heisenberg][n4][pbc]") {
    if (!runtime_supports_gpu_ftlm_sym()) return;

    const int N = 4;
    const double J = 1.0;
    auto op = make_heisenberg_translation_op(N, J, /*periodic=*/true);

    const std::vector<double> betas = {0.1, 0.5, 1.0};
    const int n_samples = 16;
    const std::uint64_t max_iter = 30;
    const unsigned long seed_offset = 7777UL;

    const std::size_t n_sectors = op->symmetry_info.sectors.size();
    REQUIRE(n_sectors > 0);

    for (std::size_t s = 0; s < n_sectors; ++s) {
        DistributedFtlmOptions cpu_opts;
        cpu_opts.n_samples       = n_samples;
        cpu_opts.n_groups        = 1;
        cpu_opts.lanczos_max_iter = max_iter;
        cpu_opts.betas           = betas;
        cpu_opts.seed_offset     = seed_offset;
        DistributedFtlmResult cpu_res =
            distributed_ftlm_symmetry(op, s, cpu_opts, MPI_COMM_WORLD);

        DistributedFtlmGPUOptions gpu_opts;
        gpu_opts.n_samples       = n_samples;
        gpu_opts.n_groups        = 1;
        gpu_opts.lanczos_max_iter = max_iter;
        gpu_opts.betas           = betas;
        gpu_opts.seed_offset     = seed_offset;
        DistributedFtlmResult gpu_res =
            distributed_ftlm_gpu_symmetry(op, s, gpu_opts, MPI_COMM_WORLD);

        REQUIRE(cpu_res.Z.size() == betas.size());
        REQUIRE(gpu_res.Z.size() == betas.size());
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double diff = std::abs(gpu_res.Z[b] - cpu_res.Z[b]);
            const double scale = std::max(1e-30, std::abs(cpu_res.Z[b]));
            INFO("sector=" << s << " beta=" << betas[b]
                 << " cpu_Z=" << cpu_res.Z[b]
                 << " gpu_Z=" << gpu_res.Z[b]
                 << " reldiff=" << diff / scale);
            // CPU vs GPU: same algorithm, same seed; the only sources
            // of disagreement are NCCL allreduce ordering (commutative
            // sums of complex doubles -- bit-stable to round-off in
            // practice but not strictly equal to MPI_Allreduce). Allow
            // 1e-9 relative.
            REQUIRE(diff / scale < 1e-9);
        }
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

    int local_result = Catch::Session().run(argc, argv);
    int global_result = 0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Finalize();
    return global_result;
}
