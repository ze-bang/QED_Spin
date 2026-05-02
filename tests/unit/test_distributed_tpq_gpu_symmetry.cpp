// =============================================================================
// test_distributed_tpq_gpu_symmetry    (Phase E step 2)
//
// MPI lockdown for `ed::distributed::distributed_tpq_gpu_symmetry`.
// Cross-checks the GPU symmetry-projected canonical TPQ energy(beta)
// against the CPU `distributed_tpq_symmetry` running on the SAME
// MPI_COMM_WORLD, SAME operator + sector, SAME seed_offset,
// delta_beta and taylor_order.
//
// Build-only on CI's CUDA build-only lane; runtime-tested via the
// `runtime_supports_gpu_tpq_sym()` SKIP gate.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_tpq.h>
#include <ed/distributed/distributed_tpq_gpu.h>
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

using ed::distributed::DistributedTpqOptions;
using ed::distributed::DistributedTpqGPUOptions;
using ed::distributed::DistributedTpqResult;
using ed::distributed::distributed_tpq_symmetry;
using ed::distributed::distributed_tpq_gpu_symmetry;

namespace {

bool runtime_supports_gpu_tpq_sym() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_tpq_gpu_symmetry");
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

TEST_CASE("distributed_tpq_gpu_symmetry: energy(beta) matches CPU TPQ symm",
          "[distributed_tpq_gpu_symmetry][heisenberg][n4][pbc]") {
    if (!runtime_supports_gpu_tpq_sym()) return;

    const int N = 4;
    const double J = 1.0;
    auto op = make_heisenberg_translation_op(N, J, /*periodic=*/true);

    const std::vector<double> betas = {0.5, 1.5};
    const int n_samples            = 8;
    const double delta_beta        = 0.05;
    const std::uint64_t taylor_ord = 12;
    const unsigned long seed_offset = 4242UL;

    const std::size_t n_sectors = op->symmetry_info.sectors.size();
    REQUIRE(n_sectors > 0);

    for (std::size_t s = 0; s < n_sectors; ++s) {
        DistributedTpqOptions cpu_opts;
        cpu_opts.n_samples        = n_samples;
        cpu_opts.n_groups         = 1;
        cpu_opts.delta_beta       = delta_beta;
        cpu_opts.taylor_order     = taylor_ord;
        cpu_opts.betas            = betas;
        cpu_opts.seed_offset      = seed_offset;
        cpu_opts.compute_variance = false;
        DistributedTpqResult cpu_res =
            distributed_tpq_symmetry(op, s, cpu_opts, MPI_COMM_WORLD);

        DistributedTpqGPUOptions gpu_opts;
        gpu_opts.n_samples        = n_samples;
        gpu_opts.n_groups         = 1;
        gpu_opts.delta_beta       = delta_beta;
        gpu_opts.taylor_order     = taylor_ord;
        gpu_opts.betas            = betas;
        gpu_opts.seed_offset      = seed_offset;
        gpu_opts.compute_variance = false;
        DistributedTpqResult gpu_res =
            distributed_tpq_gpu_symmetry(op, s, gpu_opts, MPI_COMM_WORLD);

        REQUIRE(cpu_res.energy.size() == betas.size());
        REQUIRE(gpu_res.energy.size() == betas.size());
        for (std::size_t b = 0; b < betas.size(); ++b) {
            const double diff = std::abs(gpu_res.energy[b] - cpu_res.energy[b]);
            const double scale = std::max(1.0, std::abs(cpu_res.energy[b]));
            INFO("sector=" << s << " beta=" << betas[b]
                 << " cpu_E=" << cpu_res.energy[b]
                 << " gpu_E=" << gpu_res.energy[b]
                 << " reldiff=" << diff / scale);
            // Same algorithm, same seed; differences come from the
            // NCCL allreduce vs MPI_Allreduce ordering of complex
            // doubles (commutative but not bit-equal). 1e-9 is the
            // same tolerance the FTLM symm GPU lockdown uses.
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
