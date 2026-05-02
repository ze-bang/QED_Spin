// =============================================================================
// test_distributed_lanczos_gpu_symmetry    (Phase D step 1)
//
// MPI lockdown for `ed::distributed::distributed_lanczos_gpu_symmetry`.
// Cross-checks the GPU symmetry-projected Lanczos against the CPU
// `distributed_lanczos_symmetry` running on the SAME MPI_COMM_WORLD,
// SAME operator + sector, SAME seed. Eigenvalues should agree at
// `1e-9` (the iteration noise floor at max_iter=200 on these dim<= a
// few hundred problems).
//
// Build-only on CI's CUDA build-only lane; runtime-tested via the
// `runtime_supports_gpu_lanczos_sym()` SKIP gate.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_lanczos_gpu.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/multi_gpu.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#endif

using ed::distributed::distributed_lanczos_symmetry;
using ed::distributed::distributed_lanczos_gpu_symmetry;
using ed::distributed::DistributedLanczosGPUOptions;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedSymmetryOperator;

namespace {

bool runtime_supports_gpu_lanczos_sym() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_lanczos_gpu_symmetry");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP distributed_lanczos_gpu_symmetry");
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

void check_sector(int N, double J, bool periodic,
                  std::size_t sector_idx, std::uint64_t exct,
                  unsigned long seed) {
    auto op = make_heisenberg_translation_op(N, J, periodic);
    DistributedSymmetryOperator dop(op, sector_idx, MPI_COMM_WORLD);

    DistributedLanczosOptions cpu_opts;
    cpu_opts.max_iter = 200;
    cpu_opts.exct     = exct;
    cpu_opts.seed     = seed;
    auto cpu_res = distributed_lanczos_symmetry(dop, cpu_opts);

    DistributedLanczosGPUOptions gpu_opts;
    gpu_opts.max_iter = 200;
    gpu_opts.exct     = exct;
    gpu_opts.seed     = seed;
    auto gpu_res = distributed_lanczos_gpu_symmetry(dop, gpu_opts);

    REQUIRE(gpu_res.eigenvalues.size() == cpu_res.eigenvalues.size());
    for (std::size_t k = 0; k < gpu_res.eigenvalues.size(); ++k) {
        const double diff = std::abs(gpu_res.eigenvalues[k]
                                      - cpu_res.eigenvalues[k]);
        INFO("N=" << N << " periodic=" << periodic
             << " sector=" << sector_idx << " k=" << k
             << " gpu=" << gpu_res.eigenvalues[k]
             << " cpu=" << cpu_res.eigenvalues[k]
             << " diff=" << diff);
        REQUIRE(diff <= 1e-9);
    }
}

}  // namespace

TEST_CASE("distributed_lanczos_gpu_symmetry: N=4 PBC vs CPU symm",
          "[distributed_lanczos_gpu_symmetry][heisenberg][n4][pbc]") {
    if (!runtime_supports_gpu_lanczos_sym()) return;
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                     /*sector=*/s, /*exct=*/2, /*seed=*/1234UL + s);
    }
}

TEST_CASE("distributed_lanczos_gpu_symmetry: N=6 PBC vs CPU symm",
          "[distributed_lanczos_gpu_symmetry][heisenberg][n6][pbc]") {
    if (!runtime_supports_gpu_lanczos_sym()) return;
    auto info = ed::sym::translation_group_1d(6);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/6, /*J=*/-1.5, /*periodic=*/true,
                     /*sector=*/s, /*exct=*/3, /*seed=*/4242UL + s);
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
