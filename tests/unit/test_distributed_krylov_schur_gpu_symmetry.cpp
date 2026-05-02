// =============================================================================
// test_distributed_krylov_schur_gpu_symmetry    (Phase D step 3)
//
// MPI lockdown for `ed::distributed::distributed_krylov_schur_gpu_symmetry`.
// Cross-checks the GPU symmetry-projected thick-restart Lanczos against
// the CPU `distributed_krylov_schur_symmetry` running on the SAME
// MPI_COMM_WORLD, SAME operator + sector, SAME seed. Ground-state
// agreement is checked at `1e-9` (KS does locking; higher Ritz values
// are not strictly comparable across implementations because the
// per-cycle Ritz seed selection depends on the locking residual order).
//
// Build-only on CI's CUDA build-only lane; runtime-tested via the
// `runtime_supports_gpu_ks_sym()` SKIP gate.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_krylov_schur.h>
#include <ed/distributed/distributed_krylov_schur_gpu.h>
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

using ed::distributed::distributed_krylov_schur_symmetry;
using ed::distributed::distributed_krylov_schur_gpu_symmetry;
using ed::distributed::DistributedLanczosOptions;
using ed::distributed::DistributedSymmetryOperator;

namespace {

bool runtime_supports_gpu_ks_sym() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_krylov_schur_gpu_symmetry");
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

void check_sector(int N, double J, bool periodic,
                  std::size_t sector_idx, std::uint64_t exct,
                  unsigned long seed) {
    auto op = make_heisenberg_translation_op(N, J, periodic);
    DistributedSymmetryOperator dop(op, sector_idx, MPI_COMM_WORLD);

    DistributedLanczosOptions lopts;
    lopts.max_iter = 200;
    lopts.exct     = exct;
    lopts.seed     = seed;
    auto cpu_res = distributed_krylov_schur_symmetry(dop, lopts);
    auto gpu_res = distributed_krylov_schur_gpu_symmetry(dop, lopts);

    REQUIRE(!cpu_res.eigenvalues.empty());
    REQUIRE(!gpu_res.eigenvalues.empty());
    const double diff = std::abs(gpu_res.eigenvalues[0] - cpu_res.eigenvalues[0]);
    INFO("N=" << N << " periodic=" << periodic
         << " sector=" << sector_idx
         << " gpu_E0=" << gpu_res.eigenvalues[0]
         << " cpu_E0=" << cpu_res.eigenvalues[0]
         << " diff=" << diff);
    REQUIRE(diff <= 1e-9);
}

}  // namespace

TEST_CASE("distributed_krylov_schur_gpu_symmetry: N=4 PBC vs CPU KS symm",
          "[distributed_krylov_schur_gpu_symmetry][heisenberg][n4][pbc]") {
    if (!runtime_supports_gpu_ks_sym()) return;
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        check_sector(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                     /*sector=*/s, /*exct=*/2, /*seed=*/1234UL + s);
    }
}

TEST_CASE("distributed_krylov_schur_gpu_symmetry: N=6 PBC vs CPU KS symm",
          "[distributed_krylov_schur_gpu_symmetry][heisenberg][n6][pbc]") {
    if (!runtime_supports_gpu_ks_sym()) return;
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
