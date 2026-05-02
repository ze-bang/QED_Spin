// =============================================================================
// test_distributed_symmetry_operator_gpu    (Phase C)
//
// Cross-checks `DistributedSymmetryOperatorGPU::apply` against the CPU
// `DistributedSymmetryOperator::apply` on small Heisenberg chains with
// 1D translation symmetry.
//
// Build-only on CI (no GPU on the build-only lane); runtime-tested via
// the `runtime_supports_gpu_sym()` SKIP gate, mirroring the pattern in
// test_distributed_lanczos_gpu.cpp.
//
// Tolerance: 1e-12 on per-component diff. Both paths build the same
// projected CSR; the GPU path additionally pipes complex<double>
// through NCCL pairwise SendRecv (no reduction), so machine-precision
// agreement is the expected baseline.
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/core/construct_ham.h>
#include <ed/distributed/distributed_symmetry_operator.h>
#include <ed/distributed/multi_gpu.h>
#include <ed/symmetry/group.h>
#include "common/test_harness.h"

#ifdef ED_HAVE_NCCL
#  include <ed/distributed/distributed_symmetry_operator_gpu.h>
#  include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <mpi.h>

using ed::distributed::DistributedSymmetryOperator;
using Complex = std::complex<double>;

namespace {

bool runtime_supports_gpu_sym() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP distributed_symmetry_operator_gpu");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP distributed_symmetry_operator_gpu");
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

std::shared_ptr<Operator>
make_heisenberg_translation_op(int N, double J, bool periodic) {
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(static_cast<uint64_t>(N), J,
                                         periodic).release());
    op->symmetry_info = ed::sym::translation_group_1d(N);
    return op;
}

std::vector<Complex>
deterministic_local_vector(std::size_t n, unsigned long seed) {
    std::vector<Complex> v(n);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& z : v) z = Complex(nd(gen), nd(gen));
    return v;
}

#ifdef ED_HAVE_NCCL

// Run the GPU apply once and compare against the CPU apply on the same
// inputs. Returns the max-abs diff across local rows (collectively
// reduced at the call site).
double check_one(int N, double J, bool periodic, std::size_t sector_idx,
                 unsigned long seed) {
    using namespace ed::distributed;

    auto op = make_heisenberg_translation_op(N, J, periodic);
    auto cpu = std::make_shared<DistributedSymmetryOperator>(
        op, sector_idx, MPI_COMM_WORLD);

    multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD,
                                             multi_gpu::kAutoDeviceIndex);

    DistributedSymmetryOperatorGPU dop_gpu(cpu, gpu_comm);

    const std::size_t local_n = cpu->local_size();
    auto x_local = deterministic_local_vector(local_n, seed);

    // Reference: CPU apply.
    std::vector<Complex> y_cpu(local_n, Complex(0.0, 0.0));
    cpu->apply(x_local.data(), y_cpu.data());

    // GPU apply.
    Complex* d_x = nullptr;
    Complex* d_y = nullptr;
    const std::size_t bytes_x = local_n * sizeof(Complex);
    if (local_n > 0) {
        cudaMalloc(reinterpret_cast<void**>(&d_x), bytes_x);
        cudaMalloc(reinterpret_cast<void**>(&d_y), bytes_x);
        cudaMemcpy(d_x, x_local.data(), bytes_x, cudaMemcpyHostToDevice);
        cudaMemset(d_y, 0, bytes_x);
    }
    dop_gpu.apply(gpu_comm, d_x, d_y, /*stream=*/nullptr);
    cudaDeviceSynchronize();

    std::vector<Complex> y_gpu(local_n, Complex(0.0, 0.0));
    if (local_n > 0) {
        cudaMemcpy(y_gpu.data(), d_y, bytes_x, cudaMemcpyDeviceToHost);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    double err = 0.0;
    for (std::size_t k = 0; k < local_n; ++k) {
        err = std::max(err, std::abs(y_gpu[k] - y_cpu[k]));
    }
    double global_err = 0.0;
    MPI_Allreduce(&err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global_err;
}

#endif  // ED_HAVE_NCCL

}  // namespace

TEST_CASE("DistributedSymmetryOperatorGPU: N=4 PBC, all Z_4 sectors vs CPU",
          "[distributed_symmetry_operator_gpu][heisenberg][n4][pbc]") {
    if (!runtime_supports_gpu_sym()) return;
#ifdef ED_HAVE_NCCL
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const double err = check_one(/*N=*/4, /*J=*/1.0, /*periodic=*/true,
                                      /*sector=*/s, /*seed=*/2024UL + s);
        INFO("N=4 PBC sector=" << s << " err=" << err);
        REQUIRE(err <= 1e-12);
    }
#endif
}

TEST_CASE("DistributedSymmetryOperatorGPU: N=4 OBC, all Z_4 sectors vs CPU",
          "[distributed_symmetry_operator_gpu][heisenberg][n4][obc]") {
    if (!runtime_supports_gpu_sym()) return;
#ifdef ED_HAVE_NCCL
    auto info = ed::sym::translation_group_1d(4);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const double err = check_one(/*N=*/4, /*J=*/1.0, /*periodic=*/false,
                                      /*sector=*/s, /*seed=*/77UL + s);
        INFO("N=4 OBC sector=" << s << " err=" << err);
        REQUIRE(err <= 1e-12);
    }
#endif
}

TEST_CASE("DistributedSymmetryOperatorGPU: N=6 PBC, all Z_6 sectors vs CPU",
          "[distributed_symmetry_operator_gpu][heisenberg][n6][pbc]") {
    if (!runtime_supports_gpu_sym()) return;
#ifdef ED_HAVE_NCCL
    auto info = ed::sym::translation_group_1d(6);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const double err = check_one(/*N=*/6, /*J=*/-1.5, /*periodic=*/true,
                                      /*sector=*/s, /*seed=*/4242UL + s);
        INFO("N=6 PBC sector=" << s << " err=" << err);
        REQUIRE(err <= 1e-12);
    }
#endif
}

// -----------------------------------------------------------------------------
// Custom main: MPI_Init + Catch2 + MPI_Finalize.
// -----------------------------------------------------------------------------
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
