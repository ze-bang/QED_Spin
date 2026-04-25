// =============================================================================
// test_distributed_gpu_operator    (Phase 3c stage 3)
//
// Lockdown for `ed::distributed::DistributedGPUOperator::apply`. The GPU
// operator must be functionally equivalent to the CPU
// `DistributedOperator::apply` on the same input vector for any (N,
// boundary, np) we exercise. We compare element-wise:
//
//   y_gpu_local = (H * v_global)[local slab]    via DistributedGPUOperator
//   y_cpu_local = (H * v_global)[local slab]    via DistributedOperator
//
// for a deterministic random complex `v_global` scattered the same way
// the GPU Lanczos does it.
//
// Coverage:
//   * N=4 OBC and N=6 PBC Heisenberg chains.
//   * Random complex input vector (deterministic seed).
//   * Round-trip through device memory: H2D v_local, GPU apply, D2H
//     y_gpu, then bit-comparison with the CPU DistributedOperator
//     output.
//
// Run-time gating mirrors test_distributed_lanczos_gpu: SKIP if NCCL not
// compiled in, no CUDA visible, or world_size > visible_devices (the
// MultiGpuCommunicator constructor enforces 1 GPU per rank).
// =============================================================================

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ed/distributed/distributed_operator.h>
#include <ed/distributed/distributed_gpu_operator.h>
#include <ed/distributed/multi_gpu.h>
#include "common/test_harness.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <mpi.h>

#ifdef ED_HAVE_NCCL
#  include <cuda_runtime.h>
#endif

using ed::distributed::DistributedGPUOperator;
using ed::distributed::DistributedOperator;
using Complex = std::complex<double>;

namespace {

// True iff the runtime has NCCL + at least one CUDA device per rank.
// Otherwise prints a SUCCEED() message explaining the SKIP.
bool runtime_supports_gpu_op() {
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        SUCCEED("NCCL not compiled in; SKIP DistributedGPUOperator");
        return false;
    }
#ifdef ED_HAVE_NCCL
    int n_devices = 0;
    cudaError_t e = cudaGetDeviceCount(&n_devices);
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (e != cudaSuccess || n_devices < 1) {
        SUCCEED("No visible CUDA device; SKIP DistributedGPUOperator");
        return false;
    }
    if (world_size > n_devices) {
        SUCCEED("world_size > visible device count; SKIP "
                "(multi-rank-on-same-device deliberately not exercised)");
        return false;
    }
    return true;
#else
    return false;
#endif
}

// Build a deterministic random complex vector replicated on every rank
// (via rank-0 fill + Bcast), then return the rank-local slice
// corresponding to op's slab.
std::vector<Complex>
deterministic_v_local(const DistributedOperator& op, unsigned long seed) {
    const std::uint64_t global_dim   = op.global_dim();
    const std::uint64_t local_offset = op.local_offset();
    const std::uint64_t local_n      = op.local_size();

    int rank = 0;
    MPI_Comm_rank(op.comm(), &rank);

    std::vector<Complex> v_global(static_cast<std::size_t>(global_dim));
    if (rank == 0) {
        std::mt19937_64 gen(seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (std::uint64_t i = 0; i < global_dim; ++i) {
            const double a = nd(gen);
            const double b = nd(gen);
            v_global[i] = Complex(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : v_global) z *= inv;
    }
    MPI_Bcast(v_global.data(),
              2 * static_cast<int>(global_dim),
              MPI_DOUBLE, 0, op.comm());

    std::vector<Complex> v_local(static_cast<std::size_t>(local_n));
    for (std::uint64_t i = 0; i < local_n; ++i) {
        v_local[i] = v_global[local_offset + i];
    }
    return v_local;
}

#ifdef ED_HAVE_NCCL
// Round-trip a Complex* via the device through DistributedGPUOperator::apply.
void run_gpu_apply(const DistributedGPUOperator& gop,
                   const ed::distributed::multi_gpu::MultiGpuCommunicator& comm,
                   const std::vector<Complex>& v_local,
                   std::vector<Complex>& y_gpu_local) {
    const std::size_t local_n = v_local.size();
    y_gpu_local.assign(local_n, Complex(0.0, 0.0));
    if (local_n == 0) return;

    Complex* d_v = nullptr;
    Complex* d_y = nullptr;
    const std::size_t bytes = local_n * sizeof(Complex);
    cudaMalloc(reinterpret_cast<void**>(&d_v), bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_y), bytes);
    cudaMemcpy(d_v, v_local.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, bytes);

    gop.apply(comm, d_v, d_y, /*stream=*/nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(y_gpu_local.data(), d_y, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_v);
    cudaFree(d_y);
}
#endif  // ED_HAVE_NCCL

}  // namespace

TEST_CASE("DistributedGPUOperator: N=4 OBC matches CPU DistributedOperator",
          "[distributed_gpu_op][heisenberg]") {
    if (!runtime_supports_gpu_op()) return;

#ifdef ED_HAVE_NCCL
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/4, /*J=*/1.0,
                                          /*periodic=*/false).release());
    auto dop = std::make_shared<DistributedOperator>(op, MPI_COMM_WORLD);

    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(
        MPI_COMM_WORLD, /*device_index=*/-1);

    DistributedGPUOperator gop(dop, gpu_comm);

    auto v_local = deterministic_v_local(*dop, /*seed=*/12345UL);
    std::vector<Complex> y_cpu(v_local.size(), Complex(0.0, 0.0));
    dop->apply(v_local.data(), y_cpu.data());

    std::vector<Complex> y_gpu;
    run_gpu_apply(gop, gpu_comm, v_local, y_gpu);

    REQUIRE(y_gpu.size() == y_cpu.size());
    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < y_cpu.size(); ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(y_gpu[i] - y_cpu[i]));
    }
    INFO("rank=" << dop->rank() << " local_n=" << v_local.size()
         << " max|y_gpu - y_cpu| = " << max_abs_err);
    REQUIRE(max_abs_err < 1e-12);
#endif
}

TEST_CASE("DistributedGPUOperator: N=6 PBC matches CPU DistributedOperator",
          "[distributed_gpu_op][heisenberg][pbc]") {
    if (!runtime_supports_gpu_op()) return;

#ifdef ED_HAVE_NCCL
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    auto dop = std::make_shared<DistributedOperator>(op, MPI_COMM_WORLD);

    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(
        MPI_COMM_WORLD, /*device_index=*/-1);

    DistributedGPUOperator gop(dop, gpu_comm);

    auto v_local = deterministic_v_local(*dop, /*seed=*/7UL);
    std::vector<Complex> y_cpu(v_local.size(), Complex(0.0, 0.0));
    dop->apply(v_local.data(), y_cpu.data());

    std::vector<Complex> y_gpu;
    run_gpu_apply(gop, gpu_comm, v_local, y_gpu);

    REQUIRE(y_gpu.size() == y_cpu.size());
    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < y_cpu.size(); ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(y_gpu[i] - y_cpu[i]));
    }
    INFO("rank=" << dop->rank() << " local_n=" << v_local.size()
         << " max|y_gpu - y_cpu| = " << max_abs_err);
    REQUIRE(max_abs_err < 1e-12);
#endif
}

TEST_CASE("DistributedGPUOperator: idempotent re-apply (sanity)",
          "[distributed_gpu_op][repeat]") {
    if (!runtime_supports_gpu_op()) return;

#ifdef ED_HAVE_NCCL
    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/false).release());
    auto dop = std::make_shared<DistributedOperator>(op, MPI_COMM_WORLD);

    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(
        MPI_COMM_WORLD, /*device_index=*/-1);

    DistributedGPUOperator gop(dop, gpu_comm);

    auto v_local = deterministic_v_local(*dop, /*seed=*/2026UL);
    std::vector<Complex> y_a, y_b;
    run_gpu_apply(gop, gpu_comm, v_local, y_a);
    run_gpu_apply(gop, gpu_comm, v_local, y_b);

    REQUIRE(y_a.size() == y_b.size());
    double max_abs_err = 0.0;
    for (std::size_t i = 0; i < y_a.size(); ++i) {
        max_abs_err = std::max(max_abs_err, std::abs(y_a[i] - y_b[i]));
    }
    INFO("rank=" << dop->rank() << " idempotent re-apply max diff = "
         << max_abs_err);
    REQUIRE(max_abs_err < 1e-15);
#endif
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
