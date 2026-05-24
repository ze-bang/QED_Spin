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

// =============================================================================
// Phase 3.5 (CGS2 orthogonality lockdown):
//   Call `ed::krylov::lanczos_kernel<MpiCudaBackend>` directly on the
//   N=6 PBC chain. The default options set by `distributed_lanczos_gpu`
//   (FullCGS2 + keep_basis) must keep `||V^H V - I||_inf < 1e-10` across
//   the whole basis at M up to 60. This is the regression that fires if
//   `MpiCudaBackend::dot` / `dot_many` ever drops the cross-rank
//   ncclAllReduce, or if the kernel's CGS2 path regresses.
// =============================================================================
#ifdef ED_HAVE_NCCL

#  include <ed/krylov/lanczos_kernel.h>
#  include <ed/matvec/backends/mpi_cuda_backend.cuh>
#  include <ed/distributed/distributed_gpu_operator.h>

namespace {

// Allocate a scratch device vector with the same lifetime as the kernel
// basis; the kernel uses `backend.allocate` so we mirror that.
std::vector<std::complex<double>> v_to_host(
    const ed::matvec::MpiCudaBackend& /*unused*/,
    const std::complex<double>* d_v,
    std::size_t n) {
    std::vector<std::complex<double>> out(n);
    if (n > 0) {
        cudaMemcpy(out.data(), d_v,
                   n * sizeof(std::complex<double>),
                   cudaMemcpyDeviceToHost);
    }
    return out;
}

}  // namespace

TEST_CASE("distributed_lanczos_gpu: CGS2 orthogonality "
          "|| V^H V - I ||_inf < 1e-10",
          "[distributed_lanczos_gpu][orthogonality][cgs2]") {
    if (!runtime_supports_gpu_lanczos()) return;

    auto op = std::shared_ptr<Operator>(
        ed_tests::build_heisenberg_chain(/*N=*/6, /*J=*/1.0,
                                          /*periodic=*/true).release());
    DistributedOperator dop(op, MPI_COMM_WORLD);

    // Set up MultiGpuCommunicator + MpiCudaBackend for direct kernel run.
    ed::distributed::multi_gpu::MultiGpuCommunicator gpu_comm(MPI_COMM_WORLD);
    ed::matvec::MpiCudaBackend backend(gpu_comm);

    const std::size_t local_n = dop.local_size();
    const std::uint64_t M = 30;

    // Construct a deterministic device-side seed (rank-0 fills natural
    // RNG, scatter, normalise — same shape as scatter_initial_vector
    // inside distributed_lanczos_gpu.cu). For an orthogonality lockdown
    // we don't need bit-identical RNG; a simple per-rank deterministic
    // seed suffices.
    std::vector<std::complex<double>> v0_host(local_n);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    for (std::size_t i = 0; i < local_n; ++i) {
        v0_host[i] = std::complex<double>(
            std::sin(0.31 * static_cast<double>(rank * 1024 + i)),
            std::cos(0.17 * static_cast<double>(rank * 2048 + i)));
    }
    // Normalise globally via the backend.
    {
        std::complex<double>* d_tmp = backend.allocate(local_n);
        if (local_n > 0) {
            cudaMemcpy(d_tmp, v0_host.data(),
                       local_n * sizeof(std::complex<double>),
                       cudaMemcpyHostToDevice);
        }
        const double nrm = backend.nrm2(d_tmp, local_n);
        if (nrm > 0.0) {
            const std::complex<double> inv(1.0 / nrm, 0.0);
            backend.scale(inv, d_tmp, local_n);
        }
        cudaMemcpy(v0_host.data(), d_tmp,
                   local_n * sizeof(std::complex<double>),
                   cudaMemcpyDeviceToHost);
        backend.deallocate(d_tmp, local_n);
    }

    // DistributedGPUOperator wants a shared_ptr that owns the underlying
    // DistributedOperator; we have `dop` here as a stack object owned by
    // this test scope, so wrap it in a non-owning shared_ptr (aliasing
    // ctor with explicit empty deleter via `[](DistributedOperator*) {}`)
    // exactly the way distributed_lanczos_gpu.cu does it.
    std::shared_ptr<DistributedOperator> dop_alias(
        &dop, [](DistributedOperator*) {});
    ed::distributed::DistributedGPUOperator gop(dop_alias, gpu_comm);

    std::complex<double>* d_v0 = backend.allocate(local_n);
    if (local_n > 0) {
        cudaMemcpy(d_v0, v0_host.data(),
                   local_n * sizeof(std::complex<double>),
                   cudaMemcpyHostToDevice);
    }

    auto matvec = [&](const std::complex<double>* in, std::complex<double>* out) {
        gop.apply(gpu_comm, in, out, /*stream=*/nullptr);
    };

    ed::krylov::LanczosKernelOptions kopts;
    kopts.max_iter = M;
    kopts.reorth   = ed::krylov::ReorthPolicy::FullCGS2;
    kopts.keep_basis = true;
    kopts.dim_cap  = dop.global_dim();
    kopts.breakdown_tol = 1e-14;

    auto kres = ed::krylov::lanczos_kernel(
        backend, matvec, local_n, d_v0, kopts);

    backend.deallocate(d_v0, local_n);

    REQUIRE(!kres.basis.empty());
    const std::size_t mfinal = kres.basis.size();
    INFO("CGS2 lockdown: mfinal=" << mfinal << "  local_n=" << local_n);

    // Compute |<V_i | V_j>| with MpiCudaBackend::dot (NCCL-allreduced).
    // Identity diagonal must be 1.0 within 1e-12; off-diagonal must be
    // below 1e-10. We do all pairs (i,j) since mfinal is small.
    double worst_off = 0.0;
    double worst_diag = 0.0;
    for (std::size_t i = 0; i < mfinal; ++i) {
        for (std::size_t j = i; j < mfinal; ++j) {
            std::complex<double> c = backend.dot(
                reinterpret_cast<const std::complex<double>*>(kres.basis[i].get()),
                reinterpret_cast<const std::complex<double>*>(kres.basis[j].get()),
                local_n);
            if (i == j) {
                worst_diag = std::max(worst_diag, std::abs(c.real() - 1.0));
                worst_diag = std::max(worst_diag, std::abs(c.imag()));
            } else {
                worst_off = std::max(worst_off, std::abs(c));
            }
        }
    }
    INFO("worst_diag=" << worst_diag << "  worst_off=" << worst_off);
    REQUIRE(worst_diag < 1e-10);
    REQUIRE(worst_off  < 1e-10);
}

#endif  // ED_HAVE_NCCL

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
