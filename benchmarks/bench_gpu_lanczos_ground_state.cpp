// =============================================================================
// benchmarks/bench_gpu_lanczos_ground_state.cpp
//
// Micro-benchmark for the GPU `GPULanczos::run()` ground-state path. The
// companion to bench_lanczos_ground_state.cpp on the CPU side.
//
// What we measure
// ---------------
//   * 1D Heisenberg chain, N in {12, 14, 16, 18}, PBC (matches CPU sweep).
//   * Cost of one full ground-state Lanczos run (build + iterate +
//     tridiagonal solve), excluding I/O.
//
// Each benchmark iteration creates a fresh GPULanczos instance on top of
// the same operator so the cuSPARSE CSR cache is reused. The op is built
// once outside the timed loop and cached inside the GPUOperator instance.
//
// Audit ref: GPU SOTA follow-up.
// =============================================================================

#include <benchmark/benchmark.h>

#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_lanczos.cuh>
#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

using Complex = std::complex<double>;

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::unique_ptr<GPUOperator>
make_gpu_heisenberg_chain(int N, bool periodic) {
    auto op = std::make_unique<GPUOperator>(N, /*spin=*/0.5f);
    const Complex J_real(1.0, 0.0);
    const Complex J_half(0.5, 0.0);
    const int last = periodic ? N : (N - 1);
    for (int i = 0; i < last; ++i) {
        const int j = (i + 1) % N;
        op->addTwoBodyTerm(/*op1=*/2, i, /*op2=*/2, j, J_real);
        op->addTwoBodyTerm(/*op1=*/0, i, /*op2=*/1, j, J_half);
        op->addTwoBodyTerm(/*op1=*/1, i, /*op2=*/0, j, J_half);
    }
    op->copyTransformDataToDevice();
    return op;
}

void BM_GPULanczosGroundState(benchmark::State& state) {
    if (!gpu_available()) {
        state.SkipWithError("No CUDA device available");
        return;
    }

    const auto N = static_cast<int>(state.range(0));
    const uint64_t dim = (1ULL << N);

    auto op = make_gpu_heisenberg_chain(N, /*periodic=*/true);

    // Warm-up: triggers cuSPARSE CSR build, kernel compilation, etc.
    {
        cuDoubleComplex* d_x = nullptr;
        cuDoubleComplex* d_y = nullptr;
        cudaMalloc(&d_x, dim * sizeof(cuDoubleComplex));
        cudaMalloc(&d_y, dim * sizeof(cuDoubleComplex));
        cudaMemset(d_x, 0, dim * sizeof(cuDoubleComplex));
        op->matVecGPU(d_x, d_y, static_cast<int>(dim));
        cudaDeviceSynchronize();
        cudaFree(d_x);
        cudaFree(d_y);
    }

    for (auto _ : state) {
        GPULanczos solver(op.get(), /*max_iter=*/100, /*tol=*/1e-12);
        std::vector<double> eigs;
        std::vector<std::vector<Complex>> evecs;
        solver.run(/*num_eigenvalues=*/1, eigs, evecs, /*compute_vectors=*/false);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(eigs);
        benchmark::ClobberMemory();
    }

    state.counters["dim"] = static_cast<double>(dim);
    state.counters["N"]   = static_cast<double>(N);
}

}  // namespace

BENCHMARK(BM_GPULanczosGroundState)
    ->Arg(12)->Arg(14)->Arg(16)->Arg(18)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);  // Lanczos runs are 10-200 ms each, 1s total is plenty

#else  // !WITH_CUDA

static void BM_GPULanczos_NoCUDA(benchmark::State& state) {
    for (auto _ : state) {}
    state.SkipWithError("Built without WITH_CUDA -- nothing to benchmark.");
}
BENCHMARK(BM_GPULanczos_NoCUDA);

#endif  // WITH_CUDA

BENCHMARK_MAIN();
