// =============================================================================
// benchmarks/bench_gpu_operator_apply.cpp
//
// Micro-benchmark for the GPU `GPUOperator::matVecGPU(d_x, d_y, N)` hot
// path. The companion to bench_operator_apply.cpp on the CPU side.
//
// What we measure
// ---------------
//   * 1D Heisenberg chain, N in {12, 14, 16, 18, 20}, PBC.
//   * Cost of one full H*v on a deterministic random complex unit vector
//     that already lives on the GPU (so we time the kernel, not HtoD copies).
//   * Each iteration includes a single `cudaDeviceSynchronize()` so the
//     elapsed-time stopwatch reflects actual GPU work, not just kernel
//     launch latency. Without it the measurement just reflects the CPU's
//     ability to enqueue commands.
//
// What we DO NOT measure
// ----------------------
//   * Host<->device transfers (those are amortized in any real solver).
//   * The one-time CSR build + HtoD copy. We trigger it via a warm-up
//     matVec call, then start the timed loop.
//
// Results comparison
// ------------------
//   * Sweep variants: matrix-free (legacy) vs cuSPARSE (assembled CSR).
//     The matrix-free path is forced via ED_GPU_DISABLE_CUSPARSE=1; the
//     env var is read once on first matVec, so we also expose two
//     benchmark functions and document which to run for which path.
//   * Compare against bench_operator_apply.cpp (CPU) at matching N.
//
// Audit ref: GPU SOTA follow-up.
// =============================================================================

#include <benchmark/benchmark.h>

#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <iostream>

namespace {

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

// Mirror of bench_operator_apply.cpp::make_heisenberg_chain on the GPU
// side. J = 1, optionally PBC, spin-1/2.
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

ComplexVec random_unit_vector(uint64_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVec v(dim);
    double norm2 = 0.0;
    for (uint64_t i = 0; i < dim; ++i) {
        const double re = dist(rng);
        const double im = dist(rng);
        v[i] = Complex(re, im);
        norm2 += re * re + im * im;
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (auto& x : v) x *= inv;
    return v;
}

// Core timed loop. Pre-allocates GPU buffers once, copies the input vector
// in once, then loops over matVecGPU + cudaDeviceSynchronize.
void run_gpu_apply(benchmark::State& state, bool periodic) {
    if (!gpu_available()) {
        state.SkipWithError("No CUDA device available");
        return;
    }

    const auto N = static_cast<int>(state.range(0));
    const uint64_t dim = (1ULL << N);

    auto op = make_gpu_heisenberg_chain(N, periodic);

    ComplexVec h_in = random_unit_vector(dim, /*seed=*/42);

    // Allocate two device-side vectors that persist across the timed loop.
    cuDoubleComplex* d_x = nullptr;
    cuDoubleComplex* d_y = nullptr;
    cudaMalloc(&d_x, dim * sizeof(cuDoubleComplex));
    cudaMalloc(&d_y, dim * sizeof(cuDoubleComplex));
    cudaMemcpy(d_x, h_in.data(), dim * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Warm-up triggers lazy CSR build / kernel compilation / etc.
    op->matVecGPU(d_x, d_y, static_cast<int>(dim));
    cudaDeviceSynchronize();

    for (auto _ : state) {
        op->matVecGPU(d_x, d_y, static_cast<int>(dim));
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(d_y);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * dim);
    state.SetBytesProcessed(state.iterations() * dim * 2 * 16);
    state.counters["dim"] = static_cast<double>(dim);
    state.counters["N"]   = static_cast<double>(N);

    cudaFree(d_x);
    cudaFree(d_y);
}

void BM_GPUOperatorApply_PBC(benchmark::State& state) {
    run_gpu_apply(state, /*periodic=*/true);
}
void BM_GPUOperatorApply_OBC(benchmark::State& state) {
    run_gpu_apply(state, /*periodic=*/false);
}

}  // namespace

// Sweep the same dimensions as the CPU benchmark so head-to-head ratios
// fall out trivially.
BENCHMARK(BM_GPUOperatorApply_PBC)
    ->Arg(8)->Arg(10)->Arg(12)->Arg(14)->Arg(16)->Arg(18)->Arg(20)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_GPUOperatorApply_OBC)
    ->Arg(8)->Arg(10)->Arg(12)->Arg(14)->Arg(16)->Arg(18)->Arg(20)
    ->Unit(benchmark::kMicrosecond);

#else  // !WITH_CUDA

#include <iostream>
static void BM_GPU_NoCUDA(benchmark::State& state) {
    for (auto _ : state) {
        // No-op when CUDA is not built in.
    }
    state.SkipWithError("Built without WITH_CUDA -- nothing to benchmark.");
}
BENCHMARK(BM_GPU_NoCUDA);

#endif  // WITH_CUDA

BENCHMARK_MAIN();
