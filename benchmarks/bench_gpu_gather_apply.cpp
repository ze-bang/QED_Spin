// =============================================================================
// benchmarks/bench_gpu_gather_apply.cpp
//
// Raw throughput of the CudaMatVecBackend gather SpMV (the lane the GPU
// Lanczos / thermal / spectral workflows use) on a fixed-Sz Heisenberg chain:
// device-resident vectors, no host sync inside the loop, one sync at the end.
//
//   ./build/benchmarks/bench_gpu_gather_apply 18 20 22 24
// =============================================================================
#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/construct_ham.h>
#include <ed/core/linear_operator.h>
#include <ed/matvec/backends/cuda_backend.cuh>
#include <cuda_runtime.h>

#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

using Complex = std::complex<double>;

static std::unique_ptr<FixedSzOperator> chain(std::uint64_t N) {
    auto op = std::make_unique<FixedSzOperator>(N, 0.5f, static_cast<std::int64_t>(N / 2));
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.is_two_body = true;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2; t.site_index_2 = j; t.coefficient = Complex(1.0, 0.0);
        op->transform_data_.push_back(t);
        t.op_type = 0; t.site_index = i; t.op_type_2 = 1; t.site_index_2 = j; t.coefficient = Complex(0.5, 0.0);
        op->transform_data_.push_back(t);
        t.op_type = 1; t.site_index = i; t.op_type_2 = 0; t.site_index_2 = j; t.coefficient = Complex(0.5, 0.0);
        op->transform_data_.push_back(t);
    }
    return op;
}

int main(int argc, char** argv) {
    std::vector<std::uint64_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<std::uint64_t>(std::atoll(argv[i])));
    if (sizes.empty()) sizes = {18, 20, 22};
    const int reps = std::getenv("BENCH_REPEATS") ? std::atoi(std::getenv("BENCH_REPEATS")) : 100;
    ed::matvec::CudaBackend be;
    for (auto N : sizes) {
        auto op = chain(N);
        const std::size_t n = op->dim();
        auto mv = op->bind_cuda();
        auto x = be.make_zero_vector(n), y = be.make_zero_vector(n);
        std::vector<Complex> host(n);
        for (std::size_t i = 0; i < n; ++i) host[i] = Complex(std::sin(0.001 * i), std::cos(0.002 * i));
        be.copy_from_host(host.data(), x.get(), n);
        mv(x.get(), y.get(), n);                          // warm-up (mirror build)
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) mv(x.get(), y.get(), n);
        cudaDeviceSynchronize();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        const double nnz = static_cast<double>(n) * (1.0 + 2.0 * N / 2.0);   // diag + ~N/2 hopping partners
        std::printf("{\"N\": %llu, \"dim\": %zu, \"us_per_apply\": %.1f, \"Gnnz_per_s\": %.2f, \"y0\": %.6f}\n",
                    static_cast<unsigned long long>(N), n, us, nnz / us * 1e-3, 0.0);
    }
    return 0;
}
