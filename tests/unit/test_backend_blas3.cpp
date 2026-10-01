// =============================================================================
// tests/unit/test_backend_blas3.cpp
//
// Pins the Level-3 BLAS surface on `Backend` (`gemm`)
// across the concrete backends:
//
//     CpuBackend    -- cBLAS path
//     CudaBackend   -- cuBLAS path
//
// FTLM dynamics forms its overlap matrices with it. This test covers
// correctness on small random inputs, not performance.
//
// Runtime SKIPs follow the same pattern as `test_cuda_backend.cpp`:
// build-without-CUDA hosts get the CpuBackend lane; the CudaBackend lane
// SKIPs gracefully when no GPU is visible.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/matvec/backends/cpu_backend.h>

#ifdef WITH_CUDA
#  include <ed/matvec/backends/cuda_backend.cuh>
#  include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

using Complex = std::complex<double>;

namespace {

constexpr double kTol = 1e-10;

// ---------------------------------------------------------------------------
// Reference host kernels (column-major). Naive but readable; they exist to
// give an algorithm-agnostic gold copy to compare against.
// ---------------------------------------------------------------------------

void ref_gemm(char opA, char opB,
              std::size_t m, std::size_t n, std::size_t k,
              Complex alpha,
              const Complex* A, std::size_t lda,
              const Complex* B, std::size_t ldb,
              Complex beta,
              Complex* C, std::size_t ldc) {
    auto a_at = [&](std::size_t i, std::size_t j) -> Complex {
        Complex v = (opA == 'N' || opA == 'n') ? A[i + j * lda] : A[j + i * lda];
        if (opA == 'C' || opA == 'c' || opA == 'H' || opA == 'h') v = std::conj(v);
        return v;
    };
    auto b_at = [&](std::size_t i, std::size_t j) -> Complex {
        Complex v = (opB == 'N' || opB == 'n') ? B[i + j * ldb] : B[j + i * ldb];
        if (opB == 'C' || opB == 'c' || opB == 'H' || opB == 'h') v = std::conj(v);
        return v;
    };
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            Complex acc(0.0, 0.0);
            for (std::size_t l = 0; l < k; ++l) acc += a_at(i, l) * b_at(l, j);
            C[i + j * ldc] = alpha * acc + beta * C[i + j * ldc];
        }
    }
}

double max_abs_diff(const Complex* a, const Complex* b, std::size_t n) {
    double w = 0.0;
    for (std::size_t i = 0; i < n; ++i) w = std::max(w, std::abs(a[i] - b[i]));
    return w;
}

std::vector<Complex> random_matrix(std::size_t rows, std::size_t cols,
                                   std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Complex> M(rows * cols);
    for (auto& z : M) z = Complex(uni(rng), uni(rng));
    return M;
}

#ifdef WITH_CUDA
bool cuda_available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return false; }
    return n > 0;
}
#endif

}  // namespace

// =============================================================================
// CpuBackend BLAS-3
// =============================================================================

TEST_CASE("CpuBackend::gemm matches naive reference",
          "[backend-blas3][cpu]") {
    ed::matvec::CpuBackend be;
    const std::size_t m = 7, n = 5, k = 4;
    auto A = random_matrix(m, k, 0xA1);
    auto B = random_matrix(k, n, 0xB2);
    auto C_be  = random_matrix(m, n, 0xC3);
    auto C_ref = C_be;

    const Complex alpha{0.7, -0.3}, beta{-0.4, 0.2};
    be.gemm('N', 'N', m, n, k, alpha,
            A.data(), m, B.data(), k, beta,
            C_be.data(), m);
    ref_gemm('N', 'N', m, n, k, alpha,
             A.data(), m, B.data(), k, beta,
             C_ref.data(), m);
    REQUIRE(max_abs_diff(C_be.data(), C_ref.data(), m * n) < kTol);
}

TEST_CASE("CpuBackend::gemm honors conjugate-transpose ops",
          "[backend-blas3][cpu]") {
    ed::matvec::CpuBackend be;
    const std::size_t m = 4, n = 5, k = 6;
    auto A = random_matrix(k, m, 0xA1);
    auto B = random_matrix(k, n, 0xB2);
    std::vector<Complex> C_be(m * n, Complex{0, 0});
    std::vector<Complex> C_ref(m * n, Complex{0, 0});

    be.gemm('C', 'N', m, n, k, Complex{1, 0},
            A.data(), k, B.data(), k, Complex{0, 0},
            C_be.data(), m);
    ref_gemm('C', 'N', m, n, k, Complex{1, 0},
             A.data(), k, B.data(), k, Complex{0, 0},
             C_ref.data(), m);
    REQUIRE(max_abs_diff(C_be.data(), C_ref.data(), m * n) < kTol);
}

// =============================================================================
// CudaBackend BLAS-3 (only when WITH_CUDA + a visible GPU)
// =============================================================================

#ifdef WITH_CUDA

TEST_CASE("CudaBackend::gemm matches naive reference",
          "[backend-blas3][cuda]") {
    if (!cuda_available()) { SUCCEED("no CUDA device, skipping"); return; }
    ed::matvec::CudaBackend be;
    const std::size_t m = 7, n = 5, k = 4;
    auto A = random_matrix(m, k, 0xA1);
    auto B = random_matrix(k, n, 0xB2);
    auto C_in  = random_matrix(m, n, 0xC3);
    auto C_ref = C_in;
    const Complex alpha{0.7, -0.3}, beta{-0.4, 0.2};
    ref_gemm('N', 'N', m, n, k, alpha,
             A.data(), m, B.data(), k, beta,
             C_ref.data(), m);

    auto dA = be.make_zero_vector(m * k);
    auto dB = be.make_zero_vector(k * n);
    auto dC = be.make_zero_vector(m * n);
    be.copy_from_host(A.data(), dA.get(), m * k);
    be.copy_from_host(B.data(), dB.get(), k * n);
    be.copy_from_host(C_in.data(), dC.get(), m * n);
    be.gemm('N', 'N', m, n, k, alpha,
            dA.get(), m, dB.get(), k, beta,
            dC.get(), m);
    std::vector<Complex> C_be(m * n);
    be.copy_to_host(dC.get(), C_be.data(), m * n);
    REQUIRE(max_abs_diff(C_be.data(), C_ref.data(), m * n) < kTol);
}

#endif  // WITH_CUDA
