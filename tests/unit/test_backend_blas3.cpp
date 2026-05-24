// =============================================================================
// tests/unit/test_backend_blas3.cpp
//
// Phase 1 of the Minimalist ED Collapse (May 2026): lock down the new
// Level-3 BLAS surface on `Backend` (`gemm`, `gemv`, `trsm`, `qr_thin`)
// across the four concrete backends:
//
//     CpuBackend    -- LAPACK / cBLAS path
//     CudaBackend   -- cuBLAS + cuSolver path
//     MpiBackend    -- inherited local + CholeskyQR2-based qr_thin
//     MpiCudaBackend - inherited local + NCCL CholeskyQR2 qr_thin
//
// The new BLAS-3 surface is the foundation block-Lanczos needs once it
// migrates to `block_lanczos_kernel<Backend>` (Phase 2.3 of the same
// rollout). This test covers correctness on small random inputs, not
// performance.
//
// Runtime SKIPs follow the same pattern as `test_cuda_backend.cpp` and
// `test_mpi_cuda_backend.cpp`: build-without-CUDA hosts get the
// CpuBackend lane; the CudaBackend / MpiCudaBackend lanes SKIP
// gracefully when no GPU / NCCL is visible.
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

// Check that Q (m x b column-major) has orthonormal columns
// (||Q^H Q - I||_inf <= tol).
double orthonormality_error(const Complex* Q, std::size_t m, std::size_t b) {
    std::vector<Complex> G(b * b);
    ref_gemm('C', 'N', b, b, m, Complex{1, 0}, Q, m, Q, m,
             Complex{0, 0}, G.data(), b);
    double worst = 0.0;
    for (std::size_t j = 0; j < b; ++j) {
        for (std::size_t i = 0; i < b; ++i) {
            const Complex target = (i == j) ? Complex{1, 0} : Complex{0, 0};
            worst = std::max(worst, std::abs(G[i + j * b] - target));
        }
    }
    return worst;
}

// Check that Q * R recovers the input matrix A (m x b column-major).
double qr_reconstruction_error(const Complex* Q, const Complex* R,
                                const Complex* A_in,
                                std::size_t m, std::size_t b) {
    std::vector<Complex> QR(m * b);
    ref_gemm('N', 'N', m, b, b, Complex{1, 0}, Q, m, R, b,
             Complex{0, 0}, QR.data(), m);
    return max_abs_diff(QR.data(), A_in, m * b);
}

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

TEST_CASE("CpuBackend::gemv matches naive reference",
          "[backend-blas3][cpu]") {
    ed::matvec::CpuBackend be;
    const std::size_t m = 9, n = 6;
    auto A = random_matrix(m, n, 0xA1);
    auto x = random_matrix(n, 1, 0xB2);
    auto y_be  = random_matrix(m, 1, 0xC3);
    auto y_ref = y_be;
    const Complex alpha{0.5, 0.1}, beta{-0.2, 0.3};
    be.gemv('N', m, n, alpha, A.data(), m, x.data(), 1, beta, y_be.data(), 1);
    // Reference via gemm shape (m x n) * (n x 1) -> (m x 1).
    ref_gemm('N', 'N', m, 1, n, alpha,
             A.data(), m, x.data(), n, beta,
             y_ref.data(), m);
    REQUIRE(max_abs_diff(y_be.data(), y_ref.data(), m) < kTol);
}

TEST_CASE("CpuBackend::trsm solves R X = alpha B (right, upper, non-unit)",
          "[backend-blas3][cpu]") {
    ed::matvec::CpuBackend be;
    const std::size_t m = 7, b = 4;
    // R: upper-triangular with non-zero diagonal.
    std::vector<Complex> R(b * b, Complex{0, 0});
    std::mt19937_64 rng(0xDEAD);
    std::uniform_real_distribution<double> uni(-0.5, 0.5);
    for (std::size_t j = 0; j < b; ++j) {
        for (std::size_t i = 0; i <= j; ++i) {
            R[i + j * b] = Complex(uni(rng), uni(rng));
        }
        R[j + j * b] += Complex(2.0, 0.0);   // dominant diagonal -> well-conditioned
    }
    auto B    = random_matrix(m, b, 0xB3);
    auto Bin  = B;
    be.trsm('R', 'U', 'N', 'N', m, b, Complex{1, 0},
            R.data(), b, B.data(), m);
    // Verify B * R reconstructs Bin.
    std::vector<Complex> BR(m * b, Complex{0, 0});
    ref_gemm('N', 'N', m, b, b, Complex{1, 0},
             B.data(), m, R.data(), b,
             Complex{0, 0}, BR.data(), m);
    REQUIRE(max_abs_diff(BR.data(), Bin.data(), m * b) < kTol);
}

TEST_CASE("CpuBackend::qr_thin produces orthonormal Q and recovers A=Q*R",
          "[backend-blas3][cpu][qr]") {
    ed::matvec::CpuBackend be;
    const std::size_t m = 32, b = 5;
    auto A_in = random_matrix(m, b, 0xACEFACE);
    auto A    = A_in;
    std::vector<Complex> R(b * b);
    be.qr_thin(A.data(), m, b, R.data());

    REQUIRE(orthonormality_error(A.data(), m, b) < 1e-12);
    REQUIRE(qr_reconstruction_error(A.data(), R.data(), A_in.data(), m, b) < 1e-12);
}

TEST_CASE("CpuBackend::all_reduce_sum_vec is a no-op on the host",
          "[backend-blas3][cpu]") {
    ed::matvec::CpuBackend be;
    std::vector<Complex> v{Complex(1, 2), Complex(3, -4), Complex(-5, 6)};
    auto copy = v;
    be.all_reduce_sum_vec(v.data(), v.size());
    REQUIRE(max_abs_diff(v.data(), copy.data(), v.size()) == 0.0);
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

TEST_CASE("CudaBackend::qr_thin produces orthonormal Q and recovers A=Q*R",
          "[backend-blas3][cuda][qr]") {
    if (!cuda_available()) { SUCCEED("no CUDA device, skipping"); return; }
    ed::matvec::CudaBackend be;
    const std::size_t m = 64, b = 4;
    auto A_in = random_matrix(m, b, 0xCAFEFEED);

    auto dA = be.make_zero_vector(m * b);
    be.copy_from_host(A_in.data(), dA.get(), m * b);
    std::vector<Complex> R(b * b);
    be.qr_thin(dA.get(), m, b, R.data());

    std::vector<Complex> Q(m * b);
    be.copy_to_host(dA.get(), Q.data(), m * b);

    REQUIRE(orthonormality_error(Q.data(), m, b) < 1e-10);
    REQUIRE(qr_reconstruction_error(Q.data(), R.data(), A_in.data(), m, b) < 1e-10);
}

TEST_CASE("CudaBackend::trsm round-trips against the gemm reference",
          "[backend-blas3][cuda]") {
    if (!cuda_available()) { SUCCEED("no CUDA device, skipping"); return; }
    ed::matvec::CudaBackend be;
    const std::size_t m = 12, b = 4;
    // Build a well-conditioned R.
    std::vector<Complex> R(b * b, Complex{0, 0});
    std::mt19937_64 rng(0xBEEF);
    std::uniform_real_distribution<double> uni(-0.5, 0.5);
    for (std::size_t j = 0; j < b; ++j) {
        for (std::size_t i = 0; i <= j; ++i) R[i + j * b] = Complex(uni(rng), uni(rng));
        R[j + j * b] += Complex(2.0, 0.0);
    }
    auto B   = random_matrix(m, b, 0xBABE);
    auto Bin = B;
    auto dR = be.make_zero_vector(b * b);
    auto dB = be.make_zero_vector(m * b);
    be.copy_from_host(R.data(), dR.get(), b * b);
    be.copy_from_host(B.data(), dB.get(), m * b);
    be.trsm('R', 'U', 'N', 'N', m, b, Complex{1, 0}, dR.get(), b, dB.get(), m);
    std::vector<Complex> Bsolved(m * b);
    be.copy_to_host(dB.get(), Bsolved.data(), m * b);

    // Recompute Bsolved * R; should equal Bin to working precision.
    std::vector<Complex> BR(m * b, Complex{0, 0});
    ref_gemm('N', 'N', m, b, b, Complex{1, 0},
             Bsolved.data(), m, R.data(), b,
             Complex{0, 0}, BR.data(), m);
    REQUIRE(max_abs_diff(BR.data(), Bin.data(), m * b) < 1e-10);
}

#endif  // WITH_CUDA
