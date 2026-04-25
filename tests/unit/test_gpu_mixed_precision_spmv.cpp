// =============================================================================
// test_gpu_mixed_precision_spmv (Catch2 v3, Phase 3a #3)
//
// Lockdown for the FP32 cuSPARSE SpMV path on GPUOperator. The path is
// gated by two environment variables:
//
//   ED_GPU_CUSPARSE_MIN_DIM   - lowered to "1" for these tiny tests so the
//                               cuSPARSE pathway is chosen at all (default
//                               threshold is 32768; production GPU runs at
//                               N>=2^15 already pick CUSPARSE_CSR).
//   ED_GPU_MIXED_PRECISION_SPMV - "1" enables FP32 SpMV on the FP64 host
//                                 boundary; "0"/unset stays in FP64.
//
// Two tests:
//   1. H*v matches FP64 on a 10-site Heisenberg chain (dim=1024) within
//      a relative L2 of 5e-6 (about 100x looser than the 1e-10 FP64 path).
//   2. The Lanczos ground-state eigenvalue from a mixed-precision matvec
//      pipeline matches the dense reference within 1e-5 (CPU dot/normalize
//      stay FP64, so the ground state is still recoverable to nearly FP32
//      machine epsilon).
//
// Both tests SKIP if no CUDA device is present so the WITH_CUDA build-only
// CI lane does not fail. The runtime-skip pattern mirrors
// test_cpu_gpu_equivalence.cpp.
// =============================================================================

#include "common/catch2_harness.h"

#ifdef WITH_CUDA

#include <ed/gpu/gpu_mixed_precision.h>
#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_lanczos.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

using Complex = std::complex<double>;

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        return false;
    }
    return count > 0;
}

std::unique_ptr<GPUOperator> build_gpu_heisenberg_chain(int N, bool periodic) {
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

double l2_norm(const std::vector<Complex>& v) {
    double s = 0.0;
    for (const auto& x : v) s += std::norm(x);
    return std::sqrt(s);
}

double l2_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    double s = 0.0;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) s += std::norm(a[i] - b[i]);
    return std::sqrt(s);
}

// RAII: force the cuSPARSE pathway to be selectable at tiny N, and ensure
// neither knob leaks across tests.
struct ScopedMixedPrecisionEnv {
    ScopedMixedPrecisionEnv() {
        // Save current values so the destructor can restore them (so other
        // tests in the same Catch2 run can keep their env knobs).
        const char* mp = std::getenv("ED_GPU_MIXED_PRECISION_SPMV");
        const char* md = std::getenv("ED_GPU_CUSPARSE_MIN_DIM");
        prev_mp_ = mp ? mp : "";
        prev_md_ = md ? md : "";
        had_mp_  = mp != nullptr;
        had_md_  = md != nullptr;
        // Force cuSPARSE to be selected even at tiny dimension.
        setenv("ED_GPU_CUSPARSE_MIN_DIM", "1", /*overwrite=*/1);
        unsetenv("ED_GPU_MIXED_PRECISION_SPMV");
    }
    ~ScopedMixedPrecisionEnv() {
        if (had_mp_) setenv("ED_GPU_MIXED_PRECISION_SPMV", prev_mp_.c_str(), 1);
        else         unsetenv("ED_GPU_MIXED_PRECISION_SPMV");
        if (had_md_) setenv("ED_GPU_CUSPARSE_MIN_DIM",     prev_md_.c_str(), 1);
        else         unsetenv("ED_GPU_CUSPARSE_MIN_DIM");
    }
    std::string prev_mp_;
    std::string prev_md_;
    bool        had_mp_ = false;
    bool        had_md_ = false;
};

}  // namespace

// ============================================================================
// 1. H*v matches FP64 within ~5e-6 relative L2
// ============================================================================
TEST_CASE("GPU mixed-precision H*v matches FP64 within FP32 epsilon",
          "[gpu_mixed_precision][matvec]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available -- skipping mixed-precision test.");
    }

    ScopedMixedPrecisionEnv env_guard;

    const int N = 10;                  // dim = 1024
    const uint64_t dim = 1ULL << N;
    auto v = ed_tests::random_unit_vector(dim, /*seed=*/24681357);

    // ---- FP64 reference --------------------------------------------------
    std::vector<Complex> y_fp64(dim, Complex(0.0, 0.0));
    {
        REQUIRE_FALSE(ed::gpu::gpu_mixed_precision_spmv_enabled());
        auto op_fp64 = build_gpu_heisenberg_chain(N, /*periodic=*/true);
        op_fp64->matVec(v.data(), y_fp64.data(), static_cast<int>(dim));
    }

    // ---- Mixed-precision -------------------------------------------------
    std::vector<Complex> y_mixed(dim, Complex(0.0, 0.0));
    {
        setenv("ED_GPU_MIXED_PRECISION_SPMV", "1", /*overwrite=*/1);
        REQUIRE(ed::gpu::gpu_mixed_precision_spmv_enabled());
        auto op_mixed = build_gpu_heisenberg_chain(N, /*periodic=*/true);
        op_mixed->matVec(v.data(), y_mixed.data(), static_cast<int>(dim));
    }

    const double err  = l2_diff(y_fp64, y_mixed);
    const double norm = l2_norm(y_fp64);
    const double rel  = err / std::max(norm, 1e-30);
    INFO("dim=" << dim
         << "  ||fp64 - mixed||_2 = " << err
         << "  ||fp64||_2 = "          << norm
         << "  rel = "                  << rel);
    // FP32 has ~7 decimal digits; the cumulative rounding error from
    // O(N) accumulations is bounded by ~N * eps_fp32 ~ 1024 * 1.2e-7 ~ 1e-4.
    // 5e-6 leaves a healthy margin for the actual observed error on this
    // tridiagonal-flavoured operator (which is much sparser per row).
    REQUIRE(rel < 5e-6);
}

// ============================================================================
// 2. Lanczos ground state with mixed-precision matvec matches dense ref
// ============================================================================
TEST_CASE("GPU mixed-precision Lanczos ground state matches dense reference",
          "[gpu_mixed_precision][lanczos]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available -- skipping mixed-precision test.");
    }

    ScopedMixedPrecisionEnv env_guard;

    const int N = 8;                   // dim = 256
    const uint64_t dim = 1ULL << N;

    // Dense reference via the CPU operator (cheap at dim=256).
    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0,
                                                    /*periodic=*/true);
    auto ref = ed_tests::reference_from_operator(*cpu_op, dim);
    const double E_ref = ref.eigs.front();

    // Mixed-precision GPU Lanczos.
    setenv("ED_GPU_MIXED_PRECISION_SPMV", "1", /*overwrite=*/1);
    REQUIRE(ed::gpu::gpu_mixed_precision_spmv_enabled());

    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    GPULanczos gpu_lanczos(gpu_op.get(), /*max_iter=*/100,
                           /*tolerance=*/1e-12);
    std::vector<double> eigs;
    std::vector<std::vector<Complex>> vecs;
    gpu_lanczos.run(/*num_eigenvalues=*/1, eigs, vecs,
                    /*compute_vectors=*/false);
    REQUIRE_FALSE(eigs.empty());

    INFO("dim=" << dim
         << "  E_ref="   << E_ref
         << "  E_mixed=" << eigs.front()
         << "  |Δ|="     << std::abs(eigs.front() - E_ref));
    // Ground state is preserved at much better than FP32 epsilon because
    // the *outer* dot/normalize/axpy stay in FP64 (cuBLAS Z* on FP64
    // vectors); only the per-iteration matvec runs in FP32. The Lanczos
    // recurrence's O(1) condition number on the ground-state eigenpair
    // means the ground-state energy converges to well within 1e-5 even
    // at FP32 SpMV precision.
    REQUIRE(std::abs(eigs.front() - E_ref) < 1e-5);
}

#else  // !WITH_CUDA

TEST_CASE("GPU mixed-precision SpMV: WITH_CUDA=OFF -- no-op",
          "[gpu_mixed_precision][skip]") {
    SUCCEED("Built without WITH_CUDA -- nothing to test.");
}

#endif  // WITH_CUDA
