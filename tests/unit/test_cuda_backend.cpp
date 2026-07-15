// =============================================================================
// tests/unit/test_cuda_backend.cpp
//
// Phase 2 of the matvec-unification (May 2026): pin agreement between
// `lanczos_kernel<CpuBackend>` and `lanczos_kernel<CudaBackend>` on a
// small Heisenberg chain. Two independent lanes:
//
//   1. Backend BLAS-1 round-trips (alloc + H2D + axpy + dot + nrm2 + D2H).
//   2. `lanczos_kernel(...)` ground-state energy with both backends,
//      driving the same matvec callable through cuBLAS on the device
//      side. Demonstrates that the kernel is *actually* backend-
//      agnostic: the algorithm body in `ed/krylov/lanczos_kernel.h`
//      runs unchanged against the CUDA implementation of `Backend`.
//
// Runtime SKIPs (Catch2 SUCCEED + return) keep the build-only CUDA lane
// happy on CI hosts without an attached GPU, exactly like
// `test_cpu_gpu_equivalence.cpp` does.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#ifdef WITH_CUDA

#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/backends/cuda_backend.cuh>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/thermal/kpm_dos_kernel.h>

#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_solvers.h>  // run_lanczos_eigenvalues_kernel_facade
#include <ed/solvers/lanczos.h>  // CPU `lanczos(...)` reference

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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

// Mirror of `build_heisenberg_chain` (CPU, test_harness.h) for the GPU
// term-storage path, so the cuBLAS lane sees the same Hamiltonian.
std::unique_ptr<GPUOperator>
build_gpu_heisenberg_chain(int N, bool periodic) {
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

}  // namespace

TEST_CASE("matvec::CudaBackend round-trips its BLAS-1 primitives",
          "[cuda-backend][matvec-unification][phase2]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr std::size_t n = 1024;

    ed::matvec::CudaBackend cuda;
    REQUIRE(cuda.memory_space() == ed::matvec::MemorySpace::CudaDevice);

    // Host reference data.
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Complex> h_x(n), h_y(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_x[i] = Complex(uni(rng), uni(rng));
        h_y[i] = Complex(uni(rng), uni(rng));
    }

    auto d_x = cuda.make_zero_vector(n);
    auto d_y = cuda.make_zero_vector(n);
    cuda.copy_from_host(h_x.data(), d_x.get(), n);
    cuda.copy_from_host(h_y.data(), d_y.get(), n);

    // ---- nrm2 ----
    const double nrm_dev  = cuda.nrm2(d_x.get(), n);
    double nrm_ref_sq = 0.0;
    for (auto z : h_x) nrm_ref_sq += std::norm(z);
    REQUIRE(std::abs(nrm_dev - std::sqrt(nrm_ref_sq)) < 1e-10 * (1.0 + std::abs(nrm_dev)));

    // ---- dot (conj on left) ----
    const Complex dot_dev = cuda.dot(d_x.get(), d_y.get(), n);
    Complex dot_ref(0.0, 0.0);
    for (std::size_t i = 0; i < n; ++i) dot_ref += std::conj(h_x[i]) * h_y[i];
    REQUIRE(std::abs(dot_dev - dot_ref) < 1e-10 * (1.0 + std::abs(dot_ref)));

    // ---- axpy: y <- (1+2i)*x + y ----
    const Complex alpha(1.0, 2.0);
    cuda.axpy(alpha, d_x.get(), d_y.get(), n);
    std::vector<Complex> h_y_after(n);
    cuda.copy_to_host(d_y.get(), h_y_after.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        const Complex expected = alpha * h_x[i] + h_y[i];
        REQUIRE(std::abs(h_y_after[i] - expected) < 1e-12);
    }

    // ---- scale: x <- 0.5 * x ----
    cuda.scale(Complex(0.5, 0.0), d_x.get(), n);
    std::vector<Complex> h_x_after(n);
    cuda.copy_to_host(d_x.get(), h_x_after.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(std::abs(h_x_after[i] - 0.5 * h_x[i]) < 1e-12);
    }
}

TEST_CASE("krylov::lanczos_kernel matches CPU vs CUDA backend on 6-site chain",
          "[cuda-backend][lanczos-kernel][phase2]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int    N   = 6;
    constexpr std::size_t dim = std::size_t{1} << N;

    // ---- CPU reference ----
    auto cpu_H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    ed::matvec::CpuBackend cpu;
    std::vector<Complex> v0(dim, Complex(0.0, 0.0));
    v0[0] = Complex(1.0, 0.0);

    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter = 24;
    opts.reorth   = ed::krylov::ReorthPolicy::FullCGS2;
    opts.keep_basis = true;

    auto cpu_res = ed::krylov::lanczos_kernel(
        cpu,
        [&](const Complex* in, Complex* out, std::size_t n) {
            cpu_H->apply(in, out, n);
        },
        dim, v0.data(), opts);
    REQUIRE(cpu_res.alpha.size() > 0);

    // ---- CUDA lane: same kernel, CUDA backend ----
    auto gpu_H = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    ed::matvec::CudaBackend cuda;

    auto d_v0 = cuda.make_zero_vector(dim);
    cuda.copy_from_host(v0.data(), d_v0.get(), dim);

    auto cuda_res = ed::krylov::lanczos_kernel(
        cuda,
        [&](const Complex* in, Complex* out, std::size_t n) {
            gpu_H->matVecGPU(
                reinterpret_cast<const cuDoubleComplex*>(in),
                reinterpret_cast<cuDoubleComplex*>(out),
                static_cast<int>(n));
        },
        dim, d_v0.get(), opts);
    REQUIRE(cuda_res.alpha.size() == cpu_res.alpha.size());
    REQUIRE(cuda_res.beta.size()  == cpu_res.beta.size());

    // Pin equality of every Ritz-tridiagonal coefficient. Both backends
    // ran the same algorithmic body; differences here can only come from
    // (a) cuBLAS using a different reduction tree (~1e-12 round-off) or
    // (b) the GPU SpMV path computing H*v with a different summation
    // order than the CPU path (also ~1e-12). 1e-10 is comfortably above
    // both noise floors.
    for (std::size_t k = 0; k < cpu_res.alpha.size(); ++k) {
        REQUIRE(std::abs(cpu_res.alpha[k] - cuda_res.alpha[k]) < 1e-10);
    }
    for (std::size_t k = 0; k < cpu_res.beta.size(); ++k) {
        REQUIRE(std::abs(cpu_res.beta[k] - cuda_res.beta[k]) < 1e-10);
    }
}

TEST_CASE("gpu::run_lanczos_eigenvalues_kernel_facade matches CPU lanczos on N=8 Heisenberg",
          "[cuda-backend][lanczos-kernel][phase2][facade]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int    N   = 8;
    constexpr std::size_t dim = std::size_t{1} << N;

    // CPU reference: legacy `lanczos(Hv, ...)` entry point in
    // ed/solvers/lanczos.h, eigenvalues-only. This is the same reference
    // path test_cpu_gpu_equivalence uses to pin GPULanczos::run.
    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        cpu_op->apply(in, out, static_cast<std::size_t>(n));
    };
    std::vector<double> cpu_eigs;
    ::lanczos(Hv, dim, /*max_iter=*/100, /*exct=*/1, /*tol=*/1e-12,
              cpu_eigs, /*dir=*/"", /*eigenvectors=*/false);
    REQUIRE(!cpu_eigs.empty());

    // GPU lane: the new kernel-facade entry point used in production by
    // `runGPULanczos(...)` when `eigenvectors=false`.
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    std::vector<double> facade_eigs;
    ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
        *gpu_op,
        /*N=*/static_cast<int>(dim),
        /*max_iter=*/100,
        /*num_eigs=*/1,
        /*tol=*/1e-12,
        /*seed=*/42ULL,
        facade_eigs);
    REQUIRE(!facade_eigs.empty());

    INFO("E_cpu=" << cpu_eigs[0]
         << "  E_facade=" << facade_eigs[0]
         << "  |Δ|=" << std::abs(cpu_eigs[0] - facade_eigs[0]));
    // The CPU & GPU lanczos seed their starting vectors differently
    // (std::mt19937 vs curand), so the recurrence runs through different
    // Ritz iterates -- but both converge to the same ground-state energy.
    // 1e-8 matches the threshold the legacy test_cpu_gpu_equivalence uses
    // for the same comparison.
    REQUIRE(std::abs(cpu_eigs[0] - facade_eigs[0]) < 1e-8);
}

TEST_CASE("gpu::run_lanczos_eigenvalues_kernel_facade input validation",
          "[cuda-backend][facade]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    auto gpu_op = build_gpu_heisenberg_chain(/*N=*/4, /*periodic=*/true);
    std::vector<double> out;

    REQUIRE_THROWS_AS(
        ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
            *gpu_op, /*N=*/-1, 10, 1, 1e-10, 42ULL, out),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
            *gpu_op, /*N=*/16, /*max_iter=*/0, 1, 1e-10, 42ULL, out),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
            *gpu_op, /*N=*/16, 10, /*num_eigs=*/0, 1e-10, 42ULL, out),
        std::invalid_argument);
}

TEST_CASE("gpu::run_lanczos_eigenpairs_kernel_facade returns a valid "
          "eigenpair on N=8 Heisenberg",
          "[cuda-backend][facade][ritz-recon]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int    N   = 8;
    constexpr std::size_t dim = std::size_t{1} << N;

    // CPU reference for the ground-state energy.
    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        cpu_op->apply(in, out, static_cast<std::size_t>(n));
    };
    std::vector<double> cpu_eigs;
    ::lanczos(Hv, dim, /*max_iter=*/100, /*exct=*/1, /*tol=*/1e-12,
              cpu_eigs, /*dir=*/"", /*eigenvectors=*/false);
    REQUIRE(!cpu_eigs.empty());

    // GPU facade with eigvec recovery.
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    std::vector<double> facade_eigs;
    std::vector<std::vector<Complex>> facade_vecs;
    ed::matvec::gpu::run_lanczos_eigenpairs_kernel_facade(
        *gpu_op,
        /*N=*/static_cast<int>(dim),
        /*max_iter=*/100,
        /*num_eigs=*/1,
        /*tol=*/1e-12,
        /*seed=*/42ULL,
        facade_eigs,
        facade_vecs);

    REQUIRE(facade_eigs.size() == 1);
    REQUIRE(facade_vecs.size() == 1);
    REQUIRE(facade_vecs[0].size() == dim);

    INFO("E_cpu=" << cpu_eigs[0]
         << "  E_facade=" << facade_eigs[0]
         << "  |Δ|=" << std::abs(cpu_eigs[0] - facade_eigs[0]));
    REQUIRE(std::abs(cpu_eigs[0] - facade_eigs[0]) < 1e-8);

    // Norm check: each reconstructed Ritz vector should be unit-norm
    // (up to roundoff). T's eigenvectors are orthonormal, so the sum
    //   y = sum_k S(k,0) * V_k
    // of orthonormal V_k weighted by an orthonormal column of S has
    // norm exactly 1 (modulo the breakdown_tol-controlled truncation).
    double sq = 0.0;
    for (const auto& z : facade_vecs[0]) sq += std::norm(z);
    INFO("||y_0|| = " << std::sqrt(sq));
    REQUIRE(std::abs(std::sqrt(sq) - 1.0) < 1e-8);

    // Residual check: ||H y - lambda y|| / ||y|| should be small for
    // a true Ritz pair on a fully-converged Krylov subspace.
    std::vector<Complex> Hy(dim);
    cpu_op->apply(facade_vecs[0].data(), Hy.data(), dim);
    double res2 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const Complex r = Hy[i] - facade_eigs[0] * facade_vecs[0][i];
        res2 += std::norm(r);
    }
    INFO("||H y_0 - lambda y_0|| = " << std::sqrt(res2));
    REQUIRE(std::sqrt(res2) < 1e-6);
}

TEST_CASE("gpu::run_lanczos_eigenpairs_kernel_facade recovers multiple "
          "orthonormal Ritz pairs on N=6 Heisenberg",
          "[cuda-backend][facade][ritz-recon]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int    N    = 6;
    constexpr std::size_t dim = std::size_t{1} << N;
    constexpr int    K    = 3;  // request the lowest 3 Ritz pairs

    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    std::vector<double> evals;
    std::vector<std::vector<Complex>> evecs;
    ed::matvec::gpu::run_lanczos_eigenpairs_kernel_facade(
        *gpu_op,
        /*N=*/static_cast<int>(dim),
        /*max_iter=*/50,
        /*num_eigs=*/K,
        /*tol=*/1e-12,
        /*seed=*/123ULL,
        evals,
        evecs);

    REQUIRE(evals.size() == K);
    REQUIRE(evecs.size() == K);
    for (const auto& v : evecs) REQUIRE(v.size() == dim);

    // Ascending eigenvalues.
    for (int i = 1; i < K; ++i) {
        REQUIRE(evals[i - 1] <= evals[i] + 1e-10);
    }

    // Each reconstructed Ritz vector is unit-norm and they are
    // pairwise orthogonal (since T's eigenvectors are orthonormal
    // and the Krylov basis is orthonormal, so column-i and column-j
    // are <S_i, S_j> = delta_{ij}). Slack 1e-8 captures cuBLAS
    // round-off in the axpy chain.
    auto cpu_op = ed_tests::build_heisenberg_chain(N, 1.0, true);
    for (int i = 0; i < K; ++i) {
        double sq = 0.0;
        for (const auto& z : evecs[i]) sq += std::norm(z);
        INFO("||y_" << i << "|| = " << std::sqrt(sq));
        REQUIRE(std::abs(std::sqrt(sq) - 1.0) < 1e-8);

        // Residual for THIS Ritz pair.
        std::vector<Complex> Hy(dim);
        cpu_op->apply(evecs[i].data(), Hy.data(), dim);
        double res2 = 0.0;
        for (std::size_t k = 0; k < dim; ++k) {
            const Complex r = Hy[k] - evals[i] * evecs[i][k];
            res2 += std::norm(r);
        }
        INFO("||H y_" << i << " - lambda_" << i << " y_" << i
             << "|| = " << std::sqrt(res2));
        REQUIRE(std::sqrt(res2) < 1e-6);
    }
    for (int i = 0; i < K; ++i) {
        for (int j = i + 1; j < K; ++j) {
            Complex ip(0.0, 0.0);
            for (std::size_t k = 0; k < dim; ++k) {
                ip += std::conj(evecs[i][k]) * evecs[j][k];
            }
            INFO("<y_" << i << ", y_" << j << "> = " << ip);
            REQUIRE(std::abs(ip) < 1e-8);
        }
    }
}

TEST_CASE("matvec::CudaBackend batched dot_many matches the sequential reference",
          "[cuda-backend][batched-primitives][phase1]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    // Phase 1 gap-fill regression: `dot_many` is overridden via one
    // `cublasZgemv` over a staged contiguous (n x M) basis buffer. Pin
    // that the override returns the same coefficient vector as M
    // sequential `dot()` calls within strict tolerance. Runs across a
    // range of M values, including the M=1 trivial path and a
    // FTLM-typical M=100 case.
    ed::matvec::CudaBackend cuda;
    std::mt19937_64 rng(0xD07BA7CULL);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for (std::size_t M : {std::size_t(1), std::size_t(5), std::size_t(25), std::size_t(100)}) {
        constexpr std::size_t n = 2048;

        // Random v on host then push to device.
        std::vector<Complex> h_v(n);
        for (auto& z : h_v) z = Complex(uni(rng), uni(rng));
        auto d_v = cuda.make_zero_vector(n);
        cuda.copy_from_host(h_v.data(), d_v.get(), n);

        // M random basis vectors, each its own device allocation
        // (mirrors how `lanczos_kernel` lays out the basis).
        std::vector<std::vector<Complex>> h_basis(M, std::vector<Complex>(n));
        std::vector<ed::matvec::Backend::UniqueVec> d_basis;
        std::vector<const Complex*> basis_ptrs(M);
        d_basis.reserve(M);
        for (std::size_t k = 0; k < M; ++k) {
            for (auto& z : h_basis[k]) z = Complex(uni(rng), uni(rng));
            auto buf = cuda.make_zero_vector(n);
            cuda.copy_from_host(h_basis[k].data(), buf.get(), n);
            basis_ptrs[k] = buf.get();
            d_basis.emplace_back(std::move(buf));
        }

        // Batched override.
        std::vector<Complex> coeffs_batched(M);
        cuda.dot_many(basis_ptrs.data(), M, d_v.get(), n, coeffs_batched.data());

        // Sequential reference: M single-vector `dot` calls.
        std::vector<Complex> coeffs_seq(M);
        for (std::size_t k = 0; k < M; ++k) {
            coeffs_seq[k] = cuda.dot(basis_ptrs[k], d_v.get(), n);
        }

        for (std::size_t k = 0; k < M; ++k) {
            INFO("M=" << M << " k=" << k
                 << " batched=" << coeffs_batched[k]
                 << " sequential=" << coeffs_seq[k]);
            REQUIRE(std::abs(coeffs_batched[k] - coeffs_seq[k])
                    < 1e-12 * (1.0 + std::abs(coeffs_seq[k])));
        }
    }
}

TEST_CASE("matvec::CudaBackend batched axpy_many matches the sequential reference",
          "[cuda-backend][batched-primitives][phase1]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    // Phase 1 gap-fill regression: `axpy_many` is overridden via one
    // `cublasZgemv(OP_N)` over the same staged basis buffer. Pin that
    // the override produces the same result as the equivalent loop of
    // M `axpy` calls.
    ed::matvec::CudaBackend cuda;
    std::mt19937_64 rng(0xAFEED2A9ULL);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for (std::size_t M : {std::size_t(1), std::size_t(5), std::size_t(25), std::size_t(100)}) {
        constexpr std::size_t n = 2048;

        std::vector<Complex> h_y0(n);
        for (auto& z : h_y0) z = Complex(uni(rng), uni(rng));

        // Two parallel device copies of y0: one for the batched lane,
        // one for the sequential reference. Both start identical.
        auto d_y_batched = cuda.make_zero_vector(n);
        auto d_y_seq     = cuda.make_zero_vector(n);
        cuda.copy_from_host(h_y0.data(), d_y_batched.get(), n);
        cuda.copy_from_host(h_y0.data(), d_y_seq.get(),     n);

        std::vector<std::vector<Complex>> h_basis(M, std::vector<Complex>(n));
        std::vector<ed::matvec::Backend::UniqueVec> d_basis;
        std::vector<const Complex*> basis_ptrs(M);
        d_basis.reserve(M);
        for (std::size_t k = 0; k < M; ++k) {
            for (auto& z : h_basis[k]) z = Complex(uni(rng), uni(rng));
            auto buf = cuda.make_zero_vector(n);
            cuda.copy_from_host(h_basis[k].data(), buf.get(), n);
            basis_ptrs[k] = buf.get();
            d_basis.emplace_back(std::move(buf));
        }

        std::vector<Complex> alphas(M);
        for (auto& a : alphas) a = Complex(uni(rng), uni(rng));

        // Batched override.
        cuda.axpy_many(alphas.data(), basis_ptrs.data(), M,
                       d_y_batched.get(), n);

        // Sequential reference.
        for (std::size_t k = 0; k < M; ++k) {
            cuda.axpy(alphas[k], basis_ptrs[k], d_y_seq.get(), n);
        }

        // Compare element-by-element on the host.
        std::vector<Complex> h_y_batched(n), h_y_seq(n);
        cuda.copy_to_host(d_y_batched.get(), h_y_batched.data(), n);
        cuda.copy_to_host(d_y_seq.get(),     h_y_seq.data(),     n);
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(std::abs(h_y_batched[i] - h_y_seq[i])
                    < 1e-12 * (1.0 + std::abs(h_y_seq[i])));
        }
    }
}

TEST_CASE("matvec::CudaBackend fused axpby (cublasZgeam) matches scale+axpy",
          "[cuda-backend][batched-primitives][phase1]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    // Phase 1 gap-fill regression: axpby is now a single cublasZgeam.
    // Pin element-wise agreement against the previous two-launch
    // (scale + axpy) decomposition, which is exactly the abstract
    // semantics the Backend interface promises.
    ed::matvec::CudaBackend cuda;
    constexpr std::size_t n = 4096;

    std::mt19937_64 rng(0x42424242ULL);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<Complex> h_x(n), h_y(n);
    for (auto& z : h_x) z = Complex(uni(rng), uni(rng));
    for (auto& z : h_y) z = Complex(uni(rng), uni(rng));

    auto d_x = cuda.make_zero_vector(n);
    auto d_y_fused = cuda.make_zero_vector(n);
    auto d_y_ref   = cuda.make_zero_vector(n);
    cuda.copy_from_host(h_x.data(), d_x.get(), n);
    cuda.copy_from_host(h_y.data(), d_y_fused.get(), n);
    cuda.copy_from_host(h_y.data(), d_y_ref.get(),   n);

    const Complex alpha(0.7, -0.3);
    const Complex beta (1.4,  0.2);

    cuda.axpby(alpha, d_x.get(), beta, d_y_fused.get(), n);

    // Reference: scale + axpy.
    cuda.scale(beta, d_y_ref.get(), n);
    cuda.axpy(alpha, d_x.get(), d_y_ref.get(), n);

    std::vector<Complex> h_y_fused(n), h_y_ref(n);
    cuda.copy_to_host(d_y_fused.get(), h_y_fused.data(), n);
    cuda.copy_to_host(d_y_ref.get(),   h_y_ref.data(),   n);
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(std::abs(h_y_fused[i] - h_y_ref[i])
                < 1e-12 * (1.0 + std::abs(h_y_ref[i])));
    }
}

TEST_CASE("matvec::CudaBackend pool-backed allocator survives a churn loop",
          "[cuda-backend][allocator][phase1]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    // Phase 1 gap-fill smoke test: the allocator is now backed by
    // `cudaMallocAsync` / `cudaFreeAsync` on the default device pool.
    // A long churn loop must not exhaust device memory (the pool
    // reuses returned allocations) and must remain functionally
    // correct (each fresh allocation reads/writes the values it was
    // asked to).
    ed::matvec::CudaBackend cuda;
    constexpr std::size_t n = 1024;
    constexpr int iterations = 200;

    std::mt19937_64 rng(0xC0DECAFEULL);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for (int it = 0; it < iterations; ++it) {
        // Random size in [n/4, n] to actually exercise the pool's
        // bucket logic (same-size allocations would be trivial).
        const std::size_t this_n = (n / 4) + (rng() % (n - n / 4));
        std::vector<Complex> h_x(this_n);
        for (auto& z : h_x) z = Complex(uni(rng), uni(rng));

        auto d_x = cuda.make_zero_vector(this_n);
        cuda.copy_from_host(h_x.data(), d_x.get(), this_n);

        // The nrm2 must match host-side ||x|| within roundoff. If the
        // pool ever returns a stale page with garbage in it, this
        // assertion would fire (it would also fire if cudaMallocAsync
        // silently dropped to a synchronous code path that didn't
        // initialise -- we don't assume initialisation, hence the
        // copy_from_host before reading).
        const double nrm = cuda.nrm2(d_x.get(), this_n);
        double ref_sq = 0.0;
        for (const auto& z : h_x) ref_sq += std::norm(z);
        REQUIRE(std::abs(nrm - std::sqrt(ref_sq))
                < 1e-10 * (1.0 + nrm));
        // d_x goes out of scope; UniqueVec frees via cudaFreeAsync,
        // returning the allocation to the pool for the next iter.
    }
}

TEST_CASE("lanczos_kernel<CudaBackend> `aux_ortho_ptrs` projects out the "
          "ground state through cuBLAS",
          "[cuda-backend][lanczos-kernel][aux_ortho]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int    N   = 6;
    constexpr std::size_t dim = std::size_t{1} << N;

    // ---- Reference: dense E_0 / E_1 from CPU host-space solve --------------
    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto ref    = ed_tests::reference_from_operator(*cpu_op, dim);
    REQUIRE(ref.eigs.size() >= 2);

    // ---- Pass 1: build the Krylov basis with CudaBackend, reconstruct y_0 --
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    ed::matvec::CudaBackend cuda;

    // v0_a: a random unit vector. The canonical basis vector |000…0⟩ lives
    // in a 1-D Sz sector for the all-down state on Heisenberg PBC, so H
    // returns a multiple of v0 and the Lanczos run breaks down after one
    // iteration. A random vector spans every Sz sector and gives the
    // kernel a full Krylov subspace to grow.
    auto v0_a_host = ed_tests::random_unit_vector(dim, /*seed=*/0xA110CAU);
    auto d_v0_a    = cuda.make_zero_vector(dim);
    cuda.copy_from_host(v0_a_host.data(), d_v0_a.get(), dim);

    auto gpu_matvec = [&](const Complex* in, Complex* out, std::size_t n) {
        gpu_op->matVecGPU(
            reinterpret_cast<const cuDoubleComplex*>(in),
            reinterpret_cast<cuDoubleComplex*>(out),
            static_cast<int>(n));
    };

    ed::krylov::LanczosKernelOptions opts_a;
    opts_a.max_iter   = 30;
    opts_a.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
    opts_a.keep_basis = true;
    auto R_a = ed::krylov::lanczos_kernel(cuda, gpu_matvec, dim, d_v0_a.get(), opts_a);
    const std::size_t M_a = R_a.alpha.size();
    REQUIRE(M_a >= 5);
    REQUIRE(R_a.basis.size() == M_a);

    // Diagonalise the tridiagonal on host (Eigen) and pin the lowest
    // Ritz value against the dense reference.
    Eigen::MatrixXd T_a = Eigen::MatrixXd::Zero(M_a, M_a);
    for (std::size_t i = 0; i < M_a; ++i) {
        T_a(i, i) = R_a.alpha[i];
        if (i + 1 < M_a) {
            T_a(i, i + 1) = R_a.beta[i + 1];
            T_a(i + 1, i) = R_a.beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_a(T_a);
    REQUIRE(es_a.info() == Eigen::Success);
    INFO("CUDA pass-1: E_0_lanczos=" << es_a.eigenvalues()(0)
         << "  E_0_dense=" << ref.eigs[0]);
    REQUIRE(std::abs(es_a.eigenvalues()(0) - ref.eigs[0]) < 1e-8);

    // Reconstruct y_0 on device via cuBLAS axpy_many:
    //   y_0 := sum_j S(j, 0) * V_j
    // Use the backend's batched primitive — exactly what the kernel
    // uses for its CGS2 axpy_many, just driven from outside.
    auto d_y0 = cuda.make_zero_vector(dim);
    std::vector<Complex>        coeffs(M_a);
    std::vector<const Complex*> basis_ptrs(M_a);
    for (std::size_t j = 0; j < M_a; ++j) {
        coeffs[j]     = Complex(es_a.eigenvectors()(static_cast<int>(j), 0), 0.0);
        basis_ptrs[j] = R_a.basis[j].get();
    }
    cuda.axpy_many(coeffs.data(),
                   basis_ptrs.data(), M_a,
                   d_y0.get(), dim);

    // Sanity: ||y_0|| ~ 1 on device.
    REQUIRE(std::abs(cuda.nrm2(d_y0.get(), dim) - 1.0) < 1e-8);

    // ---- Pass 2: re-run with aux_ortho_ptrs = { y_0 }, pre-project v0_b ----
    // v0_b: a different RANDOM seed, not orthogonal to y_0 a priori. We
    // mirror the kernel's documented contract: the CALLER pre-projects
    // v0 against the aux set before handing it to the kernel.
    auto v0_b_host = ed_tests::random_unit_vector(dim, /*seed=*/0xB055AU);
    auto d_v0_b    = cuda.make_zero_vector(dim);
    cuda.copy_from_host(v0_b_host.data(), d_v0_b.get(), dim);

    // Caller-side CGS2 pre-projection of v0_b against y_0 (device-side).
    for (int pass = 0; pass < 2; ++pass) {
        const Complex c = cuda.dot(d_y0.get(), d_v0_b.get(), dim);
        cuda.axpy(-c, d_y0.get(), d_v0_b.get(), dim);
    }
    const double v0b_norm = cuda.nrm2(d_v0_b.get(), dim);
    REQUIRE(v0b_norm > 1e-6);
    cuda.scale(Complex(1.0 / v0b_norm, 0.0), d_v0_b.get(), dim);

    ed::krylov::LanczosKernelOptions opts_b;
    opts_b.max_iter        = 30;
    opts_b.reorth          = ed::krylov::ReorthPolicy::FullCGS2;
    opts_b.keep_basis      = true;
    opts_b.aux_ortho_ptrs  = { d_y0.get() };

    auto R_b = ed::krylov::lanczos_kernel(cuda, gpu_matvec, dim,
                                          d_v0_b.get(), opts_b);
    const std::size_t M_b = R_b.alpha.size();
    REQUIRE(M_b >= 5);

    // Basis-to-y_0 orthogonality on device. cuBLAS dot accumulation is
    // looser than CPU dpdot under certain reduction trees; 1e-10 still
    // catches "the projection never happened" by a wide margin.
    double max_overlap = 0.0;
    for (std::size_t j = 0; j < M_b; ++j) {
        const Complex c = cuda.dot(d_y0.get(), R_b.basis[j].get(), dim);
        max_overlap = std::max(max_overlap, std::abs(c));
    }
    INFO("CUDA: max |<y_0, V_j>| after aux_ortho_ptrs = " << max_overlap);
    REQUIRE(max_overlap < 1e-10);

    // Deflated tridiagonal: smallest Ritz value should be E_1, not E_0.
    Eigen::MatrixXd T_b = Eigen::MatrixXd::Zero(M_b, M_b);
    for (std::size_t i = 0; i < M_b; ++i) {
        T_b(i, i) = R_b.alpha[i];
        if (i + 1 < M_b) {
            T_b(i, i + 1) = R_b.beta[i + 1];
            T_b(i + 1, i) = R_b.beta[i + 1];
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_b(T_b);
    REQUIRE(es_b.info() == Eigen::Success);
    INFO("CUDA deflated: E_0=" << es_b.eigenvalues()(0)
         << "  E_1 (dense)=" << ref.eigs[1]
         << "  E_0 (dense)=" << ref.eigs[0]);
    REQUIRE(std::abs(es_b.eigenvalues()(0) - ref.eigs[1]) < 1e-7);
    // The smallest Ritz value MUST NOT be the ground state.
    REQUIRE(std::abs(es_b.eigenvalues()(0) - ref.eigs[0]) >
            std::abs(ref.eigs[1] - ref.eigs[0]) - 1e-8);
}

TEST_CASE("thermal::kpm_dos_kernel<CudaBackend> matches CpuBackend on N=6 chain",
          "[cuda-backend][kpm-dos][phase-e1]") {
    // Phase E1 of the "Backend x Symmetries x Workflows" plan
    // (May 2026): pin that the new ``kpm_dos_kernel<CudaBackend>``
    // specialisation (which wraps the GPU Chebyshev/Hutchinson
    // driver in ``compute_kpm_dos_gpu_with_matvec``) produces the
    // same Z/E/Cv/S grid as the CpuBackend specialisation when both
    // see the same Hamiltonian and the same random seed.
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int          N   = 6;
    constexpr std::size_t  dim = std::size_t{1} << N;

    // ---- Build Hamiltonians on both lanes ----
    auto cpu_H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto gpu_H = build_gpu_heisenberg_chain(N, /*periodic=*/true);

    ed::matvec::CpuBackend  cpu;
    ed::matvec::CudaBackend cuda;

    ed::thermal::KpmDosOptions opts;
    opts.num_moments        = 256;
    opts.num_random_vectors = 8;
    opts.betas              = {0.25, 0.5, 1.0, 2.0};
    opts.random_seed        = 7777;
    // Lock the spectral window by hand so both lanes use the same
    // rescaling. Exact bounds for the 6-site periodic Heisenberg
    // ring at J=1 are E_min ≈ -2.802 and E_max ≈ +1.5; we pad to
    // be safe. With the window pinned the only remaining variance
    // is the Hutchinson sampling -- enough to put the two lanes in
    // numerical agreement at R=8.
    opts.e_min_override = -3.5;
    opts.e_max_override =  2.0;

    auto cpu_res = ed::thermal::kpm_dos_kernel(
        cpu,
        [&](const Complex* in, Complex* out, std::size_t n) {
            cpu_H->apply(in, out, n);
        },
        dim, static_cast<std::uint64_t>(dim), opts);

    auto cuda_res = ed::thermal::kpm_dos_kernel(
        cuda,
        [&](const Complex* in, Complex* out, std::size_t n) {
            gpu_H->matVecGPU(
                reinterpret_cast<const cuDoubleComplex*>(in),
                reinterpret_cast<cuDoubleComplex*>(out),
                static_cast<int>(n));
        },
        dim, static_cast<std::uint64_t>(dim), opts);

    REQUIRE(cpu_res.energy.size()  == opts.betas.size());
    REQUIRE(cuda_res.energy.size() == opts.betas.size());
    REQUIRE(cpu_res.specific_heat.size()  == opts.betas.size());
    REQUIRE(cuda_res.specific_heat.size() == opts.betas.size());

    // Both lanes use the *same* spectral window (we passed an
    // explicit override above) -- so any remaining discrepancy is
    // purely Hutchinson sampling noise from the two independent
    // RNGs (host mt19937 vs cuRAND).
    REQUIRE(std::abs(cpu_res.e_min_estimate  - (-3.5)) < 1e-12);
    REQUIRE(std::abs(cpu_res.e_max_estimate  -  2.0 ) < 1e-12);
    REQUIRE(std::abs(cuda_res.e_min_estimate - (-3.5)) < 1e-12);
    REQUIRE(std::abs(cuda_res.e_max_estimate -  2.0 ) < 1e-12);

    // At R=8 on D=64 the Monte-Carlo error on the energy is
    // O(1/sqrt(R*D)) ≈ 0.04, which dominates the kernel-smoothing
    // bias. The energy must lie inside the spectrum (else the
    // moment computation is broken) and be within ~0.5 of the CPU
    // reference (~10% of the bandwidth).
    for (std::size_t t = 0; t < opts.betas.size(); ++t) {
        INFO("beta=" << opts.betas[t]
             << "  E_cpu=" << cpu_res.energy[t]
             << "  E_gpu=" << cuda_res.energy[t]
             << "  Cv_cpu=" << cpu_res.specific_heat[t]
             << "  Cv_gpu=" << cuda_res.specific_heat[t]);

        // Energies inside the spectrum.
        REQUIRE(cpu_res.energy[t]  >= -3.5);
        REQUIRE(cpu_res.energy[t]  <=  2.0);
        REQUIRE(cuda_res.energy[t] >= -3.5);
        REQUIRE(cuda_res.energy[t] <=  2.0);

        // CPU vs CUDA agreement within Hutchinson noise at R=8.
        REQUIRE(std::abs(cpu_res.energy[t] - cuda_res.energy[t]) < 0.6);
    }
}

// REMOVED: "thermal::ltlm_kernel<CudaBackend> matches CpuBackend".
// ltlm_kernel (and ed/thermal/ltlm_kernel.h) were deleted in consolidation
// Family 1 (Jul 2026). This case pinned CPU/GPU AGREEMENT between the two LTLM
// lanes -- and they did agree, because both reimplemented the same
// GS-local-DOS bug (each summed |<0|psi_n>|^2 e^{-bE_n} instead of the thermal
// trace). A parity test between twins that share a defect cannot see it; this
// one passed for months while LTLM returned E0 at every temperature. The
// replacement pin compares LTLM against an INDEPENDENT reference rather than
// its own twin: test_thermal_dense_ref's "LTLM thermodynamics IS the FTLM
// trace" (identical knobs, 1e-12) plus its dense-reference cells.


TEST_CASE("gpu::lanczos_kernel facade is seed-reproducible",
          "[cuda-backend][facade]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    constexpr int N = 6;
    constexpr std::size_t dim = std::size_t{1} << N;

    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);

    auto run_with_seed = [&](unsigned long long s) {
        std::vector<double> e;
        ed::matvec::gpu::run_lanczos_eigenvalues_kernel_facade(
            *gpu_op, static_cast<int>(dim), 40, 3, 1e-12, s, e);
        return e;
    };

    auto a = run_with_seed(42ULL);
    auto b = run_with_seed(42ULL);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        // Identical seed → identical v0 → identical Krylov sequence →
        // bit-equal eigenvalues (modulo non-deterministic FP reductions
        // in cuBLAS, which are below 1e-12 here).
        REQUIRE(std::abs(a[i] - b[i]) < 1e-12);
    }
}

#else   // !WITH_CUDA

TEST_CASE("matvec::CudaBackend test placeholder (CUDA disabled)",
          "[cuda-backend][phase2]") {
    SUCCEED("CUDA support not compiled");
}

#endif  // WITH_CUDA
