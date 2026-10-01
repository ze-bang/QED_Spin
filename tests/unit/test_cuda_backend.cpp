// =============================================================================
// tests/unit/test_cuda_backend.cpp
//
// Pins agreement between
// `lanczos_kernel<CpuBackend>` and `lanczos_kernel<CudaBackend>` on a
// small Heisenberg chain. Two independent lanes:
//
//   1. Backend BLAS-1 round-trips (alloc + H2D + axpy + dot + nrm2 + D2H).
//   2. `lanczos_kernel(...)` ground-state energy with both backends,
//      driving the same matvec callable through cuBLAS on the device
//      side. Demonstrates that the kernel is *actually* backend-
//      agnostic: the algorithm body in `ed/krylov/lanczos.h`
//      runs unchanged against the CUDA implementation of `Backend`.
//
// Runtime SKIPs (Catch2 SUCCEED + return) keep the build-only CUDA lane
// happy on CI hosts without an attached GPU.
//
// The device matvec in both Lanczos cases stages each vector through the
// host Operator (staged_device_matvec).
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#ifdef WITH_CUDA

#include <ed/matvec/cpu_backend.h>
#include <ed/gpu/cuda_backend.cuh>
#include <ed/krylov/lanczos.h>

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

// Device-pointer matvec for the Lanczos-kernel tests: stage each vector
// through the host Operator (the kernel under test is the CUDA Backend, not
// a device SpMV).
std::function<void(const Complex*, Complex*, std::size_t)>
staged_device_matvec(ed::matvec::CudaBackend& be, const Operator& H) {
    return [&be, &H](const Complex* in, Complex* out, std::size_t n) {
        std::vector<Complex> hin(n), hout(n);
        be.copy_to_host(in, hin.data(), n);
        H.apply(hin.data(), hout.data(), n);
        be.copy_from_host(hout.data(), out, n);
    };
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
    ed::matvec::CudaBackend cuda;
    auto gpu_H = staged_device_matvec(cuda, *cpu_H);

    auto d_v0 = cuda.make_zero_vector(dim);
    cuda.copy_from_host(v0.data(), d_v0.get(), dim);

    auto cuda_res = ed::krylov::lanczos_kernel(
        cuda,
        [&](const Complex* in, Complex* out, std::size_t n) {
            gpu_H(in, out, n);
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

TEST_CASE("matvec::CudaBackend batched dot_many matches the sequential reference",
          "[cuda-backend][batched-primitives][phase1]") {
    if (!gpu_available()) { SUCCEED("no CUDA device available, skipping"); return; }

    // `dot_many` is overridden via one
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

    // `axpy_many` is overridden via one
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

    // axpby is a single cublasZgeam. Pin element-wise agreement against
    // the two-launch (scale + axpy) decomposition, which is exactly the
    // abstract semantics the Backend interface promises.
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

    // The allocator is backed by
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
    ed::matvec::CudaBackend cuda;
    auto gpu_op = staged_device_matvec(cuda, *cpu_op);

    // v0_a: a random unit vector. The canonical basis vector |000…0⟩ lives
    // in a 1-D Sz sector for the all-down state on Heisenberg PBC, so H
    // returns a multiple of v0 and the Lanczos run breaks down after one
    // iteration. A random vector spans every Sz sector and gives the
    // kernel a full Krylov subspace to grow.
    auto v0_a_host = ed_tests::random_unit_vector(dim, /*seed=*/0xA110CAU);
    auto d_v0_a    = cuda.make_zero_vector(dim);
    cuda.copy_from_host(v0_a_host.data(), d_v0_a.get(), dim);

    auto gpu_matvec = [&](const Complex* in, Complex* out, std::size_t n) {
        gpu_op(in, out, n);
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

#else   // !WITH_CUDA

TEST_CASE("matvec::CudaBackend test placeholder (CUDA disabled)",
          "[cuda-backend][phase2]") {
    SUCCEED("CUDA support not compiled");
}

#endif  // WITH_CUDA
