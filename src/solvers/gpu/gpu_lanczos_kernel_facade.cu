// =============================================================================
// src/solvers/gpu/gpu_lanczos_kernel_facade.cu
//
// First production migration onto the unified
// `ed::krylov::lanczos_kernel<Backend>` template (May 2026).
//
// Replaces the eigenvalues-only AND the eigenpair paths of
// `GPULanczos::run` (the 1099-LOC hand-rolled body in `gpu_lanczos.cu`)
// for the common case: full CGS2 reorthogonalisation, basis held
// entirely in device memory. Behind the scenes:
//
//   1. allocate v0 on the GPU (`CudaBackend::make_zero_vector`),
//   2. initialise v0 with a curand-based random vector (mirrors the
//      legacy GPULanczos init policy: real-only Gaussian per element),
//   3. drive `lanczos_kernel<Backend>` with a matvec callable that
//      forwards into `GPUOperator::matVecGPU(...)`,
//   4. solve the small real-symmetric tridiagonal on the host via
//      Eigen (`SelfAdjointEigenSolver`), and
//   5. either hand back the lowest `num_eigs` Ritz eigenvalues (the
//      "eigenvalues-only" entry point), or additionally reconstruct
//      the corresponding Ritz vectors by `num_eigs * M` backend
//      axpys on the retained Krylov basis and copy them to host
//      (the "eigenpairs" entry point).
//
// The legacy `GPULanczos::run` class remains the path for callers that
// need any of:
//
//   * a windowed reorthogonalisation regime (device-memory pressure),
//   * the on-disk basis spill,
//   * the early-eigenvalue-convergence early-exit.
//
// `runGPULanczos(...)` dispatches between the facade entry points and
// the legacy class on the combination of (`eigenvectors`, available
// device memory). All current callers (the orchestrator's GPU lane
// in ``ed::workflows::solve`` and the streaming-symmetry GPU kernels)
// go through `runGPULanczos`, so they pick up the unified-kernel
// path automatically.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backends/cuda_backend.cuh>
#include <ed/gpu/gpu_operator.cuh>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdlib>   // Waves 4.1+4.2: getenv (ED_GPU_LANCZOS_FULL_CGS2)
#include <iostream>
#include <stdexcept>
#include <vector>

namespace ed::matvec::gpu {

namespace {

using Complex = std::complex<double>;

// Curand-based random init, real-only per element, one curand state per
// thread seeded with (seed, tid). Matches the default behaviour of the
// legacy GPULanczos::initializeRandomVector (with complex_seed=0). The
// same starting vector is therefore generated for the same seed under
// either path, which keeps reproducibility intact.
__global__ void init_v0_curand_real(cuDoubleComplex* v,
                                    int N,
                                    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curandState st;
    curand_init(seed, idx, 0, &st);
    const double re = curand_normal_double(&st);
    v[idx] = make_cuDoubleComplex(re, 0.0);
}

// Build the real-symmetric tridiagonal T (size M x M) from the kernel
// output `(alpha, beta)` -- alpha[i] = T[i,i], beta[i+1] = T[i, i+1],
// beta[0] is the sentinel zero -- and diagonalise it on the host via
// Eigen's SelfAdjointEigenSolver. The eigensolver runs once; both
// the eigenvalues-only and the eigenpair entry points reuse it.
struct TridiagSolveResult {
    Eigen::VectorXd values;    // size M, ascending
    Eigen::MatrixXd vectors;   // M x M, columns are T's eigenvectors
};

TridiagSolveResult solve_tridiag(const std::vector<double>& alpha,
                                 const std::vector<double>& beta)
{
    TridiagSolveResult out;
    const int M = static_cast<int>(alpha.size());
    if (M == 0) {
        out.values  = Eigen::VectorXd();
        out.vectors = Eigen::MatrixXd();
        return out;
    }
    if (M == 1) {
        out.values  = Eigen::VectorXd::Constant(1, alpha[0]);
        out.vectors = Eigen::MatrixXd::Constant(1, 1, 1.0);
        return out;
    }

    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(M, M);
    for (int i = 0; i < M; ++i) T(i, i) = alpha[i];
    for (int i = 1; i < M; ++i) {
        T(i, i - 1) = beta[i];
        T(i - 1, i) = beta[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "gpu_lanczos_kernel_facade: tridiagonal diagonalisation failed");
    }
    out.values  = es.eigenvalues();
    out.vectors = es.eigenvectors();
    return out;
}

// Run the kernel from a fresh curand-initialised v0 on the GPU. Common
// preamble shared by both entry points.
struct FacadeKernelRun {
    ed::matvec::CudaBackend       backend;
    ed::krylov::LanczosKernelResult result;
    std::chrono::steady_clock::time_point t0;
};

void init_random_v0(ed::matvec::CudaBackend& cuda,
                    Complex* d_v0,
                    int N,
                    unsigned long long seed)
{
    const unsigned long long actual_seed = (seed != 0ULL) ? seed : 42ULL;
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>(
        (static_cast<std::size_t>(N) + BLOCK - 1) / BLOCK);
    init_v0_curand_real<<<grid, BLOCK>>>(
        reinterpret_cast<cuDoubleComplex*>(d_v0), N, actual_seed);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("gpu_lanczos_kernel_facade: v0 init kernel "
                        "launch failed: ") + cudaGetErrorString(err));
    }
    (void)cuda;  // The backend's stream guarantees ordering with the
                 // subsequent cuBLAS nrm2 call (HOST pointer mode).
}

void validate_inputs(int N, int max_iter, int num_eigs)
{
    if (N <= 0)
        throw std::invalid_argument(
            "gpu_lanczos_kernel_facade: N must be > 0");
    if (max_iter <= 0)
        throw std::invalid_argument(
            "gpu_lanczos_kernel_facade: max_iter must be > 0");
    if (num_eigs <= 0)
        throw std::invalid_argument(
            "gpu_lanczos_kernel_facade: num_eigs must be > 0");
}

}  // namespace

// Shared kernel-driver: validates inputs, allocates v0, seeds curand,
// runs `lanczos_kernel<CudaBackend>`, then leaves the tridiag solve and
// any Ritz reconstruction to the caller. Returns the backend instance
// (caller keeps it alive while it touches `kres.basis`) and the kernel
// result. The unique_ptr is returned by value so the device-side v0
// lifetime tracks the kernel's transient internals only.
namespace {

struct KernelOutput {
    ed::matvec::CudaBackend          backend;
    ed::krylov::LanczosKernelResult  kres;
    double                           wall_s = 0.0;
};

KernelOutput run_facade_kernel(GPUOperator& gpu_op,
                               int N, int max_iter,
                               double tol,
                               unsigned long long seed,
                               bool keep_basis = false)
{
    KernelOutput out;
    const auto t0  = std::chrono::high_resolution_clock::now();
    const std::size_t dim = static_cast<std::size_t>(N);

    auto d_v0 = out.backend.make_zero_vector(dim);
    init_random_v0(out.backend, d_v0.get(), N, seed);

    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter   = static_cast<std::size_t>(max_iter);

    // Waves 4.1 & 4.2 of the SOTA Performance rollout (May 2026):
    // for the eigenvalues-only path the GPU facade now defaults to
    //
    //   keep_basis = false  (Wave 4.1) -- skip storing
    //                       m * N * 16 B of device-side basis vectors;
    //                       eigvecs/CF spectral callers pass true.
    //   LocalDGKS3 K=1      (Wave 4.2) -- 30-50% fewer cuBLAS BLAS-1
    //                       calls vs FullCGS2 at large krylov_dim.
    //                       Env opt-in ``ED_GPU_LANCZOS_FULL_CGS2=1``
    //                       restores the pre-Wave defaults for
    //                       near-degenerate spectra.
    const bool force_cgs2 = []() {
        const char* env = std::getenv("ED_GPU_LANCZOS_FULL_CGS2");
        return env && env[0] == '1';
    }();
    if (force_cgs2 || keep_basis) {
        opts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        opts.keep_basis = true;
    } else {
        opts.reorth         = ed::krylov::ReorthPolicy::LocalDGKS3;
        opts.local_ring_size = 1;
        opts.keep_basis     = false;
    }
    (void)tol;  // GPU Lanczos converges to the full Krylov dim; Ritz-tol
                // early-exit goes through `opts.convergence_check` when
                // we want it (none of the current facade callers do).

    out.kres = ed::krylov::lanczos_kernel(
        out.backend,
        [&](const Complex* in, Complex* o, std::size_t n) {
            gpu_op.matVecGPU(
                reinterpret_cast<const cuDoubleComplex*>(in),
                reinterpret_cast<cuDoubleComplex*>(o),
                static_cast<int>(n));
        },
        dim,
        d_v0.get(),
        opts);

    const auto t1 = std::chrono::high_resolution_clock::now();
    out.wall_s = std::chrono::duration<double>(t1 - t0).count();
    return out;
}

}  // namespace

// Public entry point (forward-declared in gpu_solvers.h).
void run_lanczos_eigenvalues_kernel_facade(
    GPUOperator& gpu_op,
    int N,
    int max_iter,
    int num_eigs,
    double tol,
    unsigned long long seed,
    std::vector<double>& eigenvalues_out)
{
    validate_inputs(N, max_iter, num_eigs);

    auto ko = run_facade_kernel(gpu_op, N, max_iter, tol, seed);
    auto ts = solve_tridiag(ko.kres.alpha, ko.kres.beta);

    const std::size_t k = std::min<std::size_t>(
        static_cast<std::size_t>(num_eigs),
        static_cast<std::size_t>(ts.values.size()));
    eigenvalues_out.assign(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) eigenvalues_out[i] = ts.values(i);

    std::cout << "\nGPU Lanczos (kernel facade: lanczos_kernel<CudaBackend>):\n"
              << "  Total time  : " << ko.wall_s << " s\n"
              << "  Iterations  : " << ko.kres.iters_done << "\n"
              << "  Krylov basis: " << ko.kres.basis.size()
              << " vectors held\n"
              << "  Eigenvalues : returning " << k << " of "
              << ts.values.size() << " Ritz values\n";
}

// Eigenpair entry point. Returns eigenvalues + Ritz vectors (in host
// memory). Implementation strategy: after the unified-kernel run, T's
// diagonalisation gives an M x M eigenvector matrix S. For each
// requested Ritz index i, reconstruct
//
//     y_i = sum_{k=0}^{M-1} S(k, i) * V_k
//
// by a sequence of `Backend::axpy` calls into a device scratch buffer
// (cuBLAS zaxpy on the GPU), then copy the result to host. Mirrors
// `GPULanczos::computeRitzVectors` exactly, only routed through the
// unified Backend interface.
void run_lanczos_eigenpairs_kernel_facade(
    GPUOperator& gpu_op,
    int N,
    int max_iter,
    int num_eigs,
    double tol,
    unsigned long long seed,
    std::vector<double>& eigenvalues_out,
    std::vector<std::vector<std::complex<double>>>& eigenvectors_out)
{
    validate_inputs(N, max_iter, num_eigs);

    // Eigenpair path needs the kept basis for Ritz vector reconstruction.
    auto ko = run_facade_kernel(gpu_op, N, max_iter, tol, seed,
                                /*keep_basis=*/true);
    auto ts = solve_tridiag(ko.kres.alpha, ko.kres.beta);

    const std::size_t dim   = static_cast<std::size_t>(N);
    const std::size_t M     = static_cast<std::size_t>(ts.values.size());
    const std::size_t k     = std::min<std::size_t>(
        static_cast<std::size_t>(num_eigs), M);
    const std::size_t basis = ko.kres.basis.size();

    if (k > 0 && basis < M) {
        throw std::runtime_error(
            "run_lanczos_eigenpairs_kernel_facade: kernel retained " +
            std::to_string(basis) + " basis vectors but the tridiag is " +
            std::to_string(M) + " x " + std::to_string(M) +
            "; cannot reconstruct Ritz vectors. The unified kernel only "
            "drops a basis vector when an invariant subspace is detected "
            "(beta_{j+1} below breakdown_tol); see lanczos_kernel.h.");
    }

    eigenvalues_out.assign(k, 0.0);
    for (std::size_t i = 0; i < k; ++i) eigenvalues_out[i] = ts.values(i);

    eigenvectors_out.assign(k, {});

    if (k == 0) {
        std::cout << "\nGPU Lanczos eigenpairs (kernel facade): no Ritz "
                     "values requested.\n";
        return;
    }

    // Device scratch + host buffer for the D2H copy of each Ritz vector.
    auto scratch = ko.backend.make_zero_vector(dim);
    std::vector<Complex> host_buf(dim);

    for (std::size_t i = 0; i < k; ++i) {
        ko.backend.fill_zero(scratch.get(), dim);
        for (std::size_t j = 0; j < M; ++j) {
            const double sij = ts.vectors(static_cast<int>(j),
                                          static_cast<int>(i));
            if (sij == 0.0) continue;
            ko.backend.axpy(Complex(sij, 0.0),
                            ko.kres.basis[j].get(),
                            scratch.get(),
                            dim);
        }
        ko.backend.copy_to_host(scratch.get(), host_buf.data(), dim);
        eigenvectors_out[i].assign(host_buf.begin(), host_buf.end());
    }

    std::cout << "\nGPU Lanczos eigenpairs (kernel facade: "
                 "lanczos_kernel<CudaBackend> + Ritz recon):\n"
              << "  Total time  : " << ko.wall_s << " s\n"
              << "  Iterations  : " << ko.kres.iters_done << "\n"
              << "  Krylov basis: " << basis << " vectors held\n"
              << "  Eigenpairs  : returning " << k << " of "
              << M << " Ritz pairs (eigvec recon: " << k * M
              << " axpys + " << k << " D2H copies)\n";
}

// ---------------------------------------------------------------------------
// FTLM-shaped facade. Same kernel, same backend, but the caller supplies
// the starting vector (already on device, already normalised) and optionally
// receives the orthonormal basis as a heap-owned `cuDoubleComplex**` array.
//
// Three behavioural knobs lift directly from the FTLM solver knobs:
//
//   * `full_reorth=true`             -> FullCGS2
//   * `full_reorth=false, freq > 0`  -> PeriodicCGS2 every `freq` iter
//   * `full_reorth=false, freq == 0` -> no reorth (pure three-term)
//
// Reorthogonalisation is the kernel's job (batched CGS2 via
// `CudaBackend::dot_many`/`axpy_many`). The legacy bodies in
// `gpu_ftlm.cu` did this with M sequential `cublasZdotc` calls; the
// kernel collapses that to one `cublasZgemv` per CGS2 pass via the
// new Phase-1 overrides.
//
// Ownership note: the kernel allocates basis vectors via
// `CudaBackend::allocate` which is now pool-backed
// (`cudaMallocAsync`). When `d_basis_out != nullptr` we release()
// the kernel's UniqueVec wrappers into a heap-allocated raw-pointer
// array so the caller can keep them past the function return; the
// caller MUST free with `cudaFreeAsync(ptr, 0)` (and `delete[]`)
// per the pool's contract. When `d_basis_out == nullptr`, the
// basis is freed by the kernel's RAII destructors at scope exit.
// ---------------------------------------------------------------------------
int run_ftlm_lanczos_kernel_facade(
    GPUOperator& gpu_op,
    const void*  d_start_vec_raw,
    int          N,
    int          krylov_dim,
    bool         full_reorth,
    int          reorth_freq,
    double       tol,
    std::vector<double>& alpha_out,
    std::vector<double>& beta_out,
    void**       d_basis_out)
{
    if (N <= 0) {
        throw std::invalid_argument("run_ftlm_lanczos_kernel_facade: N must be > 0");
    }
    if (krylov_dim <= 0) {
        throw std::invalid_argument(
            "run_ftlm_lanczos_kernel_facade: krylov_dim must be > 0");
    }
    if (d_start_vec_raw == nullptr) {
        throw std::invalid_argument(
            "run_ftlm_lanczos_kernel_facade: d_start_vec must be non-null");
    }

    alpha_out.clear();
    beta_out.clear();

    const std::size_t dim = static_cast<std::size_t>(N);
    const auto* d_start = reinterpret_cast<const Complex*>(d_start_vec_raw);

    ed::matvec::CudaBackend backend;

    // Caller hands us an already-normalised starting vector. The kernel
    // re-normalises internally (defensive), so the only thing we do here
    // is allocate a fresh device buffer to hand to the kernel and copy
    // the caller's content into it. This decouples the caller's
    // device buffer from the kernel's working set.
    auto d_v0 = backend.make_zero_vector(dim);
    backend.copy(d_start, d_v0.get(), dim);

    ed::krylov::LanczosKernelOptions opts;
    opts.max_iter   = static_cast<std::size_t>(krylov_dim);
    opts.keep_basis = true;
    opts.breakdown_tol = tol;
    if (full_reorth) {
        opts.reorth = ed::krylov::ReorthPolicy::FullCGS2;
    } else if (reorth_freq > 0) {
        opts.reorth      = ed::krylov::ReorthPolicy::PeriodicCGS2;
        opts.reorth_freq = static_cast<std::size_t>(reorth_freq);
    } else {
        opts.reorth = ed::krylov::ReorthPolicy::None;
    }

    auto kres = ed::krylov::lanczos_kernel(
        backend,
        [&](const Complex* in, Complex* o, std::size_t n) {
            gpu_op.matVecGPU(
                reinterpret_cast<const cuDoubleComplex*>(in),
                reinterpret_cast<cuDoubleComplex*>(o),
                static_cast<int>(n));
        },
        dim,
        d_v0.get(),
        opts);

    alpha_out = std::move(kres.alpha);
    beta_out  = std::move(kres.beta);

    if (d_basis_out != nullptr) {
        const std::size_t M = kres.basis.size();
        auto** out = new cuDoubleComplex*[M];
        for (std::size_t k = 0; k < M; ++k) {
            // Steal the raw pointer from the kernel's RAII wrapper. The
            // wrapper's deleter (which would have called
            // CudaBackend::deallocate -> cudaFreeAsync) is now disengaged;
            // ownership transfers to the caller via *d_basis_out.
            Complex* raw = kres.basis[k].release();
            out[k] = reinterpret_cast<cuDoubleComplex*>(raw);
        }
        *d_basis_out = static_cast<void*>(out);
    }
    // If d_basis_out is null, kres.basis goes out of scope here and
    // every UniqueVec's deleter fires (cudaFreeAsync on the default
    // stream).

    return static_cast<int>(alpha_out.size());
}

}  // namespace ed::matvec::gpu

#endif  // WITH_CUDA
