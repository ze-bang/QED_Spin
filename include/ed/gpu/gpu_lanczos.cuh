#pragma once

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <vector>
#include <functional>
#include <complex>
#include <ed/gpu/gpu_operator.cuh>

/**
 * GPU-accelerated Lanczos algorithm for large-scale eigenvalue problems
 * Optimized for systems with up to 32 sites
 */
class GPULanczos {
public:
    GPULanczos(GPUOperator* op, int max_iter, double tolerance);
    ~GPULanczos();

    /// Set the starting-vector RNG seed. Matches the
    /// ``EDParameters::lanczos_seed`` field. A value of 0 keeps the
    /// historical deterministic seed (42) to preserve reproducibility
    /// of legacy GPU runs; pass a nonzero value to opt into a custom
    /// seed (e.g. when comparing CPU vs GPU runs that should agree to
    /// numerical precision -- audit STRUCTURAL_AUDIT.md S1 #21). The
    /// fully-random ``std::random_device`` policy used by the CPU
    /// Lanczos can be reproduced by passing ``std::random_device{}()``
    /// from the caller.
    void setSeed(unsigned long long seed) noexcept { user_seed_ = seed; }

    // Run Lanczos algorithm to find lowest eigenvalues
    void run(int num_eigenvalues, std::vector<double>& eigenvalues,
            std::vector<std::vector<std::complex<double>>>& eigenvectors,
            bool compute_vectors = false);
    
    // Get performance statistics
    struct Stats {
        double total_time;
        double matvec_time;
        double ortho_time;
        int iterations;
        double convergence_error;
        // full_reorth_count was retired in D-6: nothing in the orthogonalization
        // loop ever incremented it — the GPU Lanczos uses windowed/selective
        // reorth, not periodic full reorth — so the counter was always
        // zero and gave a misleading impression that full reorth was running.
        uint64_t selective_reorth_count;
        uint64_t total_reorth_ops;
    };
    
    Stats getStats() const { return stats_; }
    
private:
    GPUOperator* op_;
    int max_iter_;
    double tolerance_;
    int dimension_;
    
    // GPU memory
    cuDoubleComplex* d_v_current_;    // Current Lanczos vector
    cuDoubleComplex* d_v_prev_;       // Previous Lanczos vector
    cuDoubleComplex* d_w_;            // Work vector (H*v)
    cuDoubleComplex* d_temp_;         // Temporary vector
    
    // Starting-vector seed. 0 -> use the historical deterministic
    // seed (42) for back-compat. Set via ``setSeed`` (audit S1 #21).
    unsigned long long user_seed_ = 0;

    // Lanczos vectors stored on GPU (if memory allows)
    cuDoubleComplex** d_lanczos_vectors_;
    int num_stored_vectors_;

    // Reusable buffers for batched orthogonalization (avoid cudaMalloc/free per step)
    cuDoubleComplex** d_ortho_basis_ptrs_;  // device pointer array [num_stored_vectors_]
    cuDoubleComplex* d_ortho_overlaps_;      // overlap scratch [num_stored_vectors_]

    // -------------------------------------------------------------------------
    // Phase 8 #7: persistent device-side mirror of the *entire* ring buffer
    // pointer table.
    //
    // The pre-Phase-8 batched orthogonalize() rebuilt the windowed pointer
    // slice on the host on every iteration and `cudaMemcpy`'d it into
    // ``d_ortho_basis_ptrs_``. That is one synchronous H2D copy per Lanczos
    // step (default-stream, sized only ``num_check * 8`` bytes but blocking
    // because the destination is reused immediately by a kernel launch).
    //
    // ``d_ortho_basis_ptrs_full_`` mirrors the full
    // ``d_lanczos_vectors_[0..num_stored_vectors_-1]`` device-pointer
    // table once at allocate time. The pointers themselves never change
    // for the lifetime of GPULanczos (each ring-buffer slot owns a fixed
    // ``cudaMalloc``ed slab), so the table can stay resident on the
    // device for the whole run.
    //
    // Hot path savings: for iter <= num_stored_vectors_ (the common
    // "basis fits in GPU" case) we just pass
    // ``d_ortho_basis_ptrs_full_`` directly to the kernels -- *zero* H2D
    // traffic in the orthogonalize() critical section. Only the wrapped
    // (iter > num_stored_vectors_, windowed reorth) case still pays the
    // H2D cost, and that is the explicitly-degraded path users are
    // already warned about.
    // -------------------------------------------------------------------------
    cuDoubleComplex** d_ortho_basis_ptrs_full_;  // [num_stored_vectors_]

    // Reusable CUDA events for orthogonalize() timing (avoid create/destroy each call)
    cudaEvent_t ortho_timing_start_;
    cudaEvent_t ortho_timing_stop_;
    bool ortho_timing_events_created_;

    // -------------------------------------------------------------------------
    // Phase 8 #4: device-resident scalar buffers for the alpha = <v|H|v> step.
    //
    // Without these, ``vectorDot`` runs cuBLAS in HOST pointer mode, which
    // forces an implicit device->host sync on every Lanczos iteration:
    // cublasZdotc cannot return until the result has reached host memory.
    // The follow-up ``vectorAxpy(d_v_current, d_w, -alpha)`` then has to wait
    // for that sync before it can issue.
    //
    // With DEVICE pointer mode + a one-element device buffer for alpha, the
    // dot result lands on the device, the negation is one trivial kernel
    // launch, and the subsequent zaxpy can be queued back-to-back -- no
    // host sync at all on the alpha path. We still copy alpha to a pinned
    // host slot (cudaMemcpyAsync), but the first time we *read* it on the
    // host is after the orthogonalize() call has issued, so the latency
    // overlaps with downstream device work.
    //
    // beta = ||w|| stays in HOST mode because the convergence check that
    // immediately follows it would have to sync anyway.
    // -------------------------------------------------------------------------
    cuDoubleComplex* d_alpha_dev_;       // [1] device-side alpha
    cuDoubleComplex* d_neg_alpha_dev_;   // [1] device-side -alpha
    cuDoubleComplex* h_alpha_pinned_;    // [1] pinned host (async D2H target)

    // Tridiagonal matrix elements (on host)
    std::vector<double> alpha_;  // Diagonal
    std::vector<double> beta_;   // Off-diagonal
    
    // cuBLAS handle
    cublasHandle_t cublas_handle_;
    
    // Statistics
    Stats stats_;
    
    // Helper functions
    void allocateMemory();
    void freeMemory();
    void initializeRandomVector(cuDoubleComplex* d_vec, unsigned long long seed = 0);
    
    // Adaptive selective reorthogonalization (Parlett-Simon)
    void orthogonalize(cuDoubleComplex* d_vec, int iter,
                      std::vector<std::vector<double>>& omega,
                      const std::vector<double>& alpha,
                      const std::vector<double>& beta,
                      double ortho_threshold);
    
    void normalizeVector(cuDoubleComplex* d_vec);
    double vectorNorm(const cuDoubleComplex* d_vec);
    void vectorCopy(const cuDoubleComplex* src, cuDoubleComplex* dst);
    void vectorScale(cuDoubleComplex* d_vec, double scale);
    void vectorAxpy(const cuDoubleComplex* d_x, cuDoubleComplex* d_y,
                   const cuDoubleComplex& alpha);
    std::complex<double> vectorDot(const cuDoubleComplex* d_x,
                                   const cuDoubleComplex* d_y);
    
    // Tridiagonal solver
    void solveTridiagonal(int m, int num_eigs,
                         std::vector<double>& eigenvalues,
                         std::vector<std::vector<double>>& eigenvectors);
    
    // Ritz vector computation
    void computeRitzVectors(const std::vector<std::vector<double>>& tridiag_eigenvecs,
                           int num_vecs,
                           std::vector<std::vector<std::complex<double>>>& eigenvectors);
};

// The `GPUBlockLanczos` and `GPUKrylovSchur` solver classes were retired in
// operator-collapse Phase 2b (Jun 2026) together with their device bodies
// (`gpu_block_lanczos.cu`, `gpu_krylov_schur.cu`). Their only callers were the
// now-deleted `GPUEDWrapper::runGPUBlockLanczos[FixedSz]` /
// `runGPUKrylovSchur[FixedSz]` forwarders. Production many-eigenvalue GPU runs
// go through `GPULanczos` (above) and the unified `lanczos_kernel<CudaBackend>`
// facade. The block-specific kernel declarations in `GPULanczosKernels` below
// are kept (they cost nothing) but are no longer wired to a host class; to
// re-introduce a block / Schur GPU solver, prefer a Backend-templated kernel.

// Kernel declarations for Lanczos helpers
namespace GPULanczosKernels {

// complex_seed: 0 = real-only initialization (default, matches CPU lanczos
// convention), 1 = fully complex (legacy / ED_LANCZOS_COMPLEX_SEED=1).
__global__ void initRandomVectorKernel(cuDoubleComplex* vec, int N, unsigned long long seed,
                                       int complex_seed);

/**
 * @brief Batched dot product kernel for efficient orthogonalization
 * 
 * Computes multiple inner products in parallel using one block per vector.
 * More efficient than sequential cuBLAS calls for multiple vectors.
 */
__global__ void batchedDotProductKernel(const cuDoubleComplex* const* basis,
                                        const cuDoubleComplex* target,
                                        cuDoubleComplex* overlaps,
                                        int num_vecs, int N);

/**
 * @brief Batched orthogonalization correction kernel
 */
__global__ void batchedOrthogonalizeKernel(cuDoubleComplex* const* basis,
                                          cuDoubleComplex* target,
                                          const cuDoubleComplex* overlaps,
                                          int num_vecs, int N);

} // namespace GPULanczosKernels

#endif // WITH_CUDA
