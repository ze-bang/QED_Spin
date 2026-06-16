#pragma once

// =============================================================================
// include/ed/gpu/gpu_solvers.h   (matvec-unification Phase 4 -- GPU)
//
// Thin convenience overloads that let any callsite that already holds a
// `GPUOperator` (or a `MatVecOperator` known to be GPU-resident) drive the
// existing GPUEDWrapper kernels without having to manage the opaque
// `void* gpu_op_handle` plumbing.
//
// Two layers:
//
//   1. `ed::matvec::gpu::lanczos(const GPUOperator& op, ...)` etc. --
//      type-safe, zero-overhead forwarders. The GPUOperator subclass
//      hierarchy (GPUFixedSzOperator) participates naturally.
//
//   2. `ed::matvec::gpu::lanczos(const MatVecOperator& op, ...)` etc. --
//      runtime-checks `op.memory_space() == MemorySpace::CudaDevice`, then
//      dynamic_casts to `const GPUOperator&` and delegates to (1). This is
//      the fully unified surface promised by Phase 4 -- callers can pass
//      any concrete operator and the bridge figures out the rest. Throws
//      `std::invalid_argument` if the operator is not GPU-resident.
//
// All entry points are inline and zero-cost in release builds; the dynamic
// cast is the only addition over the void* path, paid once per solver run.
//
// Existing call sites (the orchestrator's GPU lane in ``ed::workflows::solve``
// and the streaming-symmetry GPU kernels) that go through the
// ``GPUEDWrapper::runGPU*`` static methods directly stay valid; the
// helpers below are additive sugar, not a replacement.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_ed_wrapper.h>
#include <ed/matvec/matvec.h>
#include <ed/matvec/memory_space.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <complex>
#include <functional>

namespace ed::matvec::gpu {

namespace detail {

// One-stop "is this operator runnable on the GPU solvers?" check + extract
// of the underlying GPUOperator*. The void* the GPUEDWrapper::runGPU*
// kernels want is just a re-cast of the GPUOperator pointer (see
// gpu_ed_wrapper.cu's createGPUOperatorFromCPU / createGPUFixedSzOperator
// implementations, which return `static_cast<void*>(new GPUOperator{...})`).
inline const GPUOperator& require_gpu_operator(
    const ed::matvec::MatVecOperator& op,
    const char* solver_name)
{
    if (op.memory_space() != ed::matvec::MemorySpace::CudaDevice) {
        throw std::invalid_argument(
            std::string("ed::matvec::gpu::") + solver_name +
            ": operator memory_space is not CudaDevice (got memory_space "
            "tag " + std::to_string(static_cast<int>(op.memory_space())) +
            "); the GPU solvers require a GPU-resident operator "
            "(GPUOperator / GPUFixedSzOperator). "
            "Use the CPU solver overload in ed/solvers/* instead.");
    }
    const auto* gpu = dynamic_cast<const GPUOperator*>(&op);
    if (!gpu) {
        throw std::invalid_argument(
            std::string("ed::matvec::gpu::") + solver_name +
            ": operator advertises MemorySpace::CudaDevice but is not a "
            "GPUOperator subclass; rejecting to avoid undefined behaviour. "
            "Pass a GPUOperator / GPUFixedSzOperator.");
    }
    return *gpu;
}

inline void* gpu_handle(const GPUOperator& op) {
    // Symmetric with the runGPU* call sites in ed_wrapper.h, which already
    // pass a `void*` obtained by `static_cast<void*>(GPUOperator*)`. The
    // const_cast is safe here because the GPU solver kernels do not
    // mutate the operator -- they read the SoA term tables and write into
    // user-owned device buffers.
    return static_cast<void*>(const_cast<GPUOperator*>(&op));
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Lanczos family
// ---------------------------------------------------------------------------
inline void lanczos(const GPUOperator& op,
                    int N, int max_iter, int num_eigs, double tol,
                    std::vector<double>& eigenvalues,
                    std::string dir = "", bool eigenvectors = false)
{
    GPUEDWrapper::runGPULanczos(detail::gpu_handle(op), N, max_iter, num_eigs,
                                tol, eigenvalues, std::move(dir), eigenvectors);
}

// ---------------------------------------------------------------------------
// GPU Lanczos through the unified Krylov kernel
// (`ed::krylov::lanczos_kernel<CudaBackend>`).
//
// First production migration onto the unified `Backend`-templated
// Krylov kernel (May 2026). Replaces the corresponding paths of
// `GPULanczos::run`. Full CGS2 reorth + basis held in device memory.
//
// Two entry points:
//
//   * `run_lanczos_eigenvalues_kernel_facade(...)` -- eigenvalues only.
//     Diagonalises the small tridiagonal on the host (Eigen) and
//     returns the lowest `num_eigs` Ritz values.
//
//   * `run_lanczos_eigenpairs_kernel_facade(...)` -- eigenvalues +
//     eigenvectors. After the tridiag solve, reconstructs the lowest
//     `num_eigs` Ritz vectors as
//
//         y_i = sum_{k=0}^{M-1} S(k, i) * V_k
//
//     by `num_eigs * M` backend axpys on the retained Krylov basis
//     (V_k lives in device memory; the axpys use cuBLAS). The Ritz
//     vectors are copied to host into the `eigenvectors_out`
//     std::vector<std::vector<Complex>>, matching the legacy
//     `GPULanczos::run` output type.
//
// Both implementations live in
// `src/solvers/gpu/gpu_lanczos_kernel_facade.cu`. `runGPULanczos(...)`
// in `gpu_ed_wrapper.cu` dispatches to one of the two based on
// `eigenvectors`, so every existing call site picks up the new path
// without a source change. The legacy `GPULanczos::run` is kept only
// for the windowed-reorth / on-disk basis-spill regime (when the
// basis won't fit in device memory).
// ---------------------------------------------------------------------------
void run_lanczos_eigenvalues_kernel_facade(
    GPUOperator& gpu_op,
    int N,
    int max_iter,
    int num_eigs,
    double tol,
    unsigned long long seed,
    std::vector<double>& eigenvalues_out);

void run_lanczos_eigenpairs_kernel_facade(
    GPUOperator& gpu_op,
    int N,
    int max_iter,
    int num_eigs,
    double tol,
    unsigned long long seed,
    std::vector<double>& eigenvalues_out,
    std::vector<std::vector<std::complex<double>>>& eigenvectors_out);

// FTLM-shaped Lanczos build that takes a caller-provided starting
// vector (already in device memory, already normalised) and writes
// the resulting alpha / beta tridiagonal arrays plus -- optionally --
// a caller-owned array of device basis pointers.
//
// Phase 2 of the gap-fill rollout (May 2026 day 11+): replaces the
// hand-rolled inline Lanczos bodies in `gpu_ftlm.cu`
// (`buildLanczosTridiagonalWithBasis`, `buildLanczosTridiagonalFromVector`)
// that drove their reorthogonalisation through M sequential
// `cublasZdotc` calls per step. Routes through the unified
// `lanczos_kernel<CudaBackend>` which uses batched CGS2 via
// `CudaBackend::dot_many` / `axpy_many` (a single cublasZgemv per
// pass).
//
// Ownership: when `d_basis_out` is non-null, the facade releases
// each basis vector from the kernel's RAII unique_ptr into a
// heap-allocated `cuDoubleComplex**` array which the caller must
// free with `cudaFreeAsync(...)` followed by `delete[]`. The
// allocator inside `CudaBackend` is pool-backed
// (`cudaMallocAsync`), so `cudaFreeAsync` is the matching free; a
// plain `cudaFree` is undefined behaviour per the CUDA contract.
// When `d_basis_out` is null, the kernel-owned basis is dropped
// (RAII free) before the function returns.
//
// `reorth_freq` follows the legacy convention: `full_reorth=true`
// forces full CGS2 on every step; `full_reorth=false` with
// `reorth_freq > 0` triggers `PeriodicCGS2` (CGS2 every
// `reorth_freq` steps); `full_reorth=false` with `reorth_freq == 0`
// is unreorth (pure three-term recurrence -- only safe for very
// short Krylov sequences).
//
// `tol` is the **breakdown** threshold on `beta_{j+1}`; defaults
// match `GPUFTLMSolver::tolerance_` (1e-12).
//
// Return: number of iterations actually completed (equals
// `alpha_out.size()`).
int run_ftlm_lanczos_kernel_facade(
    const std::function<void(const cuDoubleComplex*, cuDoubleComplex*, int)>& matvec,
    const void*  d_start_vec,            // cuDoubleComplex*; void* to keep cuda.h optional
    int          N,
    int          krylov_dim,
    bool         full_reorth,
    int          reorth_freq,
    double       tol,
    std::vector<double>& alpha_out,
    std::vector<double>& beta_out,
    void**       d_basis_out);           // cuDoubleComplex***; nullable

inline void lanczos(const ed::matvec::MatVecOperator& op,
                    int N, int max_iter, int num_eigs, double tol,
                    std::vector<double>& eigenvalues,
                    std::string dir = "", bool eigenvectors = false)
{
    lanczos(detail::require_gpu_operator(op, "lanczos"),
            N, max_iter, num_eigs, tol, eigenvalues, std::move(dir),
            eigenvectors);
}

// The `block_lanczos`, `krylov_schur`, `full_diagonalization`, `ftlm`,
// `microcanonical_tpq`, and `canonical_tpq` Phase-4-GPU forwarders that
// used to live here were retired in the minimalist-architecture rev
// (May 2026): only the `lanczos` pair was ever called (from
// test_cpu_gpu_equivalence.cpp; the rest were API scaffolding for an
// unused unified surface). All remaining call sites go through
// `GPUEDWrapper::runGPU*` directly from the orchestrator's GPU lane
// (``ed::workflows::solve``) and the streaming-symmetry GPU kernels.
// To re-introduce one, write a one-line forwarder following the
// `lanczos` template above.

}  // namespace ed::matvec::gpu

#endif  // WITH_CUDA
