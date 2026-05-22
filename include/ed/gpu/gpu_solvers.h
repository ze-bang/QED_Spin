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
//      hierarchy (GPUFixedSzOperator, GPUSymmetrizedOperator)
//      participates naturally.
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
// Existing call sites (ed_wrapper.h, ed_wrapper_streaming.h) that go through
// the legacy `GPUEDWrapper::runGPU*` static methods directly stay valid; the
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
            "(GPUOperator / GPUFixedSzOperator / GPUSymmetrizedOperator). "
            "Use the CPU solver overload in ed/solvers/* instead.");
    }
    const auto* gpu = dynamic_cast<const GPUOperator*>(&op);
    if (!gpu) {
        throw std::invalid_argument(
            std::string("ed::matvec::gpu::") + solver_name +
            ": operator advertises MemorySpace::CudaDevice but is not a "
            "GPUOperator subclass; rejecting to avoid undefined behaviour. "
            "Pass a GPUOperator / GPUFixedSzOperator / GPUSymmetrizedOperator.");
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

inline void lanczos(const ed::matvec::MatVecOperator& op,
                    int N, int max_iter, int num_eigs, double tol,
                    std::vector<double>& eigenvalues,
                    std::string dir = "", bool eigenvectors = false)
{
    lanczos(detail::require_gpu_operator(op, "lanczos"),
            N, max_iter, num_eigs, tol, eigenvalues, std::move(dir),
            eigenvectors);
}

inline void block_lanczos(const GPUOperator& op,
                          int N, int max_iter, int num_eigs, int block_size,
                          double tol, std::vector<double>& eigenvalues,
                          std::string dir = "", bool eigenvectors = false)
{
    GPUEDWrapper::runGPUBlockLanczos(detail::gpu_handle(op), N, max_iter,
                                     num_eigs, block_size, tol, eigenvalues,
                                     std::move(dir), eigenvectors);
}

inline void block_lanczos(const ed::matvec::MatVecOperator& op,
                          int N, int max_iter, int num_eigs, int block_size,
                          double tol, std::vector<double>& eigenvalues,
                          std::string dir = "", bool eigenvectors = false)
{
    block_lanczos(detail::require_gpu_operator(op, "block_lanczos"),
                  N, max_iter, num_eigs, block_size, tol, eigenvalues,
                  std::move(dir), eigenvectors);
}

inline void davidson(const GPUOperator& op,
                     int N, int num_eigs, int max_iter, int max_subspace,
                     double tol, std::vector<double>& eigenvalues,
                     std::string dir = "",
                     bool compute_eigenvectors = false)
{
    GPUEDWrapper::runGPUDavidson(detail::gpu_handle(op), N, num_eigs, max_iter,
                                 max_subspace, tol, eigenvalues, std::move(dir),
                                 compute_eigenvectors);
}

inline void davidson(const ed::matvec::MatVecOperator& op,
                     int N, int num_eigs, int max_iter, int max_subspace,
                     double tol, std::vector<double>& eigenvalues,
                     std::string dir = "",
                     bool compute_eigenvectors = false)
{
    davidson(detail::require_gpu_operator(op, "davidson"),
             N, num_eigs, max_iter, max_subspace, tol, eigenvalues,
             std::move(dir), compute_eigenvectors);
}

inline void krylov_schur(const GPUOperator& op,
                         int N, int num_eigs, int max_iter, double tol,
                         std::vector<double>& eigenvalues,
                         std::string dir = "",
                         bool compute_eigenvectors = false)
{
    GPUEDWrapper::runGPUKrylovSchur(detail::gpu_handle(op), N, num_eigs,
                                    max_iter, tol, eigenvalues, std::move(dir),
                                    compute_eigenvectors);
}

inline void krylov_schur(const ed::matvec::MatVecOperator& op,
                         int N, int num_eigs, int max_iter, double tol,
                         std::vector<double>& eigenvalues,
                         std::string dir = "",
                         bool compute_eigenvectors = false)
{
    krylov_schur(detail::require_gpu_operator(op, "krylov_schur"),
                 N, num_eigs, max_iter, tol, eigenvalues, std::move(dir),
                 compute_eigenvectors);
}

inline void block_krylov_schur(const GPUOperator& op,
                               int N, int num_eigs, int max_iter,
                               int block_size, double tol,
                               std::vector<double>& eigenvalues,
                               std::string dir = "",
                               bool compute_eigenvectors = false)
{
    GPUEDWrapper::runGPUBlockKrylovSchur(detail::gpu_handle(op), N, num_eigs,
                                         max_iter, block_size, tol,
                                         eigenvalues, std::move(dir),
                                         compute_eigenvectors);
}

inline void block_krylov_schur(const ed::matvec::MatVecOperator& op,
                               int N, int num_eigs, int max_iter,
                               int block_size, double tol,
                               std::vector<double>& eigenvalues,
                               std::string dir = "",
                               bool compute_eigenvectors = false)
{
    block_krylov_schur(detail::require_gpu_operator(op, "block_krylov_schur"),
                       N, num_eigs, max_iter, block_size, tol, eigenvalues,
                       std::move(dir), compute_eigenvectors);
}

inline void lobpcg(const GPUOperator& op,
                   int N, int num_eigs, int max_iter, double tol,
                   std::vector<double>& eigenvalues,
                   std::string dir = "",
                   bool compute_eigenvectors = false)
{
    GPUEDWrapper::runGPULOBPCG(detail::gpu_handle(op), N, num_eigs, max_iter,
                               tol, eigenvalues, std::move(dir),
                               compute_eigenvectors);
}

inline void lobpcg(const ed::matvec::MatVecOperator& op,
                   int N, int num_eigs, int max_iter, double tol,
                   std::vector<double>& eigenvalues,
                   std::string dir = "",
                   bool compute_eigenvectors = false)
{
    lobpcg(detail::require_gpu_operator(op, "lobpcg"),
           N, num_eigs, max_iter, tol, eigenvalues, std::move(dir),
           compute_eigenvectors);
}

inline void full_diagonalization(const GPUOperator& op,
                                 int N, int num_eigs,
                                 std::vector<double>& eigenvalues,
                                 std::string dir = "",
                                 bool compute_eigenvectors = true)
{
    GPUEDWrapper::runGPUFullDiag(detail::gpu_handle(op), N, num_eigs,
                                 eigenvalues, std::move(dir),
                                 compute_eigenvectors);
}

inline void full_diagonalization(const ed::matvec::MatVecOperator& op,
                                 int N, int num_eigs,
                                 std::vector<double>& eigenvalues,
                                 std::string dir = "",
                                 bool compute_eigenvectors = true)
{
    full_diagonalization(detail::require_gpu_operator(op, "full_diagonalization"),
                         N, num_eigs, eigenvalues, std::move(dir),
                         compute_eigenvectors);
}

// ---------------------------------------------------------------------------
// Thermal: FTLM and TPQ. (Only the most-used entry points are mirrored;
// the full FTLM observable / dynamics surface lives in GPUEDWrapper and can
// be reached the same way -- write a one-line forwarder when you need one.)
// ---------------------------------------------------------------------------
inline void ftlm(const GPUOperator& op,
                 int N, int krylov_dim, int num_samples,
                 double temp_min, double temp_max, int num_temp_bins,
                 double tolerance, std::string dir = "",
                 bool full_reorth = false, int reorth_freq = 10,
                 unsigned int random_seed = 0)
{
    GPUEDWrapper::runGPUFTLM(detail::gpu_handle(op), N, krylov_dim, num_samples,
                             temp_min, temp_max, num_temp_bins, tolerance,
                             std::move(dir), full_reorth, reorth_freq,
                             random_seed);
}

inline void ftlm(const ed::matvec::MatVecOperator& op,
                 int N, int krylov_dim, int num_samples,
                 double temp_min, double temp_max, int num_temp_bins,
                 double tolerance, std::string dir = "",
                 bool full_reorth = false, int reorth_freq = 10,
                 unsigned int random_seed = 0)
{
    ftlm(detail::require_gpu_operator(op, "ftlm"),
         N, krylov_dim, num_samples, temp_min, temp_max, num_temp_bins,
         tolerance, std::move(dir), full_reorth, reorth_freq, random_seed);
}

inline void microcanonical_tpq(const GPUOperator& op,
                               int N, int max_iter, int num_samples,
                               int temp_interval,
                               std::vector<double>& eigenvalues,
                               std::string dir = "",
                               double large_value = 1e5,
                               bool continue_quenching = false,
                               int continue_sample = 0,
                               double continue_beta = 0.0,
                               bool save_thermal_states = false,
                               double target_beta = 1000.0,
                               int num_measure_points = 20,
                               double measure_beta_min = 1.0,
                               double measure_beta_max = 1000.0)
{
    GPUEDWrapper::runGPUMicrocanonicalTPQ(
        detail::gpu_handle(op), N, max_iter, num_samples, temp_interval,
        eigenvalues, std::move(dir), large_value, continue_quenching,
        continue_sample, continue_beta, save_thermal_states, target_beta,
        num_measure_points, measure_beta_min, measure_beta_max);
}

inline void microcanonical_tpq(const ed::matvec::MatVecOperator& op,
                               int N, int max_iter, int num_samples,
                               int temp_interval,
                               std::vector<double>& eigenvalues,
                               std::string dir = "",
                               double large_value = 1e5,
                               bool continue_quenching = false,
                               int continue_sample = 0,
                               double continue_beta = 0.0,
                               bool save_thermal_states = false,
                               double target_beta = 1000.0,
                               int num_measure_points = 20,
                               double measure_beta_min = 1.0,
                               double measure_beta_max = 1000.0)
{
    microcanonical_tpq(detail::require_gpu_operator(op, "microcanonical_tpq"),
                       N, max_iter, num_samples, temp_interval, eigenvalues,
                       std::move(dir), large_value, continue_quenching,
                       continue_sample, continue_beta, save_thermal_states,
                       target_beta, num_measure_points, measure_beta_min,
                       measure_beta_max);
}

inline void canonical_tpq(const GPUOperator& op,
                          int N, double beta_max, int num_samples,
                          int temp_interval, std::vector<double>& energies,
                          std::string dir = "",
                          double delta_beta = 0.1, int taylor_order = 50,
                          int num_measure_points = 20,
                          double measure_beta_min = 1.0,
                          double measure_beta_max = 1000.0)
{
    GPUEDWrapper::runGPUCanonicalTPQ(
        detail::gpu_handle(op), N, beta_max, num_samples, temp_interval,
        energies, std::move(dir), delta_beta, taylor_order,
        num_measure_points, measure_beta_min, measure_beta_max);
}

inline void canonical_tpq(const ed::matvec::MatVecOperator& op,
                          int N, double beta_max, int num_samples,
                          int temp_interval, std::vector<double>& energies,
                          std::string dir = "",
                          double delta_beta = 0.1, int taylor_order = 50,
                          int num_measure_points = 20,
                          double measure_beta_min = 1.0,
                          double measure_beta_max = 1000.0)
{
    canonical_tpq(detail::require_gpu_operator(op, "canonical_tpq"),
                  N, beta_max, num_samples, temp_interval, energies,
                  std::move(dir), delta_beta, taylor_order,
                  num_measure_points, measure_beta_min, measure_beta_max);
}

}  // namespace ed::matvec::gpu

#endif  // WITH_CUDA
