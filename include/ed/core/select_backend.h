#pragma once
// =============================================================================
// include/ed/core/select_backend.h
//
// `ed::select_backend(LinearOperator, BackendConstraints)`: the runtime
// dispatch helper consumed by the Phase-4 orchestrators (`ed::solve`,
// `ed::thermal`, `ed::spectral`). Resolves the (mpi_size, have_cuda,
// gpu_mem_fits, user constraints) tuple into a single `BackendVariant`
// the caller can `std::visit` over.
//
// Decision order (matches the legacy `auto/solve.cpp` heuristic):
//   1. if mpi_size > 1 AND have_cuda() AND gpu_mem_fits AND allow_mpi_gpu
//          --> MpiCudaBackend
//   2. else if mpi_size > 1 AND allow_mpi          --> MpiBackend
//   3. else if have_cuda() AND gpu_mem_fits AND allow_gpu --> CudaBackend
//   4. else                                          --> CpuBackend
//
// Phase 4.1 of the Minimalist ED Collapse (May 2026).
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>

#include <ed/core/linear_operator.h>
#include <ed/matvec/backends/cpu_backend.h>

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#  include <ed/matvec/backends/cuda_backend.cuh>
#endif

#ifdef WITH_MPI
#  include <mpi.h>
#  include <ed/matvec/backends/mpi_backend.h>
#  ifdef ED_HAVE_NCCL
#    include <ed/distributed/multi_gpu.h>
#    include <ed/matvec/backends/mpi_cuda_backend.cuh>
#  endif
#endif

namespace ed {

struct BackendConstraints {
    bool allow_gpu     = true;
    bool allow_mpi     = true;
    bool allow_mpi_gpu = true;
    /// Per-rank GPU memory budget. Empty means "no limit; trust the
    /// `cudaGetDeviceProperties` query". When set, gpu_mem_fits is
    /// computed as `bytes_per_complex * geometry.local_dim *
    /// fudge_factor <= gpu_mem_bytes`.
    std::optional<std::size_t> gpu_mem_bytes;
    /// Per-element memory budget multiplier (workspace overhead). 8.0
    /// is a safe default for Lanczos / TPQ which carry ~5 N-length
    /// scratch vectors plus the basis.
    double fudge_factor = 8.0;
};

// ---------------------------------------------------------------------------
// BackendVariant carries a unique_ptr to one of the concrete Backend
// classes (so its address is stable across the orchestrator dispatch).
// Variant alternatives are conditionally compiled in based on build
// flags, matching the rest of the codebase.
// ---------------------------------------------------------------------------
using BackendVariant = std::variant<
    std::unique_ptr<ed::matvec::CpuBackend>
#ifdef WITH_CUDA
    , std::unique_ptr<ed::matvec::CudaBackend>
#endif
#ifdef WITH_MPI
    , std::unique_ptr<ed::matvec::MpiBackend>
#  ifdef ED_HAVE_NCCL
    , std::unique_ptr<ed::matvec::MpiCudaBackend>
#  endif
#endif
>;

// ---------------------------------------------------------------------------
// Runtime probes.
// ---------------------------------------------------------------------------
inline bool have_cuda() noexcept {
#ifdef WITH_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
#else
    return false;
#endif
}

inline std::size_t mpi_size_or_one() noexcept {
#ifdef WITH_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) return 1;
    int sz = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &sz);
    return static_cast<std::size_t>(sz);
#else
    return 1;
#endif
}

inline bool gpu_mem_fits(const Geometry& geom,
                          const BackendConstraints& c) noexcept {
#ifdef WITH_CUDA
    if (!have_cuda()) return false;
    // Probe the active device's free memory if no explicit budget.
    std::size_t budget;
    if (c.gpu_mem_bytes.has_value()) {
        budget = c.gpu_mem_bytes.value();
    } else {
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        budget = free_bytes;
    }
    const std::size_t per_complex = sizeof(std::complex<double>);
    const std::size_t need = static_cast<std::size_t>(
        geom.local_dim * per_complex * c.fudge_factor);
    return need <= budget;
#else
    (void)geom; (void)c;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// select_backend
// ---------------------------------------------------------------------------
inline BackendVariant select_backend(const Geometry& geom,
                                     const BackendConstraints& c = {})
{
    const std::size_t nmpi = mpi_size_or_one();
    const bool have_gpu = have_cuda();
    const bool gpu_fits = have_gpu && gpu_mem_fits(geom, c);
    (void)nmpi; (void)have_gpu;

    // CRITICAL: select_backend can only pick a Backend that matches
    // the operator's declared memory space. A Host operator forced
    // through a CudaBackend would attempt cudaMemcpy on host pointers
    // and crash. Honour the operator's preference; the caller can
    // upgrade by providing a device-side operator.
    const bool op_is_host_only =
        ed::matvec::is_host(geom.memory_space);
    const bool op_is_device =
        ed::matvec::is_device(geom.memory_space);

#ifdef WITH_MPI
    // NOTE: MpiCudaBackend (the GPU+MPI lane) requires a
    // `MultiGpuCommunicator` that the caller owns separately --- the
    // NCCL handle cannot be auto-constructed here without coordinating
    // initialisation. When the caller wants the MPI+GPU lane it
    // constructs the BackendVariant in place via the MpiCudaBackend
    // alternative; this auto-dispatch returns MpiBackend (host-staged
    // collectives over CUDA-aware MPI) when only mpi_size > 1 is true,
    // and CudaBackend when only the GPU is available.
    if (nmpi > 1 && c.allow_mpi && !op_is_device) {
        return BackendVariant{std::make_unique<ed::matvec::MpiBackend>(
            MPI_COMM_WORLD)};
    }
#endif

#ifdef WITH_CUDA
    // Phase 2 of the "Unified CPU/GPU symmetry architecture" plan
    // (May 2026): an operator can advertise device-matvec capability
    // even when its native storage is host. `bind_cuda()` is expected
    // to lazily build a GPU mirror in that case. This unblocks the
    // `qed.solve(symmetry=..., device='gpu')` lane: the symmetry
    // SectorViews are host-resident but their GPUSymmetrizedOperator
    // mirror (lazily constructed inside `bind_cuda`) runs on the GPU.
    const bool device_mv = op_is_device || geom.supports_device_matvec;
    if (device_mv && have_gpu && gpu_fits && c.allow_gpu) {
        return BackendVariant{std::make_unique<ed::matvec::CudaBackend>()};
    }
#endif

    (void)gpu_fits; (void)op_is_host_only;
    return BackendVariant{std::make_unique<ed::matvec::CpuBackend>()};
}

/// Convenience: forward through `LinearOperator::geometry()`.
inline BackendVariant select_backend(const LinearOperator& op,
                                     const BackendConstraints& c = {}) {
    return select_backend(op.geometry(), c);
}

#if defined(WITH_MPI) && defined(WITH_CUDA) && defined(ED_HAVE_NCCL)
/// `WithMpiCudaBackend` --- explicit opt-in for the MPI+GPU lane.
///
/// `select_backend` cannot auto-construct an `MpiCudaBackend` because
/// the constructor needs a `MultiGpuCommunicator` that the caller owns.
/// CLIs that pass `--gpu` on the distributed binary can use this helper
/// to wrap an externally-constructed `MpiCudaBackend` in the
/// `BackendVariant` that the orchestrators consume.
///
/// ED Cleanup Sweep Phase 4 (May 2026): added so the distributed-CLI
/// `--gpu` flag has a uniform path through the `ed::workflows::*`
/// orchestrators (matching the CPU lane's auto-selection).
inline BackendVariant WithMpiCudaBackend(
    std::unique_ptr<ed::matvec::MpiCudaBackend> be) {
    return BackendVariant{std::move(be)};
}

/// `MpiCudaBackendContext` --- RAII bundle bundling a NCCL
/// `MultiGpuCommunicator` and the `MpiCudaBackend` that references it.
///
/// The `MpiCudaBackend` keeps a reference to the
/// `MultiGpuCommunicator`; both must outlive the kernel call. This
/// context bundle owns the comm by `unique_ptr` and hands the backend
/// out via the `BackendVariant` member. Construct once at CLI startup,
/// keep alive until after the orchestrator call returns.
///
/// The context is non-movable / non-copyable on purpose: the backend
/// holds a raw reference into the comm. Stash it on the caller's stack
/// (e.g. inside `main()`) for the duration of the workflow.
struct MpiCudaBackendContext {
    std::unique_ptr<ed::distributed::multi_gpu::MultiGpuCommunicator> comm;
    BackendVariant                                                   backend;

    MpiCudaBackendContext(MpiCudaBackendContext&&) = delete;
    MpiCudaBackendContext& operator=(MpiCudaBackendContext&&) = delete;
};

/// Build a `MpiCudaBackendContext` from an MPI communicator and an
/// optional explicit CUDA device index. Defaults: `MPI_COMM_WORLD` and
/// auto-detected node-local device binding (one GPU per rank,
/// node-local rank modulo visible device count).
///
/// Construction is COLLECTIVE on `mpi_comm`. Throws
/// `std::logic_error` if `ED_HAVE_NCCL` was not set at compile time,
/// `std::runtime_error` on cuda / nccl / MPI failure.
///
/// Typical use:
///
///     auto ctx = ed::make_mpi_cuda_backend();      // collective
///     auto op  = ed::make_operator({...,
///                                   .distributed = true});
///     auto gs  = ed::workflows::solve(*op, opts);  // uses ctx.backend
///                                                   // via the caller
///                                                   // plumbing.
///
/// The orchestrator does not yet accept an externally-supplied
/// `BackendVariant`; callers needing the MPI+CUDA lane today drive the
/// kernel manually via `std::visit(ctx.backend, ...)`. The factory
/// exists so that machinery has a single, RAII-correct entry point.
inline std::unique_ptr<MpiCudaBackendContext> make_mpi_cuda_backend(
    MPI_Comm mpi_comm = MPI_COMM_WORLD,
    int      device_index =
        ed::distributed::multi_gpu::kAutoDeviceIndex)
{
    if (!ed::distributed::multi_gpu::nccl_compiled_in()) {
        throw std::logic_error(
            "ed::make_mpi_cuda_backend: NCCL not compiled in. Rebuild "
            "with -DED_USE_NCCL=ON and a NCCL-enabled toolchain.");
    }
    auto comm =
        std::make_unique<ed::distributed::multi_gpu::MultiGpuCommunicator>(
            mpi_comm, device_index);
    auto be = std::make_unique<ed::matvec::MpiCudaBackend>(*comm);
    auto ctx = std::unique_ptr<MpiCudaBackendContext>(
        new MpiCudaBackendContext{
            std::move(comm), WithMpiCudaBackend(std::move(be))});
    return ctx;
}
#endif

}  // namespace ed
