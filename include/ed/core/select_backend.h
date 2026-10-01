#pragma once
#include <chrono>
// =============================================================================
// include/ed/core/select_backend.h
//
// `ed::select_backend(LinearOperator, BackendConstraints)`: the runtime
// dispatch helper consumed by the orchestrators (`ed::workflows::solve`,
// `ed::workflows::thermal`). Resolves the (have_cuda, gpu_mem_fits,
// user constraints) tuple into a single `BackendVariant` the caller can
// `std::visit` over.
//
// Decision order:
//   0. neither allow_gpu nor require_gpu             --> CpuBackend, before any CUDA call
//   1. if have_cuda() AND gpu_mem_fits AND allow_gpu --> CudaBackend
//   2. require_gpu                                   --> throw, saying why the GPU cannot run it
//   3. else                                          --> CpuBackend
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include <ed/core/errors.h>
#include <ed/core/linear_operator.h>
#include <ed/core/log.h>
#include <ed/matvec/backends/cpu_backend.h>

#include <mutex>

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#  include <ed/matvec/backends/cuda_backend.cuh>
#endif

namespace ed {

struct BackendConstraints {
    bool allow_gpu     = true;
    /// Per-element memory budget multiplier (workspace overhead). 8.0
    /// is a safe default for Lanczos / TPQ which carry ~5 N-length
    /// scratch vectors plus the basis.
    double fudge_factor = 8.0;
    /// Minimum problem dimension for the GPU AUTO-promotion.
    /// Below this, kernel-launch + transfer overhead makes the GPU
    /// strictly slower than the CPU, and on shared-GPU hosts (WSL2)
    /// tiny solves dispatched to a contended device can return
    /// silently-wrong spectra. Callers that EXPLICITLY
    /// request the GPU (``device='gpu'``) set this to 0 -- the floor
    /// gates only the automatic promotion, never an explicit choice.
    std::size_t gpu_dim_floor = (std::size_t{1} << 14);
    /// device='gpu': the operator must run on the device. When it cannot,
    /// select_backend throws instead of returning the CPU backend:
    /// DeviceUnavailable (no usable device), DeviceUnsupported (no device
    /// kernel for this operator), ResourceLimit (not enough device memory).
    bool require_gpu = false;
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
>;

// ---------------------------------------------------------------------------
// Runtime probes.
// ---------------------------------------------------------------------------
inline bool have_cuda() noexcept {
#ifdef WITH_CUDA
    // Probed once per process. A wheel built against a newer CUDA toolkit
    // than the node's driver reports cudaErrorInsufficientDriver here;
    // one Warn record says so, then the documented CPU fallback.
    static const bool ok = [] {
        int n = 0;
        const cudaError_t err = cudaGetDeviceCount(&n);
        if (err == cudaSuccess) return n > 0;
        cudaGetLastError();
        if (err == cudaErrorInsufficientDriver
#if CUDART_VERSION >= 11000
            || err == cudaErrorSystemDriverMismatch
#endif
        ) {
            int drv = 0, rt = 0;
            cudaDriverGetVersion(&drv);
            cudaRuntimeGetVersion(&rt);
            ED_LOG(Warn,
                "CUDA DISABLED: the NVIDIA driver on this machine is "
                "too old for this build (driver API %d.%d < runtime %d.%d). "
                "Every GPU lane falls back to CPU. Fix: update the driver, "
                "or rebuild against this node's CUDA toolkit.",
                drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);
        }
        return false;
    }();
    return ok;
#else
    return false;
#endif
}

/// Device memory this process can still allocate, or nothing when the device cannot be queried.
/// The driver's free count misses what the process freed into its default memory pool (CudaBackend
/// keeps the pool's memory: release threshold UINT64_MAX), which the next allocation reuses, so
/// the pool's reserved-but-unused bytes are added. Cached for one second (cudaMemGetInfo costs
/// 20-100 ms per call under WSL2, and the automatic choice only needs an order of magnitude);
/// `fresh` bypasses the cache, for a strict device request.
inline std::optional<std::size_t> free_device_bytes(bool fresh = false) noexcept {
#ifdef WITH_CUDA
    static std::mutex m;
    static std::size_t cached_free = 0;
    static std::chrono::steady_clock::time_point cached_at{};
    const std::lock_guard<std::mutex> lock(m);
    const auto now = std::chrono::steady_clock::now();
    if (fresh || cached_at == std::chrono::steady_clock::time_point{} || now - cached_at > std::chrono::seconds(1)) {
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            cudaGetLastError();
            return std::nullopt;
        }
        int dev = -1;
        cudaMemPool_t pool = nullptr;
        std::uint64_t reserved = 0, used = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess
                && cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) == cudaSuccess
                && cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess) {
            if (reserved > used) free_bytes += static_cast<std::size_t>(reserved - used);
        } else {
            cudaGetLastError();
        }
        cached_free = free_bytes;
        cached_at   = now;
    }
    return cached_free;
#else
    (void)fresh;
    return std::nullopt;
#endif
}

/// Device memory the operator's solve needs: `fudge_factor` vectors of its dimension.
inline std::size_t gpu_bytes_needed(const Geometry& geom, const BackendConstraints& c) noexcept {
    return static_cast<std::size_t>(geom.local_dim * sizeof(std::complex<double>) * c.fudge_factor);
}

inline bool gpu_mem_fits(const Geometry& geom,
                          const BackendConstraints& c) noexcept {
#ifdef WITH_CUDA
    if (!have_cuda()) return false;
    const std::optional<std::size_t> budget =
        free_device_bytes(c.require_gpu);
    return budget.has_value() && gpu_bytes_needed(geom, c) <= *budget;
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
    // device='cpu' never initialises CUDA.
    if (!c.allow_gpu && !c.require_gpu)
        return BackendVariant{std::make_unique<ed::matvec::CpuBackend>()};
    const bool have_gpu = have_cuda();
    const bool gpu_fits = have_gpu && gpu_mem_fits(geom, c);
    (void)have_gpu;

    // CRITICAL: select_backend can only pick a Backend that matches
    // the operator's declared memory space. A Host operator forced
    // through a CudaBackend would attempt cudaMemcpy on host pointers
    // and crash. Honour the operator's preference; the caller can
    // upgrade by providing a device-side operator.
    const bool op_is_host_only =
        ed::matvec::is_host(geom.memory_space);
    const bool op_is_device =
        ed::matvec::is_device(geom.memory_space);

#ifdef WITH_CUDA
    // An operator can advertise device-matvec capability even when its
    // native storage is host; `bind_cuda()` then lazily builds a GPU
    // mirror. Symmetry sector operators use this: they are host-resident
    // but their device mirror runs on the GPU.
    const bool device_mv = op_is_device || geom.supports_device_matvec;
    // gpu_dim_floor gates only the AUTO promotion: a device-resident
    // operator has already committed to the GPU, and explicit requests
    // arrive with the floor zeroed.
    const bool dim_ok = op_is_device || c.require_gpu || geom.local_dim >= c.gpu_dim_floor;
    if (device_mv && have_gpu && gpu_fits && dim_ok) {
        return BackendVariant{std::make_unique<ed::matvec::CudaBackend>()};
    }
    if (c.require_gpu) {
        if (!have_gpu)
            throw ed::DeviceUnavailable("device='gpu', but no usable CUDA device is visible");
        if (!device_mv)
            throw ed::DeviceUnsupported("device='gpu', but this operator (dim " + std::to_string(geom.local_dim)
                                        + ") has no device kernel; use device='auto' or 'cpu'");
        const std::optional<std::size_t> budget =
            free_device_bytes(true);
        if (!budget)
            throw ed::DeviceUnavailable("device='gpu', but the device's memory cannot be queried "
                                        "(no CUDA context could be created)");
        throw ed::ResourceLimit("device='gpu', but a block of dim " + std::to_string(geom.local_dim) + " needs "
                                + std::to_string(gpu_bytes_needed(geom, c) >> 20) + " MiB of device memory and "
                                + std::to_string(*budget >> 20) + " MiB are free");
    }
#else
    if (c.require_gpu)
        throw ed::DeviceUnavailable("device='gpu', but this build has no CUDA");
#endif

    (void)gpu_fits; (void)op_is_host_only;
    return BackendVariant{std::make_unique<ed::matvec::CpuBackend>()};
}

/// Convenience: forward through `LinearOperator::geometry()`.
inline BackendVariant select_backend(const LinearOperator& op,
                                     const BackendConstraints& c = {}) {
    return select_backend(op.geometry(), c);
}

// ---------------------------------------------------------------------------
// lane_label_for<Backend>() / lane_label_from_variant(v): truthful lane
// reporting helpers.
//
// The lane is derived from the selected Backend, not from
// `H.geometry().is_device()`: a host-resident operator that advertises
// `supports_device_matvec` (e.g. a symmetry sector with a lazily built GPU
// mirror) runs on `CudaBackend` while its memory_space still reads `Host`.
//
// `lane_label_for<Backend>()` is the template form (cheap, available
// inside any `solve_on<Backend>` / `thermal_on<Backend>` body).
// `lane_label_from_variant(v)` visits the variant for callers that
// already hold a `BackendVariant`. Both return one of
// {"cpu","gpu"}.
// ---------------------------------------------------------------------------
template <typename Backend>
inline std::string lane_label_for() {
    if constexpr (std::is_same_v<Backend, ed::matvec::CpuBackend>) {
        return std::string{"cpu"};
    }
#ifdef WITH_CUDA
    else if constexpr (std::is_same_v<Backend, ed::matvec::CudaBackend>) {
        return std::string{"gpu"};
    }
#endif
    else {
        return std::string{"cpu"};
    }
}

inline std::string lane_label_from_variant(const BackendVariant& v) {
    return std::visit([](const auto& ptr) -> std::string {
        using Ptr = std::decay_t<decltype(ptr)>;
        using B   = typename Ptr::element_type;
        return lane_label_for<B>();
    }, v);
}

}  // namespace ed
