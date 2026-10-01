#pragma once
// =============================================================================
// include/ed/core/select_backend.h
//
// The one device decision. place(Device, BlockRequest) returns the lane one block
// runs on -- host or device, dense or Krylov -- for every block of every verb, from
// the 'auto' table in device.h, the block's size and capability, and two probes of
// the machine (a visible CUDA device, its free memory). with_backend(lane, fn) runs
// fn on a fresh CpuBackend or CudaBackend of that lane.
//
// The CUDA headers stay here through P2.4 (P2.7 moves them to the users).
// =============================================================================

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include <ed/core/device.h>
#include <ed/core/errors.h>
#include <ed/matvec/linear_operator.h>
#include <ed/core/log.h>
#include <ed/matvec/cpu_backend.h>

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#  include <ed/gpu/cuda_backend.cuh>
#endif

namespace ed {

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

// ---------------------------------------------------------------------------
// place(): the one device decision for every block of every verb.
// ---------------------------------------------------------------------------

/// What place() asks of the machine (a test seam: the [place] unit tests count the calls).
struct DeviceProbe {
    bool (*available)() noexcept = &have_cuda;
    std::optional<std::size_t> (*free_bytes)(bool fresh) noexcept = &free_device_bytes;
};

/// The lane one block runs on. In order:
///   a. DenseBatch: the host under Cpu or without a device ('gpu' raises DeviceUnavailable),
///      else the device (no floor, no memory check).
///   b. A block the verb solves densely runs dense on the host under every device.
///   c. Cpu: the host Krylov lanes, before any probe ('cpu' never initialises CUDA).
///   d. Auto: the device when the block has a kernel, its task may run there, dim >= the
///      task's floor, a device is visible and (fit_vectors > 0) the vectors fit; else the host.
///   e. Gpu: the device, or DeviceUnavailable / DeviceUnsupported / ResourceLimit saying why not.
/// TRANSITIONAL: a device-bound Eigs block of dim <= kDeviceDenseMaxDim or 2 want >= dim is
/// solved densely on the host.
[[nodiscard]] inline Lane place(Device d, const BlockRequest& r, const DeviceProbe& probe = {}) {
    if (r.task == Task::DenseBatch) {
        if (d == Device::Cpu) return Lane::HostDense;
        if (!probe.available()) {
            if (d == Device::Gpu)
                throw ed::DeviceUnavailable(std::string(r.verb) + ": device='gpu', but no usable CUDA device is visible");
            return Lane::HostDense;
        }
        return Lane::DeviceDense;
    }
    if (r.dense) return Lane::HostDense;
    if (d == Device::Cpu) return Lane::HostKrylov;
    const AutoRow row = auto_row(r.task);
    const std::uint64_t need = std::uint64_t{row.fit_vectors} * r.dim * sizeof(std::complex<double>);
    const bool small_eigs = r.task == Task::Eigs && (r.dim <= kDeviceDenseMaxDim || 2 * r.want >= r.dim);
    if (d == Device::Auto) {
        if (!r.device_kernel || r.task == Task::Oftlm || r.dim < row.floor || !probe.available())
            return Lane::HostKrylov;
        if (row.fit_vectors > 0) {
            const std::optional<std::size_t> free = probe.free_bytes(false);
            if (!free || need > *free) return Lane::HostKrylov;
        }
        return small_eigs ? Lane::HostDense : Lane::DeviceKrylov;
    }
    if (!probe.available())
        throw ed::DeviceUnavailable(std::string(r.verb) + ": device='gpu', but no usable CUDA device is visible");
    if (r.task == Task::Oftlm)
        throw ed::DeviceUnsupported(std::string(r.verb) + ": OFTLM (exact_states > 0) runs on the host only; with "
                                    "device='gpu' use FTLM without exact_states, or device='auto' or 'cpu'");
    if (!r.device_kernel)
        throw ed::DeviceUnsupported(std::string(r.verb) + ": device='gpu', but "
                                    + (r.what ? r.what() : "a block of dim " + std::to_string(r.dim)) + " "
                                    + r.why + "; use device='auto' or 'cpu'");
    if (small_eigs) return Lane::HostDense;
    if (row.fit_vectors > 0) {
        const std::optional<std::size_t> free = probe.free_bytes(true);
        if (!free)
            throw ed::DeviceUnavailable("device='gpu', but the device's memory cannot be queried "
                                        "(no CUDA context could be created)");
        if (need > *free)
            throw ed::ResourceLimit("device='gpu', but a block of dim " + std::to_string(r.dim) + " needs "
                                    + std::to_string(need >> 20) + " MiB of device memory and "
                                    + std::to_string(*free >> 20) + " MiB are free");
    }
    return Lane::DeviceKrylov;
}

/// fn(backend) on a fresh CpuBackend for a host lane, or a fresh CudaBackend for a device lane.
/// A device lane in a build without CUDA is a logic error (place() never returns one there).
template <class Fn>
auto with_backend(Lane lane, Fn&& fn) {
    if (on_device(lane)) {
#ifdef WITH_CUDA
        ed::matvec::CudaBackend be;
        return fn(be);
#else
        throw std::logic_error("with_backend: a device lane in a build without CUDA");
#endif
    }
    ed::matvec::CpuBackend be;
    return fn(be);
}

}  // namespace ed
