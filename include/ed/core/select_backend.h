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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <ed/core/device.h>
#include <ed/core/footprint.h>
#include <ed/core/memory.h>
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


// ---------------------------------------------------------------------------
// place(): the one device decision for every block of every verb.
// ---------------------------------------------------------------------------

/// What place() asks of the machine (a test seam: the [place] unit tests count the calls).
struct DeviceProbe {
    bool (*available)() noexcept = &have_cuda;
    std::optional<std::size_t> (*free_bytes)(bool fresh) noexcept = &ed::core::available_device_bytes;
};

/// The device working set place() checks for a block whose request does not state it.
[[nodiscard]] inline std::uint64_t device_need(const BlockRequest& r) {
    if (r.device_bytes > 0) return r.device_bytes;
    using ed::core::Path;
    ed::core::Shape s;
    s.dim = r.dim;
    s.device = true;
    if (r.task == Task::Sampled) return ed::core::footprint(Path::FtlmSample, s).device;
    if (r.want <= 1) return ed::core::footprint(Path::GsTwoPass, s).device;
    s.k = static_cast<std::size_t>(r.want);
    s.krylov = s.k + 8;
    return ed::core::footprint(Path::KrylovSchur, s).device;
}

/// The lane one block runs on. In order:
///   a. DenseBatch: the host under Cpu or without a device ('gpu' raises DeviceUnavailable),
///      else the device (no floor, no memory check).
///   b. A block the verb solves densely runs dense on the host under every device.
///   c. Cpu: the host Krylov lanes, before any probe ('cpu' never initialises CUDA).
///   d. Auto: the device when the block has a kernel, its task may run there, dim >= the
///      task's floor, a device is visible and (the row's fit) its device working set fits in free
///      device memory (not checked under ED_MEM_GUARD_OFF); else the host.
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
    const bool fit = row.fit && !ed::core::mem_guard_off();
    const std::uint64_t need = fit ? device_need(r) : 0;
    const bool small_eigs = r.task == Task::Eigs && (r.dim <= kDeviceDenseMaxDim || 2 * r.want >= r.dim);
    if (d == Device::Auto) {
        if (!r.device_kernel || r.task == Task::Oftlm || r.dim < row.floor || !probe.available())
            return Lane::HostKrylov;
        if (fit) {
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
    if (fit) {
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
