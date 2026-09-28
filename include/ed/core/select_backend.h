#pragma once
#include <chrono>
// =============================================================================
// include/ed/core/select_backend.h
//
// `ed::select_backend(LinearOperator, BackendConstraints)`: the runtime
// dispatch helper consumed by the Phase-4 orchestrators (`ed::solve`,
// `ed::thermal`, `ed::spectral`). Resolves the (have_cuda, gpu_mem_fits,
// user constraints) tuple into a single `BackendVariant` the caller can
// `std::visit` over.
//
// Decision order:
//   1. if have_cuda() AND gpu_mem_fits AND allow_gpu --> CudaBackend
//   2. else                                          --> CpuBackend
//
// `allow_mpi` / `allow_mpi_gpu` are accepted and ignored.
//
// Phase 4.1 of the Minimalist ED Collapse (May 2026).
// =============================================================================

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include <ed/core/linear_operator.h>
#include <ed/matvec/backends/cpu_backend.h>

#ifdef WITH_CUDA
#  include <cuda_runtime.h>
#  include <ed/matvec/backends/cuda_backend.cuh>
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
    /// Minimum problem dimension for the GPU AUTO-promotion (Jul 2026).
    /// Below this, kernel-launch + transfer overhead makes the GPU
    /// strictly slower than the CPU -- and on shared-GPU hosts (WSL2)
    /// tiny solves dispatched to a contended device have produced
    /// silently-wrong spectra. Matches the Python-side ``dim >= 2^14``
    /// heuristic in ``qed._resolve_device``. Callers that EXPLICITLY
    /// request the GPU (``device='gpu'``) set this to 0 -- the floor
    /// gates only the automatic promotion, never an explicit choice.
    std::size_t gpu_dim_floor = (std::size_t{1} << 14);
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
    // Probed once per process. Environment-difference failures must be
    // LOUD: a wheel built against a newer CUDA toolkit than the node's
    // driver used to swallow cudaErrorInsufficientDriver here and silently
    // degrade every GPU lane to CPU (a cluster ran days of "GPU" jobs on
    // legacy drivers before anyone noticed). One clear diagnostic, then the
    // documented CPU fallback.
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
            std::fprintf(stderr,
                "[qed] CUDA DISABLED: the NVIDIA driver on this machine is "
                "too old for this build (driver API %d.%d < runtime %d.%d). "
                "Every GPU lane falls back to CPU. Fix: update the driver, "
                "or rebuild against this node's CUDA toolkit.\n",
                drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);
        }
        return false;
    }();
    return ok;
#else
    return false;
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
        // cudaMemGetInfo costs 20-100 ms per call under WSL2 (measured: 6
        // calls = 116 ms in a 6-solve benchmark). Cache the answer for one
        // second; the feasibility check only needs an order of magnitude.
        static std::size_t cached_free = 0;
        static std::chrono::steady_clock::time_point cached_at{};
        const auto now = std::chrono::steady_clock::now();
        if (cached_at == std::chrono::steady_clock::time_point{}
                || now - cached_at > std::chrono::seconds(1)) {
            std::size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
                cudaGetLastError();
                return false;
            }
            cached_free = free_bytes;
            cached_at   = now;
        }
        budget = cached_free;
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
    // Phase 2 of the "Unified CPU/GPU symmetry architecture" plan
    // (May 2026): an operator can advertise device-matvec capability
    // even when its native storage is host. `bind_cuda()` is expected
    // to lazily build a GPU mirror in that case. This unblocks the
    // `qed.solve(symmetry=..., device='gpu')` lane: the symmetry
    // sector operators are host-resident but their CudaMatVecBackend
    // mirror (lazily constructed inside `bind_cuda`) runs on the GPU.
    const bool device_mv = op_is_device || geom.supports_device_matvec;
    // gpu_dim_floor gates only the AUTO promotion: a device-resident
    // operator has already committed to the GPU, and explicit requests
    // arrive with the floor zeroed.
    const bool dim_ok = op_is_device || geom.local_dim >= c.gpu_dim_floor;
    if (device_mv && have_gpu && gpu_fits && c.allow_gpu && dim_ok) {
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

// ---------------------------------------------------------------------------
// lane_label_for<Backend>() / lane_label_from_variant(v): truthful lane
// reporting helpers.
//
// `R.backend.lane` used to be inferred from `H.geometry().is_device()` /
// `is_distributed()`. That is correct for native Operator
// instances, but it lies about every `SectorView` (streaming-symmetry /
// fixed-Sz streaming-symmetry): the view advertises `Host` memory_space
// yet lazily wires a GPU mirror via `bind_cuda_for_sector(...)`. With
// `allow_gpu=true` and `supports_device_matvec=true`, `select_backend`
// picks `CudaBackend` -- but the legacy reporter still read "cpu",
// which is what made `qed.thermal(symmetry=..., device='gpu')` look
// stuck on CPU even though the matvec ran on the device.
//
// `lane_label_for<Backend>()` is the template form (cheap, available
// inside any `solve_on<Backend>` / `thermal_on<Backend>` body).
// `lane_label_from_variant(v)` visits the variant for callers that
// already hold a `BackendVariant`. Both return one of
// {"cpu","gpu"}. Callers should set `R.backend.mpi_size`
// separately -- the label encodes the lane class, not the rank count.
//
// Phase D of the "Backend x Symmetries x Workflows" plan (May 2026).
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
