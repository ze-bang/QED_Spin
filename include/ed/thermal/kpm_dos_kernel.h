#pragma once
// =============================================================================
// include/ed/thermal/kpm_dos_kernel.h
//
// KPM-DOS kernel — `template<Backend, MatvecFn>`. CPU body delegates
// to `ed::kpm_dos::compute_kpm_dos` in `src/solvers/cpu/kpm_dos.cpp`.
// Backend interface (axpy/dot/norm) is already abstracted by the CPU
// implementation; GPU and MPI specialisations land alongside
// `CudaBackend` and `MpiBackend`.
//
// Algorithm:
//   * Estimate spectral bounds via outer Lanczos.
//   * Rescale H so spec(H_sc) ⊂ [-1, 1].
//   * Compute first `num_moments` Chebyshev moments via Hutchinson
//     stochastic trace.
//   * Apply Jackson kernel (default) and integrate against
//     Boltzmann factor on a Chebyshev–Gauss quadrature grid to obtain
//     Z(beta), E(beta), C(beta), S(beta).
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/solvers/kpm_dos.h>

#ifdef WITH_CUDA
#  include <cuComplex.h>
#  include <ed/gpu/kpm_dos_gpu.cuh>
// Forward declarations let us reference CudaBackend in
// `std::is_same_v` without pulling its full definition (which lives in
// `cuda_backend.cuh`, a `.cuh` header that should only be included
// from nvcc-compiled translation units). The orchestrator's TU
// already drags in the full definition via `select_backend.h`, so the
// `if constexpr (std::is_same_v<...>)` branch below resolves cleanly
// there. From other consumers (CPU-only callers) the constraint
// trivially evaluates to `false` and the device branch is discarded.
namespace ed { namespace matvec { class CudaBackend; } }
#endif

namespace ed::thermal {

using Complex = std::complex<double>;

struct KpmDosOptions {
    int  num_moments              = 2048;
    int  num_random_vectors       = 20;
    int  num_quadrature_nodes     = 0;        ///< 0 -> auto = 2 * num_moments
    int  spectral_bounds_krylov   = 150;
    double spectral_bound_buffer  = 0.05;
    bool   use_jackson_kernel     = true;
    double lorentz_lambda         = 4.0;
    std::uint64_t random_seed     = 0;

    std::vector<double> betas;
    std::vector<double> dos_energies;

    /// Wave B3 (May 2026): caller-supplied spectral bounds. NaN means
    /// "estimate via internal Lanczos". When BOTH are finite the
    /// kernel skips its 150-iteration spectral-bound Lanczos. The
    /// streaming-symmetry binding sets these once and reuses across
    /// every per-sector call (the spectral bounds are global to H, so
    /// per-sector re-estimation is purely wasted work).
    double e_min_override = std::numeric_limits<double>::quiet_NaN();
    double e_max_override = std::numeric_limits<double>::quiet_NaN();
};

struct KpmDosResult {
    std::vector<double> betas;
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;
    std::vector<double> dos_grid_energies;
    std::vector<double> dos_grid_values;
    double e_min_estimate = 0.0;
    double e_max_estimate = 0.0;
};

namespace detail {

/// Convert a legacy `KPMDOSResult` (the host-side struct used by both
/// the CPU and GPU drivers) into the thermal-facade `KpmDosResult`.
inline KpmDosResult from_legacy_kpm_result(
    const ed::kpm_dos::KPMDOSResult& legacy) {
    KpmDosResult out;
    out.betas               = legacy.betas;
    out.partition_function  = legacy.partition_function;
    out.energy              = legacy.energy;
    out.specific_heat       = legacy.specific_heat;
    out.entropy             = legacy.entropy;
    out.free_energy         = legacy.free_energy;
    out.dos_grid_energies   = legacy.dos_grid_energies;
    out.dos_grid_values     = legacy.dos_grid_values;
    out.e_min_estimate      = legacy.e_min_estimate;
    out.e_max_estimate      = legacy.e_max_estimate;
    return out;
}

/// Populate a `KPMDOSParameters` (the legacy struct, used by both the
/// CPU and GPU drivers) from the facade `KpmDosOptions`. Shared by
/// the CPU and CUDA branches so the two lanes stay in lockstep.
inline ed::kpm_dos::KPMDOSParameters to_legacy_params(
    const KpmDosOptions& opts) {
    ed::kpm_dos::KPMDOSParameters params;
    params.num_moments            = opts.num_moments;
    params.num_random_vectors     = opts.num_random_vectors;
    params.num_quadrature_nodes   = opts.num_quadrature_nodes;
    params.spectral_bounds_krylov = opts.spectral_bounds_krylov;
    params.spectral_bound_buffer  = opts.spectral_bound_buffer;
    params.use_jackson_kernel     = opts.use_jackson_kernel;
    params.lorentz_lambda         = opts.lorentz_lambda;
    params.random_seed            = opts.random_seed;
    params.e_min_override         = opts.e_min_override;
    params.e_max_override         = opts.e_max_override;
    return params;
}

}  // namespace detail

/// KPM-DOS kernel facade.
///
/// Phase E1 of the "Backend x Symmetries x Workflows" plan (May 2026):
/// closes the previous CPU-only `static_assert` gap. The body now
/// dispatches on the backend type:
///   * `CpuBackend`: delegates to ``ed::kpm_dos::compute_kpm_dos``
///     (the host-side driver in src/solvers/cpu/kpm_dos.cpp).
///   * `CudaBackend`: delegates to
///     ``ed::kpm_dos::compute_kpm_dos_gpu_with_matvec`` (the
///     device-resident Chebyshev + Hutchinson driver in
///     src/solvers/gpu/kpm_dos_gpu.cu), wrapping the supplied
///     `apply_H` callable as a device-pointer matvec. The legacy
///     `GPUOperator*` entry point is preserved for CLI consumers.
template <typename Backend, typename MatvecFn>
KpmDosResult kpm_dos_kernel(const Backend& /*backend*/,
                            MatvecFn&&     apply_H,
                            std::size_t    local_n,
                            std::uint64_t  /*global_n*/,
                            const KpmDosOptions& opts)
{
    const auto params = detail::to_legacy_params(opts);

    if constexpr (std::is_same_v<Backend, ed::matvec::CpuBackend>) {
        // CPU lane: host-side compute_kpm_dos. apply_H is a host-pointer
        // callable; we just forward into the legacy MatVec signature.
        ed::kpm_dos::MatVec legacy_H =
            [&apply_H](const Complex* in, Complex* out, int n) {
                apply_H(in, out, static_cast<std::size_t>(n));
            };

        const auto legacy = ed::kpm_dos::compute_kpm_dos(
            legacy_H,
            static_cast<std::uint64_t>(local_n),
            opts.betas, opts.dos_energies, params);
        return detail::from_legacy_kpm_result(legacy);
    }
#ifdef WITH_CUDA
    else if constexpr (std::is_same_v<Backend, ed::matvec::CudaBackend>) {
        // CUDA lane: apply_H is a device-pointer matvec (see
        // bind_cuda / bind_cuda_for_sector). Wrap it as a
        // cuDoubleComplex callable and delegate to the GPU driver.
        // The Complex <-> cuDoubleComplex cast is layout-safe per
        // CUDA's documented contract.
        ed::kpm_dos::DeviceMatVec device_matvec =
            [&apply_H](const cuDoubleComplex* d_in,
                       cuDoubleComplex* d_out,
                       int n) {
                apply_H(
                    reinterpret_cast<const Complex*>(d_in),
                    reinterpret_cast<Complex*>(d_out),
                    static_cast<std::size_t>(n));
            };

        const auto legacy = ed::kpm_dos::compute_kpm_dos_gpu_with_matvec(
            std::move(device_matvec),
            static_cast<std::uint64_t>(local_n),
            opts.betas, opts.dos_energies, params);
        return detail::from_legacy_kpm_result(legacy);
    }
#endif
    else {
        // MpiBackend / MpiCudaBackend are not yet wired through the
        // KPM-DOS lane: the per-rank Lanczos for the spectral bound
        // and the Hutchinson loop both need a cross-rank reduction
        // that the CPU/CUDA drivers do not currently implement.
        // Mirrors the analogous FTLM/LTLM guard above.
        throw std::runtime_error(
            "kpm_dos_kernel: distributed backends are not yet "
            "supported. Pin BackendConstraints::allow_mpi = false "
            "to route through the CPU/CUDA lanes.");
    }
}

}  // namespace ed::thermal
