#pragma once
// =============================================================================
// include/ed/thermal/ftlm_kernel.h
//
// Phase 5 (minimalist refactor) — `ftlm_kernel<Backend, MatvecFn>`: thin
// inline facade over the existing `finite_temperature_lanczos` CPU
// implementation in `src/solvers/cpu/ftlm.cpp`. The Backend interface
// already abstracts axpy/dot/norm, so the CPU specialisation here is
// the canonical body; GPU and MPI specialisations will land alongside
// `CudaBackend` / `MpiBackend` in Phases 7-8.
//
// Algorithm:
//   * Draw `R` Gaussian random unit vectors |r>.
//   * For each |r> run a length-`M` Lanczos.
//   * Compute partition function and observables via
//        Z(beta)    = sum_r <r|e^{-beta H}|r>
//        <O>(beta)  = (1/Z) sum_r <r|e^{-beta H} O|r>
//     using the Ritz-value Lehmann representation of the Krylov
//     projection.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/solvers/ftlm.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct FtlmOptions {
    std::size_t num_samples  = 40;
    std::size_t krylov_dim   = 100;
    std::vector<double> betas;           ///< inverse-temperature grid (positive)
    std::uint64_t random_seed = 0;
    std::string output_dir;
};

struct FtlmResult {
    std::vector<double> betas;
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> heat_capacity;
    std::vector<double> entropy;
};

namespace detail {

inline FtlmResult to_ftlm_result(const ::FTLMResults& legacy,
                                 const std::vector<double>& betas) {
    FtlmResult out;
    out.betas              = betas;
    out.partition_function = legacy.thermo_data.Z_sample;
    out.energy             = legacy.thermo_data.energy;
    out.heat_capacity      = legacy.thermo_data.specific_heat;
    out.entropy            = legacy.thermo_data.entropy;
    return out;
}

}  // namespace detail

/**
 * @brief Run FTLM using the supplied `Backend` linalg primitives and an
 *        `apply_H(in, out, n)` matvec callable.
 *
 * Today the CPU body delegates to `::finite_temperature_lanczos`. The
 * `apply_H` callable allocates HOST-side scratch vectors inside the
 * legacy body, so paired with a device-only Backend (CudaBackend /
 * MpiCudaBackend) it would silently miscalibrate (host vectors handed
 * to a device-pointer matvec). We guard against that here: only
 * `CpuBackend` (and the bandwidth-equivalent `MpiBackend` when
 * `apply_H` ferries host buffers) is supported, and the kernel throws
 * loudly when paired with anything else.
 *
 * Wave B (Full unified-interface collapse, May 2026): full delegation
 * inversion (moving the 200+ LOC FTLM driver from
 * `src/solvers/cpu/ftlm.cpp` into this header and replacing the inner
 * Lanczos loop with `ed::krylov::lanczos_kernel<Backend>`) is a
 * dedicated work item tracked separately. The explicit Backend guard
 * here keeps the facade honest until the inversion lands. mTPQ and
 * cTPQ (see `mtpq_kernel.h` / `ctpq_kernel.h`) are already fully
 * backend-templated through `tpq_kernel<Backend>`.
 */
template <typename Backend, typename MatvecFn>
FtlmResult ftlm_kernel(const Backend&  /*backend*/,
                       MatvecFn&&      apply_H,
                       std::size_t     local_n,
                       std::uint64_t   /*global_n*/,
                       const FtlmOptions& opts)
{
    static_assert(
        std::is_same_v<Backend, ed::matvec::CpuBackend>,
        "ftlm_kernel: only CpuBackend is supported today. The "
        "templated FTLM kernel body is tracked under wave-b-thermal "
        "(see ed/thermal/ftlm_kernel.h comments). For the GPU lane, "
        "construct a GPUOperator and use the legacy "
        "compute_dynamical_correlation entry point until Wave B "
        "lands.");

    FTLMParameters params;
    params.krylov_dim   = static_cast<std::uint64_t>(opts.krylov_dim);
    params.num_samples  = static_cast<std::uint64_t>(opts.num_samples);
    params.random_seed  = opts.random_seed;

    const double temp_min = opts.betas.empty()
        ? 0.01 : 1.0 / *std::max_element(opts.betas.begin(), opts.betas.end());
    const double temp_max = opts.betas.empty()
        ? 100.0 : 1.0 / *std::min_element(opts.betas.begin(), opts.betas.end());
    const std::uint64_t num_bins =
        opts.betas.empty() ? 32u : static_cast<std::uint64_t>(opts.betas.size());

    std::function<void(const Complex*, Complex*, int)> legacy_H =
        [&apply_H](const Complex* in, Complex* out, int n) {
            apply_H(in, out, static_cast<std::size_t>(n));
        };

    const auto legacy = ::finite_temperature_lanczos(
        legacy_H,
        static_cast<std::uint64_t>(local_n),
        params, temp_min, temp_max, num_bins, opts.output_dir);

    return detail::to_ftlm_result(legacy, opts.betas);
}

}  // namespace ed::thermal
