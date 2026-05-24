#pragma once
// =============================================================================
// include/ed/thermal/ltlm_kernel.h
//
// LTLM (Low Temperature Lanczos Method) kernel —
// `template<Backend, MatvecFn>`. CPU body delegates to the existing
// `low_temperature_lanczos` in `src/solvers/cpu/ltlm.cpp`. GPU / MPI
// specialisations land alongside `CudaBackend` / `MpiBackend`.
//
// Algorithm:
//   * Find ground state |0>.
//   * Build a Krylov subspace from |0>.
//   * Evaluate Z, <O>, free energy with the Lehmann representation of
//     the projected H eigenpairs.
//
// LTLM is strictly more accurate than FTLM at low temperatures (it
// captures the ground state exactly) and is the natural finite-T
// driver below the FTLM/LTLM crossover.
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
#include <ed/solvers/ltlm.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct LtlmOptions {
    std::size_t num_samples         = 1;
    std::size_t krylov_dim          = 200;
    std::size_t ground_state_krylov = 100;
    std::vector<double> betas;
    std::uint64_t random_seed = 0;
    std::string output_dir;
};

struct LtlmResult {
    std::vector<double> betas;
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> heat_capacity;
    std::vector<double> entropy;
    double ground_state_energy = 0.0;
};

namespace detail {

inline LtlmResult to_ltlm_result(const ::LTLMResults& legacy,
                                 const std::vector<double>& betas) {
    LtlmResult out;
    out.betas                = betas;
    out.partition_function   = legacy.thermo_data.Z_sample;
    out.energy               = legacy.thermo_data.energy;
    out.heat_capacity        = legacy.thermo_data.specific_heat;
    out.entropy              = legacy.thermo_data.entropy;
    out.ground_state_energy  = legacy.ground_state_energy;
    return out;
}

}  // namespace detail

// Wave B (Full unified-interface collapse, May 2026): the inner
// driver in `src/solvers/cpu/ltlm.cpp::low_temperature_lanczos` is
// CPU-host today. Guard against silent miscalibration when the
// orchestrator hands the kernel a device backend.
template <typename Backend, typename MatvecFn>
LtlmResult ltlm_kernel(const Backend&  /*backend*/,
                       MatvecFn&&      apply_H,
                       std::size_t     local_n,
                       std::uint64_t   /*global_n*/,
                       const LtlmOptions& opts)
{
    static_assert(
        std::is_same_v<Backend, ed::matvec::CpuBackend>,
        "ltlm_kernel: only CpuBackend is supported today. See "
        "ftlm_kernel.h for the wave-b-thermal context.");

    LTLMParameters params;
    params.krylov_dim           = static_cast<std::uint64_t>(opts.krylov_dim);
    params.ground_state_krylov  = static_cast<std::uint64_t>(opts.ground_state_krylov);
    params.num_samples          = static_cast<std::uint64_t>(opts.num_samples);
    params.random_seed          = opts.random_seed;

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

    const auto legacy = ::low_temperature_lanczos(
        legacy_H,
        static_cast<std::uint64_t>(local_n),
        params, temp_min, temp_max, num_bins, nullptr, opts.output_dir);

    return detail::to_ltlm_result(legacy, opts.betas);
}

}  // namespace ed::thermal
