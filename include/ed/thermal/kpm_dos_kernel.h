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
#include <string>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/solvers/kpm_dos.h>

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

// Wave B (Full unified-interface collapse, May 2026): the driver in
// `src/solvers/cpu/kpm_dos.cpp::compute_kpm_dos` is CPU-host today.
// Guard against silent miscalibration when the orchestrator hands
// the kernel a device backend.
template <typename Backend, typename MatvecFn>
KpmDosResult kpm_dos_kernel(const Backend& /*backend*/,
                            MatvecFn&&     apply_H,
                            std::size_t    local_n,
                            std::uint64_t  /*global_n*/,
                            const KpmDosOptions& opts)
{
    static_assert(
        std::is_same_v<Backend, ed::matvec::CpuBackend>,
        "kpm_dos_kernel: only CpuBackend is supported today. See "
        "ftlm_kernel.h for the wave-b-thermal context.");

    ed::kpm_dos::KPMDOSParameters params;
    params.num_moments             = opts.num_moments;
    params.num_random_vectors      = opts.num_random_vectors;
    params.num_quadrature_nodes    = opts.num_quadrature_nodes;
    params.spectral_bounds_krylov  = opts.spectral_bounds_krylov;
    params.spectral_bound_buffer   = opts.spectral_bound_buffer;
    params.use_jackson_kernel      = opts.use_jackson_kernel;
    params.lorentz_lambda          = opts.lorentz_lambda;
    params.random_seed             = opts.random_seed;

    ed::kpm_dos::MatVec legacy_H =
        [&apply_H](const Complex* in, Complex* out, int n) {
            apply_H(in, out, static_cast<std::size_t>(n));
        };

    const auto legacy = ed::kpm_dos::compute_kpm_dos(
        legacy_H,
        static_cast<std::uint64_t>(local_n),
        opts.betas, opts.dos_energies, params);

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

}  // namespace ed::thermal
