// =============================================================================
// src/observables/ftlm_cross_irrep_kernel.cpp
//
// Host entry point of the cross-sector FTLM dynamical kernel (the estimator itself is
// ftlm_dynamics_kernel.h, shared with the GPU path), and the cross-sector recombination.
// =============================================================================

#include <ed/observables/ftlm_cross_irrep_kernel.h>
#include <ed/observables/ftlm_dynamics_kernel.h>

#include <ed/matvec/backends/cpu_backend.h>

#include <cmath>
#include <limits>
#include <vector>

namespace ed::observables {

FtlmCrossIrrepSectorResult ftlm_cross_irrep_kernel_one_sector(
    const std::function<void(const Complex*, Complex*, int)>& H_src,
    const std::function<void(const Complex*, Complex*, int)>& H_dst,
    const std::function<void(const Complex*, Complex*, int)>& O_apply,
    std::size_t                       dim_src,
    std::size_t                       dim_dst,
    const std::vector<double>&        temperatures,
    const std::vector<double>&        omega_grid,
    const FtlmCrossIrrepOptions&      opts)
{
    auto wrap = [](const std::function<void(const Complex*, Complex*, int)>& f) {
        return [&f](const Complex* in, Complex* out, std::size_t n) { f(in, out, static_cast<int>(n)); };
    };
    return ftlm_dynamics_kernel(ed::matvec::default_cpu_backend(), wrap(H_src), wrap(H_dst), wrap(O_apply),
                                dim_src, dim_dst, temperatures, omega_grid, opts);
}

DynamicalSpectraMerged combine_sector_dynamical_spectra(
    const std::vector<FtlmCrossIrrepSectorResult>& sector_results,
    const std::vector<double>&                     temperatures,
    std::size_t                                    num_omega)
{
    DynamicalSpectraMerged out;
    if (sector_results.empty() || num_omega == 0) {
        for (double T : temperatures) {
            out.S_real[T] = std::vector<double>(num_omega, 0.0);
            out.S_imag[T] = std::vector<double>(num_omega, 0.0);
        }
        return out;
    }

    // The per-sector S_real / S_imag / Z accumulators are already
    // dim_src-weighted (see kernel return). However, each sector
    // carries its own ``E_min`` reference inside the exp(-beta E)
    // factor, so we apply an F-shift trick analogous to
    // ``ed::core::combine_sector_thermodynamics``: pick the global
    // minimum E_min across sectors, multiply each sector's numerator
    // AND denominator by exp(-beta * (E_min^sector - E_min_global)),
    // then sum and divide. This makes the float exponents stable
    // even when one sector's GS sits far above the global GS.

    // Find global E_min:
    double E_min_global = std::numeric_limits<double>::infinity();
    for (const auto& sr : sector_results) {
        if (sr.dim_src == 0 || sr.samples_done == 0) continue;
        if (sr.E_min < E_min_global) E_min_global = sr.E_min;
    }
    if (!std::isfinite(E_min_global)) {
        // Every sector empty / failed; return zeros.
        for (double T : temperatures) {
            out.S_real[T] = std::vector<double>(num_omega, 0.0);
            out.S_imag[T] = std::vector<double>(num_omega, 0.0);
        }
        return out;
    }

    for (double T : temperatures) {
        const double beta = 1.0 / T;
        std::vector<double> num_real(num_omega, 0.0);
        std::vector<double> num_imag(num_omega, 0.0);
        double              den = 0.0;
        for (const auto& sr : sector_results) {
            if (sr.dim_src == 0 || sr.samples_done == 0) continue;
            const double sec_E_min = sr.E_min;
            const double shift     = std::exp(
                -beta * (sec_E_min - E_min_global));
            auto sR = sr.S_real.find(T);
            auto sI = sr.S_imag.find(T);
            auto sZ = sr.Z.find(T);
            if (sR == sr.S_real.end() ||
                sI == sr.S_imag.end() ||
                sZ == sr.Z.end()) continue;
            const auto& sec_real = sR->second;
            const auto& sec_imag = sI->second;
            const double sec_Z   = sZ->second;
            if (sec_real.size() != num_omega ||
                sec_imag.size() != num_omega) continue;
            for (std::size_t iw = 0; iw < num_omega; ++iw) {
                num_real[iw] += shift * sec_real[iw];
                num_imag[iw] += shift * sec_imag[iw];
            }
            den += shift * sec_Z;
        }
        if (den > 1e-300) {
            const double inv = 1.0 / den;
            for (std::size_t iw = 0; iw < num_omega; ++iw) {
                num_real[iw] *= inv;
                num_imag[iw] *= inv;
            }
        }
        out.S_real[T] = std::move(num_real);
        out.S_imag[T] = std::move(num_imag);
    }
    return out;
}

}  // namespace ed::observables
