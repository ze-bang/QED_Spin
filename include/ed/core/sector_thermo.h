#pragma once

// =============================================================================
// ed/core/sector_thermo.h
//
// Sector-level recombination of finite-temperature thermodynamic data.
//
// MOTIVATION (matvec-unification audit follow-up)
// -----------------------------------------------
// The streaming-symmetry kernel built by
// ``ed::make_streaming_symmetry_operator`` iterates over symmetry
// sectors, calling ``ed::workflows::thermal(...)`` once per sector.
// For ground-state methods (LANCZOS, BLOCK_LANCZOS, KRYLOV_SCHUR, ...)
// it is enough to collect the per-sector eigenvalues into a global
// pool and sort. For finite-temperature methods (FTLM, LTLM, KPM_DOS,
// mTPQ) we *also* need to recombine thermodynamic observables
// across sectors --- otherwise the result for ``ed::workflows::thermal``
// with ``method=FTLM`` and ``use_symmetry=true`` would be just the
// thermodynamics of the last sector touched, which is wrong.
//
// This header provides a single canonical combiner that operates on the
// generic ``ThermodynamicData`` payload populated by every finite-T
// solver. The math mirrors ``combine_ftlm_sector_results`` in
// ``src/solvers/cpu/ftlm.cpp`` but is decoupled from the FTLMResults
// envelope so it works uniformly for LTLM / KPM_DOS.
//
// MATH
// ----
// Z_total(beta) = Σ_s Z_s(beta)
//   where Z_s = exp(-beta F_s)        --- F_s is the sector free energy
//
// To avoid overflow we work with shifted partition functions
//   Z_s_shifted = exp(-beta (F_s - F_ref))     F_ref = min_s F_s
// so that Z_total = exp(-beta F_ref) Σ_s Z_s_shifted.
//
// The combined observables follow from the canonical mixture rule
//   <E>_total       = Σ_s w_s <E>_s
//   <E^2>_total     = Σ_s w_s <E^2>_s       (reconstructed via Cv = β²Var[E])
//   F_total         = F_ref - T ln(Σ_s Z_s_shifted)
//   S_total         = β(<E>_total - F_total)
//   Cv_total        = β²(<E^2>_total - <E>_total^2)
//
// where w_s = Z_s_shifted / Σ_s Z_s_shifted.
//
// This treatment is invariant under a uniform additive shift of all
// sector free energies (overall zero of energy / partition function).
// =============================================================================

#include <ed/core/ed_types.h>      // DiagonalizationMethod
#include <ed/core/thermal_types.h>  // ThermodynamicData

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ed {
namespace core {

/**
 * @brief Recombine per-sector ``ThermodynamicData`` payloads into a
 *        single full-system ``ThermodynamicData`` using free-energy
 *        Z-weighting.
 *
 * @param sector_thermo  Per-sector thermodynamic blocks. Each must share
 *                       the same ``temperatures`` grid and have populated
 *                       ``energy``, ``specific_heat``, ``free_energy``.
 * @param sector_dims    Per-sector dimensions. Currently informational
 *                       only --- the recombination weight is derived
 *                       entirely from F_s (which already absorbs the
 *                       sector dimension via the ln(D_s) entropy term
 *                       inside each solver). Provided for symmetry with
 *                       the audit's `combine_sector_kpm` signature and
 *                       so that callers do not have to discard the data.
 * @return Combined ``ThermodynamicData`` for the full Hilbert space.
 *
 * Throws ``std::invalid_argument`` if the inputs are empty, mis-sized,
 * or have inconsistent temperature grids.
 */
inline ThermodynamicData
combine_sector_thermodynamics(const std::vector<ThermodynamicData>& sector_thermo,
                              const std::vector<std::uint64_t>& sector_dims) {
    if (sector_thermo.empty()) {
        throw std::invalid_argument(
            "ed::core::combine_sector_thermodynamics: "
            "sector_thermo must be non-empty.");
    }
    if (sector_thermo.size() != sector_dims.size()) {
        throw std::invalid_argument(
            "ed::core::combine_sector_thermodynamics: "
            "sector_thermo.size() must equal sector_dims.size().");
    }

    const std::size_t n_sectors = sector_thermo.size();
    const std::size_t n_temps   = sector_thermo.front().temperatures.size();
    if (n_temps == 0) {
        throw std::invalid_argument(
            "ed::core::combine_sector_thermodynamics: "
            "temperatures grid is empty.");
    }

    for (std::size_t s = 1; s < n_sectors; ++s) {
        if (sector_thermo[s].temperatures.size() != n_temps) {
            throw std::invalid_argument(
                "ed::core::combine_sector_thermodynamics: "
                "sectors disagree on temperature-grid length.");
        }
    }

    ThermodynamicData combined;
    combined.temperatures = sector_thermo.front().temperatures;
    combined.energy.assign(n_temps, 0.0);
    combined.specific_heat.assign(n_temps, 0.0);
    combined.entropy.assign(n_temps, 0.0);
    combined.free_energy.assign(n_temps, 0.0);

    // Suppress unused-parameter warning while keeping the signature
    // symmetric with `combine_sector_kpm` (which does use the dims
    // explicitly). Sector dimensions enter the math indirectly through
    // the per-sector ln(D_s) inside each solver's free_energy.
    (void) sector_dims;

    for (std::size_t t = 0; t < n_temps; ++t) {
        const double T = combined.temperatures[t];
        if (!(T > 0.0)) {
            throw std::invalid_argument(
                "ed::core::combine_sector_thermodynamics: "
                "non-positive temperature in grid.");
        }
        const double beta = 1.0 / T;

        // Step 1: find the most stable sector at this T (smallest F).
        double F_ref = sector_thermo[0].free_energy[t];
        for (std::size_t s = 1; s < n_sectors; ++s) {
            const double F_s = sector_thermo[s].free_energy[t];
            if (std::isfinite(F_s) && F_s < F_ref) F_ref = F_s;
        }

        // Step 2: shifted partition function and Z-weights.
        std::vector<double> Z_shifted(n_sectors, 0.0);
        double Z_total = 0.0;
        for (std::size_t s = 0; s < n_sectors; ++s) {
            const double F_s = sector_thermo[s].free_energy[t];
            const double dF  = F_s - F_ref;
            double Z_s = std::exp(-beta * dF);
            if (!std::isfinite(Z_s) || Z_s < 0.0) Z_s = 0.0;
            Z_shifted[s] = Z_s;
            Z_total += Z_s;
        }

        if (!(Z_total > 1e-300) || !std::isfinite(Z_total)) {
            // All sectors numerically degenerate or invalid at this T.
            // Best-effort fallback: drop down to the most stable sector.
            combined.free_energy[t]   = F_ref;
            combined.energy[t]        = sector_thermo[0].energy[t];
            combined.specific_heat[t] = 0.0;
            combined.entropy[t]       = 0.0;
            continue;
        }

        combined.free_energy[t] = F_ref - T * std::log(Z_total);

        // Step 3: mixture-rule observables.
        double E_total  = 0.0;
        double E2_total = 0.0;
        for (std::size_t s = 0; s < n_sectors; ++s) {
            const double w_s = Z_shifted[s] / Z_total;
            const double E_s = sector_thermo[s].energy[t];
            const double C_s = sector_thermo[s].specific_heat[t];
            // From Cv_s = β²(<E²>_s - <E>_s²): <E²>_s = C_s/β² + E_s²
            const double E2_s = C_s / (beta * beta) + E_s * E_s;
            E_total  += w_s * E_s;
            E2_total += w_s * E2_s;
        }

        combined.energy[t]        = E_total;
        combined.specific_heat[t] = beta * beta * (E2_total - E_total * E_total);
        // Thermodynamic identity: S = β(E - F).
        combined.entropy[t]       = beta * (E_total - combined.free_energy[t]);
    }

    return combined;
}

/**
 * @brief Degeneracy-weighted recombination (Stage 12f, SU(2) rollout).
 *
 * ``Z = sum_s g_s * Z_s`` -- the per-S thermal formulation, where each
 * spin-S tower is sampled once in its highest-weight sector and enters
 * with multiplicity g_s = 2S + 1. Implemented by the exact identity
 * ``g Z <=> F -> F - T ln g`` (with ``S -> S + ln g`` so the per-sector
 * copies stay thermodynamically consistent) followed by the canonical
 * shifted-F mixture above. Energies and specific heats are intensive to
 * the g-scaling and pass through unchanged.
 */
inline ThermodynamicData
combine_sector_thermodynamics(const std::vector<ThermodynamicData>& sector_thermo,
                              const std::vector<std::uint64_t>& sector_dims,
                              const std::vector<double>& degeneracy) {
    if (degeneracy.size() != sector_thermo.size()) {
        throw std::invalid_argument(
            "ed::core::combine_sector_thermodynamics: degeneracy.size() "
            "must equal sector_thermo.size().");
    }
    std::vector<ThermodynamicData> weighted = sector_thermo;
    for (std::size_t s = 0; s < weighted.size(); ++s) {
        const double g = degeneracy[s];
        if (!(g > 0.0)) {
            throw std::invalid_argument(
                "ed::core::combine_sector_thermodynamics: degeneracies "
                "must be positive.");
        }
        const double ln_g = std::log(g);
        for (std::size_t t = 0; t < weighted[s].temperatures.size(); ++t) {
            const double T = weighted[s].temperatures[t];
            weighted[s].free_energy[t] -= T * ln_g;
            weighted[s].entropy[t]     += ln_g;
        }
    }
    return combine_sector_thermodynamics(weighted, sector_dims);
}

/**
 * @brief Decide whether ``method`` should trigger sector thermodynamics
 *        recombination in the streaming-symmetry kernel.
 *
 * Returns true exactly for the finite-temperature methods that populate
 * ``EDResults::thermo_data`` --- FTLM, LTLM, KPM_DOS, mTPQ.
 * The streaming-symmetry kernel uses this to decide whether to invoke
 * ``combine_sector_thermodynamics`` after the per-sector loop.
 */
inline bool method_produces_sector_thermo(DiagonalizationMethod method) {
    switch (method) {
        case DiagonalizationMethod::FTLM:
        case DiagonalizationMethod::OFTLM:
        case DiagonalizationMethod::LTLM:
        case DiagonalizationMethod::KPM_DOS:
        case DiagonalizationMethod::mTPQ:
            return true;
        default:
            return false;
    }
}

}  // namespace core
}  // namespace ed
