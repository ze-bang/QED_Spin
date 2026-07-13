#pragma once
// =============================================================================
// include/ed/thermal/tpq_thermo.h -- TPQ trajectory -> ThermodynamicData
// aggregation (Stage 11b relocation of the last surviving TPQ.{h,cpp}
// function into the thermal kernel family it serves).
//
// Takes the per-sample (beta_k, E_k, var_k) trajectories DIRECTLY (no
// HDF5/text round-trip) and interpolates onto the caller's temperature
// grid. Called by the unified ``ed::workflows::thermal`` orchestrator for
// the mTPQ lane (both the backend-templated kernel in ``mtpq_kernel.h``
// and the fp32 GPU lane in ``mtpq_f32.h`` emit these trajectories).
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <ed/core/thermal_types.h>  // ThermodynamicData

namespace ed::thermal {

/**
 * @brief Aggregate per-sample TPQ trajectories into ThermodynamicData
 *        on a user-supplied temperature grid (in-memory variant).
 *
 * Math: for each target temperature T (target_temperatures[t]), linearly
 * interpolate every sample's (beta_k, E_k, var_k) at beta_target = 1/T,
 * Welford-average across samples that bracket the target. Specific
 * heat C_v = beta^2 * <var>.
 *
 * Samples whose trajectories don't bracket a given target beta are
 * clamped to their nearest trajectory endpoint (beta=0 baseline on the
 * warm side, the asymptotic deepest-beta iterate on the cold side).
 *
 * Returns an empty ``ThermodynamicData`` (all vectors zero-length)
 * when ``sample_*.empty()`` or ``target_temperatures.empty()``.
 *
 * @param sample_inv_temps    Per-sample beta_k trajectory
 * @param sample_energies     Per-sample E_k trajectory (same length)
 * @param sample_variances    Per-sample var_k trajectory (same length)
 * @param target_temperatures Temperature grid for the output
 * @param hilbert_dim         Hilbert-space dimension D of the (sub)space
 *                            the trajectories live in. When ``> 1`` the
 *                            entropy and free energy are reconstructed in
 *                            ABSOLUTE form via thermodynamic integration
 *                            ``ln Z(beta) = ln(D) - \int_0^beta <E> dbeta'``,
 *                            so ``S(T->inf) -> ln(D)`` and ``F`` carries the
 *                            correct dimensional normalisation. This is what
 *                            makes the per-sector ``F_s`` usable as a
 *                            Boltzmann weight in
 *                            ``combine_sector_thermodynamics`` (U(1)/Sz and
 *                            spatial recombination). When ``0`` (default) the
 *                            legacy zero-baseline integration ``S(T_min)=0``
 *                            is used (kept for backwards compatibility).
 */
inline ThermodynamicData compute_tpq_thermo_from_trajectories(
    const std::vector<std::vector<double>>& sample_inv_temps,
    const std::vector<std::vector<double>>& sample_energies,
    const std::vector<std::vector<double>>& sample_variances,
    const std::vector<double>& target_temperatures,
    double hilbert_dim = 0.0
) {
    ThermodynamicData thermo{};

    const std::size_t num_samples = sample_inv_temps.size();
    const std::size_t num_T = target_temperatures.size();
    if (num_samples == 0 || num_T == 0) {
        return thermo;
    }
    // Shape sanity: every sample carries its three arrays in lockstep.
    if (sample_energies.size() != num_samples
        || sample_variances.size() != num_samples) {
        return thermo;
    }

    // Pre-sort every sample's trajectory by beta so the per-target
    // interpolation can use a single linear scan (kernels emit in
    // step-monotone order; for mTPQ beta is also step-monotone since
    // E_k drifts slowly, but defend against rounding-driven inversions).
    std::vector<std::vector<std::array<double, 3>>> sorted(num_samples);
    for (std::size_t s = 0; s < num_samples; ++s) {
        const auto& betas = sample_inv_temps[s];
        const auto& Es    = sample_energies[s];
        const auto& vars  = sample_variances[s];
        if (betas.size() != Es.size() || Es.size() != vars.size()) {
            continue;  // skip malformed sample; cf. shape-mismatch policy
        }
        sorted[s].reserve(betas.size());
        for (std::size_t k = 0; k < betas.size(); ++k) {
            if (std::isfinite(betas[k]) && std::isfinite(Es[k])
                && std::isfinite(vars[k])) {
                sorted[s].push_back({betas[k], Es[k], vars[k]});
            }
        }
        std::sort(sorted[s].begin(), sorted[s].end(),
                  [](const auto& a, const auto& b) { return a[0] < b[0]; });
    }

    // Per-target Welford running mean across samples.
    std::vector<double> energy_mean(num_T, 0.0);
    std::vector<double> cv_mean(num_T, 0.0);
    std::vector<std::uint64_t> counts(num_T, 0);

    // Per-sample contribution to every target T:
    //   * In-bracket beta:   linear interpolation of (E_k, var_k).
    //   * target_beta > beta_max_traj (asked colder than the trajectory
    //     reaches): clamp to the last trajectory point. For mTPQ this
    //     is the asymptotic ground-state-projected state, which is the
    //     correct extrapolation for T < T_min_traj. For canonical trajectories this is
    //     the deepest beta the Taylor evolution reached.
    //   * target_beta < beta_min_traj (asked warmer than beta=0): use
    //     the first trajectory point (which is the beta=0 baseline,
    //     <H> on the random seed -- the high-T limit).
    //
    // This per-sample boundary clamp replaces the previous per-bin
    // "nearest covered bin" hole-filler, which produced incorrect E(T)
    // at T_min when the trajectory bracketed (β_max_traj) was just
    // below β_target = 1/T_min and the aggregator silently copied the
    // value from the next-warmer bin.
    bool any_covered = false;
    for (std::size_t s = 0; s < num_samples; ++s) {
        const auto& tr = sorted[s];
        if (tr.empty()) continue;
        any_covered = true;
        const double beta_min = tr.front()[0];
        const double beta_max = tr.back()[0];
        for (std::size_t t = 0; t < num_T; ++t) {
            const double T = target_temperatures[t];
            const double target_beta = 1.0 / std::max(T, 1e-300);

            double E_t, var_t;
            if (target_beta <= beta_min) {
                // Above-bracket warm: clamp to first point (beta=0 baseline).
                E_t   = tr.front()[1];
                var_t = tr.front()[2];
            } else if (target_beta >= beta_max) {
                // Below-bracket cold: clamp to last point (asymptotic E).
                E_t   = tr.back()[1];
                var_t = tr.back()[2];
            } else {
                // In-bracket: linear interpolation. Kernel emits in
                // step-monotone order so the trajectory size is bounded
                // by ``max_iter``; linear scan beats upper_bound's
                // branch-mispredict cost on the small-N arrays.
                std::size_t i_low = 0;
                for (std::size_t i = 0; i + 1 < tr.size(); ++i) {
                    if (tr[i][0] <= target_beta && tr[i + 1][0] >= target_beta) {
                        i_low = i;
                        break;
                    }
                }
                const std::size_t i_high = i_low + 1;
                const double beta_l = tr[i_low][0];
                const double beta_h = tr[i_high][0];
                const double alpha = (beta_h > beta_l)
                    ? (target_beta - beta_l) / (beta_h - beta_l)
                    : 0.0;
                E_t   = tr[i_low][1] * (1.0 - alpha) + tr[i_high][1] * alpha;
                var_t = tr[i_low][2] * (1.0 - alpha) + tr[i_high][2] * alpha;
            }
            const double Cv_t = target_beta * target_beta * var_t;

            ++counts[t];
            energy_mean[t] += (E_t  - energy_mean[t]) / counts[t];
            cv_mean[t]     += (Cv_t - cv_mean[t])     / counts[t];
        }
    }

    if (!any_covered) {
        return thermo;  // no usable sample
    }

    std::vector<double> entropy(num_T, 0.0);
    std::vector<double> free_energy(num_T, 0.0);

    if (hilbert_dim > 1.0) {
        // ABSOLUTE thermodynamics via thermodynamic integration of the
        // energy over inverse temperature:
        //
        //   ln Z(beta) = ln(D) - \int_0^beta <E>(beta') dbeta'
        //   F(beta)    = -ln Z(beta) / beta
        //   S(beta)    = beta * <E>(beta) + ln Z(beta)
        //
        // This anchors the high-T entropy correctly (S(T->inf) -> ln(D),
        // since at beta=0 the integral vanishes and S = ln(D)) and gives
        // the ABSOLUTE free energy. Crucially, the absolute F is what makes
        // each sector's free energy usable as a Boltzmann weight inside
        // ``combine_sector_thermodynamics``; the previous zero-baseline
        // integration dropped the per-sector ln(dim_s) constant and so
        // produced a systematically biased Sz / spatial recombination.
        //
        // The [0, beta_first] head segment must be anchored at the TRUE
        // beta=0 energy <E>(0) = Tr(H)/D, NOT at the warmest target's
        // energy. Using E(beta_first) for the head over-counts the
        // integral by ~0.5*(E(0) - E(beta_first))*beta_first, which shows
        // up as a constant ~1% offset in S(T) across the whole curve
        // (and a matching error in F). Every mTPQ sample already records
        // the beta=0 baseline as its first trajectory point
        // (<rand|H|rand> -> Tr(H)/D as samples accumulate), so we recover
        // <E>(0) for free from the sorted fronts -- a moment-exact anchor
        // with no extra matvec.
        const double lnD = std::log(hilbert_dim);
        double E_inf = 0.0;
        std::uint64_t e_inf_cnt = 0;
        for (std::size_t s = 0; s < num_samples; ++s) {
            if (!sorted[s].empty()) {
                E_inf += sorted[s].front()[1];  // smallest-beta (~0) point
                ++e_inf_cnt;
            }
        }
        if (e_inf_cnt > 0) {
            E_inf /= static_cast<double>(e_inf_cnt);
        } else {
            E_inf = energy_mean[0];
        }

        // Visit target temperatures in ascending-beta (descending-T) order
        // without assuming the caller's grid ordering.
        std::vector<std::size_t> order(num_T);
        for (std::size_t i = 0; i < num_T; ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) {
                      const double ba = 1.0
                          / std::max(target_temperatures[a], 1e-300);
                      const double bb = 1.0
                          / std::max(target_temperatures[b], 1e-300);
                      return ba < bb;
                  });

        double integral  = 0.0;
        double prev_beta  = 0.0;
        double prev_E     = E_inf;  // exact <E>(beta=0) = Tr(H)/D anchor
        for (std::size_t r = 0; r < num_T; ++r) {
            const std::size_t t = order[r];
            const double beta = 1.0
                / std::max(target_temperatures[t], 1e-300);
            const double E_t = energy_mean[t];
            integral += 0.5 * (prev_E + E_t) * (beta - prev_beta);
            const double lnZ = lnD - integral;
            free_energy[t] = -lnZ / std::max(beta, 1e-300);
            entropy[t]     = beta * E_t + lnZ;
            prev_beta = beta;
            prev_E    = E_t;
        }
    } else {
        // Legacy: trapezoidal integration of C_v / T -> entropy (zero
        // baseline at the coldest target). F = E - T S. Kept for callers
        // that do not supply the Hilbert dimension.
        for (std::size_t i = 1; i < num_T; ++i) {
            const double T1 = target_temperatures[i - 1];
            const double T2 = target_temperatures[i];
            const double Cv1 = cv_mean[i - 1];
            const double Cv2 = cv_mean[i];
            if (T1 > 0.0 && T2 > 0.0) {
                entropy[i] = entropy[i - 1]
                           + 0.5 * (T2 - T1) * (Cv1 / T1 + Cv2 / T2);
            } else {
                entropy[i] = entropy[i - 1];
            }
        }
        for (std::size_t i = 0; i < num_T; ++i) {
            free_energy[i] = energy_mean[i]
                           - target_temperatures[i] * entropy[i];
        }
    }

    thermo.temperatures   = target_temperatures;
    thermo.energy         = std::move(energy_mean);
    thermo.specific_heat  = std::move(cv_mean);
    thermo.entropy        = std::move(entropy);
    thermo.free_energy    = std::move(free_energy);
    return thermo;
}

}  // namespace ed::thermal
