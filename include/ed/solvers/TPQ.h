// TPQ.h - Thermal Pure Quantum state support (post-processing surface)
//
// The legacy monolithic TPQ drivers (``microcanonical_tpq`` /
// ``canonical_tpq`` with their 25+-parameter signatures, the SS_rand /
// HDF5 sidecar writers, the per-site observable helpers, and the
// imaginary-time Taylor/Krylov evolvers) were deleted in the
// debt-cleanup sweep (Jul 2026): they had zero callers. The live TPQ
// path is the backend-templated kernels in
// ``include/ed/thermal/{tpq,mtpq}_kernel.h`` (plus the fp32 GPU
// lane in ``include/ed/thermal/mtpq_f32.h``), driven by the
// orchestrator's ``ed::workflows::thermal``. This header keeps only:
//
//   * ``tpq_per_sample_seed`` (re-export from tpq_seeding.h) -- the
//     historical rank-deterministic seeding recipe shared by every
//     TPQ lane, and
//   * ``compute_tpq_thermo_from_trajectories`` -- the single
//     aggregator that turns per-sample (beta_k, E_k, var_k)
//     trajectories into ``ThermodynamicData``.

#pragma once

#include <cstdint>
#include <vector>

#include <ed/core/thermal_types.h>       // ThermodynamicData
#include <ed/solvers/tpq_seeding.h>

using ed::tpq_per_sample_seed;

/**
 * @brief Aggregate per-sample TPQ trajectories into ThermodynamicData
 *        on a user-supplied temperature grid (in-memory variant).
 *
 * Consumed by the unified mTPQ kernel in
 * ``include/ed/thermal/mtpq_kernel.h`` and the fp32 GPU mTPQ
 * lane, which emit per-step (beta_k, E_k, var_k) trajectories directly
 * (no HDF5 / text-file round trip). The orchestrator's mTPQ
 * branches in ``ed::workflows::thermal`` call this to populate
 * ``ThermalResult::thermo``.
 *
 * Math: for each target temperature T (target_temperatures[t]), linearly
 * interpolate every sample's (beta_k, E_k, var_k) at beta_target = 1/T,
 * Welford-average across samples that bracket the target. Specific
 * heat C_v = beta^2 * <var>. Entropy via trapezoidal integration of
 * C_v / T from cold to hot. Free energy F = E - T * S.
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
ThermodynamicData compute_tpq_thermo_from_trajectories(
    const std::vector<std::vector<double>>& sample_inv_temps,
    const std::vector<std::vector<double>>& sample_energies,
    const std::vector<std::vector<double>>& sample_variances,
    const std::vector<double>& target_temperatures,
    double hilbert_dim = 0.0
);
