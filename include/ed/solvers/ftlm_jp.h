// =============================================================================
// include/ed/solvers/ftlm_jp.h
//
// Jaklic-Prelovsek finite-temperature dynamical / static two-point correlation
// functions via *double* Lanczos.
//
//   S_{O1,O2}(omega, T) = (1/Z) sum_{n,m} e^{-beta E_n}
//                                <n|O1^dagger|m><m|O2|n>
//                                delta(omega - (E_m - E_n))
//
// JP estimator (Jaklic & Prelovsek, PRB 49, 5065 (1994); Adv. Phys. 49, 1
// (2000)):
//
//   For each random Gaussian sample |r> in the target Hilbert sector:
//     1. Outer Lanczos on |r> in H               -> Krylov V_outer (N x M_o),
//                                                   tridiagonal -> Ritz pairs
//                                                   {E_n^(r), |n^(r)>}.
//     2. For each n with non-negligible Boltzmann weight:
//          phi_n   = O2 |n^(r)>;  norm_n = ||phi_n||
//          chi_n   = O1 |n^(r)>            (skipped when O1 = O2)
//          Inner Lanczos on phi_n/norm_n in H    -> Krylov V_inner (N x M_i),
//                                                   Ritz pairs {E_m, |m>}.
//          <m|O2|n>  = norm_n * U_inner[0, m]
//          <m|O1|n>  = sum_l conj(U_inner[l, m]) <v_l^inner|chi_n>
//          contribution_m = U_outer[0,n]^2 * conj(<m|O1|n>) * <m|O2|n>
//                                          * Lorentzian(omega - (E_m - E_n))
//
//   Z(beta)   = (D / R) sum_r sum_n e^{-beta E_n^(r)} U_outer[0,n]^2
//   S(omega)  = (D / R) sum_r sum_{n,m}  contribution_m * e^{-beta E_n^(r)}
//
//   The (D/R) factors cancel in S/Z, so we only track the unnormalised
//   accumulators and divide at the end.
//
// vs. the legacy `compute_dynamical_correlation_*` family (which builds a
// single Krylov from O |psi> and naively inserts Boltzmann factors against
// inner Ritz values), this implementation keeps the *initial* energy E_n in
// the Boltzmann weight and the *final* energy E_m in the Lorentzian, so
// the resulting S(omega, T) satisfies the detailed-balance condition
// S(-omega, T) = e^{-beta omega} S(omega, T) up to statistical noise.
//
// Phase A scope: intra-sector, CPU only. Cross-sector operators (Phase B)
// reuse the same kernel via an `OperatorAcrossSectors` adapter that swaps
// the inner Hamiltonian / basis to the partner sector.
// =============================================================================

#pragma once

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/matvec_types.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ed::ftlm::jp {

using Complex = ed::types::Complex;
using ComplexVector = std::vector<Complex>;

/// Matrix-vector product callable: apply(in, out, n).
using MatVec = ed::types::MatVec;

/// Knobs controlling the JP double-Lanczos sampler.
struct JPParameters {
    /// Outer Krylov dimension (Lanczos steps from each random sample).
    /// Determines how well the e^{-beta H} trace is sampled by each |r>.
    std::uint64_t outer_krylov_dim = 100;

    /// Inner Krylov dimension (Lanczos steps from each O2 |n^(r)> seed).
    /// Determines the spectral resolution of the Lehmann sum.
    std::uint64_t inner_krylov_dim = 100;

    /// Number of independent Gaussian samples |r>. Stat. error ~ 1/sqrt(R).
    std::uint64_t num_samples = 20;

    /// Drop outer Ritz pairs whose Boltzmann weight at the *coldest*
    /// requested temperature is below this threshold (relative to the
    /// largest weight in the sample). Skips wasted inner Lanczos passes.
    /// Set to 0 to disable.
    double outer_boltzmann_cutoff = 1.0e-12;

    /// Numerical Lanczos breakdown tolerance (used by both Krylov passes).
    double tolerance = 1.0e-12;

    /// Use full reorthogonalization in both Lanczos passes. Strongly
    /// recommended for spectral function work; the linear cost of the
    /// outer/inner Krylov reorth is dwarfed by the matvecs.
    bool full_reorthogonalization = true;

    /// Periodic-reorth fallback when full_reorthogonalization is false.
    std::uint64_t reorth_frequency = 10;

    /// Random seed (0 = std::random_device).
    std::uint64_t random_seed = 0;

    /// If non-zero, override the global energy reference subtracted from
    /// the Boltzmann factors. When zero, the kernel uses its running
    /// estimate of the ground-state energy across samples seen so far.
    /// Setting this explicitly (e.g. to a converged GS energy) makes Z
    /// numerically comparable across runs.
    double energy_shift = 0.0;
};

/// Result payload from the JP sampler.
struct JPDynamicalResult {
    /// Frequency grid (length n_omega).
    std::vector<double> frequencies;

    /// Inverse-temperature grid (length n_T). Stored as beta = 1/T to
    /// avoid the T = 0 singularity; T = 0 limit is reachable as
    /// beta -> inf.
    std::vector<double> betas;

    /// Real part of S_{O1,O2}(omega, T). Row-major:
    ///   spectral_real[t * n_omega + i] = Re S(omega_i, T_t).
    std::vector<double> spectral_real;

    /// Imaginary part of S_{O1,O2}(omega, T). Same layout. Identically
    /// zero when O1 = O2 and the operator is Hermitian.
    std::vector<double> spectral_imag;

    /// Frequency-integrated equal-time correlator <O1^dagger O2>(T)
    /// (length n_T). For self-correlation this is the static structure
    /// factor S(Q, T) up to convention.
    std::vector<Complex> static_correlator;

    /// Per-temperature partition function estimate (length n_T). Useful
    /// for ratio diagnostics; only relative magnitudes are meaningful
    /// because of the (D/R) prefactor.
    std::vector<double> partition_function;

    /// Best estimate of the ground-state energy seen across all samples
    /// (lowest outer Ritz value). Useful as a convergence check.
    double ground_state_estimate = 0.0;

    /// Running mean of the energy shift actually used.
    double energy_shift_used = 0.0;

    /// Number of (sample, outer_n) pairs that contributed to the sum
    /// (after the boltzmann_cutoff filter at the coldest temperature).
    std::uint64_t inner_lanczos_passes = 0;

    /// Number of random samples actually used (samples whose initial
    /// vector survived all internal sanity checks).
    std::uint64_t total_samples = 0;
};

/// Compute S_{O1,O2}(omega, T) via the JP double-Lanczos estimator.
///
/// `H_apply` must be the Hamiltonian on the *outer* sector (where the
/// random samples live). For an *intra-sector* correlator (Phase A), the
/// inner Lanczos uses the same H. Cross-sector wiring (Phase B) swaps in
/// a partner-sector H via the `H_inner_apply` overload below.
///
/// `O2_apply` is applied to outer Ritz vectors |n^(r)> in the outer sector
/// to seed the inner Lanczos. `O1_apply` is also applied to |n^(r)> to
/// build matrix elements <m|O1|n>; pass the same callable as `O2_apply`
/// to compute the self-correlation S_{O,O}.
///
/// `dim` is the outer-sector Hilbert dimension. `omega_min/max/n` define
/// the frequency grid. `betas` is the inverse-temperature list.
/// `eta` is the Lorentzian broadening (same units as omega).
JPDynamicalResult compute_dynamical_correlation(
    MatVec H_apply,
    MatVec O1_apply,
    MatVec O2_apply,
    std::uint64_t dim,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const JPParameters& params = {}
);

/// Cross-sector overload: outer Lanczos on `H_outer_apply` (sector alpha),
/// inner Lanczos on `H_inner_apply` (sector beta = O2 alpha). The
/// operators carry vectors between sectors and may be rectangular:
///   O2: dim_outer -> dim_inner
///   O1: dim_outer -> dim_inner
/// `dim_inner` is the partner-sector Hilbert dimension.
JPDynamicalResult compute_dynamical_correlation_cross_sector(
    MatVec H_outer_apply,
    MatVec H_inner_apply,
    MatVec O1_apply,           // dim_outer -> dim_inner
    MatVec O2_apply,           // dim_outer -> dim_inner
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const JPParameters& params = {}
);

/// Save a JPDynamicalResult to an HDF5 file under group `/jp/` (one
/// dataset per field). No-op when `path` is empty.
void save_jp_dynamical_result(const JPDynamicalResult& result,
                              const std::string& path);

/// Combine per-sector JP results into a full-symmetry-resolved spectrum
/// using the Z-weighted recipe
///   Z_total(T)            = sum_alpha Z_alpha(T)
///   S_total(omega, T)     = sum_alpha S_alpha(omega, T) * Z_alpha(T) / Z_total(T)
/// `sector_dim_weights` is the dimension of each sector (used as a
/// multiplicative correction when the per-sector result was computed
/// without the (D_alpha / R) prefactor).
JPDynamicalResult combine_sector_results(
    const std::vector<JPDynamicalResult>& per_sector,
    const std::vector<std::uint64_t>& sector_dims
);

} // namespace ed::ftlm::jp
