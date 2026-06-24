// =============================================================================
// include/ed/solvers/ftlm_kpm.h
//
// Kernel Polynomial Method (KPM / Chebyshev moment) kernel for finite-T
// dynamical correlators.
//
// Mathematical foundation
// -----------------------
//   The dynamical correlator is:
//
//     S_{O1,O2}(ω, β) = (1/Z(β)) Σ_n  e^{-β E_n}
//                        Σ_m  <n|O1†|m><m|O2|n>  δ(ω − (E_m − E_n))
//
//   For each outer eigenstate |n⟩ with energy E_n, define:
//     |f_n⟩ = O2|n⟩   (right excitation)
//     |g_n⟩ = O1|n⟩   (left excitation; equal to f_n for self-correlation)
//
//   The local spectral density at transferred frequency ω is:
//     s_n(ω) = ⟨g_n| δ(ω + E_n − H) |f_n⟩
//
//   In Chebyshev basis on H_sc = (H − b)/a  (so that spectrum ⊂ [-1,1]):
//     s_n(ω) = [1/(π a √(1−x²))] * [g₀ Re(μ₀) + 2 Σ_{k≥1} gₖ Re(μₖ) Tₖ(x)]
//
//   where:
//     x    = (ω + E_n − b) / a       (rescaled absolute energy of excited state)
//     μₖ   = ⟨g_n|Tₖ(H_sc)|f_n⟩     (k-th Chebyshev moment)
//     gₖ   = Jackson or Lorentz kernel weight
//
//   Chebyshev recursion (stores only 3 vectors):
//     |v₀⟩ = |f_n⟩
//     |v₁⟩ = H_sc |v₀⟩
//     |vₖ⟩ = 2 H_sc |v_{k-1}⟩ − |v_{k-2}⟩
//     μₖ   = ⟨g_n|vₖ⟩
//
//   Finite-temperature accumulation:
//     S(ω, β) = (1/Z) Σ_n  e^{-β(E_n − shift)} s_n(ω)
//     Z(β)    = Σ_n  e^{-β(E_n − shift)}
//
// Energy rescaling
// ----------------
//   a = (E_max − E_min + 2*buffer) / 2
//   b = (E_max + E_min) / 2
//
//   The Chebyshev expansion is valid for any (ω + E_n) ∈ [b-a, b+a].
//   Points outside this window contribute zero spectral weight.
//
// Outer eigenstates
// -----------------
//   Three outer strategies are provided:
//
//   1. LTLM: single outer Lanczos → K lowest Ritz states.
//            Good for low T (K states are the thermally-relevant ones).
//
//   2. from_states: pre-computed exact eigenstates (e.g. from full diag).
//            Exact result in the limit M → ∞.
//
//   3. JP-KPM: stochastic outer (JP random samples) with Boltzmann weighting.
//              Coming in a later revision; use LTLM for now.
//
// Kernel choice
// -------------
//   Jackson kernel: gₖᴶ = [(M+1−k) cos(πk/(M+1)) + sin(πk/(M+1))cot(π/(M+1))] / (M+1)
//     Guarantees positive-definite S(ω) ≥ 0.  Effective Gaussian broadening
//     η_eff ≈ π * bandwidth / M.
//
//   Lorentz kernel: gₖᴸ = sinh(λ(1−k/M)) / sinh(λ)
//     Produces Lorentzian broadening with η = a λ / M.  May go slightly negative
//     near band edges but gives a cleaner Lorentzian lineshape.
//
// Result format
// -------------
//   KPMResult is layout-compatible with ed::ftlm::jp::JPDynamicalResult:
//   both use row-major [t * n_omega + i] for spectral_real/imag.
// =============================================================================

#pragma once

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/matvec_types.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ed::kpm {

using Complex = ed::types::Complex;
using ComplexVector = std::vector<Complex>;
using MatVec = ed::types::MatVec;

// ---------------------------------------------------------------------------
// KPM control parameters
// ---------------------------------------------------------------------------

struct KPMParameters {
    /// Number of Chebyshev moments M.  Effective frequency resolution:
    ///   Jackson kernel: η ≈ π * (E_max − E_min) / M
    int num_moments = 512;

    /// Number of lowest outer Ritz states K (LTLM mode).
    int num_lowest_states = 100;

    /// Outer Lanczos Krylov dimension (LTLM mode).
    int outer_krylov_dim = 300;

    /// Whether to use the Jackson kernel (positive-definite).
    /// If false, use the Lorentz kernel.
    bool use_jackson_kernel = true;

    /// Lorentz kernel decay parameter λ (used only if use_jackson_kernel=false).
    double lorentz_lambda = 4.0;

    /// Full reorthogonalization in outer Lanczos.
    bool full_reorthogonalization = true;

    /// Reorthogonalization frequency (ignored if full_reorthogonalization=true).
    int reorth_frequency = 10;

    /// Lanczos convergence tolerance.
    double tolerance = 1e-12;

    /// Random seed (0 = use hardware random device).
    std::uint64_t random_seed = 0;

    /// Energy shift for Boltzmann weights (0 = auto-detect from lowest Ritz).
    double energy_shift = 0.0;

    /// Fractional buffer added beyond [E_min, E_max] for the Chebyshev domain.
    /// E.g. 0.05 means the domain is [E_min − 0.05*BW, E_max + 0.05*BW].
    double spectral_bound_buffer = 0.05;
};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

struct KPMResult {
    /// Output frequency grid (transfer energies ω = E_m − E_n).
    std::vector<double> frequencies;

    /// Inverse temperatures.
    std::vector<double> betas;

    /// Spectral function S(ω, β) in row-major layout: [t * n_omega + i].
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;

    /// Integrated spectral weight ∫S(ω,β)dω at each β (should equal SSSF static).
    std::vector<double> static_correlator;

    /// Partition function Z(β) at each β.
    std::vector<double> partition_function;

    /// Lowest energy eigenstate estimate (lowest outer Ritz value or given E[0]).
    double ground_state_estimate = 0.0;

    /// Energy shift applied in Boltzmann weights.
    double energy_shift_used = 0.0;

    /// KPM energy window: H_sc = (H − kpm_b) / kpm_a.
    double kpm_a = 1.0;
    double kpm_b = 0.0;

    /// Number of Chebyshev moments actually used.
    int num_moments_used = 0;

    /// True if Jackson kernel was applied; false for Lorentz.
    bool jackson_kernel_used = true;

    /// Number of outer states processed (K).
    std::uint64_t total_outer_states = 0;
};

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

// compute_kpm_ltlm (random-state self-window KPM driver) was retired in the
// Gen-1 Lanczos-unification cleanup (Jun 2026): no callers anywhere. Use
// compute_kpm_ltlm_from_states (the orchestrator's live KPM path).

/// Compute S_{O1,O2}(ω, β) from pre-computed exact eigenstates.
///
/// Uses all K provided states as outer states, with Chebyshev inner expansion.
/// This gives the exact result in the limit M → ∞ (arbitrarily sharp lines).
///
/// @param H_inner   Hamiltonian in the inner sector (same as outer if not cross).
/// @param O1, O2    Operators (map dim_outer → dim_inner).
/// @param eigenstates Pre-computed eigenstate vectors (normalised, ascending E).
/// @param energies    Eigenvalues corresponding to eigenstates.
KPMResult compute_kpm_ltlm_from_states(
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<ComplexVector>& eigenstates,
    const std::vector<double>& energies,
    double omega_min,
    double omega_max,
    int n_omega,
    const std::vector<double>& betas,
    const KPMParameters& params = {});

/// Combine per-sector KPMResults with Z-weighted recombination.
/// Same convention as combine_sector_results in ftlm_jp.h:
///   Z_eff_s = sector_dims[s] * Z_partial_s
KPMResult combine_sector_kpm(
    const std::vector<KPMResult>& per_sector,
    const std::vector<std::uint64_t>& sector_dims);

} // namespace ed::kpm
