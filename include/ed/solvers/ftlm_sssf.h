// =============================================================================
// include/ed/solvers/ftlm_sssf.h
//
// Static Spectral Structure Factor (SSSF) at finite temperature:
//
//   S_{O1,O2}(T) = (1/Z) Tr[ e^{-beta H} O1^dagger O2 ]
//                = (1/Z) sum_n  e^{-beta E_n} <n|O1^dagger O2|n>
//
// This is the zero-frequency (omega-integrated) limit of the dynamical
// correlator S_{O1,O2}(omega,T).  Computing it via the JP/LTLM dynamical
// kernels wastes an inner-Lanczos pass whose cost scales as
// O(inner_krylov * dim) per outer Ritz state.  The SSSF fast path avoids
// the inner Lanczos entirely:
//
//   <n|O1^dagger O2|n> = <O1 ñ_n | O2 ñ_n>
//                      = dot(O1 |ñ_n>, O2 |ñ_n>)   (one dot-product)
//
// For the self-correlation case (O1 = O2 = O):
//   <n|O^dagger O|n> = ||O|ñ_n>||^2                 (one norm)
//
// Memory cost: O(dim)  (two extra vectors O1|n> and O2|n>)
// Time cost:   O(outer_krylov * dim * num_samples)   (outer Lanczos only)
//
// JP (stochastic) variant
// -----------------------
//   For each random sample |r>:
//     Outer Lanczos -> Ritz {E_n^(r), U_outer^(r)}
//     For each n: w = U_outer[0,n]^2
//       phi_n = O2 |ñ_n>
//       chi_n = O1 |ñ_n>  (or phi_n if O1=O2)
//       S_partial[beta] += w * exp(-beta*(E_n-shift)) * Re<chi_n|phi_n>
//       Z_partial[beta] += w * exp(-beta*(E_n-shift))
//   S(T) = (D/R) S_partial / ((D/R) Z_partial) = S_partial / Z_partial
//
// LTLM (deterministic) variant
// ----------------------------
//   Single outer Lanczos -> K Ritz states (lowest K eigenstates)
//   For each n in 0..K-1:
//     phi_n = O2 |ñ_n>
//     chi_n = O1 |ñ_n>
//     S[beta] += exp(-beta*(E_n-shift)) * Re<chi_n|phi_n>
//     Z[beta] += exp(-beta*(E_n-shift))
//   S(T) = S / Z
//
// Cross-sector SSSF
// -----------------
//   When O2 maps from the outer sector (H_outer, dim_outer) to an inner
//   sector (H_inner, dim_inner), the formula becomes:
//     S(T) = (1/Z) sum_n exp(-beta*E_n^outer) <O1 ñ_n | O2 ñ_n>
//   Z is still computed from the outer eigenstates.
//   Provide `compute_sssf_jp_cross_sector` and
//   `compute_sssf_ltlm_cross_sector`.
// =============================================================================

#pragma once

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/matvec_types.h>
#include <ed/solvers/ftlm_jp.h>    // JPParameters
#include <ed/solvers/ftlm_ltlm_dyn.h>   // LTLMParameters

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ed::sssf {

using Complex = ed::types::Complex;
using ComplexVector = std::vector<Complex>;
using MatVec = ed::types::MatVec;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

struct SSSFResult {
    /// Inverse temperatures at which S(T) was computed.
    std::vector<double> betas;

    /// Static correlator Re[S_{O1,O2}(T)] at each beta (length = betas.size()).
    std::vector<double> S_static_real;

    /// Im[S_{O1,O2}(T)] at each beta.
    std::vector<double> S_static_imag;

    /// Partition function Z(beta) = sum_n w_n exp(-beta(E_n-shift)) at each beta.
    std::vector<double> partition_function;

    /// Estimate of the ground-state energy (lowest outer Ritz value seen).
    double ground_state_estimate = 0.0;

    /// Energy shift applied in Boltzmann weights.
    double energy_shift_used = 0.0;

    /// Number of outer-Lanczos random samples used (JP) or 1 (LTLM).
    std::uint64_t total_samples = 0;

    /// Number of outer Ritz states processed.
    std::uint64_t total_outer_ritz_processed = 0;
};

/// Combine per-sector SSSFResults with Z-weighted recombination.
///
/// For each sector s with sector dimension d_s, the effective partition
/// function is Z_eff_s = d_s * Z_partial_s.  The combined static
/// correlator is the Z_eff-weighted average.
///
/// @param per_sector    One SSSFResult per symmetry sector.
/// @param sector_dims   Physical Hilbert-space dimension of each sector.
SSSFResult combine_sector_sssf(
    const std::vector<SSSFResult>& per_sector,
    const std::vector<std::uint64_t>& sector_dims);

// ---------------------------------------------------------------------------
// JP (stochastic) SSSF
// ---------------------------------------------------------------------------

/// Compute the static correlator S_{O1,O2}(T) via JP random sampling.
///
/// Cheaper than the full dynamical JP kernel: no inner Lanczos, no
/// frequency grid.  Each outer Ritz state contributes one dot-product
/// <O1 ñ_n | O2 ñ_n>.
///
/// @param H      Hamiltonian MatVec
/// @param O1     First operator (O1^dagger appears in the formula)
/// @param O2     Second operator (seeds the inner dot-product)
/// @param dim    Hilbert-space dimension
/// @param betas  Inverse temperatures
/// @param params JP control parameters (outer_krylov_dim, num_samples, ...)
///               inner_krylov_dim is ignored.
SSSFResult compute_sssf_jp(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const ed::ftlm::jp::JPParameters& params = {});

/// Cross-sector JP SSSF: outer sector (H_outer, dim_outer),
/// inner sector (H_inner, dim_inner).
SSSFResult compute_sssf_jp_cross_sector(
    MatVec H_outer,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const ed::ftlm::jp::JPParameters& params = {});

// ---------------------------------------------------------------------------
// LTLM (deterministic) SSSF
// ---------------------------------------------------------------------------

/// Compute the static correlator S_{O1,O2}(T) via deterministic LTLM.
///
/// Finds K lowest Ritz states from a single outer Lanczos, then computes
/// S(T) = (1/Z) sum_{n=0}^{K-1} exp(-beta*E_n) <O1 ñ_n | O2 ñ_n>.
///
/// @param H      Hamiltonian MatVec
/// @param O1     First operator
/// @param O2     Second operator
/// @param dim    Hilbert-space dimension
/// @param betas  Inverse temperatures
/// @param params LTLM control parameters
SSSFResult compute_sssf_ltlm(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const ed::ltlm::LTLMParameters& params = {});

/// LTLM SSSF from pre-computed eigenstates.
///
/// @param O1, O2        Operators (map dim_outer -> dim_inner)
/// @param eigenstates   Pre-computed |n> vectors (normalised, mutually orthogonal)
/// @param energies      Corresponding eigenvalues E_n (ascending order)
SSSFResult compute_sssf_ltlm_from_states(
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<ComplexVector>& eigenstates,
    const std::vector<double>& energies,
    const std::vector<double>& betas);

/// Cross-sector LTLM SSSF.
SSSFResult compute_sssf_ltlm_cross_sector(
    MatVec H_outer,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<double>& betas,
    const ed::ltlm::LTLMParameters& params = {});

} // namespace ed::sssf
