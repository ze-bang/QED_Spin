// =============================================================================
// include/ed/solvers/ftlm_ltlm_dyn.h
//
// Low-Temperature Lanczos Method (LTLM) dynamical two-point correlator.
//
//   S_{O1,O2}(omega, T) = (1/Z) sum_{n=0}^{K-1} e^{-beta E_n}
//                                 sum_m conj(<m|O1|n>) <m|O2|n>
//                                        delta(omega - (E_m - E_n))
//
// where {E_n, |n>} are the K lowest approximate eigenpairs of H (Ritz pairs
// from a single high-quality outer Lanczos sweep), and {E_m, |m>} are Ritz
// pairs of the inner Lanczos started from O2|n>/||O2|n>||.
//
// Compared to the Jaklic-Prelovsek (JP) estimator implemented in ftlm_jp.h:
//
//   JP:   w_outer = U_outer[0,n]^2  (overlap with random sample)
//         Z ~ (D/R) sum_r sum_n w_{outer,n}^r exp(-beta E_n^r)
//         Statistical error ~ 1/sqrt(R·D)  for R random samples of a D-dim space.
//
//   LTLM: w_outer = 1               (exact Ritz vector, not a projection)
//         Z = sum_{n=0}^{K-1} exp(-beta E_n)
//         Zero stochastic error; truncation error ~ exp(-beta E_K) for the
//         (K+1)-th eigenstate omitted.
//
// The LTLM is preferred when:
//   - T is low (fewer states needed to converge Z and S)
//   - K lowest eigenstates are well separated from the bulk (gap > 0)
//   - Full accuracy at T=0 is required (set K=1 -> exact GS response)
//
// For high T or large Hilbert spaces JP is more efficient. A combined
// FTLM+LTLM strategy is to use LTLM at low T and JP at high T.
//
// Cross-sector variant: provide H_outer (outer sector Hamiltonian),
//   H_inner (inner sector), dim_outer, dim_inner.  The operator O2 maps
//   dim_outer → dim_inner.  Use compute_ltlm_dynamical_correlation_cross_sector.
// =============================================================================

#pragma once

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/matvec_types.h>
#include <ed/solvers/ftlm_jp.h>  // reuse JPDynamicalResult

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ed::ltlm {

using Complex = ed::types::Complex;
using ComplexVector = std::vector<Complex>;
using MatVec = ed::types::MatVec;

// Reuse the same result type as JP so callers can use a single code-path
// for JP/LTLM with only a parameter change.
using LTLMDynamicalResult = ed::ftlm::jp::JPDynamicalResult;

/// Parameters for the LTLM double-Lanczos correlator.
struct LTLMParameters {
    /// Number of lowest eigenstates K to include in the thermal sum.
    /// Must be <= outer_krylov_dim.  Truncation error vanishes when
    /// exp(-beta * E_K) << exp(-beta * E_0).
    std::uint64_t num_lowest_states = 20;

    /// Outer Krylov dimension M_outer.  Must be >= num_lowest_states.
    /// Larger values improve Ritz accuracy but cost more memory and FLOPS.
    std::uint64_t outer_krylov_dim = 200;

    /// Inner Krylov dimension M_inner.  Controls spectral resolution.
    std::uint64_t inner_krylov_dim = 100;

    /// Lanczos convergence tolerance (used for both outer and inner).
    double tolerance = 1e-12;

    /// Full reorthogonalization of Lanczos basis vectors.
    /// Recommended for accuracy when dim is small or K is large.
    bool full_reorthogonalization = true;

    /// Reorthogonalization period when full_reorthogonalization = false.
    std::uint64_t reorth_frequency = 10;

    /// Random seed for the outer Lanczos initial vector.
    /// 0 = use std::random_device.
    std::uint64_t random_seed = 0;

    /// Pin the energy shift used in exp(-beta*(E_n - energy_shift)).
    /// 0.0 = auto-detect as the lowest outer Ritz value.
    double energy_shift = 0.0;
};

/// Intra-sector LTLM dynamical correlator (O1, O2 both map dim -> dim).
///
/// Finds K lowest eigenstates via a single outer Lanczos sweep, then for
/// each eigenstate runs inner Lanczos to compute the Lehmann sum exactly.
///
/// @param H        Hamiltonian MatVec: H|v> -> out
/// @param O1       First operator  (O1† appears in the Lehmann formula)
/// @param O2       Second operator (seeds the inner Krylov from O2|n>)
/// @param dim      Hilbert-space dimension
/// @param omega_min/max/n_omega  Frequency grid specification
/// @param betas    Inverse temperatures (can be empty for static-only)
/// @param eta      Lorentzian broadening
/// @param params   LTLM control parameters
LTLMDynamicalResult compute_ltlm_dynamical_correlation(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const LTLMParameters& params = {});

/// Cross-sector LTLM: outer sector (H_outer, dim_outer) and inner sector
/// (H_inner, dim_inner).  O2 maps dim_outer->dim_inner; O1 maps
/// dim_outer->dim_inner (with O1^dagger appearing in the formula).
///
/// Typical use: Sz=m sector as outer, Sz=m+1 as inner, O2=S^+, O1=S^+.
LTLMDynamicalResult compute_ltlm_dynamical_correlation_cross_sector(
    MatVec H_outer,
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    const LTLMParameters& params = {});

/// Supply pre-computed eigenpairs {energies, eigenstates} directly.
///
/// This is useful when eigenstates have already been obtained (e.g. from an
/// ARPACK calculation) and you want to avoid re-running the outer Lanczos.
/// The eigenstates must be normalised and mutually orthogonal.
///
/// @param eigenstates  Column-major: eigenstates[n] is the n-th eigenstate
/// @param energies     Corresponding eigenvalues E_n
LTLMDynamicalResult compute_ltlm_dynamical_correlation_from_states(
    MatVec H_inner,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim_outer,
    std::uint64_t dim_inner,
    const std::vector<ComplexVector>& eigenstates,
    const std::vector<double>& energies,
    double omega_min,
    double omega_max,
    std::uint64_t n_omega,
    const std::vector<double>& betas,
    double eta,
    std::uint64_t inner_krylov_dim = 100,
    double tolerance = 1e-12,
    bool full_reorth = true,
    std::uint64_t reorth_freq = 10);

} // namespace ed::ltlm
