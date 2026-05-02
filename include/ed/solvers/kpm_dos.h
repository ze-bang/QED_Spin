// =============================================================================
// include/ed/solvers/kpm_dos.h
//
// Kernel Polynomial Method for the *density of states* of a Hermitian operator,
// with downstream finite-temperature thermodynamic observables.
//
// Mathematical foundation
// -----------------------
//   ρ(E) = Tr δ(E - H) = Σ_n δ(E - E_n)
//
//   Rescale  H_sc = (H - b)/a   so that spec(H_sc) ⊂ [-1, 1]:
//     a = (E_max - E_min)/2 * (1 + buffer)
//     b = (E_max + E_min)/2
//
//   Stochastic Chebyshev moments (Hutchinson trace):
//     μ_k = Tr T_k(H_sc)
//         ≈ (D / R) Σ_{r=1}^R ⟨r| T_k(H_sc) |r⟩
//
//   For each random vector |r⟩, three-term recursion gives ⟨r|T_k(H_sc)|r⟩
//   with one sparse mat-vec per moment (3 vectors of memory).
//
//   Reconstruction (rescaled DOS, x = (E - b)/a, x ∈ [-1, 1]):
//     a · ρ(E) = (1 / (π √(1 - x²))) [g_0 μ_0 + 2 Σ_{k=1}^{M-1} g_k μ_k T_k(x)]
//
//   where g_k is the Jackson or Lorentz kernel coefficient.
//
// Thermodynamics via Chebyshev–Gauss quadrature
// ---------------------------------------------
//   On the Chebyshev nodes x_i = cos((i + 0.5) π / N_E), i = 0..N_E-1,
//   the √(1 - x²) factor cancels the quadrature weight π/N_E, giving
//   bias-free numerical integration of any smooth f(E):
//
//     ∫_{kpm_lo}^{kpm_hi} ρ(E) f(E) dE
//        = (1 / N_E) Σ_i f(b + a x_i) [g_0 μ_0 + 2 Σ_{k≥1} g_k μ_k T_k(x_i)]
//
//   Plug in f(E) = e^{-β(E - shift)}, E e^{-β(E - shift)}, E² e^{-β(E - shift)}
//   to get Z(β), ⟨E⟩, ⟨E²⟩ on a grid of β at no extra mat-vec cost.
//
// Why this scales to N ≈ 20
// -------------------------
//   * Cost = R · M sparse mat-vecs.  Typical R = 20, M = 2048 → ~4×10⁴ matvecs.
//   * Bias from kernel smoothing: O(BW / M) Gaussian (Jackson) — irrelevant for
//     thermal integrals because they are smooth-against-ρ(E).
//   * Variance from Hutchinson trace: O(1 / √(R · D)).  Decreases with the
//     Hilbert-space dimension, i.e. *better* for larger clusters.
//   * Memory: 3 complex vectors of length D + the M Chebyshev moments.
//
// Result format
// -------------
//   KPMDOSResult contains:
//     - Chebyshev moments μ_k (after kernel weighting, ready to consume).
//     - The spectral domain (kpm_a, kpm_b) so users can rebuild ρ(E) themselves.
//     - A thermodynamic table (Z, E, C, S, F) on the requested β-grid.
//     - Optional reconstructed DOS on a user-provided E-grid.
// =============================================================================

#pragma once

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/matvec_types.h>

#include <cstdint>
#include <vector>

namespace ed::kpm_dos {

using Complex = ed::types::Complex;
using ComplexVector = std::vector<Complex>;
using MatVec = ed::types::MatVec;

// ---------------------------------------------------------------------------
// Control parameters
// ---------------------------------------------------------------------------

struct KPMDOSParameters {
    /// Number of Chebyshev moments M.  Effective DOS resolution
    /// (Jackson kernel) η ≈ π * (E_max − E_min) / M.
    int num_moments = 2048;

    /// Number of random Hutchinson samples R.
    int num_random_vectors = 20;

    /// Number of Chebyshev–Gauss quadrature nodes used for thermodynamics.
    /// Must be at least num_moments to integrate the kernel-smoothed DOS
    /// exactly; default 2*M is comfortable.
    int num_quadrature_nodes = 0;  // 0 = auto = 2*num_moments

    /// Spectral-bound estimator: outer Lanczos Krylov dimension.
    /// 100–200 is plenty for tight, well-padded bounds.
    int spectral_bounds_krylov = 150;

    /// Fractional buffer added beyond the estimated [E_min, E_max] window
    /// (avoids the √(1-x²) singularity at the band edges).
    double spectral_bound_buffer = 0.05;

    /// Use Jackson kernel (true, default, positive-definite) or Lorentz.
    bool use_jackson_kernel = true;

    /// Lorentz kernel decay parameter λ (used only if use_jackson_kernel=false).
    double lorentz_lambda = 4.0;

    /// Lanczos / numerical tolerance.
    double tolerance = 1e-12;

    /// Full reorthogonalization in spectral-bound Lanczos.
    bool full_reorthogonalization = true;

    /// Reorthogonalization frequency (ignored if full_reorthogonalization=true).
    int reorth_frequency = 10;

    /// Random seed (0 = use std::random_device).
    std::uint64_t random_seed = 0;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct KPMDOSResult {
    /// Inverse temperatures (mirror of input).
    std::vector<double> betas;

    /// Z(β), E(β), C(β), S(β), F(β) at each β.
    /// Ordering of values matches `betas`.
    std::vector<double> partition_function;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;

    /// Optional reconstructed DOS on the user-provided E-grid (may be empty).
    std::vector<double> dos_grid_energies;
    std::vector<double> dos_grid_values;

    /// Kernel-weighted Chebyshev moments g_k μ_k (for downstream re-use).
    std::vector<double> moments_weighted;

    /// Raw (un-weighted) Chebyshev moments μ_k.  Useful for diagnostics.
    std::vector<double> moments_raw;

    /// KPM rescaling: H_sc = (H − kpm_b) / kpm_a; spec(H) ⊂ [b - a, b + a].
    double kpm_a = 1.0;
    double kpm_b = 0.0;

    /// Estimated extreme eigenvalues (from spectral-bound Lanczos).
    double e_min_estimate = 0.0;
    double e_max_estimate = 0.0;

    /// Energy reference subtracted in Boltzmann factors for numerical stability
    /// (typically e_min_estimate).  Z, E, F as reported are in the *physical*
    /// reference frame — the shift has already been undone.
    double energy_shift_used = 0.0;

    /// Hilbert-space dimension D (for Hutchinson normalisation).
    std::uint64_t hilbert_dim = 0;

    /// Number of moments and samples actually used.
    int num_moments_used = 0;
    int num_random_vectors_used = 0;

    /// Whether Jackson kernel was applied.
    bool jackson_kernel_used = true;
};

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

/// Compute Chebyshev DOS moments + thermodynamics for a Hermitian H.
///
/// @param H              Matrix-vector product for H.  Must be Hermitian.
/// @param dim            Hilbert-space dimension.
/// @param betas          Inverse temperatures at which to report Z, E, C, S, F.
/// @param dos_energies   Optional energy grid for DOS reconstruction; pass an
///                       empty vector to skip DOS reconstruction.
/// @param params         Control parameters.
KPMDOSResult compute_kpm_dos(
    MatVec H,
    std::uint64_t dim,
    const std::vector<double>& betas,
    const std::vector<double>& dos_energies,
    const KPMDOSParameters& params = {});

}  // namespace ed::kpm_dos
