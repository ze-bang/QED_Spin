// =============================================================================
// include/ed/solvers/tpq_dynamical.h
//
// Thermal Pure Quantum (TPQ) state approach to finite-T dynamical correlators.
//
// Mathematical foundation
// -----------------------
//   A TPQ state at inverse temperature β is defined as:
//
//     |ψ(β)⟩ = e^{-βH/2} |r⟩ / ‖ e^{-βH/2} |r⟩ ‖
//
//   where |r⟩ is a Haar-random state.  For large Hilbert-space dimension D,
//   thermal expectation values of observables converge to their canonical
//   ensemble values with corrections O(1/√D):
//
//     ⟨O⟩_β  ≈  ⟨ψ(β)|O|ψ(β)⟩   (up to O(1/√D) fluctuations)
//     Z(β)   ≈  D ‖ e^{-βH/2} |r⟩ ‖²
//
//   The dynamical correlator follows from the real-time two-point function:
//
//     G_{O1,O2}(t, β) = ⟨ψ(β)|O1† e^{iHt} O2 e^{-iHt}|ψ(β)⟩
//                      = ⟨φ_L(β)| e^{iHt} |φ_R(β,t=0)⟩
//
//   where:
//     |φ_R(β)⟩ = O2 |ψ(β)⟩
//     |φ_L(β)⟩ = O1 |ψ(β)⟩   (or φ_R if O1=O2)
//
//   The real-time propagation e^{iHt}|φ_R⟩ is computed via Chebyshev
//   expansion of e^{iHt} on the rescaled Hamiltonian H_sc = (H-b)/a:
//
//     e^{iHt}|v⟩ = e^{ibt} Σ_k (2-δ_{k0}) i^k J_k(at) T_k(H_sc)|v⟩
//
//   where J_k is the k-th order Bessel function of the first kind.
//
//   The spectral function is obtained by Fourier-transforming G(t,β):
//
//     S_{O1,O2}(ω, β) = (1/π) Re ∫_0^{t_max} G(t,β) e^{iωt} dt
//
//   In practice we sample G at n_t equally spaced time points t_j = j·Δt
//   and use a window function w(t) to suppress Gibbs oscillations:
//
//     S(ω,β) ≈ (1/π) Re Σ_j w(t_j) G(t_j,β) e^{iωt_j} Δt
//
// Computation of |ψ(β)⟩
// ----------------------
//   The Boltzmann state e^{-βH/2}|r⟩ is computed via Chebyshev expansion:
//
//     e^{-aβ' H_sc}|r⟩ ≈ I_0(aβ') + 2 Σ_{k=1}^{M_β} (-1)^k I_k(aβ') T_k(H_sc)|r⟩
//
//   where β' = β/2 and I_k is the modified Bessel function of the first kind.
//   The rescaling shifts: H = a H_sc + b, so e^{-βH/2} = e^{-βb/2} e^{-aβ/2 H_sc}.
//
// Time-window functions
// ---------------------
//   Linear ramp:    w(t) = 1 − t/t_max
//   Hann window:    w(t) = cos²(π t / (2 t_max))
//   No window:      w(t) = 1   (produces Gibbs ringing for sharp peaks)
//
// Error estimates
// ---------------
//   - TPQ statistical error: O(1/√D) per sample; use `num_samples` random vectors.
//   - Time truncation error: O(e^{−t_max δE_min}) where δE_min = spectral gap.
//   - Chebyshev truncation for e^{iHt}: need M_t ≳ e * a * t_max moments.
//
// API
// ---
//   compute_tpq_dynamical:            single β, single operator pair
//   compute_tpq_dynamical_multi_beta: multiple β from the same random vectors
//   compute_tpq_dynamical_from_state: use pre-constructed |ψ(β)⟩ (e.g. mTPQ)
// =============================================================================

#pragma once

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/matvec_types.h>
#include <ed/solvers/ftlm_jp.h>      // JPDynamicalResult (for compat)

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>

namespace ed::tpq::dynamical {

using Complex       = ed::types::Complex;
using ComplexVector = std::vector<Complex>;
using MatVec        = ed::types::MatVec;

// ---------------------------------------------------------------------------
// Window function type for Fourier transform
// ---------------------------------------------------------------------------
enum class WindowFunction {
    Hann,    ///< cos²(πt/(2t_max))  — recommended; suppresses Gibbs
    Linear,  ///< 1 − t/t_max
    None,    ///< w=1, no windowing (fastest; may produce ringing)
};

// ---------------------------------------------------------------------------
// Control parameters
// ---------------------------------------------------------------------------
struct TPQParameters {
    /// Number of random TPQ initial states to average over.
    /// Variance ∝ 1/num_samples.  For D ≥ 1000, num_samples=5 is often enough.
    int num_samples = 5;

    /// Number of Chebyshev moments for e^{-βH/2}|r⟩ (Boltzmann state).
    /// Rule of thumb: M_beta ≳ 2 * a * beta/2 where a = bandwidth/2.
    int boltzmann_moments = 300;

    /// Number of Chebyshev moments for e^{iHt}|v⟩ (real-time propagation).
    /// Rule of thumb: M_time ≳ e * a * t_max (Bessel series truncation).
    int time_moments = 512;

    /// Maximum real time t_max for the Fourier integral.
    /// Frequency resolution: δω ≈ 2π / t_max.
    double t_max = 20.0;

    /// Number of time points (including t=0).
    int n_time = 2048;

    /// Number of output frequency points.
    int n_omega = 2001;

    /// Fractional buffer for KPM/Chebyshev energy window.
    double spectral_bound_buffer = 0.05;

    /// Window function for the Fourier integral.
    WindowFunction window = WindowFunction::Hann;

    /// Random seed (0 = hardware random device).
    std::uint64_t random_seed = 0;

    /// Energy bounds (auto-detected from a short power-iteration if both = 0).
    double E_min = 0.0;
    double E_max = 0.0;

    /// Energy shift for Boltzmann Z computation.
    double energy_shift = 0.0;

    /// Lanczos iterations for energy-bound estimation (if E_min/E_max = 0).
    int bound_lanczos_dim = 200;
};

// ---------------------------------------------------------------------------
// Result type  (layout-compatible with JPDynamicalResult)
// ---------------------------------------------------------------------------
struct TPQDynamicalResult {
    /// Output frequency grid (transfer energies).
    std::vector<double> frequencies;

    /// Inverse temperature(s) at which S was evaluated.
    std::vector<double> betas;

    /// Spectral function S(ω, β) in row-major layout [t * n_omega + i].
    std::vector<double> spectral_real;
    std::vector<double> spectral_imag;

    /// Integrated spectral weight ∫S dω per beta.
    std::vector<double> static_correlator;

    /// Partition function estimate per beta: Z ≈ D * ‖e^{-βH/2}|r⟩‖²
    std::vector<double> partition_function;

    /// Ground-state energy estimate.
    double ground_state_estimate = 0.0;

    /// Energy shift used in Boltzmann weights.
    double energy_shift_used = 0.0;

    /// Number of random samples used.
    int total_samples = 0;

    /// KPM energy window half-width and centre.
    double kpm_a = 1.0;
    double kpm_b = 0.0;

    /// Real-time correlation function G(t_j, β) for the last sample.
    /// Length: n_time * n_beta.  Useful for diagnostics and custom FT.
    std::vector<Complex> G_time;   // row-major [t * n_time + j]
    std::vector<double>  t_grid;
};

// ---------------------------------------------------------------------------
// Primary entry points
// ---------------------------------------------------------------------------

/// Compute S_{O1,O2}(ω, β) via TPQ dynamical at a single inverse temperature.
///
/// @param H         Hamiltonian MatVec.
/// @param O1        Left operator.
/// @param O2        Right operator.
/// @param dim       Hilbert-space dimension.
/// @param beta      Inverse temperature.
/// @param omega_min Minimum output frequency.
/// @param omega_max Maximum output frequency.
/// @param params    TPQ control parameters.
TPQDynamicalResult compute_tpq_dynamical(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    double beta,
    double omega_min,
    double omega_max,
    const TPQParameters& params = {});

/// Compute S_{O1,O2}(ω, β) for multiple inverse temperatures simultaneously.
///
/// All β values share the same random Boltzmann states.  This is efficient
/// when sweeping temperature at fixed (H, O1, O2): the per-temperature cost
/// is a separate Chebyshev propagation for each β but shares the time-evolution.
///
/// Note: for very different β values, the Boltzmann moments required differ;
/// params.boltzmann_moments should be sufficient for the coldest (largest) β.
TPQDynamicalResult compute_tpq_dynamical_multi_beta(
    MatVec H,
    MatVec O1,
    MatVec O2,
    std::uint64_t dim,
    const std::vector<double>& betas,
    double omega_min,
    double omega_max,
    const TPQParameters& params = {});

/// Compute S_{O1,O2}(ω) for a pre-constructed TPQ state |ψ⟩ (e.g. mTPQ).
///
/// Useful when |ψ(β)⟩ is computed externally (e.g. by iterative Lanczos
/// cooling or full-vector imaginary-time propagation).
///
/// @param H         Hamiltonian MatVec (inner sector, for time propagation).
/// @param O1, O2    Operators.
/// @param psi       Pre-computed TPQ state |ψ(β)⟩.
/// @param beta      The temperature at which |ψ⟩ was prepared (for Z).
/// @param Z_estimate Partition-function estimate Z(β) ≈ D * ‖psi_unnorm‖² / ‖psi‖²
TPQDynamicalResult compute_tpq_dynamical_from_state(
    MatVec H,
    MatVec O1,
    MatVec O2,
    const ComplexVector& psi,
    double beta,
    double Z_estimate,
    double omega_min,
    double omega_max,
    const TPQParameters& params = {});

} // namespace ed::tpq::dynamical
