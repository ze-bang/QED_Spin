#pragma once
// =============================================================================
// include/ed/observables/ftlm_cross_irrep_kernel.h
//
// FTLM (Finite-Temperature Lanczos Method) cross-irrep / cross-sector
// dynamical spectral kernel. Generalises the same-dim multi-T FTLM
// machinery (`compute_dynamical_correlation_multi_operator_multi_temperature`
// in `src/solvers/cpu/ftlm_dynamical.cpp`) to
// the case where the probe observable ``O`` is RECTANGULAR --
// i.e., it maps a source-sector orbit basis (``dim_src``) to a
// target-sector orbit basis (``dim_dst``). This is exactly the
// finite-T analog of the T=0 cross-irrep kernel
// (``cf_spectral_from_vector`` driven by
// ``ed::dssf::CrossSectorOrbitObservable``) and the missing piece
// that closes the SOTA gap flagged in
// ``docs/architecture/SYMMETRY.md`` for the
// ``DYNAMICAL_THERMAL`` row (spatial-irrep + cross-irrep).
//
// Math
// ----
// For each source sector ``k_src``, the FTLM trace estimator is
//
//   Tr_{k_src}[ exp(-beta H) A^dagger (omega - (H - E_m))^{-1} A ]
//     ~= (dim_src / R) * sum_r sum_i exp(-beta E_i^{(r)}) c_i^{(r)^2}
//        * <psi_i^{(r)} | A^dagger (omega - (H - E_i^{(r)}))^{-1} A | psi_i^{(r)}>
//
// where ``|r>`` are dim_src-dim i.i.d. Gaussian samples drawn in the
// source-sector orbit basis, ``|psi_i^{(r)}> = sum_j V[i,j] |v_j^{(r)}>``
// are the Ritz states of the outer Lanczos on ``H_src`` from ``|r>``,
// ``E_i^{(r)}`` are the corresponding Ritz energies, and
// ``c_i^{(r)} = <psi_i^{(r)} | r> = V[i, 0]`` is the first-column
// amplitude. The inner resolvent is evaluated by a *second*, target-
// sector Lanczos seeded from ``phi_i = A |psi_i^{(r)}>`` (now in
// dim_dst orbit basis), giving the standard continued-fraction
// Lehmann sum
//
//   S_i(omega) = |phi_i|^2 * sum_k V_S[0, k]^2
//                * (eta/pi) / ((omega - (lambda_k - E_i))^2 + eta^2)
//
// Total finite-T spectral function:
//
//   S_{k_src}(omega, T) = (dim_src / R)
//        * sum_r sum_i exp(-beta(E_i^{(r)} - E_min)) c_i^{(r)^2} S_i(omega)
//
// and the partition function in this sector
//
//   Z_{k_src}(T) = (dim_src / R)
//        * sum_r sum_i exp(-beta(E_i^{(r)} - E_min)) c_i^{(r)^2}.
//
// Aggregation across source sectors (in the caller / streaming-
// symmetry binding):
//
//   S_total(omega, T) = (sum_k S_k(omega, T)) / (sum_k Z_k(T)),
//
// where each sector's per-T accumulators are returned UN-normalised
// (i.e., we hand the caller the raw numerator and denominator so
// the dim_src / R factors line up correctly when sectors of
// different sizes are combined).
//
// Implementation
// --------------
// The kernel is a thin specialisation of the legacy multi-sample
// FTLM (we reuse ``build_lanczos_tridiagonal_with_basis`` and
// ``diagonalize_tridiagonal_ritz`` from the same compilation unit
// as the legacy kernel). The two differences are:
//
//   1. The outer matvec is ``H_src`` on dim_src; the inner matvec
//      is ``H_dst`` on dim_dst. Both are passed as
//      ``std::function<void(const Complex*, Complex*, int)>``,
//      identical to the legacy lambdas.
//   2. The observable application is rectangular -- the kernel
//      pre-allocates a ``dim_dst``-sized work buffer for the
//      target sector and calls the user-supplied ``O_apply`` as
//      ``O_apply(psi_src.data(), phi_dst.data(), dim_dst)``. The
//      caller is responsible for wiring this up against
//      ``ed::dssf::CrossSectorOrbitObservable::apply``.
//
// Threading: the inner H matvecs are OpenMP-parallel through the
// streaming-symmetry sector view; the outer FTLM loop runs serial
// (legacy convention -- nested OMP corrupts the heap allocator on
// some BLAS builds, see ftlm.cpp line ~3055).
//
// SOTA upgrade, May 2026 (closes the finite-T DSSF + symmetry gap).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <vector>

namespace ed::observables {

using Complex = std::complex<double>;

/// Parameters for the FTLM cross-irrep kernel. Mirrors the subset
/// of ``DynamicalResponseParameters`` actually consumed by the
/// multi-sample multi-T legacy lane, plus a few cross-sector knobs.
struct FtlmCrossIrrepOptions {
    std::size_t krylov_dim       = 200;
    std::size_t num_samples      = 30;
    double      broadening       = 0.05;
    /// Tolerance for the inner Lanczos breakdown / convergence
    /// check. Matches the legacy ``DynamicalResponseParameters::tolerance``.
    double      tolerance        = 1e-12;
    /// Full reorthogonalisation flag for the outer + inner Lanczos.
    /// Off by default to match the legacy `compute_dynamical_correlation_*`
    /// default; turn on for problems with near-degeneracies.
    bool        full_reorthogonalization = false;
    std::size_t reorth_frequency = 1;
    /// Random-seed offset; each sample uses
    /// ``random_seed + sample_idx * 12345``. Matches the legacy
    /// FTLM multi-sample convention so seeds are stable across
    /// the two paths.
    std::uint64_t random_seed    = 0;
    /// Verbose progress to stdout. Off by default.
    bool        verbose          = false;
};

/// One sector's UN-normalised FTLM cross-irrep accumulators. Keyed
/// by temperature; the caller aggregates across sectors.
struct FtlmCrossIrrepSectorResult {
    /// Per-temperature numerators: sum_r sum_i exp(-beta(E_i-E_min)) c_i^2 S_i(omega)
    /// Length equals ``omega_grid.size()``. Multiplied by dim_src on
    /// return so the cross-sector aggregator can sum directly.
    std::map<double, std::vector<double>>  S_real;
    std::map<double, std::vector<double>>  S_imag;
    /// Per-temperature denominators (the sector's partition function
    /// times dim_src / R). Same dim_src multiplication as ``S_*``.
    std::map<double, double>               Z;
    /// Source-sector dimension (used to multiply S/Z; reported for
    /// debugging).
    std::size_t                            dim_src       = 0;
    /// Target-sector dimension.
    std::size_t                            dim_dst       = 0;
    /// Number of samples actually processed (after rejecting samples
    /// whose Lanczos failed).
    std::size_t                            samples_done  = 0;
    /// Energy reference used for the thermal exponent's numerical
    /// stability shift (= min Ritz energy across all samples in this
    /// sector). Reported so the caller can sanity-check the per-
    /// sector recombination.
    double                                 E_min         = 0.0;
};

/// FTLM cross-irrep dynamical kernel for ONE source/target sector
/// pair. The caller is responsible for:
///   * resolving the target sector via the selection rule
///     (``include/ed/core/sector_loop.h::resolve_target_sector``),
///   * building the cross-sector observable apply lambda from a
///     ``ed::dssf::CrossSectorOrbitObservable``,
///   * aggregating the per-sector accumulators across source
///     sectors (typically via the bundled
///     ``combine_sector_dynamical_spectra`` helper below).
///
/// @param H_src       matvec on dim_src (source sector restricted H)
/// @param H_dst       matvec on dim_dst (target sector restricted H)
/// @param O_apply     rectangular src->dst observable apply
/// @param dim_src     source orbit-basis dimension
/// @param dim_dst     target orbit-basis dimension
/// @param temperatures temperatures (Kelvin units of energy axis)
/// @param omega_grid  frequency grid (linear samples)
/// @param opts        kernel options (see above)
FtlmCrossIrrepSectorResult ftlm_cross_irrep_kernel_one_sector(
    const std::function<void(const Complex*, Complex*, int)>& H_src,
    const std::function<void(const Complex*, Complex*, int)>& H_dst,
    const std::function<void(const Complex*, Complex*, int)>& O_apply,
    std::size_t                       dim_src,
    std::size_t                       dim_dst,
    const std::vector<double>&        temperatures,
    const std::vector<double>&        omega_grid,
    const FtlmCrossIrrepOptions&      opts);

/// F-shifted Z-weighted recombination of per-sector dynamical
/// spectra. Mirrors ``ed::core::combine_sector_thermodynamics`` but
/// for an omega-resolved spectral function. Returns the final
/// thermal-averaged S(omega, T) per temperature, ready for the
/// ``SpectralResult`` payload.
///
/// The math is the linearity of the trace:
///
///   S_total(omega, T) = (sum_k S_k(omega, T)_num) / (sum_k Z_k(T)_den).
///
/// where each per-sector pair carries the dim_src * (Σ ...) numerator
/// and dim_src * (Σ ...) denominator (NOT yet divided), and the
/// final division gives the total thermal-averaged spectral function.
///
/// We use the F-shift (subtract the global F_min(T) before
/// exponentiating) to defend against floating-point underflow when
/// per-sector Z's span many orders of magnitude. The per-sector
/// kernel above stores E_min separately, but for spectral combination
/// the Z numerators already encode their own exp(-beta E_min^sector)
/// references; we therefore implement a *global*-F-shifted
/// recombination at this layer.
///
/// @param sector_results  one entry per (k_src, k_dst) sector pair.
/// @param temperatures    same set the caller passed to the kernel.
/// @param num_omega       frequency-grid length.
/// @return  ``{T -> S_real[omega]}`` and ``{T -> S_imag[omega]}`` of
///          the combined spectral function.
struct DynamicalSpectraMerged {
    std::map<double, std::vector<double>>  S_real;
    std::map<double, std::vector<double>>  S_imag;
};

DynamicalSpectraMerged combine_sector_dynamical_spectra(
    const std::vector<FtlmCrossIrrepSectorResult>& sector_results,
    const std::vector<double>&                     temperatures,
    std::size_t                                    num_omega);

}  // namespace ed::observables
