#pragma once
// =============================================================================
// include/ed/observables/ftlm_cross_irrep_kernel.h
//
// FTLM (Finite-Temperature Lanczos Method) cross-irrep / cross-sector
// dynamical spectral kernel, for a probe observable ``O`` that is
// RECTANGULAR -- i.e., it maps a source-sector orbit basis (``dim_src``)
// to a target-sector orbit basis (``dim_dst``). This is the finite-T
// analog of the T=0 cross-irrep kernel (``cf_spectral_from_vector``
// driven by ``ed::dssf::CrossSectorOrbitObservable``).
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
// Aggregation across source sectors (in the caller, e.g. the
// ed::sectors dynamics verb):
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
// The kernel is a thin specialisation of the multi-sample FTLM
// (fully reorthogonalised ``ed::krylov::lanczos_kernel`` runs plus
// ``tridiag_eig``). The two differences are:
//
//   1. The outer matvec is ``H_src`` on dim_src; the inner matvec
//      is ``H_dst`` on dim_dst. Both are passed as
//      ``std::function<void(const Complex*, Complex*, int)>``.
//   2. The observable application is rectangular -- the kernel
//      pre-allocates a ``dim_dst``-sized work buffer for the
//      target sector and calls the user-supplied ``O_apply`` as
//      ``O_apply(psi_src.data(), phi_dst.data(), dim_dst)``. The
//      caller is responsible for wiring this up against
//      ``ed::dssf::CrossSectorOrbitObservable::apply``.
//
// Threading: the H matvecs are OpenMP-parallel internally; on the host
// the outer FTLM sample loop runs serially (nested OMP corrupts the heap
// allocator on some BLAS builds).
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

/// Parameters for the FTLM cross-irrep kernel.
struct FtlmCrossIrrepOptions {
    std::size_t krylov_dim       = 200;
    std::size_t num_samples      = 30;
    double      broadening       = 0.05;
    /// Base seed: sample s starts from gaussian_vector(dim_src, sample_engine(random_seed, s))
    /// (0 draws one).
    std::uint64_t random_seed    = 0;
    /// Applied in place to each (host) random vector before use, e.g. a projection onto one
    /// spin tower; the kernel renormalises the result. The trace then runs over the image of
    /// the transform, whose dimension is `trace_dim` (0: the whole source sector).
    std::function<void(Complex*, std::size_t)> seed_transform;
    std::size_t trace_dim        = 0;
    /// Device multi-vector source and target H (LinearOperator::bind_cuda_multi): on a CUDA run
    /// up to `batch_width` samples advance in lockstep and share each H apply. O must then be
    /// safe to apply from several threads at once.
    std::function<void(const Complex* const*, Complex* const*, std::size_t, std::size_t)> batch_src, batch_dst;
    std::size_t batch_width      = 8;
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
///   * resolving the target sector via the selection rule,
///   * building the cross-sector observable apply lambda from a
///     ``ed::dssf::CrossSectorOrbitObservable``,
///   * aggregating the per-sector accumulators across source
///     sectors (the ed::sectors dynamics verb does this).
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

}  // namespace ed::observables
