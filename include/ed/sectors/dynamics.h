#pragma once
// =============================================================================
// include/ed/sectors/dynamics.h
//
// Dynamical correlation functions S_AB(omega) = <A^dag delta(omega - H + E_i) B> over the symmetry
// sectors of H: autocorrelations (B = A, real) and cross-correlations (complex), several probes per
// call sharing the ground manifold (T = 0) and the source sectors (T > 0).
//
// A probe O is in general not invariant under the point group, time reversal or the
// spin flip (a Fourier mode O_Q maps to O_{pQ}), so the multiplicity folding that is
// exact for the spectrum and for thermodynamics is not exact here. Dynamics therefore
// walks the momentum sectors of the abelian group with no folding: every block holds
// one copy of its levels, and O|psi> is carried from its source sector into every
// target momentum sector (and every Sz or parity sector O reaches) by the
// cross-sector rep-basis scatter; ||O|psi>|| decides every selection rule.
//
//   T = 0:  the ground manifold is every level within degeneracy_tol * s_H of E0 (s_H: the sum
//           of |c| over H's terms, <ed/core/numerics.h>, so s * H keeps the same manifold);
//           S = (1/g) sum over the manifold of the continued fraction of B|a> in each
//           target sector (with A|a> projected on its Krylov vectors for a cross pair),
//           omega measured from E0.
//   T > 0:  finite-temperature Lanczos over every source sector (the Jaklic-Prelovsek
//           estimator, samples per sector), S = sum_s S_s / sum_s Z_s with Z taken over
//           ALL source sectors, including those the probes annihilate.
// =============================================================================

#include <ed/sectors/sectors.h>

#include <cstdint>
#include <vector>

namespace ed::sectors {

struct DynamicsSpec {
    std::vector<double> omega;
    double eta = 0.05;
    std::vector<double> temperatures;          ///< empty = T = 0
    std::size_t krylov = 200;
    std::size_t samples = 40;   ///< T > 0: random vectors per source sector
    std::uint64_t seed = 0;    ///< 0 = draw one
    // scale-free: relative to s_H (numerics.h)
    double degeneracy_tol = 1e-8; ///< T = 0: ground-manifold window, relative to s_H (numerics.h)
    int dense_max_dim = -1;   ///< T = 0: the ground-manifold eigensolve's crossover (EigsOptions)
    bool prune = true; ///< T = 0: the ground-manifold eigensolve prunes blocks (EigsOptions)
    Device device = Device::Cpu;   ///< continued fractions (T = 0) / FTLM (T > 0) on a GPU
    /// T > 0: the thermal pass from the same source Lanczos runs -- the thermodynamics (lnZ, E, V) and
    /// <X>(T) for every observable and every pair A_a^dag B_b of the requests (the symmetric FTLM
    /// estimator through each sample's phi_T), over every source sector the symmetries of H fold.
    /// Not under a total-spin restriction (ed::Unsupported).
    bool thermodynamics = false;
    std::vector<const ::Operator*> observables;
    std::vector<PairRequest> observable_pairs;
};

/// One correlation: S_AB(omega) = sum_m p_m <m|A^dag delta(omega - H + E_m) B|m>; B null: A's
/// autocorrelation.
struct Probe {
    const ::Operator* A = nullptr;
    const ::Operator* B = nullptr;
};

struct DynamicsCurves {
    std::vector<double> omega;
    std::vector<double> T;        ///< empty for T = 0
    /// [probe][row][omega]: one row per temperature (one row at T = 0). An autocorrelation's
    /// imaginary part is zero up to roundoff.
    std::vector<std::vector<std::vector<Complex>>> S;
    double e0 = 0.0;   ///< T = 0: the ground energy; T > 0 with thermodynamics: the lowest weighted source Ritz value
    int ground_manifold = 0;   ///< T = 0: levels averaged over
    /// With DynamicsSpec::thermodynamics: per temperature, ln Z, E and Var(H), and <X>(T) per observable
    /// then per pair (O[x][T]).
    std::vector<double> lnZ, E, V;
    std::vector<std::vector<Complex>> O;
    std::size_t target_sectors = 0;    ///< sectors the probes reached
    std::size_t device_blocks = 0;    ///< continued fractions / FTLM sources run on a GPU
    Placement placement;
    Diagnostics diagnostics;
};

[[nodiscard]] DynamicsCurves dynamics(const ::Operator& H, const Spec& s, const std::vector<Probe>& probes,
                                      const DynamicsSpec& d);

/// O's autocorrelation alone.
[[nodiscard]] inline DynamicsCurves dynamics(const ::Operator& H, const Spec& s, const ::Operator& O,
                                             const DynamicsSpec& d) {
    return dynamics(H, s, std::vector<Probe>{Probe{&O, nullptr}}, d);
}

}  // namespace ed::sectors
