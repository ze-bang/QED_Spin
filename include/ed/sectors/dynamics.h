#pragma once
// =============================================================================
// include/ed/sectors/dynamics.h
//
// Dynamical correlation functions S(omega) = <O^dag delta(omega - H + E_i) O> over the
// symmetry sectors of H.
//
// A probe O is in general not invariant under the point group, time reversal or the
// spin flip (a Fourier mode O_Q maps to O_{pQ}), so the multiplicity folding that is
// exact for the spectrum and for thermodynamics is not exact here. Dynamics therefore
// walks the momentum sectors of the abelian group with no folding: every block holds
// one copy of its levels, and O|psi> is carried from its source sector into every
// target momentum sector (and every Sz or parity sector O reaches) by the
// cross-sector rep-basis scatter; ||O|psi>|| decides every selection rule.
//
//   T = 0:  the ground manifold is every level within `degeneracy_tol` of E0;
//           S = (1/g) sum over the manifold of the continued fraction of O|a> in each
//           target sector, with omega measured from E0.
//   T > 0:  finite-temperature Lanczos over every source sector (the Jaklic-Prelovsek
//           estimator, samples per sector), S = sum_s S_s / sum_s Z_s with Z taken over
//           ALL source sectors, including those O annihilates.
// =============================================================================

#include <ed/sectors/sectors.h>

#include <cstdint>
#include <vector>

namespace ed::sectors {

struct DynamicsSpec {
    std::vector<double> omega;
    double              eta            = 0.05;
    std::vector<double> temperatures;          ///< empty = T = 0
    std::size_t         krylov         = 200;
    std::size_t         samples        = 30;   ///< T > 0: random vectors per source sector
    std::uint64_t       seed           = 0;    ///< 0 = draw one
    double              degeneracy_tol = 1e-8; ///< T = 0: ground-manifold window
    Device              device         = Device::Cpu;   ///< continued fractions (T = 0) / FTLM (T > 0) on a GPU
};

struct DynamicsCurves {
    std::vector<double>              omega;
    std::vector<double>              T;        ///< empty for T = 0
    std::vector<std::vector<double>> S;        ///< one row per temperature (one row at T = 0)
    double                           e0 = 0.0;
    int                              ground_manifold = 0;   ///< T = 0: levels averaged over
    std::size_t                      target_sectors = 0;    ///< sectors O reached
    std::size_t                      device_blocks  = 0;    ///< continued fractions / FTLM sources run on a GPU
};

[[nodiscard]] DynamicsCurves dynamics(const ::Operator& H, int n_sites, const Spec& s,
                                      const ::Operator& O, const DynamicsSpec& d);

}  // namespace ed::sectors
