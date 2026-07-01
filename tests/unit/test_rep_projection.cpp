// =============================================================================
// tests/unit/test_rep_projection.cpp
//
// Pins the closed-form orbit-projected norm (ed::symmetry::projected_norm_sq,
// rep_projection.h) against the reference orbit walk (compute_orbit_for_state).
// This is the contract every sector builder relies on once it routes its
// Pass-1.5 norm through the shared primitive: the closed form must reproduce the
// walk's survival decision EXACTLY (it sets the sector dimension) and its norm
// value to machine precision, for both the fixed-Sz and full-space subspaces
// (the two subspace families that satisfy the "orbit closed in subspace"
// precondition).
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/symmetry/fixed_sz_membership.h>
#include <ed/symmetry/group.h>             // translation_group_1d
#include <ed/symmetry/projector.h>
#include <ed/symmetry/projector_chain.h>   // compute_orbit_for_state
#include <ed/symmetry/rep_projection.h>    // build_orbit_stabilizers, projected_norm_sq
#include <ed/symmetry/rep_sector_data.h>   // sector_characters_from
#include <ed/symmetry/sector_basis.h>      // kOrbitNormSqEpsilon
#include <ed/symmetry/sector_set.h>        // enumerate_*_orbit_reps*
#include <ed/symmetry/subspace.h>          // FixedSzSubspace, FullSpaceSubspace

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

using namespace ed::symmetry;

namespace {

// Run the parity check for one rep set + subspace against every irrep sector.
template <class Subspace>
void check_parity(const SymmetryGroupInfo&          info,
                  const Subspace&                   subspace,
                  const std::vector<std::uint64_t>& reps) {
    const SpatialProjector projector(info);
    const OrbitStabilizers stabs = build_orbit_stabilizers(reps, projector);
    REQUIRE(stabs.size() == reps.size());

    const double eps = SectorBasis::kOrbitNormSqEpsilon;
    std::vector<std::uint64_t>        elems;
    std::vector<std::complex<double>> coeffs;

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const auto& phase = info.sectors[s].phase_factors;
        const auto  chi   = sector_characters_from(info, phase);
        for (std::size_t i = 0; i < reps.size(); ++i) {
            double n_ref = 0.0;
            compute_orbit_for_state(subspace, projector, reps[i], phase,
                                    elems, coeffs, n_ref);
            const bool   ref_survives = !elems.empty() && n_ref > eps;
            const double n_closed     = projected_norm_sq(stabs, i, chi);

            // Survival (the sector dimension) must be bit-identical.
            REQUIRE(ref_survives == (n_closed > eps));
            // Norm value matches to machine precision when the rep survives.
            if (ref_survives)
                REQUIRE(std::abs(n_ref - n_closed) <= 1e-9 * std::max(1.0, n_ref));
        }
    }
}

}  // namespace

TEST_CASE("projected_norm_sq == compute_orbit_for_state (fixed-Sz)",
          "[symmetry][rep_projection]") {
    for (int N : {6, 8, 10}) {
        const int               n_up = N / 2;
        const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
        const FixedSzMembershipSubspace subspace(N, n_up);
        const auto reps = enumerate_fixed_sz_orbit_reps_streaming(N, n_up, info);
        REQUIRE_FALSE(reps.empty());
        check_parity(info, subspace, reps);
    }
}

TEST_CASE("projected_norm_sq == compute_orbit_for_state (full space)",
          "[symmetry][rep_projection]") {
    for (int N : {4, 6}) {
        const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
        const FullSpaceSubspace subspace(static_cast<std::uint64_t>(N));
        const auto reps = enumerate_full_orbit_reps(info, static_cast<std::uint64_t>(N));
        REQUIRE_FALSE(reps.empty());
        check_parity(info, subspace, reps);
    }
}
