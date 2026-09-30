// =============================================================================
// tests/unit/test_orbit_table.cpp
//
// Stage-2 guards of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md):
//
//   1. The OrbitTable content hash separates subspaces and groups.
//   2. Burnside sum rule: summing the closed-form survivor counts over
//      all irreps reproduces the subspace dimension exactly
//      (Σ_k dim_k = C(N, n_up), resp. 2^N).
//   3. SectorBasis::build_prefiltered (closed-form survival + parallel
//      orbit materialization) is bit-identical to the serial
//      SectorBasis::build for every irrep: same reps, same sorted
//      orbits, same coefficients, same norms.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/symmetry/group.h>
#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/rep_sector_data.h>  // sector_characters_from
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/subspace.h>

#include <cstdint>
#include <vector>

using namespace ed::symmetry;

TEST_CASE("OrbitTable content hash distinguishes subspaces and groups",
          "[orbit_table]") {
    const int N = 12;
    const SymmetryGroupInfo info =
        ed::sym::translation_group_with_reflection_1d(N);
    const OrbitTable a = build_orbit_table_fixed_sz_streaming(N, 5, info);
    const OrbitTable b = build_orbit_table_fixed_sz_streaming(N, 6, info);
    const OrbitTable c = build_orbit_table_full(N, info);
    REQUIRE(a.content_hash != b.content_hash);
    REQUIRE(a.content_hash != c.content_hash);
    // NOTE: translation_group_with_reflection_1d falls back to the
    // maximal ABELIAN subgroup (= the pure translations), so it would
    // compile to the identical group. Use a genuinely different group:
    // the reflection-only Z2.
    const SymmetryGroupInfo r_only = ed::sym::group_from_generators(
        N, {ed::sym::reflection_1d(N)});
    const OrbitTable d = build_orbit_table_fixed_sz_streaming(N, 5, r_only);
    REQUIRE(a.content_hash != d.content_hash);
}

TEST_CASE("Burnside sum rule: closed-form sector dims tile the subspace",
          "[orbit_table]") {
    const int N = 12;
    for (const SymmetryGroupInfo& info :
         {ed::sym::translation_group_1d(N),
          ed::sym::translation_group_with_reflection_1d(N)}) {
        SECTION("fixed-Sz, |G|=" + std::to_string(info.max_clique.size())) {
            for (int n_up : {2, 5, N / 2}) {
                const OrbitTable tab =
                    build_orbit_table_fixed_sz_streaming(N, n_up, info);
                std::uint64_t total = 0;
                for (const auto& sec : info.sectors) {
                    const std::vector<Complex> chi =
                        sector_characters_from(info, sec.phase_factors);
                    for (std::size_t i = 0; i < tab.size(); ++i) {
                        if (projected_norm_sq(tab, i, chi) >
                            SectorBasis::kOrbitNormSqEpsilon)
                            ++total;
                    }
                }
                REQUIRE(total == tab.subspace_dim);
            }
        }
        SECTION("full space, |G|=" + std::to_string(info.max_clique.size())) {
            const OrbitTable tab = build_orbit_table_full(N, info);
            std::uint64_t total = 0;
            for (const auto& sec : info.sectors) {
                const std::vector<Complex> chi =
                    sector_characters_from(info, sec.phase_factors);
                for (std::size_t i = 0; i < tab.size(); ++i) {
                    if (projected_norm_sq(tab, i, chi) >
                        SectorBasis::kOrbitNormSqEpsilon)
                        ++total;
                }
            }
            REQUIRE(total == (1ULL << N));
        }
    }
}

TEST_CASE("build_prefiltered is bit-identical to serial build, every irrep",
          "[orbit_table][sector_basis]") {
    const int N    = 10;
    const int n_up = N / 2;
    const SymmetryGroupInfo info =
        ed::sym::translation_group_with_reflection_1d(N);
    const SpatialProjector projector(info);
    const FixedSzSubspace subspace = FixedSzSubspace::build(N, n_up);

    const OrbitTable tab = build_orbit_table_fixed_sz_streaming(N, n_up, info);

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const auto& qn    = info.sectors[s].quantum_numbers;
        const auto& phase = info.sectors[s].phase_factors;

        const SectorBasis ref = SectorBasis::build(
            subspace, projector, qn, phase, tab.reps, s);

        const std::vector<Complex> chi = sector_characters_from(info, phase);
        std::vector<double> prefilter(tab.size());
        for (std::size_t i = 0; i < tab.size(); ++i)
            prefilter[i] = projected_norm_sq(tab, i, chi);

        const SectorBasis got = SectorBasis::build_prefiltered(
            subspace, projector, qn, phase, tab.reps, prefilter, s);

        REQUIRE(got.dim() == ref.dim());
        const auto& a = got.sector().basis_states;
        const auto& b = ref.sector().basis_states;
        for (std::size_t i = 0; i < a.size(); ++i) {
            REQUIRE(a[i].orbit_rep == b[i].orbit_rep);
            REQUIRE(a[i].norm == b[i].norm);  // same walk, bitwise equal
            REQUIRE(a[i].orbit_elements == b[i].orbit_elements);
            REQUIRE(a[i].orbit_coefficients.size() ==
                    b[i].orbit_coefficients.size());
            for (std::size_t k = 0; k < a[i].orbit_coefficients.size(); ++k) {
                REQUIRE(a[i].orbit_coefficients[k].real() ==
                        b[i].orbit_coefficients[k].real());
                REQUIRE(a[i].orbit_coefficients[k].imag() ==
                        b[i].orbit_coefficients[k].imag());
            }
        }
        // Reverse lookup agrees.
        for (std::size_t i = 0; i < a.size(); ++i)
            REQUIRE(got.index_of(a[i].orbit_rep) ==
                    ref.index_of(a[i].orbit_rep));
    }
}
