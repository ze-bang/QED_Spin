// =============================================================================
// tests/unit/test_compiled_group.cpp
//
// Stage-1 bit-identity contract of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md):
//
//   CompiledGroup::apply(s, g) == applyPermutation(s, perm[g]) ^ flip[g]
//
// for every state and element, across the N range the byte-LUT covers
// (1..64), plus parity of the swapped-in construction consumers:
//
//   * enumerate_full_orbit_reps / enumerate_fixed_sz_orbit_reps(_streaming)
//     reproduce a scalar reference enumeration exactly (same reps, same
//     order),
//   * build_orbit_stabilizers reproduces a scalar reference stabilizer
//     table exactly,
//   * content_hash is stable under recompilation and sensitive to any
//     element change.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/basis_utils.h>          // applyPermutation (scalar reference)
#include <ed/symmetry/compiled_group.h>
#include <ed/symmetry/group.h>            // translation_group_with_reflection_1d
#include <ed/symmetry/projector.h>
#include <ed/symmetry/rep_projection.h>
#include <ed/symmetry/sector_set.h>       // enumerate_*_orbit_reps*
#include <ed/symmetry/subspace.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using ed::symmetry::CompiledGroup;

namespace {

std::vector<int> random_perm(int n, std::mt19937_64& gen) {
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    std::shuffle(p.begin(), p.end(), gen);
    return p;
}

std::uint64_t random_state(int n, std::mt19937_64& gen) {
    const std::uint64_t mask =
        (n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL);
    return gen() & mask;
}

}  // namespace

TEST_CASE("CompiledGroup::apply matches scalar applyPermutation across N",
          "[compiled_group]") {
    std::mt19937_64 gen(20260702);
    for (int n : {1, 5, 8, 13, 16, 24, 31, 32, 37, 48, 63, 64}) {
        std::vector<std::vector<int>> perms;
        for (int g = 0; g < 6; ++g) perms.push_back(random_perm(n, gen));
        // Include the identity explicitly (is_identity contract).
        std::vector<int> ident(n);
        for (int i = 0; i < n; ++i) ident[i] = i;
        perms.push_back(ident);

        const CompiledGroup cg = CompiledGroup::from_permutations(perms, n);
        REQUIRE(cg.size() == perms.size());
        REQUIRE(cg.is_identity(perms.size() - 1));

        for (int trial = 0; trial < 200; ++trial) {
            const std::uint64_t s = random_state(n, gen);
            for (std::size_t g = 0; g < perms.size(); ++g) {
                REQUIRE(cg.apply(s, g) == applyPermutation(s, perms[g]));
            }
        }
    }
}

TEST_CASE("CompiledGroup flip elements XOR after the permutation",
          "[compiled_group]") {
    std::mt19937_64 gen(42);
    const int n = 20;
    std::vector<std::vector<int>> perms{random_perm(n, gen),
                                        random_perm(n, gen)};
    const std::uint64_t all_ones = (1ULL << n) - 1ULL;
    std::vector<std::uint64_t> flips{0ULL, all_ones};

    const CompiledGroup cg = CompiledGroup::from_elements(perms, flips, n);
    for (int trial = 0; trial < 200; ++trial) {
        const std::uint64_t s = random_state(n, gen);
        REQUIRE(cg.apply(s, 0) == applyPermutation(s, perms[0]));
        REQUIRE(cg.apply(s, 1) ==
                (applyPermutation(s, perms[1]) ^ all_ones));
    }
    // A pure flip on the identity permutation is NOT the identity element.
    std::vector<int> ident(n);
    for (int i = 0; i < n; ++i) ident[i] = i;
    const CompiledGroup cg2 = CompiledGroup::from_elements(
        {ident, ident}, {0ULL, all_ones}, n);
    REQUIRE(cg2.is_identity(0));
    REQUIRE(!cg2.is_identity(1));
}

TEST_CASE("CompiledGroup content_hash: stable and element-sensitive",
          "[compiled_group]") {
    std::mt19937_64 gen(7);
    const int n = 16;
    std::vector<std::vector<int>> perms{random_perm(n, gen),
                                        random_perm(n, gen)};
    const auto a = CompiledGroup::from_permutations(perms, n);
    const auto b = CompiledGroup::from_permutations(perms, n);
    REQUIRE(a.content_hash() == b.content_hash());

    // Any element change must change the hash.
    auto perms2 = perms;
    std::swap(perms2[1][0], perms2[1][1]);
    const auto c = CompiledGroup::from_permutations(perms2, n);
    REQUIRE(a.content_hash() != c.content_hash());

    // Same permutations, different flips: different hash.
    const auto d = CompiledGroup::from_elements(
        perms, {0ULL, (1ULL << n) - 1ULL}, n);
    REQUIRE(a.content_hash() != d.content_hash());
}

// ---------------------------------------------------------------------------
// Consumer parity: the swapped-in enumerators and the stabilizer builder
// must be bit-identical to a scalar reference on a nontrivial group
// (D_N = translations x reflection on a 14-site ring).
// ---------------------------------------------------------------------------

TEST_CASE("Stage-1 enumerators are bit-identical to scalar reference",
          "[compiled_group][sector_set]") {
    const int N = 14;
    const SymmetryGroupInfo info =
        ed::sym::translation_group_with_reflection_1d(N);
    const std::size_t G = info.max_clique.size();
    REQUIRE(G > 1);

    // Scalar reference: full space.
    std::vector<std::uint64_t> ref_full;
    for (std::uint64_t s = 0; s < (1ULL << N); ++s) {
        bool is_rep = true;
        for (std::size_t g = 0; g < G; ++g) {
            if (applyPermutation(s, info.max_clique[g]) < s) {
                is_rep = false;
                break;
            }
        }
        if (is_rep) ref_full.push_back(s);
    }
    REQUIRE(ed::symmetry::enumerate_full_orbit_reps(info, N) == ref_full);

    // Scalar reference: fixed-Sz (streaming + materialized lanes).
    const int n_up = N / 2;
    std::vector<std::uint64_t> ref_sz;
    for (std::uint64_t s = 0; s < (1ULL << N); ++s) {
        if (__builtin_popcountll(s) != n_up) continue;
        bool is_rep = true;
        for (std::size_t g = 0; g < G; ++g) {
            if (applyPermutation(s, info.max_clique[g]) < s) {
                is_rep = false;
                break;
            }
        }
        if (is_rep) ref_sz.push_back(s);
    }
    REQUIRE(ed::symmetry::enumerate_fixed_sz_orbit_reps_streaming(
                N, n_up, info) == ref_sz);
    const ed::symmetry::FixedSzSubspace sub =
        ed::symmetry::FixedSzSubspace::build(N, n_up);
    REQUIRE(ed::symmetry::enumerate_fixed_sz_orbit_reps(sub, info) == ref_sz);
}

TEST_CASE("Stage-1 stabilizer table is bit-identical to scalar reference",
          "[compiled_group][rep_projection]") {
    const int N = 12;
    const SymmetryGroupInfo info =
        ed::sym::translation_group_with_reflection_1d(N);
    const ed::symmetry::SpatialProjector projector(info);
    const std::size_t G = info.max_clique.size();

    const std::vector<std::uint64_t> reps =
        ed::symmetry::enumerate_fixed_sz_orbit_reps_streaming(N, N / 2, info);
    REQUIRE(!reps.empty());

    const ed::symmetry::OrbitStabilizers tab =
        ed::symmetry::build_orbit_stabilizers(reps, projector);
    REQUIRE(tab.size() == reps.size());

    for (std::size_t i = 0; i < reps.size(); ++i) {
        std::vector<int> ref_stab;
        for (std::size_t g = 0; g < G; ++g) {
            if (applyPermutation(reps[i], info.max_clique[g]) == reps[i]) {
                ref_stab.push_back(static_cast<int>(g));
            }
        }
        REQUIRE(tab.stab_size[i] == ref_stab.size());
        const auto it = tab.nontrivial.find(i);
        if (ref_stab.size() > 1) {
            REQUIRE(it != tab.nontrivial.end());
            REQUIRE(it->second == ref_stab);
        } else {
            REQUIRE(it == tab.nontrivial.end());
        }
    }
}
