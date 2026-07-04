// =============================================================================
// tests/unit/test_sector_plan.cpp
//
// Stage-8 guards (SymmetryEngine v2): the pure composition-planning
// functions in include/ed/symmetry/sector_plan.h.
//
//   * resolve_symmetry_composition: Auto engages on symmetric/real H,
//     Off disables, Require throws on a Hamiltonian without the
//     symmetry; projection is backend-independent (Stage 8b device flips).
//   * plan_build_window: transport clamps to n_up <= N/2, covers
//     mirror-only request windows, and degenerate windows fall back.
//   * plan_tr_actions: canonical members solve, partners copy with the
//     right source; flip-synthetic indices pair within their parity
//     block; dim mismatches force a solve (paranoia).
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/symmetry/group.h>
#include <ed/symmetry/sector_plan.h>

#include <complex>
#include <vector>

using namespace ed::symmetry;
using Cx = std::complex<double>;

namespace {

ed::matvec::TermStorage heisenberg_terms() {
    ed::matvec::TermStorage t;
    t.diag_two_body.push_back({0, 1, Cx(1.0, 0.0)});
    t.offdiag_two_body.push_back({0, 1, 0, 1, Cx(0.5, 0.0)});
    t.offdiag_two_body.push_back({0, 1, 1, 0, Cx(0.5, 0.0)});
    return t;
}

ed::matvec::TermStorage zeeman_terms() {
    auto t = heisenberg_terms();
    t.diag_one_body.push_back({0, Cx(0.3, 0.0)});   // breaks flip, keeps real
    return t;
}

ed::matvec::TermStorage complex_terms() {
    auto t = heisenberg_terms();
    t.offdiag_two_body.push_back({1, 2, 0, 1, Cx(0.0, 0.2)});  // breaks real
    return t;
}

struct FakeTag {
    int           n_up;
    std::size_t   sector_index;
    std::uint64_t sector_dim;
};

}  // namespace

TEST_CASE("resolve_symmetry_composition: modes and gates", "[sector_plan]") {
    const SymmetryGroupInfo info = ed::sym::translation_group_1d(8);

    SECTION("Auto engages on symmetric + real H (CPU)") {
        const auto c = resolve_symmetry_composition(
            heisenberg_terms(), info, /*allow_gpu=*/false);
        REQUIRE(c.flip_symmetric);
        REQUIRE(c.h_real);
        REQUIRE(c.flip_transport);
        REQUIRE(c.flip_project);          // CPU: projection allowed
        REQUIRE(c.tr_pairing);
        REQUIRE(c.tr_partner.size() == info.sectors.size());
    }
    SECTION("GPU backend: projection allowed too (Stage 8b device flips)") {
        const auto c = resolve_symmetry_composition(
            heisenberg_terms(), info, /*allow_gpu=*/true);
        REQUIRE(c.flip_transport);
        REQUIRE(c.flip_project);
        REQUIRE(c.tr_pairing);
    }
    SECTION("Off disables regardless of H") {
        const auto c = resolve_symmetry_composition(
            heisenberg_terms(), info, false, SymToggle::Off, SymToggle::Off);
        REQUIRE_FALSE(c.flip_transport);
        REQUIRE_FALSE(c.tr_pairing);
        REQUIRE(c.flip_symmetric);        // detection still reported
    }
    SECTION("Require throws when H lacks the symmetry") {
        REQUIRE_THROWS_AS(
            resolve_symmetry_composition(zeeman_terms(), info, false,
                                         SymToggle::Require, SymToggle::Auto),
            std::runtime_error);
        REQUIRE_THROWS_AS(
            resolve_symmetry_composition(complex_terms(), info, false,
                                         SymToggle::Auto, SymToggle::Require),
            std::runtime_error);
        // Zeeman keeps H real: TR 'require' is satisfied there.
        const auto c = resolve_symmetry_composition(
            zeeman_terms(), info, false, SymToggle::Off, SymToggle::Require);
        REQUIRE(c.tr_pairing);
    }
}

TEST_CASE("plan_build_window: transport clamps and fallbacks", "[sector_plan]") {
    SymmetryComposition on;
    on.flip_transport = true;
    on.flip_project   = true;

    SECTION("full window halves") {
        const auto w = plan_build_window(on, 0, 12, 12);
        REQUIRE(w.lo == 0);
        REQUIRE(w.hi == 6);
        REQUIRE(w.flip_transport);
        REQUIRE(w.flip_project_half);
    }
    SECTION("mirror-only request pulls the window down") {
        const auto w = plan_build_window(on, 8, 12, 12);   // wants n_up 8..12
        REQUIRE(w.lo == 0);                                 // = 12 - 12
        REQUIRE(w.hi == 6);
        REQUIRE(w.flip_transport);
    }
    SECTION("open-ended request (-1) covers everything") {
        const auto w = plan_build_window(on, 0, -1, 12);
        REQUIRE(w.lo == 0);
        REQUIRE(w.hi == 6);
    }
    SECTION("no transport: window passes through") {
        SymmetryComposition off;
        const auto w = plan_build_window(off, 3, 9, 12);
        REQUIRE(w.lo == 3);
        REQUIRE(w.hi == 9);
        REQUIRE_FALSE(w.flip_transport);
    }
    REQUIRE(flip_mirror_target(4, 12) == 8);
}

TEST_CASE("plan_tr_actions: canonical solve, partner copy", "[sector_plan]") {
    const std::size_t nraw = 8;
    SymmetryComposition comp;
    comp.tr_pairing = true;
    comp.tr_partner = {0, 7, 6, 5, 4, 3, 2, 1};  // Z8: k <-> 8-k

    SECTION("plain sector list") {
        std::vector<FakeTag> tags;
        for (std::size_t k = 0; k < nraw; ++k)
            tags.push_back({4, k, 100});
        const auto plan = plan_tr_actions(tags, nraw, comp);
        // k = 0 and k = 4 self-conjugate; of {1,7},{2,6},{3,5} the
        // HIGHER index copies from the lower.
        REQUIRE(plan.n_skipped == 3);
        REQUIRE(plan.skip[7] == 1);
        REQUIRE(plan.source[7] == 1);
        REQUIRE(plan.skip[1] == 0);
        REQUIRE(plan.skip[0] == 0);
        REQUIRE(plan.skip[4] == 0);
    }
    SECTION("flip-synthetic indices pair within their parity block") {
        std::vector<FakeTag> tags;
        for (std::size_t idx = 0; idx < 2 * nraw; ++idx)
            tags.push_back({6, idx, 50});
        const auto plan = plan_tr_actions(tags, nraw, comp);
        REQUIRE(plan.n_skipped == 6);         // 3 pairs per parity block
        REQUIRE(plan.skip[nraw + 7] == 1);
        REQUIRE(plan.source[nraw + 7] == nraw + 1);
    }
    SECTION("dim mismatch forces a solve") {
        std::vector<FakeTag> tags = {{4, 1, 100}, {4, 7, 99}};
        const auto plan = plan_tr_actions(tags, nraw, comp);
        REQUIRE(plan.n_skipped == 0);
    }
    SECTION("partner outside the built set forces a solve") {
        std::vector<FakeTag> tags = {{4, 7, 100}};  // its partner k=1 absent
        const auto plan = plan_tr_actions(tags, nraw, comp);
        REQUIRE(plan.n_skipped == 0);
    }
}
