// =============================================================================
// tests/unit/test_nonabelian_guard.cpp
//
// Stage 0 of the general-symmetry plan: the non-abelian safety guard.
//
// The symmetry-projection layer only implements 1-D (abelian) irreps. Feeding
// non-commuting generators (e.g. translation + reflection = dihedral D_N) used
// to reduce orbits by the full non-abelian group while projecting onto 1-D
// characters -> the d>=2 irrep content was dropped -> incomplete spectrum.
//
// `group_from_generators` now detects this and restricts to a maximal abelian
// subgroup (a complete, correct reduction by a coarser factor). These tests
// prove (a) the group-theory helpers are correct, (b) the restriction fires,
// and (c) the resulting sectors are COMPLETE and reproduce the brute-force
// spectrum.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/symmetry/group.h>
#include <ed/symmetry/sector_set.h>
#include <ed/symmetry/sector_operator.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

using ed::sym::Permutation;
using Complex = std::complex<double>;

namespace {

void add_heisenberg_pbc_terms(ed::symmetry::SectorOperator& op,
                              std::uint64_t N, double J) {
    const Complex J_real(J, 0.0), J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);
        op.addTwoBodyTerm(0, i, 1, j, J_half);
        op.addTwoBodyTerm(1, i, 0, j, J_half);
    }
}

// All eigenvalues of a (small) operator via a dense column-by-column build.
template <class Op>
std::vector<double> all_eigs(const Op& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(d, d);
    std::vector<Complex> e(d, Complex(0, 0)), col(d, Complex(0, 0));
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(e.begin(), e.end(), Complex(0, 0));
        e[j] = Complex(1, 0);
        const_cast<Op&>(op).apply(e.data(), col.data(), d);
        for (std::size_t i = 0; i < d; ++i) H(i, j) = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    std::vector<double> ev(es.eigenvalues().data(),
                           es.eigenvalues().data() + es.eigenvalues().size());
    std::sort(ev.begin(), ev.end());
    return ev;
}

}  // namespace

TEST_CASE("nonabelian: is_abelian + maximal_abelian_subgroup", "[symmetry][nonabelian]") {
    const int N = 6;
    const Permutation T = ed::sym::translation(N, 1);
    const Permutation R = ed::sym::reflection_1d(N);

    // Translation + reflection on a chain do NOT commute (R T R = T^{-1}).
    REQUIRE(ed::sym::is_abelian({T}) == true);
    REQUIRE(ed::sym::is_abelian({T, R}) == false);

    const auto full = ed::sym::generate_group({T, R});   // dihedral D_6, |G|=12
    REQUIRE(full.size() == 2u * static_cast<std::size_t>(N));

    const auto agens = ed::sym::maximal_abelian_subgroup_generators({T, R}, full);
    REQUIRE(ed::sym::is_abelian(agens));
    const auto asub = ed::sym::generate_group(agens);
    REQUIRE(asub.size() == static_cast<std::size_t>(N));   // translations Z_6
    // The retained subgroup must contain the translation (preferred element).
    REQUIRE(std::find(asub.begin(), asub.end(), T) != asub.end());
}

TEST_CASE("nonabelian: abelian restriction returns the MAXIMUM-cardinality "
          "subgroup, not a smaller maximal-by-inclusion one (Z3^2 > Z2^k trap)",
          "[symmetry][nonabelian]") {
    // 3x3 toroidal grid: two independent order-3 translations T1,T2 (Z3 x Z3,
    // order 9) and the transpose reflection R (swaps the two Z3 factors). The
    // full group (Z3 x Z3) :| Z2 is non-abelian (|G| = 18); its MAXIMUM abelian
    // subgroup is the order-9 translation group. This reproduces the 3x3
    // periodic-kagome failure where a plain greedy committed to commuting
    // involutions and returned a SMALLER maximal-by-inclusion subgroup.
    const int n = 9;
    Permutation T1(n), T2(n), R(n);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            T1[3 * r + c] = 3 * ((r + 1) % 3) + c;     // shift rows  (order 3, fpf)
            T2[3 * r + c] = 3 * r + ((c + 1) % 3);     // shift cols  (order 3, fpf)
            R[3 * r + c]  = 3 * c + r;                 // transpose   (order 2)
        }
    REQUIRE(ed::sym::is_abelian({T1, T2}));            // translations commute
    REQUIRE_FALSE(ed::sym::is_abelian({T1, T2, R}));   // ... but R swaps them
    const auto full = ed::sym::generate_group({T1, T2, R});
    REQUIRE(full.size() == 18u);

    // Adversarial `preferred`: lead with the reflection (an involution). The old
    // greedy then committed to a small commuting subgroup (e.g. <R, T1T2> = Z6)
    // and never reached the larger translation group. The robust default
    // restarts from every element and keeps the MAXIMUM, so it must recover the
    // order-9 Z3 x Z3.
    const auto agens = ed::sym::maximal_abelian_subgroup_generators({R}, full);
    REQUIRE(ed::sym::is_abelian(agens));
    REQUIRE(ed::sym::generate_group(agens).size() >= 9u);
}

TEST_CASE("nonabelian: group_from_generators auto-restricts to abelian",
          "[symmetry][nonabelian]") {
    const int N = 6;
    auto info = ed::sym::group_from_generators(
        N, {ed::sym::translation(N, 1), ed::sym::reflection_1d(N)});
    // Restricted to the abelian (translation) subgroup => |max_clique| == N.
    REQUIRE(info.max_clique.size() == static_cast<std::size_t>(N));
    REQUIRE(info.sectors.size() == static_cast<std::size_t>(N));   // N momenta
}

TEST_CASE("nonabelian: ED_SYM_REQUIRE_ABELIAN=1 throws on non-abelian input",
          "[symmetry][nonabelian]") {
    const int N = 6;
    setenv("ED_SYM_REQUIRE_ABELIAN", "1", /*overwrite=*/1);
    REQUIRE_THROWS_AS(
        ed::sym::group_from_generators(
            N, {ed::sym::translation(N, 1), ed::sym::reflection_1d(N)}),
        std::invalid_argument);
    unsetenv("ED_SYM_REQUIRE_ABELIAN");
}

TEST_CASE("nonabelian: restricted sectors are COMPLETE + match brute force",
          "[symmetry][nonabelian]") {
    const int N = 6;
    auto info = ed::sym::group_from_generators(
        N, {ed::sym::translation(N, 1), ed::sym::reflection_1d(N)});

    auto ops = ed::symmetry::build_full_sector_operators_lazy(
        static_cast<std::uint64_t>(N), 0.5f, info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });

    // (1) Completeness: surviving sector dims tile the full Hilbert space.
    std::size_t total = 0;
    std::vector<double> merged;
    for (const auto& op : ops) {
        total += op->dim();
        const auto ev = all_eigs(*op);
        merged.insert(merged.end(), ev.begin(), ev.end());
    }
    REQUIRE(total == (1ULL << N));
    REQUIRE(merged.size() == (1ULL << N));
    std::sort(merged.begin(), merged.end());

    // (2) Spectrum: merged sector spectrum == brute-force full Heisenberg.
    auto full_op = ed_tests::build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    const auto ref = all_eigs(*full_op);
    REQUIRE(ref.size() == merged.size());
    for (std::size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(std::abs(ref[i] - merged[i]) < 1e-9);
    }
}
