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

    auto ops = ed::symmetry::build_full_sector_operators(
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
