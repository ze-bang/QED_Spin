// =============================================================================
// tests/unit/test_orbit_table.cpp
//
// OrbitTable guards:
//
//   1. The OrbitTable content hash separates subspaces and groups.
//   2. Burnside sum rule: summing the closed-form survivor counts over
//      all irreps of an abelian group reproduces the subspace dimension
//      exactly (Σ_k dim_k = C(N, n_up), resp. 2^N).
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/symmetry/compiled_group.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/orbit_table.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using namespace ed::symmetry;

namespace {

using Cx = std::complex<double>;

// Survival cutoff on a symmetrised orbit's norm^2.
constexpr double kNormSqEps = 1e-10;

// An abelian group given element by element, with the characters of all of
// its 1-D irreps (chars[k][g] for element g).
struct AbelianGroup {
    std::vector<std::vector<int>> elems;
    std::vector<std::vector<Cx>>  chars;
};

// Z_N ring translations T^m (m = 0 .. N-1): chi_k(T^m) = exp(2 pi i k m / N).
AbelianGroup translations(int N) {
    AbelianGroup g;
    const double two_pi = 2.0 * std::acos(-1.0);
    for (int m = 0; m < N; ++m) g.elems.push_back(ed::sym::translation(N, m));
    for (int k = 0; k < N; ++k) {
        std::vector<Cx> chi;
        for (int m = 0; m < N; ++m) chi.push_back(std::polar(1.0, two_pi * k * m / N));
        g.chars.push_back(std::move(chi));
    }
    return g;
}

// The reflection Z2 {1, R}: chi_+ = (1, 1), chi_- = (1, -1).
AbelianGroup reflection(int N) {
    AbelianGroup g;
    g.elems = {ed::sym::identity(N), ed::sym::reflection_1d(N)};
    g.chars = {{Cx(1, 0), Cx(1, 0)}, {Cx(1, 0), Cx(-1, 0)}};
    return g;
}

CompiledGroup compile(const AbelianGroup& g, int N) {
    return CompiledGroup::from_permutations(g.elems, N);
}

std::uint64_t surviving_total(const OrbitTable& tab, const AbelianGroup& g) {
    std::uint64_t total = 0;
    for (const auto& chi : g.chars)
        for (std::size_t i = 0; i < tab.size(); ++i)
            if (projected_norm_sq(tab, i, chi) > kNormSqEps) ++total;
    return total;
}

}  // namespace

TEST_CASE("OrbitTable content hash distinguishes subspaces and groups",
          "[orbit_table]") {
    const int N = 12;
    const CompiledGroup t = compile(translations(N), N);
    const OrbitTable a = build_orbit_table_fixed_sz_streaming(N, 5, t);
    const OrbitTable b = build_orbit_table_fixed_sz_streaming(N, 6, t);
    const OrbitTable c = build_orbit_table_full_compiled(N, t);
    REQUIRE(a.content_hash != b.content_hash);
    REQUIRE(a.content_hash != c.content_hash);
    const CompiledGroup r = compile(reflection(N), N);
    const OrbitTable d = build_orbit_table_fixed_sz_streaming(N, 5, r);
    REQUIRE(a.content_hash != d.content_hash);
}

TEST_CASE("Burnside sum rule: closed-form sector dims tile the subspace",
          "[orbit_table]") {
    const int N = 12;
    for (const AbelianGroup& g : {translations(N), reflection(N)}) {
        const CompiledGroup cg = compile(g, N);
        SECTION("fixed-Sz, |G|=" + std::to_string(g.elems.size())) {
            for (int n_up : {2, 5, N / 2}) {
                const OrbitTable tab =
                    build_orbit_table_fixed_sz_streaming(N, n_up, cg);
                REQUIRE(surviving_total(tab, g) == tab.subspace_dim);
            }
        }
        SECTION("full space, |G|=" + std::to_string(g.elems.size())) {
            const OrbitTable tab = build_orbit_table_full_compiled(N, cg);
            REQUIRE(surviving_total(tab, g) == (1ULL << N));
        }
    }
}
