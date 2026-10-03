// =============================================================================
// tests/unit/test_orbit_table.cpp
//
// OrbitTable guards:
//
//   1. The OrbitTable content hash separates subspaces and groups.
//   2. Burnside sum rule: summing the closed-form survivor counts over
//      all irreps of an abelian group reproduces the subspace dimension
//      exactly (Σ_k dim_k = C(N, n_up), resp. 2^N).
//   3. The tables do not depend on the thread count.
//   4. A fixed-Sz table builds its rank lookup once.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/basis/compiled_group.h>
#include <ed/basis/group.h>
#include <ed/basis/orbit_table.h>
#include <ed/basis/rep_sector.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ed::symmetry;

namespace {

using Cx = std::complex<double>;

// Survival cutoff on a symmetrised orbit's norm^2.
constexpr double kNormSqEps = 1e-10;

// An abelian group given element by element, with the characters of all of
// its 1-D irreps (chars[k][g] for element g).
struct AbelianGroup {
    std::vector<std::vector<int>> elems;
    std::vector<std::vector<Cx>> chars;
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

CompiledGroup compile(const AbelianGroup& g, int N) { return CompiledGroup::from_permutations(g.elems, N); }

std::uint64_t surviving_total(const OrbitTable& tab, const AbelianGroup& g) {
    std::uint64_t total = 0;
    for (const auto& chi : g.chars)
        for (std::size_t i = 0; i < tab.size(); ++i)
            if (projected_norm_sq_stab(tab.stabilizer_of(i), chi) > kNormSqEps) ++total;
    return total;
}

}  // namespace

TEST_CASE("OrbitTable content hash distinguishes subspaces and groups", "[orbit_table]") {
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

TEST_CASE("Burnside sum rule: closed-form sector dims tile the subspace", "[orbit_table]") {
    const int N = 12;
    for (const AbelianGroup& g : {translations(N), reflection(N)}) {
        const CompiledGroup cg = compile(g, N);
        SECTION("fixed-Sz, |G|=" + std::to_string(g.elems.size())) {
            for (int n_up : {2, 5, N / 2}) {
                const OrbitTable tab = build_orbit_table_fixed_sz_streaming(N, n_up, cg);
                REQUIRE(surviving_total(tab, g) == tab.subspace_dim);
            }
        }
        SECTION("full space, |G|=" + std::to_string(g.elems.size())) {
            const OrbitTable tab = build_orbit_table_full_compiled(N, cg);
            REQUIRE(surviving_total(tab, g) == (1ULL << N));
        }
    }
}

TEST_CASE("Orbit tables are the same at any thread count", "[orbit_table]") {
    // The scan hands chunks of the states to threads dynamically and merges them in state order
    // (audit P1-matvec-cpu-09): reps, stabiliser ids and the stabiliser sets must not depend on
    // how many threads ran it.
    const int N = 18;                                   // C(18, 9) = 48620 states: the parallel path
    const CompiledGroup cg = compile(translations(N), N);
    auto same = [](const OrbitTable& a, const OrbitTable& b) {
        return a.reps == b.reps && a.stab_id == b.stab_id && a.stab_elems == b.stab_elems
               && a.subspace_dim == b.subspace_dim && a.content_hash == b.content_hash;
    };
    auto at = [](int threads, auto build) {
#ifdef _OPENMP
        const int before = omp_get_max_threads();
        omp_set_num_threads(threads);
        OrbitTable t = build();
        omp_set_num_threads(before);
        return t;
#else
        (void)threads;
        return build();
#endif
    };
    const auto fixed = [&] { return build_orbit_table_fixed_sz_streaming(N, N / 2, cg); };
    const auto parity = [&] { return build_orbit_table_parity_compiled(N, 1, cg); };
    const auto full = [&] { return build_orbit_table_full_compiled(N, cg); };
    for (int threads : {2, 7}) {
        REQUIRE(same(at(1, fixed), at(threads, fixed)));
        REQUIRE(same(at(1, parity), at(threads, parity)));
        REQUIRE(same(at(1, full), at(threads, full)));
    }
}

TEST_CASE("A fixed-Sz table keeps one rank lookup", "[orbit_table]") {
    // Built once and kept with the table (P6-engine-setup-07): every walk over the table -- and
    // every call that reuses it from the registry -- shares it.
    const int N = 14, n_up = 7;
    const OrbitTable tab = build_orbit_table_fixed_sz_streaming(N, n_up, compile(translations(N), N));
    const auto a = rank_lookup_of(tab, N, n_up);
    REQUIRE(a != nullptr);
    REQUIRE(rank_lookup_of(tab, N, n_up) == a);
    REQUIRE(tab.bytes() >= a->shared_of_rank.size() * sizeof(std::int32_t));
    const auto fresh = make_shared_rank_lookup(tab.reps, N, n_up);
    REQUIRE(std::equal(a->shared_of_rank.begin(), a->shared_of_rank.end(), fresh->shared_of_rank.begin(),
                       fresh->shared_of_rank.end()));
    for (std::size_t i = 0; i < tab.reps.size(); ++i)
        REQUIRE(a->shared_of_rank[static_cast<std::size_t>(
                    ed::core::combinadic::rank_state(tab.reps[i], N, n_up, a->binom))]
                == static_cast<std::int32_t>(i));
}
