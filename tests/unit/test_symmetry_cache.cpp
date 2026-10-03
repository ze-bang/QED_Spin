// =============================================================================
// tests/unit/test_symmetry_cache.cpp
//
// Orbit-table registry guards:
//
//   * the acquire_*_compiled front-ends return the table the builders make
//     (the key they compute without building reproduces its content_hash);
//   * acquire_* returns the SAME shared table from the in-process registry on
//     a second call, and a different table for a different subspace.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/basis/compiled_group.h>
#include <ed/basis/group.h>
#include <ed/basis/orbit_table.h>
#include <ed/basis/symmetry_cache.h>

using namespace ed::symmetry;

namespace {

bool tables_equal(const OrbitTable& a, const OrbitTable& b) {
    return a.reps == b.reps && a.stab_id == b.stab_id && a.stab_elems == b.stab_elems
           && a.subspace_dim == b.subspace_dim && a.content_hash == b.content_hash;
}

// Z_N translations of a ring, compiled.
CompiledGroup translations(int N) {
    return CompiledGroup::from_permutations(ed::sym::generate_group({ed::sym::translation(N)}), N);
}

// The dihedral group of the ring (translations + reflection), compiled.
CompiledGroup dihedral(int N) {
    return CompiledGroup::from_permutations(
        ed::sym::generate_group({ed::sym::translation(N), ed::sym::reflection_1d(N)}), N);
}

}  // namespace

TEST_CASE("acquire returns the table the builders make", "[symmetry_cache]") {
    const int N = 12;
    for (const CompiledGroup& cg : {translations(N), dihedral(N)}) {
        for (int n_up : {4, N / 2}) {
            const OrbitTable tab = build_orbit_table_fixed_sz_streaming(N, n_up, cg);
            REQUIRE(tables_equal(tab, *acquire_orbit_table_fixed_sz_compiled(N, n_up, cg)));
        }
        const OrbitTable par = build_orbit_table_parity_compiled(N, 0, cg);
        REQUIRE(tables_equal(par, *acquire_orbit_table_parity_compiled(N, 0, cg)));
        const OrbitTable full = build_orbit_table_full_compiled(N, cg);
        REQUIRE(tables_equal(full, *acquire_orbit_table_full_compiled(N, cg)));
    }
}

TEST_CASE("acquire: in-process registry returns the same shared table", "[symmetry_cache]") {
    const int N = 11;  // distinct N so other tests' registry entries don't alias
    const CompiledGroup cg = translations(N);

    const auto a = acquire_orbit_table_fixed_sz_compiled(N, 5, cg);
    const auto b = acquire_orbit_table_fixed_sz_compiled(N, 5, cg);
    REQUIRE(a.get() == b.get());  // same object, zero rebuild

    // Different subspace: different table.
    const auto c = acquire_orbit_table_fixed_sz_compiled(N, 4, cg);
    REQUIRE(c.get() != a.get());
}

TEST_CASE("orbit table: bytes() counts what the registry budgets", "[symmetry-cache]") {
    ed::symmetry::OrbitTable t;
    t.reps = {1, 2, 3};
    t.stab_id = {0, 0, 1};
    t.stab_elems = {{0}, {0, 1}};
    REQUIRE(t.bytes() == 3 * 8 + 3 * 2 + 3 * 2);
}
