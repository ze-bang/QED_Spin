// =============================================================================
// tests/unit/test_symmetry_cache.cpp
//
// Stage-3 guards of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md):
//
//   * orbit_table_key_* reproduces the content_hash the builders stamp
//     (the cache key can be computed without building).
//   * save/load round-trips every field bit-identically.
//   * acquire_* returns the SAME shared table from the in-process
//     registry on a second call, and loads from disk in a fresh key
//     when the registry entry has been evicted (simulated via distinct
//     keys) -- plus a corrupted / truncated file falls back to a clean
//     rebuild rather than wrong data.
//   * resolve_sym_cache_dir implements the documented precedence.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/symmetry/group.h>
#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/symmetry_cache.h>

#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ed::symmetry;
namespace fs = std::filesystem;

namespace {

std::string scratch_dir(const std::string& tag) {
    auto p = fs::temp_directory_path() /
             ("qed_symcache_" + tag + "_" + std::to_string(::getpid()));
    fs::remove_all(p);
    fs::create_directories(p);
    return p.string();
}

bool tables_equal(const OrbitTable& a, const OrbitTable& b) {
    return a.reps == b.reps && a.stab_id == b.stab_id &&
           a.stab_elems == b.stab_elems &&
           a.subspace_dim == b.subspace_dim &&
           a.content_hash == b.content_hash;
}

}  // namespace

TEST_CASE("orbit_table_key_* matches the built table's content_hash",
          "[symmetry_cache]") {
    const int N = 12;
    const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
    for (int n_up : {4, N / 2}) {
        const OrbitTable tab =
            build_orbit_table_fixed_sz_streaming(N, n_up, info);
        REQUIRE(orbit_table_key_fixed_sz(N, n_up, info) == tab.content_hash);
    }
    const OrbitTable full = build_orbit_table_full(N, info);
    REQUIRE(orbit_table_key_full(N, info) == full.content_hash);
}

TEST_CASE("save/load round-trips the OrbitTable bit-identically",
          "[symmetry_cache]") {
    const int N = 12;
    const SymmetryGroupInfo info =
        ed::sym::translation_group_with_reflection_1d(N);
    const OrbitTable tab =
        build_orbit_table_fixed_sz_streaming(N, N / 2, info);
    REQUIRE(!tab.empty());

    const std::string dir = scratch_dir("roundtrip");
    REQUIRE(save_orbit_table(tab, dir));

    const auto loaded = load_orbit_table(tab.content_hash, dir);
    REQUIRE(loaded != nullptr);
    REQUIRE(tables_equal(tab, *loaded));

    // Wrong key: miss, not garbage.
    REQUIRE(load_orbit_table(tab.content_hash ^ 1, dir) == nullptr);
    fs::remove_all(dir);
}

TEST_CASE("corrupted / truncated cache files fall back to rebuild",
          "[symmetry_cache]") {
    const int N = 10;
    const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
    const OrbitTable tab =
        build_orbit_table_fixed_sz_streaming(N, N / 2, info);
    const std::string dir = scratch_dir("corrupt");
    REQUIRE(save_orbit_table(tab, dir));
    const std::string path =
        detail::otab_path(dir, tab.content_hash);

    SECTION("truncated") {
        fs::resize_file(path, fs::file_size(path) / 2);
        REQUIRE(load_orbit_table(tab.content_hash, dir) == nullptr);
    }
    SECTION("bad magic") {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(0);
        f.write("XXXXXXXX", 8);
        f.close();
        REQUIRE(load_orbit_table(tab.content_hash, dir) == nullptr);
    }
    SECTION("hash mismatch inside the file") {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(16);  // content_hash field (after magic + version)
        const std::uint64_t bogus = 0xDEADBEEFULL;
        f.write(reinterpret_cast<const char*>(&bogus), 8);
        f.close();
        REQUIRE(load_orbit_table(tab.content_hash, dir) == nullptr);
    }
    // acquire still yields a correct table (rebuild path).
    const auto rebuilt = acquire_orbit_table_fixed_sz(N, N / 2, info, dir);
    REQUIRE(tables_equal(tab, *rebuilt));
    fs::remove_all(dir);
}

TEST_CASE("acquire: in-process registry returns the same shared table",
          "[symmetry_cache]") {
    const int N = 11;  // distinct N so other tests' registry entries don't alias
    const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);

    const auto a = acquire_orbit_table_fixed_sz(N, 5, info, /*cache_dir=*/{});
    const auto b = acquire_orbit_table_fixed_sz(N, 5, info, /*cache_dir=*/{});
    REQUIRE(a.get() == b.get());  // same object, zero rebuild

    // Different subspace: different table.
    const auto c = acquire_orbit_table_fixed_sz(N, 4, info, {});
    REQUIRE(c.get() != a.get());
}

TEST_CASE("acquire: disk hit in a fresh process is simulated via save+load",
          "[symmetry_cache]") {
    const int N = 13;
    const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
    const std::string dir = scratch_dir("diskhit");

    // Cold: builds and saves.
    const auto cold = acquire_orbit_table_fixed_sz(N, 6, info, dir);
    REQUIRE(fs::exists(detail::otab_path(dir, cold->content_hash)));

    // Simulate a fresh process: bypass the registry by loading directly.
    const auto warm =
        load_orbit_table(orbit_table_key_fixed_sz(N, 6, info), dir);
    REQUIRE(warm != nullptr);
    REQUIRE(tables_equal(*cold, *warm));
    fs::remove_all(dir);
}

TEST_CASE("resolve_sym_cache_dir precedence", "[symmetry_cache]") {
    // NOTE: manipulates the process env; keep assertions local.
    ::unsetenv("ED_SYM_CACHE");
    ::unsetenv("ED_SYM_CACHE_DIR");
    REQUIRE(resolve_sym_cache_dir("", "/lat") == "/lat/basis_cache");
    REQUIRE(resolve_sym_cache_dir("/explicit", "/lat") == "/explicit");
    REQUIRE(resolve_sym_cache_dir("", "") == "");

    ::setenv("ED_SYM_CACHE_DIR", "/ovr", 1);
    REQUIRE(resolve_sym_cache_dir("", "/lat") == "/ovr");
    REQUIRE(resolve_sym_cache_dir("/explicit", "/lat") == "/explicit");
    ::unsetenv("ED_SYM_CACHE_DIR");

    ::setenv("ED_SYM_CACHE", "0", 1);
    REQUIRE(resolve_sym_cache_dir("/explicit", "/lat") == "");
    ::unsetenv("ED_SYM_CACHE");
}
