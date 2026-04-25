// =============================================================================
// test_sorted_uint64_index (Catch2 v3, Phase 3a #5)
//
// Coverage for include/ed/core/sorted_uint64_index.h, the compact
// sorted-vector replacement for `std::unordered_map<uint64_t, size_t>` used
// in the streaming-symmetry basis-lookup hot path.
//
// Sections:
//   1. Empty index returns kNotFound for every probe and is_finalized()=true.
//   2. operator[] / insert() build path matches a reference unordered_map
//      across a 10k-entry random workload (no duplicates).
//   3. Duplicate keys: last-write-wins matches unordered_map::operator[]=.
//   4. find() on a non-finalized index is documented "undefined"; in this
//      build it MUST return kNotFound for any key NOT present in the
//      already-finalized prefix, and we lock down the safer behaviour:
//      keys() / values() throw before finalize().
//   5. clear() leaves the index in a usable post-finalize state.
//   6. size_bytes() is in the right ballpark (16 B/entry for the data,
//      modulo vector capacity rounding).
//   7. Stress: 1M entries, random uint64_t keys, 100k random lookups
//      match an unordered_map oracle bit-for-bit. (Times the lookup loop
//      so a regression here is visible in the test output.)
//   8. Sorted-key invariant: keys() is strictly ascending after finalize().
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/sorted_uint64_index.h>

#include <chrono>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>

using ed::core::SortedUint64Index;

namespace {

std::vector<std::uint64_t> random_unique_keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(n * 2);
    std::vector<std::uint64_t> out;
    out.reserve(n);
    while (out.size() < n) {
        std::uint64_t k = rng();
        if (seen.insert(k).second) {
            out.push_back(k);
        }
    }
    return out;
}

}  // namespace

TEST_CASE("SortedUint64Index: empty index", "[sorted_uint64_index]") {
    SortedUint64Index idx;
    REQUIRE(idx.empty());
    REQUIRE(idx.size() == 0);
    REQUIRE(idx.is_finalized());                    // empty is trivially sorted
    REQUIRE(idx.find(0) == SortedUint64Index::kNotFound);
    REQUIRE(idx.find(0xDEADBEEFu) == SortedUint64Index::kNotFound);
    REQUIRE_FALSE(idx.contains(42));
    REQUIRE(idx.size_bytes() == 0);                 // both vectors have cap 0
}

TEST_CASE("SortedUint64Index: build via operator[] matches unordered_map oracle",
          "[sorted_uint64_index]") {
    constexpr std::size_t N = 10'000;
    auto keys = random_unique_keys(N, /*seed=*/0xA5A5A5A5ULL);

    SortedUint64Index idx;
    idx.reserve(N);
    std::unordered_map<std::uint64_t, std::size_t> oracle;
    oracle.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        idx[keys[i]]   = i;        // operator[] = idiom (matches existing call sites)
        oracle[keys[i]] = i;
    }

    REQUIRE_FALSE(idx.is_finalized());
    idx.finalize();
    REQUIRE(idx.is_finalized());
    REQUIRE(idx.size() == N);

    for (const auto& [k, v] : oracle) {
        REQUIRE(idx.find(k) == v);
    }
    // Probe a handful of keys we KNOW are absent.
    SECTION("misses return kNotFound") {
        std::mt19937_64 rng(0xBEEFULL);
        std::size_t misses = 0;
        for (std::size_t i = 0; i < 1000; ++i) {
            std::uint64_t k = rng();
            if (oracle.count(k) == 0) {
                REQUIRE(idx.find(k) == SortedUint64Index::kNotFound);
                ++misses;
            }
        }
        // With only 10k 64-bit keys the random probe should miss
        // essentially always; sanity check we exercised the path.
        REQUIRE(misses > 900);
    }
}

TEST_CASE("SortedUint64Index: insert() build path", "[sorted_uint64_index]") {
    SortedUint64Index idx;
    idx.insert(7, 100);
    idx.insert(2, 200);
    idx.insert(9, 300);
    idx.insert(5, 400);
    idx.finalize();

    REQUIRE(idx.size() == 4);
    REQUIRE(idx.find(2) == 200);
    REQUIRE(idx.find(5) == 400);
    REQUIRE(idx.find(7) == 100);
    REQUIRE(idx.find(9) == 300);
    REQUIRE(idx.find(0) == SortedUint64Index::kNotFound);
    REQUIRE(idx.find(8) == SortedUint64Index::kNotFound);

    // Sorted-key invariant.
    const auto& k = idx.keys();
    for (std::size_t i = 1; i < k.size(); ++i) {
        REQUIRE(k[i - 1] < k[i]);
    }
}

TEST_CASE("SortedUint64Index: duplicate keys, last write wins",
          "[sorted_uint64_index]") {
    SortedUint64Index idx;
    idx[42] = 1;
    idx[42] = 2;
    idx[42] = 3;
    idx[7]  = 100;
    idx[42] = 99;          // last write
    idx.finalize();

    REQUIRE(idx.size() == 2);
    REQUIRE(idx.find(42) == 99);
    REQUIRE(idx.find(7) == 100);
}

TEST_CASE("SortedUint64Index: keys() / values() throw before finalize",
          "[sorted_uint64_index]") {
    SortedUint64Index idx;
    idx[1] = 10;
    REQUIRE_FALSE(idx.is_finalized());
    REQUIRE_THROWS_AS(idx.keys(),   std::logic_error);
    REQUIRE_THROWS_AS(idx.values(), std::logic_error);
    idx.finalize();
    REQUIRE_NOTHROW(idx.keys());
    REQUIRE_NOTHROW(idx.values());
}

TEST_CASE("SortedUint64Index: clear leaves usable empty state",
          "[sorted_uint64_index]") {
    SortedUint64Index idx;
    for (std::uint64_t i = 0; i < 100; ++i) idx.insert(i, i * 2);
    idx.finalize();
    REQUIRE(idx.size() == 100);

    idx.clear();
    REQUIRE(idx.empty());
    REQUIRE(idx.is_finalized());                 // safe to find() right away
    REQUIRE(idx.find(50) == SortedUint64Index::kNotFound);
}

TEST_CASE("SortedUint64Index: size_bytes is ~16 B/entry (cap-bounded)",
          "[sorted_uint64_index]") {
    SortedUint64Index idx;
    constexpr std::size_t N = 1024;
    idx.reserve(N);
    for (std::uint64_t i = 0; i < N; ++i) idx.insert(i, i);
    idx.finalize();

    // 16 B = sizeof(uint64_t) + sizeof(size_t) per entry. Vector capacity
    // can over-allocate by up to ~2x in the worst case, so we bound with
    // a generous upper limit and a tight lower limit.
    const std::size_t bytes = idx.size_bytes();
    REQUIRE(bytes >= 16 * N);
    REQUIRE(bytes <= 32 * N + 64);
}

TEST_CASE("SortedUint64Index: 1M-entry stress vs unordered_map oracle",
          "[sorted_uint64_index][stress]") {
    constexpr std::size_t N = 1'000'000;
    constexpr std::size_t Q = 100'000;

    auto keys = random_unique_keys(N, /*seed=*/0xC0FFEEULL);

    SortedUint64Index idx;
    idx.reserve(N);
    std::unordered_map<std::uint64_t, std::size_t> oracle;
    oracle.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        idx.insert(keys[i], i);
        oracle.emplace(keys[i], i);
    }
    idx.finalize();

    std::mt19937_64 rng(0xDEADBEEFULL);
    std::vector<std::uint64_t> probes(Q);
    // Half hits, half misses.
    for (std::size_t i = 0; i < Q; ++i) {
        if (i & 1u) {
            probes[i] = keys[rng() % N];
        } else {
            std::uint64_t k = rng();
            // make sure the miss is actually a miss
            while (oracle.count(k)) k = rng();
            probes[i] = k;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::size_t hits = 0;
    for (std::uint64_t p : probes) {
        std::size_t v = idx.find(p);
        auto it = oracle.find(p);
        if (it == oracle.end()) {
            REQUIRE(v == SortedUint64Index::kNotFound);
        } else {
            REQUIRE(v == it->second);
            ++hits;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    INFO("SortedUint64Index 1M-entry probe time: " << ms << " ms for "
         << Q << " probes (" << hits << " hits)");
    REQUIRE(hits > 0);

    // Memory savings sanity check: 16 B/entry vs unordered_map's
    // ~32-40 B/entry. We don't have a portable handle on
    // unordered_map's footprint, so just assert the absolute lower
    // bound on our own.
    REQUIRE(idx.size_bytes() >= 16 * N);
    REQUIRE(idx.size_bytes() <= 64 * N);     // generous upper bound
}
