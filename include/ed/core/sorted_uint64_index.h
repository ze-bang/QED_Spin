#pragma once

// =============================================================================
// SortedUint64Index -- compact uint64_t -> size_t map         (Phase 3a #5)
// =============================================================================
//
// Drop-in replacement for `std::unordered_map<uint64_t, size_t>` in the
// streaming-symmetry / chunked-symmetry code paths. The only access pattern
// in those paths is:
//
//   1. Build phase: write `m[state] = basis_idx` once per orbit element,
//      while building each sector's basis (one map per sector).
//   2. Lookup phase: `auto it = m.find(state); if (it != m.end()) ...`,
//      called inside the SpMV inner loop (millions of times per Lanczos
//      iteration).
//
// At N = 36 with full point-group + Sz symmetry the dominant sector has
// ~3 x 10^7 representatives. `std::unordered_map<uint64_t, size_t>` carries
// ~32-40 B/entry of overhead (16 B for {key, value}, plus per-bucket
// pointers, plus next-pointers in the open-chaining bucket lists). This
// module stores the same data as two parallel `std::vector`s (struct of
// arrays):
//
//     std::vector<uint64_t>  keys_;     // sorted ascending after finalize()
//     std::vector<size_t>    values_;   // values_[i] is the value for keys_[i]
//
// That's 16 B/entry flat -- ~2x memory reduction at the dominant scale,
// pushing the ceiling on a 64 GB workstation from "barely fits" to
// "comfortably fits". Lookups are `std::lower_bound` on the sorted keys
// (~23 comparisons for 10^7 keys, each on a hot cache line, vs.
// `unordered_map::find` which typically incurs 2-4 L3 misses on a
// multi-million-entry table).
//
// Build pattern:
//
//     SortedUint64Index lookup;
//     lookup.reserve(estimated_n);
//     for (...) {
//         lookup[state] = basis_idx;     // appends, marks unfinalized
//     }
//     lookup.finalize();                 // sort once at the end
//
// Lookup pattern (drop-in for unordered_map):
//
//     const size_t k = lookup.find(state);   // returns kNotFound on miss
//     if (k == SortedUint64Index::kNotFound) continue;
//     // ... use k as size_t ...
//
// Thread safety: build phase is NOT thread-safe (each sector builds its
// own index serially in the existing pass-2). Lookup phase is read-only
// after `finalize()` and is safe to call concurrently from multiple
// threads (the same pattern as the existing `unordered_map` usage --
// matvec parallelism is over rows, all reading the same lookup).
//
// Persistence: this class deliberately does NOT serialize itself; the
// existing HDF5 cache stores the underlying CSR orbit data and rebuilds
// the lookup on load (see streaming_symmetry.h `loadOrbitBasisHDF5`),
// so swapping the in-memory container does not require a schema bump.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ed::core {

class SortedUint64Index {
public:
    /// Sentinel returned by `find()` when the key is not present. Chosen
    /// as `SIZE_MAX` so it cannot collide with a legitimate basis index
    /// in any reasonable simulation (a valid basis index fits in `int64_t`
    /// since `int64_t` is what the rest of the codebase uses for `size`).
    static constexpr std::size_t kNotFound =
        std::numeric_limits<std::size_t>::max();

    SortedUint64Index() = default;

    /// Reserve capacity for `n` entries. Cheap; safe to over-estimate.
    void reserve(std::size_t n) {
        keys_.reserve(n);
        values_.reserve(n);
    }

    /// Append a (key, value) pair. After any insert(), the index is
    /// "unfinalized" and `find()` is undefined; the caller MUST call
    /// `finalize()` before the first lookup. Provided as the cheaper
    /// alternative to `operator[]` when the caller has the value
    /// in hand.
    void insert(std::uint64_t key, std::size_t value) {
        keys_.push_back(key);
        values_.push_back(value);
        finalized_ = false;
    }

    /// Drop-in for `unordered_map<uint64_t, size_t>::operator[]` during
    /// the BUILD phase. Always appends; on `finalize()` the LAST value
    /// for a duplicate key wins (matches `unordered_map::operator[] =`
    /// overwrite semantics). The reference is into the appended slot,
    /// so the caller's idiom `lookup[k] = v` works unchanged.
    std::size_t& operator[](std::uint64_t key) {
        keys_.push_back(key);
        values_.emplace_back();      // default-constructed std::size_t (= 0)
        finalized_ = false;
        return values_.back();
    }

    /// Sort by key + de-duplicate (last value wins). Idempotent: cheap
    /// no-op if the index is already finalized.
    void finalize() {
        if (finalized_) return;
        if (keys_.empty()) {
            finalized_ = true;
            return;
        }
        // Sort an index permutation rather than zipping the two arrays
        // back into a vector<pair> (avoids 16 B/entry temporary in
        // exactly the regime we're trying to relieve).
        std::vector<std::size_t> perm(keys_.size());
        std::iota(perm.begin(), perm.end(), std::size_t{0});
        std::sort(perm.begin(), perm.end(),
                  [this](std::size_t a, std::size_t b) {
                      return keys_[a] < keys_[b];
                  });

        std::vector<std::uint64_t> sk;
        std::vector<std::size_t>   sv;
        sk.reserve(keys_.size());
        sv.reserve(values_.size());
        for (std::size_t i : perm) {
            if (!sk.empty() && sk.back() == keys_[i]) {
                // Duplicate key: last write wins (unordered_map semantics).
                sv.back() = values_[i];
            } else {
                sk.push_back(keys_[i]);
                sv.push_back(values_[i]);
            }
        }
        keys_   = std::move(sk);
        values_ = std::move(sv);
        finalized_ = true;
    }

    /// O(log N) lookup. Returns `kNotFound` if the key is not present.
    /// Precondition: `finalize()` has been called since the last
    /// `insert()` / `operator[]`.
    std::size_t find(std::uint64_t key) const {
        if (keys_.empty()) return kNotFound;
        auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
        if (it == keys_.end() || *it != key) return kNotFound;
        return values_[static_cast<std::size_t>(it - keys_.begin())];
    }

    /// Convenience: presence check.
    bool contains(std::uint64_t key) const { return find(key) != kNotFound; }

    /// Number of distinct keys after `finalize()`. Pre-finalize this
    /// counts every appended pair (including duplicates).
    std::size_t size() const { return keys_.size(); }
    bool empty() const { return keys_.empty(); }

    /// Wipe all entries. Leaves the index "finalized" (empty is
    /// trivially sorted) so the next `find()` returns `kNotFound`
    /// without first needing to call `finalize()`.
    void clear() {
        keys_.clear();
        values_.clear();
        finalized_ = true;
    }

    /// Approximate resident-memory footprint in bytes (capacity-based,
    /// matches what the OS sees). Used by the diagnostic printouts in
    /// streaming_symmetry.h.
    std::size_t size_bytes() const {
        return keys_.capacity() * sizeof(std::uint64_t)
             + values_.capacity() * sizeof(std::size_t);
    }

    /// True iff `finalize()` has been called since the last write.
    /// Test hook only -- production code is expected to finalize once
    /// at the end of a sector build and never read before then.
    bool is_finalized() const { return finalized_; }

    /// Read-only access to the sorted key array, e.g. for diagnostics
    /// or for callers that want to walk the entries in sorted order.
    /// Throws if not finalized.
    const std::vector<std::uint64_t>& keys() const {
        if (!finalized_) {
            throw std::logic_error(
                "SortedUint64Index::keys() called before finalize()");
        }
        return keys_;
    }
    const std::vector<std::size_t>& values() const {
        if (!finalized_) {
            throw std::logic_error(
                "SortedUint64Index::values() called before finalize()");
        }
        return values_;
    }

private:
    std::vector<std::uint64_t> keys_;
    std::vector<std::size_t>   values_;
    bool finalized_ = true;       // empty is trivially "sorted"
};

}  // namespace ed::core
