#pragma once
// =============================================================================
// include/ed/symmetry/symmetry_cache.h
//
// In-process sharing for the OrbitTable: a small FIFO of shared_ptr<const
// OrbitTable> keyed by ``OrbitTable::content_hash`` (CompiledGroup element
// hash x subspace signature x engine version -- Hamiltonian couplings
// deliberately excluded, which is why parameter sweeps hit). Repeated
// qed.eigs/thermal/spectrum calls in one process reuse the table without any
// rebuild; tables are immutable once built. Every hit is spot-verified against
// the caller's group and subspace before it is used.
//
// The acquire_* front-ends below are what the sector builders call; the
// raw build_orbit_table_* functions in orbit_table.h remain the compute
// kernels.
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <utility>

#include <ed/symmetry/orbit_table.h>
#include <ed/symmetry/sym_profile.h>

namespace ed::symmetry {

namespace detail {

// ---------------------------------------------------------------------------
// In-process registry: content-keyed FIFO of the most recent tables.
// Small (tables are 10 B/rep; 8 entries cover a full multi-Sz sweep loop),
// mutex-guarded, always on.
// ---------------------------------------------------------------------------
class OrbitTableRegistry {
public:
    static OrbitTableRegistry& instance() {
        static OrbitTableRegistry r;
        return r;
    }

    [[nodiscard]] std::shared_ptr<const OrbitTable> find(std::uint64_t key) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : entries_)
            if (e->content_hash == key) return e;
        return nullptr;
    }

    void insert(std::shared_ptr<const OrbitTable> tab) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& e : entries_)
            if (e->content_hash == tab->content_hash) return;
        entries_.push_back(std::move(tab));
        while (entries_.size() > kMaxEntries) entries_.pop_front();
    }

    /// Drop a poisoned entry (a hit that failed physical verification) so a
    /// rebuilt table can take its key -- ``insert`` dedupes by hash and would
    /// otherwise keep serving the bad entry forever.
    void erase(std::uint64_t key) {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if ((*it)->content_hash == key) { entries_.erase(it); return; }
    }

private:
    static constexpr std::size_t kMaxEntries = 8;
    std::mutex mu_;
    std::deque<std::shared_ptr<const OrbitTable>> entries_;
};

// Physical verification of a cache hit: registry entries are keyed by a
// salted content hash, and correctness must NOT ride on hash quality
// (structured inputs can collide). Spot-verify sampled reps against the CALLER's group + subspace:
// membership (bit range, popcount / parity) and canonical-minimum under the
// group action. A wrong-table hit fails with near-certainty; cost is
// <= 64 x |G| LUT applies, negligible next to any solve.
[[nodiscard]] inline bool
orbit_table_consistent(const OrbitTable&    t,
                       const CompiledGroup& cg,
                       std::uint64_t        n_bits,
                       int                  n_up,
                       int                  parity) noexcept {
    if (t.reps.empty()) return true;
    const std::uint64_t mask =
        (n_bits >= 64) ? ~0ULL : ((std::uint64_t{1} << n_bits) - 1ULL);
    const std::size_t n = t.reps.size();
    const std::size_t samples = std::min<std::size_t>(n, 64);
    for (std::size_t i = 0; i < samples; ++i) {
        const std::size_t idx =
            (samples == 1) ? 0 : i * (n - 1) / (samples - 1);
        const std::uint64_t r = t.reps[idx];
        if (r & ~mask) return false;
        const int pc = __builtin_popcountll(r);
        if (n_up >= 0 && pc != n_up) return false;
        if (parity >= 0 && (pc & 1) != parity) return false;
        for (std::size_t g = 0; g < cg.size(); ++g)
            if (cg.apply(r, g) < r) return false;   // not canonical here
    }
    return true;
}

template <class BuildFn, class VerifyFn>
[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_impl(std::uint64_t key, BuildFn&& build, VerifyFn&& verify) {
    auto& reg = OrbitTableRegistry::instance();
    if (auto hit = reg.find(key)) {
        if (verify(*hit)) {
            if (sym_profile_enabled())
                ED_LOG(Info, "[sym-profile] orbit-table registry HIT (%zu reps)", hit->size());
            return hit;
        }
        ED_LOG(Warn, "[symmetry-cache] orbit-table registry hit FAILED physical "
                     "verification (key collision or stale entry) -- rebuilding");
        reg.erase(key);
    }
    auto tab = std::make_shared<OrbitTable>(build());
    reg.insert(tab);
    return tab;
}

}  // namespace detail

/// Acquire the fixed-Sz orbit table of a CompiledGroup (possibly
/// flip-extended). The key is computed without building the table, so the
/// registry is consulted first (flip elements change the group hash and
/// therefore the key).
[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_fixed_sz_compiled(std::uint64_t        n_bits,
                                      int                  n_up,
                                      const CompiledGroup& cg) {
    const std::uint64_t key = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(n_up + 1) * 0xD6E8FEB86659FD93ULL);
    return detail::acquire_impl(
        key,
        [&] { return build_orbit_table_fixed_sz_streaming(n_bits, n_up, cg); },
        [&](const OrbitTable& t) {
            return detail::orbit_table_consistent(t, cg, n_bits, n_up, -1);
        });
}

[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_parity_compiled(std::uint64_t        n_bits,
                                    int                  parity,
                                    const CompiledGroup& cg) {
    const std::uint64_t key = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(parity + 7) * 0xA24BAED4963EE407ULL);
    return detail::acquire_impl(
        key,
        [&] { return build_orbit_table_parity_compiled(n_bits, parity, cg); },
        [&](const OrbitTable& t) {
            return detail::orbit_table_consistent(t, cg, n_bits, -1, parity);
        });
}

[[nodiscard]] inline std::shared_ptr<const OrbitTable>
acquire_orbit_table_full_compiled(std::uint64_t        n_bits,
                                  const CompiledGroup& cg) {
    const std::uint64_t key = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL);
    return detail::acquire_impl(
        key,
        [&] { return build_orbit_table_full_compiled(n_bits, cg); },
        [&](const OrbitTable& t) {
            return detail::orbit_table_consistent(t, cg, n_bits, -1, -1);
        });
}

}  // namespace ed::symmetry
