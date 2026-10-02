#pragma once
// =============================================================================
// include/ed/basis/orbit_table.h
//
// OrbitTable: the single irrep-INDEPENDENT group-level artifact of the
// symmetry engine.
//
// One fused scan over the subspace produces, per orbit:
//
//   * the canonical (min-image) representative, and
//   * its stabilizer Stab(rep) = { g : g(rep) = rep }, stored as an index
//     into a DEDUPED list of stabilizer element sets (distinct stabilizer
//     subgroups number ~1-100 for any realistic lattice group, so the
//     per-rep cost is one uint16).
//
// Rep enumeration ("pass 1") and stabilizer collection ("pass 1.5") are
// fused into one pass: a state that survives the early-exit min-image test
// has, in the same |G| loop, already seen every element that fixes it.
// Non-reps keep the early-exit cost; reps pay the full |G| walk exactly once.
//
// Everything per-irrep derives from this table in O(#reps):
//
//   norm²(rep, χ) = |Σ_{h ∈ Stab(rep)} χ(h)|² / |Stab(rep)|
//
// (closed form; precondition: the full G-orbit of a rep lies inside the
// subspace -- true for the full 2^N space and every popcount-preserving
// fixed-Sz subspace, since the compiled elements here are pure site
// permutations).
//
// ``content_hash`` combines the CompiledGroup hash with the subspace
// signature -- the persistent-cache key.
// =============================================================================

#include <algorithm>
#include <complex>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ed/basis/combinadic.h>
#include <ed/basis/compiled_group.h>
#include <ed/basis/gosper.h>
#include <ed/basis/sym_profile.h>

namespace ed::symmetry {

struct SharedRankLookup;   // rep_sector.h

struct OrbitTable {
    std::vector<std::uint64_t> reps;      // canonical reps, ascending
    std::vector<std::uint16_t> stab_id;   // per rep -> index into stab_elems
    std::vector<std::vector<std::uint16_t>> stab_elems;  // deduped stabilizer
                                                         // element-index sets
    std::uint64_t subspace_dim = 0;       // C(N, n_up) or 2^N
    std::uint64_t content_hash = 0;       // (group, subspace, engine version)
    /// A fixed-Sz table's dense rank -> rep index lookup (rep_sector.h rank_lookup_of), built on
    /// first use and kept with the table, so every walk over it shares one.
    struct RankSlot {
        std::mutex mu;
        bool tried = false;
        std::shared_ptr<const SharedRankLookup> table;
        std::atomic<std::uint64_t> bytes{0};
    };
    std::shared_ptr<RankSlot> rank_slot = std::make_shared<RankSlot>();

    [[nodiscard]] std::size_t size() const noexcept { return reps.size(); }
    /// Bytes it holds (10 per rep, the stabiliser sets, and its rank lookup once built).
    [[nodiscard]] std::uint64_t bytes() const noexcept {
        std::uint64_t b = reps.size() * sizeof(std::uint64_t) + stab_id.size() * sizeof(std::uint16_t);
        for (const auto& s : stab_elems) b += s.size() * sizeof(std::uint16_t);
        return b + (rank_slot ? rank_slot->bytes.load() : 0);
    }
    [[nodiscard]] bool        empty() const noexcept { return reps.empty(); }

    [[nodiscard]] const std::vector<std::uint16_t>&
    stabilizer_of(std::size_t rep_i) const noexcept {
        return stab_elems[stab_id[rep_i]];
    }
};

/// Closed-form orbit-projected norm² of a rep with stabiliser ``st`` in the 1-D irrep
/// with per-element characters ``chi`` (length |G|): |Σ_{h∈Stab}χ(h)|²/|Stab|, with the
/// |Stab|=1 fast path. For rep i of a table: projected_norm_sq_stab(tab.stabilizer_of(i), chi).
[[nodiscard]] inline double
projected_norm_sq_stab(const std::vector<std::uint16_t>& st,
                       const std::vector<std::complex<double>>& chi) {
    if (st.size() == 1 || chi.empty()) return 1.0;
    std::complex<double> sum(0.0, 0.0);
    for (std::uint16_t g : st) sum += chi[g];
    return std::norm(sum) / static_cast<double>(st.size());
}

namespace detail {

/// Engine version folded into the cache key: bump when the table layout or
/// the canonical-rep convention changes.
inline constexpr std::uint64_t kOrbitTableVersion = 1;

/// Thread-local stabilizer dedup: element-set -> local id.
struct StabDedup {
    std::vector<std::vector<std::uint16_t>>       sets;
    std::unordered_map<std::uint64_t, std::vector<std::uint16_t>> by_hash;

    static std::uint64_t hash_set(const std::vector<std::uint16_t>& v) {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::uint16_t x : v) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        h ^= v.size();
        h *= 1099511628211ULL;
        return h;
    }

    std::uint16_t id_of(const std::vector<std::uint16_t>& st) {
        const std::uint64_t h = hash_set(st);
        auto [it, inserted] = by_hash.try_emplace(h);
        if (inserted) {
            it->second = {static_cast<std::uint16_t>(sets.size())};
            sets.push_back(st);
            return it->second[0];
        }
        // Rare hash-collision guard: linear-verify the recorded ids.
        for (std::uint16_t id : it->second) {
            if (sets[id] == st) return id;
        }
        it->second.push_back(static_cast<std::uint16_t>(sets.size()));
        sets.push_back(st);
        return it->second.back();
    }
};

/// Fused per-state visit: rejects non-reps by early exit on the first
/// smaller image; for survivors records the stabilizer element set.
/// Returns true iff ``s`` is its orbit's canonical rep.
inline bool visit_state(std::uint64_t                s,
                        const CompiledGroup&         cg,
                        std::size_t                  G,
                        std::vector<std::uint16_t>&  stab_scratch) {
    stab_scratch.clear();
    for (std::size_t g = 0; g < G; ++g) {
        const std::uint64_t img = cg.apply(s, g);
        if (img < s) return false;
        if (img == s) stab_scratch.push_back(static_cast<std::uint16_t>(g));
    }
    return true;
}

/// The fused rep + stabiliser scan of every subspace: items 0..total-1 in ascending state order,
/// first(i) the state of item i, next(s) the state after s, keep(s) whether s is in the subspace.
/// The item range is cut into 64 chunks per thread taken dynamically -- a state's cost varies
/// with how early its canonicalisation exits and with its stabiliser, so equal static ranges left
/// threads idle -- and merged in item order: reps ascend, and the stabiliser sets keep their
/// first-occurrence numbering, so the table is the same, bit for bit, at any thread count.
template <class First, class Next, class Keep>
inline void scan_orbits(std::uint64_t total, const CompiledGroup& cg, First first, Next next, Keep keep,
                        OrbitTable& tab) {
    const std::size_t G = cg.size();
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if (total < (std::uint64_t{1} << 14) || nthreads < 1) nthreads = 1;
    const std::uint64_t n_chunks = std::max<std::uint64_t>(
        1, std::min<std::uint64_t>(total, 64ull * static_cast<std::uint64_t>(nthreads)));
    struct Local {
        std::vector<std::uint64_t> reps;
        std::vector<std::uint16_t> stab_id;
        StabDedup                  dedup;
    };
    std::vector<Local> local(static_cast<std::size_t>(n_chunks));
    // chunk c covers items [c total / n_chunks, (c + 1) total / n_chunks), without overflow
    const std::uint64_t q = total / n_chunks, r = total % n_chunks;
    const auto bound = [q, r](std::uint64_t c) { return c * q + std::min(c, r); };
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
    for (long long c = 0; c < static_cast<long long>(n_chunks); ++c) {
        const std::uint64_t begin = bound(static_cast<std::uint64_t>(c));
        const std::uint64_t end   = bound(static_cast<std::uint64_t>(c) + 1);
        Local& out = local[static_cast<std::size_t>(c)];
        std::vector<std::uint16_t> stab_scratch;
        stab_scratch.reserve(G ? G : 1);
        if (begin == end) continue;
        std::uint64_t s = first(begin);
        for (std::uint64_t i = begin; i < end; ++i) {
            if (keep(s)) {
                if (G == 0) {
                    out.reps.push_back(s);
                    stab_scratch.assign(1, 0);
                    out.stab_id.push_back(out.dedup.id_of(stab_scratch));
                } else if (visit_state(s, cg, G, stab_scratch)) {
                    out.reps.push_back(s);
                    out.stab_id.push_back(out.dedup.id_of(stab_scratch));
                }
            }
            if (i + 1 < end) s = next(s);
        }
    }

    // Merge: concatenate the chunks in order; remap chunk-local stabiliser ids into the global list.
    std::size_t tot = 0;
    for (const auto& l : local) tot += l.reps.size();
    tab.reps.reserve(tot);
    tab.stab_id.reserve(tot);
    StabDedup global;
    for (auto& l : local) {
        std::vector<std::uint16_t> remap(l.dedup.sets.size());
        for (std::size_t k = 0; k < l.dedup.sets.size(); ++k)
            remap[k] = global.id_of(l.dedup.sets[k]);
        tab.reps.insert(tab.reps.end(), l.reps.begin(), l.reps.end());
        for (std::uint16_t id : l.stab_id) tab.stab_id.push_back(remap[id]);
        l = Local{};   // release as we go
    }
    tab.stab_elems = std::move(global.sets);
}

}  // namespace detail

/// Fused rep + stabilizer scan over the fixed-Sz subspace (streaming
/// Gosper walk; no C(N,n_up) basis materialization). ``reps`` ascending.
/// Compiled-group core: the caller supplies the (possibly flip-extended)
/// CompiledGroup. PRECONDITION for flip elements: every element must
/// preserve the popcount of every subspace state (the all-ones spin flip
/// does exactly at half filling, n_up == N/2) -- otherwise the min-image
/// convention and the closed-form norms are invalid.
[[nodiscard]] inline OrbitTable
build_orbit_table_fixed_sz_streaming(std::uint64_t        n_bits,
                                     int                  n_up,
                                     const CompiledGroup& cg) {
    SymPhaseTimer prof("pass1+1.5 fused orbit-table (fixed-Sz, streaming)");
    OrbitTable tab;
    if (n_up < 0 || static_cast<std::uint64_t>(n_up) > n_bits) return tab;

    ed::core::combinadic::BinomialTable binom(static_cast<int>(n_bits));
    const std::uint64_t total = binom.at(static_cast<int>(n_bits), n_up);
    tab.subspace_dim = total;
    if (total == 0) return tab;

    const std::size_t G = cg.size();
    tab.content_hash = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(n_up + 1) * 0xD6E8FEB86659FD93ULL);

    if (n_up == 0) {  // state 0 is its own rep; every element fixes it
        tab.reps.push_back(0);
        std::vector<std::uint16_t> st;
        for (std::size_t g = 0; g < G; ++g)
            st.push_back(static_cast<std::uint16_t>(g));
        if (st.empty()) st.push_back(0);
        tab.stab_elems.push_back(std::move(st));
        tab.stab_id.push_back(0);
        prof.set_items(1);
        return tab;
    }

    detail::scan_orbits(
        total, cg,
        [&](std::uint64_t i) {
            return ed::core::combinadic::unrank_to_state(i, static_cast<int>(n_bits), n_up, binom);
        },
        [](std::uint64_t s) { return next_bit_permutation(s); }, [](std::uint64_t) { return true; }, tab);
    prof.set_items(tab.reps.size());
    return tab;
}

/// The flip-extended group G' = G x Z2 -- elements
/// [g_0..g_{|G|-1}, g_0*F, .., g_{|G|-1}*F] with F = XOR all-ones
/// (the global spin flip; it commutes with every site permutation, so
/// this IS the direct product). Only meaningful on subspaces F preserves
/// (half filling; parity halves with N even; the full space).
[[nodiscard]] inline CompiledGroup
make_flip_extended_group_from_perms(std::vector<std::vector<int>> perms,
                                    std::uint64_t                 n_bits) {
    if (perms.empty()) {  // trivial spatial group: identity + flip
        std::vector<int> ident(static_cast<std::size_t>(n_bits));
        for (std::size_t i = 0; i < ident.size(); ++i)
            ident[i] = static_cast<int>(i);
        perms.push_back(ident);
    }
    const std::size_t Gs = perms.size();
    const std::uint64_t all_ones =
        (n_bits >= 64) ? ~0ULL : ((1ULL << n_bits) - 1ULL);
    std::vector<std::vector<int>> perms2 = perms;
    perms2.insert(perms2.end(), perms.begin(), perms.end());
    std::vector<std::uint64_t> flips(2 * Gs, 0ULL);
    for (std::size_t g = Gs; g < 2 * Gs; ++g) flips[g] = all_ones;
    return CompiledGroup::from_elements(perms2, flips,
                                        static_cast<int>(n_bits));
}

/// Fused rep + stabilizer scan over the full 2^N Hilbert space for an
/// arbitrary CompiledGroup (perm (+) flip-mask elements -- the full
/// 2^N space is closed under EVERY such element, so the min-image
/// convention and closed-form norms hold unconditionally). This is the
/// entry the flip-extended full-space sectors use.
[[nodiscard]] inline OrbitTable
build_orbit_table_full_compiled(std::uint64_t        n_bits,
                                const CompiledGroup& cg) {
    SymPhaseTimer prof("pass1+1.5 fused orbit-table (full, compiled)");
    OrbitTable tab;
    const std::uint64_t dim = (1ULL << n_bits);
    tab.subspace_dim = dim;

    tab.content_hash = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL);

    detail::scan_orbits(
        dim, cg, [](std::uint64_t i) { return i; }, [](std::uint64_t s) { return s + 1; },
        [](std::uint64_t) { return true; }, tab);
    prof.set_items(tab.reps.size());
    return tab;
}

/// Fused rep + stabilizer scan over the Sz-PARITY subspace
/// (popcount(s) mod 2 == parity, dim 2^{N-1}). Every CompiledGroup
/// element here must preserve popcount parity: site permutations
/// always do; the all-ones flip does iff N is even (the caller
/// enforces the closure rule).
[[nodiscard]] inline OrbitTable
build_orbit_table_parity_compiled(std::uint64_t        n_bits,
                                  int                  parity,
                                  const CompiledGroup& cg) {
    SymPhaseTimer prof("pass1+1.5 fused orbit-table (Sz-parity)");
    OrbitTable tab;
    const std::uint64_t dim_all = (1ULL << n_bits);
    tab.subspace_dim = dim_all / 2;

    tab.content_hash = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL)
        ^ (static_cast<std::uint64_t>(parity + 7) * 0xA24BAED4963EE407ULL);

    detail::scan_orbits(
        dim_all, cg, [](std::uint64_t i) { return i; }, [](std::uint64_t s) { return s + 1; },
        [parity](std::uint64_t s) { return (static_cast<int>(__builtin_popcountll(s)) & 1) == parity; }, tab);
    prof.set_items(tab.reps.size());
    return tab;
}

}  // namespace ed::symmetry
