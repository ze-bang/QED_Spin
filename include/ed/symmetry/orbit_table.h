#pragma once
// =============================================================================
// include/ed/symmetry/orbit_table.h
//
// OrbitTable: the single irrep-INDEPENDENT group-level artifact of the
// SymmetryEngine v2 plan (docs/architecture/SYMMETRY_V2_DESIGN.md, Stage 2).
//
// One fused scan over the subspace produces, per orbit:
//
//   * the canonical (min-image) representative, and
//   * its stabilizer Stab(rep) = { g : g(rep) = rep }, stored as an index
//     into a DEDUPED list of stabilizer element sets (distinct stabilizer
//     subgroups number ~1-100 for any realistic lattice group, so the
//     per-rep cost is one uint16).
//
// This fuses the previous "Pass 1" (rep enumeration) and "Pass 1.5"
// (build_orbit_stabilizers) into one pass: a state that survives the
// early-exit min-image test has, in the same |G| loop, already seen every
// element that fixes it. Non-reps keep the early-exit cost; reps pay the
// full |G| walk exactly once (they previously paid it twice).
//
// Everything per-irrep derives from this table in O(#reps):
//
//   norm²(rep, χ) = |Σ_{h ∈ Stab(rep)} χ(h)|² / |Stab(rep)|
//
// (the rep_projection.h closed form, precondition: the full G-orbit of a
// rep lies inside the subspace -- true for the full 2^N space and every
// popcount-preserving fixed-Sz subspace, since the compiled elements here
// are pure site permutations).
//
// ``content_hash`` combines the CompiledGroup hash with the subspace
// signature -- the Stage-3 persistent-cache key.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ed/core/combinadic.h>
#include <ed/symmetry/compiled_group.h>
#include <ed/symmetry/gosper.h>
#include <ed/symmetry/sym_profile.h>
#include <ed/core/symmetry_metadata.h>  // SymmetryGroupInfo

namespace ed::symmetry {

struct OrbitTable {
    std::vector<std::uint64_t> reps;      // canonical reps, ascending
    std::vector<std::uint16_t> stab_id;   // per rep -> index into stab_elems
    std::vector<std::vector<std::uint16_t>> stab_elems;  // deduped stabilizer
                                                         // element-index sets
    std::uint64_t subspace_dim = 0;       // C(N, n_up) or 2^N
    std::uint64_t content_hash = 0;       // (group, subspace, engine version)

    [[nodiscard]] std::size_t size() const noexcept { return reps.size(); }
    [[nodiscard]] bool        empty() const noexcept { return reps.empty(); }

    [[nodiscard]] const std::vector<std::uint16_t>&
    stabilizer_of(std::size_t rep_i) const noexcept {
        return stab_elems[stab_id[rep_i]];
    }
};

/// Closed-form orbit-projected norm² of rep ``rep_i`` in the 1-D irrep with
/// per-element characters ``chi`` (length |G|). Identical math to
/// ``ed::symmetry::projected_norm_sq(OrbitStabilizers, ...)``
/// (rep_projection.h) -- |Σ_{h∈Stab}χ(h)|²/|Stab| with the |Stab|=1 fast
/// path -- just fed from the fused table.
[[nodiscard]] inline double
projected_norm_sq_stab(const std::vector<std::uint16_t>& st,
                       const std::vector<std::complex<double>>& chi) {
    if (st.size() == 1 || chi.empty()) return 1.0;
    std::complex<double> sum(0.0, 0.0);
    for (std::uint16_t g : st) sum += chi[g];
    return std::norm(sum) / static_cast<double>(st.size());
}

[[nodiscard]] inline double
projected_norm_sq(const OrbitTable&                        tab,
                  std::size_t                              rep_i,
                  const std::vector<std::complex<double>>& chi) {
    return projected_norm_sq_stab(tab.stabilizer_of(rep_i), chi);
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

/// Fused per-state visit: rejects non-reps with the same early exit the
/// legacy Pass-1 used; for survivors records the stabilizer element set.
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

}  // namespace detail

/// Fused rep + stabilizer scan over the fixed-Sz subspace (streaming
/// Gosper walk; no C(N,n_up) basis materialization). ``reps`` ascending,
/// bit-identical to ``enumerate_fixed_sz_orbit_reps_streaming``;
/// ``stab_elems`` content-identical to ``build_orbit_stabilizers``.
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

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if (total < (std::uint64_t{1} << 14)) nthreads = 1;
    if (nthreads < 1) nthreads = 1;

    struct Local {
        std::vector<std::uint64_t> reps;
        std::vector<std::uint16_t> stab_id;
        detail::StabDedup          dedup;
    };
    std::vector<Local> local(static_cast<std::size_t>(nthreads));

#ifdef _OPENMP
#   pragma omp parallel num_threads(nthreads)
#endif
    {
        int tid = 0, nt = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nt  = omp_get_num_threads();
#endif
        const std::uint64_t chunk = total / static_cast<std::uint64_t>(nt);
        const std::uint64_t rem   = total % static_cast<std::uint64_t>(nt);
        const std::uint64_t begin =
            static_cast<std::uint64_t>(tid) * chunk +
            std::min<std::uint64_t>(static_cast<std::uint64_t>(tid), rem);
        const std::uint64_t count =
            chunk + (static_cast<std::uint64_t>(tid) < rem ? 1u : 0u);

        Local& out = local[static_cast<std::size_t>(tid)];
        std::vector<std::uint16_t> stab_scratch;
        stab_scratch.reserve(G ? G : 1);
        if (count > 0) {
            std::uint64_t s = ed::core::combinadic::unrank_to_state(
                begin, static_cast<int>(n_bits), n_up, binom);
            for (std::uint64_t i = 0; i < count; ++i) {
                if (G == 0) {
                    out.reps.push_back(s);
                    stab_scratch.assign(1, 0);
                    out.stab_id.push_back(out.dedup.id_of(stab_scratch));
                } else if (detail::visit_state(s, cg, G, stab_scratch)) {
                    out.reps.push_back(s);
                    out.stab_id.push_back(out.dedup.id_of(stab_scratch));
                }
                if (i + 1 < count) s = next_bit_permutation(s);
            }
        }
    }

    // Merge: concatenate ascending thread chunks; remap thread-local
    // stabilizer ids into the global deduped list.
    std::size_t tot = 0;
    for (const auto& l : local) tot += l.reps.size();
    tab.reps.reserve(tot);
    tab.stab_id.reserve(tot);
    detail::StabDedup global;
    for (auto& l : local) {
        std::vector<std::uint16_t> remap(l.dedup.sets.size());
        for (std::size_t k = 0; k < l.dedup.sets.size(); ++k)
            remap[k] = global.id_of(l.dedup.sets[k]);
        tab.reps.insert(tab.reps.end(), l.reps.begin(), l.reps.end());
        for (std::uint16_t id : l.stab_id) tab.stab_id.push_back(remap[id]);
    }
    tab.stab_elems = std::move(global.sets);
    prof.set_items(tab.reps.size());
    return tab;
}

/// SymmetryGroupInfo convenience wrapper (pure site permutations).
[[nodiscard]] inline OrbitTable
build_orbit_table_fixed_sz_streaming(std::uint64_t            n_bits,
                                     int                      n_up,
                                     const SymmetryGroupInfo& info) {
    const CompiledGroup cg = info.max_clique.empty()
        ? CompiledGroup{}
        : CompiledGroup::from_permutations(
              info.max_clique, static_cast<int>(info.max_clique[0].size()));
    return build_orbit_table_fixed_sz_streaming(n_bits, n_up, cg);
}

/// Stage 5b: the flip-extended group G' = G x Z2 -- elements
/// [g_0..g_{|G|-1}, g_0*F, .., g_{|G|-1}*F] with F = XOR all-ones
/// (the global spin flip; it commutes with every site permutation, so
/// this IS the direct product). Only meaningful on the half-filling
/// subspace, where F preserves popcount.
[[nodiscard]] inline CompiledGroup
make_flip_extended_group(const SymmetryGroupInfo& info, std::uint64_t n_bits) {
    std::vector<std::vector<int>> perms = info.max_clique;
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

    const std::size_t G = cg.size();
    tab.content_hash = cg.content_hash()
        ^ (detail::kOrbitTableVersion * 0x9E3779B97F4A7C15ULL)
        ^ (n_bits * 0x2545F4914F6CDD1DULL);

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if (dim < (std::uint64_t{1} << 14)) nthreads = 1;
    if (nthreads < 1) nthreads = 1;

    struct Local {
        std::vector<std::uint64_t> reps;
        std::vector<std::uint16_t> stab_id;
        detail::StabDedup          dedup;
    };
    std::vector<Local> local(static_cast<std::size_t>(nthreads));

#ifdef _OPENMP
#   pragma omp parallel num_threads(nthreads)
#endif
    {
        int tid = 0, nt = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nt  = omp_get_num_threads();
#endif
        const std::uint64_t chunk = dim / static_cast<std::uint64_t>(nt);
        const std::uint64_t rem   = dim % static_cast<std::uint64_t>(nt);
        const std::uint64_t begin =
            static_cast<std::uint64_t>(tid) * chunk +
            std::min<std::uint64_t>(static_cast<std::uint64_t>(tid), rem);
        const std::uint64_t count =
            chunk + (static_cast<std::uint64_t>(tid) < rem ? 1u : 0u);

        Local& out = local[static_cast<std::size_t>(tid)];
        std::vector<std::uint16_t> stab_scratch;
        stab_scratch.reserve(G ? G : 1);
        for (std::uint64_t i = 0; i < count; ++i) {
            const std::uint64_t s = begin + i;
            if (G == 0) {
                out.reps.push_back(s);
                stab_scratch.assign(1, 0);
                out.stab_id.push_back(out.dedup.id_of(stab_scratch));
            } else if (detail::visit_state(s, cg, G, stab_scratch)) {
                out.reps.push_back(s);
                out.stab_id.push_back(out.dedup.id_of(stab_scratch));
            }
        }
    }

    std::size_t tot = 0;
    for (const auto& l : local) tot += l.reps.size();
    tab.reps.reserve(tot);
    tab.stab_id.reserve(tot);
    detail::StabDedup global;
    for (auto& l : local) {
        std::vector<std::uint16_t> remap(l.dedup.sets.size());
        for (std::size_t k = 0; k < l.dedup.sets.size(); ++k)
            remap[k] = global.id_of(l.dedup.sets[k]);
        tab.reps.insert(tab.reps.end(), l.reps.begin(), l.reps.end());
        for (std::uint16_t id : l.stab_id) tab.stab_id.push_back(remap[id]);
    }
    tab.stab_elems = std::move(global.sets);
    prof.set_items(tab.reps.size());
    return tab;
}

/// Fused rep + stabilizer scan over the full 2^N Hilbert space. ``reps``
/// bit-identical to ``enumerate_full_orbit_reps``.
[[nodiscard]] inline OrbitTable
build_orbit_table_full(std::uint64_t n_bits, const SymmetryGroupInfo& info) {
    const CompiledGroup cg = info.max_clique.empty()
        ? CompiledGroup{}
        : CompiledGroup::from_permutations(
              info.max_clique, static_cast<int>(info.max_clique[0].size()));
    return build_orbit_table_full_compiled(n_bits, cg);
}

}  // namespace ed::symmetry
