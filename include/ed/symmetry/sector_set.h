#pragma once
// =============================================================================
// include/ed/symmetry/sector_set.h
//
// SymmetrySectorSet: the multi-sector enumerator that turns a
// ``SymmetryGroupInfo`` + a Hamiltonian term-builder into a vector of
// standalone per-sector ``SectorOperator`` objects -- one owning operator
// per non-vanishing symmetry sector.
//
// This is the P5 building block that replaces the
// ``StreamingSymmetryOperator::generateSymmetrySectorsStreaming`` +
// nested ``SectorView`` fan-out: instead of one monolithic operator that
// bakes every sector's orbit data into its class identity and exposes
// each sector through a back-referencing view, the caller gets a flat
// list of independent ``ed::LinearOperator``s that the orchestrator can
// solve / thermal / spectral-analyse uniformly.
//
// Two construction lanes, mirroring the two legacy streaming operators:
//   * ``build_full_sector_operators``   -- orbits over the full 2^N
//        Hilbert space (twin of ``StreamingSymmetryOperator``).
//   * ``build_fixed_sz_sector_operators`` -- orbits over a fixed-Sz
//        subspace (twin of ``FixedSzStreamingSymmetryOperator``).
//
// Both take a ``TermBuilder`` callback that populates each freshly-built
// ``SectorOperator``'s inherited term list (via ``addOneBodyTerm`` /
// ``addTwoBodyTerm`` / ``addThreeBodyTerm``). The callback runs once per
// surviving sector; the SAME Hamiltonian is applied to every sector
// (the basis differs, not the terms).
//
// Orbit-rep enumeration is lifted here from the legacy
// ``unique_orbit_reps_`` "Pass 1" (and the test-local helpers) so it is
// reusable and bit-identical:
//   * full space:   a state b is a rep iff b == min_{g in G} g(b).
//   * fixed-Sz:     same, restricted to in-subspace images (site
//                   permutations preserve popcount, so every orbit image
//                   of a fixed-Sz state is itself a fixed-Sz state).
//
// P5 of the operator-collapse refactor (Jun 2026).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include <ed/core/basis_utils.h>          // generateFixedSzBasis, LinIndexTable, applyPermutation
#include <ed/core/combinadic.h>           // BinomialTable, unrank_to_state (streaming partition)
#include <ed/symmetry/compiled_group.h>   // CompiledGroup (Stage 1, SymmetryEngine v2)
#include <ed/symmetry/orbit_table.h>      // OrbitTable, fused pass1+1.5 (Stage 2)
#include <ed/symmetry/symmetry_cache.h>   // acquire_orbit_table_* (Stage 3 cache)
#include <ed/symmetry/gosper.h>           // next_bit_permutation (streaming enumeration)
#include <ed/symmetry/sym_profile.h>      // SymPhaseTimer (ED_SYM_PROFILE=1)
#include <ed/symmetry/fixed_sz_membership.h>  // FixedSzMembershipSubspace (tableless)
#include <ed/symmetry/symmetry_sector_data.h>  // SymmetrySector
#include <ed/symmetry/projector.h>
#include <ed/symmetry/rep_projection.h>   // OrbitStabilizers, projected_norm_sq
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/subspace.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// Stage 1 (SymmetryEngine v2): compile the group's site permutations to the
// byte-LUT form once per construction entry point. Empty-safe: an empty
// max_clique yields an empty CompiledGroup and the enumeration loops below
// degenerate to "every state is its own rep", matching the legacy scalar
// path.
// ---------------------------------------------------------------------------
[[nodiscard]] inline CompiledGroup
compile_spatial_group(const SymmetryGroupInfo& info) {
    if (info.max_clique.empty()) return {};
    return CompiledGroup::from_permutations(
        info.max_clique, static_cast<int>(info.max_clique[0].size()));
}

// ---------------------------------------------------------------------------
// Orbit-rep enumeration (reusable; bit-identical to the legacy Pass 1).
// ---------------------------------------------------------------------------

/// Deduplicated, ascending orbit representatives over the full 2^N Hilbert
/// space. A state ``b`` is its orbit's canonical rep iff ``b`` is the numeric
/// minimum over all group images ``g(b)``. Matches the legacy
/// ``StreamingSymmetryOperator::unique_orbit_reps_`` enumeration.
[[nodiscard]] inline std::vector<std::uint64_t>
enumerate_full_orbit_reps(const SymmetryGroupInfo& info, std::uint64_t n_bits)
{
    SymPhaseTimer prof("pass1 rep-scan (full)");
    std::vector<std::uint64_t> reps;
    const std::uint64_t dim = (1ULL << n_bits);
    const std::size_t   G   = info.max_clique.size();
    const CompiledGroup cg  = compile_spatial_group(info);

    // Parallel orbit-min scan over [0, 2^N). Each thread scans a contiguous
    // ascending range and collects its local reps; concatenating chunks in
    // thread order preserves global ascending order (bit-identical to the
    // serial scan). O(#reps) memory; no full-space materialization.
    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if (dim < (std::uint64_t{1} << 14)) nthreads = 1;
    if (nthreads < 1) nthreads = 1;

    std::vector<std::vector<std::uint64_t>> local(
        static_cast<std::size_t>(nthreads));

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
        std::vector<std::uint64_t>& out = local[static_cast<std::size_t>(tid)];
        for (std::uint64_t i = 0; i < count; ++i) {
            const std::uint64_t s = begin + i;
            bool is_rep = true;
            for (std::size_t g = 0; g < G; ++g) {
                if (cg.apply(s, g) < s) { is_rep = false; break; }
            }
            if (is_rep) out.push_back(s);
        }
    }

    std::size_t tot = 0;
    for (const auto& v : local) tot += v.size();
    reps.reserve(tot);
    for (auto& v : local) reps.insert(reps.end(), v.begin(), v.end());
    prof.set_items(reps.size());
    return reps;
}

/// Deduplicated, ascending fixed-Sz orbit representatives. A fixed-Sz basis
/// state is a rep iff it is the numeric minimum of its orbit restricted to
/// in-subspace images. Matches the legacy
/// ``FixedSzStreamingSymmetryOperator::unique_orbit_reps_`` enumeration.
[[nodiscard]] inline std::vector<std::uint64_t>
enumerate_fixed_sz_orbit_reps(const FixedSzSubspace& sub,
                              const SymmetryGroupInfo& info)
{
    // A fixed-Sz state ``s`` is its orbit's canonical rep iff ``s`` is the
    // numeric minimum over its in-subspace images -- so we can detect reps
    // in a single pass (no global dedup container) and only the survivors
    // enter the output. Site permutations preserve popcount, so every orbit
    // image of a fixed-Sz state is itself in the subspace; the explicit
    // ``index_of`` guard is kept for defensive parity with the legacy loop.
    // Sorted-vector output (ascending) replaces the old ``std::set`` insert
    // (O(n log n) once vs O(n log n) with per-insert tree allocations).
    SymPhaseTimer prof("pass1 rep-scan (fixed-Sz, materialized)");
    const CompiledGroup cg = compile_spatial_group(info);
    std::vector<std::uint64_t> reps;
    for (std::uint64_t s : sub.basis_states()) {
        bool is_rep = true;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            const std::uint64_t img = cg.apply(s, g);
            if (sub.index_of(img) >= 0 && img < s) { is_rep = false; break; }
        }
        if (is_rep) reps.push_back(s);
    }
    std::sort(reps.begin(), reps.end());
    reps.erase(std::unique(reps.begin(), reps.end()), reps.end());
    return reps;
}

/// Streaming, materialization-free twin of `enumerate_fixed_sz_orbit_reps`.
///
/// Walks every length-`n_bits` state of popcount `n_up` via Gosper's hack (no
/// C(N,n_up) basis vector, no LinIndexTable) and keeps the orbit minima:
/// ``s`` is a rep iff ``s == min_g applyPermutation(s, info.max_clique[g])``
/// (early-break on the first smaller image). Since site permutations preserve
/// popcount, every image is in-subspace, so no subspace membership lookup is
/// needed -- the result is bit-identical (same reps, ascending order) to the
/// legacy `enumerate_fixed_sz_orbit_reps(FixedSzSubspace::build(...), info)`.
///
/// OpenMP-parallel: the combinadic rank range [0, C(n,n_up)) is split across
/// threads, each unranking its start state and iterating Gosper for its share;
/// per-thread reps are concatenated in ascending rank order. O(#reps) memory.
[[nodiscard]] inline std::vector<std::uint64_t>
enumerate_fixed_sz_orbit_reps_streaming(std::uint64_t            n_bits,
                                        int                      n_up,
                                        const SymmetryGroupInfo& info)
{
    std::vector<std::uint64_t> reps;
    if (n_up < 0 || static_cast<std::uint64_t>(n_up) > n_bits) return reps;
    if (n_up == 0) { reps.push_back(0); return reps; }  // state 0 is its own rep

    ed::core::combinadic::BinomialTable binom(static_cast<int>(n_bits));
    const std::uint64_t total =
        binom.at(static_cast<int>(n_bits), n_up);
    if (total == 0) return reps;

    SymPhaseTimer prof("pass1 rep-scan (fixed-Sz, streaming)");
    const std::size_t   G  = info.max_clique.size();
    const CompiledGroup cg = compile_spatial_group(info);
    auto is_rep = [&](std::uint64_t s) -> bool {
        for (std::size_t g = 0; g < G; ++g) {
            if (cg.apply(s, g) < s) return false;
        }
        return true;
    };

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    if (total < (std::uint64_t{1} << 14)) nthreads = 1;  // serial for tiny ranges
    if (nthreads < 1) nthreads = 1;

    std::vector<std::vector<std::uint64_t>> local(
        static_cast<std::size_t>(nthreads));

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

        std::vector<std::uint64_t>& out = local[static_cast<std::size_t>(tid)];
        if (count > 0) {
            std::uint64_t s = ed::core::combinadic::unrank_to_state(
                begin, static_cast<int>(n_bits), n_up, binom);
            for (std::uint64_t i = 0; i < count; ++i) {
                if (is_rep(s)) out.push_back(s);
                if (i + 1 < count) s = next_bit_permutation(s);
            }
        }
    }

    std::size_t tot = 0;
    for (const auto& v : local) tot += v.size();
    reps.reserve(tot);
    for (auto& v : local) reps.insert(reps.end(), v.begin(), v.end());
    prof.set_items(reps.size());
    return reps;  // already globally ascending (thread chunks are ascending)
}

// Stage 10e: the ED_SYM_STREAMING_ENUM / ED_SYM_FUSED_PASS15 bisection
// gates are RETIRED -- the fused streaming orbit-table lane has been the
// only tested production path since Stage 2 landed, and every retained
// legacy branch was untested unintended-behaviour surface. The legacy
// enumerators below survive solely as the bit-identity references
// exercised by test_compiled_group.cpp.

// ---------------------------------------------------------------------------
// rep_sector_data_from_sector: extract the CSR-free on-the-fly representative
// description ("On-the-fly representative SpMV" plan, Jun 2026) from an
// already-built ``SymmetrySector`` + its owning group info.
//
// Reads ONLY ``orbit_rep`` + ``norm`` from each basis state (cheap, O(dim);
// never touches the orbit images), and composes the |G| per-sector characters
// + flattened group permutations from the group info. ``n_up`` is the shared
// popcount of the representatives (uniform for a fixed-Sz sector); it stays
// ``-1`` for a full-Hilbert sym-only sector, which makes the result NOT usable
// (the rep matvec needs a fixed-Sz combinadic rank table) so those sectors
// transparently fall back to the orbit-CSR mirror.
// ---------------------------------------------------------------------------
[[nodiscard]] inline RepSectorData
rep_sector_data_from_sector(const ::SymmetrySector&  sec,
                            const SymmetryGroupInfo& info,
                            int                      n_sites)
{
    RepSectorData d;
    d.n_sites    = n_sites;
    d.group_size = static_cast<int>(info.max_clique.size());
    d.reps.reserve(sec.basis_states.size());
    d.inv_norms.reserve(sec.basis_states.size());
    int  n_up    = -1;
    bool uniform = true;
    for (const auto& bs : sec.basis_states) {
        d.reps.push_back(bs.orbit_rep);
        d.inv_norms.push_back(bs.norm > 0.0 ? 1.0 / bs.norm : 0.0);
        const int pc = __builtin_popcountll(bs.orbit_rep);
        if (n_up < 0) n_up = pc;
        else if (pc != n_up) uniform = false;
    }
    d.n_up = uniform ? n_up : -1;
    if (!info.power_representation.empty() && !sec.phase_factors.empty()) {
        d.characters = sector_characters_from(info, sec.phase_factors);
    }
    d.perms_flat = flatten_group_perms(info, n_sites);
    return d;
}

// ---------------------------------------------------------------------------
// Per-sector SectorOperator factory.
//
// ``TermBuilder`` is any callable ``void(SectorOperator&)`` that appends the
// Hamiltonian terms to a freshly-built sector operator (typically a lambda
// closing over the lattice couplings, calling ``op.addTwoBodyTerm(...)`` etc).
// One SectorOperator is produced per surviving sector (``SectorBasis::dim()
// > 0``); empty sectors (all orbits cancel in the irrep) are skipped, so the
// returned vector may be shorter than ``info.sectors.size()``.
// ---------------------------------------------------------------------------

// NOTE (Stage 11c-2a, Jul 2026): the EAGER builders
// (``build_full_sector_operators`` / ``build_fixed_sz_sector_operators``)
// and the ``make_sector_operator_adopt`` / ``SectorBasis::adopt`` bridge
// were deleted -- the lazy rep-first builders below are the only
// construction path, and the unit suites that used the eager lane as a
// reference now pin the same invariants (dims, tiling, ordering, Bethe GS)
// on the lazy operators, materialising via ``ensureHostCsr()`` where they
// inspect orbit data. ``SectorBasis::build`` stays: it is the lazy
// ``csr_provider``'s on-demand materialiser.

// ---------------------------------------------------------------------------
// build_fixed_sz_sector_operators_lazy: CSR-FREE direct-enumeration twin of
// ``build_fixed_sz_sector_operators`` (operator-collapse Phase 3, Jun 2026).
//
// Where the eager builder materialises every sector's full orbit CSR up-front
// (~24 GiB/sector at N=32), this builder produces ``SectorOperator``s that:
//   * know their dimension + real/complex character up-front from a CSR-free
//     "Pass 1.5" orbit-norm scan (walk each rep's orbit, keep only the norm,
//     discard the orbit images), and
//   * defer the GPU RepSectorData / host orbit CSR exactly like the streaming
//     handle's ``make_rep_sector_operator_lazy`` -- but WITHOUT a
//     ``FixedSzStreamingSymmetryOperator`` carrier. The deferred providers
//     instead co-own (via shared_ptr) the group info, the fixed-Sz subspace,
//     and the orbit-rep list, so they stay valid for the operators' whole
//     lifetime even though ``make_sector_operators`` builds nothing else.
//
// This is the Phase 3 piece that lets the production sector loop drop the
// streaming carrier as a *public* type while preserving the memory
// optimisation: the RepSectorData is precomputed (cheap, CSR-free) and the
// CPU ``apply`` fallback rebuilds the orbit CSR on demand via the same
// ``SectorBasis::build`` the eager lane uses -- so a GPU-only run never
// allocates the host orbit CSR.
// ---------------------------------------------------------------------------
// build_rep_sectors_over_table: THE one flip-aware rep-sector constructor
// (structural consolidation, Jul 2026).
//
// Given a subspace's orbit table (built with either the plain spatial
// CompiledGroup or the flip-extended G' = G x Z2 -- fixed-Sz N/2, full
// 2^N, or a parity half), emit one CSR-free rep sector per
// (irrep k [, flip sign]): chi' = (chi, +/-chi) over the doubled
// element list when ``n_signs == 2``, closed-form norms from the
// table's deduped stabilizers, rd.n_up as supplied (-1 = no popcount
// filter). Slot layout of the RETURNED vector: idx = k + sign_i *
// num_irreps (callers add their own slot base for multi-subspace
// sets). Replaces three copy-adapted builder blocks (the historical
// N/2 projection, the full-space flip sectors, and the parity loop).
// ---------------------------------------------------------------------------
template <class TermBuilder>
[[nodiscard]] inline std::vector<std::unique_ptr<SectorOperator>>
build_rep_sectors_over_table(
    std::uint64_t                              n_bits,
    float                                      spin_l,
    const std::shared_ptr<SymmetryGroupInfo>&  info_sp,
    const std::shared_ptr<const OrbitTable>&   otab,
    int                                        n_signs,   // 1 plain, 2 flip
    int                                        rd_n_up,   // -1 or fixed n_up
    const std::shared_ptr<const SharedRankLookup>& srl,
    TermBuilder&&                              terms,
    const char*                                csr_unavailable_msg)
{
    const std::size_t num_irreps =
        std::max<std::size_t>(info_sp->sectors.size(), 1);
    auto reps_fp = std::shared_ptr<const std::vector<std::uint64_t>>(
        otab, &otab->reps);

    // Element tables: spatial perms (doubled with all-ones masks for
    // the flip-extended case).
    const std::vector<int> base_perms =
        flatten_group_perms(*info_sp, static_cast<int>(n_bits));
    std::vector<int> perms_use = base_perms;
    if (perms_use.empty()) {
        perms_use.resize(n_bits);
        for (std::uint64_t i = 0; i < n_bits; ++i)
            perms_use[i] = static_cast<int>(i);
    }
    const std::size_t Gs = perms_use.size() / n_bits;
    std::vector<std::uint64_t> flips_use;
    if (n_signs == 2) {
        perms_use.insert(perms_use.end(), perms_use.begin(),
                         perms_use.begin()
                             + static_cast<std::ptrdiff_t>(Gs * n_bits));
        const std::uint64_t all_ones =
            (n_bits >= 64) ? ~0ULL : ((1ULL << n_bits) - 1ULL);
        flips_use.assign(2 * Gs, 0ULL);
        for (std::size_t g = Gs; g < 2 * Gs; ++g) flips_use[g] = all_ones;
    }
    const std::size_t G_use = (n_signs == 2) ? 2 * Gs : Gs;

    const std::size_t n_slots = num_irreps * static_cast<std::size_t>(n_signs);
    std::vector<std::unique_ptr<SectorOperator>> slots(n_slots);

    #pragma omp parallel for schedule(dynamic) if(n_slots > 1)
    for (std::ptrdiff_t si = 0;
         si < static_cast<std::ptrdiff_t>(n_slots); ++si) {
        const std::size_t idx    = static_cast<std::size_t>(si);
        const std::size_t k      = idx % num_irreps;
        const int         fsign  = (idx / num_irreps == 0) ? +1 : -1;

        std::vector<Complex> chi(G_use, Complex(1.0, 0.0));
        if (!info_sp->sectors.empty()) {
            const std::vector<Complex> chi_k = sector_characters_from(
                *info_sp, info_sp->sectors[k].phase_factors);
            for (std::size_t g = 0; g < Gs; ++g) {
                chi[g] = chi_k[g];
                if (n_signs == 2)
                    chi[Gs + g] = static_cast<double>(fsign) * chi_k[g];
            }
        } else if (n_signs == 2) {
            chi[1] = Complex(static_cast<double>(fsign), 0.0);
        }

        RepSectorData rd;
        rd.n_sites    = static_cast<int>(n_bits);
        rd.group_size = static_cast<int>(G_use);
        rd.reps.reserve(reps_fp->size());
        rd.inv_norms.reserve(reps_fp->size());
        if (srl) {
            rd.shared_rank = srl;
            rd.local_of_shared.assign(reps_fp->size(), std::int32_t{-1});
        }
        for (std::size_t i = 0; i < reps_fp->size(); ++i) {
            const double norm_sq = projected_norm_sq_stab(
                otab->stabilizer_of(i), chi);
            if (norm_sq <= SectorBasis::kOrbitNormSqEpsilon) continue;
            rd.reps.push_back((*reps_fp)[i]);
            rd.inv_norms.push_back(1.0 / std::sqrt(norm_sq));
            if (srl) {
                rd.local_of_shared[i] =
                    static_cast<std::int32_t>(rd.reps.size() - 1);
            }
        }
        if (rd.reps.empty()) continue;

        rd.n_up       = rd_n_up;
        rd.characters = chi;
        rd.perms_flat = perms_use;
        if (n_signs == 2) rd.flip_masks = flips_use;

        bool is_real = true;
        for (const auto& c : rd.characters)
            if (std::abs(c.imag()) > 1e-12) { is_real = false; break; }
        const std::uint64_t dim =
            static_cast<std::uint64_t>(rd.reps.size());
        auto rep_sp = std::make_shared<RepSectorData>(std::move(rd));

        auto op = std::make_unique<SectorOperator>(
            n_bits, spin_l, SectorBasis{});
        terms(*op);
        const std::string msg = csr_unavailable_msg;
        op->configureRepLazy(
            dim, G_use, is_real,
            [rep_sp]() { return *rep_sp; },
            [msg]() -> ::SymmetrySector {
                throw std::runtime_error(msg);
            },
            /*csr_available=*/false);
        slots[idx] = std::move(op);
    }
    return slots;
}

// ---------------------------------------------------------------------------
// build_parity_sector_operators_lazy: Sz-PARITY sectors (monomial-group
// consolidation, Jul 2026).
//
// For a Hamiltonian whose every term changes n_up by an EVEN amount
// (S+S+ / S-S- anisotropic exchange breaking U(1)), the diagonal Z2
// remnant (-1)^{n_up} splits the space into two 2^{N-1} halves. Each
// half x spatial irrep is ONE RepSectorData with ``n_up = -1`` (no
// popcount filter -- H never leaves the half) whose reps come from the
// parity-filtered fused scan; the CPU/GPU rep lanes consume it through
// the existing full-space (n_up = -1) machinery unchanged.
//
// Closure rule: the all-ones flip preserves popcount parity iff N is
// even, so ``flip_sectors`` additionally splits every (parity, k) into
// (parity, k, +/-) via the flip-extended group -- exactly the N/2 and
// full-space constructions. (For odd N the caller must not request
// flip sectors here; flip is then a parity <-> parity transport map.)
//
// ``which_parity``: 0 = even only, 1 = odd only, -1 = BOTH (one sector
// set covering the whole space -- the pooled-GS / thermal / full-dense
// mode). Synthetic ids: k + slot * num_irreps with slots enumerating
// (parity, flip sign); tag quantum numbers append the parity label
// (+1 even / -1 odd) and, when flip sectors are on, the flip sign.
// ---------------------------------------------------------------------------
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_parity_sector_operators_lazy(std::uint64_t            n_bits,
                                   float                    spin_l,
                                   int                      which_parity,
                                   const SymmetryGroupInfo& info,
                                   TermBuilder&&            terms,
                                   std::vector<std::size_t>* out_sector_ids = nullptr,
                                   const std::string&       cache_dir = {},
                                   bool                     flip_sectors = false)
{
    auto info_sp = std::make_shared<SymmetryGroupInfo>(info);
    std::vector<std::unique_ptr<SectorOperator>> ops;
    if (flip_sectors && (n_bits % 2 != 0)) {
        throw std::runtime_error(
            "build_parity_sector_operators_lazy: the all-ones flip only "
            "preserves Sz parity for even N (closure rule).");
    }
    const std::size_t num_irreps =
        std::max<std::size_t>(info_sp->sectors.size(), 1);
    const int n_signs = flip_sectors ? 2 : 1;
    const CompiledGroup cg = flip_sectors
        ? make_flip_extended_group(*info_sp, n_bits)
        : (info_sp->max_clique.empty()
               ? CompiledGroup{}
               : CompiledGroup::from_permutations(
                     info_sp->max_clique, static_cast<int>(n_bits)));

    const int p_lo = (which_parity < 0) ? 0 : which_parity;
    const int p_hi = (which_parity < 0) ? 1 : which_parity;
    std::size_t slot_base = 0;
    for (int parity = p_lo; parity <= p_hi; ++parity) {
        auto ptab = acquire_orbit_table_parity_compiled(
            n_bits, parity, cg, cache_dir);
        auto slots = build_rep_sectors_over_table(
            n_bits, spin_l, info_sp, ptab, n_signs, /*rd_n_up=*/-1,
            /*srl=*/nullptr, terms,
            "Sz-parity sector: the orbit-CSR lane does not carry parity "
            "subspaces; keep the default rep lane (unset ED_SYM_REP=0).");
        for (std::size_t idx = 0; idx < slots.size(); ++idx) {
            if (!slots[idx]) continue;
            if (out_sector_ids)
                out_sector_ids->push_back(
                    (idx % num_irreps)
                    + (slot_base + idx / num_irreps) * num_irreps);
            ops.push_back(std::move(slots[idx]));
        }
        slot_base += static_cast<std::size_t>(n_signs);
    }
    return ops;
}

// ---------------------------------------------------------------------------
// Forward declaration (defined below): the fixed-Sz lazy builder delegates
// its Stage-8c flip-projected half-filling case to the all-Sz builder.
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_all_sz_sector_operators(
    std::uint64_t n_bits, float spin_l, const SymmetryGroupInfo& info,
    TermBuilder&& terms, std::int64_t n_up_min = 0,
    std::int64_t n_up_max = -1,
    std::vector<std::pair<int, std::size_t>>* out_n_up_sector_ids = nullptr,
    const std::string& cache_dir = {}, bool flip_project_half = false);

template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_fixed_sz_sector_operators_lazy(std::uint64_t            n_bits,
                                     float                    spin_l,
                                     std::int64_t             n_up,
                                     const SymmetryGroupInfo& info,
                                     TermBuilder&&            terms,
                                     std::vector<std::size_t>* out_sector_ids = nullptr,
                                     int                      mpi_rank = 0,
                                     int                      mpi_size = 1,
                                     const std::vector<int>*  sector_owner = nullptr,
                                     const std::string&       cache_dir = {},
                                     bool                     flip_project_half = false)
{
    // Stage 8c (SymmetryEngine v2): flip-projected half-filling block for
    // the SINGLE-Sz lane (GS solve). Delegates to the all-Sz builder's
    // proven (k, +/-) machinery restricted to this one n_up. Single-rank
    // only (the all-Sz builder carries no MPI sector ownership); callers
    // gate on eigenvalues-only workloads upstream (the flip-projected
    // eigenvectors live in the projected basis and the orbit-CSR
    // reconstruction lane is not flip-aware).
    if (flip_project_half && mpi_size == 1
        && n_up * 2 == static_cast<std::int64_t>(n_bits)) {
        std::vector<std::pair<int, std::size_t>> ids;
        auto ops = build_all_sz_sector_operators(
            n_bits, spin_l, info, std::forward<TermBuilder>(terms),
            n_up, n_up, &ids, cache_dir, /*flip_project_half=*/true);
        if (out_sector_ids) {
            out_sector_ids->clear();
            for (const auto& [nu, idx] : ids) {
                (void)nu;
                out_sector_ids->push_back(idx);
            }
        }
        return ops;
    }
    // Streaming construction (Jun 2026): the orbit expansion uses a tableless
    // popcount-membership subspace -- bit-identical to ``FixedSzSubspace`` here
    // because every site-permutation image of a fixed-Sz state preserves
    // popcount, so the membership guard in ``compute_orbit_for_state`` behaves
    // identically -- and it is trivially copyable, so the deferred CSR provider
    // captures it by value instead of co-owning a materialized C(N,n_up) basis
    // vector + Lin table (the ~72 GB wall at N=36). Rep enumeration is the
    // streaming Gosper scan by default; ``ED_SYM_STREAMING_ENUM=0`` falls back
    // to the legacy materialized enumerator (built solely to enumerate, then
    // discarded).
    auto info_sp = std::make_shared<SymmetryGroupInfo>(info);
    const FixedSzMembershipSubspace subspace(n_bits, static_cast<int>(n_up));

    // Stage 2 (SymmetryEngine v2): the default lane runs ONE fused
    // pass1+1.5 scan (reps + deduped stabilizers in the same |G| walk).
    // ``ED_SYM_FUSED_PASS15=0`` / ``ED_SYM_STREAMING_ENUM=0`` retain the
    // legacy two-pass / materialized variants for bisection.
    // Stage 10e: the fused streaming orbit table is the only production
    // lane (legacy enumerators remain as test references only).

    // Stage 3: the fused lane acquires the (registry/disk-cached) shared
    // table; ``reps_sp`` is an ALIASING shared_ptr into it, so the deferred
    // CSR providers co-own the whole immutable table with zero copies.
    std::shared_ptr<const std::vector<std::uint64_t>> reps_sp;
    std::shared_ptr<const OrbitTable> otab_sp;
    {
        otab_sp = acquire_orbit_table_fixed_sz(
            n_bits, static_cast<int>(n_up), *info_sp, cache_dir);
        reps_sp = std::shared_ptr<const std::vector<std::uint64_t>>(
            otab_sp, &otab_sp->reps);
    }
    const SpatialProjector  projector(*info_sp);
    const std::size_t        group_size = info_sp->max_clique.size();
    const std::size_t        num_sectors = info_sp->sectors.size();

    // Stage 4: ONE shared rank table per (N, n_up) -- every irrep sector
    // co-owns it and carries only the small local remap, replacing the
    // per-sector C(N,n_up) x int32 dense tables. Budget-gated like the
    // legacy per-sector build (rep_rank_table_enabled).
    std::shared_ptr<const SharedRankLookup> srl;
    {
        ed::core::combinadic::BinomialTable b(static_cast<int>(n_bits));
        if (rep_rank_table_enabled(
                b.at(static_cast<int>(n_bits), static_cast<int>(n_up)))) {
            SymPhaseTimer prof("stage4 shared rank table (one per n_up)");
            srl = make_shared_rank_lookup(
                *reps_sp, static_cast<int>(n_bits), static_cast<int>(n_up));
        }
    }

    // Fix 4: Compute perms_flat once; reuse across all sectors.
    const std::vector<int> shared_perms_flat =
        flatten_group_perms(*info_sp, static_cast<int>(n_bits));

    // Stage 2: the per-irrep Pass 1.5 loop is embarrassingly parallel (each
    // sector reads only shared read-only data and writes its own slot).
    // TermBuilder must be safe to invoke concurrently on distinct operators
    // (all in-tree builders are copy-only), matching the all-Sz lane's
    // existing contract.
    std::vector<std::unique_ptr<SectorOperator>> slot(num_sectors);
    #pragma omp parallel for schedule(dynamic) if(num_sectors > 1)
    for (std::ptrdiff_t si = 0;
         si < static_cast<std::ptrdiff_t>(num_sectors); ++si) {
        const std::size_t s = static_cast<std::size_t>(si);
        // Across-sector MPI: run the per-sector Pass 1.5 orbit walk only for
        // this rank's raw sectors (distributes the walk + the per-sector
        // RepSectorData memory). The shared rep enumeration above stays per-rank.
        if (mpi_size > 1) {
            const int owner_r = sector_owner ? (*sector_owner)[s]
                : static_cast<int>(s % static_cast<std::size_t>(mpi_size));
            if (owner_r != mpi_rank) continue;
        }
        const std::vector<Complex>& phase = info_sp->sectors[s].phase_factors;

        // Pass 1.5: CSR-free per-sector dimension + RepSectorData. Walk each
        // rep's orbit, keep ONLY (rep, 1/norm) for the survivors, discard the
        // orbit images. Same survival cutoff + ordering as SectorBasis::build,
        // so the rep-index <-> basis-index mapping is bit-identical.
        RepSectorData rd;
        rd.n_sites    = static_cast<int>(n_bits);
        rd.group_size = static_cast<int>(group_size);
        rd.reps.reserve(reps_sp->size());
        rd.inv_norms.reserve(reps_sp->size());
        if (srl) {  // Stage 4: per-sector local remap over the shared reps
            rd.shared_rank = srl;
            rd.local_of_shared.assign(reps_sp->size(), std::int32_t{-1});
        }
        int  sec_n_up = -1;
        bool uniform  = true;
        // χ_s(g) over the group (used both for the fused norm and stored on rd
        // below); computed once per sector here so the fused rep loop is O(1) per
        // generic rep and O(|Stab|) for the rare non-trivial-stabilizer reps.
        const std::vector<Complex> chi =
            sector_characters_from(*info_sp, phase);
        for (std::size_t i = 0; i < reps_sp->size(); ++i) {
            const std::uint64_t rep = (*reps_sp)[i];
            double norm_sq;
            {
                norm_sq =
                    projected_norm_sq_stab(otab_sp->stabilizer_of(i), chi);
                if (norm_sq <= SectorBasis::kOrbitNormSqEpsilon) continue;
            }
            rd.reps.push_back(rep);
            rd.inv_norms.push_back(1.0 / std::sqrt(norm_sq));
            if (srl) {
                rd.local_of_shared[i] =
                    static_cast<std::int32_t>(rd.reps.size() - 1);
            }
            const int pc = __builtin_popcountll(rep);
            if (sec_n_up < 0) sec_n_up = pc;
            else if (pc != sec_n_up) uniform = false;
        }
        if (rd.reps.empty()) continue;  // empty sector: nothing to solve

        rd.n_up = uniform ? sec_n_up : -1;
        if (!info_sp->power_representation.empty() && !phase.empty()) {
            rd.characters = chi;
        }
        rd.perms_flat = shared_perms_flat;  // Fix 4: reuse pre-computed copy

        bool is_real = true;
        for (const auto& c : rd.characters) {
            if (std::abs(c.imag()) > 1e-12) { is_real = false; break; }
        }

        const std::uint64_t dim = static_cast<std::uint64_t>(rd.reps.size());
        auto rep_sp = std::make_shared<RepSectorData>(std::move(rd));

        auto op = std::make_unique<SectorOperator>(n_bits, spin_l, SectorBasis{});
        terms(*op);
        op->configureRepLazy(
            dim, group_size, is_real,
            /*rep_provider=*/[rep_sp]() { return *rep_sp; },
            /*csr_provider=*/[reps_sp, info_sp, n_bits, n_up, s]()
                -> ::SymmetrySector {
                // First CPU ``apply`` only: rebuild this sector's orbit CSR
                // with the same enumerator the eager lane uses, over the
                // tableless membership subspace (no materialized basis vector).
                const FixedSzMembershipSubspace sub(n_bits,
                                                    static_cast<int>(n_up));
                const SpatialProjector proj(*info_sp);
                return SectorBasis::build(
                           sub, proj,
                           info_sp->sectors[s].quantum_numbers,
                           info_sp->sectors[s].phase_factors,
                           *reps_sp, /*sector_id=*/s)
                    .sector();
            });
        slot[s] = std::move(op);
    }
    std::vector<std::unique_ptr<SectorOperator>> ops;
    ops.reserve(num_sectors);
    for (std::size_t s = 0; s < num_sectors; ++s) {
        if (!slot[s]) continue;
        ops.push_back(std::move(slot[s]));
        if (out_sector_ids) out_sector_ids->push_back(s);
    }
    return ops;
}

// ---------------------------------------------------------------------------
// build_full_sector_operators_lazy: CSR-free rep-walk twin of
// ``build_full_sector_operators`` for the FULL 2^N space (no Sz restriction).
//
// Mirrors ``build_fixed_sz_sector_operators_lazy`` but over ``FullSpaceSubspace``
// + ``enumerate_full_orbit_reps``: O(#reps) memory (no materialized orbit CSR),
// the same stabilizer-fused Pass 1.5 via the shared rep_projection primitive, and
// a deferred CSR provider. This is the memory-bounded, ≈|G|×-fast construction
// lane for PURE SPATIAL symmetry on non-Sz-conserving (or Sz-unrestricted) models
// at large N -- the case the eager full builder cannot reach (it materializes the
// orbit CSR over the full 2^N space). ``rd.n_up`` is forced to -1 so the rep-walk
// policy applies NO popcount filter (every state lives in the full space; a
// non-conserving term may connect different popcounts).
//
// Closed-form precondition holds: every G-orbit image of a full-space state is
// itself in the full space, so no image is ever dropped (parity-tested in
// tests/unit/test_rep_projection.cpp, full-space case).
// ---------------------------------------------------------------------------
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_full_sector_operators_lazy(std::uint64_t            n_bits,
                                 float                    spin_l,
                                 const SymmetryGroupInfo& info,
                                 TermBuilder&&            terms,
                                 std::vector<std::size_t>* out_sector_ids = nullptr,
                                 int                      mpi_rank = 0,
                                 int                      mpi_size = 1,
                                 const std::vector<int>*  sector_owner = nullptr,
                                 const std::string&       cache_dir = {},
                                 bool                     flip_sectors = false)
{
    auto info_sp = std::make_shared<SymmetryGroupInfo>(info);
    const FullSpaceSubspace subspace(n_bits);

    // -----------------------------------------------------------------
    // FULL-SPACE prod-sigma^x sectors (monomial-group consolidation,
    // Jul 2026). When U(1) Sz is broken but [H, X] == 0 (X = prod
    // sigma^x -- e.g. anisotropic S+S+ exchange), the full 2^N space
    // still splits by the flip quantum number: extend G to
    // G' = G x Z2 exactly as the fixed-Sz N/2 projection does (flip is
    // one more CompiledGroup element: identity perm (+) all-ones XOR
    // mask; the full space is trivially closed under it) and emit
    // (k, +/-) sectors with chi' = (chi, +/-chi). Halves every irrep
    // block on top of the spatial reduction. CPU/GPU rep lanes only
    // (the orbit-CSR fallback throws, as at N/2); single-rank.
    // -----------------------------------------------------------------
    if (flip_sectors && mpi_size == 1) {
        const CompiledGroup cg_flip =
            make_flip_extended_group(*info_sp, n_bits);
        auto flip_tab = acquire_orbit_table_full_compiled(
            n_bits, cg_flip, cache_dir);
        auto slots = build_rep_sectors_over_table(
            n_bits, spin_l, info_sp, flip_tab, /*n_signs=*/2,
            /*rd_n_up=*/-1, /*srl=*/nullptr, terms,
            "flip-projected full-space sector: the orbit-CSR lane is not "
            "flip-aware. Keep the default rep lane (unset ED_SYM_REP=0) "
            "or disable the flip projection (ED_SYM_SPIN_FLIP_PROJECT=0).");
        std::vector<std::unique_ptr<SectorOperator>> ops;
        for (std::size_t idx = 0; idx < slots.size(); ++idx) {
            if (!slots[idx]) continue;
            if (out_sector_ids) out_sector_ids->push_back(idx);
            ops.push_back(std::move(slots[idx]));
        }
        (void)mpi_rank; (void)sector_owner;
        return ops;
    }

    // Stage 2 (SymmetryEngine v2): fused pass1+1.5 + irrep-parallel Pass 1.5
    // loop. Stage 10e: the fused lane is the only production path.
    std::shared_ptr<const std::vector<std::uint64_t>> reps_sp;
    std::shared_ptr<const OrbitTable> otab_sp;
    {
        otab_sp = acquire_orbit_table_full(n_bits, *info_sp, cache_dir);
        reps_sp = std::shared_ptr<const std::vector<std::uint64_t>>(
            otab_sp, &otab_sp->reps);
    }

    const SpatialProjector projector(*info_sp);
    const std::size_t group_size  = info_sp->max_clique.size();
    const std::size_t num_sectors = info_sp->sectors.size();

    const std::vector<int> shared_perms_flat =
        flatten_group_perms(*info_sp, static_cast<int>(n_bits));

    std::vector<std::unique_ptr<SectorOperator>> slot(num_sectors);
    #pragma omp parallel for schedule(dynamic) if(num_sectors > 1)
    for (std::ptrdiff_t si = 0;
         si < static_cast<std::ptrdiff_t>(num_sectors); ++si) {
        const std::size_t s = static_cast<std::size_t>(si);
        if (mpi_size > 1) {
            const int owner_r = sector_owner ? (*sector_owner)[s]
                : static_cast<int>(s % static_cast<std::size_t>(mpi_size));
            if (owner_r != mpi_rank) continue;
        }
        const std::vector<Complex>& phase = info_sp->sectors[s].phase_factors;
        RepSectorData rd;
        rd.n_sites    = static_cast<int>(n_bits);
        rd.group_size = static_cast<int>(group_size);
        rd.reps.reserve(reps_sp->size());
        rd.inv_norms.reserve(reps_sp->size());
        const std::vector<Complex> chi =
            sector_characters_from(*info_sp, phase);
        for (std::size_t i = 0; i < reps_sp->size(); ++i) {
            const std::uint64_t rep = (*reps_sp)[i];
            double norm_sq;
            {
                norm_sq = projected_norm_sq_stab(otab_sp->stabilizer_of(i), chi);
                if (norm_sq <= SectorBasis::kOrbitNormSqEpsilon) continue;
            }
            rd.reps.push_back(rep);
            rd.inv_norms.push_back(1.0 / std::sqrt(norm_sq));
        }
        if (rd.reps.empty()) continue;

        rd.n_up = -1;   // full space: NO popcount filter in the rep-walk policy
        if (!info_sp->power_representation.empty() && !phase.empty())
            rd.characters = chi;
        rd.perms_flat = shared_perms_flat;

        bool is_real = true;
        for (const auto& c : rd.characters)
            if (std::abs(c.imag()) > 1e-12) { is_real = false; break; }

        const std::uint64_t dim = static_cast<std::uint64_t>(rd.reps.size());
        auto rep_sp = std::make_shared<RepSectorData>(std::move(rd));
        auto op = std::make_unique<SectorOperator>(n_bits, spin_l, SectorBasis{});
        terms(*op);
        op->configureRepLazy(
            dim, group_size, is_real,
            /*rep_provider=*/[rep_sp]() { return *rep_sp; },
            /*csr_provider=*/[reps_sp, info_sp, n_bits, s]() -> ::SymmetrySector {
                const FullSpaceSubspace sub(n_bits);
                const SpatialProjector  proj(*info_sp);
                return SectorBasis::build(
                           sub, proj,
                           info_sp->sectors[s].quantum_numbers,
                           info_sp->sectors[s].phase_factors,
                           *reps_sp, /*sector_id=*/s)
                    .sector();
            });
        slot[s] = std::move(op);
    }
    std::vector<std::unique_ptr<SectorOperator>> ops;
    ops.reserve(num_sectors);
    for (std::size_t s = 0; s < num_sectors; ++s) {
        if (!slot[s]) continue;
        ops.push_back(std::move(slot[s]));
        if (out_sector_ids) out_sector_ids->push_back(s);
    }
    return ops;
}

// ---------------------------------------------------------------------------
// build_all_sz_sector_operators: one-pass all-Sz lazy builder.
//
// Calls ``enumerate_full_orbit_reps`` ONCE (O(2^N × |G|)), partitions the
// reps by popcount (site permutations preserve Sz), then runs the same
// CSR-free "Pass 1.5" as ``build_fixed_sz_sector_operators_lazy`` for every
// (n_up, irrep) sector in [n_up_min, n_up_max].
//
// The orbit CSR is deferred via ``configureRepLazy`` so that:
//   * The build phase is O(2^N × |G|) -- a single full-space scan instead
//     of N+1 fixed-Sz scans.
//   * The CSR is materialised on the first ``apply()`` call per sector,
//     which happens inside the (possibly OMP-parallel) thermal/solve loop,
//     so CSR data is in L1/L2 cache immediately when the Krylov iteration
//     starts.
//
// ``out_n_up_sector_ids`` receives (n_up, raw_irrep_index) pairs in the same
// order as the returned operators so callers can attach SectorTags.
// ---------------------------------------------------------------------------
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_all_sz_sector_operators(
    std::uint64_t            n_bits,
    float                    spin_l,
    const SymmetryGroupInfo& info,
    TermBuilder&&            terms,
    std::int64_t             n_up_min,
    std::int64_t             n_up_max,
    std::vector<std::pair<int, std::size_t>>* out_n_up_sector_ids,
    const std::string&       cache_dir,
    bool                     flip_project_half)
{
    if (n_up_max < 0) n_up_max = static_cast<std::int64_t>(n_bits);
    n_up_min = std::max<std::int64_t>(0, n_up_min);
    n_up_max = std::min<std::int64_t>(
        n_up_max, static_cast<std::int64_t>(n_bits));

    // Co-own the group info for all deferred CSR provider lambdas.
    auto info_sp = std::make_shared<SymmetryGroupInfo>(info);

    // One-pass full-space orbit rep enumeration. Site permutations preserve
    // popcount (lattice automorphisms, not spin flips), so every orbit image
    // of a fixed-Sz state has the same n_up. Partitioning by popcount gives
    // exactly the fixed-Sz reps for each n_up without N+1 separate scans.
    {
        // Stage 2 (SymmetryEngine v2): ONE fused pass1+1.5 scan over the full
        // space. A state's stabilizer is a property of the state under the
        // group -- independent of which Sz subspace we later view it in -- so
        // the full-space stabilizer ids partition by popcount alongside the
        // reps, and the per-n_up ``build_orbit_stabilizers`` rebuilds vanish.
        const std::shared_ptr<const OrbitTable> otab_full_sp =
            acquire_orbit_table_full(n_bits, *info_sp, cache_dir);
        const std::vector<std::uint64_t>& all_reps = otab_full_sp->reps;
        // (populated into per-n_up rep + stab-id lists below)
        std::vector<std::vector<std::uint64_t>> reps_by_n_up(
            static_cast<std::size_t>(n_bits + 1));
        std::vector<std::vector<std::uint16_t>> stabid_by_n_up(
            static_cast<std::size_t>(n_bits + 1));
        for (std::size_t k = 0; k < all_reps.size(); ++k) {
            const std::uint64_t r = all_reps[k];
            const auto pc =
                static_cast<std::size_t>(__builtin_popcountll(r));
            if (pc < reps_by_n_up.size()) {
                reps_by_n_up[pc].push_back(r);
                stabid_by_n_up[pc].push_back(otab_full_sp->stab_id[k]);
            }
        }

        const SpatialProjector projector(*info_sp);
        const std::size_t group_size = info_sp->max_clique.size();
        const std::size_t num_irreps = info_sp->sectors.size();

        std::vector<std::unique_ptr<SectorOperator>> ops;
        ops.reserve(
            static_cast<std::size_t>(n_up_max - n_up_min + 1) * num_irreps);

        // Fix 4: Compute the group permutation table once for all sectors.
        // All (n_up, irrep) sectors share the same group, so the 156 separate
        // flatten_group_perms() calls (each allocating |G|*N ints) collapse to
        // one allocation.  Each RepSectorData still owns its own copy (required
        // by RepSymmetryBasisPolicy::perms raw-pointer), but the source is only
        // computed once.
        const std::vector<int> shared_perms_flat =
            flatten_group_perms(*info_sp, static_cast<int>(n_bits));

        for (std::int64_t n_up = n_up_min; n_up <= n_up_max; ++n_up) {
            const auto nu = static_cast<std::size_t>(n_up);
            if (nu >= reps_by_n_up.size() || reps_by_n_up[nu].empty())
                continue;

            // -----------------------------------------------------------------
            // Stage 5b (SymmetryEngine v2): flip-projected half-filling block.
            // At n_up == N/2 the global spin flip F preserves the subspace, so
            // the group extends to G' = G x Z2 (F commutes with every site
            // permutation). Each spatial irrep k splits into (k, +) and (k, -)
            // with chi' = (chi_k, +/-chi_k), halving the LARGEST Sz block's
            // sector dims. CPU rep lane only (the caller gates on that); the
            // orbit-CSR fallback is unsupported for flip sectors and throws.
            // -----------------------------------------------------------------
            if (flip_project_half
                && n_up * 2 == static_cast<std::int64_t>(n_bits)) {
                const CompiledGroup cg_flip =
                    make_flip_extended_group(*info_sp, n_bits);
                auto flip_tab = acquire_orbit_table_fixed_sz_compiled(
                    n_bits, static_cast<int>(n_up), cg_flip, cache_dir);
                std::shared_ptr<const SharedRankLookup> srl_f;
                {
                    ed::core::combinadic::BinomialTable b(
                        static_cast<int>(n_bits));
                    if (rep_rank_table_enabled(b.at(static_cast<int>(n_bits),
                                                    static_cast<int>(n_up)))) {
                        srl_f = make_shared_rank_lookup(
                            flip_tab->reps, static_cast<int>(n_bits),
                            static_cast<int>(n_up));
                    }
                }
                auto slots = build_rep_sectors_over_table(
                    n_bits, spin_l, info_sp, flip_tab, /*n_signs=*/2,
                    static_cast<int>(n_up), srl_f, terms,
                    "flip-projected sector: the orbit-CSR lane is not "
                    "flip-aware. Keep the default rep lane (unset "
                    "ED_SYM_REP=0) or disable the in-sector flip "
                    "projection (ED_SYM_SPIN_FLIP_PROJECT=0).");
                for (std::size_t idx = 0; idx < slots.size(); ++idx) {
                    if (!slots[idx]) continue;
                    if (out_n_up_sector_ids)
                        out_n_up_sector_ids->push_back(
                            {static_cast<int>(n_up), idx});
                    ops.push_back(std::move(slots[idx]));
                }
                continue;   // plain (unprojected) N/2 block replaced
            }

            // Tableless popcount-membership subspace for the orbit expansion
            // (bit-identical to FixedSzSubspace for fixed-Sz; no materialized
            // C(N,n_up) basis vector / Lin table). Trivially copyable -> shared
            // read-only by the OMP region and captured by value in each
            // deferred CSR provider.
            auto reps_sp = std::make_shared<std::vector<std::uint64_t>>(
                reps_by_n_up[nu]);

            const FixedSzMembershipSubspace subspace(
                n_bits, static_cast<int>(n_up));

            // Stabilizer-fused Pass 1.5: the per-n_up stabilizer ids were
            // partitioned out of the ONE full-space fused scan above
            // (popcount-preserving subspace => closed-form precondition
            // holds); each irrep below reads its norm straight from them.
            const std::vector<std::uint16_t>& stab_ids = stabid_by_n_up[nu];

            // Stage 4: one shared rank table per n_up, co-owned by all
            // |G| irrep sectors of this Sz block (budget-gated).
            std::shared_ptr<const SharedRankLookup> srl;
            {
                ed::core::combinadic::BinomialTable b(static_cast<int>(n_bits));
                if (rep_rank_table_enabled(b.at(static_cast<int>(n_bits),
                                                static_cast<int>(n_up)))) {
                    srl = make_shared_rank_lookup(
                        *reps_sp, static_cast<int>(n_bits),
                        static_cast<int>(n_up));
                }
            }

            // Fix 3: Parallelize the irrep loop.
            // Each irrep s is independent: orbit walks read only shared
            // read-only data (subspace, projector, phase_factors, reps_sp)
            // and write to a thread-private SectorOperator slot.
            // Thread-private scratch avoids contention on elems/coeffs.
            // Results are collected into a slot array; gaps (empty sectors)
            // are filled with nullptr and skipped in the collection pass.
            const auto ns = static_cast<std::ptrdiff_t>(num_irreps);
            std::vector<std::unique_ptr<SectorOperator>> slot(
                static_cast<std::size_t>(ns));
            std::vector<std::pair<int, std::size_t>> slot_ids(
                static_cast<std::size_t>(ns), {-1, 0});

            #pragma omp parallel for schedule(dynamic) if(ns > 1)
            for (std::ptrdiff_t si = 0; si < ns; ++si) {
                const std::size_t s = static_cast<std::size_t>(si);
                const std::vector<Complex>& phase =
                    info_sp->sectors[s].phase_factors;

                // Pass 1.5: CSR-free per-sector dimension + RepSectorData.
                // Walk each rep's orbit, keep only (rep, 1/norm) for the
                // survivors. Identical to build_fixed_sz_sector_operators_lazy.
                // Thread-private scratch vectors eliminate contention.
                std::vector<std::uint64_t> elems;
                std::vector<Complex>       coeffs;
                RepSectorData rd;
                rd.n_sites    = static_cast<int>(n_bits);
                rd.group_size = static_cast<int>(group_size);
                rd.reps.reserve(reps_sp->size());
                rd.inv_norms.reserve(reps_sp->size());
                if (srl) {  // Stage 4: local remap over the per-n_up reps
                    rd.shared_rank = srl;
                    rd.local_of_shared.assign(reps_sp->size(),
                                              std::int32_t{-1});
                }
                int  sec_n_up = -1;
                bool uniform  = true;
                const std::vector<Complex> chi =
                    sector_characters_from(*info_sp, phase);

                for (std::size_t i = 0; i < reps_sp->size(); ++i) {
                    const std::uint64_t rep = (*reps_sp)[i];
                    double norm_sq;
                    {
                        norm_sq = projected_norm_sq_stab(
                            otab_full_sp->stab_elems[stab_ids[i]], chi);
                        if (norm_sq <= SectorBasis::kOrbitNormSqEpsilon) continue;
                    }
                    rd.reps.push_back(rep);
                    rd.inv_norms.push_back(1.0 / std::sqrt(norm_sq));
                    if (srl) {
                        rd.local_of_shared[i] =
                            static_cast<std::int32_t>(rd.reps.size() - 1);
                    }
                    const int pc = __builtin_popcountll(rep);
                    if (sec_n_up < 0) sec_n_up = pc;
                    else if (pc != sec_n_up) uniform = false;
                }
                if (rd.reps.empty()) continue;

                rd.n_up = uniform ? sec_n_up : -1;
                if (!info_sp->power_representation.empty() && !phase.empty())
                    rd.characters = chi;
                // Fix 4: Use pre-computed perms_flat (one copy per sector,
                // no recomputation of the |G|*N loop for every sector).
                rd.perms_flat = shared_perms_flat;

                bool is_real = true;
                for (const auto& c : rd.characters) {
                    if (std::abs(c.imag()) > 1e-12) { is_real = false; break; }
                }

                const std::uint64_t dim =
                    static_cast<std::uint64_t>(rd.reps.size());
                auto rep_sp = std::make_shared<RepSectorData>(std::move(rd));

                auto op = std::make_unique<SectorOperator>(
                    n_bits, spin_l, SectorBasis{});
                terms(*op);
                op->configureRepLazy(
                    dim, group_size, is_real,
                    /*rep_provider=*/[rep_sp]() { return *rep_sp; },
                    /*csr_provider=*/
                    [reps_sp, info_sp, n_bits, n_up, s]()
                        -> ::SymmetrySector {
                        // First CPU apply: rebuild orbit CSR for this sector
                        // over the tableless membership subspace.
                        const FixedSzMembershipSubspace sub(
                            n_bits, static_cast<int>(n_up));
                        const SpatialProjector proj(*info_sp);
                        return SectorBasis::build(
                                   sub, proj,
                                   info_sp->sectors[s].quantum_numbers,
                                   info_sp->sectors[s].phase_factors,
                                   *reps_sp, /*sector_id=*/s)
                            .sector();
                    });

                slot[s]     = std::move(op);
                slot_ids[s] = {static_cast<int>(n_up), s};
            }

            // Collect surviving sectors in (n_up, s) order.
            for (std::size_t s = 0; s < num_irreps; ++s) {
                if (!slot[s]) continue;
                if (out_n_up_sector_ids)
                    out_n_up_sector_ids->push_back(slot_ids[s]);
                ops.push_back(std::move(slot[s]));
            }
        }
        return ops;
    }
}

} // namespace ed::symmetry

