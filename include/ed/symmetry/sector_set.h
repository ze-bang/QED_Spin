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
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include <omp.h>

#include <ed/core/basis_utils.h>          // generateFixedSzBasis, LinIndexTable, applyPermutation
#include <ed/symmetry/symmetry_sector_data.h>  // SymmetrySector
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/subspace.h>

namespace ed::symmetry {

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
    std::vector<std::uint64_t> reps;
    const std::uint64_t dim = (1ULL << n_bits);
    reps.reserve(dim / std::max<std::size_t>(1, info.max_clique.size()) + 1);
    for (std::uint64_t s = 0; s < dim; ++s) {
        std::uint64_t mn = s;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            mn = std::min(mn, applyPermutation(s, info.max_clique[g]));
        }
        if (mn == s) reps.push_back(s);
    }
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
    std::vector<std::uint64_t> reps;
    for (std::uint64_t s : sub.basis_states()) {
        bool is_rep = true;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            const std::uint64_t img = applyPermutation(s, info.max_clique[g]);
            if (sub.index_of(img) >= 0 && img < s) { is_rep = false; break; }
        }
        if (is_rep) reps.push_back(s);
    }
    std::sort(reps.begin(), reps.end());
    reps.erase(std::unique(reps.begin(), reps.end()), reps.end());
    return reps;
}

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

/// Full-Hilbert-space symmetry sectors. ``n_bits`` / ``spin_l`` are the
/// lattice constants; ``info`` carries the group + irrep metadata; ``terms``
/// populates each sector operator's term list.
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_full_sector_operators(std::uint64_t            n_bits,
                            float                    spin_l,
                            const SymmetryGroupInfo& info,
                            TermBuilder&&            terms,
                            std::vector<std::size_t>* out_sector_ids = nullptr)
{
    const FullSpaceSubspace full(n_bits);
    const SpatialProjector  projector(info);
    const std::vector<std::uint64_t> reps =
        enumerate_full_orbit_reps(info, n_bits);

    std::vector<std::unique_ptr<SectorOperator>> ops;
    ops.reserve(info.sectors.size());
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        SectorBasis sb = SectorBasis::build(
            full, projector,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);
        if (sb.dim() == 0) continue;  // orbit fully cancels in this irrep
        auto op = std::make_unique<SectorOperator>(
            n_bits, spin_l, std::move(sb));
        terms(*op);
        ops.push_back(std::move(op));
        // Record the RAW sector index (the loop variable) so callers can
        // reconstruct the per-sector ``SectorTag`` (quantum numbers + dim)
        // without the now-compacted output losing the irrep label.
        if (out_sector_ids) out_sector_ids->push_back(s);
    }
    return ops;
}

/// Fixed-Sz symmetry sectors. ``n_up`` selects the magnetisation sector;
/// the owning ``FixedSzSubspace`` is built internally and kept alive for the
/// duration of the build (the orbit reps + SectorBasis are computed against
/// it). ``info`` / ``terms`` as above.
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_fixed_sz_sector_operators(std::uint64_t            n_bits,
                                float                    spin_l,
                                std::int64_t             n_up,
                                const SymmetryGroupInfo& info,
                                TermBuilder&&            terms,
                                std::vector<std::size_t>* out_sector_ids = nullptr)
{
    const FixedSzSubspace  fixed = FixedSzSubspace::build(n_bits, n_up);
    const SpatialProjector projector(info);
    const std::vector<std::uint64_t> reps =
        enumerate_fixed_sz_orbit_reps(fixed, info);

    std::vector<std::unique_ptr<SectorOperator>> ops;
    ops.reserve(info.sectors.size());
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        SectorBasis sb = SectorBasis::build(
            fixed, projector,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);
        if (sb.dim() == 0) continue;  // orbit fully cancels in this irrep
        auto op = std::make_unique<SectorOperator>(
            n_bits, spin_l, std::move(sb));
        terms(*op);
        // Populate the CSR-free on-the-fly representative source so
        // ``bind_cuda()`` can take the resident rep path under
        // ``ED_GPU_SYMMETRY_REP``. Fixed-Sz sectors yield a usable record.
        op->setRepSectorData(rep_sector_data_from_sector(
            op->basis().sector(), info, static_cast<int>(n_bits)));
        ops.push_back(std::move(op));
        if (out_sector_ids) out_sector_ids->push_back(s);
    }
    return ops;
}

// ---------------------------------------------------------------------------
// Migration factory: build a standalone SectorOperator from an
// already-materialised legacy sector.
//
// The legacy ``StreamingSymmetryOperator`` /
// ``FixedSzStreamingSymmetryOperator`` materialise a sector's orbit data on
// demand (``getSector(k)`` runs ``ensureSectorMaterialized_``). This helper
// ADOPTS that materialised ``SymmetrySector`` into an owning ``SectorBasis``
// (no orbit recompute) and copies the host operator's Hamiltonian term list
// into a fresh ``SectorOperator`` -- yielding an independent per-sector
// operator equivalent to the legacy ``SectorView`` but free of any
// back-reference to the parent. ``host`` supplies the lattice constants
// (``getNumBits`` / ``getSpin``) and the canonical term list
// (``transform_data_`` / ``three_body_data_``); ``sector`` is the
// materialised orbit data (taken by value); ``group_size`` is |G|.
//
// This is the bridge used by the production sector loop (behind an env gate)
// to route through the collapse-target SectorOperator while keeping the
// legacy operator as the orbit-enumeration + cross-check source.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::unique_ptr<SectorOperator>
make_sector_operator_adopt(const ::Operator& host,
                           SymmetrySector    sector,
                           std::size_t       group_size)
{
    SectorBasis sb = SectorBasis::adopt(std::move(sector), group_size);
    auto op = std::make_unique<SectorOperator>(
        host.getNumBits(), host.getSpin(), std::move(sb));
    // Copy the canonical Hamiltonian term list verbatim; the symmetry
    // backend applies these terms within the orbit walk exactly as the
    // legacy applySymmetrizedUnified path does (which reads the same
    // term_view_). ``copyTermsFrom`` keeps the host's term members behind
    // a single intentional API rather than touching them directly.
    op->copyTermsFrom(host);
    // Populate the CSR-free on-the-fly representative source (reps + 1/norm
    // from the adopted sector, characters + perms from the host's group
    // info) so ``bind_cuda()`` can take the resident rep path under
    // ``ED_GPU_SYMMETRY_REP``. Reads only orbit_rep + norm from the sector.
    op->setRepSectorData(rep_sector_data_from_sector(
        op->basis().sector(), host.symmetry_info,
        static_cast<int>(host.getNumBits())));
    return op;
}

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
template <class TermBuilder>
[[nodiscard]] std::vector<std::unique_ptr<SectorOperator>>
build_fixed_sz_sector_operators_lazy(std::uint64_t            n_bits,
                                     float                    spin_l,
                                     std::int64_t             n_up,
                                     const SymmetryGroupInfo& info,
                                     TermBuilder&&            terms,
                                     std::vector<std::size_t>* out_sector_ids = nullptr)
{
    // Co-own the inputs the deferred providers capture. The group info backs
    // the SpatialProjector (a non-owning view). The fixed-Sz subspace is
    // self-referential (``FixedSzSubspace::build`` sets ``basis_ptr_`` to its
    // own ``owned_basis_``), so it cannot be moved into a ``shared_ptr``
    // without dangling -- we instead own the basis vector + Lin table directly
    // (shared, stable heap storage) and hand out cheap ``FixedSzSubspace::view``
    // observers over them. ``reps`` drives both the Pass 1.5 norm scan and the
    // on-demand CSR rebuild.
    auto info_sp  = std::make_shared<SymmetryGroupInfo>(info);
    auto basis_sp = std::make_shared<std::vector<std::uint64_t>>(
        generateFixedSzBasis(n_bits, n_up));
    auto lin_sp   = std::make_shared<LinIndexTable>();
    lin_sp->build(n_bits, n_up, *basis_sp);

    const FixedSzSubspace subspace =
        FixedSzSubspace::view(n_bits, n_up, *basis_sp, *lin_sp);
    auto reps_sp = std::make_shared<std::vector<std::uint64_t>>(
        enumerate_fixed_sz_orbit_reps(subspace, *info_sp));

    const SpatialProjector  projector(*info_sp);
    const std::size_t        group_size = info_sp->max_clique.size();
    const std::size_t        num_sectors = info_sp->sectors.size();

    std::vector<std::unique_ptr<SectorOperator>> ops;
    ops.reserve(num_sectors);

    // Fix 4: Compute perms_flat once; reuse across all sectors.
    const std::vector<int> shared_perms_flat =
        flatten_group_perms(*info_sp, static_cast<int>(n_bits));

    std::vector<std::uint64_t> elems;
    std::vector<Complex>       coeffs;

    for (std::size_t s = 0; s < num_sectors; ++s) {
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
        int  sec_n_up = -1;
        bool uniform  = true;
        for (std::uint64_t rep : *reps_sp) {
            double norm_sq = 0.0;
            compute_orbit_for_state(subspace, projector, rep, phase,
                                    elems, coeffs, norm_sq);
            if (elems.empty() ||
                norm_sq <= SectorBasis::kOrbitNormSqEpsilon) {
                continue;  // orbit fully cancels in this irrep
            }
            rd.reps.push_back(rep);
            rd.inv_norms.push_back(1.0 / std::sqrt(norm_sq));
            const int pc = __builtin_popcountll(rep);
            if (sec_n_up < 0) sec_n_up = pc;
            else if (pc != sec_n_up) uniform = false;
        }
        if (rd.reps.empty()) continue;  // empty sector: nothing to solve

        rd.n_up = uniform ? sec_n_up : -1;
        if (!info_sp->power_representation.empty() && !phase.empty()) {
            rd.characters = sector_characters_from(*info_sp, phase);
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
            /*csr_provider=*/[basis_sp, lin_sp, reps_sp, info_sp, n_bits, n_up, s]()
                -> ::SymmetrySector {
                // First CPU ``apply`` only: rebuild this sector's orbit CSR
                // with the same enumerator the eager lane uses. A fresh view
                // over the (still-alive) shared basis + Lin table avoids the
                // self-referential ``FixedSzSubspace`` move hazard.
                const FixedSzSubspace sub =
                    FixedSzSubspace::view(n_bits, n_up, *basis_sp, *lin_sp);
                const SpatialProjector proj(*info_sp);
                return SectorBasis::build(
                           sub, proj,
                           info_sp->sectors[s].quantum_numbers,
                           info_sp->sectors[s].phase_factors,
                           *reps_sp, /*sector_id=*/s)
                    .sector();
            });
        ops.push_back(std::move(op));
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
    std::int64_t             n_up_min = 0,
    std::int64_t             n_up_max = -1,
    std::vector<std::pair<int, std::size_t>>* out_n_up_sector_ids = nullptr)
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
        const std::vector<std::uint64_t> all_reps =
            enumerate_full_orbit_reps(*info_sp, n_bits);
        // (populated into per-n_up rep lists below)
        std::vector<std::vector<std::uint64_t>> reps_by_n_up(
            static_cast<std::size_t>(n_bits + 1));
        for (std::uint64_t r : all_reps) {
            const auto pc =
                static_cast<std::size_t>(__builtin_popcountll(r));
            if (pc < reps_by_n_up.size())
                reps_by_n_up[pc].push_back(r);
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

            // Co-own per-n_up basis + Lin table for deferred CSR providers.
            // ``FixedSzSubspace::view`` takes stable heap references, so we
            // heap-allocate both and share them via shared_ptr.
            auto basis_sp = std::make_shared<std::vector<std::uint64_t>>(
                generateFixedSzBasis(n_bits, n_up));
            auto lin_sp = std::make_shared<LinIndexTable>();
            lin_sp->build(n_bits, n_up, *basis_sp);
            auto reps_sp = std::make_shared<std::vector<std::uint64_t>>(
                reps_by_n_up[nu]);

            const FixedSzSubspace subspace =
                FixedSzSubspace::view(n_bits, n_up, *basis_sp, *lin_sp);

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
                int  sec_n_up = -1;
                bool uniform  = true;

                for (std::uint64_t rep : *reps_sp) {
                    double norm_sq = 0.0;
                    compute_orbit_for_state(subspace, projector, rep, phase,
                                            elems, coeffs, norm_sq);
                    if (elems.empty() ||
                        norm_sq <= SectorBasis::kOrbitNormSqEpsilon)
                        continue;
                    rd.reps.push_back(rep);
                    rd.inv_norms.push_back(1.0 / std::sqrt(norm_sq));
                    const int pc = __builtin_popcountll(rep);
                    if (sec_n_up < 0) sec_n_up = pc;
                    else if (pc != sec_n_up) uniform = false;
                }
                if (rd.reps.empty()) continue;

                rd.n_up = uniform ? sec_n_up : -1;
                if (!info_sp->power_representation.empty() && !phase.empty())
                    rd.characters = sector_characters_from(*info_sp, phase);
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
                    [basis_sp, lin_sp, reps_sp, info_sp, n_bits, n_up, s]()
                        -> ::SymmetrySector {
                        // First CPU apply: rebuild orbit CSR for this sector.
                        const FixedSzSubspace sub =
                            FixedSzSubspace::view(n_bits, n_up,
                                                  *basis_sp, *lin_sp);
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

