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
#include <set>
#include <vector>

#include <ed/core/streaming_symmetry.h>   // SymmetryGroupInfo, applyPermutation
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
                            TermBuilder&&            terms)
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
                                TermBuilder&&            terms)
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
// make_rep_sector_operator_lazy: CSR-FREE per-sector operator for the GPU
// on-the-fly representative path ("scan other region" optimisation, Jun 2026).
//
// Unlike ``make_sector_operator_adopt`` (which requires an already-materialised
// ``SymmetrySector`` -- i.e. the ~24 GiB/sector host orbit CSR), this builds a
// SectorOperator that knows its dimension + real/complex character up-front
// (cheap, CSR-free) and DEFERS:
//   * the CSR-free RepSectorData (GPU rep matvec source) to ``bind_cuda`` via
//     ``host.getRepSectorData(k)`` -- regenerates the group action on device,
//     never stores the orbit CSR; and
//   * the host orbit CSR (CPU ``apply`` fallback) to first CPU use via
//     ``host.getSector(k)`` -- so a GPU-only run NEVER materialises it.
//
// Only valid for the fixed-Sz streaming operator (the rep matvec needs a
// fixed-Sz combinadic rank table). The host operator MUST outlive every
// returned SectorOperator (the providers capture it by pointer) -- the same
// lifetime contract ``StreamingSymmetryHandle`` already imposes.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::unique_ptr<SectorOperator>
make_rep_sector_operator_lazy(
        const ::FixedSzStreamingSymmetryOperator& host,
        std::size_t                               sector_idx)
{
    const std::uint64_t dim =
        host.getSectorDimension(sector_idx);
    const std::size_t group_size =
        static_cast<std::size_t>(host.getGroupSize());

    // Cheap CSR-free real/complex test: the |G| per-sector characters depend
    // only on the group info + the sector's phase factors (both known at
    // generation), never on the orbit CSR. A momentum sector with any complex
    // character must stay on the complex matvec path.
    bool is_real = true;
    if (!host.symmetry_info.power_representation.empty()) {
        const auto chi = sector_characters_from(
            host.symmetry_info, host.getSectorPhaseFactors(sector_idx));
        for (const auto& c : chi) {
            if (std::abs(c.imag()) > 1e-12) { is_real = false; break; }
        }
    }

    auto op = std::make_unique<SectorOperator>(
        host.getNumBits(), host.getSpin(), SectorBasis{});
    op->copyTermsFrom(host);

    const ::FixedSzStreamingSymmetryOperator* host_ptr = &host;
    op->configureRepLazy(
        dim, group_size, is_real,
        /*rep_provider=*/[host_ptr, sector_idx]() {
            return host_ptr->getRepSectorData(sector_idx);
        },
        /*csr_provider=*/[host_ptr, sector_idx]() -> ::SymmetrySector {
            return host_ptr->getSector(sector_idx);  // const ref -> copy
        });
    return op;
}

} // namespace ed::symmetry

