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
#include <cstdint>
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
    std::set<std::uint64_t> reps;
    for (std::uint64_t s : sub.basis_states()) {
        std::uint64_t mn = s;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            const std::uint64_t img = applyPermutation(s, info.max_clique[g]);
            if (sub.index_of(img) >= 0) mn = std::min(mn, img);
        }
        reps.insert(mn);
    }
    return std::vector<std::uint64_t>(reps.begin(), reps.end());
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
    // term_view_).
    op->transform_data_  = host.transform_data_;
    op->three_body_data_ = host.three_body_data_;
    return op;
}

} // namespace ed::symmetry

