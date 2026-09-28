#pragma once
// =============================================================================
// include/ed/matvec/symmetry_basis_policy.h
//
// SymmetryBasisPolicy + factory helper.
//
// A SymmetryBasisPolicy is the thin, POD-style lookup view into a single
// orbit-CSR symmetry sector (full-space or fixed-Sz). It exposes:
//
//   * dim()        --- sector_dim (number of orbit representatives)
//   * state_of(i)  --- orbit-rep bitstring of orbit i
//   * index_of(s)  --- orbit index of computational state s, or -1
//
// plus the compile-time trait flags of the symmetry lane. The type also
// tags the symmetry SubspaceOperator (``ed::symmetry::SectorOperator``),
// whose matvec runs through the representative policy
// (``RepSymmetryBasisPolicy``) and its kernels.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>

#include <ed/symmetry/symmetry_sector_data.h>   // SymmetrySector, SymBasisState,
                                          // SectorLookupHandle
#include <ed/core/sorted_uint64_index.h>  // kNotFound sentinel

namespace ed::matvec::basis {

// ---------------------------------------------------------------------------
// SymmetryBasisPolicy
//
// Serves BOTH the full-Hilbert symmetry sector (StreamingSymmetryOperator)
// and the fixed-Sz symmetry sector.
// The only mode-specific data lives inside ``SectorLookupHandle``
// (dense table + optional LinIndexTable indirection); the policy reads
// the same SymmetrySector / SymBasisState layout in both cases.
// ---------------------------------------------------------------------------
struct SymmetryBasisPolicy {
    const ::SymmetrySector*  sector;       // non-owning view
    ::SectorLookupHandle     lookup;       // POD (3 pointers)
    double                   group_norm;   // 1.0 / |G|

    [[nodiscard]] inline uint64_t dim() const noexcept {
        return sector->basis_states.size();
    }

    [[nodiscard]] inline uint64_t state_of(uint64_t idx) const noexcept {
        // Canonical orbit representative. Only consulted by the trivial
        // (non-orbit-walk) path; kept for ABI completeness.
        return sector->basis_states[idx].orbit_rep;
    }

    [[nodiscard]] inline int64_t index_of(uint64_t state) const noexcept {
        const std::size_t k = lookup.find(state);
        return (k == ed::core::SortedUint64Index::kNotFound)
            ? int64_t{-1}
            : static_cast<int64_t>(k);
    }

    // -------- compile-time traits --------------------------------------
    // Off-diagonal terms can take an orbit element s into a computational
    // state s' that is not in any orbit of this sector (true for both
    // full-space and fixed-Sz symmetry); index_of() returns -1 and the
    // emit is skipped.
    static constexpr bool may_leave_basis   = true;
    static constexpr bool needs_orbit_walk  = true;
    static constexpr bool has_coeff_modifier = true;
};

// ---------------------------------------------------------------------------
// Convenience factory. Saves call sites from spelling out the field
// types each time they need a fresh view onto a sector.
// ---------------------------------------------------------------------------
[[nodiscard]] inline SymmetryBasisPolicy
make_symmetry_basis(const ::SymmetrySector& sector,
                    const ::SectorLookupHandle& lookup,
                    double group_size) noexcept
{
    return SymmetryBasisPolicy{
        /*sector=*/&sector,
        /*lookup=*/lookup,
        /*group_norm=*/(group_size > 0.0 ? 1.0 / group_size : 0.0),
    };
}

} // namespace ed::matvec::basis
