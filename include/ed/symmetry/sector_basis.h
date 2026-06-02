#pragma once
// =============================================================================
// include/ed/symmetry/sector_basis.h
//
// SectorBasis: the owning host-side description of ONE symmetry-projected
// sector. Completes the basis-owning "triplet" started by the Orthogonal
// symmetry composition wave:
//
//     FullSpaceSubspace   (subspace.h)  -> FullBasisPolicy      view
//     FixedSzSubspace     (subspace.h)  -> FixedSzBasisPolicy   view
//     SectorBasis         (this file)   -> SymmetryBasisPolicy  view
//
// A SectorBasis is the symmetry analogue of a Subspace: it OWNS the orbit
// data (one ``SymmetrySector`` plus the state->orbit lookup index) and
// hands out the non-owning POD ``SymmetryBasisPolicy`` that the unified
// matvec kernels (``ed::matvec::kernel::apply_terms`` and its GPU twin)
// consume. It is the single owning artefact the future
// ``Operator<SymmetryBasisPolicy, MemSpace>`` holds, replacing the
// per-sector state that ``StreamingSymmetryOperator`` /
// ``FixedSzStreamingSymmetryOperator`` currently bake into their class
// identity.
//
// Why this exists (operator-collapse refactor, Jun 2026):
// ------------------------------------------------------
// The four legacy operator classes (Operator / FixedSzOperator /
// StreamingSymmetryOperator / FixedSzStreamingSymmetryOperator) differ
// ONLY in their basis. The basis axis is already expressed as a POD
// policy + an owning producer for the trivial cases (Subspace). The
// symmetry case lacked a standalone owning producer -- the orbit data
// lived inside the operator and was reachable only through the nested
// ``SectorView``. SectorBasis lifts that orbit data into a free-standing,
// operator-independent value so a single operator template can hold any
// of the three host policies uniformly.
//
// Construction routes:
//   * ``SectorBasis::build(subspace, projector, sector_meta)``
//        Enumerates the sector's orbits over the chosen Subspace using
//        the shared ``compute_orbit_for_state`` host helper (bit-identical
//        to the legacy ``computeOrbitData*`` member methods), then builds
//        the state->orbit lookup index.
//   * ``SectorBasis::adopt(std::move(sector), group_size)``
//        Adopts an already-materialised ``SymmetrySector`` (used during
//        migration so the legacy streaming operators can hand their
//        per-sector orbit data to the new abstraction without recompute).
//
// Lookup strategy: the shipped path uses the ``SortedUint64Index``
// fallback (O(log |orbit_total|) per find), exposed through a
// ``SectorLookupHandle`` with ``dense == nullptr`` and ``lin == nullptr``
// so the policy's ``index_of`` consults the sorted keys directly. The
// dense / Lin-indirection fast paths remain available on the legacy
// operators and can be layered onto SectorBasis in a later wave without
// changing this surface.
// =============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ed/core/sorted_uint64_index.h>
#include <ed/core/streaming_symmetry.h>      // SymmetrySector, SymBasisState,
                                             // SectorLookupHandle
#include <ed/matvec/symmetry_basis_policy.h>  // SymmetryBasisPolicy + factory
#include <ed/symmetry/projector.h>
#include <ed/symmetry/projector_chain.h>
#include <ed/symmetry/subspace.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// SectorBasis -- owning orbit data + lookup for one symmetry sector.
//
// Movable, non-copyable: it owns a ``SortedUint64Index`` whose addresses
// the ``SectorLookupHandle`` (and hence any live ``SymmetryBasisPolicy``)
// references. Copying would silently dangle those handles, so we delete
// the copy operations and require explicit moves.
// ---------------------------------------------------------------------------
class SectorBasis {
public:
    // Survival cutoff on a symmetrised orbit's norm^2; matches the legacy
    // StreamingSymmetryOperator (``norm_sq > 1e-10``) so the basis ordering
    // and dimension stay bit-identical to the legacy sectors.
    static constexpr double kOrbitNormSqEpsilon = 1e-10;

    SectorBasis() = default;
    SectorBasis(const SectorBasis&)            = delete;
    SectorBasis& operator=(const SectorBasis&) = delete;
    SectorBasis(SectorBasis&&)                 = default;
    SectorBasis& operator=(SectorBasis&&)      = default;

    // -----------------------------------------------------------------
    // adopt: take ownership of an already-materialised SymmetrySector.
    //
    // Used by the migration shim so a legacy ``StreamingSymmetryOperator``
    // / ``FixedSzStreamingSymmetryOperator`` can hand off its per-sector
    // orbit CSR (sectors_[k]) without recomputing it. ``group_size`` is
    // |G| (== max_clique.size()); it sets the ``group_norm = 1/|G|``
    // weight the symmetry policy applies per emit.
    //
    // The sector's ``basis_states`` must already be orbit-sorted
    // (``SymBasisState::sortOrbit()`` called) exactly as the legacy
    // build does -- adopt() rebuilds only the state->orbit lookup index,
    // not the orbit coefficients.
    // -----------------------------------------------------------------
    [[nodiscard]] static SectorBasis
    adopt(SymmetrySector sector, std::size_t group_size) {
        SectorBasis sb;
        sb.sector_      = std::move(sector);
        sb.group_size_  = group_size;
        sb.rebuild_lookup_();
        return sb;
    }

    // -----------------------------------------------------------------
    // build: enumerate this sector's orbits over ``subspace`` using the
    // single shared ``compute_orbit_for_state`` host helper, producing
    // orbit data bit-identical to the legacy ``computeOrbitData`` /
    // ``computeOrbitDataFixedSz`` member methods.
    //
    // ``orbit_reps`` is the deduplicated list of orbit representatives in
    // this sector (one per ``SymBasisState``); the caller computes it
    // exactly as the legacy "Pass 1" does (minimum computational state
    // over the orbit, restricted to ``subspace``). Passing it in keeps
    // SectorBasis agnostic to the rep-enumeration strategy (full scan vs
    // lazy streaming) -- it only owns the per-rep orbit expansion.
    //
    // ``sector_meta`` carries the irrep label (quantum_numbers +
    // phase_factors) verbatim from ``SectorMetadata``.
    // -----------------------------------------------------------------
    template <class SubspaceT>
    [[nodiscard]] static SectorBasis
    build(const SubspaceT&                  subspace,
          const SpatialProjector&           projector,
          const std::vector<int>&           quantum_numbers,
          const std::vector<Complex>&       phase_factors,
          const std::vector<std::uint64_t>& orbit_reps,
          std::uint64_t                     sector_id = 0)
    {
        SectorBasis sb;
        sb.group_size_ = projector.group_info().max_clique.size();

        SymmetrySector& sec = sb.sector_;
        sec.sector_id        = sector_id;
        sec.quantum_numbers  = quantum_numbers;
        sec.phase_factors    = phase_factors;
        sec.basis_states.reserve(orbit_reps.size());

        std::vector<std::uint64_t> elems;
        std::vector<Complex>       coeffs;
        double                     norm_sq = 0.0;

        for (std::uint64_t rep : orbit_reps) {
            compute_orbit_for_state(subspace, projector, rep,
                                    phase_factors, elems, coeffs, norm_sq);
            // Survival threshold matches the legacy streaming build
            // (StreamingSymmetryOperator pass 2 / pass 1.5): an orbit
            // whose symmetrised norm^2 falls at or below 1e-10 is a
            // numerically-null state in this irrep and is NOT a basis
            // state. Using the same cutoff keeps the orbit count and the
            // index ordering bit-identical to the legacy sectors.
            if (elems.empty() || norm_sq <= kOrbitNormSqEpsilon) {
                continue;  // orbit fully cancels in this irrep
            }
            SymBasisState st(rep, quantum_numbers, std::sqrt(norm_sq));
            st.orbit_elements     = elems;
            st.orbit_coefficients = coeffs;
            st.sortOrbit();  // O(log) findCoeff in the matvec hot loop
            sec.basis_states.push_back(std::move(st));
        }

        sb.rebuild_lookup_();
        return sb;
    }

    // -----------------------------------------------------------------
    // Basis introspection (mirrors the Subspace surface).
    // -----------------------------------------------------------------
    [[nodiscard]] std::uint64_t dim() const noexcept {
        return sector_.basis_states.size();
    }

    [[nodiscard]] std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return sector_.basis_states[idx].orbit_rep;
    }

    [[nodiscard]] std::int64_t index_of(std::uint64_t state) const noexcept {
        const std::size_t k = lookup_.find(state);
        return (k == ed::core::SortedUint64Index::kNotFound)
            ? std::int64_t{-1}
            : static_cast<std::int64_t>(k);
    }

    [[nodiscard]] std::size_t group_size() const noexcept {
        return group_size_;
    }

    [[nodiscard]] const SymmetrySector& sector() const noexcept {
        return sector_;
    }

    // -----------------------------------------------------------------
    // policy(): the non-owning POD view the matvec kernels consume.
    //
    // The returned ``SymmetryBasisPolicy`` holds a pointer into this
    // object's ``sector_`` and a ``SectorLookupHandle`` referencing this
    // object's ``lookup_``. It stays valid only as long as this
    // SectorBasis is alive AND not moved-from; treat it as a borrow.
    // -----------------------------------------------------------------
    [[nodiscard]] ed::matvec::basis::SymmetryBasisPolicy
    policy() const noexcept {
        return ed::matvec::basis::make_symmetry_basis(
            sector_, lookup_handle_(), static_cast<double>(group_size_));
    }

    // ABI alignment with the ed::matvec::basis::* policies + Subspace
    // surface: a SectorBasis carries the same compile-time trait flags
    // as the symmetry policy it produces.
    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = true;
    static constexpr bool has_coeff_modifier = true;
    static constexpr bool is_distributed     = false;

private:
    // Build the state->orbit lookup index from the orbit elements of every
    // SymBasisState. Maps every computational state appearing in any orbit
    // to its owning orbit index (the array position of the SymBasisState).
    void rebuild_lookup_() {
        lookup_ = ed::core::SortedUint64Index{};
        std::size_t total = 0;
        for (const auto& st : sector_.basis_states) {
            total += st.orbit_elements.size();
        }
        lookup_.reserve(total);
        for (std::size_t i = 0; i < sector_.basis_states.size(); ++i) {
            for (std::uint64_t s : sector_.basis_states[i].orbit_elements) {
                lookup_[s] = i;
            }
        }
        lookup_.finalize();
    }

    [[nodiscard]] SectorLookupHandle lookup_handle_() const noexcept {
        SectorLookupHandle h;
        h.fallback = &lookup_;
        h.dense    = nullptr;
        h.lin      = nullptr;
        return h;
    }

    SymmetrySector             sector_{};
    ed::core::SortedUint64Index lookup_{};
    std::size_t                group_size_ = 1;
};

} // namespace ed::symmetry
