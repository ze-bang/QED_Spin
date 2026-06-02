#pragma once
// =============================================================================
// include/ed/matvec/symmetry_basis_policy.h
//
// SymmetryBasisPolicy + factory helpers. Wave 1 of the
// "unify all 16 matvec cells under apply_terms<BasisPolicy, Scalar, Backend>"
// plan (May 2026).
//
// A SymmetryBasisPolicy is the thin, POD-style view into a single
// symmetry sector of either StreamingSymmetryOperator (full Hilbert)
// or FixedSzStreamingSymmetryOperator (fixed-Sz). It exposes:
//
//   * dim()        --- sector_dim (number of orbit representatives)
//   * state_of(i)  --- orbit-rep bitstring of orbit i
//   * index_of(s)  --- orbit index of computational state s, or -1
//   * iter_orbit(i, cb)  --- walks all (s, alpha_s / norm_i) pairs in
//                            orbit i; cb is invoked once per orbit
//                            element with a Complex pre-phase.
//   * coeff_modifier(s, s', i, k)  --- per-emit factor
//                            conj(beta_{s'}) * group_norm / norm_k
//                            (or the .real() variant when Scalar=double).
//
// Wire-up: the kernel ``ed::matvec::kernel::apply_terms`` honors
// ``BasisPolicy::needs_orbit_walk`` (compile-time true here) by walking
// each orbit before invoking the per-term scatter, and honors
// ``BasisPolicy::has_coeff_modifier`` (true here) by multiplying every
// emitted contribution by ``coeff_modifier`` before the radix-sort
// buffer push. Trivial policies (Full / FixedSz) leave both flags
// false; the kernel ``if constexpr`` branches elide the extra work for
// those policies, so the symmetry path adds zero overhead to the
// non-symmetric matvec.
//
// The math (identical to the legacy ``applyHamiltonianTermsFullSpace``
// / ``applyHamiltonianTermsReal``):
//
//   out[k] += in[j] * (alpha_{s,j} / norm_j) * <s'|H|s>
//                   * conj(beta_{s',k}) * group_norm / norm_k
//
// where (s, alpha_{s,j}) runs over the orbit of representative j, and
// the per-term scatter computes <s'|H|s> via the standard SoA
// (diag_one_body, offdiag_one_body, ...) bins on the host Operator.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>

#include <ed/core/streaming_symmetry.h>   // SymmetrySector, SymBasisState,
                                          // SectorLookupHandle
#include <ed/core/sorted_uint64_index.h>  // kNotFound sentinel

namespace ed::matvec::basis {

// ---------------------------------------------------------------------------
// SymmetryBasisPolicy
//
// Serves BOTH the full-Hilbert symmetry sector (StreamingSymmetryOperator)
// and the fixed-Sz symmetry sector (FixedSzStreamingSymmetryOperator).
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
    static constexpr bool is_distributed    = false;

    // -------- orbit walk ----------------------------------------------
    // Wave 1 ABI: the kernel uses this to amortise the symmetry-adapted
    // weighting across all |orbit| computational states in one input
    // column. Cb is invoked with (s, alpha_s / norm_j) for every orbit
    // element with non-negligible coefficient.
    template <class Callback>
    inline void iter_orbit(uint64_t src_idx, Callback&& cb) const {
        const auto& state_j     = sector->basis_states[src_idx];
        // Cached reciprocal norm (refreshed by SymBasisState::sortOrbit(),
        // the universal finalize point) -- turns a division into a multiply.
        const double inv_norm_j = state_j.inv_norm;
        const std::size_t M = state_j.orbit_elements.size();
        for (std::size_t k = 0; k < M; ++k) {
            const std::complex<double> alpha = state_j.orbit_coefficients[k];
            if (std::abs(alpha) < 1e-15) continue;
            cb(state_j.orbit_elements[k],
               std::complex<double>(alpha.real() * inv_norm_j,
                                    alpha.imag() * inv_norm_j));
        }
    }

    // -------- per-emit modifier ---------------------------------------
    // Wave 1 ABI: returns ``conj(beta_{s'}) * group_norm / norm_k`` when
    // ``Scalar == complex<double>``, and ``beta_{s'}.real() * group_norm
    // / norm_k`` when ``Scalar == double``. The Sector::sortOrbit() pass
    // makes beta lookup O(log |orbit|).
    template <class Scalar>
    [[nodiscard]] inline Scalar
    coeff_modifier(uint64_t /*s*/, uint64_t s_prime,
                   uint64_t /*src_idx*/, uint64_t dst_idx) const noexcept
    {
        const auto& state_k = sector->basis_states[dst_idx];
        // group_norm * inv_norm replaces a per-emit division by the cached
        // reciprocal norm (set in SymBasisState::sortOrbit()). When norm == 0
        // inv_norm == 0, so scale == 0 -- bit-identical to the prior guard.
        const double scale  = group_norm * state_k.inv_norm;
        const std::complex<double> beta_s_prime = state_k.findCoeff(s_prime);
        if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
            return std::conj(beta_s_prime) * scale;
        } else {
            static_assert(std::is_same_v<Scalar, double>,
                          "SymmetryBasisPolicy::coeff_modifier supports "
                          "Scalar = std::complex<double> or double only.");
            return beta_s_prime.real() * scale;
        }
    }

    // -------- distributed no-ops --------------------------------------
    [[nodiscard]] inline bool is_local(uint64_t /*g*/) const noexcept { return true; }
    [[nodiscard]] inline uint64_t local_offset() const noexcept { return 0; }
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
