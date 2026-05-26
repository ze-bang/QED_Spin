#pragma once
// =============================================================================
// include/ed/symmetry/projector.h
//
// Projector: ONE group representation, applied to a Subspace as
//
//     P = (1/|G|) * Sum_{g in G} chi(g)* g
//
// Part of the "Orthogonal symmetry composition" wave (May 2026). A
// Projector is the SECOND orthogonal axis of the new
// (Subspace x ProjectorChain) decomposition. A ProjectorChain (Phase R3)
// is just an ordered list of Projectors -- think of each Projector as
// one factor in a product of commuting symmetry groups acting on the
// chosen Subspace:
//
//   chain = []                                -> mode "none" / "Sz"
//   chain = [SpatialProjector]                -> mode "Symm" / "Sz+Symm"
//   chain = [SpatialProjector, InternalZ2]    -> future spin-flip
//   chain = [Spatial, InternalZ2, TimeRev]    -> future T + Z2
//   chain = [..., CasimirProjector]           -> future SU(2) Casimir
//
// At build time the orbit/character composition collapses the chain
// into the standard (orbit_elements, orbit_coefficients, norm) tuple
// stored on each ``SymBasisState``. The unified matvec kernels
// (``ed::matvec::kernel::apply_terms`` and its GPU twin) consume only
// that flat scalar form, so a Projector NEVER appears in the matvec
// hot loop -- it is consumed purely host-side.
//
// Currently we ship ONE concrete projector:
//
//   * SpatialProjector  -- the existing site-permutation group from
//                          ``SymmetryGroupInfo``. Per-sector character
//                          is composed from the per-generator
//                          ``phase_factors`` and the cached
//                          ``power_representation`` (Bloch convention
//                          chi = product_k exp(-2 pi i q_k * p_k / o_k)).
//
// Future projectors (Phase 4 of the plan, not implemented here) all
// fit the same trait surface:
//
//   * InternalZ2Projector  -- e.g. global spin flip ``state ^ all_ones``;
//                              characters {+1, -1}; one generator of
//                              order 2. ``is_antiunitary = false``.
//   * AntiunitaryProjector -- e.g. time reversal; same shape but
//                              ``is_antiunitary = true`` so the orbit
//                              builder uses chi(g) rather than chi(g)*.
//   * CasimirProjector     -- e.g. SU(2) total-S filter. Not a 1-D
//                              character; either filter on the
//                              ``Subspace`` (Casimir-polynomial route)
//                              or fork the matvec ABI (coupled-basis
//                              route). Documented in the wave plan.
//
// The Projector concept is duck-typed (templated chain build code in
// projector_chain.h uses ``auto`` and ``decltype``), so adding a new
// projector is "implement the methods below + push onto the chain";
// there is no closed-set switch statement to extend.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/core/basis_utils.h>     // applyPermutation
#include <ed/core/symmetry_metadata.h>  // SymmetryGroupInfo + SectorMetadata

namespace ed::symmetry {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Projector duck-type contract
// ---------------------------------------------------------------------------
//
// Every Projector type used by ``build_symmetry_basis`` is expected
// to expose the following members. We do NOT enforce this via a C++20
// ``concept`` here because the chain is heterogeneous (different
// projector types in the same chain) -- the orbit builder dispatches
// via ``std::variant`` (Phase R3). The required methods are:
//
//   std::size_t size() const noexcept;
//       Number of group elements |G|.
//
//   std::uint64_t apply(std::uint64_t state, std::size_t g) const;
//       Action of the g'th group element on a computational basis
//       state. ``g == 0`` MUST be the identity.
//
//   Complex character(std::size_t g,
//                     const std::vector<int>& sector_quantum_numbers) const;
//       chi_q(g) for the sector picked by ``sector_quantum_numbers``.
//
//   std::size_t sector_count() const noexcept;
//       Number of inequivalent irreducible representations supported.
//
//   const std::vector<int>& quantum_numbers(std::size_t s) const noexcept;
//       Quantum-number tuple labelling sector ``s``.
//
//   static constexpr bool is_antiunitary = false;
//       True only for time-reversal-like projectors. The build path
//       uses chi(g) rather than conj(chi(g)) when this is set.
//
//   static constexpr bool preserves_sz = true;
//       Hint: does this projector commute with the Sz operator?
//       Currently informational (the build path uses
//       ``Subspace::index_of`` to drop out-of-subspace orbit elements
//       regardless).
//
//   std::string name() const noexcept;
//       Diagnostic label used in build logs / error messages.
//
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SpatialProjector
//
// Wraps a ``SymmetryGroupInfo`` (the host-resident JSON-loaded blob in
// ``Operator::symmetry_info``) and exposes the per-sector spatial
// character + state action that the legacy ``computeOrbitData`` and
// ``computeOrbitDataFixedSz`` functions previously inlined.
//
// The projector is a non-owning view; the ``SymmetryGroupInfo`` it
// wraps must outlive the projector. ``SpatialProjector`` is trivially
// copyable.
//
// Per-sector character composition (Bloch convention, identical to
// the legacy inline loop at streaming_symmetry.h:1781-1788):
//
//   chi_q(g) = product_k phase_factors[k] ^ power_representation[g][k]
//
// where ``phase_factors`` holds ``exp(-2 pi i q_k / o_k)`` for each of
// the ``num_generators`` generators with orders ``o_k`` and the
// sector's quantum-number tuple ``q``.
// ---------------------------------------------------------------------------
class SpatialProjector {
public:
    explicit SpatialProjector(const SymmetryGroupInfo& info) noexcept
        : info_(&info) {}

    [[nodiscard]] std::size_t size() const noexcept {
        return info_->max_clique.size();
    }

    [[nodiscard]] std::uint64_t apply(std::uint64_t state,
                                      std::size_t   g) const {
        return applyPermutation(state, info_->max_clique[g]);
    }

    [[nodiscard]] std::size_t sector_count() const noexcept {
        return info_->sectors.size();
    }

    [[nodiscard]] const std::vector<int>&
    quantum_numbers(std::size_t s) const noexcept {
        return info_->sectors[s].quantum_numbers;
    }

    /// Per-generator phase factors for sector ``s`` (one Complex per
    /// generator; the character at group element ``g`` is the product
    /// of these raised to the powers ``power_representation[g][k]``).
    [[nodiscard]] const std::vector<Complex>&
    phase_factors(std::size_t s) const noexcept {
        return info_->sectors[s].phase_factors;
    }

    /// Composite character at group element ``g`` for sector ``s``.
    ///
    /// Mirrors the legacy inline loop verbatim:
    ///
    ///   character = product_k pf[k]^p[g][k]
    ///
    /// where ``pf = phase_factors(s)`` and
    /// ``p[g][k] = power_representation[g][k]``.
    [[nodiscard]] Complex character(std::size_t g, std::size_t s) const {
        const auto& powers = info_->power_representation[g];
        const auto& pf     = info_->sectors[s].phase_factors;
        Complex c(1.0, 0.0);
        for (std::size_t k = 0; k < powers.size(); ++k) {
            const Complex phase = pf[k];
            for (int p = 0; p < powers[k]; ++p) {
                c *= phase;
            }
        }
        return c;
    }

    /// Convenience overload: character indexed by the sector's
    /// quantum-number tuple. Looks up the matching sector and
    /// delegates to the index-based overload above.
    [[nodiscard]] Complex
    character(std::size_t g,
              const std::vector<int>& sector_quantum_numbers) const
    {
        for (std::size_t s = 0; s < info_->sectors.size(); ++s) {
            if (info_->sectors[s].quantum_numbers == sector_quantum_numbers) {
                return character(g, s);
            }
        }
        throw std::out_of_range(
            "SpatialProjector::character: no sector with the given "
            "quantum_numbers tuple is loaded.");
    }

    [[nodiscard]] const SymmetryGroupInfo& group_info() const noexcept {
        return *info_;
    }

    // ----- Trait surface (consumed by build_symmetry_basis) ---------
    static constexpr bool is_antiunitary = false;
    static constexpr bool preserves_sz   = true;

    [[nodiscard]] std::string name() const noexcept {
        return "SpatialProjector(|G|=" + std::to_string(size())
             + ", sectors=" + std::to_string(sector_count()) + ")";
    }

private:
    const SymmetryGroupInfo* info_ = nullptr;
};

// ---------------------------------------------------------------------------
// Future seats -- declared as forward-only no-op classes so chain code
// can be ABI-compiled against them today and the implementations land
// in follow-up PRs without touching this header. None of these are
// wired into the orbit builder yet.
//
// The four-mode bench matrix is closed today via empty chains or chains
// of exactly one SpatialProjector; the placeholders below exist only
// to document the extension surface.
// ---------------------------------------------------------------------------

/// Spin-flip-like global Z_2 (state -> state ^ mask). Forward-declared
/// no-op for ABI smoke tests in Phase R5. Real implementation is a
/// follow-up workstream and lives behind ``automorphism_results/internal_symmetries.json``.
class InternalZ2Projector {
public:
    explicit InternalZ2Projector(std::uint64_t mask, bool even = true) noexcept
        : mask_(mask), parity_(even ? +1 : -1) {}

    [[nodiscard]] std::size_t size() const noexcept { return 2; }

    [[nodiscard]] std::uint64_t apply(std::uint64_t state,
                                      std::size_t   g) const noexcept {
        return (g == 0) ? state : (state ^ mask_);
    }

    [[nodiscard]] std::size_t sector_count() const noexcept { return 2; }

    [[nodiscard]] std::vector<int> quantum_numbers(std::size_t s) const {
        return {s == 0 ? +1 : -1};
    }

    [[nodiscard]] Complex character(std::size_t g, std::size_t s) const {
        if (g == 0) return Complex(1.0, 0.0);
        return Complex(s == 0 ? +1.0 : -1.0, 0.0);
    }

    static constexpr bool is_antiunitary = false;
    static constexpr bool preserves_sz   = false;

    [[nodiscard]] std::string name() const noexcept {
        return std::string("InternalZ2Projector(parity=")
             + (parity_ > 0 ? "even" : "odd") + ")";
    }

private:
    std::uint64_t mask_   = 0;
    int           parity_ = +1;
};

/// Antiunitary projector (e.g. time reversal). Same shape as the
/// abelian ones; the orbit-builder branch on ``is_antiunitary`` flips
/// conjugation of the character. ABI placeholder for Phase R5.
class AntiunitaryProjector {
public:
    AntiunitaryProjector() noexcept = default;

    [[nodiscard]] std::size_t size() const noexcept { return 2; }

    [[nodiscard]] std::uint64_t apply(std::uint64_t state,
                                      std::size_t /*g*/) const noexcept {
        return state;
    }

    [[nodiscard]] std::size_t sector_count() const noexcept { return 2; }

    [[nodiscard]] std::vector<int> quantum_numbers(std::size_t s) const {
        return {s == 0 ? +1 : -1};
    }

    [[nodiscard]] Complex character(std::size_t g, std::size_t s) const {
        if (g == 0) return Complex(1.0, 0.0);
        return Complex(s == 0 ? +1.0 : -1.0, 0.0);
    }

    static constexpr bool is_antiunitary = true;
    static constexpr bool preserves_sz   = true;

    [[nodiscard]] std::string name() const noexcept {
        return "AntiunitaryProjector(time-reversal-like)";
    }
};

} // namespace ed::symmetry
