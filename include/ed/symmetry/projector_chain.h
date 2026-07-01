#pragma once
// =============================================================================
// include/ed/symmetry/projector_chain.h
//
// ProjectorChain + build_symmetry_basis: the host-side orbit/character
// composition that collapses an ordered list of Projectors acting on
// a chosen Subspace into the standard ``(orbit_elements,
// orbit_coefficients, norm)`` tuple stored on each ``SymBasisState``.
//
// Part of the "Orthogonal symmetry composition" wave (May 2026).
//
// Design contract: the FINAL host-side artefact is bit-identical to
// what ``StreamingSymmetryOperator::computeOrbitData`` and
// ``FixedSzStreamingSymmetryOperator::computeOrbitDataFixedSz``
// produce today. The new functions below are designed to REPLACE
// those two legacy member methods with one template parameterised on
// the Subspace; the legacy code paths now delegate to the helper here
// (see streaming_symmetry.h).
//
// Mathematics (Bloch convention, identical to the legacy inline
// loops):
//
//   For each computational state |s> with representative ``basis``,
//   compute
//       Sum_{g in G}  conj(chi(g))  * |g.s>
//   accumulate the per-element coefficient in a hash map keyed by
//   ``g.s``, drop entries below 1e-15, then
//       norm_sq = (Sum |alpha_s|^2) / |G|.
//
// Subspace filter: when the subspace excludes ``g.s`` (e.g. fixed-Sz
// orbit element that has the wrong popcount, which only happens if
// the projector does NOT preserve Sz -- site permutations always do),
// the contribution is skipped before the hash-map update. This is
// the only behavioural difference between the full-space and the
// fixed-Sz inline loops, and it is now consolidated into ONE generic
// branch driven by ``Subspace::index_of``.
//
// Future projectors (Z_2 spin-flip, time-reversal, ...) extend the
// chain by adding another factor; the inner loop becomes a Cartesian
// walk over the chain. The current shipped configuration is always
// ``[SpatialProjector]`` for symmetry-projected operators and ``[]``
// (empty) for plain Operator / FixedSzOperator.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
// <unordered_map> removed: compute_orbit_for_state uses a stack flat-array
// accumulator (Fix 6) that is faster for |G|<=256 with zero heap allocation.
#include <utility>
#include <variant>
#include <vector>

#include <ed/symmetry/projector.h>
#include <ed/symmetry/subspace.h>

namespace ed::symmetry {

using Complex = std::complex<double>;

// ---------------------------------------------------------------------------
// Numerical tolerance: coefficients with magnitude below this threshold
// are dropped from the orbit storage. Matches the legacy inline loops
// at streaming_symmetry.h:1800 and :3718.
// ---------------------------------------------------------------------------
inline constexpr double kOrbitCoefficientEpsilon = 1e-15;

// ---------------------------------------------------------------------------
// compute_orbit_for_state(subspace, projector, basis, phase_factors, ...)
//
// Single-projector overload (the universal current production case --
// every shipped symmetry sector applies exactly one SpatialProjector).
// Output is bit-identical to the legacy ``computeOrbitData`` /
// ``computeOrbitDataFixedSz`` inline member methods.
//
// Parameters:
//   subspace        --  ``FullSpaceSubspace`` or ``FixedSzSubspace``.
//                       Drives the "is this orbit element in the
//                       subspace?" check that distinguished the two
//                       legacy code paths.
//   projector       --  Concrete ``SpatialProjector`` view onto the
//                       owning ``SymmetryGroupInfo``.
//   basis           --  Computational basis state to expand into its
//                       orbit.
//   phase_factors   --  The sector's ``phase_factors`` (per-generator
//                       complex characters); kept here verbatim from
//                       the legacy ABI so the SectorMetadata layout is
//                       untouched.
//
// Outputs:
//   orbit_elements  --  Sorted-by-insertion list of computational
//                       basis states reached by g.basis for some g.
//                       (The legacy code calls ``sortOrbit()`` once
//                       AFTER this function to sort by state -- that
//                       step still happens at the call site.)
//   orbit_coeffs    --  Parallel to ``orbit_elements``; the symmetrised
//                       coefficient at each orbit element.
//   norm_sq         --  Sum-of-squares of the orbit coefficients,
//                       divided by |G|. Caller takes sqrt() to obtain
//                       the SymBasisState norm.
// ---------------------------------------------------------------------------
template <class SubspaceT>
inline void compute_orbit_for_state(
    const SubspaceT&                 subspace,
    const SpatialProjector&          projector,
    std::uint64_t                    basis,
    const std::vector<Complex>&      phase_factors,
    std::vector<std::uint64_t>&      orbit_elements,
    std::vector<Complex>&            orbit_coefficients,
    double&                          norm_sq)
{
    // Fix 6: Replace std::unordered_map with a stack-allocated flat accumulator.
    // An orbit has at most |G| distinct elements. For all realistic lattice
    // symmetry groups |G| <= 256; a plain linear-dedup pass over a small
    // stack buffer is faster than hash-map construction + destruction for
    // orbit sizes up to ~200 (zero heap allocation, CPU-cache resident).
    // Stack cost: 256 × 24 bytes = 6 KB per call — comfortably within any thread.
    // The outer callers pre-allocate ``orbit_elements`` / ``orbit_coefficients``
    // so the only allocation here is the fixed-size stack array.
    struct OrbElem { std::uint64_t state; Complex coeff; };
    constexpr std::size_t kMaxOrbit = 256;  // covers any realistic lattice group
    OrbElem buf[kMaxOrbit];
    std::size_t n_buf = 0;

    const auto& info = projector.group_info();
    const std::size_t G = info.max_clique.size();

    // ``phase_factors`` comes in two layouts: PER-ELEMENT (length |G|, value
    // χ(max_clique[g]) -- the group_from_generators / distributed convention) or
    // PER-GENERATOR (length num_generators -- the JSON convention, from which
    // χ(g) is reconstructed via power_representation). Detect by length.
    const bool per_element = (phase_factors.size() == G);

    for (std::size_t g = 0; g < G; ++g) {
        Complex character(1.0, 0.0);
        if (per_element) {
            character = phase_factors[g];
        } else {
            const auto& powers = info.power_representation[g];
            for (std::size_t k = 0; k < powers.size(); ++k) {
                const Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) character *= phase;
            }
        }
        const std::uint64_t permuted = projector.apply(basis, g);
        if (subspace.index_of(permuted) < 0) continue;
        const Complex c = std::conj(character);
        // Linear dedup: the orbit is tiny (≤|G|) so O(|G|²) beats hash overhead.
        bool found = false;
        for (std::size_t j = 0; j < n_buf; ++j) {
            if (buf[j].state == permuted) { buf[j].coeff += c; found = true; break; }
        }
        if (!found) buf[n_buf++] = {permuted, c};
    }

    orbit_elements.clear();
    orbit_coefficients.clear();
    norm_sq = 0.0;
    for (std::size_t j = 0; j < n_buf; ++j) {
        if (std::abs(buf[j].coeff) > kOrbitCoefficientEpsilon) {
            orbit_elements.push_back(buf[j].state);
            orbit_coefficients.push_back(buf[j].coeff);
            norm_sq += std::norm(buf[j].coeff);
        }
    }
    if (G > 0) norm_sq /= static_cast<double>(G);
}

// ---------------------------------------------------------------------------
// ProjectorChain (currently only used by tests / future axes)
//
// A type-erased ordered list of projectors. We use std::variant to
// carry the (currently) two heterogeneous future axes plus the
// production SpatialProjector. For today the chain in any production
// path contains 0 or 1 entries; the multi-projector branch is exercised
// only by Phase R5's ABI-smoke tests.
//
// When the chain has more than one entry, every projector in the
// chain must be a duck-type with the same surface as
// ``SpatialProjector`` (size / apply / character / quantum_numbers
// / is_antiunitary). For the abelian Cartesian product, characters
// multiply pointwise across the chain.
// ---------------------------------------------------------------------------
using ChainNode = std::variant<SpatialProjector,
                               InternalZ2Projector,
                               AntiunitaryProjector>;

class ProjectorChain {
public:
    ProjectorChain() = default;

    void push_back(SpatialProjector       p) { nodes_.emplace_back(std::move(p)); }
    void push_back(InternalZ2Projector    p) { nodes_.emplace_back(std::move(p)); }
    void push_back(AntiunitaryProjector   p) { nodes_.emplace_back(std::move(p)); }

    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }
    [[nodiscard]] bool        empty() const noexcept { return nodes_.empty(); }

    /// Total group order = product of component orders. For an empty
    /// chain |G| = 1 (the trivial group consisting only of the
    /// identity), which makes the no-symmetry case fold cleanly into
    /// the same orbit-build path: every orbit has exactly one element
    /// (itself) with coefficient 1 and norm 1.
    [[nodiscard]] std::size_t total_order() const noexcept {
        std::size_t G = 1;
        for (const auto& n : nodes_) {
            std::visit([&](const auto& p) { G *= p.size(); }, n);
        }
        return G;
    }

    [[nodiscard]] const std::vector<ChainNode>& nodes() const noexcept {
        return nodes_;
    }

private:
    std::vector<ChainNode> nodes_;
};

// ---------------------------------------------------------------------------
// Empty-chain helper: an "orbit" of one element (the state itself)
// with coefficient 1 and norm_sq = 1/1 = 1. Used by tests that exercise
// the chain wiring without touching the real character logic.
// ---------------------------------------------------------------------------
template <class SubspaceT>
inline void compute_trivial_orbit(
    const SubspaceT&                 subspace,
    std::uint64_t                    basis,
    std::vector<std::uint64_t>&      orbit_elements,
    std::vector<Complex>&            orbit_coefficients,
    double&                          norm_sq)
{
    orbit_elements.clear();
    orbit_coefficients.clear();
    if (subspace.index_of(basis) < 0) {
        norm_sq = 0.0;
        return;
    }
    orbit_elements.push_back(basis);
    orbit_coefficients.emplace_back(1.0, 0.0);
    norm_sq = 1.0;
}

} // namespace ed::symmetry
