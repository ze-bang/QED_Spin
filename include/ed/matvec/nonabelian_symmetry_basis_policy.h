#pragma once
// =============================================================================
// include/ed/matvec/nonabelian_symmetry_basis_policy.h
//
// NonAbelianSymmetryBasisPolicy — the d≥2 generalisation of SymmetryBasisPolicy.
// It lets the NON-abelian symmetry-adapted basis ride the SAME engine
// (CpuMatVecBackend + apply_terms) as the abelian/Sz reductions, instead of a
// parallel matvec.
//
// One difference from the abelian policy, and one only: MULTIPLICITY. An abelian
// sector's orbits partition the Hilbert space, so a connected state s' maps to
// exactly ONE basis index (SymmetryBasisPolicy::index_of). A non-abelian irrep of
// dimension d_Γ carries each free orbit with multiplicity d_Γ, so several SAB
// vectors share an orbit and s' belongs to MANY of them. This policy therefore
// exposes `for_each_target(s', cb)` (multi-index) and sets `multi_target = true`;
// the kernel's `emit` takes its `if constexpr (multi_target)` branch and skips the
// single-index path. Every existing policy lacks the trait, so (via the
// `policy_multi_target_v` detector) their codegen is byte-identical.
//
// The SAB is orthonormal, so each SymBasisState has norm = 1 and the matvec uses
// group_norm = 1: the destination weight is simply conj(c_k(s')) = the dst SAB
// coefficient, read by SymBasisState::findCoeff (no 1/|G|, no 1/norm).
// =============================================================================

#include <complex>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <ed/symmetry/symmetry_sector_data.h>  // ::SymmetrySector, ::SymBasisState

namespace ed::matvec::basis {

// Owning side-table: computational state -> every SAB basis index whose orbit
// contains it (abelian would have lists of length 1; this is the multiplicity
// generalisation). Built once per sector; the policy holds a non-owning view.
struct NonAbelianMultiLookup {
    std::unordered_map<std::uint64_t, std::vector<std::int32_t>> map;

    void build(const ::SymmetrySector& sec) {
        map.clear();
        for (std::int32_t k = 0; k < static_cast<std::int32_t>(sec.basis_states.size()); ++k)
            for (std::uint64_t s : sec.basis_states[static_cast<std::size_t>(k)].orbit_elements)
                map[s].push_back(k);
    }
};

struct NonAbelianSymmetryBasisPolicy {
    const ::SymmetrySector*      sector = nullptr;   // non-owning
    const NonAbelianMultiLookup* lookup = nullptr;   // non-owning

    // -------- compile-time traits --------------------------------------
    static constexpr bool may_leave_basis   = true;
    static constexpr bool needs_orbit_walk  = true;
    static constexpr bool has_coeff_modifier = false;  // superseded by for_each_target
    static constexpr bool multi_target      = true;
    static constexpr bool is_distributed    = false;

    [[nodiscard]] inline std::uint64_t dim() const noexcept {
        return sector->basis_states.size();
    }
    [[nodiscard]] inline std::uint64_t state_of(std::uint64_t i) const noexcept {
        return sector->basis_states[i].orbit_rep;
    }
    // In-sector gate: any basis index containing s, or -1 if s left the sector.
    [[nodiscard]] inline std::int64_t index_of(std::uint64_t s) const noexcept {
        auto it = lookup->map.find(s);
        return (it == lookup->map.end() || it->second.empty())
            ? std::int64_t{-1}
            : static_cast<std::int64_t>(it->second.front());
    }

    // Source walk: orbit elements of SAB vector i with their coefficients
    // (norm = 1, so no 1/norm scaling).
    template <class Callback>
    inline void iter_orbit(std::uint64_t i, Callback&& cb) const {
        const auto& bs = sector->basis_states[i];
        const std::size_t M = bs.orbit_elements.size();
        for (std::size_t k = 0; k < M; ++k) {
            const std::complex<double> a = bs.orbit_coefficients[k];
            if (std::abs(a) < 1e-15) continue;
            cb(bs.orbit_elements[k], a);
        }
    }

    // Multi-target scatter: every SAB vector k containing s' receives the
    // contribution weighted by conj(c_k(s')).
    template <class Callback>
    inline void for_each_target(std::uint64_t s_prime, Callback&& cb) const {
        auto it = lookup->map.find(s_prime);
        if (it == lookup->map.end()) return;
        for (std::int32_t k : it->second) {
            const std::complex<double> c =
                sector->basis_states[static_cast<std::size_t>(k)].findCoeff(s_prime);
            cb(static_cast<std::uint64_t>(k), std::conj(c));
        }
    }

    // Stub: never called (has_coeff_modifier == false, multi_target == true), but
    // kept so the kernel's discarded single-index branch is well-formed for name
    // lookup on any compiler.
    template <class Scalar>
    [[nodiscard]] inline Scalar
    coeff_modifier(std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t) const noexcept {
        return Scalar(0);
    }

    [[nodiscard]] inline bool is_local(std::uint64_t) const noexcept { return true; }
    [[nodiscard]] inline std::uint64_t local_offset() const noexcept { return 0; }
};

}  // namespace ed::matvec::basis
