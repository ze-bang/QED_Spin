#pragma once
// =============================================================================
// include/ed/symmetry/rep_sector_data.h
//
// RepSectorData: the compact, CSR-FREE description of one symmetry sector
// consumed by the on-the-fly representative GPU matvec
// (``apply_terms_rep_symmetry_scatter`` /
// ``ed::symmetry::make_sector_matvec_gpu_rep``).
//
// "On-the-fly representative SpMV for streaming symmetry" plan (Jun 2026).
//
// Where a ``SymmetrySector`` stores, per representative, ALL |G| orbit images
// and their character coefficients (an O(full-Sz-dim) structure), a
// RepSectorData stores only:
//
//   * ``reps``       -- the representative computational state per orbit
//                       (the sector basis index IS the array index).
//   * ``inv_norms``  -- ``1 / norm_i`` per orbit (same ordering as ``reps``).
//   * ``characters`` -- the per-SECTOR character ``chi_k(g)`` (one complex per
//                       group element; this object describes ONE irrep block).
//   * ``perms_flat`` -- the |G| site permutations, row-major
//                       ``perms_flat[g*n_sites + site]`` (same bit convention
//                       as the host ``applyPermutation``).
//
// The group action and projection phases are regenerated arithmetically on
// the device from ``perms_flat`` + ``characters``; nothing of size O(dim) is
// stored or streamed. This is a plain value type (no CUDA), so CPU
// translation units can build it and hand it to the CUDA mirror factory.
// =============================================================================

#include <complex>
#include <cstdint>
#include <vector>

namespace ed::symmetry {

struct RepSectorData {
    std::vector<std::uint64_t>        reps;        // representative per orbit index
    std::vector<double>               inv_norms;   // 1/norm per orbit index
    std::vector<std::complex<double>> characters;  // chi_k(g), length group_size
    std::vector<int>                  perms_flat;  // group_size * n_sites, row-major
    int group_size = 0;
    int n_sites    = 0;
    int n_up       = -1;  // -1 => not a fixed-Sz sector (rep path needs n_up >= 0)

    [[nodiscard]] std::uint64_t dim() const noexcept {
        return static_cast<std::uint64_t>(reps.size());
    }

    // A RepSectorData is usable by the rep matvec only when it carries a
    // fixed-Sz magnetisation (the device reverse lookup is a combinadic rank
    // table over C(n_sites, n_up)) and a non-empty group action.
    [[nodiscard]] bool usable() const noexcept {
        return n_up >= 0 && group_size > 0 && n_sites > 0
            && !reps.empty()
            && characters.size() == static_cast<std::size_t>(group_size)
            && perms_flat.size() ==
                   static_cast<std::size_t>(group_size) * n_sites;
    }
};

// Compose the per-sector spatial character ``chi_k(g)`` for every group
// element from the per-generator ``phase_factors`` and the cached
// ``power_representation`` -- the SAME Bloch-convention product that
// ``ed::symmetry::compute_orbit_for_state`` / ``SpatialProjector::character``
// use, so the on-the-fly phase matches the orbit-CSR reference bit-for-bit:
//
//   chi_k(g) = product_gen phase_factors[gen] ^ power_representation[g][gen]
//
// Templated on the group-info type to avoid pulling the (heavier) symmetry
// metadata header into this value-type header; any type exposing
// ``max_clique`` + ``power_representation`` (i.e. ``SymmetryGroupInfo``)
// satisfies it.
template <class GroupInfoT>
[[nodiscard]] inline std::vector<std::complex<double>>
sector_characters_from(const GroupInfoT&                        info,
                       const std::vector<std::complex<double>>& phase_factors) {
    const std::size_t G = info.max_clique.size();
    std::vector<std::complex<double>> chi(G, std::complex<double>(1.0, 0.0));
    for (std::size_t g = 0; g < G; ++g) {
        const auto& powers = info.power_representation[g];
        std::complex<double> c(1.0, 0.0);
        for (std::size_t k = 0; k < powers.size(); ++k) {
            const std::complex<double> phase = phase_factors[k];
            for (int p = 0; p < powers[k]; ++p) c *= phase;
        }
        chi[g] = c;
    }
    return chi;
}

// Flatten a group's site permutations (``max_clique``) into the row-major
// ``perms_flat`` layout the device policy consumes.
template <class GroupInfoT>
[[nodiscard]] inline std::vector<int>
flatten_group_perms(const GroupInfoT& info, int n_sites) {
    const std::size_t G = info.max_clique.size();
    std::vector<int> flat(G * static_cast<std::size_t>(n_sites), 0);
    for (std::size_t g = 0; g < G; ++g) {
        const auto& perm = info.max_clique[g];
        for (int i = 0; i < n_sites; ++i) {
            flat[g * static_cast<std::size_t>(n_sites) + i] =
                (i < static_cast<int>(perm.size())) ? perm[i] : i;
        }
    }
    return flat;
}

} // namespace ed::symmetry
