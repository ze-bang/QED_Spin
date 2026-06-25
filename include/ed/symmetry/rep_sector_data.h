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

#include <ed/core/combinadic.h>  // BinomialTable + rank_state (O(1) reverse lookup)

namespace ed::symmetry {

struct RepSectorData {
    std::vector<std::uint64_t>        reps;        // representative per orbit index
    std::vector<double>               inv_norms;   // 1/norm per orbit index
    std::vector<std::complex<double>> characters;  // chi_k(g), length group_size
    std::vector<int>                  perms_flat;  // group_size * n_sites, row-major
    int group_size = 0;
    int n_sites    = 0;
    int n_up       = -1;  // -1 => not a fixed-Sz sector (rep path needs n_up >= 0)

    // Optional O(1) reverse lookup (host twin of the GPU dense rank table,
    // symmetry_spmv_optimizations.pdf Section 3.3). ``rep_index_of_rank`` maps
    // the combinadic rank of a representative to its orbit index (-1 for a
    // non-representative); ``binom`` is the matching Pascal table. EMPTY by
    // default => the policy falls back to a binary search over ``reps``. Built
    // once per sector (``build_rank_table``) and reused across every solver
    // iteration. Cost: C(n_sites, n_up) * 4 B (~2.4 GiB at N=32, n_up=16).
    std::vector<std::int32_t>          rep_index_of_rank;
    ed::core::combinadic::BinomialTable binom;

    // Byte-decomposition lookup table for fast apply_perm on N≤32 systems.
    // Replaces the N-iteration scalar bit-scatter loop with 4 table lookups,
    // saving ~60% of instruction count at N=32 (4 L2 hits vs 32 scalar ops).
    //
    // Layout: perm_lut_data[(g * perm_lut_bpw + byte_idx) * 256 + byte_val]
    //   = the bits that input byte byte_idx with value byte_val contributes
    //     to the 32-bit output of group element g's permutation.
    // perm_lut_bpw = ceil(n_sites / 8); always 4 for N=32.
    // Size: group_size * perm_lut_bpw * 256 * 4 bytes
    //   (~128 KB for |G|=32, N=32 -- L2-resident).
    // Built by build_perm_lut(); empty/unused when n_sites > 32.
    std::vector<std::uint32_t> perm_lut_data;
    int                        perm_lut_bpw = 0;

    [[nodiscard]] std::uint64_t dim() const noexcept {
        return static_cast<std::uint64_t>(reps.size());
    }

    [[nodiscard]] bool has_rank_table() const noexcept {
        return !rep_index_of_rank.empty();
    }

    // Number of int32 entries a full rank table would need for this sector
    // (== C(n_sites, n_up)). 0 when the sector cannot carry a rank table.
    [[nodiscard]] std::uint64_t rank_table_entries() const noexcept {
        if (n_up < 0 || n_sites <= 0) return 0;
        ed::core::combinadic::BinomialTable b(n_sites);
        return b.at(n_sites, n_up);
    }

    // Build the dense rank -> orbit-index table from ``reps`` only (no orbit
    // images materialised; bit-identical to the GPU build in
    // streaming_symmetry_gpu_mirror.cu). Idempotent / no-op when already built
    // or when the sector is not a usable fixed-Sz sector.
    void build_rank_table() {
        if (has_rank_table()) return;
        if (n_up < 0 || n_sites <= 0 || reps.empty()) return;
        binom.resize(n_sites);
        const std::uint64_t dim_full_sz = binom.at(n_sites, n_up);
        if (dim_full_sz == 0) return;
        rep_index_of_rank.assign(static_cast<std::size_t>(dim_full_sz),
                                 std::int32_t{-1});
        for (std::size_t i = 0; i < reps.size(); ++i) {
            const std::int64_t r = ed::core::combinadic::rank_state(
                reps[i], n_sites, n_up, binom);
            if (r >= 0 && static_cast<std::uint64_t>(r) < dim_full_sz) {
                rep_index_of_rank[static_cast<std::size_t>(r)] =
                    static_cast<std::int32_t>(i);
            }
        }
    }

    // Build the byte-decomposition LUT for N≤32. Idempotent / no-op when
    // already built or when n_sites > 32. See ``perm_lut_data`` for layout.
    void build_perm_lut() {
        if (!perm_lut_data.empty()) return;
        if (n_sites <= 0 || n_sites > 32 || perms_flat.empty()) return;
        const int G   = group_size;
        const int N   = n_sites;
        const int BPW = (N + 7) / 8;   // 4 for N=32
        perm_lut_bpw  = BPW;
        perm_lut_data.assign(static_cast<std::size_t>(G) * BPW * 256, 0u);
        for (int g = 0; g < G; ++g) {
            const int* p = perms_flat.data() + g * N;
            // Invert the permutation: p[i] = src for output bit i
            //   => p_inv[j] = dst for input bit j
            int p_inv[32] = {};
            for (int i = 0; i < N; ++i) p_inv[p[i]] = i;
            for (int byte_idx = 0; byte_idx < BPW; ++byte_idx) {
                const int bit_base = byte_idx * 8;
                for (int byte_val = 0; byte_val < 256; ++byte_val) {
                    std::uint32_t out = 0;
                    for (int b = 0; b < 8 && bit_base + b < N; ++b) {
                        if ((byte_val >> b) & 1)
                            out |= (1u << p_inv[bit_base + b]);
                    }
                    const std::size_t idx =
                        (static_cast<std::size_t>(g) * BPW + byte_idx) * 256
                        + static_cast<std::size_t>(byte_val);
                    perm_lut_data[idx] = out;
                }
            }
        }
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
    // phase_factors: PER-ELEMENT (length |G|, χ(max_clique[g]) directly) or
    // PER-GENERATOR (length num_generators, reconstruct via power_representation).
    const bool per_element = (phase_factors.size() == G);
    for (std::size_t g = 0; g < G; ++g) {
        if (per_element) {
            chi[g] = phase_factors[g];
            continue;
        }
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
