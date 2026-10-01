#pragma once
// =============================================================================
// include/ed/symmetry/rep_sector_data.h
//
// RepSectorData: the compact, CSR-FREE description of one symmetry sector
// consumed by the on-the-fly representative GPU matvec
// (``apply_terms_rep_symmetry_scatter`` /
// ``ed::symmetry::make_sector_matvec_gpu_rep``).
//
// Instead of storing, per representative, ALL |G| orbit images and their
// character coefficients (an O(full-Sz-dim) structure), a RepSectorData
// stores only:
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
#include <cstdlib>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ed/config/env_registry.h>
#include <ed/core/combinadic.h>  // BinomialTable + rank_state (O(1) reverse lookup)
#include <ed/matvec/rep_symmetry_basis_policy.h>  // RepSymmetryBasisPolicy (make_policy)

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// O(1) rep reverse-lookup gate.
//
// The CSR-free rep matvec can resolve ``state -> orbit index`` either by a
// binary search over the sorted ``reps`` array (O(log dim), zero extra memory)
// or by a dense combinadic rank table (O(1), C(n_sites,n_up) int32). For an
// iterative solver doing thousands of matvecs the table is reused every
// iteration, so it pays for itself -- but it costs ~2.4 GiB at N=32, so it is
// gated by a memory budget: the table is built when its bytes fit in
// ED_SYM_REP_RANKTABLE_BUDGET_GIB (default 8).
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool rep_rank_table_enabled(std::uint64_t table_entries) noexcept {
    if (table_entries == 0) return false;
    double budget_gib = 8.0;
    if (const char* b = ed::env::raw("ED_SYM_REP_RANKTABLE_BUDGET_GIB")) {
        const double parsed = std::atof(b);
        if (parsed > 0.0) budget_gib = parsed;
    }
    const long double table_bytes =
        static_cast<long double>(table_entries) * sizeof(std::int32_t);
    const long double budget_bytes =
        static_cast<long double>(budget_gib) * 1024.0L * 1024.0L * 1024.0L;
    return table_bytes <= budget_bytes;
}

// ---------------------------------------------------------------------------
// SharedRankLookup -- ONE dense ``combinadic rank -> shared-rep-index``
// table per (n_sites, n_up), shared across every irrep sector of that
// subspace, instead of a per-sector C(N,n_up) x int32 table (2.4 GiB EACH
// at N=32 half-filling). Each sector then carries only the small
// ``local_of_shared`` remap (int32 x #reps, ~76 MB at N=32).
// ---------------------------------------------------------------------------
struct SharedRankLookup {
    std::vector<std::int32_t>           shared_of_rank;  // rank -> shared idx, -1
    /// Unique per table (device caches key on it; an address can be reused after a free).
    std::uint64_t                       uid = 0;
    ed::core::combinadic::BinomialTable binom;
    int                                 n_sites = 0;
    int                                 n_up    = -1;
};

[[nodiscard]] inline std::shared_ptr<const SharedRankLookup>
make_shared_rank_lookup(const std::vector<std::uint64_t>& shared_reps,
                        int n_sites, int n_up)
{
    if (n_up < 0 || n_sites <= 0) return nullptr;
    auto srl = std::make_shared<SharedRankLookup>();
    static std::atomic<std::uint64_t> next_uid{1};
    srl->uid     = next_uid.fetch_add(1);
    srl->n_sites = n_sites;
    srl->n_up    = n_up;
    srl->binom.resize(n_sites);
    const std::uint64_t dim_full_sz = srl->binom.at(n_sites, n_up);
    if (dim_full_sz == 0) return nullptr;
    srl->shared_of_rank.assign(static_cast<std::size_t>(dim_full_sz),
                               std::int32_t{-1});
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (long long i = 0; i < static_cast<long long>(shared_reps.size()); ++i) {
        const std::int64_t r = ed::core::combinadic::rank_state(
            shared_reps[static_cast<std::size_t>(i)], n_sites, n_up,
            srl->binom);
        if (r >= 0 && static_cast<std::uint64_t>(r) < dim_full_sz) {
            srl->shared_of_rank[static_cast<std::size_t>(r)] =
                static_cast<std::int32_t>(i);
        }
    }
    return srl;
}

struct RepSectorData {
    std::vector<std::uint64_t>        reps;        // representative per orbit index
    std::vector<double>               inv_norms;   // 1/norm per orbit index
    std::vector<std::complex<double>> characters;  // chi_k(g), length group_size
    std::vector<int>                  perms_flat;  // group_size * n_sites, row-major
    int group_size = 0;
    int n_sites    = 0;
    int n_up       = -1;  // -1 => not a fixed-Sz sector (full-space sentinel)

    // Optional O(1) reverse lookup (host twin of the GPU dense rank table,
    // symmetry_spmv_optimizations.pdf Section 3.3). ``rep_index_of_rank`` maps
    // the combinadic rank of a representative to its orbit index (-1 for a
    // non-representative); ``binom`` is the matching Pascal table. EMPTY by
    // default => the policy falls back to a binary search over ``reps``. Built
    // once per sector (``build_rank_table``) and reused across every solver
    // iteration. Cost: C(n_sites, n_up) * 4 B (~2.4 GiB at N=32, n_up=16).
    std::vector<std::int32_t>          rep_index_of_rank;
    ed::core::combinadic::BinomialTable binom;

    // Per-element XOR masks for flip-extended groups (element action =
    // permute_bits(s, perm) ^ flip_masks[g]). Empty = pure permutations.
    // When non-empty the length must equal ``group_size`` and ``perms_flat``
    // carries the permutation part of every element (the flip half repeats
    // the spatial permutations). The device mirror carries the same masks,
    // so flip-extended sectors run on both CPU and GPU.
    std::vector<std::uint64_t> flip_masks;

    [[nodiscard]] bool has_flips() const noexcept {
        for (std::uint64_t m : flip_masks)
            if (m != 0) return true;
        return false;
    }

    // Byte-decomposition lookup table for fast apply_perm on N≤64 systems.
    // Replaces the N-iteration scalar bit-scatter loop with ceil(N/8) table
    // lookups (~60% fewer instructions at N=32: 4 L2 hits vs 32 scalar ops).
    //
    // Layout: perm_lut_data[(g * perm_lut_bpw + byte_idx) * 256 + byte_val]
    // -- 64-bit output words so every N <= 64 gets the byte-decomposition
    // fast path.
    // perm_lut_bpw = ceil(n_sites / 8); 5 for N=36.
    // Size: group_size * perm_lut_bpw * 256 * 8 bytes (~740 KB at N=36,
    // |G|=72 -- L2-resident on host and device).
    // Built by build_perm_lut(); empty when perms are absent.
    std::vector<std::uint64_t> perm_lut_data;
    int                        perm_lut_bpw = 0;

    [[nodiscard]] std::uint64_t dim() const noexcept {
        return static_cast<std::uint64_t>(reps.size());
    }

    [[nodiscard]] bool has_rank_table() const noexcept {
        return !rep_index_of_rank.empty();
    }

    // Two-level reverse lookup: the SHARED per-(N,n_up) rank table
    // (co-owned across all irrep sectors) + this sector's small
    // shared-idx -> local-idx remap. Preferred over the dense per-sector
    // table when present (``make_policy`` honors it).
    std::shared_ptr<const SharedRankLookup> shared_rank;
    std::vector<std::int32_t>               local_of_shared;  // -1 = cancelled here

    [[nodiscard]] bool has_two_level() const noexcept {
        return shared_rank != nullptr && !local_of_shared.empty();
    }

    // Number of int32 entries a full rank table would need for this sector
    // (== C(n_sites, n_up)). 0 when the sector cannot carry a rank table.
    [[nodiscard]] std::uint64_t rank_table_entries() const noexcept {
        if (n_sites <= 0) return 0;
        if (n_up < 0) {
            // State-indexed identity-rank table over the full 2^N.
            return (n_sites <= 31) ? (1ULL << n_sites) : 0;
        }
        ed::core::combinadic::BinomialTable b(n_sites);
        return b.at(n_sites, n_up);
    }

    // Build the dense rank -> orbit-index table from ``reps`` only (no orbit
    // images materialised; bit-identical to the GPU build in
    // streaming_symmetry_gpu_mirror.cu). Idempotent / no-op when already built
    // or when the sector has no reps.
    void build_rank_table() {
        if (has_rank_table()) return;
        if (n_sites <= 0 || reps.empty()) return;
        if (n_up < 0) {
            // Full-space / parity / flip-extended sectors: the rank of
            // a state over the full 2^N enumeration is the state
            // itself (identity), so the reverse table is state-indexed
            // (2^N int32; the caller budget-gates via
            // rank_table_entries + rep_rank_table_enabled).
            if (n_sites > 31) return;
            const std::uint64_t dim_all = (1ULL << n_sites);
            rep_index_of_rank.assign(static_cast<std::size_t>(dim_all),
                                     std::int32_t{-1});
            for (std::size_t i = 0; i < reps.size(); ++i) {
                rep_index_of_rank[static_cast<std::size_t>(reps[i])] =
                    static_cast<std::int32_t>(i);
            }
            binom.resize(n_sites);   // policy precondition (unused here)
            return;
        }
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

    // Build the byte-decomposition LUT for N≤64. Idempotent / no-op when
    // already built or when n_sites > 64. See ``perm_lut_data`` for layout.
    void build_perm_lut() {
        if (!perm_lut_data.empty()) return;
        if (n_sites <= 0 || n_sites > 64 || perms_flat.empty()) return;
        const int G   = group_size;
        const int N   = n_sites;
        const int BPW = (N + 7) / 8;   // 5 for N=36
        perm_lut_bpw  = BPW;
        perm_lut_data.assign(static_cast<std::size_t>(G) * BPW * 256, 0ULL);
        for (int g = 0; g < G; ++g) {
            const int* p = perms_flat.data() + g * N;
            // Invert the permutation: p[i] = src for output bit i
            //   => p_inv[j] = dst for input bit j
            int p_inv[64] = {};
            for (int i = 0; i < N; ++i) p_inv[p[i]] = i;
            for (int byte_idx = 0; byte_idx < BPW; ++byte_idx) {
                const int bit_base = byte_idx * 8;
                for (int byte_val = 0; byte_val < 256; ++byte_val) {
                    std::uint64_t out = 0;
                    for (int b = 0; b < 8 && bit_base + b < N; ++b) {
                        if ((byte_val >> b) & 1)
                            out |= (1ULL << p_inv[bit_base + b]);
                    }
                    const std::size_t idx =
                        (static_cast<std::size_t>(g) * BPW + byte_idx) * 256
                        + static_cast<std::size_t>(byte_val);
                    perm_lut_data[idx] = out;
                }
            }
        }
    }

    // Non-owning host policy view over this data. THE single source of the
    // RepSectorData -> RepSymmetryBasisPolicy mapping: the matvec factory
    // (``rep_policy_from``) and the little-group engine both route through
    // here so the two-level rank table / flip masks / perm LUT wiring can
    // never drift between consumers. The returned view holds raw
    // pointers into this object's vectors -- keep it alive for the policy's
    // lifetime.
    [[nodiscard]] ed::matvec::basis::RepSymmetryBasisPolicy
    make_policy() const noexcept {
        ed::matvec::basis::RepSymmetryBasisPolicy p;
        p.reps       = reps.data();
        p.inv_norms  = inv_norms.data();
        p.perms      = perms_flat.data();
        p.characters = characters.data();
        p.dim_       = reps.size();
        p.group_size = group_size;
        p.n_sites    = n_sites;
        p.n_up       = n_up;
        // Two-level lookup takes precedence: shared rank table (one per
        // (N, n_up)) + per-sector local remap. Then the dense per-sector
        // table; index_of_rep falls back to binary search when neither is
        // set.
        if (has_two_level()) {
            p.shared_rank_of  = shared_rank->shared_of_rank.data();
            p.local_of_shared = local_of_shared.data();
            p.binom           = &shared_rank->binom;
        } else if (has_rank_table()) {
            p.rep_index_of_rank = rep_index_of_rank.data();
            p.binom             = &binom;
        }
        // Fast apply_perm: byte-decomposition LUT (ceil(N/8) lookups vs N iters).
        if (!perm_lut_data.empty()) {
            p.perm_lut     = perm_lut_data.data();
            p.perm_lut_bpw = perm_lut_bpw;
        }
        // Flip-extended elements (perm THEN xor).
        if (!flip_masks.empty()) {
            p.flips = flip_masks.data();
        }
        return p;
    }

    // A RepSectorData is usable by the rep matvec only when it carries a
    // non-empty group action with matching characters / permutations, and
    // either a fixed-Sz magnetisation or the full-space sentinel.
    [[nodiscard]] bool usable() const noexcept {
        // n_up == -1 is the full-space sentinel (the rep policy skips the
        // popcount filter) and must be accepted.
        return n_up >= -1 && group_size > 0 && n_sites > 0
            && !reps.empty()
            && characters.size() == static_cast<std::size_t>(group_size)
            && perms_flat.size() ==
                   static_cast<std::size_t>(group_size) * n_sites;
    }
};

} // namespace ed::symmetry
