#pragma once
// =============================================================================
// include/ed/matvec/rep_symmetry_basis_policy.h
//
// RepSymmetryBasisPolicy: the HOST twin of
// ``ed::matvec::basis::DeviceRepSymmetryBasisPolicy``
// (include/ed/gpu/device_basis_policy.cuh).
//
// The CPU form of the on-the-fly representative SpMV. Like the device
// policy it stores NO orbit
// CSR; it keeps only the per-orbit representative state + ``1/norm``, the |G|
// site permutations, and the per-sector characters ``chi_k(g)``. The group
// action and the projection phase are regenerated arithmetically inside
// ``index_and_projection`` -- nothing of size O(dim) beyond the representative
// list itself.
//
// Reverse lookup ``state -> orbit index``:
//   * PRIMARY: binary search on the sorted ``reps`` array (O(log dim), zero
//     extra memory). ``reps`` is guaranteed ascending by the producer
//     (``OrbitTable::reps`` is ascending and sector construction keeps
//     survivors in that order).
//   * OPTIONAL O(1): when ``rep_index_of_rank`` is supplied (length
//     C(n_sites, n_up)) the combinadic rank of the representative indexes it
//     directly, exactly as the device policy does. The dense table costs
//     C(N,n_up) int32, so it is built only within a memory budget (see
//     ``rep_rank_table_enabled`` in rep_sector.h).
//
// Math (equivalent to the explicit orbit-sum formulation; see the device
// policy header for the full derivation):
//   For a connected state ``s'`` reached by a term from ``reps[i]`` the
//   destination orbit index ``k`` has representative ``r_b = min_g g(s')`` and
//   projection ``conj(beta_{s'}) / norm_k`` with
//       conj(beta_{s'}) = sum_{h: h(s') = r_b}  conj(chi_k(h)).
//   The rep-symmetry kernel supplies ``pre_phase = inv_norms[i]`` for the
//   single representative row; the group_norm (1/|G|) and the orbit walk of
//   the orbit-sum form collapse into this single representative term.
//
// This is a non-owning POD view: the backing arrays live in a
// ``RepSectorData`` (or any caller-owned buffers) that MUST outlive the
// policy. Trivially copyable so the matvec kernel keeps it in registers.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <ed/basis/combinadic.h>
#include <ed/basis/sublattice_code.h>

namespace ed::matvec::basis {

struct RepSymmetryBasisPolicy {
    using Complex = std::complex<double>;

    const std::uint64_t* reps       = nullptr;  // sorted ascending, length dim_
    const double*        inv_norms   = nullptr;  // 1/norm_i, length dim_
    const int*           perms       = nullptr;  // group_size * n_sites, row-major
    const Complex*       characters  = nullptr;  // chi_k(g), length group_size
    std::uint64_t        dim_        = 0;
    int                  group_size  = 1;
    int                  n_sites     = 0;
    int                  n_up        = -1;

    // Optional O(1) reverse lookup. When ``rep_index_of_rank`` is non-null it
    // maps combinadic rank(rep) -> orbit index (-1 if absent); ``binom`` is the
    // matching Pascal table. Both null => binary-search ``reps``.
    const std::int32_t*                       rep_index_of_rank = nullptr;
    const ed::core::combinadic::BinomialTable* binom            = nullptr;

    // Per-element XOR flip masks (nullptr = pure permutations). Applied AFTER the permutation; for the global
    // spin flip (all-ones mask) the order is immaterial since the mask is
    // permutation-invariant.
    const std::uint64_t* flips = nullptr;

    // Two-level O(1) lookup, preferred when set:
    // rank -> SHARED rep index (one dense table per (N, n_up), shared across
    // every irrep sector) -> this sector's local index via the small
    // ``local_of_shared`` remap (-1 = orbit cancels in this irrep).
    const std::int32_t* shared_rank_of  = nullptr;  // C(N,n_up) entries, shared
    const std::int32_t* local_of_shared = nullptr;  // per sector, shared-rep count

    // Rank buckets (RepSectorData::build_buckets): offsets into `reps` of the reps whose key (rank
    // at fixed Sz, else the state) lies in each run of 2^bucket_shift keys above bucket_base.
    const std::uint32_t* bucket_off   = nullptr;   // n_buckets + 1 offsets
    std::uint64_t        n_buckets    = 0;
    std::uint64_t        bucket_base  = 0;
    int                  bucket_shift = 0;

    // Optional byte-decomposition LUT for fast apply_perm (N <= 64).
    // When non-null, replaces the N-iteration scalar bit-scatter loop with
    // ``perm_lut_bpw`` table lookups (5 for N=36). Pointer into
    // RepSectorData::perm_lut_data; null when not built. 64-bit words so
    // every N <= 64 rides the fast path.
    const std::uint64_t* perm_lut     = nullptr;
    int                  perm_lut_bpw = 0;

    // An irrep of dimension d > 1 (rep_sector.h, P6.3): D(g) for every element, and per rep its
    // stabiliser class (rank and C) and its first state. Null for d = 1.
    int                  irrep_dim    = 1;
    const Complex*       irrep_D      = nullptr;   // |G| d x d, row-major
    const Complex*       class_C      = nullptr;   // per class: d x d, row-major
    const std::uint8_t*  class_rank   = nullptr;   // per class
    const std::uint16_t* rep_class    = nullptr;   // per rep
    const std::uint64_t* state_offset = nullptr;   // per rep, and one past

    // The group's sublattice code (<ed/basis/sublattice_code.h>; RepSectorData::make_policy sets it):
    // representatives are least in its key order, found from the candidates alone. Empty: the plain
    // order, every element scanned.
    ed::symmetry::SublatticeView slc;

    [[nodiscard]] inline std::uint64_t dim() const noexcept { return dim_; }

    [[nodiscard]] inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return reps[idx];
    }

    [[nodiscard]] inline double inv_norm_of(std::uint64_t idx) const noexcept {
        return inv_norms[idx];
    }

    // Apply the g'th site permutation (same bit convention as the host
    // ``applyPermutation`` and the device ``apply_perm``).
    //
    // Fast path: uses the byte-decomposition LUT (ceil(N/8) table lookups
    // instead of N scalar bit-scatter ops). At N=32, |G|=32 this saves ~60% of
    // instruction count vs the scalar loop (4 L2 hits vs 32 iterations×3 ops).
    [[nodiscard]] inline std::uint64_t
    apply_perm(std::uint64_t s, int g) const noexcept {
        const std::uint64_t flip = (flips != nullptr) ? flips[g] : 0ULL;
        if (perm_lut != nullptr) {
            const std::uint64_t* lut_g = perm_lut
                + static_cast<std::size_t>(g) * perm_lut_bpw * 256;
            std::uint64_t r = 0;
            for (int b = 0; b < perm_lut_bpw; ++b)
                r |= lut_g[b * 256 + static_cast<int>((s >> (b * 8)) & 0xFF)];
            return r ^ flip;
        }
        // Scalar fallback when the LUT is not built.
        const int* p = perms + static_cast<std::size_t>(g) * n_sites;
        std::uint64_t r = 0;
        for (int i = 0; i < n_sites; ++i)
            r |= ((s >> p[i]) & 1ULL) << i;
        return r ^ flip;
    }

    // fn(g, image, key) for the elements that can map ``state`` to its representative, in ascending
    // element order, with the key the images are compared by: every element and the image itself
    // (the plain order), or the sublattice candidates and the image's key.
    template <class Fn>
    inline void for_each_image(std::uint64_t state, Fn&& fn) const noexcept {
        if (slc.engaged()) {
            slc.for_each_candidate(slc.key(state), [&](int g) {
                const std::uint64_t img = apply_perm(state, g);
                fn(g, img, slc.key(img));
            });
            return;
        }
        for (int g = 0; g < group_size; ++g) {
            const std::uint64_t img = apply_perm(state, g);
            fn(g, img, img);
        }
    }

    // Representative of ``state``: the orbit's least member (in the key order with a sublattice code).
    [[nodiscard]] inline std::uint64_t representative(std::uint64_t state) const noexcept {
        std::uint64_t rb = state, best = ~std::uint64_t{0};
        for_each_image(state, [&](int, std::uint64_t img, std::uint64_t k) {
            if (k < best) { best = k; rb = img; }
        });
        return rb;
    }

    // Orbit index of a representative ``rb`` (-1 if not a surviving rep).
    [[nodiscard]] inline std::int64_t index_of_rep(std::uint64_t rb) const noexcept {
        if (shared_rank_of != nullptr && local_of_shared != nullptr
            && binom != nullptr) {
            const std::int64_t r =
                ed::core::combinadic::rank_state(rb, n_sites, n_up, *binom);
            const std::int32_t g = shared_rank_of[r];
            if (g < 0) return -1;
            const std::int32_t k = local_of_shared[g];
            return (k < 0) ? -1 : static_cast<std::int64_t>(k);
        }
        if (rep_index_of_rank != nullptr
            && (n_up < 0 || binom != nullptr)) {
            // n_up < 0: full-space/parity sectors use the identity
            // rank (state-indexed table).
            const std::int64_t r = (n_up < 0)
                ? static_cast<std::int64_t>(rb)
                : ed::core::combinadic::rank_state(rb, n_sites, n_up,
                                                   *binom);
            const std::int32_t k = rep_index_of_rank[r];
            return (k < 0) ? -1 : static_cast<std::int64_t>(k);
        }
        // One bucket of the sorted reps, else all of them.
        const std::uint64_t* first = reps;
        const std::uint64_t* last  = reps + dim_;
        if (bucket_off != nullptr) {
            if (n_up >= 0 && __builtin_popcountll(rb) != n_up) return -1;
            const std::uint64_t key = (n_up < 0) ? rb
                : static_cast<std::uint64_t>(ed::core::combinadic::rank_state(rb, n_sites, n_up, *binom));
            if (key < bucket_base) return -1;
            const std::uint64_t b = (key - bucket_base) >> bucket_shift;
            if (b >= n_buckets) return -1;
            first = reps + bucket_off[b];
            last  = reps + bucket_off[b + 1];
        }
        const std::uint64_t* it    = std::lower_bound(first, last, rb);
        if (it == last || *it != rb) return -1;
        return static_cast<std::int64_t>(it - reps);
    }

    [[nodiscard]] inline std::int64_t index_of(std::uint64_t state) const noexcept {
        // n_up < 0 marks the full 2^N space (no Sz constraint) -- skip the
        // popcount membership filter (every state is in-subspace).
        if (n_up >= 0 && __builtin_popcountll(state) != n_up) return -1;
        return index_of_rep(representative(state));
    }


    // Fused destination index + projection phase for a connected state, the
    // host equivalent of the device ``index_and_projection``. Returns the
    // orbit index ``k`` (or -1) and writes ``conj(beta_state) * inv_norms[k]``
    // into ``proj_out``.
    //
    // One pass over the group with a running minimum: the character sum restarts whenever a
    // smaller image appears and grows on ties, so it ends as the sum over the elements that
    // map the state to its representative, in ascending element order (the same terms, in
    // the same order, as a separate second pass). No image buffer, no bound on |G|.
    [[nodiscard]] inline std::int64_t
    index_and_projection(std::uint64_t state, Complex& proj_out) const noexcept {
        if (n_up >= 0 && __builtin_popcountll(state) != n_up) return -1;
        std::uint64_t rb = ~std::uint64_t{0}, best = ~std::uint64_t{0};
        double acc_re = 0.0, acc_im = 0.0;
        for_each_image(state, [&](int g, std::uint64_t img, std::uint64_t k) {
            if (k < best) {
                best = k;
                rb = img;
                acc_re = 0.0 + characters[g].real();   // conj: +real (from 0.0, as a sum)
                acc_im = 0.0 - characters[g].imag();   //       -imag
            } else if (k == best) {
                acc_re += characters[g].real();
                acc_im -= characters[g].imag();
            }
        });
        const std::int64_t k = index_of_rep(rb);
        if (k < 0) return -1;
        const double s = inv_norms[static_cast<std::size_t>(k)];
        proj_out = Complex(acc_re * s, acc_im * s);
        return k;
    }

    // The d > 1 form (rep_sector.h): the orbit index k of `state` (or -1) and, into A (d x d,
    // row-major), A = sum_{g: g state = rep_k} D(g)^T -- the same running-minimum pass.
    [[nodiscard]] inline std::int64_t
    index_and_matrix(std::uint64_t state, Complex* A) const noexcept {
        if (n_up >= 0 && __builtin_popcountll(state) != n_up) return -1;
        const int d = irrep_dim;
        const std::size_t dd = static_cast<std::size_t>(d) * static_cast<std::size_t>(d);
        std::uint64_t rb = ~std::uint64_t{0}, best = ~std::uint64_t{0};
        for_each_image(state, [&](int g, std::uint64_t img, std::uint64_t k) {
            if (k > best) return;
            if (k < best) {
                best = k;
                rb = img;
                for (std::size_t e = 0; e < dd; ++e) A[e] = Complex(0.0, 0.0);
            }
            const Complex* Dg = irrep_D + static_cast<std::size_t>(g) * dd;
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) A[i * d + j] += Dg[j * d + i];
        });
        return index_of_rep(rb);
    }

    /// d > 1: rep k's rank, its class's C (d x d, row-major; columns >= rank zero) and its first state.
    [[nodiscard]] inline int rank_of(std::uint64_t k) const noexcept { return class_rank[rep_class[k]]; }
    [[nodiscard]] inline const Complex* C_of(std::uint64_t k) const noexcept {
        return class_C + static_cast<std::size_t>(rep_class[k]) * static_cast<std::size_t>(irrep_dim * irrep_dim);
    }
    [[nodiscard]] inline std::uint64_t first_state_of(std::uint64_t k) const noexcept { return state_offset[k]; }

    // ----- Trait surface --------------------------------------------------
    // ``is_rep_symmetry`` selects the dedicated rep-symmetry kernel + forces
    // the complex matrix-free path in CpuMatVecBackend (no CSR, no real-input
    // fast path that would drop momentum phases). ``may_leave_basis`` is true
    // (off-diagonal terms reach other orbits); ``needs_orbit_walk`` is false
    // (the kernel applies H to the single representative).
    static constexpr bool is_rep_symmetry    = true;
    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = true;
};

}  // namespace ed::matvec::basis
