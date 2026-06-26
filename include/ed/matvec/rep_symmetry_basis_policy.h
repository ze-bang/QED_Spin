#pragma once
// =============================================================================
// include/ed/matvec/rep_symmetry_basis_policy.h
//
// RepSymmetryBasisPolicy: the HOST twin of
// ``ed::matvec::basis::DeviceRepSymmetryBasisPolicy``
// (include/ed/matvec/device_basis_policy.cuh).
//
// "Optimized symmetry ED + NLCE" plan (Jun 2026) -- the CPU port of the
// on-the-fly representative SpMV. Like the device policy it stores NO orbit
// CSR; it keeps only the per-orbit representative state + ``1/norm``, the |G|
// site permutations, and the per-sector characters ``chi_k(g)``. The group
// action and the projection phase are regenerated arithmetically inside
// ``index_and_projection`` -- nothing of size O(dim) beyond the representative
// list itself.
//
// Reverse lookup ``state -> orbit index``:
//   * PRIMARY: binary search on the sorted ``reps`` array (O(log dim), zero
//     extra memory). ``reps`` is guaranteed ascending by the producer
//     (``unique_orbit_reps_`` is ``std::sort``-ed; ``getRepSectorData`` keeps
//     survivors in that order).
//   * OPTIONAL O(1): when ``rep_index_of_rank`` is supplied (length
//     C(n_sites, n_up)) the combinadic rank of the representative indexes it
//     directly, exactly as the device policy does. Off by default (the dense
//     table costs C(N,n_up) int32 which is only worth it when reused).
//
// Math (matches the orbit-CSR reference bit-for-bit; see the device policy
// header for the full derivation):
//   For a connected state ``s'`` reached by a term from ``reps[i]`` the
//   destination orbit index ``k`` has representative ``r_b = min_g g(s')`` and
//   projection ``conj(beta_{s'}) / norm_k`` with
//       conj(beta_{s'}) = sum_{h: h(s') = r_b}  conj(chi_k(h)).
//   The rep-symmetry kernel supplies ``pre_phase = inv_norms[i]`` for the
//   single representative row; the group_norm (1/|G|) and the orbit walk of
//   the orbit-CSR reference collapse into this single representative term.
//
// This is a non-owning POD view: the backing arrays live in a
// ``RepSectorData`` (or any caller-owned buffers) that MUST outlive the
// policy. Trivially copyable so the matvec kernel keeps it in registers.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <ed/core/combinadic.h>

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

    // Optional byte-decomposition LUT for fast apply_perm (N≤32).
    // When non-null, replaces the N-iteration scalar bit-scatter loop with
    // ``perm_lut_bpw`` table lookups (~4 for N=32). Pointer into
    // RepSectorData::perm_lut_data; null when N>32 or not built.
    const std::uint32_t* perm_lut     = nullptr;
    int                  perm_lut_bpw = 0;

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
    // N≤32 fast path: uses the byte-decomposition LUT (4 table lookups instead
    // of N scalar bit-scatter ops). At N=32, |G|=32 this saves ~60% of
    // instruction count vs the scalar loop (4 L2 hits vs 32 iterations×3 ops).
    [[nodiscard]] inline std::uint64_t
    apply_perm(std::uint64_t s, int g) const noexcept {
        if (perm_lut != nullptr) {
            const std::uint32_t* lut_g = perm_lut
                + static_cast<std::size_t>(g) * perm_lut_bpw * 256;
            std::uint32_t r = 0;
            for (int b = 0; b < perm_lut_bpw; ++b)
                r |= lut_g[b * 256 + static_cast<int>((s >> (b * 8)) & 0xFF)];
            return static_cast<std::uint64_t>(r);
        }
        // Scalar fallback for N>32 (or when LUT not built).
        const int* p = perms + static_cast<std::size_t>(g) * n_sites;
        std::uint64_t r = 0;
        for (int i = 0; i < n_sites; ++i)
            r |= ((s >> p[i]) & 1ULL) << i;
        return r;
    }

    // Representative of ``state``: numeric minimum over the orbit.
    [[nodiscard]] inline std::uint64_t representative(std::uint64_t state) const noexcept {
        std::uint64_t rb = state;
        for (int g = 1; g < group_size; ++g) {
            const std::uint64_t img = apply_perm(state, g);
            if (img < rb) rb = img;
        }
        return rb;
    }

    // Orbit index of a representative ``rb`` (-1 if not a surviving rep).
    [[nodiscard]] inline std::int64_t index_of_rep(std::uint64_t rb) const noexcept {
        if (rep_index_of_rank != nullptr && binom != nullptr) {
            const std::int64_t r =
                ed::core::combinadic::rank_state(rb, n_sites, n_up, *binom);
            const std::int32_t k = rep_index_of_rank[r];
            return (k < 0) ? -1 : static_cast<std::int64_t>(k);
        }
        // Binary search the sorted representative array.
        const std::uint64_t* first = reps;
        const std::uint64_t* last  = reps + dim_;
        const std::uint64_t* it    = std::lower_bound(first, last, rb);
        if (it == last || *it != rb) return -1;
        return static_cast<std::int64_t>(it - first);
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
    // Optimization (Fix 1): all |G| permuted images are computed ONCE into a
    // stack array. The first pass finds the minimum (representative), the second
    // pass accumulates the projection phase from the same images — halving the
    // apply_perm call count vs the prior two-pass approach (representative() +
    // separate scan). group_size is bounded by realistic lattice groups (≤256).
    [[nodiscard]] inline std::int64_t
    index_and_projection(std::uint64_t state, Complex& proj_out) const noexcept {
        if (n_up >= 0 && __builtin_popcountll(state) != n_up) return -1;

        // Compute all |G| images once; find the minimum (= representative).
        // 256 covers any realistic lattice automorphism group.
        std::uint64_t images[256];
        std::uint64_t rb = state;
        for (int g = 0; g < group_size; ++g) {
            images[g] = apply_perm(state, g);
            if (images[g] < rb) rb = images[g];
        }

        const std::int64_t k = index_of_rep(rb);
        if (k < 0) return -1;

        // Accumulate conj(chi_k) for all h mapping state → rb, using the
        // already-computed images[] — no apply_perm calls here.
        double acc_re = 0.0, acc_im = 0.0;
        for (int h = 0; h < group_size; ++h) {
            if (images[h] == rb) {
                acc_re += characters[h].real();   // conj: +real
                acc_im -= characters[h].imag();   //       -imag
            }
        }
        const double s = inv_norms[static_cast<std::size_t>(k)];
        proj_out = Complex(acc_re * s, acc_im * s);
        return k;
    }

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
    static constexpr bool is_distributed     = false;

    [[nodiscard]] inline bool is_local(std::uint64_t /*g*/) const noexcept {
        return true;
    }
    [[nodiscard]] inline std::uint64_t local_offset() const noexcept { return 0; }
};

}  // namespace ed::matvec::basis
