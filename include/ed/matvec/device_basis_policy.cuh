#pragma once
// =============================================================================
// include/ed/matvec/device_basis_policy.cuh
//
// Device-resident BasisPolicy view: the GPU twin of the host
// ``RepSymmetryBasisPolicy`` (``include/ed/matvec/rep_symmetry_basis_policy.h``).
// A POD struct (raw device pointers + scalar fields) that mirrors the host
// policy's ABI via ``__device__``-callable methods.
//
// Contract for adding a new policy:
//   1. Declare a ``DeviceXxxBasisPolicy`` POD with ``__device__`` methods
//      matching the host ABI (``dim``, ``state_of``, ``index_of``, and
//      optionally ``index_and_projection``).
//   2. Add the same compile-time traits (``may_leave_basis``,
//      ``needs_orbit_walk``, ``has_coeff_modifier``).
//   3. Provide host-side code that uploads any backing arrays (basis
//      states, lookup tables, ...) to device memory and returns the POD
//      view (cf. ``GpuRepSectorMirror::basis_view`` in
//      ``src/symmetry/streaming_symmetry_gpu_mirror.cu``).
//
// Memory ownership: the owning host object (``GpuRepSectorMirror`` for
// this policy) RAII-manages the device allocations. The bare POD view is
// non-owning and trivially copyable -- safe to pass by value to a
// ``__global__`` kernel.
// =============================================================================

#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <cstddef>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/matvec/basis_policy.h>
#include <ed/gpu/combinadic.cuh>

namespace ed::matvec::basis {

// ---------------------------------------------------------------------------
// Device sentinel for "state not in basis". Matches the host ABI:
// host returns int64_t{-1}; device returns uint64_t{kDeviceNotFound} to
// avoid signed types in the hot path (atomicAdd loads, etc.).
// ---------------------------------------------------------------------------
inline constexpr std::uint64_t kDeviceNotFound = static_cast<std::uint64_t>(-1);

// ===========================================================================
// DeviceRepSymmetryBasisPolicy  -- on-the-fly representative SpMV
//
// An orbit-CSR layout would materialise, per orbit representative, ALL |G|
// computational images and their character coefficients -- a structure whose
// total length equals the FULL fixed-Sz dimension C(N, n_up) (~601M entries
// at N=32, n_up=16), which defeats the memory win of symmetry.
//
// This policy stores NONE of that. It keeps only:
//   * ``reps[i]``      -- the representative computational state of orbit i
//                         (the sector basis index ``i`` IS the orbit index).
//   * ``inv_norms[i]`` -- ``1 / norm_i`` for orbit i.
//   * ``perms``        -- the |G| site permutations (group action), flat
//                         row-major ``perms[g*n_sites + site]`` (the same
//                         convention as the host ``applyPermutation``).
//   * ``characters[g]``-- the per-SECTOR character ``chi_k(g)`` (one complex
//                         per group element; this is a SINGLE irrep block).
//   * ``rep_index_of_rank`` -- reverse lookup ``rank(representative) ->
//                         orbit index``, length ``C(n_sites, n_up)``; ``-1``
//                         when that fixed-Sz state is not a representative
//                         of a surviving orbit in this irrep.
//
// The group action and the projection phase are regenerated arithmetically
// inside ``index_and_projection`` from ``perms`` + ``characters`` -- no
// O(dim) orbit table. This is the standard Sandvik / HPhi / QuSpin
// representative scheme; the matvec applies H to the single representative
// of each row (``needs_orbit_walk == false``) and the rep-symmetry kernel
// in ``term_kernels_gpu.cuh`` supplies ``pre_phase = inv_norms[i]``.
//
// Math (equivalent to the explicit orbit-sum formulation):
//   For a connected state ``s'`` reached by a term from ``reps[i]`` we need
//   the destination orbit index ``k`` and the projection
//   ``conj(beta_{s'}) / norm_k`` where ``beta_{s'}`` is the coefficient of
//   ``s'`` in basis vector ``k``. Writing ``r_b = min_g g(s')`` (the
//   representative of s'),
//       conj(beta_{s'}) = sum_{h: h(s') = r_b}  conj(chi_k(h))
//   (because the coefficient of g(r_b) in ``sum_g conj(chi(g)) |g r_b>`` is
//   ``conj(chi(g))``, and {g: g(r_b)=s'} = {h^{-1}: h(s')=r_b} with
//   ``chi(h^{-1}) = conj(chi(h))`` for unit-modulus characters). The
//   group_norm (1/|G|) and the orbit walk of the orbit-sum form collapse into
//   the single representative term, so it does NOT appear here.
// ===========================================================================
struct DeviceRepSymmetryBasisPolicy {
    const std::uint64_t*    reps              = nullptr;  // length dim_
    const double*           inv_norms         = nullptr;  // length dim_, 1/norm_i
    const int*              perms             = nullptr;  // group_size * n_sites
    const cuDoubleComplex*  characters        = nullptr;  // length group_size, chi_k(g)
    const std::int32_t*     rep_index_of_rank = nullptr;  // length C(n_sites,n_up)
    // Per-element XOR flip masks for flip-extended groups (element
    // action = perm THEN xor). nullptr = pure permutations. Mirrors the
    // host RepSymmetryBasisPolicy::flips field.
    const std::uint64_t*    flips             = nullptr;  // length group_size
    // Two-level reverse lookup: ONE dense rank -> shared-rep-index table
    // per (N, n_up), shared across every irrep sector's mirror, plus this
    // sector's small local remap. When both are set they take precedence
    // over the per-sector ``rep_index_of_rank`` (which is then not even
    // uploaded). Mirrors the host
    // RepSymmetryBasisPolicy::{shared_rank_of, local_of_shared}.
    const std::int32_t*     shared_rank_of    = nullptr;  // C(N,n_up), shared
    const std::int32_t*     local_of_shared   = nullptr;  // per sector
    // Byte-decomposition permutation LUT (device twin of the host
    // RepSymmetryBasisPolicy fast path): out = OR_b lut[g][b][byte_b(s)].
    // Uses perm_lut_bpw = ceil(N/8) L2-resident gathers per image instead
    // of a serial n_sites-iteration bit walk (36 dependent global loads per
    // image at N=36) -- the canonicalization walk is THE production hot
    // loop (dim x terms x |G| images per matvec; the bit walk measured
    // 26 s/matvec at the 126M-dim 36-site block).
    const std::uint64_t*    perm_lut          = nullptr;
    int                     perm_lut_bpw      = 0;
    std::uint64_t           dim_              = 0;
    int                     group_size        = 1;
    int                     n_sites           = 0;
    int                     n_up              = -1;

    __host__ __device__ inline std::uint64_t dim() const noexcept {
        return dim_;
    }

    // Apply the g'th site permutation to a computational state (same bit
    // convention as the host ``applyPermutation``: output bit i is sourced
    // from input bit perms[g*n_sites + i]).
    __device__ inline std::uint64_t apply_perm(std::uint64_t s, int g) const noexcept {
        if (perm_lut != nullptr) {
            const std::uint64_t* lut_g = perm_lut
                + static_cast<std::size_t>(g) * perm_lut_bpw * 256;
            std::uint64_t r = 0;
            #pragma unroll 5
            for (int b = 0; b < perm_lut_bpw; ++b)
                r |= lut_g[b * 256 + static_cast<int>((s >> (b * 8)) & 0xFF)];
            return (flips != nullptr) ? (r ^ flips[g]) : r;
        }
        const int* p = perms + static_cast<std::size_t>(g) * n_sites;
        std::uint64_t r = 0;
        for (int i = 0; i < n_sites; ++i) {
            r |= ((s >> p[i]) & 1ULL) << i;
        }
        return (flips != nullptr) ? (r ^ flips[g]) : r;
    }

    __device__ inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return reps[idx];
    }

    // Full-space sectors (n_up < 0): no popcount filter, and the
    // "combinadic rank" of a state over the full 2^N space is the
    // state itself (identity), so the reverse table is indexed by rb.
    __device__ inline std::uint64_t rank_of_rep(std::uint64_t rb) const noexcept {
        return (n_up >= 0)
            ? static_cast<std::uint64_t>(
                  ed::gpu::combinadic::rank_state(rb, n_sites, n_up))
            : rb;
    }

    // Reverse lookup rb -> orbit index (-1 sentinel folded to the caller's
    // kDeviceNotFound). Two-level (shared rank table + local remap) when
    // available, per-sector dense table otherwise; when NEITHER table is
    // resident (C(N, n_up) too large for a dense table -- e.g. 36-site
    // half filling at 9.1e9 ranks -- and the host sector carries no
    // two-level lookup), binary-search the sorted ``reps`` array directly.
    // ``reps`` is guaranteed ascending by the producer, exactly like the
    // host RepSymmetryBasisPolicy's PRIMARY lookup.
    __device__ inline std::int32_t index_of_rep_dev(std::uint64_t rb) const noexcept {
        if (shared_rank_of != nullptr) {
            const std::int32_t g = shared_rank_of[rank_of_rep(rb)];
            return (g < 0) ? std::int32_t{-1} : local_of_shared[g];
        }
        if (rep_index_of_rank != nullptr) {
            return rep_index_of_rank[rank_of_rep(rb)];
        }
        std::uint64_t lo = 0, hi = dim_;
        while (lo < hi) {
            const std::uint64_t mid = lo + ((hi - lo) >> 1);
            if (reps[mid] < rb) lo = mid + 1;
            else                hi = mid;
        }
        return (lo < dim_ && reps[lo] == rb)
            ? static_cast<std::int32_t>(lo) : std::int32_t{-1};
    }

    __device__ inline std::uint64_t index_of(std::uint64_t state) const noexcept {
        if (n_up >= 0 && __popcll(state) != n_up) return kDeviceNotFound;
        std::uint64_t rb = state;
        for (int g = 1; g < group_size; ++g) {
            const std::uint64_t img = apply_perm(state, g);
            if (img < rb) rb = img;
        }
        const std::int32_t k = index_of_rep_dev(rb);
        return (k < 0) ? kDeviceNotFound : static_cast<std::uint64_t>(k);
    }

    // Look up the destination orbit index AND the projection phase for a
    // connected computational state ``state`` in ONE shot, regenerating both
    // from the group action on the fly (no orbit table).
    //
    // One pass over the group with a running minimum (as the host policy): the character sum
    // restarts whenever a smaller image appears and grows on ties. No per-thread image
    // buffer (it would live in local memory and cost occupancy), no bound on |G|.
    __device__ inline std::uint64_t
    index_and_projection(std::uint64_t state, cuDoubleComplex& proj_out) const noexcept {
        if (n_up >= 0 && __popcll(state) != n_up) return kDeviceNotFound;
        std::uint64_t rb = ~std::uint64_t{0};
        double acc_re = 0.0, acc_im = 0.0;
        for (int g = 0; g < group_size; ++g) {
            const std::uint64_t img = apply_perm(state, g);
            if (img < rb) {
                rb = img;
                acc_re = 0.0 + cuCreal(characters[g]);   // conj: +real
                acc_im = 0.0 - cuCimag(characters[g]);   //       -imag
            } else if (img == rb) {
                acc_re += cuCreal(characters[g]);
                acc_im -= cuCimag(characters[g]);
            }
        }
        const std::int32_t k = index_of_rep_dev(rb);
        if (k < 0) return kDeviceNotFound;
        const double s = inv_norms[k];
        proj_out = make_cuDoubleComplex(acc_re * s, acc_im * s);
        return static_cast<std::uint64_t>(k);
    }

    // Compile-time traits: the rep-symmetry kernel applies H to the single
    // representative (no orbit walk) but still leaves the basis on
    // off-diagonal terms and needs the per-emit projection phase.
    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = true;
};
}  // namespace ed::matvec::basis

#endif  // WITH_CUDA
