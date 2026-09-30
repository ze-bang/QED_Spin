#pragma once
// =============================================================================
// include/ed/matvec/device_basis_policy.cuh
//
// Device-resident BasisPolicy views: the GPU twin of the host
// BasisPolicy concept in ``include/ed/matvec/basis_policy.h``. Each
// specialization is a POD struct (raw device pointers + scalar fields)
// that mirrors the host policy's ABI 1:1 via ``__device__``-callable
// methods: DeviceFullBasisPolicy, DeviceFixedSzBasisPolicy and
// DeviceRepSymmetryBasisPolicy (the twins of ``FullBasisPolicy`` /
// ``FixedSzBasisPolicy`` / ``RepSymmetryBasisPolicy``).
//
// Contract for adding a new policy (see ``docs/architecture/
// ADD_NEW_GPU_CELL.md``):
//   1. Declare a ``DeviceXxxBasisPolicy`` POD with ``__device__`` methods
//      matching the host ABI (``dim``, ``state_of``, ``index_of``, and
//      optionally ``index_and_projection``).
//   2. Add the same compile-time traits (``may_leave_basis``,
//      ``needs_orbit_walk``, ``has_coeff_modifier``).
//   3. Provide a host-side ``to_device(HostPolicy)`` helper that
//      uploads any backing arrays (basis states, orbit CSR, ...) to
//      device memory and returns the POD view.
//
// Memory ownership: each ``DeviceXxxBasisPolicyHolder`` (in this header
// or in the owning operator class) RAII-manages the device allocations.
// The bare POD view is non-owning and trivially copyable -- safe to pass
// by value to a ``__global__`` kernel.
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
// 1. DeviceFullBasisPolicy
//
// Trivial: state == array index. No device allocation needed.
// ===========================================================================
struct DeviceFullBasisPolicy {
    std::uint64_t n_bits = 0;

    __host__ __device__ inline std::uint64_t dim() const noexcept {
        return 1ULL << n_bits;
    }
    __host__ __device__ inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return idx;
    }
    __host__ __device__ inline std::uint64_t index_of(std::uint64_t state) const noexcept {
        // Full basis: bitstring IS the array index. The kernel gates this
        // behind ``may_leave_basis`` so the check is elided for the full
        // basis, but we keep the signature consistent for template uniformity.
        return state < (1ULL << n_bits) ? state : kDeviceNotFound;
    }

    // Compile-time traits (mirror host FullBasisPolicy).
    static constexpr bool may_leave_basis   = false;
    static constexpr bool needs_orbit_walk  = false;
    static constexpr bool has_coeff_modifier = false;
};

[[nodiscard]] inline DeviceFullBasisPolicy
to_device(const FullBasisPolicy& host) noexcept {
    return DeviceFullBasisPolicy{host.n_bits};
}

// ===========================================================================
// 2. DeviceFixedSzBasisPolicy
//
// Holds device pointers to:
//   * basis_states (sorted, length dim)
//   * Lin index table (open-addressing hash) for O(1) state -> idx
//
// The owning host mirror (the fixed-Sz ``CudaMatVecBackend``, or a Phase 2 mirror
// inside ``StreamingSymmetryOperator``) builds the device tables once
// at construction; this struct is a non-owning view.
// ===========================================================================
struct DeviceFixedSzBasisPolicy {
    const std::uint64_t* basis_states = nullptr;   // sorted, length dim_
    std::uint64_t        dim_         = 0;

    // Open-addressing hash table for state -> idx lookup. Empty slot
    // marked by key == UINT64_MAX. (16-byte entry: key 8 + value 4 +
    // pad 4, naturally aligned for 64-bit loads.)
    struct HashEntry {
        std::uint64_t key;
        std::uint32_t value;
        std::uint32_t _pad;
    };
    const HashEntry* hash_table   = nullptr;
    std::uint32_t    hash_mask    = 0;  // table size = hash_mask + 1, must be power of 2

    // GPU audit (2026-09): Lin (1990) two-table lookup, the same construction
    // the host FixedSzBasisPolicy uses. J_l has 2^(N-N/2) entries (8 B each),
    // J_r 2^(N/2) entries (4 B): 3 MB at N = 36, resident in L2 for every N,
    // versus the 2 x dim x 16 B hash table (86 MB at N = 24) it replaces.
    // When ``lin_jl != nullptr`` this path is taken; the hash stays as the
    // fallback for callers that only supply a sorted basis list.
    const std::uint64_t* lin_jl      = nullptr;
    const std::uint32_t* lin_jr      = nullptr;
    std::uint32_t        lin_n_lower = 0;
    std::uint64_t        lin_mask    = 0;
    int                  lin_n_up    = -1;

    __host__ __device__ inline std::uint64_t dim() const noexcept {
        return dim_;
    }
    __device__ inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return basis_states[idx];
    }
    __device__ inline std::uint64_t index_of(std::uint64_t state) const noexcept {
        if (lin_jl != nullptr) {
            if (__popcll(state) != lin_n_up) return kDeviceNotFound;
            const std::uint64_t base = lin_jl[state >> lin_n_lower];
            if (base == static_cast<std::uint64_t>(-1)) return kDeviceNotFound;
            return base + lin_jr[state & lin_mask];
        }
        // Open-addressing linear probing on a power-of-two table.
        std::uint64_t h = (state * 11400714819323198485ULL) & hash_mask;  // Fibonacci hash
        for (;;) {
            const HashEntry e = hash_table[h];
            if (e.key == static_cast<std::uint64_t>(-1)) return kDeviceNotFound;
            if (e.key == state) return static_cast<std::uint64_t>(e.value);
            h = (h + 1) & hash_mask;
        }
    }

    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = false;
};

// ---------------------------------------------------------------------------
// DeviceFixedSzBasisPolicyHolder
//
// RAII owner of the device-resident backing arrays a
// ``DeviceFixedSzBasisPolicy`` view points at: the sorted basis_states
// array and the open-addressing state->index hash table. The bare POD
// view returned by ``view()`` is non-owning and trivially copyable -- safe
// to hand by value to the unified GPU kernel -- but its pointers are only
// valid while this holder is alive.
//
// ``build()`` constructs the hash on the HOST using the EXACT same probe
// scheme the device ``index_of`` uses (Fibonacci multiply + linear probe on
// a power-of-two table, empty slot == UINT64_MAX), then uploads both
// arrays. The table is sized to a <=0.5 load factor so probe chains stay
// short. This mirrors how ``streaming_symmetry_gpu_mirror.cu`` builds its
// per-sector device hash; here it is generalised behind a reusable owner so
// the FixedSz CUDA lane (and any caller holding a sorted fixed-Sz basis)
// can share one upload path.
// ---------------------------------------------------------------------------
class DeviceFixedSzBasisPolicyHolder {
public:
    DeviceFixedSzBasisPolicyHolder() = default;
    ~DeviceFixedSzBasisPolicyHolder() { reset(); }
    DeviceFixedSzBasisPolicyHolder(const DeviceFixedSzBasisPolicyHolder&) = delete;
    DeviceFixedSzBasisPolicyHolder& operator=(const DeviceFixedSzBasisPolicyHolder&) = delete;
    DeviceFixedSzBasisPolicyHolder(DeviceFixedSzBasisPolicyHolder&& o) noexcept { steal(o); }
    DeviceFixedSzBasisPolicyHolder& operator=(DeviceFixedSzBasisPolicyHolder&& o) noexcept {
        if (this != &o) { reset(); steal(o); }
        return *this;
    }

    // Build + upload the device tables from a lexicographically sorted
    // fixed-Sz basis. Throws std::runtime_error on any CUDA failure.
    void build(const std::vector<std::uint64_t>& sorted_basis_states,
               const LinIndexTable* lin = nullptr) {
        reset();
        dim_ = sorted_basis_states.size();
        if (dim_ == 0) return;

        check_(cudaMalloc(&d_states_, dim_ * sizeof(std::uint64_t)),
               "cudaMalloc(basis_states)");
        check_(cudaMemcpy(d_states_, sorted_basis_states.data(),
                          dim_ * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(basis_states)");

        if (lin != nullptr && !lin->J_l().empty()) {
            // Lin two-table lookup (GPU audit 2026-09): no hash at all.
            const auto& jl = lin->J_l();
            const auto& jr = lin->J_r();
            check_(cudaMalloc(&d_jl_, jl.size() * sizeof(std::uint64_t)), "cudaMalloc(J_l)");
            check_(cudaMemcpy(d_jl_, jl.data(), jl.size() * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice), "cudaMemcpy(J_l)");
            check_(cudaMalloc(&d_jr_, jr.size() * sizeof(std::uint32_t)), "cudaMalloc(J_r)");
            check_(cudaMemcpy(d_jr_, jr.data(), jr.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice), "cudaMemcpy(J_r)");
            lin_n_lower_ = static_cast<std::uint32_t>(lin->n_lower());
            lin_mask_    = lin->lower_mask();
            lin_n_up_    = static_cast<int>(lin->n_up());
            return;
        }

        // Power-of-two table sized for a <=0.5 load factor.
        std::uint64_t table_size = 1;
        while (table_size < dim_ * 2ULL) table_size <<= 1;
        hash_mask_ = static_cast<std::uint32_t>(table_size - 1);

        using HashEntry = DeviceFixedSzBasisPolicy::HashEntry;
        constexpr std::uint64_t kEmpty = static_cast<std::uint64_t>(-1);
        std::vector<HashEntry> host_hash(table_size, HashEntry{kEmpty, 0u, 0u});
        for (std::uint64_t idx = 0; idx < dim_; ++idx) {
            const std::uint64_t state = sorted_basis_states[idx];
            std::uint64_t h = (state * 11400714819323198485ULL) & hash_mask_;
            while (host_hash[h].key != kEmpty) h = (h + 1) & hash_mask_;
            host_hash[h].key   = state;
            host_hash[h].value = static_cast<std::uint32_t>(idx);
        }

        check_(cudaMalloc(&d_hash_, table_size * sizeof(HashEntry)),
               "cudaMalloc(hash_table)");
        check_(cudaMemcpy(d_hash_, host_hash.data(),
                          table_size * sizeof(HashEntry), cudaMemcpyHostToDevice),
               "cudaMemcpy(hash_table)");
    }

    [[nodiscard]] DeviceFixedSzBasisPolicy view() const noexcept {
        DeviceFixedSzBasisPolicy p;
        p.basis_states = d_states_;
        p.dim_         = dim_;
        p.hash_table   = d_hash_;
        p.hash_mask    = hash_mask_;
        p.lin_jl       = d_jl_;
        p.lin_jr       = d_jr_;
        p.lin_n_lower  = lin_n_lower_;
        p.lin_mask     = lin_mask_;
        p.lin_n_up     = lin_n_up_;
        return p;
    }

    [[nodiscard]] std::uint64_t dim() const noexcept { return dim_; }

private:
    static void check_(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DeviceFixedSzBasisPolicyHolder: ") + what +
                " failed: " + cudaGetErrorString(err));
        }
    }
    void reset() noexcept {
        if (d_states_) { cudaFree(d_states_); d_states_ = nullptr; }
        if (d_hash_)   { cudaFree(d_hash_);   d_hash_   = nullptr; }
        if (d_jl_)     { cudaFree(d_jl_);     d_jl_     = nullptr; }
        if (d_jr_)     { cudaFree(d_jr_);     d_jr_     = nullptr; }
        dim_ = 0;
        hash_mask_ = 0;
        lin_n_lower_ = 0; lin_mask_ = 0; lin_n_up_ = -1;
    }
    void steal(DeviceFixedSzBasisPolicyHolder& o) noexcept {
        d_states_  = o.d_states_;  o.d_states_ = nullptr;
        d_hash_    = o.d_hash_;    o.d_hash_   = nullptr;
        d_jl_      = o.d_jl_;      o.d_jl_     = nullptr;
        d_jr_      = o.d_jr_;      o.d_jr_     = nullptr;
        dim_       = o.dim_;       o.dim_      = 0;
        hash_mask_ = o.hash_mask_; o.hash_mask_ = 0;
        lin_n_lower_ = o.lin_n_lower_; lin_mask_ = o.lin_mask_; lin_n_up_ = o.lin_n_up_;
    }

    std::uint64_t*                       d_states_  = nullptr;
    DeviceFixedSzBasisPolicy::HashEntry* d_hash_    = nullptr;
    std::uint64_t*                       d_jl_      = nullptr;
    std::uint32_t*                       d_jr_      = nullptr;
    std::uint64_t                        dim_       = 0;
    std::uint32_t                        hash_mask_ = 0;
    std::uint32_t                        lin_n_lower_ = 0;
    std::uint64_t                        lin_mask_    = 0;
    int                                  lin_n_up_    = -1;
};

// ===========================================================================
// 3. DeviceRepSymmetryBasisPolicy  -- on-the-fly representative SpMV
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
// Math (matches the orbit-CSR reference bit-for-bit; derivation in the plan):
//   For a connected state ``s'`` reached by a term from ``reps[i]`` we need
//   the destination orbit index ``k`` and the projection
//   ``conj(beta_{s'}) / norm_k`` where ``beta_{s'}`` is the coefficient of
//   ``s'`` in basis vector ``k``. Writing ``r_b = min_g g(s')`` (the
//   representative of s'),
//       conj(beta_{s'}) = sum_{h: h(s') = r_b}  conj(chi_k(h))
//   (because the coefficient of g(r_b) in ``sum_g conj(chi(g)) |g r_b>`` is
//   ``conj(chi(g))``, and {g: g(r_b)=s'} = {h^{-1}: h(s')=r_b} with
//   ``chi(h^{-1}) = conj(chi(h))`` for unit-modulus characters). The
//   group_norm (1/|G|) and the orbit walk of the reference collapse into the
//   single representative term, so it does NOT appear here.
// ===========================================================================
struct DeviceRepSymmetryBasisPolicy {
    const std::uint64_t*    reps              = nullptr;  // length dim_
    const double*           inv_norms         = nullptr;  // length dim_, 1/norm_i
    const int*              perms             = nullptr;  // group_size * n_sites
    const cuDoubleComplex*  characters        = nullptr;  // length group_size, chi_k(g)
    const std::int32_t*     rep_index_of_rank = nullptr;  // length C(n_sites,n_up)
    // Stage 8b (SymmetryEngine v2): per-element XOR flip masks for
    // flip-extended groups (element action = perm THEN xor). nullptr =
    // pure permutations (every pre-8b sector). Mirrors the host
    // RepSymmetryBasisPolicy::flips field.
    const std::uint64_t*    flips             = nullptr;  // length group_size
    // Stage-4 two-level reverse lookup, DEVICE twin (Jul 2026): ONE dense
    // rank -> shared-rep-index table per (N, n_up), shared across every
    // irrep sector's mirror, plus this sector's small local remap. When
    // both are set they take precedence over the per-sector
    // ``rep_index_of_rank`` (which is then not even uploaded). Mirrors the
    // host RepSymmetryBasisPolicy::{shared_rank_of, local_of_shared}.
    const std::int32_t*     shared_rank_of    = nullptr;  // C(N,n_up), shared
    const std::int32_t*     local_of_shared   = nullptr;  // per sector
    // Byte-decomposition permutation LUT (Jul 2026, device twin of the host
    // RepSymmetryBasisPolicy fast path): out = OR_b lut[g][b][byte_b(s)].
    // Replaces the serial n_sites-iteration bit walk (36 dependent global
    // loads per image at N=36) with perm_lut_bpw = ceil(N/8) L2-resident
    // gathers -- the canonicalization walk is THE production hot loop
    // (dim x terms x |G| images per matvec; measured 26 s/matvec at the
    // 126M-dim 36-site block before this).
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
    // buffer (it lived in local memory and cost occupancy), no bound on |G|.
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
