#pragma once
// =============================================================================
// include/ed/matvec/device_basis_policy.cuh
//
// Phase 1 of the "Unified CPU/GPU symmetry architecture" plan (May 2026).
//
// Device-resident BasisPolicy views: the GPU twin of the host
// BasisPolicy concept in ``include/ed/matvec/basis_policy.h``. Each
// specialization is a POD struct (raw device pointers + scalar fields)
// that mirrors the host policy's ABI 1:1 via ``__device__``-callable
// methods. The unified kernel ``ed::matvec::kernel::gpu::apply_terms_gpu``
// reads these the same way the CPU ``apply_terms`` reads
// ``FullBasisPolicy`` / ``FixedSzBasisPolicy`` / ``SymmetryBasisPolicy`` /
// ``FixedSzSymmetryBasisPolicy``.
//
// Contract for adding a new policy (see ``docs/architecture/
// ADD_NEW_GPU_CELL.md``):
//   1. Declare a ``DeviceXxxBasisPolicy`` POD with ``__device__`` methods
//      matching the host ABI (``dim``, ``state_of``, ``index_of``, and
//      optionally ``iter_orbit`` / ``coeff_modifier`` / ``is_local`` /
//      ``local_offset``).
//   2. Add the same compile-time traits (``may_leave_basis``,
//      ``needs_orbit_walk``, ``has_coeff_modifier``, ``is_distributed``).
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
    static constexpr bool is_distributed    = false;

    __device__ inline bool is_local(std::uint64_t /*g*/) const noexcept { return true; }
    __device__ inline std::uint64_t local_offset() const noexcept { return 0; }
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
// The owning host class (``GPUFixedSzOperator``, or a Phase 2 mirror
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

    __host__ __device__ inline std::uint64_t dim() const noexcept {
        return dim_;
    }
    __device__ inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return basis_states[idx];
    }
    __device__ inline std::uint64_t index_of(std::uint64_t state) const noexcept {
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
    static constexpr bool is_distributed     = false;

    __device__ inline bool is_local(std::uint64_t /*g*/) const noexcept { return true; }
    __device__ inline std::uint64_t local_offset() const noexcept { return 0; }
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
    void build(const std::vector<std::uint64_t>& sorted_basis_states) {
        reset();
        dim_ = sorted_basis_states.size();
        if (dim_ == 0) return;

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

        check_(cudaMalloc(&d_states_, dim_ * sizeof(std::uint64_t)),
               "cudaMalloc(basis_states)");
        check_(cudaMemcpy(d_states_, sorted_basis_states.data(),
                          dim_ * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
               "cudaMemcpy(basis_states)");
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
        dim_ = 0;
        hash_mask_ = 0;
    }
    void steal(DeviceFixedSzBasisPolicyHolder& o) noexcept {
        d_states_  = o.d_states_;  o.d_states_ = nullptr;
        d_hash_    = o.d_hash_;    o.d_hash_   = nullptr;
        dim_       = o.dim_;       o.dim_      = 0;
        hash_mask_ = o.hash_mask_; o.hash_mask_ = 0;
    }

    std::uint64_t*                       d_states_  = nullptr;
    DeviceFixedSzBasisPolicy::HashEntry* d_hash_    = nullptr;
    std::uint64_t                        dim_       = 0;
    std::uint32_t                        hash_mask_ = 0;
};

// ===========================================================================
// 3. DeviceSymmetryBasisPolicy
//
// Holds device-resident orbit CSR data + a pre-baked coefficient modifier
// hash. Mirrors the optimization the former GPUSymmetrizedOperator
// did: the per-emit phase
// ``conj(beta_{s'}) * group_norm / norm_k`` is precomputed at upload
// time and stored in the hash, so the kernel's coeff_modifier lookup
// is a single hash hit, not a recomputation.
//
// Each entry maps a computational state s' to (basis_idx k, projection).
// ===========================================================================
struct DeviceSymmetryHashEntry {
    std::uint64_t   key;          // state s' (UINT64_MAX = empty)
    std::uint32_t   value;        // basis index k
    std::uint32_t   _pad;
    cuDoubleComplex projection;   // conj(beta_{s'}) * group_norm / norm_k
};

struct DeviceSymmetryBasisPolicy {
    // Orbit CSR: orbit_offsets[j..j+1) -> orbit_elements[k] are the
    // computational states in orbit j; orbit_coefficients[k] is alpha_s.
    //
    // Phase I of the "Close CPU / GPU Gaps" plan (May 2026):
    // ``orbit_inv_norms[j]`` stores ``1.0 / norm_j`` (pre-baked at
    // mirror construction) so the inner kernel loop only multiplies
    // instead of dividing once per orbit walk. The legacy name
    // ``orbit_norms`` (raw norms) was removed in this rename --
    // the only consumer was a single ``1.0 / basis.orbit_norms[i]``
    // inside ``apply_terms_gpu_scatter`` which now reads
    // ``basis.orbit_inv_norms[i]`` directly.
    const std::uint64_t*    orbit_elements     = nullptr;
    const cuDoubleComplex*  orbit_coefficients = nullptr;
    const std::uint32_t*    orbit_offsets      = nullptr;  // length dim_+1
    const double*           orbit_inv_norms    = nullptr;  // length dim_, 1/norm_j
    std::uint64_t           dim_               = 0;
    double                  group_norm         = 1.0;      // 1.0 / |G|

    // -------------------------------------------------------------------
    // Reverse lookup: state -> (basis_idx, projection).
    //
    // Phase E.1 of the "Kill the GPU State-Lookup Hash" plan (May 2026)
    // adds a two-level dense indirection ("rank table") that replaces
    // the open-addressing hash table for the sym+Sz workload. When
    // ``sz_to_sec != nullptr`` the device lookup is:
    //   1. ``r = combinadic::rank_state(state, n_sites, n_up)`` --
    //      constant-cache compute, zero global memory traffic.
    //   2. ``sec_idx = sz_to_sec[r]`` -- a single coalesced int32 read
    //      (or ``-1`` if state is absent from this sector).
    //   3. ``proj = sz_to_proj[r]`` -- one coalesced complex read.
    // Both tables have length ``C(n_sites, n_up) = dim_full_sz``,
    // smaller than the legacy 16 GiB+ hash for the user's
    // sym+Sz at N=32, n_up=20 workload.
    //
    // For sym-only sectors (n_up_ < 0) the rank-table path is not
    // used: the dim of the full Hilbert ``2^N`` makes the dense
    // approach unworkable at large N. ``sz_to_sec == nullptr`` is the
    // signal to fall back to the legacy hash, which stays populated
    // for those sectors only.
    // -------------------------------------------------------------------
    const DeviceSymmetryHashEntry* hash_table = nullptr;
    std::uint32_t                  hash_mask  = 0;
    const std::int32_t*            sz_to_sec  = nullptr;
    const cuDoubleComplex*         sz_to_proj = nullptr;
    int                            n_sites    = 0;
    int                            n_up       = -1;  // -1 -> sym-only, use hash

    __host__ __device__ inline std::uint64_t dim() const noexcept {
        return dim_;
    }
    __device__ inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        // Orbit representative: first element of orbit `idx`.
        const std::uint32_t off = orbit_offsets[idx];
        return orbit_elements[off];
    }
    __device__ inline std::uint64_t index_of(std::uint64_t state) const noexcept {
        if (sz_to_sec != nullptr) {
            if (__popcll(state) != n_up) return kDeviceNotFound;
            const std::int64_t r =
                ed::gpu::combinadic::rank_state(state, n_sites, n_up);
            const std::int32_t k = sz_to_sec[r];
            return (k < 0) ? kDeviceNotFound : static_cast<std::uint64_t>(k);
        }
        std::uint64_t h = (state * 11400714819323198485ULL) & hash_mask;
        for (;;) {
            const DeviceSymmetryHashEntry e = hash_table[h];
            if (e.key == static_cast<std::uint64_t>(-1)) return kDeviceNotFound;
            if (e.key == state) return static_cast<std::uint64_t>(e.value);
            h = (h + 1) & hash_mask;
        }
    }
    // Look up both the basis index AND the pre-baked projection in one hash probe.
    // Returns kDeviceNotFound for the index if the state is not in the basis.
    __device__ inline std::uint64_t
    index_and_projection(std::uint64_t state, cuDoubleComplex& proj_out) const noexcept {
        if (sz_to_sec != nullptr) {
            if (__popcll(state) != n_up) return kDeviceNotFound;
            const std::int64_t r =
                ed::gpu::combinadic::rank_state(state, n_sites, n_up);
            const std::int32_t k = sz_to_sec[r];
            if (k < 0) return kDeviceNotFound;
            proj_out = sz_to_proj[r];
            return static_cast<std::uint64_t>(k);
        }
        std::uint64_t h = (state * 11400714819323198485ULL) & hash_mask;
        for (;;) {
            const DeviceSymmetryHashEntry e = hash_table[h];
            if (e.key == static_cast<std::uint64_t>(-1)) return kDeviceNotFound;
            if (e.key == state) {
                proj_out = e.projection;
                return static_cast<std::uint64_t>(e.value);
            }
            h = (h + 1) & hash_mask;
        }
    }

    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = true;
    static constexpr bool has_coeff_modifier = true;
    static constexpr bool is_distributed     = false;

    __device__ inline bool is_local(std::uint64_t /*g*/) const noexcept { return true; }
    __device__ inline std::uint64_t local_offset() const noexcept { return 0; }
};

// ===========================================================================
// 4. DeviceFixedSzSymmetryBasisPolicy
//
// Identical layout to DeviceSymmetryBasisPolicy -- the only difference
// from cell 3B (Symm) to cell 4B (Sz+Symm) is which orbit
// representatives populate ``orbit_elements`` (the Sz+Symm version
// filters by popcount at host construction). The kernel ABI is the same.
// ===========================================================================
using DeviceFixedSzSymmetryBasisPolicy = DeviceSymmetryBasisPolicy;

// ===========================================================================
// 5. DeviceRepSymmetryBasisPolicy  -- on-the-fly representative SpMV
//
// "On-the-fly representative SpMV for streaming symmetry" plan (Jun 2026).
//
// The orbit-CSR policies above (cells 3B / 4B) materialise, per orbit
// representative, ALL |G| computational images and their character
// coefficients -- a structure whose total length equals the FULL fixed-Sz
// dimension C(N, n_up). At N=32, n_up=16 that is ~601M entries (~4.8 GiB of
// indices + ~9.6 GiB of coefficients) plus a ~9.6 GiB ``sz_to_proj`` reverse
// table. In lazy mode that is streamed across PCIe every SpMV; even resident
// it defeats the memory win of symmetry.
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
    // Fix (GPU, mirrors CPU Fix 1): precompute all |G| permuted images into a
    // thread-local array (CUDA local memory, L1-cached for sequential access).
    // The representative (min image) and the character accumulation both read
    // from ``images[]`` — halving apply_perm call count vs the prior two-pass
    // approach.  For large N where apply_perm costs N scalar iterations, the
    // savings dominate over the local-memory latency.
    //
    // kMaxGGpu = 64 covers 4×4 lattices with C4v symmetry (|G|=64); groups
    // larger than 64 fall back to the original two-pass algorithm at runtime
    // so correctness is never sacrificed.
    __device__ inline std::uint64_t
    index_and_projection(std::uint64_t state, cuDoubleComplex& proj_out) const noexcept {
        if (n_up >= 0 && __popcll(state) != n_up) return kDeviceNotFound;

        // kMaxGGpu = 256 covers all realistic lattice automorphism groups
        // (full D4h on 4×8 N=32 lattice: |G|=256; D6h on hexagonal: |G|=12×6=72).
        // At N=32, apply_perm costs N~32 cycles; precomputing images[] saves the
        // second pass (~N×|G| instructions). images[256] goes to CUDA local memory
        // (L1-cached for sequential g=0..G-1 access); L1 hit ~1 cycle vs
        // apply_perm ~N cycles — unambiguous win for N≥8.
        static constexpr int kMaxGGpu = 256;

        if (group_size <= kMaxGGpu) {
            // Optimized path: one scan to compute images and find the min.
            std::uint64_t images[kMaxGGpu];
            std::uint64_t rb = state;
            for (int g = 0; g < group_size; ++g) {
                images[g] = apply_perm(state, g);
                if (images[g] < rb) rb = images[g];
            }
            const std::int32_t k = index_of_rep_dev(rb);
            if (k < 0) return kDeviceNotFound;
            // Accumulate from precomputed images — no more apply_perm here.
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            for (int h = 0; h < group_size; ++h) {
                if (images[h] == rb) {
                    const cuDoubleComplex c = characters[h];
                    acc.x += cuCreal(c);   // conj: +real
                    acc.y -= cuCimag(c);   //       -imag
                }
            }
            const double s = inv_norms[k];
            proj_out = make_cuDoubleComplex(acc.x * s, acc.y * s);
            return static_cast<std::uint64_t>(k);
        }

        // Fallback for |G| > kMaxGGpu: original two-pass approach.
        std::uint64_t rb = state;
        for (int g = 1; g < group_size; ++g) {
            const std::uint64_t img = apply_perm(state, g);
            if (img < rb) rb = img;
        }
        const std::int32_t k = index_of_rep_dev(rb);
        if (k < 0) return kDeviceNotFound;
        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
        for (int h = 0; h < group_size; ++h) {
            if (apply_perm(state, h) == rb) {
                const cuDoubleComplex c = characters[h];
                acc.x += cuCreal(c);
                acc.y -= cuCimag(c);
            }
        }
        const double s = inv_norms[k];
        proj_out = make_cuDoubleComplex(acc.x * s, acc.y * s);
        return static_cast<std::uint64_t>(k);
    }

    // Compile-time traits: the rep-symmetry kernel applies H to the single
    // representative (no orbit walk) but still leaves the basis on
    // off-diagonal terms and needs the per-emit projection phase.
    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = false;
    static constexpr bool has_coeff_modifier = true;
    static constexpr bool is_distributed     = false;

    __device__ inline bool is_local(std::uint64_t /*g*/) const noexcept { return true; }
    __device__ inline std::uint64_t local_offset() const noexcept { return 0; }
};

// ---------------------------------------------------------------------------
// DeviceSymmetryBasisPolicyHolder
//
// RAII owner of the device-resident backing a ``DeviceSymmetryBasisPolicy``
// view points at: the orbit CSR (offsets / elements / coefficients /
// inv_norms) plus the (state s' -> basis_idx k, pre-baked projection) hash
// table. ``build_from_orbits()`` takes the plain host orbit CSR (the same
// data the host ``SymmetryBasisPolicy`` exposes via
// ``sector->basis_states[k].orbit_{elements,coefficients}`` + ``norm``),
// pre-bakes ``conj(alpha) * group_norm / norm_k`` into the hash exactly as
// ``streaming_symmetry_gpu_mirror.cu``'s legacy hash path does, and uploads
// everything. This keeps the header free of the ``::SymmetrySector``
// dependency: the CUDA-symmetry factory extracts the raw CSR from the host
// policy and hands it here.
//
// Only the legacy open-addressing hash path is built (no dense rank table);
// it is sector-agnostic (works for sym-only and Sz+sym alike) and is the
// correct equivalence reference for the host ``coeff_modifier`` weighting.
// ---------------------------------------------------------------------------
class DeviceSymmetryBasisPolicyHolder {
public:
    DeviceSymmetryBasisPolicyHolder() = default;
    ~DeviceSymmetryBasisPolicyHolder() { reset(); }
    DeviceSymmetryBasisPolicyHolder(const DeviceSymmetryBasisPolicyHolder&) = delete;
    DeviceSymmetryBasisPolicyHolder& operator=(const DeviceSymmetryBasisPolicyHolder&) = delete;
    DeviceSymmetryBasisPolicyHolder(DeviceSymmetryBasisPolicyHolder&& o) noexcept { steal(o); }
    DeviceSymmetryBasisPolicyHolder& operator=(DeviceSymmetryBasisPolicyHolder&& o) noexcept {
        if (this != &o) { reset(); steal(o); }
        return *this;
    }

    // orbit_offsets : length dim+1, CSR row offsets into the element arrays.
    // orbit_elements/orbit_coefficients : length orbit_offsets.back().
    // orbit_norms : length dim, the raw representative norms (norm_k).
    // group_norm : 1.0 / |G|.
    void build_from_orbits(const std::vector<std::uint32_t>&          orbit_offsets,
                           const std::vector<std::uint64_t>&          orbit_elements,
                           const std::vector<std::complex<double>>&   orbit_coefficients,
                           const std::vector<double>&                 orbit_norms,
                           double                                     group_norm) {
        reset();
        if (orbit_offsets.empty()) return;
        dim_        = orbit_norms.size();
        group_norm_ = group_norm;
        if (dim_ == 0) return;

        const std::size_t total = orbit_elements.size();

        // Host CSR staging (coefficients -> cuDoubleComplex, norms -> 1/norm).
        std::vector<cuDoubleComplex> h_coefs(total);
        for (std::size_t i = 0; i < total; ++i) {
            h_coefs[i] = make_cuDoubleComplex(orbit_coefficients[i].real(),
                                              orbit_coefficients[i].imag());
        }
        std::vector<double> h_inv_norms(dim_);
        for (std::size_t k = 0; k < dim_; ++k) {
            h_inv_norms[k] = (orbit_norms[k] > 0.0) ? (1.0 / orbit_norms[k]) : 0.0;
        }

        // Open-addressing projection hash (power-of-two, ~2x occupancy).
        std::uint64_t table_size = 16;
        while (table_size < total * 2ULL) table_size <<= 1;
        hash_mask_ = static_cast<std::uint32_t>(table_size - 1);
        constexpr std::uint64_t kEmpty = static_cast<std::uint64_t>(-1);
        std::vector<DeviceSymmetryHashEntry> h_hash(
            table_size, DeviceSymmetryHashEntry{kEmpty, 0u, 0u,
                                                make_cuDoubleComplex(0.0, 0.0)});
        for (std::size_t k = 0; k < dim_; ++k) {
            const double scale = group_norm_ * h_inv_norms[k];
            const std::uint32_t lo = orbit_offsets[k];
            const std::uint32_t hi = orbit_offsets[k + 1];
            for (std::uint32_t j = lo; j < hi; ++j) {
                const std::uint64_t s_prime = orbit_elements[j];
                const std::complex<double> alpha = orbit_coefficients[j];
                const double proj_re =  alpha.real() * scale;   // conj(alpha) real part
                const double proj_im = -alpha.imag() * scale;   // conj(alpha) imag part
                std::uint64_t h = (s_prime * 11400714819323198485ULL) & hash_mask_;
                for (;;) {
                    if (h_hash[h].key == kEmpty) {
                        h_hash[h].key        = s_prime;
                        h_hash[h].value      = static_cast<std::uint32_t>(k);
                        h_hash[h].projection = make_cuDoubleComplex(proj_re, proj_im);
                        break;
                    }
                    if (h_hash[h].key == s_prime) break;  // orbits disjoint; defensive
                    h = (h + 1) & hash_mask_;
                }
            }
        }

        check_(cudaMalloc(&d_offsets_, orbit_offsets.size() * sizeof(std::uint32_t)),
               "cudaMalloc(orbit_offsets)");
        check_(cudaMemcpy(d_offsets_, orbit_offsets.data(),
                          orbit_offsets.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "cudaMemcpy(orbit_offsets)");
        if (total > 0) {
            check_(cudaMalloc(&d_elements_, total * sizeof(std::uint64_t)),
                   "cudaMalloc(orbit_elements)");
            check_(cudaMemcpy(d_elements_, orbit_elements.data(),
                              total * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice), "cudaMemcpy(orbit_elements)");
            check_(cudaMalloc(&d_coefs_, total * sizeof(cuDoubleComplex)),
                   "cudaMalloc(orbit_coefficients)");
            check_(cudaMemcpy(d_coefs_, h_coefs.data(),
                              total * sizeof(cuDoubleComplex),
                              cudaMemcpyHostToDevice), "cudaMemcpy(orbit_coefficients)");
        }
        check_(cudaMalloc(&d_inv_norms_, dim_ * sizeof(double)),
               "cudaMalloc(orbit_inv_norms)");
        check_(cudaMemcpy(d_inv_norms_, h_inv_norms.data(),
                          dim_ * sizeof(double), cudaMemcpyHostToDevice),
               "cudaMemcpy(orbit_inv_norms)");
        check_(cudaMalloc(&d_hash_, table_size * sizeof(DeviceSymmetryHashEntry)),
               "cudaMalloc(hash_table)");
        check_(cudaMemcpy(d_hash_, h_hash.data(),
                          table_size * sizeof(DeviceSymmetryHashEntry),
                          cudaMemcpyHostToDevice), "cudaMemcpy(hash_table)");
    }

    [[nodiscard]] DeviceSymmetryBasisPolicy view() const noexcept {
        DeviceSymmetryBasisPolicy p;
        p.orbit_elements     = d_elements_;
        p.orbit_coefficients = d_coefs_;
        p.orbit_offsets      = d_offsets_;
        p.orbit_inv_norms    = d_inv_norms_;
        p.dim_               = dim_;
        p.group_norm         = group_norm_;
        p.hash_table         = d_hash_;
        p.hash_mask          = hash_mask_;
        p.sz_to_sec          = nullptr;  // legacy hash path
        p.sz_to_proj         = nullptr;
        p.n_sites            = 0;
        p.n_up               = -1;
        return p;
    }

    [[nodiscard]] std::uint64_t dim() const noexcept { return dim_; }

private:
    static void check_(cudaError_t err, const char* what) {
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DeviceSymmetryBasisPolicyHolder: ") + what +
                " failed: " + cudaGetErrorString(err));
        }
    }
    void reset() noexcept {
        if (d_elements_)  { cudaFree(d_elements_);  d_elements_  = nullptr; }
        if (d_coefs_)     { cudaFree(d_coefs_);     d_coefs_     = nullptr; }
        if (d_offsets_)   { cudaFree(d_offsets_);   d_offsets_   = nullptr; }
        if (d_inv_norms_) { cudaFree(d_inv_norms_); d_inv_norms_ = nullptr; }
        if (d_hash_)      { cudaFree(d_hash_);      d_hash_      = nullptr; }
        dim_ = 0; hash_mask_ = 0; group_norm_ = 1.0;
    }
    void steal(DeviceSymmetryBasisPolicyHolder& o) noexcept {
        d_elements_  = o.d_elements_;  o.d_elements_  = nullptr;
        d_coefs_     = o.d_coefs_;     o.d_coefs_     = nullptr;
        d_offsets_   = o.d_offsets_;   o.d_offsets_   = nullptr;
        d_inv_norms_ = o.d_inv_norms_; o.d_inv_norms_ = nullptr;
        d_hash_      = o.d_hash_;      o.d_hash_      = nullptr;
        dim_         = o.dim_;         o.dim_         = 0;
        hash_mask_   = o.hash_mask_;   o.hash_mask_   = 0;
        group_norm_  = o.group_norm_;  o.group_norm_  = 1.0;
    }

    std::uint64_t*          d_elements_  = nullptr;
    cuDoubleComplex*        d_coefs_     = nullptr;
    std::uint32_t*          d_offsets_   = nullptr;
    double*                 d_inv_norms_ = nullptr;
    DeviceSymmetryHashEntry* d_hash_     = nullptr;
    std::uint64_t           dim_         = 0;
    std::uint32_t           hash_mask_   = 0;
    double                  group_norm_  = 1.0;
};

}  // namespace ed::matvec::basis

#endif  // WITH_CUDA
