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
#include <vector>

#include <ed/matvec/basis_policy.h>

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
    // marked by key == UINT64_MAX. Same layout as
    // ``GPUFixedSzOperator::GPUStateLookupEntry`` so a Phase 1b port
    // can reuse the existing build helper.
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

// ===========================================================================
// 3. DeviceSymmetryBasisPolicy
//
// Holds device-resident orbit CSR data + a pre-baked coefficient modifier
// hash. Mirrors the optimization the existing GPUSymmetrizedOperator
// already does: the per-emit phase
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

    // Reverse lookup: state -> (basis_idx, projection) via an
    // open-addressing hash.
    const DeviceSymmetryHashEntry* hash_table = nullptr;
    std::uint32_t                  hash_mask  = 0;

    __host__ __device__ inline std::uint64_t dim() const noexcept {
        return dim_;
    }
    __device__ inline std::uint64_t state_of(std::uint64_t idx) const noexcept {
        // Orbit representative: first element of orbit `idx`.
        const std::uint32_t off = orbit_offsets[idx];
        return orbit_elements[off];
    }
    __device__ inline std::uint64_t index_of(std::uint64_t state) const noexcept {
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

}  // namespace ed::matvec::basis

#endif  // WITH_CUDA
