#pragma once
// =============================================================================
// include/ed/matvec/memory_space.h
//
// MemorySpace: tag identifying where the bytes that back a Vector / matvec
// input-output buffer live. This is the *only* property a solver needs in
// order to decide which Backend (CPU / CUDA / MPI / MPI+CUDA) to use to
// drive the surrounding linear algebra (axpy, dot, norm, scale, copy).
//
// The MatVec layer treats this as an opaque tag --- it does not own the
// runtime (no CUDA context, no MPI communicator) here, that lives on the
// Backend object. The split keeps memory_space.h header-only and free of
// optional dependencies (no CUDA / MPI includes required to use Vector).
//
// Convention: each concrete MatVecOperator subclass declares a single
// MemorySpace from which it expects its `in` buffer to come and into which
// it will write `out`. Solvers compose this with a Backend of the matching
// space; mismatches throw at solver-construction time. This is exactly the
// pattern used by Trilinos Tpetra / Kokkos: tag the data, dispatch the
// runtime.
//
// Phase 1 of the matvec-unification revamp.
// =============================================================================

#include <cstdint>
#include <string_view>

namespace ed::matvec {

enum class MemorySpace : std::uint8_t {
    // Bytes live in host (CPU) RAM, accessible by any thread on this rank.
    // Covers single-node + multi-threaded OpenMP. The default.
    Host = 0,

    // Bytes live in CUDA device memory (cudaMalloc'd) on the current device.
    // The owning Vector knows the device id; the Backend supplies the
    // CUDA stream + cuBLAS / cuSPARSE handles for vector primitives.
    CudaDevice = 1,

    // Bytes live in host RAM and the full global vector is sharded across
    // MPI ranks (1-D row slab decomposition). Each rank owns a contiguous
    // chunk; reductions and halo exchanges run through the MPI backend.
    DistributedHost = 2,

    // Same as DistributedHost but the per-rank slab lives in CUDA device
    // memory. Reductions go through NCCL (preferred) or staged through
    // host MPI.
    DistributedCudaDevice = 3,
};

constexpr std::string_view to_string(MemorySpace s) noexcept {
    switch (s) {
        case MemorySpace::Host:                  return "Host";
        case MemorySpace::CudaDevice:            return "CudaDevice";
        case MemorySpace::DistributedHost:       return "DistributedHost";
        case MemorySpace::DistributedCudaDevice: return "DistributedCudaDevice";
    }
    return "Unknown";
}

constexpr bool is_distributed(MemorySpace s) noexcept {
    return s == MemorySpace::DistributedHost
        || s == MemorySpace::DistributedCudaDevice;
}

constexpr bool is_device(MemorySpace s) noexcept {
    return s == MemorySpace::CudaDevice
        || s == MemorySpace::DistributedCudaDevice;
}

constexpr bool is_host(MemorySpace s) noexcept {
    return s == MemorySpace::Host
        || s == MemorySpace::DistributedHost;
}

} // namespace ed::matvec
