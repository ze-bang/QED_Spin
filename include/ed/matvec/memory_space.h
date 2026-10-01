#pragma once
// =============================================================================
// include/ed/matvec/memory_space.h
//
// MemorySpace: tag identifying where the bytes that back a vector / matvec
// input-output buffer live. This is the *only* property a solver needs in
// order to decide which Backend (CPU / CUDA) to use to drive the surrounding
// linear algebra (axpy, dot, norm, scale, copy).
//
// The MatVec layer treats this as an opaque tag --- it does not own the
// runtime (no CUDA context) here, that lives on the Backend object. The split
// keeps memory_space.h header-only and free of optional dependencies (no CUDA
// includes required).
//
// Each Backend declares the MemorySpace its vectors live in; operators apply
// on host buffers and bind a device apply through bind_cuda().
// =============================================================================

#include <cstdint>
#include <string_view>

namespace ed::matvec {

enum class MemorySpace : std::uint8_t {
    // Bytes live in host (CPU) RAM, accessible by any thread.
    // Covers single-node + multi-threaded OpenMP. The default.
    Host = 0,

    // Bytes live in CUDA device memory (cudaMalloc'd) on the current device.
    // The Backend supplies the CUDA stream + cuBLAS handles for vector
    // primitives.
    CudaDevice = 1,
};

constexpr std::string_view to_string(MemorySpace s) noexcept {
    switch (s) {
        case MemorySpace::Host:       return "Host";
        case MemorySpace::CudaDevice: return "CudaDevice";
    }
    return "Unknown";
}

constexpr bool is_device(MemorySpace s) noexcept {
    return s == MemorySpace::CudaDevice;
}

constexpr bool is_host(MemorySpace s) noexcept {
    return s == MemorySpace::Host;
}

} // namespace ed::matvec
