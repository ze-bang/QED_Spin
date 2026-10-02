#pragma once
// =============================================================================
// include/ed/core/device.h
//
// The device vocabulary: where a verb may run (Device), where one block's solve
// runs (Lane), what the block is solved for (Task), and THE 'auto' table that
// place() (select_backend.h) reads. No CUDA headers.
// =============================================================================

#include <cstdint>
#include <functional>
#include <limits>
#include <string>

namespace ed {

/// Where a verb's blocks may run. Cpu: the host only, and CUDA is never initialised.
/// Gpu: every Krylov solve runs on the device, or the verb raises naming the block.
/// Auto: the device for blocks above the 'auto' floor of their task (auto_row).
enum class Device { Cpu, Gpu, Auto };

/// Where one block's solve runs: the four Placement counters.
enum class Lane : std::uint8_t { HostDense, HostKrylov, DeviceDense, DeviceKrylov };

[[nodiscard]] constexpr bool on_device(Lane l) noexcept {
    return l == Lane::DeviceDense || l == Lane::DeviceKrylov;
}
[[nodiscard]] constexpr bool is_dense(Lane l) noexcept {
    return l == Lane::HostDense || l == Lane::DeviceDense;
}

/// What a block is solved for.
enum class Task : std::uint8_t {
    Eigs,           ///< lowest levels of a block, and its prune estimate
    Sampled,        ///< FTLM / mTPQ thermal sampling
    Oftlm,          ///< FTLM with exact low states: host only
    DenseBatch,     ///< a whole spectrum, batched (exact thermal, spectrum)
    DynamicsCf,     ///< T = 0 continued fraction of one target sector
    DynamicsFtlm,   ///< T > 0 FTLM dynamics of one source sector
};

/// One row of the 'auto' table: a block goes to the device at dim >= floor, when (fit) its
/// device working set (BlockRequest::device_bytes) fits in free device memory.
struct AutoRow {
    std::uint64_t floor;
    bool          fit;
};

/// THE 'auto' table.
[[nodiscard]] constexpr AutoRow auto_row(Task t) noexcept {
    switch (t) {
        case Task::Eigs:         return {std::uint64_t{1} << 14, true};
        case Task::Sampled:      return {std::uint64_t{1} << 14, true};
        case Task::Oftlm:        return {std::numeric_limits<std::uint64_t>::max(), false};
        case Task::DenseBatch:   return {0, false};
        case Task::DynamicsCf:   return {std::uint64_t{1} << 14, false};
        case Task::DynamicsFtlm: return {std::uint64_t{1} << 16, false};
    }
    return {std::numeric_limits<std::uint64_t>::max(), false};
}

/// A host-placed RepSectorMatVec may still apply H with the device gather on host vectors at
/// dim >= this ('auto' and 'gpu' only; ED_SYM_LG_GPU=1 drops the floor, =0 vetoes the gather).
inline constexpr std::uint64_t kHostGatherFloor = std::uint64_t{1} << 20;

/// Host blocks below this dimension run concurrently, one thread each (the deferred thermal
/// blocks and the small dynamics sources).
inline constexpr std::uint64_t kHostPoolMaxDim = std::uint64_t{1} << 16;

/// TRANSITIONAL: a device-bound Eigs block with dim <= this, or 2 want >= dim, is solved densely
/// on the host. P6.2's calibrated crossover and P7.4's measured small-block crossover retire it.
inline constexpr std::uint64_t kDeviceDenseMaxDim = 32;

/// One block, as place() sees it.
struct BlockRequest {
    Task          task = Task::Eigs;
    std::uint64_t dim  = 0;
    /// The verb solves this block exactly on the host (eigs: dim <= its dense floor; thermal:
    /// the exact-small fallback).
    bool          dense = false;
    std::uint64_t want  = 1;               ///< Eigs only: levels owed (the transitional rule)
    /// The least device memory the solve needs (ed/core/footprint.h), checked where the task's
    /// row says so; 0: place() estimates it from the task (Eigs: the two-pass GS vector, or
    /// Krylov-Schur at its smallest cycle of want + 8; Sampled: one FTLM sample).
    std::uint64_t device_bytes = 0;
    bool          device_kernel = false;   ///< H and every operator the solve applies bind to a device
    const char*   verb = "";
    /// Built only for a refusal: "the block of star K, irrep I, n_up N (dim D)".
    std::function<std::string()> what;
    const char*   why = "has no device kernel";
};

}  // namespace ed
