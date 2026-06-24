#pragma once
// =============================================================================
// include/ed/planner/system_capabilities.h
//
// Runtime probe of the machine the ED task will run on. This is the C++
// counterpart of Python's `qed.feasibility.probe_host()` and deliberately
// shares its env-override contract (QED_HOST_*) so the two agree on the same
// host.
//
// Used by `ed::planner::ExecutionPlanner` to turn a TaskCostModel into a
// concrete ExecutionPlan (CSR vs matrix-free, device lane, reorth, basis
// strategy) gated on what the machine can actually hold.
//
// Best-effort: every field falls back to a conservative default if its probe
// fails, and records how it was determined in `notes`.
// =============================================================================

#include <cstdint>
#include <string>
#include <vector>

namespace ed::planner {

struct SystemCapabilities {
    /// Total physical RAM (bytes). Falls back to 16 GiB if undetected.
    std::uint64_t ram_total_bytes = std::uint64_t{16} << 30;
    /// Available RAM (bytes) -- MemAvailable when readable, else ram_total.
    std::uint64_t ram_avail_bytes = std::uint64_t{16} << 30;
    /// Logical CPU cores online.
    int           n_cores = 1;
    /// CUDA-visible GPUs (0 if none / no CUDA build).
    int           n_gpus = 0;
    /// Free VRAM (bytes) on the smallest visible GPU (0 if none).
    std::uint64_t vram_avail_bytes = 0;
    /// Total VRAM (bytes) on the smallest visible GPU (0 if none).
    std::uint64_t vram_total_bytes = 0;
    /// MPI world size if MPI is initialized, else best guess (scheduler env).
    int           n_mpi_ranks = 1;

    bool has_cuda_build = false;
    bool has_mpi_build  = false;
    bool has_nccl_build = false;

    std::vector<std::string> notes;

    [[nodiscard]] double ram_total_gb() const noexcept {
        return static_cast<double>(ram_total_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    [[nodiscard]] double ram_avail_gb() const noexcept {
        return static_cast<double>(ram_avail_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
    [[nodiscard]] double vram_avail_gb() const noexcept {
        return static_cast<double>(vram_avail_bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    [[nodiscard]] std::string summary() const;
};

/// Probe the host. Cached on first call; pass `refresh = true` to re-probe.
/// Honors QED_HOST_MEMORY_GB / QED_HOST_GPU_MEMORY_GB / QED_HOST_N_GPUS /
/// QED_HOST_N_MPI_RANKS env overrides (same contract as probe_host()).
[[nodiscard]] SystemCapabilities probe_system(bool refresh = false);

}  // namespace ed::planner
