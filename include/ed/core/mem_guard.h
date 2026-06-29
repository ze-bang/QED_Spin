// =============================================================================
// include/ed/core/mem_guard.h
//
// Leaf-level memory guard: a cheap "estimated working set vs available RAM"
// check to throw a CLEAN error BEFORE a large allocation, instead of letting
// the run OOM-crash mid-flight. This is NOT a planner / cost model -- it is one
// estimate and one comparison. It replaces only the "completion guarantee"
// safety net that the execution planner used to provide.
//
// Override with ED_MEM_GUARD_OFF=1 (dispatch anyway, accepting the OOM risk).
// If available RAM cannot be determined, the guard is a no-op (never blocks).
// =============================================================================
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace ed::core {

/// Best-effort available RAM in bytes (0 = unknown). Prefers /proc/meminfo
/// MemAvailable (counts reclaimable cache); falls back to sysconf.
[[nodiscard]] inline std::uint64_t available_ram_bytes() noexcept {
    std::ifstream mi("/proc/meminfo");
    if (mi) {
        std::string key;
        while (mi >> key) {
            if (key == "MemAvailable:") {
                std::uint64_t kb = 0;
                if (mi >> kb) return kb * 1024ull;
                break;
            }
            mi.ignore(1 << 20, '\n');
        }
    }
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long psize = sysconf(_SC_PAGESIZE);
    if (pages > 0 && psize > 0)
        return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(psize);
#endif
    return 0;
}

/// Throw a clean error if `est_bytes` would not fit in ~90% of available RAM.
/// No-op when ED_MEM_GUARD_OFF is set or RAM is unknown.
inline void guard_working_set(std::uint64_t est_bytes, const char* what) {
    if (std::getenv("ED_MEM_GUARD_OFF")) return;
    const std::uint64_t avail = available_ram_bytes();
    if (avail == 0) return;  // can't tell -> don't block
    const double budget = 0.90 * static_cast<double>(avail);
    if (static_cast<double>(est_bytes) > budget) {
        const auto GiB = [](double b) {
            return std::to_string(static_cast<std::uint64_t>(b / (1024.0 * 1024.0 * 1024.0)));
        };
        throw std::runtime_error(
            std::string(what) + ": estimated working set ~" + GiB(est_bytes) +
            " GiB exceeds ~" + GiB(budget) + " GiB available RAM. Reduce the "
            "problem (sz / symmetry / fewer samples / smaller Krylov dim), give "
            "it more memory or MPI ranks, or set ED_MEM_GUARD_OFF=1 to override.");
    }
}

}  // namespace ed::core
