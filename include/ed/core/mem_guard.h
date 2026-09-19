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

#include <ed/config/env_registry.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace ed::core {

/// Node-wide available RAM in bytes (0 = unknown): /proc/meminfo MemAvailable
/// (counts reclaimable cache), else sysconf.
[[nodiscard]] inline std::uint64_t node_available_ram_bytes() noexcept {
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

/// Headroom left under the tightest cgroup-v2 memory limit that contains this
/// process (memory.max - memory.current, walking from our cgroup up to the root);
/// 0 = no limit found (not in a limited cgroup, or cgroup v1 / unreadable).
///
/// Under a batch scheduler this is the number that matters: a SLURM job gets a
/// cgroup of --mem bytes on a node whose MemAvailable may be ten times larger, and
/// sizing a Krylov basis from MemAvailable is how a job gets OOM-killed.
[[nodiscard]] inline std::uint64_t cgroup_available_ram_bytes() noexcept {
    std::ifstream cg("/proc/self/cgroup");
    std::string line, path;
    while (cg && std::getline(cg, line))
        if (line.rfind("0::", 0) == 0) { path = line.substr(3); break; }
    if (path.empty()) return 0;
    auto read_u64 = [](const std::string& file, std::uint64_t& out) -> bool {
        std::ifstream f(file);
        std::string s;
        if (!(f >> s) || s == "max") return false;
        try { out = std::stoull(s); } catch (...) { return false; }
        return true;
    };
    std::uint64_t best = 0;
    bool found = false;
    for (std::string p = path;; ) {
        const std::string dir = "/sys/fs/cgroup" + (p == "/" ? std::string() : p);
        std::uint64_t lim = 0, cur = 0;
        if (read_u64(dir + "/memory.max", lim) && read_u64(dir + "/memory.current", cur)) {
            const std::uint64_t room = lim > cur ? lim - cur : 0;
            if (!found || room < best) best = room;
            found = true;
        }
        if (p.empty() || p == "/") break;
        const auto slash = p.find_last_of('/');
        p = (slash == 0 || slash == std::string::npos) ? std::string("/") : p.substr(0, slash);
    }
    return found ? std::max<std::uint64_t>(best, 1) : 0;
}

/// Best-effort RAM this process may still allocate, in bytes (0 = unknown): the
/// smaller of the node's MemAvailable and the headroom under our cgroup limit.
[[nodiscard]] inline std::uint64_t available_ram_bytes() noexcept {
    const std::uint64_t node = node_available_ram_bytes();
    const std::uint64_t job  = cgroup_available_ram_bytes();
    if (job == 0) return node;
    if (node == 0) return job;
    return node < job ? node : job;
}

/// Throw a clean error if `est_bytes` would not fit in ~90% of available RAM.
/// No-op when ED_MEM_GUARD_OFF is set or RAM is unknown.
inline void guard_working_set(std::uint64_t est_bytes, const char* what) {
    if (ed::env::flag("ED_MEM_GUARD_OFF", false)) return;
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
