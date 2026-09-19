// =============================================================================
// tests/unit/test_mem_guard.cpp
//
// The available-memory probe must never report more than the job may allocate:
// under SLURM the cgroup limit, not the node's MemAvailable, is the ceiling.
// =============================================================================
#include "common/catch2_harness.h"

#include <algorithm>
#include <cstdint>

#include <ed/core/mem_guard.h>

TEST_CASE("mem_guard: available RAM is the tighter of node and cgroup", "[mem_guard]") {
    // The three probes read live counters at different instants while other tests
    // allocate and free (ctest runs in parallel), so bracket the combined probe with
    // samples on both sides and allow a slack. The slack is far below the gap the
    // test exists to catch: a job-limited cgroup against a node's free memory.
    const std::uint64_t node0  = ed::core::node_available_ram_bytes();
    const std::uint64_t group0 = ed::core::cgroup_available_ram_bytes();
    const std::uint64_t avail  = ed::core::available_ram_bytes();
    const std::uint64_t node1  = ed::core::node_available_ram_bytes();
    const std::uint64_t group1 = ed::core::cgroup_available_ram_bytes();
    const std::uint64_t slack  = std::uint64_t{512} << 20;
    INFO("node " << (node0 >> 20) << "/" << (node1 >> 20) << " MiB, cgroup "
         << (group0 >> 20) << "/" << (group1 >> 20) << " MiB, available " << (avail >> 20) << " MiB");
    REQUIRE(node0 > 0);                      // every Linux host reports MemAvailable
    REQUIRE(avail > 0);
    REQUIRE(avail <= std::max(node0, node1) + slack);
    if (group0 > 0 && group1 > 0) REQUIRE(avail <= std::max(group0, group1) + slack);
}

TEST_CASE("mem_guard: a working set larger than the ceiling is refused", "[mem_guard]") {
    const std::uint64_t avail = ed::core::available_ram_bytes();
    REQUIRE_THROWS(ed::core::guard_working_set(avail * 2, "test"));
    REQUIRE_NOTHROW(ed::core::guard_working_set(avail / 100, "test"));
}
