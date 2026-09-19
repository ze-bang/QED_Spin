// =============================================================================
// tests/unit/test_mem_guard.cpp
//
// The available-memory probe must never report more than the job may allocate:
// under SLURM the cgroup limit, not the node's MemAvailable, is the ceiling.
// =============================================================================
#include "common/catch2_harness.h"

#include <cstdint>

#include <ed/core/mem_guard.h>

TEST_CASE("mem_guard: available RAM is the tighter of node and cgroup", "[mem_guard]") {
    const std::uint64_t node  = ed::core::node_available_ram_bytes();
    const std::uint64_t group = ed::core::cgroup_available_ram_bytes();
    const std::uint64_t avail = ed::core::available_ram_bytes();
    REQUIRE(node > 0);                       // every Linux host reports MemAvailable
    REQUIRE(avail > 0);
    REQUIRE(avail <= node);
    if (group > 0) REQUIRE(avail <= group);
    INFO("node " << (node >> 20) << " MiB, cgroup " << (group >> 20) << " MiB");
}

TEST_CASE("mem_guard: a working set larger than the ceiling is refused", "[mem_guard]") {
    const std::uint64_t avail = ed::core::available_ram_bytes();
    REQUIRE_THROWS(ed::core::guard_working_set(avail * 2, "test"));
    REQUIRE_NOTHROW(ed::core::guard_working_set(avail / 100, "test"));
}
