// =============================================================================
// tests/unit/test_mem_guard.cpp
//
// The available-memory probe must never report more than the job may allocate:
// under SLURM the cgroup limit, not the node's MemAvailable, is the ceiling.
// =============================================================================
#include "common/catch2_harness.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <ed/core/footprint.h>
#include <ed/core/memory.h>

TEST_CASE("mem_guard: available RAM is the tighter of node and cgroup", "[mem_guard]") {
    // The three probes read live counters at different instants while other tests
    // allocate and free (ctest runs in parallel), so bracket the combined probe with
    // samples on both sides and allow a slack. The slack is far below the gap the
    // test exists to catch: a job-limited cgroup against a node's free memory.
    const std::uint64_t node0 = ed::core::node_available_ram_bytes();
    const std::uint64_t group0 = ed::core::cgroup_available_ram_bytes();
    const std::uint64_t avail = ed::core::available_ram_bytes();
    const std::uint64_t node1 = ed::core::node_available_ram_bytes();
    const std::uint64_t group1 = ed::core::cgroup_available_ram_bytes();
    const std::uint64_t slack = std::uint64_t{512} << 20;
    INFO("node " << (node0 >> 20) << "/" << (node1 >> 20) << " MiB, cgroup " << (group0 >> 20) << "/" << (group1 >> 20)
                 << " MiB, available " << (avail >> 20) << " MiB");
    REQUIRE(node0 > 0);                      // every Linux host reports MemAvailable
    REQUIRE(avail > 0);
    REQUIRE(avail <= std::max(node0, node1) + slack);
    if (group0 > 0 && group1 > 0) REQUIRE(avail <= std::max(group0, group1) + slack);
}

TEST_CASE("mem_guard: a working set larger than the ceiling is refused", "[mem_guard]") {
    const std::uint64_t avail = ed::core::available_ram_bytes();
    REQUIRE_THROWS_AS(ed::core::guard_working_set(avail * 2, "test"), ed::ResourceLimit);
    REQUIRE_NOTHROW(ed::core::guard_working_set(avail / 100, "test"));
}

TEST_CASE("mem_guard: ED_MEM_GUARD_OFF stands every guard down", "[mem_guard]") {
    const char* old = std::getenv("ED_MEM_GUARD_OFF");
    const std::string saved = old ? old : "";
    setenv("ED_MEM_GUARD_OFF", "1", 1);
    REQUIRE(ed::core::mem_guard_off());
    REQUIRE_NOTHROW(ed::core::guard_working_set(ed::core::available_ram_bytes() * 2, "test"));
    if (old)
        setenv("ED_MEM_GUARD_OFF", saved.c_str(), 1);
    else
        unsetenv("ED_MEM_GUARD_OFF");
}

TEST_CASE("footprint: the paths count the vectors the kernels hold", "[mem_guard][footprint]") {
    using ed::core::Path;
    ed::core::Shape s;
    s.dim = 1000;
    s.krylov = 100;
    const std::uint64_t V = 16 * 1000;
    // FTLM without a kept basis does not grow with the Krylov depth (L5-memory-04).
    REQUIRE(ed::core::footprint(Path::FtlmSample, s).host == 5 * V);
    REQUIRE(ed::core::footprint(Path::FtlmSampleKept, s).host == 106 * V);
    s.tower = true;
    REQUIRE(ed::core::footprint(Path::FtlmSample, s).host == 6 * V);
    s.tower = false;
    // On the device: per sample in lockstep; the kept basis has a staging copy.
    s.device = true;
    s.width = 8;
    REQUIRE(ed::core::footprint(Path::FtlmSample, s).device == 32 * V);
    REQUIRE(ed::core::footprint(Path::FtlmSample, s).host == 8 * V);
    REQUIRE(ed::core::footprint(Path::FtlmSampleKept, s).device == 8 * 205 * V);
    REQUIRE(ed::core::footprint(Path::Mtpq, s).device == 16 * V);
    s.width = 1;
    REQUIRE(ed::core::footprint(Path::Mtpq, s).device == 5 * V);
    s.k = 4;
    s.krylov = 68;
    // thick restart: k = 4 keeps p = 4 + max(2, 8) = 12 Ritz vectors
    REQUIRE(ed::core::footprint(Path::KrylovSchur, s).device == (2 * 68 + 12 + 8 + 8) * V);
    s.device = false;
    REQUIRE(ed::core::footprint(Path::KrylovSchur, s).host == (68 + 12 + 4 + 3) * V);
    REQUIRE(ed::core::footprint(Path::DenseVectors, s).host == 32ull * 1000 * 1000);
    REQUIRE(ed::core::footprint(Path::Multiplet, s).host == 6 * V + 8 * 1000);
    s.dim = std::uint64_t{1} << 40;   // saturates instead of wrapping
    REQUIRE(ed::core::footprint(Path::DenseVectors, s).host > (std::uint64_t{1} << 63));
}

TEST_CASE("footprint: finite-temperature dynamics counts both sectors", "[mem_guard][footprint]") {
    using ed::core::Path;
    ed::core::Shape s;
    s.dim = 1000;
    s.dim_target = 3000;
    s.krylov = 50;
    const std::uint64_t Vs = 16 * 1000, Vt = 16 * 3000;
    REQUIRE(ed::core::footprint(Path::DynamicsFtlm, s).host == 55 * Vs + 55 * Vt);
    s.device = true;
    s.width = 2;
    REQUIRE(ed::core::footprint(Path::DynamicsFtlm, s).device == 2 * (54 * Vs + 55 * Vt + 50 * Vt));
    REQUIRE(ed::core::footprint(Path::DynamicsFtlm, s).host == 2 * Vs);
}
