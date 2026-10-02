// =============================================================================
// test_sector_csr_budget  (Catch2 v3)
//
// Pins ed::planner::sector_csr_within_budget -- THE single reduced-CSR budget
// decision, shared by the abelian CpuMatVecBackend build sites and the
// little-group engine's RepSectorMatVec, so the two lanes cannot drift apart.
//
// The property that matters is that the knob bounds the TOTAL in-flight CSR
// footprint, not one sector's. The sector-parallel lanes build each sector's
// CSR lazily inside an `omp parallel for` over sectors; a per-sector check
// would let N threads each pass an 8 GiB check independently and allocate
// N x 8 GiB.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/memory.h>
#include <ed/matvec/csr_policy.h>

#include <cmath>
#include <cstdlib>
#include <string>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace {

// Pick (dim, terms_per_row) whose upper-bound estimate is ~= gib gigabytes.
// est = dim * terms * 20 + (dim + 1) * 8
std::uint64_t dim_for_gib(double gib, std::uint64_t terms_per_row) {
    const double bytes = gib * static_cast<double>(1ULL << 30);
    return static_cast<std::uint64_t>(bytes / (terms_per_row * 20.0 + 8.0));
}

struct EnvGuard {
    std::string saved;
    bool had;
    explicit EnvGuard(const char* v) {
        const char* cur = std::getenv("ED_SYM_SECTOR_CSR_BUDGET_GIB");
        had = cur != nullptr;
        if (had) saved = cur;
        if (v) ::setenv("ED_SYM_SECTOR_CSR_BUDGET_GIB", v, 1);
        else   ::unsetenv("ED_SYM_SECTOR_CSR_BUDGET_GIB");
    }
    ~EnvGuard() {
        if (had) ::setenv("ED_SYM_SECTOR_CSR_BUDGET_GIB", saved.c_str(), 1);
        else     ::unsetenv("ED_SYM_SECTOR_CSR_BUDGET_GIB");
    }
};

}  // namespace

TEST_CASE("sector_csr_within_budget: serial lane admits up to the knob",
          "[planner][csr_budget]") {
    EnvGuard g("4");
    constexpr std::uint64_t terms = 8;

    // Comfortably under 4 GiB -> materialize the reduced CSR (the fast tier).
    REQUIRE(ed::planner::sector_csr_within_budget(dim_for_gib(1.0, terms), terms));
    // Over 4 GiB -> decline; the caller degrades to the CSR-free rep walk.
    REQUIRE_FALSE(ed::planner::sector_csr_within_budget(dim_for_gib(9.0, terms), terms));
}

TEST_CASE("sector_csr_within_budget: the knob is an AGGREGATE across concurrent builders",
          "[planner][csr_budget]") {
#ifndef _OPENMP
    SUCCEED("OpenMP not enabled; the concurrent-builder split is a no-op.");
#else
    EnvGuard g("4");
    constexpr std::uint64_t terms = 8;

    // A sector needing 1 GiB fits the 4 GiB knob on its own...
    const std::uint64_t d1 = dim_for_gib(1.0, terms);
    REQUIRE(ed::planner::sector_csr_within_budget(d1, terms));

    // ...but with 8 sectors in flight, 8 x 1 GiB = 8 GiB would blow a 4 GiB
    // budget, so the same sector must now be declined. Emulate the sector-
    // parallel lane: an outer team whose body asks the budget question.
    int admitted = 0;
    const int want = 8;
#  pragma omp parallel num_threads(want) reduction(+ : admitted)
    {
        if (omp_get_team_size(1) == want
            && ed::planner::sector_csr_within_budget(d1, terms))
            admitted++;
    }
    INFO("threads that would have materialized a 1 GiB CSR under a 4 GiB knob: "
         << admitted);
    REQUIRE(admitted == 0);

    // Small enough that even 8-way concurrency stays under the knob (8 x 0.25
    // = 2 GiB < 4): still admitted. The split throttles, it does not veto.
    const std::uint64_t d_small = dim_for_gib(0.25, terms);
    int admitted_small = 0;
#  pragma omp parallel num_threads(want) reduction(+ : admitted_small)
    {
        if (ed::planner::sector_csr_within_budget(d_small, terms))
            admitted_small++;
    }
    REQUIRE(admitted_small == want);
#endif
}

TEST_CASE("block_csr_budget_bytes: unset, the budget follows the RAM the job may still allocate",
          "[planner][csr_budget]") {
    // P6.1 (audit P1-matvec-cpu-03): a fixed 8 GiB ignored the job's memory.
    EnvGuard g(nullptr);
    const char* off = std::getenv("ED_MEM_GUARD_OFF");
    if (off != nullptr) ::unsetenv("ED_MEM_GUARD_OFF");
    const std::uint64_t avail = ed::core::available_ram_bytes();
    if (avail == 0) {
        REQUIRE(ed::planner::block_csr_budget_bytes() == (std::uint64_t{8} << 30));
    } else {
        const double b0 = static_cast<double>(ed::planner::block_csr_budget_bytes());
        // the RAM moves a little between the two reads: 5%
        REQUIRE(std::abs(b0 - 0.55 * static_cast<double>(avail)) <= 0.05 * 0.55 * static_cast<double>(avail));
        const std::uint64_t ws = avail / 4;
        const double b1 = static_cast<double>(ed::planner::block_csr_budget_bytes(ws));
        REQUIRE(std::abs(b1 - (0.55 * static_cast<double>(avail) - static_cast<double>(ws)))
                <= 0.05 * 0.55 * static_cast<double>(avail));
        REQUIRE(ed::planner::block_csr_budget_bytes(avail) == 0);   // a working set larger than the share
    }
    ::setenv("ED_MEM_GUARD_OFF", "1", 1);
    REQUIRE(ed::planner::block_csr_budget_bytes(~std::uint64_t{0} / 2) == ~std::uint64_t{0});
    if (off != nullptr) ::setenv("ED_MEM_GUARD_OFF", off, 1);
    else ::unsetenv("ED_MEM_GUARD_OFF");
}

TEST_CASE("CsrBudget: one block's operators share it", "[planner][csr_budget]") {
    ed::planner::CsrBudget b(100);
    REQUIRE(b.take(60));
    REQUIRE_FALSE(b.take(50));     // H took 60; S^2 does not fit
    REQUIRE(b.take(40));
    REQUIRE(b.left() == 0);
    b.give(25);                    // a build used less than its estimate
    REQUIRE(b.left() == 25);
}
