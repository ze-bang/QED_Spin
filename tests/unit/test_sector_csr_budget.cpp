// =============================================================================
// test_sector_csr_budget  (Catch2 v3)
//
// Pins ed::planner::sector_csr_within_budget -- THE single reduced-CSR budget
// decision, shared by the abelian CpuMatVecBackend build sites and the
// little-group engine's RepSectorMatVec. (Twin drift here is exactly how the
// abelian lane once shipped with no guard at all while the engine had one.)
//
// The property that matters is that the knob bounds the TOTAL in-flight CSR
// footprint, not one sector's. The sector-parallel lanes build each sector's
// CSR lazily inside an `omp parallel for` over sectors, so before the Jul-2026
// audit N threads could each pass an 8 GiB check independently and allocate
// N x 8 GiB -- a guard that believed it was bounding one.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/planner/sym_matvec_policy_hook.h>

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
