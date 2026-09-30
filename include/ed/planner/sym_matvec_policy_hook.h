#pragma once
// =============================================================================
// include/ed/planner/sym_matvec_policy_hook.h
//
// Symmetry-sector matvec strategy (rep walk vs reduced sector CSR) and the
// budget for materialising a reduced sector matrix.
//
// A symmetry sector is applied through the rep policy
// (RepSymmetryBasisPolicy): hold only the orbit-rep list (O(#reps) memory)
// and regenerate the projection arithmetically per emit (~|G| group ops via
// the N<=32 perm-LUT), or build the reduced sector matrix once and run a
// plain SpMV. ED_SYM_REDUCED_CSR overrides the choice; the budget below
// guards the CSR.
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <ed/config/env_registry.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::planner {

/// Symmetry matvec strategy (memory up, apply-speed up).
///   * RepStream         -- RepSymmetryBasisPolicy, regenerate the projection per
///                          matvec (~|G| ops/emit). O(#reps) memory; the scalable
///                          floor that can never OOM.
///   * RepReducedCsr     -- rep policy + build the reduced sector matrix ONCE,
///                          then plain O(1)/nnz SpMV. Fast tier; +O(dim*nnz) mem.
///   * Auto              -- no override -> the default (reduced-CSR).
enum class SymMatvecRepr : int {
    Auto              = -1,
    RepStream         =  0,
    RepReducedCsr     =  1,
};

/// Resolve the EFFECTIVE strategy: the env knob is the manual escape hatch
/// (cached process-globally): ``ED_SYM_REDUCED_CSR=1`` forces RepReducedCsr,
/// ``=0`` the CSR-free rep walk; unset -> the default below. Consumed by
/// CpuMatVecBackend and the little-group engine (the reduced-CSR sub-choice).
[[nodiscard]] inline int resolved_sym_matvec_repr() noexcept {
    static const int env_override = [] {
        if (const char* e = ed::env::raw("ED_SYM_REDUCED_CSR")) {
            if (e[0] == '1' && e[1] == '\0')
                return static_cast<int>(SymMatvecRepr::RepReducedCsr);
            if (e[0] == '0' && e[1] == '\0')
                return static_cast<int>(SymMatvecRepr::RepStream);  // CSR-free rep walk
        }
        return static_cast<int>(SymMatvecRepr::Auto);
    }();
    if (env_override != static_cast<int>(SymMatvecRepr::Auto)) return env_override;
    // DEFAULT (no env): reduced-CSR -- typically group_size x faster
    // than the rep walk (the 27-site BFG benchmark measured the rep walk at
    // ~19-42 s/matvec vs reduced-CSR at ~0.2-0.4 s/matvec). Memory-bound large
    // systems can opt out with ED_SYM_REDUCED_CSR=0 (CSR-free rep walk).
    return static_cast<int>(SymMatvecRepr::RepReducedCsr);
}

/// How many sectors are being built CONCURRENTLY around this call.
///
/// The sector-parallel lanes run `omp parallel for` over sectors and build each
/// sector's reduced CSR lazily inside the loop body, i.e. on an OMP worker. The
/// outermost active team size is therefore the number of reduced CSRs that can
/// be in flight at once. Reads the level-1 team (not the innermost) so an inner
/// Lanczos/BLAS region cannot mistake its own width for the sector fan-out.
[[nodiscard]] inline unsigned concurrent_sector_builders() noexcept {
#ifdef _OPENMP
    if (omp_in_parallel() && omp_get_active_level() >= 1) {
        const int t = omp_get_team_size(1);
        if (t > 1) return static_cast<unsigned>(t);
    }
#endif
    return 1u;
}

/// Stage 9f consolidation: ONE budget decision for materializing a reduced
/// sector matrix, shared by the abelian CpuMatVecBackend and the little-group
/// engine's RepSectorMatVec. (Twin-lane drift here is exactly how the abelian
/// lane shipped WITHOUT a guard while the engine had one -- keep a single
/// definition.) The estimate is an UPPER BOUND: each off-diagonal term
/// contributes at most one entry per source row; the budget knob is
/// ``ED_SYM_SECTOR_CSR_BUDGET_GIB`` (default 8, read per call so tests can
/// toggle without restart). An over-budget sector falls back to the CSR-free
/// walk on its own -- frontier sectors (N=36 half filling: hundreds of GB)
/// need no env var.
///
/// Jul 2026 audit: the budget is an AGGREGATE, not per-sector. It used to be
/// evaluated per sector with no knowledge of the outer fan-out, so N sector
/// threads could each pass an 8 GiB check and allocate N x 8 GiB against a
/// guard that believed it was bounding one -- the same OOM class Stage 9f was
/// written to close, reachable when blocks are solved inside an outer
/// parallel loop. Dividing by the concurrent builder count keeps the
/// TOTAL in-flight CSR footprint under the knob regardless of thread count;
/// an over-budget sector still degrades to the CSR-free walk, never OOMs.
[[nodiscard]] inline bool sector_csr_within_budget(
        std::uint64_t dim, std::uint64_t terms_per_row) noexcept {
    const std::uint64_t est_bytes =
        dim * terms_per_row * (16u /* complex value */ + 4u /* col idx */)
        + (dim + 1) * 8u /* row ptr */;
    double budget_gib = 8.0;
    if (const char* v = ed::env::raw("ED_SYM_SECTOR_CSR_BUDGET_GIB")) {
        const double b = std::atof(v);
        if (b > 0.0) budget_gib = b;
    }
    budget_gib /= static_cast<double>(concurrent_sector_builders());
    return static_cast<double>(est_bytes)
           <= budget_gib * static_cast<double>(1ULL << 30);
}

}  // namespace ed::planner
