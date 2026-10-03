#pragma once
// =============================================================================
// include/ed/matvec/csr_policy.h
//
// Symmetry-sector matvec strategy (rep walk vs reduced sector CSR) and the
// budget for materialising a reduced sector matrix.
//
// A symmetry sector is applied through the rep policy
// (RepSymmetryBasisPolicy): hold only the orbit-rep list (O(#reps) memory)
// and regenerate the projection arithmetically per emit (~|G| group ops via
// the N<=64 perm-LUT), or build the reduced sector matrix once and run a
// plain SpMV. ED_SYM_REDUCED_CSR overrides the choice; the budget below
// guards the CSR.
// =============================================================================

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <ed/core/config.h>
#include <ed/core/memory.h>

#ifdef _OPENMP
#include <omp.h>
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
    Auto = -1,
    RepStream = 0,
    RepReducedCsr = 1,
};

/// Resolve the EFFECTIVE strategy: the env knob is the manual escape hatch
/// (cached process-globally): ``ED_SYM_REDUCED_CSR=1`` forces RepReducedCsr,
/// ``=0`` the CSR-free rep walk; unset -> the default below. Consumed by
/// CpuMatVecBackend and the little-group engine (the reduced-CSR sub-choice).
[[nodiscard]] inline int resolved_sym_matvec_repr() noexcept {
    static const int env_override = [] {
        if (const std::optional<bool> on = ed::env::tristate("ED_SYM_REDUCED_CSR"))
            return static_cast<int>(*on ? SymMatvecRepr::RepReducedCsr : SymMatvecRepr::RepStream); // CSR-free rep walk
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

/// The bytes of a reduced sector CSR of `dim` rows with at most `per_row` entries a row, at its
/// build's peak: its values in a dictionary (sector_rows.h build_cross_csr: a 4-byte column and a
/// 1-2 byte value id an entry, the chunks' slabs released as the arrays fill), 7 bytes an entry and
/// the row pointers. A matrix with too many distinct values for a dictionary falls back to full
/// values (20 bytes an entry) only within the caller's cap, checked against its exact size.
[[nodiscard]] inline std::uint64_t csr_estimate_bytes(std::uint64_t dim, std::uint64_t per_row) noexcept {
    return dim * per_row * 7u + (dim + 1) * 8u /* row ptr */;
}

/// The bytes the reduced CSRs built around this call may take, ONE rule for every lane:
///   ED_SYM_SECTOR_CSR_BUDGET_GIB when set -- an absolute cap (0 admits nothing);
///   else 0.55 of the RAM the job may still allocate, less `working_set` (what the block's solver
///   will hold beside the CSR: footprint.h), which follows the job's memory instead of a fixed
///   8 GiB (audit P1-matvec-cpu-03); unlimited under ED_MEM_GUARD_OFF; 8 GiB when the RAM cannot
///   be read.
/// Either way it is an AGGREGATE, divided by the number of concurrent sector builders: blocks
/// solved inside an outer parallel loop keep the TOTAL in-flight CSR footprint under it (a
/// per-sector check would let N threads each allocate the whole budget).
[[nodiscard]] inline std::uint64_t block_csr_budget_bytes(std::uint64_t working_set = 0) noexcept {
    constexpr double GiB = 1073741824.0;
    double bytes = 0.0;
    const double knob = ed::env::real("ED_SYM_SECTOR_CSR_BUDGET_GIB", std::nan(""));
    if (!std::isnan(knob)) {
        bytes = std::max(0.0, knob) * GiB;
    } else if (ed::core::mem_guard_off()) {
        return ~std::uint64_t{0};
    } else if (const std::uint64_t avail = ed::core::available_ram_bytes(); avail > 0) {
        bytes = std::max(0.0, 0.55 * static_cast<double>(avail) - static_cast<double>(working_set));
    } else {
        bytes = 8.0 * GiB;
    }
    return static_cast<std::uint64_t>(bytes / static_cast<double>(concurrent_sector_builders()));
}

/// ONE budget decision for materializing a reduced sector matrix without a block budget (below),
/// shared by every lane. The estimate is an UPPER BOUND: each off-diagonal term contributes at most
/// one entry per source row. An over-budget sector falls back to the CSR-free walk on its own --
/// frontier sectors (N=36 half filling: hundreds of GB) need no env var.
[[nodiscard]] inline bool sector_csr_within_budget(std::uint64_t dim, std::uint64_t terms_per_row) noexcept {
    // The CSR stores 32-bit column indices: a sector of 2^32 or more states takes the walk.
    if (dim >= (std::uint64_t{1} << 32)) return false;
    return csr_estimate_bytes(dim, terms_per_row) <= block_csr_budget_bytes();
}

/// The bytes the reduced CSRs of ONE block's operators may hold together: the verb sizes it with
/// block_csr_budget_bytes(working set of the block's solver) before the solver allocates anything,
/// and each operator's lazy build takes its estimate from it -- H first, then S^2, then the
/// observables; an operator whose CSR no longer fits runs the walk.
class CsrBudget {
public:
    explicit CsrBudget(std::uint64_t bytes) noexcept : left_(bytes) {}
    /// Take `bytes` if they are left.
    [[nodiscard]] bool take(std::uint64_t bytes) noexcept {
        std::uint64_t cur = left_.load();
        while (cur >= bytes)
            if (left_.compare_exchange_weak(cur, cur - bytes)) return true;
        return false;
    }
    /// Return what a build took but did not use.
    void give(std::uint64_t bytes) noexcept { left_.fetch_add(bytes); }
    [[nodiscard]] std::uint64_t left() const noexcept { return left_.load(); }

private:
    std::atomic<std::uint64_t> left_;
};

} // namespace ed::planner
