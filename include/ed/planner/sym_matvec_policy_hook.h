#pragma once
// =============================================================================
// include/ed/planner/sym_matvec_policy_hook.h
//
// Planner -> symmetry-backend hook for the rep-walk vs orbit-CSR decision.
//
// A symmetry sector can be applied two ways:
//   * REP WALK (RepSymmetryBasisPolicy, DEFAULT): hold only the orbit-rep list
//     (O(#reps) memory) and regenerate the projection arithmetically per emit
//     (~|G| group ops via the N<=32 perm-LUT). Cost per SpMV:
//     sector_dim × n_terms × group_size  (= full-Sz no-sym SpMV cost).
//   * ORBIT-CSR (SymmetryBasisPolicy, user-override only): materialize the
//     per-sector orbit basis (orbit elements + stored coefficients,
//     ~dim*|orbit|*24 B). Applies H to all |G| orbit elements per rep;
//     each output state still requires find_representative (O(|G|) ops).
//     Cost per SpMV: sector_dim × group_size × n_terms × group_size
//     = group_size× MORE expensive than rep-walk. Not auto-selected.
//
// This is the symmetry analogue of csr_policy_hook's matrix-free-vs-CSR choice.
// The planner cost model picks orbit-CSR when it fits the probed memory budget
// (speed win) and rep walk otherwise (the scalable default), then publishes the
// decision here. make_backend_ reads it; when unset (Auto) it falls back to the
// producer's rep_lazy() heuristic, so behaviour is unchanged unless the planner
// runs. Same PROCESS-GLOBAL atomic rationale as csr_policy_hook (the backend is
// built lazily, possibly on an OpenMP worker thread).
// =============================================================================

#include <atomic>
#include <cstdint>
#include <cstdlib>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace ed::planner {

/// Symmetry matvec strategy -- ONE planner-owned ordinal (memory up, apply-speed
/// up). Replaces the old rep-vs-orbit bool PLUS the independent ED_SYM_REDUCED_CSR
/// env gate, so the planner is the single chooser and env vars are overrides only.
///   * RepStream         -- RepSymmetryBasisPolicy, regenerate the projection per
///                          matvec (~|G| ops/emit). O(#reps) memory; the scalable
///                          floor that can never OOM.
///   * RepReducedCsr     -- rep policy + build the reduced sector matrix ONCE,
///                          then plain O(1)/nnz SpMV. Fast tier; +O(dim*nnz) mem.
///   * Auto              -- no plan ran -> the default (reduced-CSR).
/// (``OrbitMaterialized`` -- the legacy orbit-CSR walk -- was retired in
/// Stage 11c-2b together with its ``ED_SYM_REP=0`` escape; the rep policy
/// is the ONE matvec representation.)
enum class SymMatvecRepr : int {
    Auto              = -1,
    RepStream         =  0,
    RepReducedCsr     =  1,
};

namespace detail {
inline std::atomic<int>& sym_matvec_repr_slot() noexcept {
    static std::atomic<int> slot{static_cast<int>(SymMatvecRepr::Auto)};
    return slot;
}
}  // namespace detail

/// Set the active symmetry-matvec override (called by the planner before a solve).
inline void set_sym_matvec_repr(SymMatvecRepr r) noexcept {
    detail::sym_matvec_repr_slot().store(static_cast<int>(r),
                                         std::memory_order_relaxed);
}

/// Clear the override (restore the rep_lazy()/env behaviour).
inline void clear_sym_matvec_repr() noexcept {
    set_sym_matvec_repr(SymMatvecRepr::Auto);
}

/// Read the raw planner slot (no env folding). Returns a SymMatvecRepr value.
[[nodiscard]] inline int sym_matvec_repr() noexcept {
    return detail::sym_matvec_repr_slot().load(std::memory_order_relaxed);
}

/// Resolve the EFFECTIVE strategy with precedence: env override -> planner slot
/// -> Auto. The env knob is the manual escape hatch (cached process-globally):
/// ``ED_SYM_REDUCED_CSR=1`` forces RepReducedCsr, ``=0`` the CSR-free rep
/// walk. Consumed by CpuMatVecBackend (the reduced-CSR sub-choice).
[[nodiscard]] inline int resolved_sym_matvec_repr() noexcept {
    static const int env_override = [] {
        if (const char* e = std::getenv("ED_SYM_REDUCED_CSR")) {
            if (e[0] == '1' && e[1] == '\0')
                return static_cast<int>(SymMatvecRepr::RepReducedCsr);
            if (e[0] == '0' && e[1] == '\0')
                return static_cast<int>(SymMatvecRepr::RepStream);  // CSR-free rep walk
        }
        return static_cast<int>(SymMatvecRepr::Auto);
    }();
    if (env_override != static_cast<int>(SymMatvecRepr::Auto)) return env_override;
    const int slot = sym_matvec_repr();
    if (slot != static_cast<int>(SymMatvecRepr::Auto)) return slot;
    // DEFAULT (no env, no plan): reduced-CSR -- typically group_size x faster
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
/// written to close, just reachable only through an explicit
/// ED_SYM_SECTOR_PARALLEL=1 (the AUTO gate caps max_sector_dim at 4096, where
/// these CSRs are trivial). Dividing by the concurrent builder count keeps the
/// TOTAL in-flight CSR footprint under the knob regardless of thread count;
/// an over-budget sector still degrades to the CSR-free walk, never OOMs.
[[nodiscard]] inline bool sector_csr_within_budget(
        std::uint64_t dim, std::uint64_t terms_per_row) noexcept {
    const std::uint64_t est_bytes =
        dim * terms_per_row * (16u /* complex value */ + 4u /* col idx */)
        + (dim + 1) * 8u /* row ptr */;
    double budget_gib = 8.0;
    if (const char* v = std::getenv("ED_SYM_SECTOR_CSR_BUDGET_GIB")) {
        const double b = std::atof(v);
        if (b > 0.0) budget_gib = b;
    }
    budget_gib /= static_cast<double>(concurrent_sector_builders());
    return static_cast<double>(est_bytes)
           <= budget_gib * static_cast<double>(1ULL << 30);
}

/// RAII guard: set the override for a scope, restore the previous value on exit.
class ScopedSymMatvecRepr {
public:
    explicit ScopedSymMatvecRepr(SymMatvecRepr r) noexcept
        : prev_(detail::sym_matvec_repr_slot().load(std::memory_order_relaxed)) {
        set_sym_matvec_repr(r);
    }
    ~ScopedSymMatvecRepr() {
        detail::sym_matvec_repr_slot().store(prev_, std::memory_order_relaxed);
    }
    ScopedSymMatvecRepr(const ScopedSymMatvecRepr&)            = delete;
    ScopedSymMatvecRepr& operator=(const ScopedSymMatvecRepr&) = delete;

private:
    int prev_;
};

}  // namespace ed::planner
