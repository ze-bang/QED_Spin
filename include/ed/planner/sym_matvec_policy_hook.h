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
#include <cstdlib>

namespace ed::planner {

/// Symmetry matvec strategy -- ONE planner-owned ordinal (memory up, apply-speed
/// up). Replaces the old rep-vs-orbit bool PLUS the independent ED_SYM_REDUCED_CSR
/// env gate, so the planner is the single chooser and env vars are overrides only.
///   * RepStream         -- RepSymmetryBasisPolicy, regenerate the projection per
///                          matvec (~|G| ops/emit). O(#reps) memory; the scalable
///                          floor that can never OOM.
///   * RepReducedCsr     -- rep policy + build the reduced sector matrix ONCE,
///                          then plain O(1)/nnz SpMV. Fast tier; +O(dim*nnz) mem.
///   * OrbitMaterialized -- SymmetryBasisPolicy: materialized orbit lists
///                          (~dim*|orbit|*24 B) walked with stored coeffs. Kept
///                          for back-compat; also the representation the
///                          observable/DSSF lane reuses.
///   * Auto              -- no plan ran -> the producer's rep_lazy() heuristic.
/// `Rep` / `OrbitCsr` are kept as aliases for pre-ordinal call sites.
enum class SymMatvecRepr : int {
    Auto              = -1,
    RepStream         =  0,
    RepReducedCsr     =  1,
    OrbitMaterialized =  2,
    Rep               =  0,   ///< deprecated alias == RepStream
    OrbitCsr          =  2,   ///< deprecated alias == OrbitMaterialized
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
/// -> Auto. The env knobs are the manual escape hatch (cached process-globally,
/// matching the legacy gates): ``ED_SYM_REP=0`` forces OrbitMaterialized (the old
/// "disable rep" switch) and takes precedence; ``ED_SYM_REDUCED_CSR=1`` forces
/// RepReducedCsr. Consumed by SectorOperator::make_backend_ (rep-vs-orbit policy)
/// and CpuMatVecBackend (the reduced-CSR sub-choice).
[[nodiscard]] inline int resolved_sym_matvec_repr() noexcept {
    static const int env_override = [] {
        if (const char* e = std::getenv("ED_SYM_REP"))
            if (e[0] == '0' && e[1] == '\0')
                return static_cast<int>(SymMatvecRepr::OrbitMaterialized);
        if (const char* e = std::getenv("ED_SYM_REDUCED_CSR"))
            if (e[0] == '1' && e[1] == '\0')
                return static_cast<int>(SymMatvecRepr::RepReducedCsr);
        return static_cast<int>(SymMatvecRepr::Auto);
    }();
    if (env_override != static_cast<int>(SymMatvecRepr::Auto)) return env_override;
    return sym_matvec_repr();
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
