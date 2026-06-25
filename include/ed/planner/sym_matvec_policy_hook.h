#pragma once
// =============================================================================
// include/ed/planner/sym_matvec_policy_hook.h
//
// Planner -> symmetry-backend hook for the rep-walk vs orbit-CSR decision.
//
// A symmetry sector can be applied two ways:
//   * REP WALK (RepSymmetryBasisPolicy): hold only the orbit-rep list
//     (O(#reps) memory) and regenerate the projection arithmetically per emit
//     (~|G| group ops, mitigated by the N<=32 perm-LUT + dense rank table).
//   * ORBIT-CSR (SymmetryBasisPolicy): materialize the per-sector orbit basis
//     (orbit elements + stored coefficients, ~dim*|orbit|*24 B) and run a
//     plain stored-coefficient matvec -- faster per apply, much heavier memory.
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

namespace ed::planner {

/// Symmetry matvec representation override. -1 = no override (use the producer's
/// rep_lazy() heuristic / ED_SYM_REP env), 0 = force the rep walk, 1 = force the
/// materialized orbit-CSR.
enum class SymMatvecRepr : int { Auto = -1, Rep = 0, OrbitCsr = 1 };

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

/// Read the active override. Returns -1 / 0 / 1 (see SymMatvecRepr).
[[nodiscard]] inline int sym_matvec_repr() noexcept {
    return detail::sym_matvec_repr_slot().load(std::memory_order_relaxed);
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
