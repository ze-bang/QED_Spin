#pragma once
// =============================================================================
// include/ed/planner/basis_policy_hook.h
//
// Leaf hook (zero heavy deps) letting the capability-aware execution planner
// steer fixed-Sz basis construction between the materialized representation
// (C(N,n_up) basis vector + Lin table; O(1) lookup) and the tableless
// combinadic representation (only an O(N^2) BinomialTable; O(N) lookup, no
// C(N,n_up)-sized allocation -- the ~72 GB wall at N=36).
//
// The fixed-Sz Operator constructor (SubspaceOperator<FixedSzBasisPolicy>)
// consults `prefer_tableless_fixed_sz()` to pick FixedSzSubspace::build vs
// build_tableless. Precedence, highest first:
//   1. env ED_FIXED_SZ_TABLELESS (=1 force tableless, =0 force materialized)
//   2. the active planner override set here
//   3. default: materialized (legacy behaviour, unchanged when no plan runs)
//
// Process-global atomic for the same reason as csr_policy_hook: the operator
// may be built off the orchestrator thread.
// =============================================================================

#include <atomic>
#include <cstdlib>

namespace ed::planner {

enum class BasisRepr : int { Default = -1, Materialized = 0, Tableless = 1 };

namespace detail {
inline std::atomic<int>& basis_repr_slot() noexcept {
    static std::atomic<int> slot{static_cast<int>(BasisRepr::Default)};
    return slot;
}
}  // namespace detail

inline void set_basis_repr(BasisRepr r) noexcept {
    detail::basis_repr_slot().store(static_cast<int>(r), std::memory_order_relaxed);
}
inline void clear_basis_repr() noexcept { set_basis_repr(BasisRepr::Default); }
[[nodiscard]] inline int basis_repr() noexcept {
    return detail::basis_repr_slot().load(std::memory_order_relaxed);
}

/// Resolve whether a fixed-Sz operator should use the tableless combinadic
/// basis. Env overrides the planner override, which overrides the default.
[[nodiscard]] inline bool prefer_tableless_fixed_sz() noexcept {
    if (const char* v = std::getenv("ED_FIXED_SZ_TABLELESS")) {
        if (v[0] == '1' && v[1] == '\0') return true;
        if (v[0] == '0' && v[1] == '\0') return false;
    }
    return basis_repr() == static_cast<int>(BasisRepr::Tableless);
}

}  // namespace ed::planner
