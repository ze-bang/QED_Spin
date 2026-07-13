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

#include <algorithm>
#include <atomic>
#include <cstdint>
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

/// Scoped override: set the basis representation for the lifetime of the token
/// (e.g. across one FixedSzOperator construction) and restore the previous value
/// on destruction. Mirrors ScopedCsrOverride / ScopedSymMatvecRepr.
class ScopedBasisRepr {
public:
    explicit ScopedBasisRepr(BasisRepr r) noexcept : prev_(basis_repr()) {
        set_basis_repr(r);
    }
    ~ScopedBasisRepr() {
        detail::basis_repr_slot().store(prev_, std::memory_order_relaxed);
    }
    ScopedBasisRepr(const ScopedBasisRepr&)            = delete;
    ScopedBasisRepr& operator=(const ScopedBasisRepr&) = delete;
private:
    int prev_;
};

/// Resolve whether a fixed-Sz operator should use the tableless combinadic
/// basis. Env overrides the planner override, which overrides the default.
[[nodiscard]] inline bool prefer_tableless_fixed_sz() noexcept {
    if (const char* v = std::getenv("ED_FIXED_SZ_TABLELESS")) {
        if (v[0] == '1' && v[1] == '\0') return true;
        if (v[0] == '0' && v[1] == '\0') return false;
    }
    return basis_repr() == static_cast<int>(BasisRepr::Tableless);
}

/// Byte budget for the materialized fixed-Sz basis list (the dimension-aware
/// default below). ED_FIXED_SZ_TABLE_BUDGET_GIB overrides; default 16 GiB
/// keeps every N <= 32 workload on the O(1)-lookup materialized path
/// (C(32,16) x 8 B = 4.8 GB -- the CPU no-sym lane is 6.6x faster there)
/// while N >= 35 half-filling flips to tableless automatically instead of
/// silently allocating 72.6 GB the consumer may never read.
[[nodiscard]] inline double fixed_sz_table_budget_bytes() noexcept {
    double gib = 16.0;
    if (const char* v = std::getenv("ED_FIXED_SZ_TABLE_BUDGET_GIB")) {
        const double parsed = std::atof(v);
        if (parsed > 0.0) gib = parsed;
    }
    return gib * 1073741824.0;
}

/// Dimension-aware resolution: env force, then the planner override, then a
/// BUDGET default -- materialize iff the basis list (C(n_bits, n_up) x 8 B)
/// fits ED_FIXED_SZ_TABLE_BUDGET_GIB. This replaces the legacy
/// always-materialize default that eagerly built the 72.6 GB list at 36-site
/// half filling even for consumers that only read the term list.
[[nodiscard]] inline bool
prefer_tableless_fixed_sz(std::uint64_t n_bits, std::int64_t n_up) noexcept {
    if (const char* v = std::getenv("ED_FIXED_SZ_TABLELESS")) {
        if (v[0] == '1' && v[1] == '\0') return true;
        if (v[0] == '0' && v[1] == '\0') return false;
    }
    const int r = basis_repr();
    if (r != static_cast<int>(BasisRepr::Default))
        return r == static_cast<int>(BasisRepr::Tableless);
    if (n_up < 0) return false;
    long double dim = 1.0L;
    const std::int64_t kk =
        std::min<std::int64_t>(n_up, static_cast<std::int64_t>(n_bits) - n_up);
    if (kk < 0) return false;
    for (std::int64_t i = 0; i < kk; ++i) {
        dim *= static_cast<long double>(static_cast<std::int64_t>(n_bits) - i);
        dim /= static_cast<long double>(i + 1);
    }
    return dim * 8.0L
        > static_cast<long double>(fixed_sz_table_budget_bytes());
}

}  // namespace ed::planner
