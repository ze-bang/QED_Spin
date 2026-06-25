#pragma once
// =============================================================================
// include/ed/planner/csr_policy_hook.h
//
// Leaf hook (zero dependencies) that lets the capability-aware execution
// planner (`ed::planner::ExecutionPlanner`) override the matvec backend's
// CSR-vs-matrix-free decision WITHOUT threading an ExecutionPlan through the
// heavily-templated Operator / CpuMatVecBackend constructors.
//
// The matvec backend factory reads the override inside
// `ed::matvec::detail::read_tunables` / `read_symmetry_tunables`. Precedence,
// highest first:
//
//   1. explicit env vars (ED_CSR_FORCE / ED_CSR_DIM_MAX / ED_SYM_CSR_DIM_MAX)
//   2. the active CSR override set here by the planner
//   3. the static `default_csr_cutoff` (legacy behaviour when no plan runs)
//
// Why a separate leaf header and not `execution_planner.h`: `matvec_backend.h`
// is a low-level header included all over the term-kernel layer. Depending on
// the full planner (which transitively pulls in the cost model, ReorthPolicy,
// etc.) would create an include cycle. This header has no includes beyond
// <atomic>, so the edge `matvec_backend.h -> csr_policy_hook.h` is a safe leaf.
//
// The override is a PROCESS-GLOBAL atomic (not thread_local): the matvec
// backend is built lazily on first `apply()`, which may happen on an OpenMP
// worker thread, so a thread_local set on the orchestrator thread would not be
// visible. A task runs under one plan at a time (sector-parallel thermal shares
// the same plan across sectors), so a single global snapshot is correct.
// =============================================================================

#include <atomic>

namespace ed::planner {

/// CSR override state. -1 = no override (use env / static cutoff),
/// 0 = force matrix-free, 1 = force CSR assembly.
enum class CsrOverride : int { None = -1, MatrixFree = 0, Csr = 1 };

namespace detail {
inline std::atomic<int>& csr_override_slot() noexcept {
    static std::atomic<int> slot{static_cast<int>(CsrOverride::None)};
    return slot;
}
}  // namespace detail

/// Set the active CSR override (called by the planner before a solve).
inline void set_csr_override(CsrOverride o) noexcept {
    detail::csr_override_slot().store(static_cast<int>(o),
                                      std::memory_order_relaxed);
}

/// Clear the override (restore env/static-cutoff behaviour).
inline void clear_csr_override() noexcept {
    set_csr_override(CsrOverride::None);
}

/// Read the active override. Returns -1 / 0 / 1 (see CsrOverride).
[[nodiscard]] inline int csr_override() noexcept {
    return detail::csr_override_slot().load(std::memory_order_relaxed);
}

/// RAII guard: set the override for a scope, restore the previous value on exit.
class ScopedCsrOverride {
public:
    explicit ScopedCsrOverride(CsrOverride o) noexcept
        : prev_(csr_override()) {
        set_csr_override(o);
    }
    ~ScopedCsrOverride() {
        detail::csr_override_slot().store(prev_, std::memory_order_relaxed);
    }
    ScopedCsrOverride(const ScopedCsrOverride&) = delete;
    ScopedCsrOverride& operator=(const ScopedCsrOverride&) = delete;
private:
    int prev_;
};

}  // namespace ed::planner
