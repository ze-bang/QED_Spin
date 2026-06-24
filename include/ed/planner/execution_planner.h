#pragma once
// =============================================================================
// include/ed/planner/execution_planner.h
//
// THE capability-aware ED execution planner. Given a task (TaskDescriptor) and
// a machine (SystemCapabilities), emit ONE ExecutionPlan that drives every
// downstream knob:
//
//   * matvec strategy : CSR vs matrix-free  -- capability-gated, NO hard dim
//                       cap. CSR is chosen when its assembled footprint plus
//                       the solver working set fits the memory budget AND the
//                       one-time assembly amortizes over the expected matvecs.
//   * device lane     : cpu / gpu / mpi / mpi_gpu  -- gated on fit + build.
//   * reorth policy   : per-method default, downgraded if the full-reorth
//                       basis would not fit.
//   * basis strategy  : how the (symmetry / fixed-Sz) basis is built + looked
//                       up; flags infeasibility (estimate + select + guard).
//
// `apply(plan)` pushes the matvec decision into the matvec backend via the
// leaf `csr_policy_hook.h` so the heavily-templated Operator constructors do
// not need to change. Device / reorth are consumed by the orchestrator at the
// select_backend / Lanczos-kernel call sites.
//
// User / env overrides always win over the plan (manual escape hatch).
// =============================================================================

#include <string>
#include <vector>

#include <ed/krylov/lanczos_kernel.h>   // ed::krylov::ReorthPolicy
#include <ed/planner/system_capabilities.h>
#include <ed/planner/task_cost_model.h>

namespace ed::planner {

enum class MatvecStrategy : std::uint8_t { MatrixFree, Csr };
enum class DeviceLane     : std::uint8_t { Cpu, Gpu, Mpi, MpiGpu };
enum class BasisStrategy  : std::uint8_t {
    FullDense,          ///< no reduction; dense full-Hilbert basis
    BinarySearchReps,   ///< symmetry/Sz reps, O(log dim) reverse lookup (tableless)
    DenseRankTable,     ///< symmetry/Sz reps + dense combinadic O(1) rank table
    Distributed         ///< reps sharded across MPI ranks (selected, may be a guard)
};

// Explicit user constraints. Each "auto"/None means "let the planner decide".
struct UserConstraints {
    std::string device   = "auto";   ///< auto | cpu | gpu | mpi | mpi_gpu
    std::string matvec   = "auto";   ///< auto | csr | matrix_free
    int         n_ranks  = 0;        ///< 0 -> probe / cores
    // Fraction of the probed memory budget the plan is allowed to consume.
    double      memory_safety = 0.8;
};

struct ExecutionPlan {
    MatvecStrategy matvec   = MatvecStrategy::MatrixFree;
    DeviceLane     device   = DeviceLane::Cpu;
    int            n_ranks  = 1;
    ed::krylov::ReorthPolicy reorth = ed::krylov::ReorthPolicy::LocalDGKS3;
    int            krylov_dim_cap = 0;   ///< 0 -> solver default
    BasisStrategy  basis    = BasisStrategy::FullDense;
    /// Fixed-Sz (no-symmetry) lane: use the tableless combinadic basis (only an
    /// O(N^2) BinomialTable) instead of the materialized C(N,n_up) basis vector
    /// + Lin table. Set when the materialized basis would not fit the budget.
    bool           tableless_fixed_sz = false;

    bool                     feasible = true;
    std::string              bottleneck = "ok";  // ok|memory|basis_construction|build|kernel
    double                   est_memory_gb = 0.0;
    double                   est_seconds   = 0.0;
    std::vector<std::string> notes;
    std::vector<std::string> suggestions;

    [[nodiscard]] std::string summary() const;
    [[nodiscard]] const char* matvec_str()  const noexcept;
    [[nodiscard]] const char* device_str()  const noexcept;
    [[nodiscard]] const char* basis_str()   const noexcept;
    [[nodiscard]] const char* reorth_str()  const noexcept;
};

// Produce the plan. Pure function of (task, caps, constraints) -- no global
// state is read or written here (see `apply`).
[[nodiscard]] ExecutionPlan plan_execution(const TaskDescriptor&      task,
                                           const SystemCapabilities&  caps,
                                           const UserConstraints&     uc = {});

// Push the plan's matvec decision into the backend (via csr_policy_hook).
// Returns a token the caller keeps alive for the duration of the solve; on
// destruction it restores the previous override. Device / reorth are applied
// by the orchestrator at their own call sites.
void apply_csr_decision(const ExecutionPlan& plan);

// Push the plan's fixed-Sz basis decision into the operator builder (via
// basis_policy_hook), so a subsequently-constructed FixedSzOperator picks the
// tableless combinadic basis when the plan says so. Consumed by both the GS and
// finite-temperature lanes (they share the fixed-Sz operator construction).
void apply_basis_decision(const ExecutionPlan& plan);

}  // namespace ed::planner
