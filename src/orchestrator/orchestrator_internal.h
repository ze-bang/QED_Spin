#pragma once
// =============================================================================
// src/orchestrator/orchestrator_internal.h -- PRIVATE to the workflow
// orchestrator.
//
// The two entry points dispatch through `ed::select_backend` to choose
// one of the four concrete Backends (Cpu / Cuda / Mpi / MpiCuda) and
// invoke the matching templated kernel from Phase 2:
//
//     ed::solve     -> lanczos_kernel<Backend>            (single eig)
//                  or krylov_schur_kernel<Backend>       (many eigs / harder problems)
//                  or full_diag fallback                  (small dim)
//     ed::thermal   -> tpq_kernel<Backend>  (mTPQ)
//                  or the FTLM / OFTLM kernels
//
// Every lane returns the uniform Result shape from
// `include/ed/core/results.h`; nothing is written to disk.
//
// Carries the include block every orchestrator translation unit needs, the
// Hermitian-input guard the entry points share, and the declarations of
// the shared plumbing helpers whose definitions live in orch_common.cpp.
// Nothing outside src/orchestrator/ includes this header: the public surface
// is include/ed/orchestrator.h.
//
// The split is a pure structural move of the former src/orchestrator.cpp --
// same lanes, same RAII nesting (ThreadBudgetScope -> pin_omp_threads_once ->
// select_backend), same env reads, same dispatch order.
//
// File map
//   orch_common.cpp    shared plumbing: the exact-small thermal env probe
//   orch_solve.cpp     solve() + the backend-templated solve_on<Backend>
//                      lanes (Lanczos / KrylovSchur / FullDiag). The ONLY
//                      translation unit
//                      that instantiates a backend-templated eigensolver,
//                      so the CudaBackend instantiation of the solve path
//                      lives here and nowhere else.
//   orch_thermal.cpp   thermal(): exact-small eigenspectrum fallback, mTPQ
//                      sampling, FTLM / OFTLM lanes, all-Sz sweep
// =============================================================================

#include <ed/config/env_registry.h>   // typed environment accessors
#include <ed/orchestrator.h>

#include <ed/core/mem_guard.h>           // leaf working-set guard (clean error vs OOM)
#include <ed/krylov/krylov_schur_kernel.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/ritz_convergence.h>
#include <ed/krylov/tridiag_eigensolver.h>  // solve_tridiag* (MPI-free)
#include <ed/krylov/subspace_policy.h>      // krylov_subspace_dim / krylov_vector_budget
// Execution planner / feasibility "dictator" removed (sensible defaults +
// env-override leaf hooks instead). The matvec/symmetry leaf policy hooks
// (sym_matvec_policy_hook, basis_policy_hook, the ED_CSR_* env) still provide
// the default + env-override behaviour, consumed lazily inside the operator
// backends; the orchestrator no longer overrides them.
#include <ed/symmetry/canonical_thermo.h>   // canonical_thermo_from_eigs (single impl)

#include <fstream>   // /proc/meminfo
#include <string>
#include <unistd.h>  // sysconf
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/parallel/numa.h>            // pin_omp_threads_once
#include <ed/parallel/thread_budget.h>   // auto_threads_for_dim + ThreadBudgetScope
#include <ed/thermal/tpq_thermo.h>  // compute_tpq_thermo_from_trajectories aggregator
#include <ed/solvers/lanczos.h>  // FullDiag fallback (zheevd on the dense matrix)
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/oftlm_kernel.h>
#include <cstdio>
#include <ed/thermal/mtpq_kernel.h>
#include <ed/thermal/mtpq_f32.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>      // getenv (Wave 1.1 real-H fast-path opt-out)
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>  // std::is_same_v (Wave 1.1)
#include <variant>

namespace ed::workflows {

// ---------------------------------------------------------------------------
// Helpers shared by more than one orchestrator translation unit. They were
// anonymous-namespace statics inside the single orchestrator.cpp; they now
// carry external linkage with their one definition in orch_common.cpp.
// ---------------------------------------------------------------------------
namespace orch_detail {

/// ED_THERMAL_EXACT_SMALL=0 forces the real sampling kernel even at
/// D <= SMALL_THERMAL_DIM. Read per call so a test can toggle it without
/// restarting the process.
[[nodiscard]] bool exact_small_thermal_enabled() noexcept;

}  // namespace orch_detail

// Correctness (2026-09-11): every solver lane assumes a Hermitian operator
// (Lanczos tridiagonalises the symmetric part silently; the rep kernels apply
// H^dagger). ``LinearOperator::is_hermitian`` is now a structural check on the
// term list for ``Operator`` and its subclasses; refuse early and loudly.
inline void require_hermitian_input(const LinearOperator& H, const char* verb) {
    if (!H.is_hermitian()) {
        throw std::invalid_argument(
            std::string(verb) + ": the operator is not Hermitian (an off-diagonal term "
            "has no adjoint partner with the conjugate coefficient, or a diagonal term "
            "carries a complex coefficient). Add the Hermitian-conjugate terms.");
    }
}

}  // namespace ed::workflows
