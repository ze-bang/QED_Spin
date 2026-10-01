#pragma once
// =============================================================================
// src/orchestrator/orchestrator_internal.h -- PRIVATE to the workflow
// orchestrator.
//
// The two entry points dispatch through `ed::select_backend` to choose
// a concrete Backend (Cpu, or Cuda when built WITH_CUDA) and invoke the
// matching templated kernel:
//
//     ed::solve     -> lanczos_kernel<Backend>            (single eig)
//                  or krylov_schur_kernel<Backend>       (many eigs / harder problems)
//                  or full_diag fallback                  (small dim)
//     ed::thermal   -> mtpq_kernel<Backend>  (mTPQ)
//                  or the FTLM / OFTLM kernels
//
// Every lane returns the uniform Result shape from
// `include/ed/core/results.h`; nothing is written to disk.
//
// Carries the include block every orchestrator translation unit needs and the
// Hermitian-input guard the entry points share.
// Nothing outside src/orchestrator/ includes this header: the public surface
// is include/ed/orchestrator.h.
//
// Every entry point nests its RAII the same way (ThreadBudgetScope ->
// pin_omp_threads_once -> select_backend).
//
// File map
//   orch_solve.cpp     solve() + the backend-templated solve_on<Backend>
//                      lanes (Lanczos / KrylovSchur / FullDiag). The ONLY
//                      translation unit
//                      that instantiates a backend-templated eigensolver,
//                      so the CudaBackend instantiation of the solve path
//                      lives here and nowhere else.
// =============================================================================

#include <ed/config/env_registry.h>   // typed environment accessors
#include <ed/orchestrator.h>

#include <ed/core/mem_guard.h>           // leaf working-set guard (clean error vs OOM)
#include <ed/krylov/krylov_schur_kernel.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/ritz_convergence.h>
#include <ed/krylov/tridiag_eigensolver.h>  // solve_tridiag*
#include <ed/krylov/subspace_policy.h>      // krylov_subspace_dim / krylov_vector_budget
// The matvec/symmetry leaf policy hooks (sym_matvec_policy_hook and the
// ED_CSR_* env) provide the default + env-override behaviour, consumed lazily
// inside the operator backends; the orchestrator does not override them.
#include <ed/symmetry/canonical_thermo.h>   // canonical_thermo_from_eigs (single impl)

#include <fstream>   // /proc/meminfo
#include <string>
#include <unistd.h>  // sysconf
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/parallel/numa.h>            // pin_omp_threads_once
#include <ed/parallel/thread_budget.h>   // auto_threads_for_dim + ThreadBudgetScope
#include <ed/thermal/tpq_thermo.h>  // mtpq_canonical_thermo, mtpq_steps_for
#include <ed/core/errors.h>         // ed::ConvergenceError, ed::ResourceLimit
#include <ed/solvers/lanczos.h>  // FullDiag fallback (zheevd on the dense matrix)
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/oftlm_kernel.h>
#include <cstdio>
#include <ed/thermal/mtpq_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <filesystem>
#include <ed/core/log.h>
#include <random>
#include <stdexcept>
#include <type_traits>  // std::is_same_v
#include <variant>

namespace ed::workflows {

// Every solver lane assumes a Hermitian operator (Lanczos tridiagonalises
// the symmetric part silently; the rep kernels apply H^dagger).
// ``LinearOperator::is_hermitian`` is a structural check on the
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
