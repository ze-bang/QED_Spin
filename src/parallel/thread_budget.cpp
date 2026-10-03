// =============================================================================
// src/parallel/thread_budget.cpp
//
// Implementation of the auto thread-count budget. See
// include/ed/parallel/thread_budget.h for the design rationale.
//
// OpenBLAS integration uses a *weak symbol*: when the binary is linked
// against an OpenBLAS that exports ``openblas_set_num_threads`` /
// ``openblas_get_num_threads`` (the case for both the OpenBLAS-pthread
// and OpenBLAS-OpenMP backends, including the system packages on Debian
// / Ubuntu / RHEL), this file calls those symbols at runtime. When the
// build is linked against a different BLAS (Netlib reference BLAS,
// Apple Accelerate, MKL via the legacy interface), the weak symbols
// resolve to NULL and the BLAS-thread-count restore is a silent no-op.
// =============================================================================

#include "ed/parallel/thread_budget.h"
#include <ed/core/config.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Weak references into OpenBLAS. Resolve to NULL at runtime if the BLAS
// in use is not OpenBLAS, which is fine -- we just skip the BLAS-side
// thread cap in that case.
extern "C" {
int openblas_get_num_threads(void) __attribute__((weak));
void openblas_set_num_threads(int) __attribute__((weak));
}

bool auto_threads_disabled() { return !ed::env::flag("ED_AUTO_THREADS", true); }

int omp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace

namespace ed::parallel {

int auto_threads_for_dim(std::uint64_t dim) {
    const int max_t = omp_max_threads();
    if (auto_threads_disabled() || max_t <= 1) return max_t;

    // One OMP/BLAS worker per kPerK*1024 basis states (kPerK * 16 KiB of complex double per thread),
    // growing with dim up to the whole team: a small block on a big team spends its steps on
    // fork/join, a large one is memory-bandwidth bound and scales with the memory channels it
    // reaches. The old ceiling of 8 threads was tuned on a 2-channel desktop (Lanczos optimum at
    // 4-8 threads); on the 12-channel EPYC nodes it left most cores idle on every large thermal
    // block (audit P4-thermal-03).
    constexpr std::uint64_t kPerK = 8;
    const std::uint64_t denom = kPerK * 1024ULL;
    const std::uint64_t want64 = std::max<std::uint64_t>(1, dim / denom);
    return static_cast<int>(std::min<std::uint64_t>(want64, static_cast<std::uint64_t>(max_t)));
}

ThreadBudgetScope::ThreadBudgetScope(int threads, int blas_threads) {
    if (threads <= 0) return;
    // Clamp to hardware concurrency. Callers pass a large sentinel (e.g. 1<<20)
    // to mean "use all cores"; without this, omp_set_num_threads(1<<20) asks the
    // runtime to spawn ~1e6 threads -> pthread_create EAGAIN ("Resource
    // temporarily unavailable") on a cgroup / RLIMIT_NPROC-bounded node. The
    // doc comment at the call sites always claimed this clamp happened here.
#ifdef _OPENMP
    {
        const int hw = omp_get_num_procs();
        if (hw > 0 && threads > hw) threads = hw;
    }
#endif
    requested_ = threads;

#ifdef _OPENMP
    prev_omp_ = omp_get_max_threads();
    if (prev_omp_ != threads) {
        omp_set_num_threads(threads);
        restore_omp_ = true;
    }
#endif

    int blas = (blas_threads > 0) ? blas_threads : 1;
#ifdef _OPENMP
    {
        const int hw = omp_get_num_procs();
        if (hw > 0 && blas > hw) blas = hw;
    }
#endif
    if (openblas_get_num_threads && openblas_set_num_threads) {
        prev_openblas_ = openblas_get_num_threads();
        if (prev_openblas_ != blas) {
            openblas_set_num_threads(blas);
            restore_blas_ = true;
        }
    }
}

ThreadBudgetScope::~ThreadBudgetScope() {
#ifdef _OPENMP
    if (restore_omp_ && prev_omp_ > 0) { omp_set_num_threads(prev_omp_); }
#endif
    if (restore_blas_ && prev_openblas_ > 0 && openblas_set_num_threads) { openblas_set_num_threads(prev_openblas_); }
}

}  // namespace ed::parallel
