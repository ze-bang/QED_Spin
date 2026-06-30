// =============================================================================
// src/parallel/thread_budget.cpp
//
// Implementation of the auto thread-count budget (Phase 6 #2). See
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
int  openblas_get_num_threads(void) __attribute__((weak));
void openblas_set_num_threads(int)  __attribute__((weak));
}

bool auto_threads_disabled() {
    const char* env = std::getenv("ED_AUTO_THREADS");
    if (!env || env[0] == '\0') return false;
    if (std::strcmp(env, "0")     == 0) return true;
    if (std::strcmp(env, "false") == 0) return true;
    if (std::strcmp(env, "FALSE") == 0) return true;
    if (std::strcmp(env, "no")    == 0) return true;
    if (std::strcmp(env, "NO")    == 0) return true;
    return false;
}

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

    // Two knobs:
    //   * ED_AUTO_THREADS_PER_K (default 8) -- aim for one OMP/BLAS worker per
    //     K*1024 basis states (i.e. K * 16 KiB of complex double per thread).
    //     This grows the team linearly with dim until we cross the hardware
    //     ceiling.
    //   * ED_AUTO_THREADS_CEIL (default 8) -- soft cap on the number of
    //     threads regardless of dim, because the dominant Lanczos kernels
    //     (SpMV, zaxpy, dznrm2) are memory-bandwidth bound and saturate
    //     well below ``max_threads`` on every machine we have measured.
    //     Set to 0 to disable the soft cap (use ``min(dim/per_k, max_t)``).
    //
    // On a 32-core x86_64 box (Ryzen 7950X-class, OpenBLAS-pthread, gcc-13)
    // we measured the following Lanczos(N) total runtime in ms vs OMP+BLAS
    // thread count for the unsymmetrised 1D Heisenberg PBC chain:
    //
    //     N \\ threads     1    2    4    8    16
    //     -----------   ----------------------------
    //     14             7    7    6   11    11
    //     16            32   28   24   38    60
    //     18           320  244  219  198   248
    //     20          1647 1486 1094 1374  1310
    //
    // The optimum sits at 4-8 threads for every N >= 16; the default
    // ``ED_AUTO_THREADS_CEIL=8`` captures that without needing per-machine
    // tuning. HPC users on systems with significantly more memory
    // bandwidth (multi-socket, MI300A, GH200) can raise the ceiling.
    std::uint64_t per_k = 8;
    if (const char* env = std::getenv("ED_AUTO_THREADS_PER_K")) {
        const long long parsed = std::strtoll(env, nullptr, 10);
        if (parsed > 0) per_k = static_cast<std::uint64_t>(parsed);
    }

    int ceil = 8;
    if (const char* env = std::getenv("ED_AUTO_THREADS_CEIL")) {
        const long long parsed = std::strtoll(env, nullptr, 10);
        if (parsed >= 0) ceil = static_cast<int>(parsed);
    }
    // ``ceil <= 0`` disables the soft cap entirely.
    if (ceil <= 0 || ceil > max_t) ceil = max_t;

    const std::uint64_t denom = per_k * 1024ULL;
    if (denom == 0) return std::min(ceil, max_t);

    const std::uint64_t want64 = std::max<std::uint64_t>(1, dim / denom);
    const int want = static_cast<int>(std::min<std::uint64_t>(
        want64, static_cast<std::uint64_t>(max_t)));
    return std::clamp(want, 1, std::min(ceil, max_t));
}

ThreadBudgetScope::ThreadBudgetScope(int threads) {
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

    if (openblas_get_num_threads && openblas_set_num_threads) {
        prev_openblas_ = openblas_get_num_threads();
        if (prev_openblas_ != threads) {
            openblas_set_num_threads(threads);
            restore_blas_ = true;
        }
    }
}

ThreadBudgetScope::~ThreadBudgetScope() {
#ifdef _OPENMP
    if (restore_omp_ && prev_omp_ > 0) {
        omp_set_num_threads(prev_omp_);
    }
#endif
    if (restore_blas_ && prev_openblas_ > 0 &&
        openblas_set_num_threads) {
        openblas_set_num_threads(prev_openblas_);
    }
}

}  // namespace ed::parallel
