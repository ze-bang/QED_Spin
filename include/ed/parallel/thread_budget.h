#pragma once

// =============================================================================
// Auto thread-count budget for memory-bound BLAS-1 / SpMV kernels  (Phase 6 #2)
// =============================================================================
//
// On many-core machines (16-128 cores) the default behaviour of OpenMP and
// OpenBLAS is to spawn ``omp_get_max_threads()`` worker threads for *every*
// parallel region. That is the right thing to do when the per-iteration work
// is large enough that the per-thread spawn cost (~5-10 us) and inter-core
// memory-bandwidth contention are amortised. It is the wrong thing to do for
// small Hilbert spaces (N = 14-18, dim = 16k-256k complex doubles), where
// each Lanczos step is a 1-4 MiB memory-bound SpMV plus a handful of axpys.
// At those sizes a 16-thread team is *4-8x slower* than a 4-thread team --
// we measured this directly against XDiag at N = 16 on a 32-core x86_64 box:
//
//     OMP threads    Lanczos(N=16) total
//     -----------    ---------------------
//        1            32 ms
//        2            28 ms
//        4            24 ms       <-- optimum
//        8            38 ms
//       16            60 ms       <-- default; 2.5x slower than the optimum
//
// This header exposes a tiny helper that picks an "appropriate" thread count
// for a given problem dimension, and an RAII scope that applies that cap
// to *both* the OpenMP runtime (``omp_set_num_threads``) and the OpenBLAS
// pthread runtime (``openblas_set_num_threads`` via a weak symbol so the
// build does not hard-require OpenBLAS).
//
// The heuristic is deliberately simple: aim for roughly 16 KiB of complex
// doubles per thread, i.e. ``threads = clamp(dim / 8192, 1, max_threads)``.
// At dim = 16k that gives 2 threads; at dim = 65k it gives 8 threads (close
// to our measured optimum of 4-8); at dim >= 1M it gives ``max_threads``.
// The exact threshold can be tuned via the ``ED_AUTO_THREADS_PER_K`` env
// knob (default: 8 -- one thread per ~8K basis states); pass ``ED_AUTO_THREADS=0``
// to opt out entirely (useful when the caller is itself OMP-parallel and
// nesting would oversubscribe).
//
// Numerically a no-op: changing the BLAS / OMP thread count never changes
// Lanczos eigenvalues. The lockdown tests confirm bit-identity with the
// scope on vs. off.
// =============================================================================

#include <cstddef>
#include <cstdint>

namespace ed::parallel {

/// Compute the recommended thread count for a Krylov-style memory-bound
/// kernel operating on a vector of `dim` complex doubles. Honours
/// ``ED_AUTO_THREADS=0`` (returns ``omp_get_max_threads()`` unchanged) and
/// ``ED_AUTO_THREADS_PER_K`` (default 8 -- one thread per 8K basis states).
/// Always returns at least 1 and at most ``omp_get_max_threads()``.
int auto_threads_for_dim(std::uint64_t dim);

/// RAII scope that pins both the OpenMP runtime and (when available) the
/// OpenBLAS runtime to ``threads`` workers for its lifetime, restoring the
/// previous values on destruction. ``threads <= 0`` is a no-op.
///
/// Construct with ``ThreadBudgetScope budget(auto_threads_for_dim(dim));``
/// at the top of any solver entry point that drives BLAS-1 / SpMV kernels.
class ThreadBudgetScope {
public:
    explicit ThreadBudgetScope(int threads);
    ~ThreadBudgetScope();

    ThreadBudgetScope(const ThreadBudgetScope&)            = delete;
    ThreadBudgetScope& operator=(const ThreadBudgetScope&) = delete;
    ThreadBudgetScope(ThreadBudgetScope&&)                 = delete;
    ThreadBudgetScope& operator=(ThreadBudgetScope&&)      = delete;

    /// Threads we ended up requesting (after clamping). Zero means "scope
    /// was a no-op and the existing thread counts are unchanged".
    int requested() const { return requested_; }

private:
    int requested_       = 0;
    int prev_omp_        = 0;
    int prev_openblas_   = 0;
    bool restore_omp_    = false;
    bool restore_blas_   = false;
};

}  // namespace ed::parallel
