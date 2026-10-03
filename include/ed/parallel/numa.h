#pragma once

// =============================================================================
// OpenMP thread-pinning hook
// =============================================================================
//
// On a multi-socket node the basis-sized vectors of a Lanczos run land on the
// NUMA node of the thread that first writes them (Linux first-touch). Pinning
// the OpenMP workers to cores once per process keeps "this thread owns this
// chunk" true across the parallel regions that follow.
//
// This header intentionally avoids a libnuma dependency. The knob is
// DEFAULT-OFF and never changes numerical results, only thread affinity.
//
// Env knob (re-read on each call so unit tests can flip it between
// invocations; parsed as a registry Flag, see <ed/core/config.h>):
//
//   ED_NUMA_PIN_THREADS       0 (off, default) | 1 (on)
//       When on, the first `pin_omp_threads_once()` called from serial code
//       binds OpenMP thread t to the t-th CPU the process may run on (its
//       cgroup cpuset or taskset mask, wrapping around) via
//       `pthread_setaffinity_np`. A call inside a parallel region waits for
//       the next one. When OMP_PROC_BIND binds the threads already, they are
//       left as they are. Failures are logged (Warn) and not counted.
// =============================================================================

namespace ed::parallel {

/// ED_NUMA_PIN_THREADS as a registry Flag (default false). Re-reads each call.
bool numa_pin_threads_enabled();

/// The number of times the per-process pinning has been *applied* (zero: not
/// pinned yet, the knob is off, OMP_PROC_BIND binds the threads, or pinning
/// failed). Test-friendly accessor.
int pin_omp_threads_application_count();

/// Apply OpenMP thread-to-CPU pinning once per process (see the knob above).
/// Repeat calls are no-ops; on non-Linux platforms, without OpenMP or with the
/// knob off, the call is a silent no-op. Called at the top of every verb.
void pin_omp_threads_once();

}  // namespace ed::parallel
