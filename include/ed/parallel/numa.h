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
// invocations):
//
//   ED_NUMA_PIN_THREADS       0 (off, default) | 1 (on)
//       When 1, `pin_omp_threads_once()` enters a `#pragma omp parallel`
//       region on first call and binds each worker thread to a single
//       hardware logical CPU via `pthread_setaffinity_np`. Pinning is
//       idempotent: repeat calls are silent no-ops within a process.
//       The mapping is the simple compact one (thread_num -> cpu
//       thread_num); honour OMP_PROC_BIND / OMP_PLACES if you need a
//       different layout.
// =============================================================================

namespace ed::parallel {

/// Read ED_NUMA_PIN_THREADS (default false). Re-reads each call.
bool numa_pin_threads_enabled();

/// Returns the number of times the per-process pinning has been *applied*
/// (zero means "not pinned yet, or knob is off"). Test-friendly accessor.
int  pin_omp_threads_application_count();

/// Apply OpenMP thread-to-CPU pinning once per process.
///
/// First call (with ED_NUMA_PIN_THREADS=1) enters `#pragma omp parallel`
/// and calls `pthread_setaffinity_np` on each worker thread to bind it to
/// a single CPU. Subsequent calls are no-ops (idempotent within a
/// process). On non-Linux platforms or if the env knob is off, the call
/// is a silent no-op.
///
/// Safe to call from anywhere -- typically invoked at the top of a solve /
/// thermal entry point, just before the first big OpenMP region.
void pin_omp_threads_once();

}  // namespace ed::parallel
