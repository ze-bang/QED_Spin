#pragma once

// =============================================================================
// NUMA-aware first-touch + thread-pinning hooks                  (Phase 3a #4)
// =============================================================================
//
// Single-node NUMA bandwidth is the silent killer for ED at N >= 36. Every
// Lanczos step touches the basis-sized vectors v_curr / v_prev / v_next / w
// plus the streamed Krylov basis (or its tile copy in re-orth). On a 2-socket
// system, if those buffers were allocated on whichever NUMA node the
// allocating thread happened to be on, every OpenMP worker thread later has
// to fetch them across the inter-socket link -- typically a 2-4x bandwidth
// hit on the dominant SpMV / dot / axpy kernels.
//
// Linux uses a *first-touch* page placement policy by default: the page is
// mapped to the NUMA node of the thread that first writes to it. So the
// minimum-effort fix is:
//
//   1. After allocating a basis-sized buffer, run a parallel zero-write
//      (`first_touch`) so each OpenMP worker writes the chunk it will
//      later own. Zero is irrelevant -- what matters is the page fault.
//
//   2. Pin OpenMP threads to cores once per process so step 1's "this
//      thread later owns this chunk" assumption actually holds across
//      subsequent parallel regions.
//
// This header intentionally avoids a libnuma dependency. The first cut works
// on any Linux kernel + glibc; explicit `numa_alloc_onnode` / `mbind`
// scheduling can be added later as a follow-up if profiling justifies it.
//
// Both knobs are DEFAULT-OFF. Turning them on never changes numerical
// results -- they only change page placement and thread affinity. The
// lockdown tests verify Lanczos eigenvalues are bit-identical with knobs
// on vs. off.
//
// Env knobs (re-read on each call so unit tests can flip them between
// invocations):
//
//   ED_NUMA_FIRST_TOUCH       0 (off, default) | 1 (on)
//       When 1, `first_touch(p, count)` walks the buffer in OpenMP
//       parallel and writes a zero to every element. Below the
//       FIRST_TOUCH_MIN_BYTES threshold (~256 KB by default) the call is
//       a no-op -- pinning a tiny scratch vector to one node hurts more
//       than it helps.
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

#include <cstddef>
#include <cstdint>

namespace ed::parallel {

/// Threshold below which `first_touch` is a no-op. Pinning a 256 KB scratch
/// vector to one NUMA node by parallel-touch is pointless -- the buffer fits
/// in L2 and never spills to DRAM. Public so tests can see the same
/// constant.
constexpr std::size_t kFirstTouchMinBytes = 256 * 1024;  // 256 KB

/// Read ED_NUMA_FIRST_TOUCH (default false). Re-reads each call.
bool numa_first_touch_enabled();

/// Read ED_NUMA_PIN_THREADS (default false). Re-reads each call.
bool numa_pin_threads_enabled();

/// Returns the number of times the per-process pinning has been *applied*
/// (zero means "not pinned yet, or knob is off"). Test-friendly accessor.
int  pin_omp_threads_application_count();

/// Parallel zero-write of `count` complex doubles starting at `data`.
///
/// On Linux + first-touch policy, this fixes the ownership of each page to
/// the OpenMP thread that touches it. Subsequent parallel SpMV / dot / axpy
/// over the same buffer (with the same OpenMP schedule) will hit local
/// DRAM instead of the inter-socket interconnect.
///
/// Behaviour:
///   * No-op if ED_NUMA_FIRST_TOUCH is off.
///   * No-op if `count * sizeof(Complex) < kFirstTouchMinBytes`.
///   * Otherwise: `#pragma omp parallel for schedule(static)` writing
///     `Complex(0.0, 0.0)` to every element. Schedule is `static` so the
///     same thread sees the same chunk on later parallel iterations
///     (most BLAS implementations also use static for `zaxpy`/`zscal`).
///
/// `Complex` is `std::complex<double>` -- declared as a void* to avoid
/// pulling `<complex>` into this header. Callers static_cast to T* on the
/// way in.
void first_touch_complex(void* data, std::size_t count);

/// Generic first-touch for arbitrary trivially-zeroable types (int, float,
/// double, ...). Same gating as `first_touch_complex`. Provided so the
/// CSR `int*` index arrays can also be first-touched without a wrapper.
void first_touch_bytes(void* data, std::size_t bytes);

/// Apply OpenMP thread-to-CPU pinning once per process.
///
/// First call (with ED_NUMA_PIN_THREADS=1) enters `#pragma omp parallel`
/// and calls `pthread_setaffinity_np` on each worker thread to bind it to
/// a single CPU. Subsequent calls are no-ops (idempotent within a
/// process). On non-Linux platforms or if the env knob is off, the call
/// is a silent no-op.
///
/// Safe to call from anywhere -- typically invoked at the top of
/// `lanczos()` / `lanczos_selective_reorth()` / FTLM samplers, just
/// before the first big OpenMP region.
void pin_omp_threads_once();

/// Diagnostic snapshot used by tests + the verbose Lanczos output.
struct NumaInfo {
    int          num_omp_threads;        ///< `omp_get_max_threads()` snapshot.
    int          num_logical_cpus;       ///< from `sysconf(_SC_NPROCESSORS_ONLN)`.
    bool         first_touch_active;     ///< ED_NUMA_FIRST_TOUCH=1 right now.
    bool         threads_pinned;         ///< pin_omp_threads_once() already ran.
    std::int64_t first_touch_call_count; ///< monotonic counter (test hook).
    std::int64_t first_touch_bytes_seen; ///< sum of bytes touched (test hook).
};

NumaInfo describe_numa_state();

}  // namespace ed::parallel
