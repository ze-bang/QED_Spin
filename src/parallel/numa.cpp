// =============================================================================
// src/parallel/numa.cpp
//
// NUMA first-touch + thread-pinning hooks (Phase 3a #4). See
// include/ed/parallel/numa.h for the design rationale and env knobs.
//
// This implementation has zero external dependencies beyond pthread + OpenMP
// + glibc -- libnuma is intentionally not required. The first-touch helper
// uses the kernel's default first-touch placement policy (each page goes to
// the NUMA node of the thread that first writes to it), and the pinning
// helper uses pthread_setaffinity_np on the OpenMP worker threads.
// =============================================================================

#include "ed/parallel/numa.h"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace ed::parallel {

using Complex = std::complex<double>;

namespace {

bool parse_bool_env(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    // Accept the common spellings; any other non-empty value -> false to
    // keep "ED_NUMA_PIN_THREADS=foo" from silently turning the knob on.
    if (v[0] == '1') return true;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

// Process-wide pinning state. We only ever apply pinning *once* per
// process: pthread_setaffinity_np is irreversible from the application
// side (we'd have to remember the inherited mask to restore it), and
// repeated calls would just thrash the kernel's scheduler bookkeeping.
std::once_flag g_pin_once_flag;
std::atomic<int> g_pin_application_count{0};
std::atomic<bool> g_threads_are_pinned{false};

// Test-friendly counters. Atomic so they're safe under concurrent
// first_touch calls (which can happen if FTLM samples each call into
// Lanczos in parallel).
std::atomic<std::int64_t> g_first_touch_calls{0};
std::atomic<std::int64_t> g_first_touch_bytes{0};

}  // anonymous namespace

bool numa_first_touch_enabled() {
    return parse_bool_env("ED_NUMA_FIRST_TOUCH");
}

bool numa_pin_threads_enabled() {
    return parse_bool_env("ED_NUMA_PIN_THREADS");
}

int pin_omp_threads_application_count() {
    return g_pin_application_count.load(std::memory_order_relaxed);
}

void first_touch_bytes(void* data, std::size_t bytes) {
    if (data == nullptr || bytes == 0) return;
    if (!numa_first_touch_enabled()) return;
    if (bytes < kFirstTouchMinBytes) return;

    g_first_touch_calls.fetch_add(1, std::memory_order_relaxed);
    g_first_touch_bytes.fetch_add(static_cast<std::int64_t>(bytes),
                                  std::memory_order_relaxed);

    // memset is fine for the touch -- we only need each page to be written
    // by exactly one thread under a static schedule. Using memset (instead
    // of a hand-rolled byte loop) lets the compiler emit AVX-512 stores
    // and lets each thread move at full DRAM bandwidth.
    auto* p = static_cast<unsigned char*>(data);
    const std::size_t total = bytes;
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (max_threads <= 1) {
        std::memset(p, 0, total);
        return;
    }
    // Static schedule across thread chunks of contiguous bytes. The chunk
    // is rounded up to a 4 KB page boundary so two threads never share a
    // single page (otherwise the page belongs to whichever thread happened
    // to fault first, defeating the purpose).
    constexpr std::size_t page = 4096;
    const std::size_t per_thread =
        ((total + max_threads - 1) / max_threads + page - 1) / page * page;
    #pragma omp parallel num_threads(max_threads)
    {
        const int tid = omp_get_thread_num();
        const std::size_t lo = std::min(total, static_cast<std::size_t>(tid)
                                                   * per_thread);
        const std::size_t hi = std::min(total, lo + per_thread);
        if (hi > lo) {
            std::memset(p + lo, 0, hi - lo);
        }
    }
#else
    std::memset(p, 0, total);
#endif
}

void first_touch_complex(void* data, std::size_t count) {
    first_touch_bytes(data, count * sizeof(Complex));
}

void pin_omp_threads_once() {
    if (!numa_pin_threads_enabled()) return;
#if !defined(__linux__)
    // pthread_setaffinity_np is glibc/Linux only. Other platforms get a
    // silent no-op so the rest of the build stays portable.
    return;
#else
    std::call_once(g_pin_once_flag, []() {
#ifdef _OPENMP
        const int max_threads = omp_get_max_threads();
        const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        if (max_threads <= 0 || ncpu <= 0) return;
        // Compact mapping: thread t -> CPU (t mod ncpu). On a typical
        // 2-socket box with hyperthreads, this lands threads 0..N-1 on
        // contiguous logical cores -- callers who want a different
        // layout (spread across sockets, etc.) should set OMP_PROC_BIND
        // / OMP_PLACES via the environment, which OpenMP applies before
        // we ever get here.
        #pragma omp parallel num_threads(max_threads)
        {
            const int tid = omp_get_thread_num();
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(static_cast<size_t>(tid % ncpu), &mask);
            // Best effort: ignore the return code. If pinning fails (e.g.
            // we're inside a cgroup that already restricts the mask), we
            // don't want to bring the solver down.
            (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                         &mask);
        }
        g_threads_are_pinned.store(true, std::memory_order_release);
#endif  // _OPENMP
        g_pin_application_count.fetch_add(1, std::memory_order_relaxed);
    });
#endif  // __linux__
}

NumaInfo describe_numa_state() {
    NumaInfo out{};
#ifdef _OPENMP
    out.num_omp_threads = omp_get_max_threads();
#else
    out.num_omp_threads = 1;
#endif
#if defined(__linux__)
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    out.num_logical_cpus = (n > 0) ? static_cast<int>(n) : 1;
#else
    out.num_logical_cpus = 1;
#endif
    out.first_touch_active = numa_first_touch_enabled();
    out.threads_pinned = g_threads_are_pinned.load(std::memory_order_acquire);
    out.first_touch_call_count =
        g_first_touch_calls.load(std::memory_order_relaxed);
    out.first_touch_bytes_seen =
        g_first_touch_bytes.load(std::memory_order_relaxed);
    return out;
}

}  // namespace ed::parallel
