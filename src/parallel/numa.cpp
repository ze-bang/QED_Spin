// =============================================================================
// src/parallel/numa.cpp
//
// OpenMP thread-pinning hook. See include/ed/parallel/numa.h
// for the rationale and the env knob.
//
// No external dependencies beyond pthread + OpenMP + glibc -- libnuma is
// intentionally not required.
// =============================================================================

#include "ed/parallel/numa.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
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

}  // anonymous namespace

bool numa_pin_threads_enabled() {
    return parse_bool_env("ED_NUMA_PIN_THREADS");
}

int pin_omp_threads_application_count() {
    return g_pin_application_count.load(std::memory_order_relaxed);
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
#endif  // _OPENMP
        g_pin_application_count.fetch_add(1, std::memory_order_relaxed);
    });
#endif  // __linux__
}

}  // namespace ed::parallel
