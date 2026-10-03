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

#include <ed/core/config.h>
#include <ed/core/log.h>

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace ed::parallel {

namespace {

// Pinning is applied at most once per process: pthread_setaffinity_np is irreversible from the
// application side (the inherited masks would have to be remembered to restore them).
std::mutex g_pin_mutex;
bool g_pin_done = false;   // guarded by g_pin_mutex
std::atomic<int> g_pin_application_count{0};

}  // anonymous namespace

bool numa_pin_threads_enabled() { return ed::env::flag("ED_NUMA_PIN_THREADS", false); }

int pin_omp_threads_application_count() { return g_pin_application_count.load(std::memory_order_relaxed); }

void pin_omp_threads_once() {
    if (!numa_pin_threads_enabled()) return;
#if defined(__linux__) && defined(_OPENMP)
    // Inside a parallel region only the calling team would be pinned, for good: wait for the
    // next call from serial code.
    if (omp_in_parallel()) return;
    std::lock_guard<std::mutex> lock(g_pin_mutex);
    if (g_pin_done) return;
    g_pin_done = true;
    // OMP_PROC_BIND / OMP_PLACES bind the threads already (and bind this one to a single place,
    // so its mask no longer names the process's CPUs): leave them as they are.
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        ED_LOG(Info, "ED_NUMA_PIN_THREADS: OMP_PROC_BIND binds the threads; not pinning them again");
        return;
    }
    // The CPUs this process may run on (its cgroup cpuset or taskset mask): thread t takes the
    // t-th of them, wrapping around.
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        ED_LOG(Warn, "ED_NUMA_PIN_THREADS: sched_getaffinity failed; the threads are not pinned");
        return;
    }
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
    if (cpus.empty()) return;
    std::atomic<int> failed{0};
#pragma omp parallel
    {
        const std::size_t t = static_cast<std::size_t>(omp_get_thread_num());
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpus[t % cpus.size()], &mask);
        if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0)
            failed.fetch_add(1, std::memory_order_relaxed);
    }
    if (failed.load() > 0) {
        ED_LOG(Warn, "ED_NUMA_PIN_THREADS: %d OpenMP thread(s) could not be pinned", failed.load());
        return;
    }
    g_pin_application_count.fetch_add(1, std::memory_order_relaxed);
#endif
}

}  // namespace ed::parallel
