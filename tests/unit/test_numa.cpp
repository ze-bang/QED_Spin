// =============================================================================
// test_numa (Catch2 v3)
//
// Coverage for the OpenMP thread-pinning hook in include/ed/parallel/numa.h.
// The knob is default-off and must never change numerical results -- only
// OMP thread affinity. The lockdown:
//
//   1. Knob default: numa_pin_threads_enabled() is false when the env var
//      is unset.
//   2. Knob parsing: the registry's Flag ("1" / "true" / "yes" / "on" -> on);
//      a value that is not a flag ("foo") is malformed, which every verb
//      refuses before it reads the knob.
//   3. Pinning inside a parallel region waits for serial code; once applied,
//      every thread runs on one CPU of the process's allowed set.
//   4. pin_omp_threads_once is idempotent: repeated calls with the knob on
//      bump the application count by at most one per process.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/config.h>
#include <ed/parallel/numa.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

namespace {

// RAII helper to set ED_NUMA_PIN_THREADS within a Catch2 SECTION and
// restore it on exit.
struct ScopedNumaEnv {
    bool had_pin_threads = false;
    std::string saved_pin_threads;

    ScopedNumaEnv() {
        if (const char* v = std::getenv("ED_NUMA_PIN_THREADS")) {
            had_pin_threads = true;
            saved_pin_threads = v;
        }
        unsetenv("ED_NUMA_PIN_THREADS");
    }

    ~ScopedNumaEnv() {
        if (had_pin_threads) {
            setenv("ED_NUMA_PIN_THREADS", saved_pin_threads.c_str(), 1);
        } else {
            unsetenv("ED_NUMA_PIN_THREADS");
        }
    }
};

}  // namespace

// ============================================================================
// 1-2. Default-off behaviour and knob parsing
// ============================================================================
TEST_CASE("ED_NUMA_PIN_THREADS defaults to off and parses the standard truthy strings",
          "[numa][knob]") {
    ScopedNumaEnv guard;

    SECTION("unset") {
        REQUIRE_FALSE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("ED_NUMA_PIN_THREADS=1") {
        setenv("ED_NUMA_PIN_THREADS", "1", 1);
        REQUIRE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("ED_NUMA_PIN_THREADS=true") {
        setenv("ED_NUMA_PIN_THREADS", "true", 1);
        REQUIRE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("ED_NUMA_PIN_THREADS=yes") {
        setenv("ED_NUMA_PIN_THREADS", "yes", 1);
        REQUIRE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("ED_NUMA_PIN_THREADS=on") {
        setenv("ED_NUMA_PIN_THREADS", "on", 1);
        REQUIRE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("ED_NUMA_PIN_THREADS=foo is malformed (the verbs refuse it)") {
        setenv("ED_NUMA_PIN_THREADS", "foo", 1);
        const auto bad = ed::env::malformed();
        REQUIRE(std::find(bad.begin(), bad.end(), std::string("ED_NUMA_PIN_THREADS=foo")) != bad.end());
    }

    SECTION("empty string -> off") {
        setenv("ED_NUMA_PIN_THREADS", "", 1);
        REQUIRE_FALSE(ed::parallel::numa_pin_threads_enabled());
    }
}

// ============================================================================
// 3. Deferred inside a parallel region; pinned threads stay in the allowed set
// ============================================================================
TEST_CASE("pin_omp_threads_once pins every thread to one CPU of the allowed set",
          "[numa][pin][cpuset]") {
#if defined(__linux__) && defined(_OPENMP)
    ScopedNumaEnv guard;
    setenv("ED_NUMA_PIN_THREADS", "1", 1);
    if (ed::parallel::pin_omp_threads_application_count() != 0) {
        SUCCEED("already pinned in this process");
        return;
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    REQUIRE(sched_getaffinity(0, sizeof(allowed), &allowed) == 0);

    // A call from inside a parallel region pins nothing and leaves the once for later.
    #pragma omp parallel num_threads(2)
    {
        #pragma omp master
        if (omp_get_num_threads() > 1) ed::parallel::pin_omp_threads_once();
    }
    REQUIRE(ed::parallel::pin_omp_threads_application_count() == 0);

    ed::parallel::pin_omp_threads_once();
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        REQUIRE(ed::parallel::pin_omp_threads_application_count() == 0);   // left to the runtime
        return;
    }
    REQUIRE(ed::parallel::pin_omp_threads_application_count() == 1);
    std::vector<int> ok(static_cast<std::size_t>(omp_get_max_threads()), 0);
    #pragma omp parallel
    {
        cpu_set_t mine;
        CPU_ZERO(&mine);
        if (sched_getaffinity(0, sizeof(mine), &mine) == 0 && CPU_COUNT(&mine) == 1) {
            int in = 0;
            for (int c = 0; c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &mine) && CPU_ISSET(c, &allowed)) in = 1;
            const std::size_t t = static_cast<std::size_t>(omp_get_thread_num());
            if (t < ok.size()) ok[t] = in;
        }
    }
    for (std::size_t t = 0; t < ok.size(); ++t) {
        INFO("thread " << t);
        REQUIRE(ok[t] == 1);
    }
#else
    SUCCEED("pinning needs Linux and OpenMP");
#endif
}

// ============================================================================
// 4. pin_omp_threads_once is idempotent
// ============================================================================
TEST_CASE("pin_omp_threads_once is idempotent within a process",
          "[numa][pin][idempotent]") {
    ScopedNumaEnv guard;
    setenv("ED_NUMA_PIN_THREADS", "1", 1);
    REQUIRE(ed::parallel::numa_pin_threads_enabled());

    const int before = ed::parallel::pin_omp_threads_application_count();

    ed::parallel::pin_omp_threads_once();
    const int after_first = ed::parallel::pin_omp_threads_application_count();
    ed::parallel::pin_omp_threads_once();
    ed::parallel::pin_omp_threads_once();
    const int after_third = ed::parallel::pin_omp_threads_application_count();

    // At MOST one application per process. If a previous test already
    // pinned, the count may have been >= 1 before we got here, in which
    // case the bump from this test is zero.
    REQUIRE(after_first - before <= 1);
    REQUIRE(after_third == after_first);  // no extra applications.
}
