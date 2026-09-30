// =============================================================================
// test_numa (Catch2 v3, Phase 3a #4)
//
// Coverage for the OpenMP thread-pinning hook in include/ed/parallel/numa.h.
// The knob is default-off and must never change numerical results -- only
// OMP thread affinity. The lockdown:
//
//   1. Knob default: numa_pin_threads_enabled() is false when the env var
//      is unset.
//   2. Knob parsing: "1" / "true" / "yes" / "on" -> true; any other
//      non-empty value -> false (so "ED_NUMA_PIN_THREADS=foo" doesn't
//      silently turn pinning on).
//   3. pin_omp_threads_once is idempotent: repeated calls with the knob on
//      bump the application count by at most one (process-wide
//      std::call_once).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/parallel/numa.h>

#include <cstdlib>
#include <string>

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

    SECTION("ED_NUMA_PIN_THREADS=foo (bogus value -> off, fail-safe)") {
        setenv("ED_NUMA_PIN_THREADS", "foo", 1);
        REQUIRE_FALSE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("empty string -> off") {
        setenv("ED_NUMA_PIN_THREADS", "", 1);
        REQUIRE_FALSE(ed::parallel::numa_pin_threads_enabled());
    }
}

// ============================================================================
// 3. pin_omp_threads_once is idempotent
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

    // call_once semantics: at MOST one application per process. If a
    // previous test already pinned, the count may have been >= 1 before
    // we got here, in which case the bump from this test is zero.
    REQUIRE(after_first - before <= 1);
    REQUIRE(after_third == after_first);  // no extra applications.
}
