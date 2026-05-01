// =============================================================================
// test_numa (Catch2 v3, Phase 3a #4)
//
// Coverage for the NUMA-aware first-touch + thread-pinning helpers in
// include/ed/parallel/numa.h. Both knobs are default-off; turning them on
// must NEVER change numerical results -- only DRAM page placement and OMP
// thread affinity. So the lockdown is structured as:
//
//   1. Knob defaults: first_touch_enabled() / pin_threads_enabled() return
//      false when neither env var is set.
//   2. Knob parsing: "1" / "true" / "yes" / "on" -> true; any other
//      non-empty value -> false (so "ED_NUMA_PIN_THREADS=foo" doesn't
//      silently turn pinning on).
//   3. first_touch is a no-op when the knob is off (counter unchanged).
//   4. first_touch is a no-op when the buffer is below the
//      kFirstTouchMinBytes threshold even with the knob on (a 1 KB
//      vector is too small to NUMA-shard sensibly).
//   5. first_touch on a large buffer with the knob on: counter +=1,
//      bytes_seen += sizeof(buffer), and the buffer is zeroed afterwards.
//   6. pin_omp_threads_once is idempotent: two calls with the knob on
//      bump the application count by exactly one (process-wide std::call_once).
//   7. End-to-end Lanczos eigenvalue invariance: with knobs on vs off,
//      lanczos_selective_reorth on a 6-site Heisenberg chain produces
//      bit-identical (up to FP roundoff) ground-state eigenvalues.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/parallel/numa.h>
#include <ed/solvers/lanczos.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>

namespace {

// RAII helper to set ED_NUMA_* env vars within a Catch2 SECTION and
// restore them on exit. Mirrors the ScopedCheckpointEnv pattern used by
// test_lanczos_checkpoint.cpp.
struct ScopedNumaEnv {
    bool had_first_touch = false;
    bool had_pin_threads = false;
    std::string saved_first_touch;
    std::string saved_pin_threads;

    ScopedNumaEnv() {
        if (const char* v = std::getenv("ED_NUMA_FIRST_TOUCH")) {
            had_first_touch = true;
            saved_first_touch = v;
        }
        if (const char* v = std::getenv("ED_NUMA_PIN_THREADS")) {
            had_pin_threads = true;
            saved_pin_threads = v;
        }
        unsetenv("ED_NUMA_FIRST_TOUCH");
        unsetenv("ED_NUMA_PIN_THREADS");
    }

    ~ScopedNumaEnv() {
        if (had_first_touch) {
            setenv("ED_NUMA_FIRST_TOUCH", saved_first_touch.c_str(), 1);
        } else {
            unsetenv("ED_NUMA_FIRST_TOUCH");
        }
        if (had_pin_threads) {
            setenv("ED_NUMA_PIN_THREADS", saved_pin_threads.c_str(), 1);
        } else {
            unsetenv("ED_NUMA_PIN_THREADS");
        }
    }
};

}  // namespace

// ============================================================================
// 1. Default-off behaviour
// ============================================================================
TEST_CASE("ed::parallel knobs default to off and parse the standard truthy strings",
          "[numa][knob]") {
    ScopedNumaEnv guard;

    SECTION("both unset") {
        REQUIRE_FALSE(ed::parallel::numa_first_touch_enabled());
        REQUIRE_FALSE(ed::parallel::numa_pin_threads_enabled());
    }

    SECTION("ED_NUMA_FIRST_TOUCH=1") {
        setenv("ED_NUMA_FIRST_TOUCH", "1", 1);
        REQUIRE(ed::parallel::numa_first_touch_enabled());
    }

    SECTION("ED_NUMA_FIRST_TOUCH=true") {
        setenv("ED_NUMA_FIRST_TOUCH", "true", 1);
        REQUIRE(ed::parallel::numa_first_touch_enabled());
    }

    SECTION("ED_NUMA_FIRST_TOUCH=yes") {
        setenv("ED_NUMA_FIRST_TOUCH", "yes", 1);
        REQUIRE(ed::parallel::numa_first_touch_enabled());
    }

    SECTION("ED_NUMA_FIRST_TOUCH=on") {
        setenv("ED_NUMA_FIRST_TOUCH", "on", 1);
        REQUIRE(ed::parallel::numa_first_touch_enabled());
    }

    SECTION("ED_NUMA_FIRST_TOUCH=foo (bogus value -> off, fail-safe)") {
        setenv("ED_NUMA_FIRST_TOUCH", "foo", 1);
        REQUIRE_FALSE(ed::parallel::numa_first_touch_enabled());
    }

    SECTION("empty string -> off") {
        setenv("ED_NUMA_FIRST_TOUCH", "", 1);
        REQUIRE_FALSE(ed::parallel::numa_first_touch_enabled());
    }

    SECTION("ED_NUMA_PIN_THREADS=1") {
        setenv("ED_NUMA_PIN_THREADS", "1", 1);
        REQUIRE(ed::parallel::numa_pin_threads_enabled());
    }
}

// ============================================================================
// 2. first_touch_complex no-op when knob is off
// ============================================================================
TEST_CASE("first_touch_complex is a no-op when ED_NUMA_FIRST_TOUCH is off",
          "[numa][first_touch][off]") {
    ScopedNumaEnv guard;
    REQUIRE_FALSE(ed::parallel::numa_first_touch_enabled());

    const auto before = ed::parallel::describe_numa_state();
    // 1 MB vector -- comfortably above the kFirstTouchMinBytes threshold.
    std::vector<std::complex<double>> big(64 * 1024,
                                          std::complex<double>(7.0, -3.0));
    ed::parallel::first_touch_complex(big.data(), big.size());
    const auto after = ed::parallel::describe_numa_state();

    REQUIRE(after.first_touch_call_count == before.first_touch_call_count);
    REQUIRE(after.first_touch_bytes_seen == before.first_touch_bytes_seen);
    // No-op: payload preserved.
    REQUIRE(big.front() == std::complex<double>(7.0, -3.0));
    REQUIRE(big.back()  == std::complex<double>(7.0, -3.0));
}

// ============================================================================
// 3. first_touch_complex below the threshold is a no-op even with knob on
// ============================================================================
TEST_CASE("first_touch_complex is a no-op for buffers below kFirstTouchMinBytes",
          "[numa][first_touch][threshold]") {
    ScopedNumaEnv guard;
    setenv("ED_NUMA_FIRST_TOUCH", "1", 1);
    REQUIRE(ed::parallel::numa_first_touch_enabled());

    const auto before = ed::parallel::describe_numa_state();
    // 16 KB << 256 KB threshold. Should NOT be touched even with knob on.
    std::vector<std::complex<double>> small(1024,
                                            std::complex<double>(0.5, 0.25));
    REQUIRE(small.size() * sizeof(std::complex<double>) <
            ed::parallel::kFirstTouchMinBytes);

    ed::parallel::first_touch_complex(small.data(), small.size());
    const auto after = ed::parallel::describe_numa_state();

    REQUIRE(after.first_touch_call_count == before.first_touch_call_count);
    REQUIRE(after.first_touch_bytes_seen == before.first_touch_bytes_seen);
    REQUIRE(small.front() == std::complex<double>(0.5, 0.25));
}

// ============================================================================
// 4. first_touch_complex on a large buffer: bumps counters + zeroes payload
// ============================================================================
TEST_CASE("first_touch_complex zeroes the buffer and bumps the counters when active",
          "[numa][first_touch][on]") {
    ScopedNumaEnv guard;
    setenv("ED_NUMA_FIRST_TOUCH", "1", 1);
    REQUIRE(ed::parallel::numa_first_touch_enabled());

    const auto before = ed::parallel::describe_numa_state();
    std::vector<std::complex<double>> big(64 * 1024,
                                          std::complex<double>(11.0, 13.0));
    REQUIRE(big.size() * sizeof(std::complex<double>) >=
            ed::parallel::kFirstTouchMinBytes);

    ed::parallel::first_touch_complex(big.data(), big.size());
    const auto after = ed::parallel::describe_numa_state();

    REQUIRE(after.first_touch_call_count ==
            before.first_touch_call_count + 1);
    REQUIRE(after.first_touch_bytes_seen ==
            before.first_touch_bytes_seen +
                static_cast<std::int64_t>(big.size() *
                                          sizeof(std::complex<double>)));

    // Every element should have been zero-touched.
    for (auto z : big) {
        REQUIRE(z == std::complex<double>(0.0, 0.0));
    }
}

// ============================================================================
// 5. first_touch_bytes parallels first_touch_complex
// ============================================================================
TEST_CASE("first_touch_bytes zeroes raw byte buffers when active",
          "[numa][first_touch][bytes]") {
    ScopedNumaEnv guard;
    setenv("ED_NUMA_FIRST_TOUCH", "1", 1);

    std::vector<unsigned char> buf(512 * 1024, 0xAB);
    REQUIRE(buf.size() >= ed::parallel::kFirstTouchMinBytes);

    ed::parallel::first_touch_bytes(buf.data(), buf.size());

    for (std::size_t i = 0; i < buf.size(); i += 4096) {  // probe one byte per page
        REQUIRE(buf[i] == 0);
    }
    REQUIRE(buf.back() == 0);
}

// ============================================================================
// 6. pin_omp_threads_once is idempotent
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

// ============================================================================
// 7. Knobs do not change Lanczos eigenvalues
// ============================================================================
TEST_CASE("ED_NUMA_FIRST_TOUCH=1 does not change Lanczos ground-state energy",
          "[numa][lanczos_invariance]") {
    ScopedNumaEnv guard;

    using ed_tests::build_heisenberg_chain;
    using ed_tests::reference_from_operator;
    using Complex = std::complex<double>;

    const uint64_t N_sites = 6;
    auto op = build_heisenberg_chain(N_sites, /*J=*/1.0, /*periodic=*/true);
    const uint64_t dim = 1ULL << N_sites;
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };

    auto run_lanczos = [&]() -> std::vector<double> {
        std::vector<double> evals;
        lanczos_selective_reorth(Hv, dim, /*max_iter=*/60, /*exct=*/1,
                                 /*tol=*/1e-10, evals, /*dir=*/"",
                                 /*eigenvectors=*/false);
        return evals;
    };

    // Reference run: knobs OFF.
    auto evals_off = run_lanczos();

    // Repeat with first-touch ON.
    setenv("ED_NUMA_FIRST_TOUCH", "1", 1);
    auto evals_on = run_lanczos();
    unsetenv("ED_NUMA_FIRST_TOUCH");

    REQUIRE(!evals_off.empty());
    REQUIRE(!evals_on.empty());

    // Both runs must produce the SAME ground-state energy (the only
    // eigenvalue Lanczos is guaranteed to converge in this regime). The
    // off-vs-on difference must be at most a few rounding ULPs, since
    // first-touch only changes WHERE the bytes are zero-written and not
    // WHAT they hold by the time Lanczos starts overwriting them.
    std::sort(evals_off.begin(), evals_off.end());
    std::sort(evals_on.begin(),  evals_on.end());
    INFO("E0 OFF = " << evals_off.front()
         << ", E0 ON = "  << evals_on.front()
         << ", |Δ| = "    << std::abs(evals_off.front() - evals_on.front()));
    REQUIRE(std::abs(evals_off.front() - evals_on.front()) < 1e-10);

    // And both must match the dense ground state.
    auto ref = reference_from_operator(*op, dim);
    REQUIRE(!ref.eigs.empty());
    REQUIRE(std::abs(evals_off.front() - ref.eigs.front()) < 1e-9);
    REQUIRE(std::abs(evals_on.front()  - ref.eigs.front()) < 1e-9);
}
