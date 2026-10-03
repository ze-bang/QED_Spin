#pragma once
// =============================================================================
// include/ed/core/interrupt.h -- a cooperative interrupt for long solves.
//
// The Python bindings install a check that raises KeyboardInterrupt when a signal is
// pending (PyErr_CheckSignals under the GIL); the engine polls it between blocks and at
// every Krylov / mTPQ step. A poll runs the check only on the thread that installed it and
// outside OpenMP parallel regions (an exception there would terminate the process), at most
// every 50 ms per thread. Without an installed check (C++ callers, tests) a poll is a no-op.
// =============================================================================

#include <chrono>
#include <functional>
#include <thread>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::core {

struct InterruptHook {
    std::function<void()> check;   ///< throws to interrupt; empty = none installed
    std::thread::id owner;   ///< the only thread that runs it
};

inline InterruptHook& interrupt_hook() {
    static InterruptHook h;
    return h;
}

/// Install (or, with an empty function, clear) the check; the calling thread owns it.
inline void set_interrupt_check(std::function<void()> check) {
    auto& h = interrupt_hook();
    h.check = std::move(check);
    h.owner = std::this_thread::get_id();
}

/// Run the installed check if this is its thread, outside a parallel region, and 50 ms have
/// passed since this thread last ran it. May throw (whatever the check throws).
inline void poll_interrupt() {
    auto& h = interrupt_hook();
    if (!h.check) return;
#ifdef _OPENMP
    if (omp_in_parallel()) return;
#endif
    if (std::this_thread::get_id() != h.owner) return;
    static thread_local std::chrono::steady_clock::time_point last{};
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::milliseconds(50)) return;
    last = now;
    h.check();
}

}  // namespace ed::core
