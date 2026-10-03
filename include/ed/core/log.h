// =============================================================================
// include/ed/core/log.h
//
// The library's one console channel. Nothing in include/ or src/ writes to
// stdout or stderr directly; every message goes through ed::logging::write into
// one process-wide sink (scripts/check_no_print.sh enforces it).
//
//   level  Off < Error < Warn (default) < Info < Debug.
//          ED_LOG(Info, "fmt", ...) (printf style) and ED_LOGS(Info, a << b)
//          (stream style) format only when the level is enabled: a disabled
//          message costs one relaxed atomic load.
//   sink   the default holds records in a bounded queue until drain() hands
//          them to the caller -- the Python bindings replay them into
//          logging.getLogger("qed") on the calling thread after a verb returns,
//          so no engine thread ever calls into Python. set_stream(stderr) writes
//          each record at once instead (live progress); set_sink() installs any
//          other callable.
//
// Thread-safe: engine and OpenMP threads may log; the sink runs under one mutex.
// =============================================================================
#pragma once

#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ed::logging {

enum class Level : int { Off = 0, Error = 1, Warn = 2, Info = 3, Debug = 4 };

struct Record {
    Level       level;
    std::string message;
};

using Sink = std::function<void(Level, const std::string&)>;

[[nodiscard]] inline const char* name(Level l) noexcept {
    switch (l) {
        case Level::Error: return "error";
        case Level::Warn:  return "warn";
        case Level::Info:  return "info";
        case Level::Debug: return "debug";
        default:           return "off";
    }
}

namespace detail {
struct State {
    std::atomic<int>    level{static_cast<int>(Level::Warn)};
    std::mutex          mtx;
    Sink                sink;                 ///< empty: the queue
    std::deque<Record>  queue;
    std::size_t         capacity = 10000;     ///< records kept until drain(); later ones are counted
    std::size_t         dropped  = 0;
};
[[nodiscard]] inline State& state() {
    static State& s = *new State;   // never destroyed: objects torn down at exit may still log
    return s;
}
}  // namespace detail

[[nodiscard]] inline Level level() noexcept {
    return static_cast<Level>(detail::state().level.load(std::memory_order_relaxed));
}

inline void set_level(Level l) noexcept {
    detail::state().level.store(static_cast<int>(l), std::memory_order_relaxed);
}

[[nodiscard]] inline bool enabled(Level l) noexcept {
    return l != Level::Off
        && static_cast<int>(l) <= detail::state().level.load(std::memory_order_relaxed);
}

/// Route records to `s`; an empty Sink restores the queue.
inline void set_sink(Sink s) {
    auto& st = detail::state();
    std::lock_guard<std::mutex> lk(st.mtx);
    st.sink = std::move(s);
}

/// Write each record to `f` (stdout or stderr) as it arrives; nullptr restores the queue.
inline void set_stream(std::FILE* f) {
    if (!f) { set_sink({}); return; }
    set_sink([f](Level l, const std::string& m) {
        std::fprintf(f, "[qed %s] %s\n", name(l), m.c_str());
        std::fflush(f);
    });
}

inline void write(Level l, std::string msg) {
    auto& st = detail::state();
    std::lock_guard<std::mutex> lk(st.mtx);
    if (st.sink) { st.sink(l, msg); return; }
    if (st.queue.size() >= st.capacity) { ++st.dropped; return; }
    st.queue.push_back({l, std::move(msg)});
}

/// The queued records, oldest first, and an empty queue. Overflow is reported as one
/// trailing Warn record.
[[nodiscard]] inline std::vector<Record> drain() {
    auto& st = detail::state();
    std::lock_guard<std::mutex> lk(st.mtx);
    std::vector<Record> out(std::make_move_iterator(st.queue.begin()),
                            std::make_move_iterator(st.queue.end()));
    st.queue.clear();
    if (st.dropped) {
        out.push_back({Level::Warn, std::to_string(st.dropped) + " log record(s) dropped: the queue "
                       "holds " + std::to_string(st.capacity) + "; log to a stream for unbounded output"});
        st.dropped = 0;
    }
    return out;
}

[[nodiscard]]
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline std::string format(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string out;
    if (n > 0) {
        out.resize(static_cast<std::size_t>(n) + 1);
        std::vsnprintf(out.data(), out.size(), fmt, ap2);
        out.resize(static_cast<std::size_t>(n));
    }
    va_end(ap2);
    return out;
}

}  // namespace ed::logging

#define ED_LOG(lvl, ...)                                                              \
    do {                                                                              \
        if (::ed::logging::enabled(::ed::logging::Level::lvl))                        \
            ::ed::logging::write(::ed::logging::Level::lvl,                           \
                                 ::ed::logging::format(__VA_ARGS__));                 \
    } while (0)

#define ED_LOGS(lvl, expr)                                                            \
    do {                                                                              \
        if (::ed::logging::enabled(::ed::logging::Level::lvl)) {                      \
            std::ostringstream ed_log_os_;                                            \
            ed_log_os_ << expr;                                                       \
            ::ed::logging::write(::ed::logging::Level::lvl, ed_log_os_.str());        \
        }                                                                             \
    } while (0)
