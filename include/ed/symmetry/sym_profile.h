#pragma once
// =============================================================================
// include/ed/symmetry/sym_profile.h
//
// Stage 0 of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): make symmetry-construction
// cost visible. ``ED_SYM_PROFILE=1`` prints one stderr line per
// construction phase:
//
//     [sym-profile] <phase>: <seconds> s  (<items> items)
//
// Zero overhead when the env var is unset (one cached bool test per
// scope). This is intentionally stderr-only plumbing -- the phases it
// wraps (rep enumeration, stabilizer table, per-irrep sector build)
// are host-side, single-shot, and upstream of every backend, so a
// process-global text channel is the right weight.
// =============================================================================

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ed::symmetry {

[[nodiscard]] inline bool sym_profile_enabled() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("ED_SYM_PROFILE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

/// RAII phase timer. ``items`` (optional) is reported alongside the
/// wallclock -- pass the rep/sector count once known via ``set_items``.
class SymPhaseTimer {
public:
    explicit SymPhaseTimer(const char* phase) noexcept
        : phase_(phase),
          on_(sym_profile_enabled()),
          t0_(on_ ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{}) {}

    void set_items(std::uint64_t n) noexcept { items_ = n; }

    ~SymPhaseTimer() {
        if (!on_) return;
        const double s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0_)
                             .count();
        if (items_ != kNoItems) {
            std::fprintf(stderr,
                         "[sym-profile] %s: %.3f s  (%llu items)\n",
                         phase_, s,
                         static_cast<unsigned long long>(items_));
        } else {
            std::fprintf(stderr, "[sym-profile] %s: %.3f s\n", phase_, s);
        }
    }

    SymPhaseTimer(const SymPhaseTimer&)            = delete;
    SymPhaseTimer& operator=(const SymPhaseTimer&) = delete;

private:
    static constexpr std::uint64_t kNoItems = ~0ULL;

    const char*                                 phase_;
    bool                                        on_;
    std::chrono::steady_clock::time_point      t0_;
    std::uint64_t                               items_ = kNoItems;
};

}  // namespace ed::symmetry
