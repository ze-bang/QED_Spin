// tpq_seeding.h — unified per-sample RNG seeding for TPQ (CPU + GPU).
//
// Lives in its own tiny header so that CUDA translation units can include
// it without dragging in Eigen / BLAS through the full TPQ.h.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <string>

namespace ed {

/**
 * @brief Compute the per-sample seed for TPQ random initial states.
 *
 * Behaviour:
 *   - If env var ED_TPQ_BASE_SEED is set to a non-zero unsigned integer S,
 *     the seed is splitmix64(S + sample). This makes both CPU and GPU TPQ
 *     fully reproducible AND identical across implementations for the same
 *     S, which is exactly what cross-validation needs.
 *   - Otherwise (default), the seed is splitmix64(time(NULL)) XOR
 *     splitmix64(sample + 1), matching the legacy "fresh random per run"
 *     behaviour but with better decorrelation between consecutive samples
 *     than the old `time(NULL) + sample` formula (which gives nearly
 *     identical seeds for nearby samples).
 *
 * The splitmix64 mixer guarantees that closely-spaced sample indices map
 * to uncorrelated 64-bit seeds.
 */
inline std::uint64_t tpq_per_sample_seed(std::uint64_t sample) {
    auto splitmix64 = [](std::uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    const char* s = std::getenv("ED_TPQ_BASE_SEED");
    std::uint64_t base = 0;
    if (s && s[0] != '\0') {
        try { base = std::stoull(s); } catch (...) { base = 0; }
    }
    if (base != 0) {
        // Deterministic mode: identical across CPU and GPU for the same base.
        return splitmix64(base + sample);
    }
    // Non-deterministic mode: time-seeded but per-sample decorrelated.
    return splitmix64(static_cast<std::uint64_t>(std::time(nullptr)))
         ^ splitmix64(sample + 1ULL);
}

} // namespace ed
