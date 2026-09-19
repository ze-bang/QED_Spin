#pragma once
// =============================================================================
// include/ed/thermal/sample_seed.h
//
// The random-vector stream of the sampled thermal methods (FTLM, LTLM, OFTLM):
// one reproducible engine per sample, derived from the caller's base seed. One
// definition, so every lane that samples draws the same vectors for the same
// seed. Header-only and host-only (the engines feed host-side vector draws).
// =============================================================================

#include <cstdint>
#include <random>

namespace ed::thermal {

// A base seed of 0 means "nondeterministic": draw one from std::random_device.
// Any other value is used verbatim.
[[nodiscard]] inline std::uint64_t resolve_base_seed(std::uint64_t seed) {
    if (seed != 0) return seed;
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
}

// The engine of sample `sample` (0-based): splitmix64 of base + phi * (sample + 1),
// fed to std::mt19937 (which keeps the low 32 bits). Samples are independent of
// one another and of the thread that draws them.
[[nodiscard]] inline std::mt19937 sample_engine(std::uint64_t base_seed, std::uint64_t sample) {
    std::uint64_t z = base_seed + 0x9E3779B97F4A7C15ULL * (sample + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return std::mt19937(static_cast<std::mt19937::result_type>(z));
}

}  // namespace ed::thermal
