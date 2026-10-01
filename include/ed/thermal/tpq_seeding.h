// tpq_seeding.h — unified per-sample RNG seeding for TPQ (CPU + GPU).
//
// Lives in its own tiny header so that CUDA translation units can include
// it without dragging in Eigen / BLAS through the TPQ kernel headers.

#pragma once

#include <cstdint>
#include <ctime>

namespace ed {

/**
 * @brief Compute the per-sample seed for TPQ random initial states.
 *
 * The seed is splitmix64(time(NULL)) XOR splitmix64(sample + 1): fresh
 * random per run, with closely-spaced sample indices mapped to
 * uncorrelated 64-bit seeds by the splitmix64 mixer.
 */
inline std::uint64_t tpq_per_sample_seed(std::uint64_t sample) {
    auto splitmix64 = [](std::uint64_t z) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    return splitmix64(static_cast<std::uint64_t>(std::time(nullptr)))
         ^ splitmix64(sample + 1ULL);
}

} // namespace ed
