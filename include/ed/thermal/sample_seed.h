#pragma once
// =============================================================================
// include/ed/thermal/sample_seed.h
//
// The random-vector stream of the sampled methods (FTLM, OFTLM, mTPQ, FTLM
// dynamics): one reproducible engine per sample, derived from the caller's base
// seed, and one Gaussian draw. One definition, so every lane that samples draws
// the same vectors for the same seed. Header-only and host-only (the vectors
// are drawn on the host and staged on the backend).
// =============================================================================

#include <ed/core/lapack.h>
#include <ed/core/errors.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

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

// The base seed of one block of a call (a dynamics source sector) from the call's base seed and the
// block's identity `key`: splitmix64 of base + phi * (key + 1). A block then draws the same samples
// whichever other blocks the call holds and in whatever order it runs them.
[[nodiscard]] inline std::uint64_t block_seed(std::uint64_t base_seed, std::uint64_t key) {
    std::uint64_t z = base_seed + 0x9E3779B97F4A7C15ULL * (key + 1);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// The stream of a run's auxiliary draw (mTPQ's spectral-bound start): the one before sample 0,
// so it never coincides with a sample's.
inline constexpr std::uint64_t kAuxStream = ~std::uint64_t{0};

// Scale a host vector to unit norm; returns the norm it had (0: left as it was). dznrm2 + zscal
// up to 2^31 - 1 entries (the 32-bit BLAS index; the draws of every block that size stay
// bit-identical), a 64-bit loop above.
inline double normalize_host(std::complex<double>* v, std::size_t n) {
    double norm = 0.0;
    if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        norm = cblas_dznrm2(static_cast<int>(n), v, 1);   // narrow-ok: n <= INT_MAX here
        if (!(norm > 0.0)) return 0.0;
        const std::complex<double> s(1.0 / norm, 0.0);
        cblas_zscal(static_cast<int>(n), &s, v, 1);       // narrow-ok: n <= INT_MAX here
        return norm;
    }
    double n2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) n2 += std::norm(v[i]);
    norm = std::sqrt(n2);
    if (!(norm > 0.0)) return 0.0;
    const double inv = 1.0 / norm;
    for (std::size_t i = 0; i < n; ++i) v[i] *= inv;
    return norm;
}

// A unit vector of n i.i.d. standard complex Gaussians (real and imaginary parts N(0, 1), drawn
// as Complex(nd(gen), nd(gen)) from a fresh distribution), normalised by normalize_host: an
// isotropic draw on the complex unit sphere, the Hutchinson / Jaklic-Prelovsek trace estimator.
[[nodiscard]] inline std::vector<std::complex<double>> gaussian_vector(std::size_t n, std::mt19937& gen) {
    std::normal_distribution<double> ndist(0.0, 1.0);
    std::vector<std::complex<double>> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = std::complex<double>(ndist(gen), ndist(gen));
    (void)normalize_host(v.data(), n);
    return v;
}

}  // namespace ed::thermal
