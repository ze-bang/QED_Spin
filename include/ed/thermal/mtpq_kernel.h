#pragma once
// =============================================================================
// include/ed/thermal/mtpq_kernel.h
//
// Microcanonical TPQ facade --- thin wrapper around the unified
// `ed::thermal::tpq_kernel<Backend>` (see `tpq_kernel.h`). Generates a
// rank-deterministic random seed using the historical
// `tpq_per_sample_seed` recipe, runs the iteration loop on the supplied
// Backend, and accumulates the per-iterate energy expectation.
//
// Phase 2.4 of the Minimalist ED Collapse (May 2026): replaces the
// CPU-only `::microcanonical_tpq` forward with a backend-templated
// kernel call. The legacy `::microcanonical_tpq` body in
// `src/solvers/cpu/TPQ.cpp` is retained for the CLI/HDF5 paths but
// will be migrated to a thin HDF5 wrapper over this facade as part of
// the follow-up tightening (tracked in the Phase 2.4 docs).
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/solvers/TPQ.h>
#include <ed/thermal/tpq_kernel.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct MtpqOptions {
    std::size_t num_samples    = 1;
    std::size_t max_iter       = 1000;
    std::size_t temp_interval  = 1;
    double      large_value    = 1.0e5;
    double      target_beta    = 1000.0;
    std::uint64_t random_seed  = 0;
    std::string output_dir;
};

struct MtpqResult {
    std::vector<double> energies;
};

namespace detail {

// Generate a length-N rank-deterministic random unit vector on host
// using the historical TPQ seeding policy (Gaussian + L2 normalise).
inline std::vector<Complex> mtpq_make_seed(std::size_t N, std::uint64_t seed) {
    std::vector<Complex> v(N);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    double sumsq = 0.0;
    for (auto& z : v) {
        const double a = nd(gen), b = nd(gen);
        z = Complex(a, b);
        sumsq += a * a + b * b;
    }
    const double inv = (sumsq > 0.0) ? (1.0 / std::sqrt(sumsq)) : 1.0;
    for (auto& z : v) z *= inv;
    return v;
}

}  // namespace detail

template <typename Backend, typename MatvecFn>
MtpqResult mtpq_kernel(Backend&       backend,
                       MatvecFn&&     apply_H,
                       std::size_t    local_n,
                       std::uint64_t  /*global_n*/,
                       const MtpqOptions& opts)
{
    MtpqResult out;
    out.energies.reserve(opts.num_samples);

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        const std::uint64_t seed = opts.random_seed
                                    ? (opts.random_seed + s)
                                    : ::tpq_per_sample_seed(s);
        auto host_seed = detail::mtpq_make_seed(local_n, seed);

        auto seed_dev = backend.make_zero_vector(local_n);
        backend.copy_from_host(host_seed.data(), seed_dev.get(), local_n);

        TpqKernelOptions kopts;
        kopts.method      = TpqMethod::Microcanonical;
        kopts.max_iter    = opts.max_iter;
        kopts.large_value = opts.large_value;
        kopts.normalize_each_step = true;

        // Capture the final-iterate energy via on_step (only commit the
        // last one).  Energy = <psi|H|psi>, real part.
        double final_E = 0.0;
        auto scratch   = backend.make_zero_vector(local_n);
        auto on_step = [&](const TpqStepInfo<Backend>& info) -> bool {
            apply_H(info.psi, scratch.get(), info.local_n);
            const Complex e = backend.dot(info.psi, scratch.get(), info.local_n);
            final_E = std::real(e);
            return true;
        };
        auto kres = tpq_kernel<Backend>(backend, apply_H, local_n,
                                        seed_dev.get(), kopts, on_step);
        out.energies.push_back(final_E);
    }
    return out;
}

}  // namespace ed::thermal
