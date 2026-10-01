#pragma once
// =============================================================================
// include/ed/thermal/mtpq_kernel.h
//
// Microcanonical TPQ facade --- thin wrapper around the unified
// `ed::thermal::tpq_kernel<Backend>` (see `tpq_kernel.h`). Draws a
// random seed vector per sample (`tpq_per_sample_seed`, or
// `random_seed + s` when a seed is given), runs the iteration loop on the
// supplied Backend, and accumulates the per-iterate energy expectation.
// The trajectory aggregator lives in `include/ed/thermal/tpq_thermo.h`.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <type_traits>

#include <ed/matvec/backend.h>
#include <ed/matvec/matvec_batcher.h>
#include <ed/thermal/tpq_seeding.h>
#include <ed/thermal/tpq_kernel.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct MtpqOptions {
    std::size_t num_samples    = 1;
    std::size_t max_iter       = 1000;
    double      large_value    = 1.0e5;
    std::uint64_t random_seed  = 0;

    /// Host-side transform applied to every
    /// TPQ sample seed before staging (e.g. the Lowdin total-spin
    /// projection). Must leave a normalisable vector; a zero result
    /// throws (the targeted subspace has no weight in this block).
    std::function<void(Complex*, std::size_t)> seed_transform;

    /// Device multi-vector H: on a CUDA run up to `batch_width` samples advance in lockstep and
    /// share each H apply (see FtlmOptions::batch_matvec).
    ed::LinearOperator::MultiMatvecFn batch_matvec;
    std::size_t batch_width = 8;
};

struct MtpqResult {
    /// Final-iterate energy per sample (an upper bound on the ground-state energy).
    std::vector<double> energies;

    /// Per sample, the trajectory psi_k = (L - H)^k psi_0 / ||.||, psi_0 a unit random vector:
    /// sample_energies[s][k] = <psi_k|H|psi_k> for k = 0..K, and sample_log_norms[s][k - 1] =
    /// ln ||(L - H) psi_{k-1}|| for k = 1..K. They give every moment <psi_0|(L - H)^j|psi_0>
    /// (j <= 2K + 1) the canonical estimator of tpq_thermo.h sums.
    std::vector<std::vector<double>> sample_energies;
    std::vector<std::vector<double>> sample_log_norms;
};

namespace detail {

// Generate a length-N random unit vector on host, deterministic in
// `seed` (Gaussian + L2 normalise).
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
    out.sample_energies.reserve(opts.num_samples);
    out.sample_log_norms.reserve(opts.num_samples);

    // One sample's trajectory; samples are independent and are stored in sample order, however
    // they were run.
    struct Sample { double final_E = 0.0; std::vector<double> Es, log_norms; };
    auto sample = [&](auto& backend, auto&& apply_H, std::size_t s) {
        const std::uint64_t seed = opts.random_seed
                                    ? (opts.random_seed + s)
                                    : ed::tpq_per_sample_seed(s);
        auto host_seed = detail::mtpq_make_seed(local_n, seed);
        // Subspace projection of the TPQ seed (e.g. Lowdin total-spin), renormalised: the
        // moments are those of a unit start vector.
        if (opts.seed_transform) {
            opts.seed_transform(host_seed.data(), local_n);
            double sumsq = 0.0;
            for (const auto& z : host_seed) sumsq += std::norm(z);
            if (!(sumsq > 0.0)) {
                throw std::runtime_error(
                    "mtpq_kernel: the seed transform annihilated sample "
                    + std::to_string(s) + " (the targeted subspace has "
                    "no weight in this block)");
            }
            const double inv = 1.0 / std::sqrt(sumsq);
            for (auto& z : host_seed) z *= inv;
        }

        auto seed_dev = backend.make_zero_vector(local_n);
        backend.copy_from_host(host_seed.data(), seed_dev.get(), local_n);

        TpqKernelOptions kopts;
        kopts.max_iter    = opts.max_iter;
        kopts.large_value = opts.large_value;

        // E_k comes from the step's own H apply; the norm of (L - H) psi_{k-1} before
        // normalisation is the step's growth factor.
        Sample out_s;
        out_s.Es.reserve(opts.max_iter + 1);
        out_s.log_norms.reserve(opts.max_iter);
        auto on_step = [&](const TpqStepInfo<Backend>& info) -> bool {
            const double E_k = info.energy;
            // L must lie above the spectrum: (L - H) is then positive and every moment is too.
            if (!(E_k < opts.large_value))
                throw std::runtime_error("mtpq_kernel: the iterate's energy " + std::to_string(E_k)
                                         + " is not below the shift L = " + std::to_string(opts.large_value)
                                         + " (L must exceed the largest eigenvalue)");
            if (info.step >= 1) {
                if (!(info.norm_before_normalize > 0.0))
                    throw std::runtime_error("mtpq_kernel: (L - H) psi vanished at step "
                                             + std::to_string(info.step));
                out_s.log_norms.push_back(std::log(info.norm_before_normalize));
            }
            out_s.Es.push_back(E_k);
            out_s.final_E = E_k;
            return true;
        };
        auto kres = tpq_kernel<std::decay_t<decltype(backend)>>(backend, apply_H, local_n, seed_dev.get(),
                                                                kopts, on_step);
        (void)kres;
        return out_s;
    };

    std::vector<Sample> samples(opts.num_samples);
    bool batched = false;
#ifdef WITH_CUDA
    if constexpr (std::is_same_v<std::decay_t<Backend>, ed::matvec::CudaBackend>) {
        // Up to batch_width samples in lockstep sharing each H apply (see MatvecBatcher).
        if (opts.batch_matvec && opts.batch_width > 1 && opts.num_samples > 1) {
            batched = true;
            for (std::size_t s0 = 0; s0 < opts.num_samples; s0 += opts.batch_width) {
                const std::size_t k = std::min(opts.batch_width, opts.num_samples - s0);
                ed::matvec::MatvecBatcher b;
                const auto H = b.wrap(opts.batch_matvec);
                b.run(k, [&](std::size_t i) {
                    ed::matvec::CudaBackend be;
                    samples[s0 + i] = sample(be, H, s0 + i);
                });
            }
        }
    }
#endif
    if (!batched)
        for (std::size_t s = 0; s < opts.num_samples; ++s) samples[s] = sample(backend, apply_H, s);
    for (auto& smp : samples) {
        out.energies.push_back(smp.final_E);
        out.sample_energies.push_back(std::move(smp.Es));
        out.sample_log_norms.push_back(std::move(smp.log_norms));
    }
    return out;
}

}  // namespace ed::thermal
