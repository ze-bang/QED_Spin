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

#include <ed/core/errors.h>
#include <ed/krylov/lanczos_kernel.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/matvec_batcher.h>
#include <ed/solvers/lanczos.h>      // estimate_spectral_bounds; LAPACKE_dstev
#include <ed/thermal/tpq_seeding.h>
#include <ed/thermal/tpq_kernel.h>
#include <ed/thermal/tpq_thermo.h>

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

/// One mTPQ run of a block, as the thermal verb asks for it.
struct MtpqRun {
    std::size_t   samples = 1;
    /// Steps per sample; 0 sizes them for the coldest beta (mtpq_steps_for), with one retry at
    /// twice the count when the trajectory falls short.
    std::size_t   steps   = 0;
    std::uint64_t seed    = 0;
    std::function<void(Complex*, std::size_t)> seed_transform;
    ed::LinearOperator::MultiMatvecFn batch_matvec;   ///< device: samples share each H apply
};

/// The canonical mTPQ thermodynamics of an n-dimensional block at `betas`. Recipe:
///   1. spectral bounds (E_min, E_max) from a short Lanczos on this backend;
///   2. L just above E_max (the series' terms peak near j* = beta (L - E_min), so L - E_max is
///      pure cost);
///   3. steps so the series converges at the coldest beta: j* + 8 sqrt(j*) terms; a target the
///      trajectory cannot reach is refused, never clamped.
template <typename Backend, typename MatvecFn>
MtpqThermo mtpq(Backend& be, MatvecFn&& H, std::size_t n, const std::vector<double>& betas,
                const MtpqRun& run) {
    std::vector<double> temperatures;
    temperatures.reserve(betas.size());
    for (double b : betas) temperatures.push_back(b > 0.0 ? 1.0 / b : 0.0);
    double beta_max = 0.0;
    for (double b : betas) beta_max = std::max(beta_max, b);
    if (!(beta_max > 0.0)) beta_max = 100.0;

    MtpqOptions kopts;
    kopts.num_samples    = run.samples;
    kopts.random_seed    = run.seed;
    kopts.seed_transform = run.seed_transform;
    kopts.batch_matvec   = run.batch_matvec;

    double e_min_est = 0.0, e_max_est = 0.0;
    bool have_bounds = false;
    if constexpr (std::is_same_v<std::decay_t<Backend>, ed::matvec::CpuBackend>) {
        // The shared Lanczos spectral-bound estimator on a host-pointer wrapper of the matvec.
        std::function<void(const Complex*, Complex*, int)> legacy_H =
            [&H](const std::complex<double>* in, std::complex<double>* out, int m) {
                H(in, out, static_cast<std::size_t>(m));
            };
        const std::uint64_t bdim = n;
        std::mt19937 gen(run.seed ? static_cast<unsigned>(run.seed) : 0x9E3779B9u);
        try {
            const int kry = static_cast<int>(std::min<std::uint64_t>(60, std::max<std::uint64_t>(bdim, 1)));
            ::estimate_spectral_bounds(legacy_H, bdim, kry, /*tol=*/1e-10, gen, e_min_est, e_max_est);
            have_bounds = std::isfinite(e_min_est) && std::isfinite(e_max_est) && e_max_est >= e_min_est;
        } catch (...) {
            have_bounds = false;
        }
    } else {
        // Device lanes: the same estimate from a short Lanczos run on this backend (vectors stay
        // device-resident); the extreme Ritz values of 60 steps.
        const std::uint64_t bdim = n;
        std::vector<Complex> seed_host(bdim);
        std::mt19937_64 gen(run.seed ? run.seed : 0x9E3779B97F4A7C15ULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& z : seed_host) z = Complex(nd(gen), nd(gen));
        auto seed = be.make_zero_vector(bdim);
        be.copy_from_host(seed_host.data(), seed.get(), bdim);
        ed::krylov::LanczosKernelOptions bo;
        bo.max_iter   = static_cast<std::size_t>(std::min<std::uint64_t>(60, std::max<std::uint64_t>(bdim, 1)));
        bo.reorth     = ed::krylov::ReorthPolicy::None;
        bo.keep_basis = false;
        try {
            const auto lk = ed::krylov::lanczos_kernel(be, H, bdim, seed.get(), bo);
            std::vector<double> d = lk.alpha, e;
            for (std::size_t i = 1; i < lk.alpha.size(); ++i) e.push_back(lk.beta[i]);
            e.resize(std::max<std::size_t>(d.size(), 1));
            if (!d.empty() && LAPACKE_dstev(LAPACK_COL_MAJOR, 'N', static_cast<lapack_int>(d.size()),
                                            d.data(), e.data(), nullptr, 1) == 0) {
                e_min_est = *std::min_element(d.begin(), d.end());
                e_max_est = *std::max_element(d.begin(), d.end());
                have_bounds = std::isfinite(e_min_est) && std::isfinite(e_max_est) && e_max_est >= e_min_est;
            }
        } catch (...) {
            have_bounds = false;
        }
    }
    if (!have_bounds)
        throw ed::ConvergenceError("mTPQ: the spectral bounds of the block could not be estimated");
    // L just above the spectrum. The margin covers the Lanczos estimate of E_max, a lower bound on it.
    const double W = e_max_est - e_min_est;
    const double L = e_max_est + std::max({0.05 * W, 1e-6 * std::max(1.0, std::abs(e_max_est)), 1e-9});
    kopts.large_value = L;

    constexpr std::size_t MTPQ_HARD_CAP = 200000;
    const bool auto_steps = run.steps == 0;
    std::size_t steps = auto_steps ? mtpq_steps_for(beta_max, L, e_min_est) : run.steps;
    if (steps > MTPQ_HARD_CAP)
        throw ed::ResourceLimit("mTPQ: T_min = " + std::to_string(1.0 / beta_max) + " needs "
                                + std::to_string(steps) + " steps per sample (cap "
                                + std::to_string(MTPQ_HARD_CAP) + "); ask for a warmer T_min");
    for (int attempt = 0;; ++attempt) {
        kopts.max_iter = std::max<std::size_t>(steps, 1);
        MtpqResult kres = mtpq_kernel<Backend>(be, H, n, kopts);
        if (temperatures.empty()) return MtpqThermo{};
        MtpqThermo mt = mtpq_canonical_thermo(kres.sample_energies, kres.sample_log_norms, L, temperatures,
                                              static_cast<double>(n));
        if (mt.unconverged.empty()) return mt;
        // Too cold for the trajectory. An auto-sized run had underestimated the spectral range:
        // run once more with twice the steps. Never clamp.
        double T_reached = std::numeric_limits<double>::infinity();
        for (std::size_t t = 0; t < temperatures.size(); ++t)
            if (std::find(mt.unconverged.begin(), mt.unconverged.end(), t) == mt.unconverged.end())
                T_reached = std::min(T_reached, temperatures[t]);
        if (auto_steps && attempt == 0 && 2 * steps <= MTPQ_HARD_CAP) {
            steps *= 2;
            continue;
        }
        throw ed::ConvergenceError(
            "mTPQ: " + std::to_string(steps) + " steps reach T = "
            + (std::isfinite(T_reached) ? std::to_string(T_reached) : std::string("none of the targets"))
            + " but the grid asks for T = " + std::to_string(1.0 / beta_max)
            + (auto_steps ? std::string("") : std::string("; raise krylov or leave it unset")));
    }
}

}  // namespace ed::thermal
