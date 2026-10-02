#pragma once
// =============================================================================
// include/ed/thermal/mtpq.h
//
// Microcanonical TPQ on any Backend. Sample s starts from
// gaussian_vector(n, sample_engine(base, s)) (sample_seed.h) and iterates
// psi_{k+1} = (L - H) psi_k / ||(L - H) psi_k||, recording E_k = <psi_k|H|psi_k>
// and the growth factors; `mtpq` sizes the run (spectral bounds, L, steps) and
// turns the trajectories into canonical curves (tpq_thermo.h).
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
#include <ed/krylov/lanczos.h>
#include <ed/krylov/tridiag.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/batcher.h>
#include <ed/thermal/sample_seed.h>
#include <ed/thermal/tpq_thermo.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct MtpqOptions {
    std::size_t num_samples    = 1;
    std::size_t max_iter       = 1000;
    double      large_value    = 1.0e5;
    /// Base seed of the sample engines; 0 draws one (resolve_base_seed).
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

template <typename Backend, typename MatvecFn>
MtpqResult mtpq_kernel(Backend&       backend,
                       MatvecFn&&     apply_H,
                       std::size_t    local_n,
                       const MtpqOptions& opts)
{
    if (local_n == 0) throw std::invalid_argument("mtpq_kernel: empty block");
    MtpqResult out;
    out.energies.reserve(opts.num_samples);
    out.sample_energies.reserve(opts.num_samples);
    out.sample_log_norms.reserve(opts.num_samples);
    const std::uint64_t base_seed = resolve_base_seed(opts.random_seed);
    const double L = opts.large_value;

    // One sample's trajectory; samples are independent and are stored in sample order, however
    // they were run.
    struct Sample { std::vector<double> Es, log_norms; };
    auto sample = [&](auto& be, auto&& apply_H, std::size_t s) {
        std::mt19937 gen = sample_engine(base_seed, s);
        std::vector<Complex> host_seed = gaussian_vector(local_n, gen);
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
        auto psi  = be.make_zero_vector(local_n);
        auto hpsi = be.make_zero_vector(local_n);
        be.copy_from_host(host_seed.data(), psi.get(), local_n);
        {
            const double n0 = be.nrm2(psi.get(), local_n);
            if (n0 > 0.0) be.scale(Complex(1.0 / n0, 0.0), psi.get(), local_n);
        }
        Sample out_s;
        out_s.Es.reserve(opts.max_iter + 1);
        out_s.log_norms.reserve(opts.max_iter);
        // One H apply per step: hpsi = H psi_k gives E_k and, below, psi_{k+1} = (L - H) psi_k
        // (one axpby and a pointer swap).
        auto energy = [&](std::size_t k) {
            apply_H(psi.get(), hpsi.get(), local_n);
            const double E = std::real(be.dot(psi.get(), hpsi.get(), local_n));
            // L must lie above the spectrum: (L - H) is then positive and every moment is too.
            if (!(E < L))
                throw ed::ConvergenceError("mtpq_kernel: the energy " + std::to_string(E) + " of step "
                                           + std::to_string(k) + " is not below the shift L = "
                                           + std::to_string(L) + " (L must exceed the largest eigenvalue)");
            out_s.Es.push_back(E);
        };
        energy(0);
        for (std::size_t k = 1; k <= opts.max_iter; ++k) {
            ed::core::poll_interrupt();
            be.axpby(Complex(L, 0.0), psi.get(), Complex(-1.0, 0.0), hpsi.get(), local_n);
            std::swap(psi, hpsi);
            const double nrm = be.nrm2(psi.get(), local_n);
            if (!(nrm > 0.0))
                throw ed::ConvergenceError("mtpq_kernel: (L - H) psi vanished at step " + std::to_string(k));
            be.scale(Complex(1.0 / nrm, 0.0), psi.get(), local_n);
            out_s.log_norms.push_back(std::log(nrm));
            energy(k);
        }
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
        out.energies.push_back(smp.Es.back());
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
    /// Base seed of the run (0 draws one): the bound estimate and every sample derive from it.
    std::uint64_t seed    = 0;
    std::function<void(Complex*, std::size_t)> seed_transform;
    ed::LinearOperator::MultiMatvecFn batch_matvec;   ///< device: samples share each H apply
};

/// The canonical mTPQ curves (ln Z, E, V) of an n-dimensional block at `betas`. Recipe:
///   1. spectral bounds (E_min, E_max): the extreme Ritz values of a 60-step Lanczos (no
///      reorthogonalisation) on this backend, from the run's auxiliary Gaussian start;
///   2. L just above E_max (the series' terms peak near j* = beta (L - E_min), so L - E_max is
///      pure cost);
///   3. steps so the series converges at the coldest beta: j* + 8 sqrt(j*) terms; a target the
///      trajectory cannot reach is refused, never clamped.
template <typename Backend, typename MatvecFn>
Curves mtpq(Backend& be, MatvecFn&& H, std::size_t n, const std::vector<double>& betas,
                const MtpqRun& run) {
    if (n == 0) throw std::invalid_argument("mtpq: empty block");
    std::vector<double> temperatures;
    temperatures.reserve(betas.size());
    for (double b : betas) temperatures.push_back(b > 0.0 ? 1.0 / b : 0.0);
    double beta_max = 0.0;
    for (double b : betas) beta_max = std::max(beta_max, b);
    if (!(beta_max > 0.0)) beta_max = 100.0;

    const std::uint64_t base_seed = resolve_base_seed(run.seed);
    MtpqOptions kopts;
    kopts.num_samples    = run.samples;
    kopts.random_seed    = base_seed;
    kopts.seed_transform = run.seed_transform;
    kopts.batch_matvec   = run.batch_matvec;

    double e_min_est = 0.0, e_max_est = 0.0;
    {
        std::mt19937 gen = sample_engine(base_seed, kAuxStream);
        const std::vector<Complex> start = gaussian_vector(n, gen);
        auto v0 = be.make_zero_vector(n);
        be.copy_from_host(start.data(), v0.get(), n);
        ed::krylov::LanczosKernelOptions bo;
        bo.max_iter   = std::min<std::size_t>(60, n);
        bo.reorth     = ed::krylov::ReorthPolicy::None;
        bo.keep_basis = false;
        const auto lk = ed::krylov::lanczos_kernel(be, H, n, v0.get(), bo);
        const auto t  = ed::krylov::tridiag_eig(lk.alpha, lk.beta, lk.alpha.size(), /*vectors=*/false);
        if (t.values.empty())
            throw ed::ConvergenceError("mTPQ: the spectral bounds of the block could not be estimated");
        e_min_est = t.values.front();
        e_max_est = t.values.back();
    }
    // L just above the spectrum. The margin covers the Lanczos estimate of E_max, a lower bound on it.
    const double W = e_max_est - e_min_est;
    const double L = e_max_est + std::max({0.05 * W, 1e-6 * std::max(std::abs(e_max_est), W),
                                           std::numeric_limits<double>::min()});   // relative: s * H alike
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
        if (betas.empty()) return Curves{};
        MtpqThermo mt = mtpq_canonical_thermo(kres.sample_energies, kres.sample_log_norms, L, betas,
                                              static_cast<double>(n));
        if (mt.unconverged.empty()) return std::move(mt.curves);
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
            + (auto_steps ? std::string("") : std::string("; raise steps or leave it unset")));
    }
}

}  // namespace ed::thermal
