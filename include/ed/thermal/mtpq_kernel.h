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
// kernel call. The legacy monolith (`src/solvers/cpu/TPQ.cpp`) is gone:
// its drivers were deleted in the Jul-2026 debt cleanup and its last
// survivor, the trajectory aggregator, now lives in
// `include/ed/thermal/tpq_thermo.h` (Stage 11b).
// =============================================================================

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/thermal/tpq_seeding.h>
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

    /// User-supplied inverse temperatures at which the kernel should
    /// snapshot (copy to host) the running TPQ state. The kernel walks
    /// every step and tracks, per probe-beta, the step whose
    /// estimator-beta is closest; the host-staged state at that step
    /// lands in ``MtpqResult::state_snapshots``. Empty (default) means
    /// "no state snapshots" -- the kernel allocates no host buffers and
    /// runs identically to the pre-snapshot path.
    std::vector<double> probe_betas;
};

struct MtpqResult {
    /// Final-iterate energy per sample (legacy field; kept for callers
    /// that only need the long-imaginary-time E proxy).
    std::vector<double> energies;

    /// Per-sample (beta_k, E_k, var_k) trajectories for the
    /// orchestrator's ThermodynamicData post-processor. Each outer
    /// vector has length ``num_samples``; inner vector lengths equal
    /// the number of kernel steps that ran for that sample (the
    /// kernel exits early if ``max_iter`` is hit or ``on_step``
    /// returns false). beta_k is the classical microcanonical-TPQ
    /// estimator
    ///   beta_k = 2 k / (large_value - E_k)
    /// (step >= 1; step == 0 contributes a beta=0 baseline so the
    /// recombiner has a high-T anchor). var_k = <H^2> - E_k^2,
    /// extracted as ||H psi||^2 - E_k^2 from the existing scratch
    /// vector at no extra matvec cost.
    std::vector<std::vector<double>> sample_inv_temps;
    std::vector<std::vector<double>> sample_energies;
    std::vector<std::vector<double>> sample_variances;

    /// Host-side TPQ state snapshots. Outer index = sample,
    /// inner index = ``MtpqOptions::probe_betas`` slot. Each inner
    /// vector has length ``local_n`` once populated, and the matching
    /// ``state_snapshot_betas[s][p]`` carries the actual kernel-step
    /// beta the snapshot was taken at (the nearest beta_k to the
    /// caller's requested probe beta). Empty when ``probe_betas`` was
    /// empty.
    std::vector<std::vector<std::vector<Complex>>> state_snapshots;
    std::vector<std::vector<double>>               state_snapshot_betas;
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
    out.sample_inv_temps.reserve(opts.num_samples);
    out.sample_energies.reserve(opts.num_samples);
    out.sample_variances.reserve(opts.num_samples);
    const bool want_snapshots = !opts.probe_betas.empty();
    if (want_snapshots) {
        out.state_snapshots.assign(
            opts.num_samples,
            std::vector<std::vector<Complex>>(opts.probe_betas.size()));
        out.state_snapshot_betas.assign(
            opts.num_samples,
            std::vector<double>(opts.probe_betas.size(),
                                std::numeric_limits<double>::quiet_NaN()));
    }

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        const std::uint64_t seed = opts.random_seed
                                    ? (opts.random_seed + s)
                                    : ed::tpq_per_sample_seed(s);
        auto host_seed = detail::mtpq_make_seed(local_n, seed);

        auto seed_dev = backend.make_zero_vector(local_n);
        backend.copy_from_host(host_seed.data(), seed_dev.get(), local_n);

        TpqKernelOptions kopts;
        kopts.method      = TpqMethod::Microcanonical;
        kopts.max_iter    = opts.max_iter;
        kopts.large_value = opts.large_value;
        kopts.normalize_each_step = true;

        // Capture both the final-iterate energy (legacy) AND the
        // per-step (beta_k, E_k, var_k) trajectory for ThermodynamicData
        // recombination by the orchestrator. The variance comes free
        // from the existing scratch vector via <H^2> = ||H psi||^2 -- no
        // extra matvec, just one extra dot product per step.
        double final_E = 0.0;
        std::vector<double> traj_betas;
        std::vector<double> traj_Es;
        std::vector<double> traj_vars;
        // Reserve max_iter + 1 (the step=0 baseline + max_iter steps).
        traj_betas.reserve(opts.max_iter + 1);
        traj_Es.reserve(opts.max_iter + 1);
        traj_vars.reserve(opts.max_iter + 1);
        auto scratch   = backend.make_zero_vector(local_n);
        // Per-probe-beta best-so-far distance tracker. The on_step
        // callback walks each user-requested probe beta and, whenever
        // the current step's estimator beta_k is closer than what we
        // have stashed, copies the running TPQ state to host and
        // overwrites the snapshot slot. The host buffer is allocated
        // lazily on the first hit so empty `probe_betas` skips all
        // staging.
        std::vector<double> best_dist;
        std::vector<Complex> host_buf;
        if (want_snapshots) {
            best_dist.assign(opts.probe_betas.size(),
                             std::numeric_limits<double>::infinity());
        }
        auto on_step = [&](const TpqStepInfo<Backend>& info) -> bool {
            apply_H(info.psi, scratch.get(), info.local_n);
            const Complex e = backend.dot(info.psi, scratch.get(), info.local_n);
            const double E_k = std::real(e);
            final_E = E_k;
            // <H^2> = ||H psi||^2 since H is Hermitian and psi is normalised.
            const Complex hh = backend.dot(scratch.get(), scratch.get(),
                                           info.local_n);
            const double H2_k = std::real(hh);
            const double var_k = std::max(H2_k - E_k * E_k, 0.0);
            // mTPQ effective inverse temperature.
            //   step == 0: beta = 0 (high-T baseline anchor)
            //   step >= 1: beta_k = 2 k / (L - E_k) when (L - E_k) > 0
            // Skip points where the denominator would be non-positive
            // (E_k > L means the seed/state has energy above the
            // microcanonical shift -- the formula is undefined there).
            double beta_k = 0.0;
            if (info.step >= 1) {
                const double denom = opts.large_value - E_k;
                if (denom > 0.0) {
                    beta_k = 2.0 * static_cast<double>(info.step) / denom;
                }
            }
            traj_betas.push_back(beta_k);
            traj_Es.push_back(E_k);
            traj_vars.push_back(var_k);
            if (want_snapshots) {
                bool host_copied = false;
                for (std::size_t p = 0; p < opts.probe_betas.size(); ++p) {
                    const double d = std::abs(beta_k - opts.probe_betas[p]);
                    if (d < best_dist[p]) {
                        if (!host_copied) {
                            host_buf.assign(info.local_n, Complex{0.0, 0.0});
                            info.backend->copy_to_host(info.psi,
                                                       host_buf.data(),
                                                       info.local_n);
                            host_copied = true;
                        }
                        out.state_snapshots[s][p] = host_buf;
                        out.state_snapshot_betas[s][p] = beta_k;
                        best_dist[p] = d;
                    }
                }
            }
            return true;
        };
        auto kres = tpq_kernel<Backend>(backend, apply_H, local_n,
                                        seed_dev.get(), kopts, on_step);
        out.energies.push_back(final_E);
        out.sample_inv_temps.push_back(std::move(traj_betas));
        out.sample_energies.push_back(std::move(traj_Es));
        out.sample_variances.push_back(std::move(traj_vars));
    }
    return out;
}

}  // namespace ed::thermal
