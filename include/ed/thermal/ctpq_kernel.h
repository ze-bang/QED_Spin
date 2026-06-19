#pragma once
// =============================================================================
// include/ed/thermal/ctpq_kernel.h
//
// Canonical TPQ facade --- thin wrapper around the unified
// `ed::thermal::tpq_kernel<Backend>` driving Taylor imaginary-time
// evolution. See `tpq_kernel.h` for the iteration loop.
//
// Phase 2.4 of the Minimalist ED Collapse (May 2026): replaces the
// CPU-only `::canonical_tpq` forward with a backend-templated kernel
// call. The legacy `::canonical_tpq` body in `src/solvers/cpu/TPQ.cpp`
// is retained for the CLI/HDF5 paths but will be migrated to a thin
// HDF5 wrapper over this facade in the follow-up tightening.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/solvers/TPQ.h>
#include <ed/thermal/mtpq_kernel.h>   // detail::mtpq_make_seed
#include <ed/thermal/tpq_kernel.h>

namespace ed::thermal {

using Complex = std::complex<double>;

struct CtpqOptions {
    std::size_t num_samples    = 1;
    std::size_t temp_interval  = 1;
    std::size_t taylor_order   = 50;
    double      beta_max       = 1000.0;
    double      delta_beta     = 0.1;
    std::uint64_t random_seed  = 0;
    std::string output_dir;

    /// User-supplied inverse temperatures at which the cTPQ kernel
    /// should snapshot (copy to host) the running TPQ state. Mirrors
    /// ``MtpqOptions::probe_betas``. Empty (default) -> no snapshots.
    std::vector<double> probe_betas;
};

struct CtpqResult {
    /// Final-iterate energy per sample (legacy field).
    std::vector<double> energies;

    /// Per-sample (beta_k, E_k, var_k) trajectories for
    /// ThermodynamicData recombination. For cTPQ, beta_k is exact
    /// (= step * delta_beta) -- no estimator needed -- because the
    /// kernel propagates by e^{-delta_beta/2 * H} via Taylor expansion
    /// so that after k steps the state is e^{-beta_k/2 * H}|r⟩, and
    /// E_k = ⟨ψ|H|ψ⟩ is exactly E_canonical(beta_k).
    std::vector<std::vector<double>> sample_inv_temps;
    std::vector<std::vector<double>> sample_energies;
    std::vector<std::vector<double>> sample_variances;

    /// Host-side state snapshots at the betas closest to
    /// ``CtpqOptions::probe_betas``. Same layout as
    /// ``MtpqResult::state_snapshots`` (outer = sample,
    /// inner = probe-beta slot). Empty when ``probe_betas`` was empty.
    std::vector<std::vector<std::vector<Complex>>> state_snapshots;
    std::vector<std::vector<double>>               state_snapshot_betas;
};

template <typename Backend, typename MatvecFn>
CtpqResult ctpq_kernel(Backend&       backend,
                       MatvecFn&&     apply_H,
                       std::size_t    local_n,
                       std::uint64_t  /*global_n*/,
                       const CtpqOptions& opts)
{
    CtpqResult out;
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

    const std::size_t steps = (opts.delta_beta > 0.0)
        ? static_cast<std::size_t>(
              std::max(1.0, std::ceil(opts.beta_max / opts.delta_beta)))
        : 0;

    for (std::size_t s = 0; s < opts.num_samples; ++s) {
        const std::uint64_t seed = opts.random_seed
                                    ? (opts.random_seed + s)
                                    : ::tpq_per_sample_seed(s);
        auto host_seed = detail::mtpq_make_seed(local_n, seed);

        auto seed_dev = backend.make_zero_vector(local_n);
        backend.copy_from_host(host_seed.data(), seed_dev.get(), local_n);

        TpqKernelOptions kopts;
        kopts.method       = TpqMethod::CanonicalTaylor;
        kopts.delta_beta   = opts.delta_beta;
        kopts.taylor_order = opts.taylor_order;
        kopts.beta_steps   = steps;
        kopts.normalize_each_step = true;

        double final_E = 0.0;
        std::vector<double> traj_betas;
        std::vector<double> traj_Es;
        std::vector<double> traj_vars;
        traj_betas.reserve(steps + 1);
        traj_Es.reserve(steps + 1);
        traj_vars.reserve(steps + 1);
        auto scratch   = backend.make_zero_vector(local_n);
        std::vector<double> best_dist;
        std::vector<Complex> host_buf;
        if (want_snapshots) {
            best_dist.assign(opts.probe_betas.size(),
                             std::numeric_limits<double>::infinity());
        }
        auto on_step = [&](const TpqStepInfo<Backend>& info) -> bool {
            apply_H(info.psi, scratch.get(), info.local_n);
            const Complex e = backend.dot(info.psi, scratch.get(),
                                          info.local_n);
            const double E_k = std::real(e);
            final_E = E_k;
            const Complex hh = backend.dot(scratch.get(), scratch.get(),
                                           info.local_n);
            const double H2_k = std::real(hh);
            const double var_k = std::max(H2_k - E_k * E_k, 0.0);
            traj_betas.push_back(info.beta);
            traj_Es.push_back(E_k);
            traj_vars.push_back(var_k);
            if (want_snapshots) {
                bool host_copied = false;
                for (std::size_t p = 0; p < opts.probe_betas.size(); ++p) {
                    const double d = std::abs(info.beta - opts.probe_betas[p]);
                    if (d < best_dist[p]) {
                        if (!host_copied) {
                            host_buf.assign(info.local_n, Complex{0.0, 0.0});
                            info.backend->copy_to_host(info.psi,
                                                       host_buf.data(),
                                                       info.local_n);
                            host_copied = true;
                        }
                        out.state_snapshots[s][p] = host_buf;
                        out.state_snapshot_betas[s][p] = info.beta;
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
