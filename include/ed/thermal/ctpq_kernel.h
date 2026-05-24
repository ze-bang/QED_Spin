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
};

struct CtpqResult {
    std::vector<double> energies;
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
