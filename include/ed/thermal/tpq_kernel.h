#pragma once
// =============================================================================
// include/ed/thermal/tpq_kernel.h
//
// Backend-templated microcanonical TPQ kernel: iterate
// |psi_{k+1}> = (L*I - H)|psi_k> / ||(L*I - H)|psi_k>||, handing each
// iterate (and its energy moments) to a per-step callback that reads off
// the inverse temperature beta_k = 2 k / (L - E_k).
//
// The kernel itself does ONLY the iteration; observable measurement and
// the temperature bookkeeping live in the calling driver. Consumers:
//
//   * the `ed::workflows::thermal` orchestrator via `mtpq_kernel.h`
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include <ed/matvec/backend.h>

namespace ed::thermal {

using Complex = std::complex<double>;

enum class TpqMethod : std::uint8_t {
    Microcanonical = 0,
};

/// Configuration for the TPQ iteration kernel.
struct TpqKernelOptions {
    TpqMethod   method        = TpqMethod::Microcanonical;
    /// Total number of iterations (k = 0..max_iter).
    std::size_t max_iter      = 0;
    /// Large value L in |psi_{k+1}> = (L*I - H)|psi_k>.
    double      large_value   = 1.0e5;
    /// Renormalise after every step? Default true matches mTPQ /
    /// imaginary-time-evolution convention.
    bool        normalize_each_step = true;
};

/// Snapshot passed to the per-step callback so the driver can extract
/// energies, variances, observables, etc. The callback may inspect
/// `psi` (BACKEND memory!) using the same `Backend` it gave the kernel
/// (dot, copy_to_host, ...).
template <typename Backend>
struct TpqStepInfo {
    Backend*       backend;
    const Complex* psi;        ///< current state, backend memory, length local_n
    std::size_t    local_n;
    std::size_t    step;       ///< 0-based step counter
    double         beta;       ///< always 0 (the driver derives beta_k)
    double         norm_before_normalize;
    /// The kernel computes H psi_k once per step and
    /// hands the moments to the callback, so drivers need no second matvec.
    bool           moments_valid = false;
    double         energy        = 0.0;   ///< Re <psi|H|psi>   (psi normalised)
    double         h2            = 0.0;   ///< <psi|H^2|psi> = ||H psi||^2
};

/// Optional convergence/early-stop callback. Return `false` to halt the
/// iteration. Receives the step snapshot.
template <typename Backend>
using TpqStepCallback = std::function<bool(const TpqStepInfo<Backend>&)>;

/// Result of `tpq_kernel`. `psi_final` points to backend memory; the
/// caller owns it via the returned `UniqueVec`.
struct TpqKernelResult {
    ed::matvec::Backend::UniqueVec psi_final;
    std::size_t                    steps_done = 0;
};

/// Run TPQ iteration on `apply_H` starting from `psi_seed` (backend
/// memory). The kernel takes ownership of a fresh backend buffer holding
/// the evolving state; `psi_seed` is left untouched.
template <typename Backend, typename MatvecFn>
TpqKernelResult tpq_kernel(Backend&                        be,
                            MatvecFn&&                       apply_H,
                            std::size_t                      local_n,
                            const Complex*                   psi_seed,
                            const TpqKernelOptions&          opts,
                            TpqStepCallback<Backend>         on_step = {})
{
    if (local_n == 0) {
        throw std::invalid_argument("tpq_kernel: local_n == 0");
    }

    auto psi    = be.make_zero_vector(local_n);
    auto scratchA = be.make_zero_vector(local_n);
    be.copy(psi_seed, psi.get(), local_n);

    // Normalise seed defensively.
    {
        const double n0 = be.nrm2(psi.get(), local_n);
        if (n0 > 0.0) be.scale(Complex(1.0 / n0, 0.0), psi.get(), local_n);
    }

    std::size_t steps = 0;
    if (opts.method == TpqMethod::Microcanonical) {
        // One matvec per step. The product H psi_k that forms
        // psi_{k+1} = (L - H) psi_k also yields the moments E_k = Re<psi_k|H psi_k>
        // and <H^2>_k = ||H psi_k||^2 handed to the callback; the state
        // update is one axpby plus a pointer swap (no copy).
        auto moments = [&](double& E, double& H2) {
            apply_H(psi.get(), scratchA.get(), local_n);          // scratchA = H psi
            E  = std::real(be.dot(psi.get(), scratchA.get(), local_n));
            H2 = std::real(be.dot(scratchA.get(), scratchA.get(), local_n));
        };
        double E0 = 0.0, H20 = 0.0;
        moments(E0, H20);
        if (on_step) {
            TpqStepInfo<Backend> info{&be, psi.get(), local_n,
                                      /*step=*/0, /*beta=*/0.0,
                                      /*norm_before_normalize=*/1.0};
            info.moments_valid = true; info.energy = E0; info.h2 = H20;
            if (!on_step(info)) {
                TpqKernelResult R;
                R.psi_final = std::move(psi);
                R.steps_done = 0;
                return R;
            }
        }
        const std::size_t total = opts.max_iter;
        for (std::size_t k = 1; k <= total; ++k) {
            // scratchA = L psi - H psi  (H psi already in scratchA)
            be.axpby(Complex(opts.large_value, 0.0), psi.get(),
                     Complex(-1.0, 0.0), scratchA.get(), local_n);
            std::swap(psi, scratchA);
            const double nrm = be.nrm2(psi.get(), local_n);
            if (opts.normalize_each_step && nrm > 0.0) {
                be.scale(Complex(1.0 / nrm, 0.0), psi.get(), local_n);
            }
            ++steps;
            double Ek = 0.0, H2k = 0.0;
            moments(Ek, H2k);                                      // also prepares the next step
            if (on_step) {
                TpqStepInfo<Backend> info{&be, psi.get(), local_n,
                                          k, /*beta=*/0.0, nrm};
                info.moments_valid = true; info.energy = Ek; info.h2 = H2k;
                if (!on_step(info)) break;
            }
        }
    }

    TpqKernelResult R;
    R.psi_final = std::move(psi);
    R.steps_done = steps;
    return R;
}

}  // namespace ed::thermal
