#pragma once
// =============================================================================
// include/ed/thermal/tpq_kernel.h
//
// Unified, backend-templated TPQ kernel. Covers BOTH thermodynamic
// flavours used throughout the project:
//
//     Method::Microcanonical    --- iterate |psi_{k+1}> = (L*I - H)|psi_k>
//                                  / ||(L*I - H)|psi_k>||. Each iterate
//                                  corresponds to an inverse-temperature
//                                  beta_k = (large_value - <H>_k) / variance
//                                  (read off by the per-step callback).
//
//     Method::CanonicalTaylor   --- propagate by e^{-delta_beta/2 · H} via a
//                                  Taylor expansion of order `taylor_order`,
//                                  renormalise after every step. Each step
//                                  advances beta by delta_beta so that after
//                                  k steps the state is e^{-β/2 H}|r⟩ with
//                                  β = k·delta_beta, giving the canonical
//                                  thermal energy E(β) from ⟨ψ|H|ψ⟩.
//
// The kernel itself does ONLY the iteration; all HDF5 result bookkeeping,
// observable measurement, MPI sample partitioning, log-spaced temperature
// checkpoint logic etc lives in the calling driver. This split allows the
// same kernel body to drive:
//
//   * `src/solvers/cpu/TPQ.cpp::microcanonical_tpq` / `::canonical_tpq`
//   * `src/solvers/gpu/gpu_tpq.cu`
//   * `src/distributed/distributed_tpq*.cpp`
//   * future `ed::thermal()` orchestrator in Phase 4.2
//
// Phase 2.4 of the Minimalist ED Collapse (May 2026): see
// `ed/thermal/mtpq_kernel.h` and `ed/thermal/ctpq_kernel.h` for the
// thin facade headers that the test_kernel_facades test consumes.
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
    CanonicalTaylor = 1,
};

/// Configuration for the unified TPQ iteration kernel.
struct TpqKernelOptions {
    TpqMethod   method        = TpqMethod::Microcanonical;
    /// Microcanonical: total number of iterations (k = 0..max_iter).
    /// CanonicalTaylor: ignored; `beta_steps` drives the loop instead.
    std::size_t max_iter      = 0;
    /// Microcanonical large_value L in |psi_{k+1}> = (L*I - H)|psi_k>.
    double      large_value   = 1.0e5;
    /// CanonicalTaylor: Taylor expansion order p. 50 is the historical
    /// default; the truncation is benign for delta_beta * ||H|| <= 1.
    std::size_t taylor_order  = 50;
    /// CanonicalTaylor: imaginary-time step. The kernel calls back at
    /// every step, so the driver decides what to record.
    double      delta_beta    = 0.1;
    /// CanonicalTaylor: number of delta_beta steps to take. The driver
    /// can shorten the loop via `on_step` returning false.
    std::size_t beta_steps    = 0;
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
    double         beta;       ///< current inverse temperature (cTPQ) or 0 (mTPQ)
    double         norm_before_normalize;
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

namespace detail {

template <typename Backend, typename MatvecFn>
void apply_microcanonical_step(Backend& be, MatvecFn&& apply_H,
                               double large_value,
                               Complex* psi, Complex* scratch,
                               std::size_t local_n) {
    // scratch <- H psi
    apply_H(psi, scratch, local_n);
    // scratch <- L * psi - scratch == (L - H) psi
    be.scale(Complex(-1.0, 0.0), scratch, local_n);
    be.axpy(Complex(large_value, 0.0), psi, scratch, local_n);
    // psi <- scratch
    be.copy(scratch, psi, local_n);
}

template <typename Backend, typename MatvecFn>
void apply_taylor_step(Backend& be, MatvecFn&& apply_H,
                       double delta_beta, std::size_t taylor_order,
                       Complex* psi, Complex* term, Complex* Hterm,
                       std::size_t local_n) {
    // result = sum_{n=0}^{p} (-delta_beta)^n / n! * H^n psi
    // Implemented Horner-free: term <- H * term every iteration,
    // accumulate scaled term into psi (which doubles as `result`).
    // We need a fresh copy of psi at n=0 so accumulation does not
    // contaminate the H multiplications; stage that into `term`.
    be.copy(psi, term, local_n);  // term = H^0 psi
    double coef = 1.0;
    for (std::size_t order = 1; order <= taylor_order; ++order) {
        apply_H(term, Hterm, local_n);
        std::swap(term, Hterm);  // pointer-swap (caller-owned buffers)
        coef *= (-delta_beta) / static_cast<double>(order);
        be.axpy(Complex(coef, 0.0), term, psi, local_n);
    }
}

}  // namespace detail

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
    auto scratchB = be.make_zero_vector(local_n);
    be.copy(psi_seed, psi.get(), local_n);

    // Normalise seed defensively.
    {
        const double n0 = be.nrm2(psi.get(), local_n);
        if (n0 > 0.0) be.scale(Complex(1.0 / n0, 0.0), psi.get(), local_n);
    }

    // Step 0 callback so drivers can record the beta=0 baseline.
    if (on_step) {
        TpqStepInfo<Backend> info{&be, psi.get(), local_n,
                                  /*step=*/0, /*beta=*/0.0,
                                  /*norm_before_normalize=*/1.0};
        if (!on_step(info)) {
            TpqKernelResult R;
            R.psi_final = std::move(psi);
            R.steps_done = 0;
            return R;
        }
    }

    std::size_t steps = 0;
    if (opts.method == TpqMethod::Microcanonical) {
        const std::size_t total = opts.max_iter;
        for (std::size_t k = 1; k <= total; ++k) {
            detail::apply_microcanonical_step(be, apply_H, opts.large_value,
                                              psi.get(), scratchA.get(),
                                              local_n);
            const double nrm = be.nrm2(psi.get(), local_n);
            if (opts.normalize_each_step && nrm > 0.0) {
                be.scale(Complex(1.0 / nrm, 0.0), psi.get(), local_n);
            }
            ++steps;
            if (on_step) {
                TpqStepInfo<Backend> info{&be, psi.get(), local_n,
                                          k, /*beta=*/0.0, nrm};
                if (!on_step(info)) break;
            }
        }
    } else {
        const std::size_t total = opts.beta_steps;
        double beta = 0.0;
        for (std::size_t k = 1; k <= total; ++k) {
            // The canonical-TPQ energy formula is
            //   E_r(β) = ⟨r|e^{-βH/2} H e^{-βH/2}|r⟩ / ⟨r|e^{-βH}|r⟩
            // which requires the state to be e^{-βH/2}|r⟩.  Each imaginary-
            // time step must therefore advance the exponent by Δβ/2, not Δβ,
            // so that after k steps the state is e^{-k(Δβ/2)H}|r⟩ = e^{-β/2 H}|r⟩
            // and ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ = E_canonical(β = k·Δβ).  Using Δβ instead
            // of Δβ/2 would produce E_canonical(2β) — a factor-of-2 error in
            // the temperature axis.
            detail::apply_taylor_step(be, apply_H, opts.delta_beta / 2.0,
                                      opts.taylor_order,
                                      psi.get(), scratchA.get(),
                                      scratchB.get(), local_n);
            const double nrm = be.nrm2(psi.get(), local_n);
            if (opts.normalize_each_step && nrm > 0.0) {
                be.scale(Complex(1.0 / nrm, 0.0), psi.get(), local_n);
            }
            beta += opts.delta_beta;
            ++steps;
            if (on_step) {
                TpqStepInfo<Backend> info{&be, psi.get(), local_n,
                                          k, beta, nrm};
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
