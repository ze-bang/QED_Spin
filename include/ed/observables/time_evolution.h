#pragma once
// =============================================================================
// include/ed/observables/time_evolution.h
//
// time_evolution_correlator(H, A, B, psi, t_grid, dt):
//   C_{AB}(t) = <psi| A^†(t) B |psi> = <psi| e^{iHt} A^† e^{-iHt} B |psi>
//
// Phase-6 primitive 5 of 5. The CPU body composes the existing Krylov
// time-evolution primitive (`time_evolve_krylov`) from
// `ed/solvers/dynamics.h` with the two operator applies (A and B) and
// the backend dot product. Replaces the dynamical-correlator pattern
// in `tpq_dynamical.cpp` and the time-evolution helpers in
// `ed/solvers/dynamics.h` with a single template body that takes a
// reference state and a time grid.
//
// Implementation strategy
//   * Build |phi_0> = B |psi>.
//   * For each t in t_grid: time-evolve |phi> = e^{-iH dt} |phi>
//     using `time_evolve_krylov`, then evaluate
//        C(t) = <A psi | phi(t)>
//     via `backend.dot()`.
//   * The Krylov dimension and step size live in `opts`.
// =============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/dynamics.h>

namespace ed::observables {

using Complex = std::complex<double>;

struct TimeEvolutionOptions {
    std::size_t krylov_dim = 50;
    double      dt         = 1e-2;
    double      t_max      = 10.0;
    double      tolerance  = 1e-10;
    bool        normalize  = false;
};

struct TimeEvolutionResult {
    std::vector<double>  t;
    std::vector<Complex> correlator;
};

template <typename Backend>
TimeEvolutionResult time_evolution_correlator(
    const Backend&                              backend,
    const ed::matvec::MatVecOperator&           H,
    const ed::matvec::MatVecOperator&           A,
    const ed::matvec::MatVecOperator&           B,
    const Complex*                              psi,
    std::size_t                                 local_n,
    const TimeEvolutionOptions&                 opts)
{
    const std::size_t n = local_n;

    // |Apsi> = A |psi>; |phi> = B |psi> at t = 0.
    std::vector<Complex> Apsi(n);
    std::vector<Complex> phi(n);
    A.apply(psi, Apsi.data(), n);
    B.apply(psi,  phi.data(), n);

    std::function<void(const Complex*, Complex*, int)> H_apply =
        [&H](const Complex* in, Complex* out, int nn) {
            H.apply(in, out, static_cast<std::size_t>(nn));
        };

    const std::size_t num_steps =
        opts.dt > 0.0 ? static_cast<std::size_t>(opts.t_max / opts.dt) : 0;

    TimeEvolutionResult out;
    out.t.reserve(num_steps + 1);
    out.correlator.reserve(num_steps + 1);

    // t = 0
    out.t.push_back(0.0);
    out.correlator.push_back(backend.dot(Apsi.data(), phi.data(), n));

    for (std::size_t step = 0; step < num_steps; ++step) {
        ::time_evolve_krylov(H_apply, phi,
                             static_cast<std::uint64_t>(n),
                             opts.dt,
                             static_cast<std::uint64_t>(opts.krylov_dim),
                             opts.normalize);
        out.t.push_back(static_cast<double>(step + 1) * opts.dt);
        out.correlator.push_back(
            backend.dot(Apsi.data(), phi.data(), n));
    }
    return out;
}

}  // namespace ed::observables
