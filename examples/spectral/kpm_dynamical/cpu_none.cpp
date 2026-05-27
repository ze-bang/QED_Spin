// =============================================================================
// examples/spectral/kpm_dynamical/cpu_none.cpp
//
// spectral | KPM dynamical | CPU (OpenMP) | full Hilbert (no symmetry)
//
// Pillar 4 of the May 2026 "Save and DSSF Upgrades" plan. Demonstrates
// the new ``method = "kpm_dynamical"`` lane: Chebyshev expansion of
// ``delta(omega - H)`` against the ground-state seed, evaluated on the
// user-supplied omega grid via
// ``ed::observables::kpm_dynamical_correlator``.
//
// Twin: examples/spectral/kpm_dynamical/cpu_none.py.
// Requires: (no special requirements; runs on the default CPU build)
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>

#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = 4;
    (void)guard;

    auto H_op = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto obs = std::make_unique<Operator>(N, /*spin=*/0.5f);
    obs->addOneBodyTerm(2, 0, std::complex<double>(1.0, 0.0));  // S^z_0

    auto h_spec = ed_example::in_memory_spec(std::move(H_op), N);
    auto obs_op = ed::make_operator(ed_example::in_memory_spec(std::move(obs), N));
    std::vector<const ed::LinearOperator*> observables = { obs_op.get() };

    ed::api::SpectralOptions opts;
    opts.method       = "kpm_dynamical";
    opts.device       = "cpu";
    opts.omega_min    = -4.0;
    opts.omega_max    =  4.0;
    opts.num_omega    = 41;
    opts.kpm_moments  = 400;
    opts.kpm_kernel   = "Jackson";

    auto result = ed::api::spectral(std::move(h_spec), observables, opts);

    std::cout << std::fixed << std::setprecision(4);
    const std::size_t mid = result.omega.size() / 2;
    ed_example::rank0_print("omega.size()      = ", result.omega.size(), "\n");
    ed_example::rank0_print("S(w=-1.0) approx  = ", result.S_real[15], "\n");
    ed_example::rank0_print("S(w= 0.0) approx  = ", result.S_real[mid], "\n");
    ed_example::rank0_print("S(w=+1.0) approx  = ", result.S_real[25], "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // omega.size()      = 41
    // S(w=-1.0) approx  = 0.0000
    // S(w= 0.0) approx  = 0.0000
    // S(w=+1.0) approx  = 2.8864
    // ===========================================================================
    return 0;
}
