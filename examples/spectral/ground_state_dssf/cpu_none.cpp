// =============================================================================
// examples/spectral/ground_state_dssf/cpu_none.cpp
//
// spectral | ground-state DSSF | CPU (OpenMP) | full Hilbert (no symmetry)
//
// dynamical structure factor S_zz(omega) at three omega points. Twin: examples/spectral/ground_state_dssf/cpu_none.py
//
// Requires: (no special requirements; runs on the default CPU build)
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>

#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = 8;
    (void)guard;

    auto H_op = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto obs = std::make_unique<Operator>(N, /*spin=*/0.5f);
    // Staggered S_z(q=pi) = sum_i (-1)^i S_z_i. Non-trivially mixes with H
    // (S_z_total is a conserved charge and would make the DSSF identically
    // zero -- using the staggered combination produces a meaningful spectrum).
    for (std::uint64_t i = 0; i < N; ++i) {
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        obs->addOneBodyTerm(2, i, std::complex<double>(sign, 0.0)); // (-1)^i Sz
    }

    auto h_spec   = ed_example::in_memory_spec(std::move(H_op), N);
    auto obs_op   = ed::make_operator(ed_example::in_memory_spec(std::move(obs), N));
    std::vector<const ed::LinearOperator*> observables = { obs_op.get() };

    ed::api::SpectralOptions opts;
    opts.method     = "ground_state_cf";
    opts.device     = "cpu";
    opts.omega_min  = -5.0;
    opts.omega_max  =  5.0;
    opts.num_omega  = 11;
    opts.eta        = 0.1;
    opts.krylov_dim = 80;

    auto result = ed::api::spectral(std::move(h_spec), observables, opts);

    const std::size_t mid = result.omega.size() / 2;
    std::cout << std::fixed << std::setprecision(6);
    ed_example::rank0_print("S(w=-5.0) = ", result.S_real.front(), "\n");
    ed_example::rank0_print("S(w= 0.0) = ", result.S_real[mid],   "\n");
    ed_example::rank0_print("S(w= 5.0) = ", result.S_real.back(),  "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // S(w=-5.0) = 0.006872
    // S(w= 0.0) = 0.703124
    // S(w= 5.0) = 0.013104
    // ===========================================================================
    return 0;
}
