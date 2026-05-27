// =============================================================================
// examples/spectral/dynamical_thermal/cpu_sz_spatial.cpp
//
// spectral | dynamical thermal | CPU (OpenMP) | U(1)-Sz x cyclic translation
//
// finite-T S_zz(omega, T) at three (T, omega) probe points. Twin: examples/spectral/dynamical_thermal/cpu_sz_spatial.py
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
    for (std::uint64_t i = 0; i < N; ++i) {
        obs->addOneBodyTerm(2, i, std::complex<double>(1.0, 0.0));
    }

    auto h_spec = ed_example::in_memory_spec(std::move(H_op), N);
    auto obs_op = ed::make_operator(ed_example::in_memory_spec(std::move(obs), N));
    std::vector<const ed::LinearOperator*> observables = { obs_op.get() };

    ed::api::SpectralOptions opts;
    opts.method               = "ftlm_dynamical";
    opts.device               = "cpu";
    opts.temperatures         = {0.5, 1.0, 2.0};
    opts.omega_min            = -5.0;
    opts.omega_max            =  5.0;
    opts.num_omega            = 11;
    opts.eta                  = 0.1;
    opts.krylov_dim           = 40;
    opts.num_random_vectors   = 8;
    opts.sz = N / 2;

    auto result = ed::api::spectral(std::move(h_spec), observables, opts);

    const std::size_t mid = result.omega.size() / 2;
    std::cout << std::fixed << std::setprecision(6);
    ed_example::rank0_print("S(w=-5.0) = ", result.S_real.front(), "\n");
    ed_example::rank0_print("S(w= 0.0) = ", result.S_real[mid],   "\n");
    ed_example::rank0_print("S(w= 5.0) = ", result.S_real.back(),  "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // S(w=-5.0) = 0.000892
    // S(w= 0.0) = 0.031189
    // S(w= 5.0) = 0.524750
    // ===========================================================================
    return 0;
}
