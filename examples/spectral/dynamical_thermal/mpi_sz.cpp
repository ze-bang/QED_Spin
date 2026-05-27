// =============================================================================
// examples/spectral/dynamical_thermal/mpi_sz.cpp
//
// spectral | dynamical thermal | MPI (distributed) | U(1)-Sz, half-filled (Sz=0, n_up=N/2)
//
// finite-T S_zz(omega, T) at three (T, omega) probe points. Twin: examples/spectral/dynamical_thermal/mpi_sz.py
//
// Requires: WITH_MPI build + launch via mpirun -n <ranks>
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
    opts.device               = "mpi";
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
    // (filled in by refresh_expected_output.py once the CPU binaries are built)
    // ===========================================================================
    return 0;
}
