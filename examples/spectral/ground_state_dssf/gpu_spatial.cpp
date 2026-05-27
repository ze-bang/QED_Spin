// =============================================================================
// examples/spectral/ground_state_dssf/gpu_spatial.cpp
//
// spectral | ground-state DSSF | single GPU (cuBLAS/cuSPARSE) | cyclic translation group Z_N
//
// dynamical structure factor S_zz(omega) at three omega points. Twin: examples/spectral/ground_state_dssf/gpu_spatial.py
//
// Requires: WITH_CUDA build + a visible CUDA device
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
        obs->addOneBodyTerm(2, i, std::complex<double>(1.0, 0.0)); // Sz
    }

    auto h_spec   = ed_example::in_memory_spec(std::move(H_op), N);
    auto obs_op   = ed::make_operator(ed_example::in_memory_spec(std::move(obs), N));
    std::vector<const ed::LinearOperator*> observables = { obs_op.get() };

    ed::api::SpectralOptions opts;
    opts.method     = "ground_state_cf";
    opts.device     = "gpu";
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
    // (filled in by refresh_expected_output.py once the CPU binaries are built)
    // ===========================================================================
    return 0;
}
