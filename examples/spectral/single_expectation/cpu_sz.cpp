// =============================================================================
// examples/spectral/single_expectation/cpu_sz.cpp
//
// spectral | single expectation | CPU (OpenMP) | U(1)-Sz, half-filled (Sz=0, n_up=N/2)
//
// ground-state expectation of H via qed.solve. Twin: examples/spectral/single_expectation/cpu_sz.py
//
// Requires: (no special requirements; runs on the default CPU build)
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = 8;
    (void)guard;

    auto op   = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto spec = ed_example::in_memory_spec(std::move(op), N);

    ed::api::SolveOptions opts;
    opts.num_eigenvalues = 2;
    opts.solver          = "LANCZOS";
    opts.device          = "cpu";
    opts.tolerance       = 1e-10;
    opts.sz = N / 2;

    auto result = ed::api::solve(std::move(spec), opts);

    std::cout << std::setprecision(10);
    ed_example::rank0_print("<O> = ", result.eigenvalues[0], "  (E_0)\n");
    ed_example::rank0_print("E[0] = ", result.eigenvalues[0], "\n");
    ed_example::rank0_print("E[1] = ", result.eigenvalues[1], "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // <O> = -3.651093409  (E_0)
    // E[0] = -3.651093409
    // E[1] = -3.128419064
    // ===========================================================================
    return 0;
}
