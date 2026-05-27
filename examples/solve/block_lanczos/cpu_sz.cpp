// =============================================================================
// examples/solve/block_lanczos/cpu_sz.cpp
//
// solve | BLOCK_LANCZOS | CPU (OpenMP) | U(1)-Sz, half-filled sector (Sz=0, n_up=N/2)
//
// block-Lanczos for 4 lowest eigenvalues (BLAS-3 path). Twin: examples/solve/block_lanczos/cpu_sz.py
//
// Requires: (no special requirements; runs on the default CPU build)
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>
#include <ed/api/symmetry_helpers.h>

#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    constexpr std::uint64_t N = 8;
    (void)guard;

    auto op = ed_example::heisenberg_chain(N, /*pbc=*/true);

    auto spec = ed_example::in_memory_spec(std::move(op), N);

    ed::api::SolveOptions opts;
    opts.num_eigenvalues = 4;
    opts.solver          = "BLOCK_LANCZOS";
    opts.device          = "cpu";
    opts.sz              = static_cast<int>(N / 2);

    opts.tolerance       = 1e-10;

    auto result = ed::api::solve(std::move(spec), opts);

    std::cout << std::setprecision(10);
    for (std::size_t k = 0; k < result.eigenvalues.size(); ++k) {
        ed_example::rank0_print("E[", k, "] = ", result.eigenvalues[k], "\n");
    }
    const double E0 = result.eigenvalues.front();
    const double Eref = ed_example::bethe_E0(N);
    if (std::isfinite(Eref)) {
        ed_example::rank0_print("|E0 - E0_Bethe| = ",
                                 std::scientific, std::setprecision(2),
                                 std::abs(E0 - Eref), "\n");
    }

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // E[0] = -3.651093409
    // E[1] = -3.128419064
    // E[2] = -3.128419064
    // E[3] = -3.128419064
    // |E0 - E0_Bethe| = 4.44e-16
    // ===========================================================================
    return 0;
}
