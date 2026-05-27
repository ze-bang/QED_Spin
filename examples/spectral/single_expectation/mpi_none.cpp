// =============================================================================
// examples/spectral/single_expectation/mpi_none.cpp
//
// spectral | single expectation | MPI (distributed) | full Hilbert (no symmetry)
//
// ground-state expectation of H via qed.solve. Twin: examples/spectral/single_expectation/mpi_none.py
//
// Requires: WITH_MPI build + launch via mpirun -n <ranks>
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
    opts.device          = "mpi";
    opts.tolerance       = 1e-10;

    auto result = ed::api::solve(std::move(spec), opts);

    std::cout << std::setprecision(10);
    ed_example::rank0_print("<O> = ", result.eigenvalues[0], "  (E_0)\n");
    ed_example::rank0_print("E[0] = ", result.eigenvalues[0], "\n");
    ed_example::rank0_print("E[1] = ", result.eigenvalues[1], "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // (filled in by refresh_expected_output.py once the CPU binaries are built)
    // ===========================================================================
    return 0;
}
