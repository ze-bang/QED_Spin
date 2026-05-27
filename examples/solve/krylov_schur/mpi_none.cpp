// =============================================================================
// examples/solve/krylov_schur/mpi_none.cpp
//
// solve | KRYLOV_SCHUR | MPI (distributed) | full Hilbert (no symmetry)
//
// thick-restart Krylov-Schur for 5 lowest eigenvalues. Twin: examples/solve/krylov_schur/mpi_none.py
//
// Requires: WITH_MPI build + launch via mpirun -n <ranks>
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
    opts.num_eigenvalues = 5;
    opts.solver          = "KRYLOV_SCHUR";
    opts.device          = "mpi";

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
    // E[0] = -3.6510934089   (filled in by refresh_expected_output.py)  (captured on the mpi reference runner)
    // E[1] = ...
    // E[2] = ...
    // E[3] = ...
    // E[4] = ...
    // |E0 - E0_Bethe| ~ 1e-10
    // ===========================================================================
    return 0;
}
