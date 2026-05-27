// =============================================================================
// examples/solve/lanczos/mpi_sz_spatial.cpp
//
// solve | LANCZOS | MPI (distributed) | U(1)-Sz x cyclic translation (Sz=0)
//
// single ground-state eigenvalue via Lanczos. Twin: examples/solve/lanczos/mpi_sz_spatial.py
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
    op->symmetry_info = ed::find_symmetries(static_cast<int>(N), "translation");

    auto spec = ed_example::in_memory_spec(std::move(op), N);

    ed::api::SolveOptions opts;
    opts.num_eigenvalues = 1;
    opts.solver          = "LANCZOS";
    opts.device          = "mpi";
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
    // E[0] = -3.6510934089   (filled in by refresh_expected_output.py)  (captured on the mpi reference runner)
    // |E0 - E0_Bethe| ~ 1e-10
    // ===========================================================================
    return 0;
}
