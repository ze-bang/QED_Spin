// =============================================================================
// examples/thermal/ftlm/mpi_gpu_none.cpp
//
// thermal | FTLM | multi-rank multi-GPU (NCCL) | full Hilbert (no symmetry)
//
// Finite-Temperature Lanczos: random vectors x Lanczos. Twin: examples/thermal/ftlm/mpi_gpu_none.py
//
// Requires: WITH_MPI + WITH_CUDA + NCCL; launch via mpirun -n <ranks>
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

    ed::api::ThermalOptions opts;
    opts.method      = "FTLM";
    opts.device      = "mpi_gpu";
    opts.T_min       = 0.1;
    opts.T_max       = 10.0;
    opts.num_T       = 8;
    opts.num_samples = 8;
    opts.random_seed = 0;
    opts.use_sz_if_conserved        = false;
    opts.use_symmetry_if_available  = false;
    opts.krylov_dim  = 40;

    auto result = ed::api::thermal(std::move(spec), opts);

    const auto& T  = result.thermo.temperatures;
    const auto& E  = result.thermo.energy;
    const auto& Cv = result.thermo.specific_heat;

    std::cout << std::fixed << std::setprecision(4);
    ed_example::rank0_print("gs_E    = ", result.ground_state_energy, "\n");
    if (!T.empty()) {
        const std::size_t mid = T.size() / 2;
        ed_example::rank0_print(
            "T[0]    = ", T.front(), "  E = ", E.front(), "  Cv = ", Cv.front(), "\n");
        ed_example::rank0_print(
            "T[mid]  = ", T[mid],   "  E = ", E[mid],   "  Cv = ", Cv[mid],   "\n");
        ed_example::rank0_print(
            "T[-1]   = ", T.back(),  "  E = ", E.back(),  "  Cv = ", Cv.back(),  "\n");
    }

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // gs_E    = -3.6510934089
    // T[0]    = 0.10   E = ...   Cv = ...
    // T[mid]  = ...    E = ...   Cv = ...
    // T[-1]   = 10.00  E = ...   Cv = ...
    // (filled in by refresh_expected_output.py)
    // ===========================================================================
    return 0;
}
