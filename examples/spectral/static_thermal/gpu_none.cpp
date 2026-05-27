// =============================================================================
// examples/spectral/static_thermal/gpu_none.cpp
//
// spectral | static thermal | single GPU (cuBLAS/cuSPARSE) | full Hilbert (no symmetry)
//
// static thermodynamic averages at a single T via qed.thermal. Twin: examples/spectral/static_thermal/gpu_none.py
//
// Requires: WITH_CUDA build + a visible CUDA device
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
    opts.device      = "gpu";
    opts.T_min       = 0.5;
    opts.T_max       = 2.0;
    opts.num_T       = 4;
    opts.num_samples = 8;
    opts.random_seed = 0;

    auto result = ed::api::thermal(std::move(spec), opts);

    const auto& T  = result.thermo.temperatures;
    const auto& E  = result.thermo.energy;
    const auto& Cv = result.thermo.specific_heat;

    std::cout << std::fixed << std::setprecision(4);
    ed_example::rank0_print("gs_E    = ", result.ground_state_energy, "\n");
    if (!T.empty()) {
        const std::size_t iT = T.size() / 2;
        ed_example::rank0_print(
            "T_probe = ", T[iT], "  E = ", E[iT], "  Cv = ", Cv[iT], "\n");
    }

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // (filled in by refresh_expected_output.py once the CPU binaries are built)
    // ===========================================================================
    return 0;
}
