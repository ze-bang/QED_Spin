// =============================================================================
// examples/spectral/static_thermal/cpu_sz.cpp
//
// spectral | static thermal | CPU (OpenMP) | U(1)-Sz, half-filled (Sz=0, n_up=N/2)
//
// static thermodynamic averages at a single T via qed.thermal. Twin: examples/spectral/static_thermal/cpu_sz.py
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

    ed::api::ThermalOptions opts;
    opts.method      = "FTLM";
    opts.device      = "cpu";
    opts.T_min       = 0.5;
    opts.T_max       = 2.0;
    opts.num_T       = 4;
    opts.num_samples = 8;
    opts.random_seed = 0;
    opts.use_sz_if_conserved = true;
    opts.sz_min = static_cast<int>(N / 2);
    opts.sz_max = static_cast<int>(N / 2);

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
    // gs_E    = -2.8204
    // T_probe = 1.5000  E = -1.3940  Cv = 1.0370
    // ===========================================================================
    return 0;
}
