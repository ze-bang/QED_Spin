// =============================================================================
// examples/thermal/mtpq/cpu_none.cpp
//
// thermal | mTPQ | CPU (OpenMP) | full Hilbert (no symmetry)
//
// Micro-canonical Thermal Pure Quantum (Taylor truncation). Twin: examples/thermal/mtpq/cpu_none.py
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
    opts.method      = "mTPQ";
    opts.device      = "cpu";
    opts.T_min       = 0.1;
    opts.T_max       = 10.0;
    opts.num_T       = 8;
    opts.num_samples = 4;
    opts.random_seed = 0;
    opts.use_sz_if_conserved        = false;
    opts.use_symmetry_if_available  = false;

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
    // gs_E    = -3.6176
    // T[0]    = 0.1000  E = -3.4757  Cv = 6.0048
    // T[mid]  = 5.7571  E = -0.3316  Cv = 0.0480
    // T[-1]   = 10.0000  E = -0.2147  Cv = 0.0157
    // ===========================================================================
    return 0;
}
