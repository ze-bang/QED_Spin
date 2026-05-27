// =============================================================================
// examples/thermal/mtpq/cpu_save.cpp
//
// thermal | mTPQ | CPU (OpenMP) | full Hilbert (no symmetry)
// + persistent trajectory + state-vector snapshots at probe-betas (HDF5)
//
// Pillar 1 of the May 2026 "Save and DSSF Upgrades" plan. Demonstrates:
//
//     opts.output_dir  = "<dir>";
//     opts.probe_betas = {beta_1, beta_2, ...};
//
// causes `ed::api::thermal` (== `ed::workflows::thermal`) to persist:
//
//   * /tpq/samples/sample_<s>/thermodynamics  --
//        per-sample (beta, E, var, doublon, step) rows for the whole
//        mTPQ trajectory (one row per kernel step).
//   * /tpq/samples/sample_<s>/states/beta_<b> --
//        host-side state vector at the kernel-step beta closest to each
//        user-requested probe beta.
//
// `result.hdf5_path` reports the resulting file (mirrored into Python's
// `EDResults.eigenvectors_path` / `ThermalResult.hdf5_path`).
//
// The companion Python example reloads the persisted state via h5py and
// feeds it into `qed.spectral(method="GroundStateCF", initial_state=...)`
// for the TPQ-to-CF spectral pipeline; the C++ example below loads it
// via `HDF5IO::loadTPQState` and reports `<psi|psi> = 1` to verify the
// round-trip.
//
// Twin: examples/thermal/mtpq/cpu_save.py
//
// Requires: HDF5 (already required by the default CPU build)
// =============================================================================

#include "../../_shared/common.h"

#include <ed/api.h>
#include <ed/core/hdf5_io.h>

#include <cmath>
#include <complex>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    (void)guard;
    constexpr std::uint64_t N = 8;

    auto op   = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto spec = ed_example::in_memory_spec(std::move(op), N);

    const std::string outdir = "ed_thermal_save_demo";
    std::filesystem::remove_all(outdir);

    ed::api::ThermalOptions opts;
    opts.method      = "mTPQ";
    opts.device      = "cpu";
    opts.T_min       = 0.1;
    opts.T_max       = 10.0;
    opts.num_T       = 8;
    opts.num_samples = 1;
    opts.random_seed = 42;
    opts.max_iterations          = 200;
    opts.use_sz_if_conserved     = false;
    opts.use_symmetry_if_available = false;
    opts.output_dir              = outdir;
    opts.probe_betas             = {1.0, 5.0};

    auto result = ed::api::thermal(std::move(spec), opts);

    std::cout << std::setprecision(6);
    ed_example::rank0_print("hdf5_path = ", result.hdf5_path, "\n");
    ed_example::rank0_print("gs_E      = ", result.ground_state_energy, "\n");

    // Round-trip: enumerate the snapshots persisted in the file and
    // load the one whose effective beta is closest to 5.0. The kernel
    // rounds each user-requested probe beta to the nearest
    // trajectory-step beta, so we ask for an exact beta match using
    // the value we discover in the HDF5 listing.
    auto snapshots = HDF5IO::listTPQStates(result.hdf5_path,
                                             /*sample_filter=*/0);
    double best_dist = std::numeric_limits<double>::infinity();
    double best_beta = 0.0;
    for (const auto& info : snapshots) {
        const double d = std::abs(info.beta - 5.0);
        if (d < best_dist) {
            best_dist = d;
            best_beta = info.beta;
        }
    }
    std::vector<std::complex<double>> psi_reload;
    const bool ok = HDF5IO::loadTPQState(result.hdf5_path,
                                          /*sample_index=*/0,
                                          best_beta,
                                          psi_reload);

    double norm2 = 0.0;
    for (const auto& z : psi_reload) {
        norm2 += std::norm(z);
    }
    ed_example::rank0_print("snapshot count           = ",
                             snapshots.size(), "\n");
    ed_example::rank0_print("closest beta to 5.0      = ",
                             std::setprecision(4), best_beta, "\n");
    ed_example::rank0_print("loaded |psi| dim         = ",
                             psi_reload.size(), "\n");
    ed_example::rank0_print("loaded <psi|psi>         = ",
                             std::setprecision(10), norm2, "\n");
    ed_example::rank0_print("HDF5IO::loadTPQState ok  = ", ok, "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // hdf5_path = ed_thermal_save_demo/ed_results.h5
    // gs_E      = -3.6216
    // snapshot count           = 2
    // closest beta to 5.0      = 5.002
    // loaded |psi| dim         = 256
    // loaded <psi|psi>         = 1
    // HDF5IO::loadTPQState ok  = 1
    // ===========================================================================
    return 0;
}
