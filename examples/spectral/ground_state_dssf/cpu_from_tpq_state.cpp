// =============================================================================
// examples/spectral/ground_state_dssf/cpu_from_tpq_state.cpp
//
// spectral | ground-state DSSF from a *finite-T TPQ state* | CPU (OpenMP)
//
// Pillar 3 of the May 2026 "Save and DSSF Upgrades" plan. Demonstrates
// the TPQ-to-CF spectral pipeline:
//
//   1. Generate an mTPQ state at beta = 1 and persist it to HDF5 via
//      `opts.output_dir + opts.probe_betas`.
//   2. Reload the persisted state with `HDF5IO::loadTPQState`.
//   3. Feed it into `ed::api::spectral` as `opts.initial_state`; the
//      `GroundStateCF` branch skips the internal Lanczos seed step and
//      runs the continued-fraction kernel directly on the warm state.
//
// This is the finite-temperature spectral lane the user described: warm
// the seed via TPQ, then run CF for an inexpensive S(omega, beta).
//
// Twin: examples/spectral/ground_state_dssf/cpu_from_tpq_state.py
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

namespace {

ed::OperatorSpec build_chain_spec(std::uint64_t N) {
    auto op = ed_example::heisenberg_chain(N, /*pbc=*/true);
    return ed_example::in_memory_spec(std::move(op), N);
}

}  // namespace

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    (void)guard;
    constexpr std::uint64_t N = 6;

    // -- Step 1: persist an mTPQ state at beta = 1.
    const std::string outdir = "ed_tpq_to_cf_demo";
    std::filesystem::remove_all(outdir);

    ed::api::ThermalOptions topts;
    topts.method                  = "mTPQ";
    topts.device                  = "cpu";
    topts.T_min                   = 0.5;
    topts.T_max                   = 4.0;
    topts.num_T                   = 8;
    topts.num_samples             = 1;
    topts.random_seed             = 42;
    topts.max_iterations          = 200;
    topts.use_sz_if_conserved     = false;
    topts.use_symmetry_if_available = false;
    topts.output_dir              = outdir;
    topts.probe_betas             = {1.0};

    auto tres = ed::api::thermal(build_chain_spec(N), topts);
    ed_example::rank0_print("tpq.hdf5_path = ", tres.hdf5_path, "\n");

    // -- Step 2: reload the persisted state from HDF5.
    auto snapshots = HDF5IO::listTPQStates(tres.hdf5_path, /*sample_filter=*/0);
    double best_dist = std::numeric_limits<double>::infinity();
    double best_beta = 0.0;
    for (const auto& info : snapshots) {
        const double d = std::abs(info.beta - 1.0);
        if (d < best_dist) {
            best_dist = d;
            best_beta = info.beta;
        }
    }
    std::vector<std::complex<double>> psi_beta;
    const bool ok = HDF5IO::loadTPQState(tres.hdf5_path,
                                          /*sample_index=*/0,
                                          best_beta,
                                          psi_beta);
    ed_example::rank0_print("loadTPQState ok       = ", ok, "\n");
    ed_example::rank0_print("|psi_beta| dim        = ", psi_beta.size(), "\n");
    ed_example::rank0_print("nearest snapshot beta = ",
                             std::setprecision(4), best_beta, "\n");

    // -- Step 3: feed the warm state into GroundStateCF as the seed.
    auto h_spec = build_chain_spec(N);
    auto obs_local = std::make_unique<Operator>(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i < N; ++i) {
        obs_local->addOneBodyTerm(2, i, std::complex<double>(1.0, 0.0));
    }
    auto obs_op   = ed::make_operator(ed_example::in_memory_spec(std::move(obs_local), N));
    std::vector<const ed::LinearOperator*> observables = { obs_op.get() };

    ed::api::SpectralOptions sopts;
    sopts.method        = "ground_state_cf";
    sopts.device        = "cpu";
    sopts.omega_min     = -5.0;
    sopts.omega_max     =  5.0;
    sopts.num_omega     = 11;
    sopts.eta           = 0.1;
    sopts.krylov_dim    = 80;
    sopts.initial_state = psi_beta;

    auto sres = ed::api::spectral(std::move(h_spec), observables, sopts);

    const std::size_t mid = sres.omega.size() / 2;
    std::cout << std::fixed << std::setprecision(6);
    ed_example::rank0_print("S(w=-5.0) = ", sres.S_real.front(), "\n");
    ed_example::rank0_print("S(w= 0.0) = ", sres.S_real[mid],   "\n");
    ed_example::rank0_print("S(w= 5.0) = ", sres.S_real.back(),  "\n");

    // === Expected output (smoke-tested by the CI harness; non-zero spectral weight ===
    // === at finite omega confirms the TPQ-to-CF pipeline ran end-to-end.) =============
    // tpq.hdf5_path = ed_tpq_to_cf_demo/ed_results.h5
    // loadTPQState ok       = 1
    // |psi_beta| dim        = 64
    // nearest snapshot beta = 1.006
    // S(w=-5.0) = 0.000729
    // S(w= 0.0) = 0.623659
    // S(w= 5.0) = 0.003318
    // ===========================================================================
    return 0;
}
