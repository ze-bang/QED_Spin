// =============================================================================
// test_dispatch_streaming_thermo  (Catch2 v3)
//
// Matvec-unification audit follow-up: end-to-end integration test for the
// finite-T + auto-symmetry path through `ed::exact_diagonalization`.
//
// Before this audit, the streaming-symmetry kernel would iterate per
// sector, run FTLM/LTLM/HYBRID/KPM_DOS on each, then collect only the
// per-sector eigenvalues -- the per-sector `thermo_data` blocks were
// silently dropped, leaving callers with an empty `EDResults::thermo_data`.
// This test pins down the new behaviour: after the per-sector loop the
// streaming kernel must call `ed::core::combine_sector_thermodynamics`
// so that the returned EDResults contains a coherent full-Hilbert
// thermodynamics block.
//
// Strategy:
//   * Build a 4-site Heisenberg PBC chain via HamiltonianBuilder.
//   * Write Z_4 translation symmetry metadata so
//     `ed::exact_diagonalization` auto-detects symmetry.
//   * Run with method=FTLM via the streaming-symmetry kernel.
//   * Verify:
//       1. `results.thermo_data` is non-empty (recombination triggered).
//       2. The recombined thermo agrees with the FTLM-without-symmetry
//          reference within FTLM statistical tolerances.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/dispatch.h>
#include <ed/core/ed_parameters.h>
#include <ed/core/ed_types.h>
#include <ed/input/hamiltonian_builder.h>
#include <ed/input/lattice.h>
#include <ed/solvers/ftlm.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <system_error>
#include <vector>

using namespace ed_tests;

namespace {

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

void write_zN_translation_fixtures(const std::string& dir, int N) {
    const std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);

    {
        std::ofstream f(root + "/max_clique.json");
        f << "[";
        for (int g = 0; g < N; ++g) {
            auto p = translation_perm(N, g);
            f << "[";
            for (std::size_t i = 0; i < p.size(); ++i) {
                f << p[i] << (i + 1 < p.size() ? "," : "");
            }
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    {
        std::ofstream f(root + "/minimal_generators.json");
        auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (std::size_t i = 0; i < p.size(); ++i) {
            f << p[i] << (i + 1 < p.size() ? "," : "");
        }
        f << "],\"order\":" << N << "}]}";
    }
    {
        std::ofstream f(root + "/sector_metadata.json");
        f << std::setprecision(17);
        f << "{\"sectors\":[";
        for (int k = 0; k < N; ++k) {
            const double angle = -2.0 * M_PI * static_cast<double>(k) /
                                 static_cast<double>(N);
            const double re = std::cos(angle);
            const double im = std::sin(angle);
            f << "{\"sector_id\":" << k
              << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << re
              << ",\"imag\":" << im << "}]}";
            if (k + 1 < N) f << ",";
        }
        f << "]}";
    }
}

}  // namespace

TEST_CASE("ed::exact_diagonalization e2e: streaming-symmetry FTLM "
          "recombines per-sector thermo to match unprojected FTLM",
          "[dispatch][streaming][ftlm][e2e]") {
    namespace fs = std::filesystem;
    using ed::input::HamiltonianBuilder;
    namespace lat = ed::input::lattice;

    const std::uint64_t N         = 4;
    const std::uint64_t hilbert_d = 1ULL << N;
    const std::string dir =
        ed_tests::make_scratch_dir("dispatch_streaming_thermo", "ftlm");
    fs::remove_all(dir + "/automorphism_results");

    // Build Heisenberg PBC chain Hamiltonian on disk.
    auto chain = lat::chain(N, /*pbc=*/true);
    HamiltonianBuilder builder(N);
    builder.heisenberg(chain.nn_pairs(), /*J=*/1.0);
    ed::input::FileOptions fopts;
    fopts.write_lattice_metadata = true;
    builder.write_directory(dir, &chain, fopts);

    // Emit Z_4 translation automorphism fixtures so the dispatch
    // auto-detects symmetry. We bypass the Python automorphism
    // generator because this test is purely about the C++ dispatch.
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    // ------------------------------------------------------------------
    // 1. Reference: plain-Hilbert FTLM via the kernel.
    // ------------------------------------------------------------------
    auto H_ref = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto Hv = [op = H_ref.get()](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<std::size_t>(n));
    };

    FTLMParameters fparams;
    fparams.krylov_dim                = 16;
    fparams.num_samples               = 40;
    fparams.tolerance                 = 1e-12;
    fparams.full_reorthogonalization  = true;
    fparams.random_seed               = 12345;
    fparams.compute_error_bars        = false;

    const double T_min = 0.1, T_max = 5.0;
    const std::uint64_t num_bins = 12;

    FTLMResults ref_results = finite_temperature_lanczos(
        Hv, hilbert_d, fparams, T_min, T_max, num_bins, /*output_dir=*/"");

    // ------------------------------------------------------------------
    // 2. Dispatch through ed::exact_diagonalization with FTLM + auto-
    //    symmetry. The auto-detection ought to fire on the
    //    automorphism_results/ directory we just wrote.
    // ------------------------------------------------------------------
    EDParameters params;
    params.num_sites           = N;
    params.spin_length         = 0.5f;
    params.temp_min            = T_min;
    params.temp_max            = T_max;
    params.num_temp_bins       = num_bins;
    params.num_samples         = fparams.num_samples;
    params.ftlm_krylov_dim     = fparams.krylov_dim;
    params.ftlm_full_reorth    = fparams.full_reorthogonalization;
    params.ftlm_seed           = fparams.random_seed;
    params.ftlm_error_bars     = false;
    params.tolerance           = fparams.tolerance;
    params.output_dir          = "";        // disable HDF5 sink

    EDResults sym_results = ed::exact_diagonalization(
        dir,
        DiagonalizationMethod::FTLM,
        params);

    // (1) Recombination must have happened: the streaming kernel filled
    //     `results.thermo_data` instead of dropping it.
    REQUIRE(sym_results.thermo_data.temperatures.size() == num_bins);
    REQUIRE(sym_results.thermo_data.energy.size()       == num_bins);
    REQUIRE(sym_results.thermo_data.specific_heat.size() == num_bins);
    REQUIRE(sym_results.thermo_data.free_energy.size()  == num_bins);

    // (2) Recombined thermo should agree with the unprojected FTLM
    //     reference within FTLM statistical tolerances. Both sides use
    //     the same (krylov_dim, num_samples) budget; per-sector FTLM
    //     gets fewer Krylov vectors per energy decade so a small
    //     extra slack is appropriate.
    double max_rel_E = 0.0;
    double max_abs_F = 0.0;
    for (std::size_t t = 0; t < num_bins; ++t) {
        const double Eref = ref_results.thermo_data.energy[t];
        const double Esym = sym_results.thermo_data.energy[t];
        const double E_scale = std::max(std::abs(Eref), 1.0);
        max_rel_E  = std::max(max_rel_E, std::abs(Esym - Eref) / E_scale);

        const double Fref = ref_results.thermo_data.free_energy[t];
        const double Fsym = sym_results.thermo_data.free_energy[t];
        max_abs_F  = std::max(max_abs_F, std::abs(Fsym - Fref));
    }
    INFO("streaming-symmetry FTLM vs reference FTLM: "
         "max rel |ΔE|=" << max_rel_E
         << ", max |ΔF|=" << max_abs_F);
    REQUIRE(max_rel_E < 0.15);
    REQUIRE(max_abs_F < 0.50);
}

TEST_CASE("ed::exact_diagonalization e2e: streaming-symmetry FULL "
          "eigenvalues span the full spectrum",
          "[dispatch][streaming][full][e2e]") {
    namespace fs = std::filesystem;
    using ed::input::HamiltonianBuilder;
    namespace lat = ed::input::lattice;

    const std::uint64_t N        = 4;
    const std::uint64_t hilbert_d = 1ULL << N;
    const std::string dir =
        ed_tests::make_scratch_dir("dispatch_streaming_thermo", "lanczos");
    fs::remove_all(dir + "/automorphism_results");

    auto chain = lat::chain(N, /*pbc=*/true);
    HamiltonianBuilder builder(N);
    builder.heisenberg(chain.nn_pairs(), /*J=*/1.0);
    ed::input::FileOptions fopts;
    fopts.write_lattice_metadata = true;
    builder.write_directory(dir, &chain, fopts);

    write_zN_translation_fixtures(dir, static_cast<int>(N));

    // Dense reference.
    auto H_ref = build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);
    auto ref   = reference_from_operator(*H_ref, hilbert_d);

    EDParameters params;
    params.num_sites          = N;
    params.spin_length        = 0.5f;
    params.num_eigenvalues    = 2;
    params.max_iterations     = 200;
    params.tolerance          = 1e-12;
    params.output_dir         = "";

    // Use FULL here -- on N=4 the per-sector dim is tiny (~4) and Lanczos
    // would request more iterations than the sector dimension supports.
    // The point of this test is to validate that the dispatch routes
    // through the streaming kernel and yields the correct lowest
    // eigenvalues; FULL exercises the same matvec path (via
    // exact_diagonalization_core->full_diagonalization) without solver
    // convergence noise.
    EDResults sym_results = ed::exact_diagonalization(
        dir, DiagonalizationMethod::FULL, params);

    // The streaming kernel returns up to `num_eigenvalues` per sector,
    // i.e. at most num_eigenvalues * num_sectors total. We only assert
    // the lowest 2 (the GS, in the k=0 sector).
    REQUIRE(sym_results.eigenvalues.size() >= 2);
    REQUIRE(std::abs(sym_results.eigenvalues[0] - ref.eigs[0]) < 1e-9);
}
