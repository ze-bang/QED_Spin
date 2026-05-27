// =============================================================================
// test_thermal_save  (Catch2 v3)
//
// Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026): pins the uniform
// persistence contract for `ed::workflows::thermal(H, opts)`.
//
// Contract:
//   * When ``opts.output_dir`` is set and the run is single-rank, the
//     orchestrator writes ``<output_dir>/ed_results.h5`` and surfaces the
//     resulting path via ``ThermalResult::hdf5_path`` (the same field the
//     Python facade mirrors into ``EDResults.eigenvectors_path``).
//   * FTLM / LTLM / KPM_DOS persist the aggregated thermodynamic curves
//     (``T, E, Cv, S, F``) under ``/ftlm/averaged/<...>``.
//   * mTPQ / cTPQ persist the per-sample trajectory rows
//     (``/tpq/samples/sample_<s>/thermodynamics``); when
//     ``opts.probe_betas`` is set they ALSO snapshot the running state
//     vector at the nearest kernel-step beta and write it to
//     ``/tpq/samples/sample_<s>/states/beta_<b>``.
//   * mTPQ ``ThermalResult::tpq_state_snapshots`` carries the host-side
//     state vectors so callers can chain them into ``ed::workflows::spectral``
//     (the TPQ-to-CF pipeline) without an HDF5 round-trip.
//
// Covered methods: FTLM, LTLM, KPM_DOS (thermo-only) + mTPQ, cTPQ
// (trajectory + probe-beta state snapshots).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/hdf5_io.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;

namespace {

constexpr std::uint64_t N_SITES = 4;
using Complex = std::complex<double>;

std::unique_ptr<Operator> heisen() {
    return build_heisenberg_chain(N_SITES, 1.0, /*periodic=*/true);
}

bool dataset_exists(const std::string& filepath, const std::string& path) {
    try {
        H5::H5File f(filepath, H5F_ACC_RDONLY);
        return f.exists(path);
    } catch (const H5::Exception&) {
        return false;
    }
}

}  // namespace

TEST_CASE("ed::thermal persists FTLM thermo curves",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "ftlm");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::FTLM;
    opts.num_samples   = 2;
    opts.krylov_dim    = 30;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/ftlm/averaged/energy"));
    CHECK(dataset_exists(h5, "/ftlm/averaged/temperatures"));
    // Thermo carrier on the in-memory result.
    CHECK(R.thermo.energy.size() == opts.num_temp_bins);

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists LTLM thermo curves",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "ltlm");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::LTLM;
    opts.num_samples   = 2;
    opts.krylov_dim    = 30;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/ftlm/averaged/energy"));
    CHECK(dataset_exists(h5, "/ftlm/averaged/temperatures"));

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists KpmDos thermo curves",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "kpm");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::KpmDos;
    opts.num_samples   = 2;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.kpm_num_moments        = 64;
    opts.kpm_num_random_vectors = 4;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/ftlm/averaged/energy"));
    CHECK(dataset_exists(h5, "/ftlm/averaged/temperatures"));

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists mTPQ trajectory + probe-beta state snapshots",
          "[orchestrator][thermal-save][tpq]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "mtpq");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::mTPQ;
    opts.num_samples   = 1;
    opts.krylov_dim    = 40;        // = mTPQ max_iter
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;
    opts.probe_betas   = {1.0, 5.0};

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/tpq/samples/sample_0/thermodynamics"));
    // In-memory trajectory mirror.
    REQUIRE(R.tpq_sample_betas.size() == opts.num_samples);
    REQUIRE_FALSE(R.tpq_sample_betas[0].empty());
    REQUIRE(R.tpq_sample_energies[0].size() == R.tpq_sample_betas[0].size());

    // State snapshots: at most one per (sample, probe_beta).
    REQUIRE_FALSE(R.tpq_state_snapshots.empty());
    for (const auto& snap : R.tpq_state_snapshots) {
        REQUIRE(snap.psi.size() == (std::size_t{1} << N_SITES));
        // Cross-check the on-disk dataset for the same beta exists.
        char beta_buf[64];
        std::snprintf(beta_buf, sizeof(beta_buf),
                      "/tpq/samples/sample_%zu/states/beta_%.6f",
                      snap.sample_index, snap.effective_beta);
        CHECK(dataset_exists(h5, std::string(beta_buf)));
    }

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists cTPQ trajectory + probe-beta state snapshots",
          "[orchestrator][thermal-save][tpq]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "ctpq");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::cTPQ;
    opts.num_samples   = 1;
    opts.krylov_dim    = 40;          // cap on Taylor steps
    opts.delta_beta    = 0.1;
    opts.taylor_order  = 8;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;
    opts.probe_betas   = {1.0, 2.0};

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/tpq/samples/sample_0/thermodynamics"));
    REQUIRE(R.tpq_sample_betas.size() == opts.num_samples);
    REQUIRE_FALSE(R.tpq_state_snapshots.empty());
    for (const auto& snap : R.tpq_state_snapshots) {
        REQUIRE(snap.psi.size() == (std::size_t{1} << N_SITES));
    }

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal leaves hdf5_path empty when output_dir is unset",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::FTLM;
    opts.num_samples   = 1;
    opts.krylov_dim    = 16;
    opts.temp_min      = 0.5;
    opts.temp_max      = 2.0;
    opts.num_temp_bins = 4;
    // output_dir intentionally empty.
    auto R = ed::workflows::thermal(*H, opts);
    CHECK(R.hdf5_path.empty());
}

TEST_CASE("ed::thermal honours /dev/null sentinel",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::FTLM;
    opts.num_samples   = 1;
    opts.krylov_dim    = 16;
    opts.temp_min      = 0.5;
    opts.temp_max      = 2.0;
    opts.num_temp_bins = 4;
    opts.output_dir    = "/dev/null";
    auto R = ed::workflows::thermal(*H, opts);
    CHECK(R.hdf5_path.empty());
}
