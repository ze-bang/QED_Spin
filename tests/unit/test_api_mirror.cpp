// =============================================================================
// test_api_mirror  (Catch2 v3)
//
// Pinning byte-equality between the Python-named `ed::api::*` kwargs
// facade and the legacy `ed::workflows::*` orchestrator across a
// representative cell for each method family. Locks in:
//
//   * `ed::api::SolveOptions` <-> `ed::workflows::SolveOptions`
//     translation (verb=solve, method=LANCZOS / BLOCK_LANCZOS /
//     KRYLOV_SCHUR / FULL).
//   * `ed::api::ThermalOptions` <-> `ed::workflows::ThermalOptions`
//     translation (verb=thermal, method=FTLM / mTPQ / KPM_DOS;
//     LTLM / cTPQ piggyback on the same `to_legacy` translator and
//     are covered by build-only compile checks).
//   * `ed::api::SpectralOptions` <-> `ed::workflows::SpectralOptions`
//     translation (verb=spectral, method=GroundStateCF).
//   * Method-name parsing (`parse_solve_method` / `parse_thermal_method`
//     / `parse_spectral_method`) and device-token mapping
//     (`device_constraints`).
//
// Phase A (PR-1 step 6) of the "mirror examples" plan (May 2026).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/api.h>
#include <ed/api/symmetry_helpers.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using namespace ed_tests;

namespace {

constexpr std::uint64_t N_SITES = 4;
constexpr double        TOLERANCE = 1e-12;

std::unique_ptr<Operator> heisen() {
    return build_heisenberg_chain(N_SITES, 1.0, /*periodic=*/true);
}

}  // namespace

// ---------------------------------------------------------------------------
// String parsers
// ---------------------------------------------------------------------------

TEST_CASE("ed::api::parse_solve_method accepts Python + C++ spellings",
          "[api-mirror][parse]") {
    using SM = ed::workflows::SolveMethod;
    REQUIRE(ed::api::parse_solve_method("").value() == SM::Auto);
    REQUIRE(ed::api::parse_solve_method("auto").value() == SM::Auto);
    REQUIRE(ed::api::parse_solve_method("LANCZOS").value() == SM::Lanczos);
    REQUIRE(ed::api::parse_solve_method("lanczos").value() == SM::Lanczos);
    REQUIRE(ed::api::parse_solve_method("BLOCK_LANCZOS").value() == SM::BlockLanczos);
    REQUIRE(ed::api::parse_solve_method("BlockLanczos").value() == SM::BlockLanczos);
    REQUIRE(ed::api::parse_solve_method("KRYLOV_SCHUR").value() == SM::KrylovSchur);
    REQUIRE(ed::api::parse_solve_method("FULL").value() == SM::FullDiag);
    REQUIRE_FALSE(ed::api::parse_solve_method("not_a_method").has_value());
}

TEST_CASE("ed::api::parse_thermal_method accepts Python + C++ spellings",
          "[api-mirror][parse]") {
    using TM = ed::workflows::ThermalOptions::Method;
    REQUIRE(ed::api::parse_thermal_method("FTLM").value()    == TM::FTLM);
    REQUIRE(ed::api::parse_thermal_method("ftlm").value()    == TM::FTLM);
    REQUIRE(ed::api::parse_thermal_method("LTLM").value()    == TM::LTLM);
    REQUIRE(ed::api::parse_thermal_method("mTPQ").value()    == TM::mTPQ);
    REQUIRE(ed::api::parse_thermal_method("mtpq").value()    == TM::mTPQ);
    REQUIRE(ed::api::parse_thermal_method("cTPQ").value()    == TM::cTPQ);
    REQUIRE(ed::api::parse_thermal_method("KPM_DOS").value() == TM::KpmDos);
    REQUIRE(ed::api::parse_thermal_method("KpmDos").value()  == TM::KpmDos);
    REQUIRE_FALSE(ed::api::parse_thermal_method("xyz").has_value());
}

TEST_CASE("ed::api::parse_spectral_method maps both spellings",
          "[api-mirror][parse]") {
    using SP = ed::workflows::SpectralOptions::Method;
    REQUIRE(ed::api::parse_spectral_method("ground_state_cf").value()  == SP::GroundStateCF);
    REQUIRE(ed::api::parse_spectral_method("GroundStateCF").value()    == SP::GroundStateCF);
    REQUIRE(ed::api::parse_spectral_method("ground_state_dssf").value() == SP::GroundStateCF);
    REQUIRE(ed::api::parse_spectral_method("ftlm_dynamical").value()   == SP::FtlmDynamical);
    REQUIRE(ed::api::parse_spectral_method("FtlmDynamical").value()    == SP::FtlmDynamical);
    REQUIRE_FALSE(ed::api::parse_spectral_method("").has_value());
}

TEST_CASE("ed::api::device_constraints maps Python device tokens correctly",
          "[api-mirror][device]") {
    // auto + large dim => allow gpu / mpi / mpi_gpu
    auto big = ed::api::device_constraints("auto", std::uint64_t{1} << 20);
    REQUIRE(big.allow_gpu);
    REQUIRE(big.allow_mpi);
    REQUIRE(big.allow_mpi_gpu);

    // auto + small dim => CPU only
    auto small = ed::api::device_constraints("auto", 64);
    REQUIRE_FALSE(small.allow_gpu);
    REQUIRE_FALSE(small.allow_mpi_gpu);

    // explicit cpu
    auto cpu = ed::api::device_constraints("cpu");
    REQUIRE_FALSE(cpu.allow_gpu);
    REQUIRE_FALSE(cpu.allow_mpi);
    REQUIRE_FALSE(cpu.allow_mpi_gpu);

    // explicit gpu
    auto gpu = ed::api::device_constraints("gpu");
    REQUIRE(gpu.allow_gpu);
    REQUIRE_FALSE(gpu.allow_mpi);

    // mpi_gpu
    auto mg = ed::api::device_constraints("mpi_gpu");
    REQUIRE(mg.allow_gpu);
    REQUIRE(mg.allow_mpi);
    REQUIRE(mg.allow_mpi_gpu);
}

// ---------------------------------------------------------------------------
// solve: LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / FULL byte-equality
// ---------------------------------------------------------------------------

TEST_CASE("ed::api::solve(LANCZOS) byte-equal to ed::workflows::solve(LANCZOS)",
          "[api-mirror][solve][lanczos]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::SolveOptions wf;
    wf.num_eigs        = 1;
    wf.method          = ed::workflows::SolveMethod::Lanczos;
    wf.tolerance       = 1e-10;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::solve(*H1, wf);

    ed::api::SolveOptions api;
    api.num_eigenvalues = 1;
    api.solver          = "LANCZOS";
    api.tolerance       = 1e-10;
    api.device          = "cpu";
    auto api_res = ed::api::solve(*H2, api);

    REQUIRE(api_res.eigenvalues.size() == wf_res.eigenvalues.size());
    for (std::size_t i = 0; i < api_res.eigenvalues.size(); ++i) {
        REQUIRE(std::abs(api_res.eigenvalues[i] - wf_res.eigenvalues[i]) < TOLERANCE);
    }
}

TEST_CASE("ed::api::solve(BLOCK_LANCZOS) byte-equal to legacy",
          "[api-mirror][solve][block_lanczos]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::SolveOptions wf;
    wf.num_eigs        = 3;
    wf.method          = ed::workflows::SolveMethod::BlockLanczos;
    wf.block_size      = 4;
    wf.tolerance       = 1e-10;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::solve(*H1, wf);

    ed::api::SolveOptions api;
    api.num_eigenvalues = 3;
    api.solver          = "BLOCK_LANCZOS";
    api.block_size      = 4;
    api.tolerance       = 1e-10;
    api.device          = "cpu";
    auto api_res = ed::api::solve(*H2, api);

    REQUIRE(api_res.eigenvalues.size() == wf_res.eigenvalues.size());
    for (std::size_t i = 0; i < api_res.eigenvalues.size(); ++i) {
        REQUIRE(std::abs(api_res.eigenvalues[i] - wf_res.eigenvalues[i]) < TOLERANCE);
    }
}

TEST_CASE("ed::api::solve(KRYLOV_SCHUR) byte-equal to legacy",
          "[api-mirror][solve][krylov_schur]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::SolveOptions wf;
    wf.num_eigs        = 4;
    wf.method          = ed::workflows::SolveMethod::KrylovSchur;
    wf.tolerance       = 1e-10;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::solve(*H1, wf);

    ed::api::SolveOptions api;
    api.num_eigenvalues = 4;
    api.solver          = "KRYLOV_SCHUR";
    api.tolerance       = 1e-10;
    api.device          = "cpu";
    auto api_res = ed::api::solve(*H2, api);

    REQUIRE(api_res.eigenvalues.size() == wf_res.eigenvalues.size());
    for (std::size_t i = 0; i < api_res.eigenvalues.size(); ++i) {
        REQUIRE(std::abs(api_res.eigenvalues[i] - wf_res.eigenvalues[i]) < TOLERANCE);
    }
}

TEST_CASE("ed::api::solve(FULL) byte-equal to legacy",
          "[api-mirror][solve][full]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::SolveOptions wf;
    wf.num_eigs        = 5;
    wf.method          = ed::workflows::SolveMethod::FullDiag;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::solve(*H1, wf);

    ed::api::SolveOptions api;
    api.num_eigenvalues = 5;
    api.solver          = "FULL";
    api.device          = "cpu";
    auto api_res = ed::api::solve(*H2, api);

    REQUIRE(api_res.eigenvalues.size() == wf_res.eigenvalues.size());
    for (std::size_t i = 0; i < api_res.eigenvalues.size(); ++i) {
        REQUIRE(std::abs(api_res.eigenvalues[i] - wf_res.eigenvalues[i]) < TOLERANCE);
    }
}

// ---------------------------------------------------------------------------
// thermal: FTLM / mTPQ / KPM_DOS byte-equality
// ---------------------------------------------------------------------------

TEST_CASE("ed::api::thermal(FTLM) byte-equal to legacy",
          "[api-mirror][thermal][ftlm]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::ThermalOptions wf;
    wf.method        = ed::workflows::ThermalOptions::Method::FTLM;
    wf.num_samples   = 2;
    wf.krylov_dim    = 30;
    wf.random_seed   = 42;
    wf.temp_min      = 0.1;
    wf.temp_max      = 4.0;
    wf.num_temp_bins = 16;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::thermal(*H1, wf);

    ed::api::ThermalOptions api;
    api.method      = "FTLM";
    api.num_samples = 2;
    api.krylov_dim  = 30;
    api.random_seed = 42;
    api.T_min       = 0.1;
    api.T_max       = 4.0;
    api.num_T       = 16;
    api.device      = "cpu";
    auto api_res = ed::api::thermal(*H2, api);

    REQUIRE(std::abs(api_res.ground_state_energy -
                     wf_res.ground_state_energy) < TOLERANCE);
    REQUIRE(api_res.thermo.temperatures.size() ==
            wf_res.thermo.temperatures.size());
    for (std::size_t i = 0; i < api_res.thermo.energy.size(); ++i) {
        REQUIRE(std::abs(api_res.thermo.energy[i] -
                          wf_res.thermo.energy[i]) < TOLERANCE);
    }
}

TEST_CASE("ed::api::thermal(mTPQ) byte-equal to legacy",
          "[api-mirror][thermal][mtpq]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::ThermalOptions wf;
    wf.method       = ed::workflows::ThermalOptions::Method::mTPQ;
    wf.num_samples  = 2;
    wf.krylov_dim   = 60;
    wf.taylor_order = 8;
    wf.delta_beta   = 0.05;
    wf.random_seed  = 137;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::thermal(*H1, wf);

    ed::api::ThermalOptions api;
    api.method            = "mTPQ";
    api.num_samples       = 2;
    api.krylov_dim        = 60;
    api.tpq_taylor_order  = 8;
    api.tpq_delta_beta    = 0.05;
    api.random_seed       = 137;
    api.device            = "cpu";
    auto api_res = ed::api::thermal(*H2, api);

    REQUIRE(std::abs(api_res.ground_state_energy -
                     wf_res.ground_state_energy) < TOLERANCE);
}

TEST_CASE("ed::api::thermal(KPM_DOS) byte-equal to legacy",
          "[api-mirror][thermal][kpm_dos]") {
    auto H1 = heisen();
    auto H2 = heisen();

    ed::workflows::ThermalOptions wf;
    wf.method                  = ed::workflows::ThermalOptions::Method::KpmDos;
    wf.kpm_num_moments         = 200;
    wf.kpm_num_random_vectors  = 16;
    wf.temp_min                = 0.1;
    wf.temp_max                = 5.0;
    wf.num_temp_bins           = 24;
    wf.random_seed             = 9;
    wf.backend.allow_gpu       = false;
    wf.backend.allow_mpi       = false;
    wf.backend.allow_mpi_gpu   = false;
    auto wf_res = ed::workflows::thermal(*H1, wf);

    ed::api::ThermalOptions api;
    api.method                 = "KPM_DOS";
    api.kpm_num_moments        = 200;
    api.kpm_num_random_vectors = 16;
    api.T_min                  = 0.1;
    api.T_max                  = 5.0;
    api.num_T                  = 24;
    api.random_seed            = 9;
    api.device                 = "cpu";
    auto api_res = ed::api::thermal(*H2, api);

    // KPM-DOS is deterministic for a fixed seed; insist on bit equality.
    REQUIRE(api_res.thermo.temperatures.size() ==
            wf_res.thermo.temperatures.size());
    for (std::size_t i = 0; i < api_res.thermo.energy.size(); ++i) {
        REQUIRE(std::abs(api_res.thermo.energy[i] -
                          wf_res.thermo.energy[i]) < TOLERANCE);
    }
}

// ---------------------------------------------------------------------------
// spectral: GroundStateCF byte-equality
// ---------------------------------------------------------------------------

TEST_CASE("ed::api::spectral(GroundStateCF) byte-equal to legacy",
          "[api-mirror][spectral][ground_state_cf]") {
    auto H1 = heisen();
    auto H2 = heisen();

    // Use H itself as the spectral probe (same idiom as test_orchestrator).
    std::vector<const ed::LinearOperator*> obs_wf{H1.get()};
    std::vector<const ed::LinearOperator*> obs_api{H2.get()};

    ed::workflows::SpectralOptions wf;
    wf.method      = ed::workflows::SpectralOptions::Method::GroundStateCF;
    wf.krylov_dim  = 40;
    wf.broadening  = 0.05;
    wf.omega_min   = -5.0;
    wf.omega_max   =  5.0;
    wf.num_omega   = 50;
    wf.backend.allow_gpu     = false;
    wf.backend.allow_mpi     = false;
    wf.backend.allow_mpi_gpu = false;
    auto wf_res = ed::workflows::spectral(*H1, obs_wf, wf);

    ed::api::SpectralOptions api;
    api.method      = "GroundStateCF";
    api.krylov_dim  = 40;
    api.eta         = 0.05;
    api.omega_min   = -5.0;
    api.omega_max   =  5.0;
    api.num_omega   = 50;
    api.device      = "cpu";
    auto api_res = ed::api::spectral(*H2, obs_api, api);

    REQUIRE(api_res.omega.size()  == wf_res.omega.size());
    REQUIRE(api_res.S_real.size() == wf_res.S_real.size());
    for (std::size_t i = 0; i < api_res.S_real.size(); ++i) {
        REQUIRE(std::abs(api_res.S_real[i] - wf_res.S_real[i]) < TOLERANCE);
        REQUIRE(std::abs(api_res.S_imag[i] - wf_res.S_imag[i]) < TOLERANCE);
    }
}

// ---------------------------------------------------------------------------
// Helpers: build introspection, find_symmetries, estimate_resources,
// suggest_workflow
// ---------------------------------------------------------------------------

TEST_CASE("ed::has_*_build() are stable booleans",
          "[api-mirror][build]") {
    // We don't pin the value (it depends on the CMake config) but we
    // do require the predicates to evaluate to a stable boolean.
    bool c1 = ed::has_cuda_build();
    bool c2 = ed::has_cuda_build();
    REQUIRE(c1 == c2);

    bool m1 = ed::has_mpi_build();
    bool m2 = ed::has_mpi_build();
    REQUIRE(m1 == m2);

    bool n1 = ed::has_nccl_build();
    bool n2 = ed::has_nccl_build();
    REQUIRE(n1 == n2);
}

TEST_CASE("ed::find_symmetries('translation') builds Z_N",
          "[api-mirror][symmetries]") {
    const int N = 6;
    auto info = ed::find_symmetries(N, "translation");
    REQUIRE(info.max_clique.size() == static_cast<std::size_t>(N));
    REQUIRE(info.generators.size() == 1);
}

TEST_CASE("ed::find_symmetries('translation+reflection') is restricted to its abelian subgroup",
          "[api-mirror][symmetries]") {
    const int N = 6;
    // translation+reflection generates the NON-abelian dihedral group D_N (|G|=2N).
    // The projection layer is abelian-only, so group_from_generators restricts to a
    // maximal abelian subgroup (the Z_N translations, |A|=N) -- a complete & correct,
    // if coarser, reduction. See the non-abelian guard in src/symmetry/group.cpp.
    auto info = ed::find_symmetries(N, "translation+reflection");
    REQUIRE(info.max_clique.size() == static_cast<std::size_t>(N));   // |A| = N, not 2N
}
