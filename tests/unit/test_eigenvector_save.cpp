// =============================================================================
// test_eigenvector_save  (Catch2 v3)
//
// Locks in the May 2026 contract for `ed::workflows::solve(H, opts)` and the
// Python-named `ed::api::solve(...)` facade: whenever
//
//     opts.compute_vectors == true       (Python: compute_eigenvectors=True)
//     opts.output_dir      != ""         (Python: output_dir="...")
//
// the solver MUST persist eigenvalues + eigenvectors to
// `<output_dir>/ed_results.h5` and surface the path via
// `GroundStateResult::hdf5_path` (mirrored into Python's
// `EDResults.eigenvectors_path`). Before this test, only the FullDiag lane
// honored the contract: Lanczos / BlockLanczos / KrylovSchur silently
// dropped `output_dir` (kernel `Options::output_dir` was declared but not
// read by the kernel, so `R.hdf5_path` stayed empty and downstream Python
// `EDResults.eigenvectors_path` was always "").
//
// This suite walks every CPU-single-rank solver method, runs `solve` with
// `compute_vectors=true` against a 4-site Heisenberg ring, and asserts:
//
//   1. The HDF5 file exists at `<output_dir>/ed_results.h5`.
//   2. `R.hdf5_path` reports that file.
//   3. `HDF5IO::loadEigenvalues` round-trips the eigenvalues bit-for-bit.
//   4. `HDF5IO::loadEigenvector` round-trips the ground-state vector to
//      within a Hilbert-space phase (|<psi_loaded|psi_memory>| ~ 1).
//
// Plus the Python-facing `ed::api::solve(...)` smoke (one cell) asserting
// the facade preserves the same hdf5_path contract.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/api.h>
#include <ed/core/hdf5_io.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <complex>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;

namespace {

constexpr std::uint64_t N_SITES = 4;
constexpr double        TOLERANCE = 1e-10;
using Complex = std::complex<double>;

std::unique_ptr<Operator> heisen() {
    return build_heisenberg_chain(N_SITES, 1.0, /*periodic=*/true);
}

// |<u|v>| -- phase-agnostic overlap magnitude.
double overlap_magnitude(const std::vector<Complex>& u,
                         const std::vector<Complex>& v) {
    REQUIRE(u.size() == v.size());
    Complex acc{0.0, 0.0};
    for (std::size_t i = 0; i < u.size(); ++i) acc += std::conj(u[i]) * v[i];
    return std::abs(acc);
}

void check_solve_persists(ed::workflows::SolveMethod method,
                          const std::string& tag) {
    auto H = heisen();

    const std::string outdir = make_scratch_dir("solve_save", tag);

    ed::workflows::SolveOptions opts;
    opts.num_eigs         = 1;
    opts.method           = method;
    opts.compute_vectors  = true;
    opts.tolerance        = 1e-12;
    opts.output_dir       = outdir;
    // BlockLanczos / KrylovSchur prefer a small block at this dim.
    opts.block_size       = 2;

    auto R = ed::workflows::solve(*H, opts);

    const std::string h5_path = outdir + "/ed_results.h5";

    INFO("method=" << tag << " outdir=" << outdir);
    CHECK(std::filesystem::exists(h5_path));
    CHECK(R.hdf5_path == h5_path);

    // Round-trip eigenvalues.
    auto e_loaded = HDF5IO::loadEigenvalues(h5_path);
    REQUIRE(e_loaded.size() >= R.eigenvalues.size());
    for (std::size_t i = 0; i < R.eigenvalues.size(); ++i) {
        CHECK(std::abs(e_loaded[i] - R.eigenvalues[i]) < TOLERANCE);
    }

    // Round-trip the ground-state vector.
    //
    // FullDiag persists per `saveDiagonalizationResults` (eigenvectors live
    // in the HDF5 file but never land in `R.eigenvectors->host` -- the
    // orchestrator drops the dense vectors after the LAPACK call). For the
    // other three methods we compare against `R.eigenvectors->host[0]`.
    auto psi_loaded = HDF5IO::loadEigenvector(h5_path, /*index=*/0);
    REQUIRE(!psi_loaded.empty());
    if (method != ed::workflows::SolveMethod::FullDiag) {
        REQUIRE(R.eigenvectors.has_value());
        REQUIRE(!R.eigenvectors->host.empty());
        const auto& psi_mem = R.eigenvectors->host[0];
        REQUIRE(psi_loaded.size() == psi_mem.size());
        const double ov = overlap_magnitude(psi_loaded, psi_mem);
        // The eigenvector is determined up to a global phase; the
        // magnitude of <psi_loaded|psi_memory> must be very close to 1.
        CHECK(ov > 1.0 - 1e-9);
    }

    std::filesystem::remove_all(outdir);
}

}  // namespace

TEST_CASE("ed::workflows::solve persists eigenvectors for Lanczos",
          "[orchestrator][eigenvector-save]") {
    check_solve_persists(ed::workflows::SolveMethod::Lanczos, "lanczos");
}

TEST_CASE("ed::workflows::solve persists eigenvectors for BlockLanczos",
          "[orchestrator][eigenvector-save]") {
    check_solve_persists(ed::workflows::SolveMethod::BlockLanczos,
                         "block_lanczos");
}

TEST_CASE("ed::workflows::solve persists eigenvectors for KrylovSchur",
          "[orchestrator][eigenvector-save]") {
    check_solve_persists(ed::workflows::SolveMethod::KrylovSchur,
                         "krylov_schur");
}

TEST_CASE("ed::workflows::solve persists eigenvectors for FullDiag",
          "[orchestrator][eigenvector-save]") {
    check_solve_persists(ed::workflows::SolveMethod::FullDiag, "full");
}

TEST_CASE("ed::api::solve mirrors the hdf5_path contract",
          "[api-mirror][eigenvector-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("api_save", "lanczos");

    ed::api::SolveOptions opts;
    opts.num_eigenvalues     = 1;
    opts.solver              = "lanczos";
    opts.compute_eigenvectors = true;
    opts.tolerance           = 1e-12;
    opts.output_dir          = outdir;

    auto R = ed::api::solve(*H, opts);

    CHECK(std::filesystem::exists(outdir + "/ed_results.h5"));
    CHECK(R.hdf5_path == outdir + "/ed_results.h5");
    REQUIRE(R.eigenvectors.has_value());
    REQUIRE(!R.eigenvectors->host.empty());

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::workflows::solve leaves hdf5_path empty when output_dir is unset",
          "[orchestrator][eigenvector-save]") {
    auto H = heisen();
    ed::workflows::SolveOptions opts;
    opts.num_eigs        = 1;
    opts.method          = ed::workflows::SolveMethod::Lanczos;
    opts.compute_vectors = true;
    opts.tolerance       = 1e-12;
    // output_dir intentionally empty.
    auto R = ed::workflows::solve(*H, opts);
    CHECK(R.hdf5_path.empty());
    REQUIRE(R.eigenvectors.has_value());
    REQUIRE(!R.eigenvectors->host.empty());
}

TEST_CASE("ed::workflows::solve honors /dev/null sentinel",
          "[orchestrator][eigenvector-save]") {
    auto H = heisen();
    ed::workflows::SolveOptions opts;
    opts.num_eigs        = 1;
    opts.method          = ed::workflows::SolveMethod::Lanczos;
    opts.compute_vectors = true;
    opts.tolerance       = 1e-12;
    opts.output_dir      = "/dev/null";  // disabled sentinel
    auto R = ed::workflows::solve(*H, opts);
    CHECK(R.hdf5_path.empty());
}
