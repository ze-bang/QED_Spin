// =============================================================================
// tests/integration/test_unified_interface_e2e.cpp
//
// End-to-end acceptance suite for the unified ED interface
// (`ed::make_operator + ed::workflows::{solve,thermal,spectral}`) shipped
// by the Full Unified-Interface Collapse (May 2026).
//
// Goal: a single test binary that drives every defined use case of the
// unified surface against the same reference Hamiltonian and asserts
// numerical agreement against textbook values. Catch2 organises the
// matrix as one SECTION per use case so a single failure pinpoints the
// broken cell without manual bisection.
//
// Reference: 6-site periodic AFM Heisenberg ring at J = 1.
//
//   * Ground state E_0 = -2.802775637731995   (Bethe ansatz)
//   * The ground state lives in the Sz = 0 sector (n_up = 3).
//   * Sz = 0 sector dimension = 6 choose 3 = 20.
//   * Sz = 0 ground state E_0^{Sz=0} = -2.802775637731995   (same value)
//
// Use case matrix exercised below:
//
//   [A] OperatorSpec source variants
//     A1. InMemoryOperator                            -- programmatic
//     A2. FilePaths (InterAll.dat, no Trans / Counter / Three)
//     A3. DirectoryPath                               -- same files
//                                                       on disk
//   [B] Spec axes
//     B1. fixed_sz: project to the n_up = 3 sector
//     B2. (streaming_symmetry / distributed deferred to per-feature
//          tests; the factory throw-paths are already covered by
//          test_make_operator.cpp.)
//   [C] Workflow + method combinations
//     C1. workflows::solve with SolveMethod::Lanczos        -> E_0
//     C2. workflows::solve with SolveMethod::FullDiag       -> E_0
//     C3. workflows::solve with SolveMethod::KrylovSchur    -> 4 eigs
//     C4. workflows::solve with SolveMethod::BlockLanczos   -> 4 eigs
//     C5. workflows::solve with SolveMethod::Auto           -> auto-pick
//     C6. workflows::thermal with mTPQ                      -> backend
//                                                              templated
//     C7. workflows::spectral with GroundStateCF            -> non-empty
//   [D] Result-type contract
//     D1. BackendMetadata.lane reads "cpu"
//     D2. KrylovDiagnostics.num_iterations populated on Krylov methods
//     D3. compute_vectors flag round-trips: when true the result
//         carries the ground-state vector.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/linear_operator.h>
#include <ed/core/make_operator.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// Bethe-ansatz reference for the 6-site periodic AFM Heisenberg ring at J=1.
constexpr double kE0_N6 = -2.802775637731995;
// Loose tolerance for Lanczos-with-default-seed convergence on a 64x64
// problem. The unified interface uses a fixed deterministic seed, so the
// tolerance is reproducible but not bit-tight.
constexpr double kLanczosTol = 1e-8;
constexpr double kFullDiagTol = 1e-12;

// Build the 6-site periodic Heisenberg chain on the InMemoryOperator
// lane. Mirrors ed_tests::build_heisenberg_chain but is duplicated here
// so the test file is self-contained and the value coincides exactly
// with the FilePaths + DirectoryPath construction below.
std::unique_ptr<Operator> build_chain_in_memory(std::uint64_t N) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const std::complex<double> one(1.0, 0.0);
    const std::complex<double> half(0.5, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op->addTwoBodyTerm(2, i, 2, j, one);    // Sz_i Sz_j
        op->addTwoBodyTerm(0, i, 1, j, half);   // (1/2) S+_i S-_j
        op->addTwoBodyTerm(1, i, 0, j, half);   // (1/2) S-_i S+_j
    }
    return op;
}

// Write the same 6-site periodic Heisenberg chain to an InterAll.dat
// deck the file-based source variants can consume. 18 lines: three
// terms per bond, 6 bonds, periodic.
void write_chain_interall(const std::filesystem::path& path,
                          std::uint64_t N,
                          double J = 1.0) {
    std::ofstream f(path);
    REQUIRE(f);
    f << "==== InterAll (chain) ====\n";
    f << "num_terms " << (3 * N) << "\n";
    f << "========================\n";
    f << "========================\n";
    f << "========================\n";
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        // Sz Sz
        f << "2 " << i << " 2 " << j << " " << J << " 0.0\n";
        // (1/2) S+_i S-_j
        f << "0 " << i << " 1 " << j << " " << (0.5 * J) << " 0.0\n";
        // (1/2) S-_i S+_j
        f << "1 " << i << " 0 " << j << " " << (0.5 * J) << " 0.0\n";
    }
}

// Plain in-memory Operators advertise supports_device_matvec on WITH_CUDA
// builds (operator-collapse Phase 2a), so the iterative workflows
// (Lanczos / KrylovSchur / mTPQ / FTLM / CF) auto-dispatch to the GPU lane
// whenever a CUDA device is visible -- exactly like the symmetry-sector
// operators already did. Dense FullDiag stays on the CPU lane. Tests that want
// to pin the CPU lane regardless of hardware set opts.backend.allow_gpu=false
// (see the dedicated BackendConstraints case below). These tiny fixtures
// zero ``gpu_dim_floor`` (the Jul-2026 auto-promotion floor) so the GPU
// lane keeps its e2e coverage despite dims far below the production floor.
inline std::string expected_iterative_lane() {
    return ed::have_cuda() ? "gpu" : "cpu";
}

}  // namespace

TEST_CASE("[unified-e2e] InMemoryOperator source through every solve method",
          "[unified][e2e][solve][in_memory]") {
    constexpr std::uint64_t N = 6;

    SECTION("C1: Lanczos converges to the Bethe-ansatz ground state") {
        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
        spec.num_sites = N;
        spec.spin_l    = 0.5f;
        auto op = ed::make_operator(std::move(spec));
        REQUIRE(op);

        ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
        opts.num_eigs        = 1;
        opts.method          = ed::SolveMethod::Lanczos;
        opts.tolerance       = 1e-12;
        opts.max_iter        = 200;
        opts.compute_vectors = false;

        auto r = ed::workflows::solve(*op, opts);
        REQUIRE_FALSE(r.eigenvalues.empty());
        REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);
        REQUIRE(r.backend.lane == expected_iterative_lane());
    }

    SECTION("C2: FullDiag matches the Bethe-ansatz ground state tightly") {
        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
        spec.num_sites = N;
        spec.spin_l    = 0.5f;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
        opts.num_eigs = 4;
        opts.method   = ed::SolveMethod::FullDiag;
        auto r = ed::workflows::solve(*op, opts);

        REQUIRE(r.eigenvalues.size() >= 4);
        REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kFullDiagTol);
        // First excited multiplet is degenerate at E_1 = -1.802...
        REQUIRE(r.eigenvalues[1] > kE0_N6);
        REQUIRE(r.eigenvalues[1] < 0.0);
    }

    SECTION("C3: KrylovSchur recovers 4 lowest eigenvalues") {
        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
        opts.num_eigs   = 4;
        opts.method     = ed::SolveMethod::KrylovSchur;
        opts.tolerance  = 1e-10;
        opts.max_iter   = 100;
        auto r = ed::workflows::solve(*op, opts);

        REQUIRE(r.eigenvalues.size() >= 4);
        REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);
        // Sorted ascending.
        for (std::size_t k = 1; k < 4; ++k) {
            REQUIRE(r.eigenvalues[k] >= r.eigenvalues[k - 1] - 1e-12);
        }
    }

    SECTION("C4: BlockLanczos recovers 4 lowest eigenvalues") {
        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
        opts.num_eigs   = 4;
        opts.block_size = 4;
        opts.method     = ed::SolveMethod::BlockLanczos;
        opts.tolerance  = 1e-10;
        opts.max_iter   = 100;
        auto r = ed::workflows::solve(*op, opts);

        REQUIRE(r.eigenvalues.size() >= 4);
        REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);
    }

    SECTION("C5: Auto picks a method and converges") {
        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
        opts.num_eigs = 1;
        opts.method   = ed::SolveMethod::Auto;
        auto r = ed::workflows::solve(*op, opts);

        REQUIRE_FALSE(r.eigenvalues.empty());
        REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);
    }
}

TEST_CASE("[unified-e2e] FilePaths source loads InterAll.dat and "
          "diagonalises identically",
          "[unified][e2e][solve][file_paths]") {
    constexpr std::uint64_t N = 6;
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "unified_e2e_filepaths";
    fs::create_directories(tmp);
    fs::path inter = tmp / "InterAll.dat";
    write_chain_interall(inter, N);

    ed::OperatorSpec spec;
    spec.source    = ed::FilePaths{inter.string(), "", "", ""};
    spec.num_sites = N;
    spec.spin_l    = 0.5f;
    auto op = ed::make_operator(std::move(spec));
    REQUIRE(op);

    ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.num_eigs  = 1;
    opts.method    = ed::SolveMethod::Lanczos;
    opts.tolerance = 1e-12;
    opts.max_iter  = 200;
    auto r = ed::workflows::solve(*op, opts);

    REQUIRE_FALSE(r.eigenvalues.empty());
    REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);

    fs::remove_all(tmp);
}

TEST_CASE("[unified-e2e] DirectoryPath source loads from a directory and "
          "diagonalises identically",
          "[unified][e2e][solve][directory_path]") {
    constexpr std::uint64_t N = 6;
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "unified_e2e_directory";
    fs::create_directories(tmp);
    write_chain_interall(tmp / "InterAll.dat", N);

    ed::OperatorSpec spec;
    spec.source    = ed::DirectoryPath{tmp.string()};
    spec.num_sites = N;
    spec.spin_l    = 0.5f;
    auto op = ed::make_operator(std::move(spec));

    ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.num_eigs  = 2;
    opts.method    = ed::SolveMethod::Lanczos;
    opts.tolerance = 1e-12;
    opts.max_iter  = 200;
    auto r = ed::workflows::solve(*op, opts);

    REQUIRE(r.eigenvalues.size() >= 2);
    REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);

    fs::remove_all(tmp);
}

TEST_CASE("[unified-e2e] fixed_sz axis projects to the Sz = 0 sector",
          "[unified][e2e][solve][fixed_sz]") {
    constexpr std::uint64_t N = 6;
    namespace fs = std::filesystem;
    fs::path tmp = fs::temp_directory_path() / "unified_e2e_fixed_sz";
    fs::create_directories(tmp);
    write_chain_interall(tmp / "InterAll.dat", N);

    ed::OperatorSpec spec;
    spec.source    = ed::DirectoryPath{tmp.string()};
    spec.num_sites = N;
    spec.fixed_sz  = static_cast<int>(N / 2);  // n_up = 3 -> Sz = 0
    auto op = ed::make_operator(std::move(spec));
    REQUIRE(op);
    // 6 choose 3 = 20.
    REQUIRE(op->dim() == 20);

    ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.num_eigs       = 1;
    opts.method         = ed::SolveMethod::Lanczos;
    opts.tolerance      = 1e-12;
    opts.max_iter       = 200;
    opts.use_fixed_sz   = true;
    opts.n_up           = static_cast<int>(N / 2);

    auto r = ed::workflows::solve(*op, opts);
    REQUIRE_FALSE(r.eigenvalues.empty());
    // Ground state of the 6-site periodic Heisenberg chain lives in Sz = 0
    // -> same Bethe-ansatz value.
    REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);

    fs::remove_all(tmp);
}

TEST_CASE("[unified-e2e] thermal mTPQ end-to-end on the in-memory operator",
          "[unified][e2e][thermal][mtpq]") {
    constexpr std::uint64_t N = 4;  // smaller for thermal stability

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
    spec.num_sites = N;
    auto op = ed::make_operator(std::move(spec));

    ed::ThermalOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.method      = ed::ThermalOptions::Method::mTPQ;
    opts.num_samples = 1;
    opts.krylov_dim  = 50;
    opts.random_seed = 0xC0FFEEULL;

    auto r = ed::workflows::thermal(*op, opts);
    REQUIRE(r.backend.lane == expected_iterative_lane());
    // mTPQ produces a finite-temperature trajectory; with 1 sample the
    // `ground_state_energy` field is the running minimum over the
    // trajectory (high-T plateau down to whatever beta the run reached),
    // not a tight ground-state estimate. The contract we assert here is
    // (i) the lane completed without throwing, (ii) the field is finite
    // and bounded above by the exact dim*0.5 = 2.0 spectrum maximum.
    REQUIRE(std::isfinite(r.ground_state_energy));
    REQUIRE(r.ground_state_energy <= 2.0);
}

TEST_CASE("[unified-e2e] thermal FTLM end-to-end on the in-memory operator",
          "[unified][e2e][thermal][ftlm]") {
    constexpr std::uint64_t N = 4;

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
    spec.num_sites = N;
    auto op = ed::make_operator(std::move(spec));

    ed::ThermalOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.method      = ed::ThermalOptions::Method::FTLM;
    opts.num_samples = 2;
    opts.krylov_dim  = 30;
    opts.random_seed = 0xBEEFULL;
    opts.temp_min    = 0.1;
    opts.temp_max    = 4.0;
    opts.num_temp_bins = 16;

    auto r = ed::workflows::thermal(*op, opts);
    REQUIRE(r.backend.lane == expected_iterative_lane());
}

TEST_CASE("[unified-e2e] spectral CF ground-state on the in-memory operator",
          "[unified][e2e][spectral][cf]") {
    constexpr std::uint64_t N = 4;

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
    spec.num_sites = N;
    auto op = ed::make_operator(std::move(spec));

    // Self-correlator: use H as its own observable. This is a smoke
    // test for the kernel plumbing; the physics doesn't matter here.
    ed::OperatorSpec spec_obs;
    spec_obs.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
    spec_obs.num_sites = N;
    auto obs_op = ed::make_operator(std::move(spec_obs));

    ed::SpectralOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.method     = ed::SpectralOptions::Method::GroundStateCF;
    opts.num_omega  = 16;
    opts.omega_min  = -5.0;
    opts.omega_max  =  5.0;
    opts.broadening = 0.1;
    opts.krylov_dim = 60;

    std::vector<const ed::LinearOperator*> obs{ obs_op.get() };
    auto r = ed::workflows::spectral(*op, obs, opts);

    REQUIRE(r.omega.size() == 16);
    REQUIRE(r.S_real.size() == 16);
    REQUIRE(r.S_imag.size() == 16);
    REQUIRE(r.backend.lane == expected_iterative_lane());
}

TEST_CASE("[unified-e2e] solve carries Krylov diagnostics and backend "
          "metadata",
          "[unified][e2e][diagnostics]") {
    constexpr std::uint64_t N = 6;

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
    spec.num_sites = N;
    auto op = ed::make_operator(std::move(spec));

    ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.num_eigs        = 1;
    opts.method          = ed::SolveMethod::Lanczos;
    opts.tolerance       = 1e-12;
    opts.max_iter        = 200;
    opts.compute_vectors = true;

    auto r = ed::workflows::solve(*op, opts);

    REQUIRE(r.backend.lane == expected_iterative_lane());
    REQUIRE(r.backend.wall_seconds >= 0.0);
    // Krylov-method results expose `iters_done` > 0.
    REQUIRE(r.krylov.iters_done > 0u);
    // compute_vectors=true -> the result optional is engaged and
    // carries at least the ground-state vector in one of the
    // EigenvectorRef storage strategies.
    REQUIRE(r.eigenvectors.has_value());
    const auto& evref = r.eigenvectors.value();
    const bool has_host_eigvecs =
        !evref.host.empty() &&
        evref.host.front().size() == op->dim();
    const bool has_hdf5_eigvecs   = !evref.hdf5_path.empty();
    const bool has_backend_eigvec = evref.on_backend;
    REQUIRE((has_host_eigvecs || has_hdf5_eigvecs || has_backend_eigvec));
}

TEST_CASE("[unified-e2e] BackendConstraints can force the CPU lane "
          "even when CUDA is built",
          "[unified][e2e][backend_constraints]") {
    constexpr std::uint64_t N = 6;

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_chain_in_memory(N)};
    spec.num_sites = N;
    auto op = ed::make_operator(std::move(spec));

    ed::SolveOptions opts;
        opts.backend.gpu_dim_floor = 0;  // tiny fixture: keep exercising the GPU lane
    opts.num_eigs       = 1;
    opts.method         = ed::SolveMethod::Lanczos;
    opts.backend.allow_gpu = false;  // explicit CPU pin
    opts.backend.allow_mpi = false;

    auto r = ed::workflows::solve(*op, opts);
    REQUIRE(r.backend.lane == "cpu");
    REQUIRE(std::abs(r.eigenvalues[0] - kE0_N6) < kLanczosTol);
}
