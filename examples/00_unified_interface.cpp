// =============================================================================
// examples/00_unified_interface.cpp
//
// Canonical end-to-end demonstration of the unified ED interface
// introduced by the Full Unified-Interface Collapse (May 2026).
//
// The shape of every C++ ED workflow now consists of just three lines:
//
//   1. Construct an `OperatorSpec` describing the Hamiltonian source
//      and the orthogonal axes (fixed_sz / streaming_symmetry /
//      distributed).
//   2. Build the operator via `ed::make_operator(spec)`.
//   3. Run the workflow via `ed::workflows::solve(*op, opts)`,
//      `ed::workflows::thermal(*op, opts)`, or
//      `ed::workflows::spectral(*op, observables, opts)`.
//
// The backend (CPU / GPU / MPI / MPI+GPU) is auto-selected internally
// from the operator's `geometry()` plus the constraints supplied in
// the options struct's `backend` field.
//
// To run this example end-to-end:
//
//   cmake -B build -DED_BUILD_EXAMPLES=ON
//   cmake --build build --target ex00_unified_interface
//   ./build/examples/ex00_unified_interface
//
// This example walks through every defined use case of the unified
// interface:
//
//   [1] InMemoryOperator source -- programmatic Hamiltonian
//   [2] InMemoryOperator + fixed_sz axis -- Sz=0 sector
//   [3] FilePaths source -- write InterAll.dat then load it back
//   [4] DirectoryPath source -- same files via a directory route
//   [5] Multiple solve methods -- Lanczos / FullDiag / KrylovSchur
//   [6] compute_vectors = true -- retrieve eigenvectors
//   [7] Thermal mTPQ -- finite-temperature trajectory
//   [8] Spectral GroundStateCF -- dynamical S(omega)
//   [9] BackendConstraints -- pin the CPU lane explicitly
//
// No HDF5, no CLI integration, no environment dependencies -- just
// `Operator -> make_operator -> workflows::*` --- exactly the shape
// every downstream caller (CLI, examples, Python) should adopt.
// =============================================================================

#include <ed/core/make_operator.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

namespace {

std::unique_ptr<Operator> build_heisenberg_chain(std::uint64_t N,
                                                 double J = 1.0,
                                                 bool periodic = true) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        if (!periodic && j == 0) break;
        // S_i \cdot S_j = Sz_i Sz_j + 0.5 (S+_i S-_j + S-_i S+_j).
        op->addTwoBodyTerm(2, i, 2, j,
                           std::complex<double>{J,       0.0});
        op->addTwoBodyTerm(0, i, 1, j,
                           std::complex<double>{0.5 * J, 0.0});
        op->addTwoBodyTerm(1, i, 0, j,
                           std::complex<double>{0.5 * J, 0.0});
    }
    return op;
}

/// Serialize a periodic-Heisenberg chain to an InterAll.dat deck so the
/// file-based source variants (`FilePaths` / `DirectoryPath`) can load
/// it back. Matches the HPhi text format consumed by `ed::make_operator`.
void write_heisenberg_interall(const std::filesystem::path& path,
                               std::uint64_t N,
                               double J = 1.0) {
    std::ofstream f(path);
    f << "==== InterAll (Heisenberg chain) ====\n";
    f << "num_terms " << (3 * N) << "\n";
    f << "========================\n";
    f << "========================\n";
    f << "========================\n";
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        f << "2 " << i << " 2 " << j << " " << J << " 0.0\n";
        f << "0 " << i << " 1 " << j << " " << (0.5 * J) << " 0.0\n";
        f << "1 " << i << " 0 " << j << " " << (0.5 * J) << " 0.0\n";
    }
}

}  // namespace

int main() {
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "===============================================\n";
    std::cout << "  Unified ED interface: end-to-end walkthrough\n";
    std::cout << "===============================================\n\n";

    constexpr std::uint64_t N = 8;

    // -----------------------------------------------------------------
    // 1. The simplest possible call: ground state on a programmatically
    //    built Operator. `make_operator(InMemoryOperator{...})` just
    //    forwards the supplied pointer; the orchestrator does the work.
    // -----------------------------------------------------------------
    {
        std::cout << "[1] Ground-state Lanczos on an in-memory operator\n";
        std::cout << "    (N=" << N << " periodic Heisenberg chain).\n";

        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N)};
        spec.num_sites = N;
        spec.spin_l    = 0.5f;

        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.num_eigs        = 2;
        opts.method          = ed::SolveMethod::Lanczos;
        opts.compute_vectors = false;
        opts.tolerance       = 1e-12;

        auto gs = ed::workflows::solve(*op, opts);
        std::cout << "    E0 = " << gs.eigenvalues[0] << "\n";
        std::cout << "    E1 = " << gs.eigenvalues[1] << "\n";
        std::cout << "    backend lane = " << gs.backend.lane
                  << "  (wall = " << gs.backend.wall_seconds << " s)\n\n";
    }

    // -----------------------------------------------------------------
    // 2. Same problem with the fixed-Sz axis flipped on. The factory
    //    returns a FixedSzOperator; the orchestrator runs the same
    //    Lanczos kernel but on the smaller (N choose N/2) sector.
    // -----------------------------------------------------------------
    {
        std::cout << "[2] Same operator, fixed_sz = N/2 (Sz=0 sector).\n";
        std::cout << "    Programmatic Operator: we rebuild the term storage\n";
        std::cout << "    into a FixedSzOperator below.\n";

        // FixedSzOperator construction needs the same term storage
        // walked into a fresh instance.  For the in-memory route the
        // simplest path is to construct the FixedSzOperator inline.
        auto fop = std::make_unique<FixedSzOperator>(
            N, 0.5f, static_cast<std::int64_t>(N / 2));
        for (std::uint64_t i = 0; i < N; ++i) {
            const std::uint64_t j = (i + 1) % N;
            fop->addTwoBodyTerm(2, i, 2, j,
                                std::complex<double>{1.0,  0.0});
            fop->addTwoBodyTerm(0, i, 1, j,
                                std::complex<double>{0.5,  0.0});
            fop->addTwoBodyTerm(1, i, 0, j,
                                std::complex<double>{0.5,  0.0});
        }

        ed::OperatorSpec spec;
        spec.source =
            ed::InMemoryOperator{std::move(fop)};
        spec.num_sites = N;
        spec.fixed_sz  = static_cast<int>(N / 2);

        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.num_eigs = 1;
        opts.method   = ed::SolveMethod::Lanczos;

        auto gs = ed::workflows::solve(*op, opts);
        std::cout << "    fixed-Sz E0 = " << gs.eigenvalues[0]
                  << "  (dim = " << op->dim() << ")\n\n";
    }

    // -----------------------------------------------------------------
    // 3. Thermal: mTPQ on the same Hamiltonian. mTPQ / cTPQ go through
    //    the fully backend-templated `tpq_kernel<Backend>` (Phase 2.4
    //    of the original Minimalist ED Collapse). FTLM / LTLM / KpmDos
    //    are CPU-only today; the kernel facade static_asserts on
    //    Backend type so the dispatch is honest.
    // -----------------------------------------------------------------
    {
        std::cout << "[3] Thermal mTPQ on the in-memory operator.\n";

        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N)};
        spec.num_sites = N;

        auto op = ed::make_operator(std::move(spec));

        ed::ThermalOptions opts;
        opts.method      = ed::ThermalOptions::Method::mTPQ;
        opts.num_samples = 1;
        opts.krylov_dim  = 80;
        opts.random_seed = 0xC0FFEEULL;

        auto thr = ed::workflows::thermal(*op, opts);
        std::cout << "    mTPQ E_min = " << thr.ground_state_energy
                  << "  (backend lane = " << thr.backend.lane << ")\n\n";
    }

    // -----------------------------------------------------------------
    // 4. Spectral: ground-state continued-fraction spectrum at a few
    //    omega points.  Observables are LinearOperators too, so the
    //    factory path is the same (just a different `OperatorSpec`).
    //    Here we pass H twice (the auto-correlator case).
    // -----------------------------------------------------------------
    {
        std::cout << "[4] Spectral S(omega) (continued-fraction, Sz total).\n";

        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N)};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        // Re-use H as the observable for this demo (Sz-total is a more
        // physical observable; the example focuses on the workflow
        // surface, not the operator definitions).
        ed::OperatorSpec spec_obs;
        spec_obs.source = ed::InMemoryOperator{build_heisenberg_chain(N)};
        spec_obs.num_sites = N;
        auto obs_op = ed::make_operator(std::move(spec_obs));

        ed::SpectralOptions opts;
        opts.method      = ed::SpectralOptions::Method::GroundStateCF;
        opts.num_omega   = 8;
        opts.omega_min   = -8.0;
        opts.omega_max   = +8.0;
        opts.broadening  = 0.1;
        opts.krylov_dim  = 80;

        std::vector<const ed::LinearOperator*> obs{ obs_op.get() };
        auto sp = ed::workflows::spectral(*op, obs, opts);
        std::cout << "    omega range = [" << sp.omega.front() << ", "
                  << sp.omega.back() << "]\n";
        std::cout << "    S_real[0] = " << sp.S_real.front()
                  << "  S_real[-1] = " << sp.S_real.back() << "\n\n";
    }

    // -----------------------------------------------------------------
    // 5. File-based source: write an InterAll.dat deck and load it back
    //    via `FilePaths`. This is the lane every external Hamiltonian
    //    generator (HPhi, custom Python builders, ...) lands on.
    // -----------------------------------------------------------------
    {
        std::cout << "[5] FilePaths source: write InterAll.dat then load.\n";
        namespace fs = std::filesystem;
        const fs::path tmp = fs::temp_directory_path()
                              / "ex00_unified_interface_files";
        fs::create_directories(tmp);
        const fs::path inter = tmp / "InterAll.dat";
        write_heisenberg_interall(inter, N);

        ed::OperatorSpec spec;
        spec.source    = ed::FilePaths{inter.string(), "", "", ""};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.num_eigs = 1;
        opts.method   = ed::SolveMethod::Lanczos;
        auto gs = ed::workflows::solve(*op, opts);
        std::cout << "    loaded from " << inter.string() << "\n";
        std::cout << "    E0 = " << gs.eigenvalues[0] << "\n\n";

        fs::remove_all(tmp);
    }

    // -----------------------------------------------------------------
    // 6. Directory-based source: same deck, but loaded via
    //    `DirectoryPath`. The CLI's `./ED ...` binary uses this lane
    //    exclusively because deck directories carry multiple text
    //    files (InterAll.dat, Trans.dat, CounterTerm.dat, ThreeBodyG.dat).
    // -----------------------------------------------------------------
    {
        std::cout << "[6] DirectoryPath source: load every text file in a dir.\n";
        namespace fs = std::filesystem;
        const fs::path tmp = fs::temp_directory_path()
                              / "ex00_unified_interface_dir";
        fs::create_directories(tmp);
        write_heisenberg_interall(tmp / "InterAll.dat", N);

        ed::OperatorSpec spec;
        spec.source    = ed::DirectoryPath{tmp.string()};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.num_eigs = 2;
        opts.method   = ed::SolveMethod::Lanczos;
        auto gs = ed::workflows::solve(*op, opts);
        std::cout << "    loaded from directory " << tmp.string() << "\n";
        std::cout << "    E0 = " << gs.eigenvalues[0]
                  << "  E1 = " << gs.eigenvalues[1] << "\n\n";

        fs::remove_all(tmp);
    }

    // -----------------------------------------------------------------
    // 7. Multiple solve methods: same Hamiltonian, four different
    //    Krylov / dense paths. Each one returns the same eigenvalue
    //    to within the solver's tolerance.
    // -----------------------------------------------------------------
    {
        std::cout << "[7] Solve-method matrix: Lanczos / FullDiag /\n"
                     "    KrylovSchur / BlockLanczos all converge to E0.\n";

        auto run = [&](ed::SolveMethod m, const char* name) {
            ed::OperatorSpec spec;
            spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N)};
            spec.num_sites = N;
            auto op = ed::make_operator(std::move(spec));

            ed::SolveOptions opts;
            opts.num_eigs   = 4;
            opts.method     = m;
            opts.tolerance  = 1e-10;
            opts.max_iter   = 200;
            opts.block_size = 4;
            auto r = ed::workflows::solve(*op, opts);
            // Some legacy lanes mutate std::cout's flags; restore.
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "      " << std::setw(13) << std::left << name
                      << "  E0 = " << r.eigenvalues[0]
                      << "  iters = " << r.krylov.iters_done << "\n";
        };
        run(ed::SolveMethod::Lanczos,      "Lanczos");
        run(ed::SolveMethod::FullDiag,     "FullDiag");
        run(ed::SolveMethod::KrylovSchur,  "KrylovSchur");
        run(ed::SolveMethod::BlockLanczos, "BlockLanczos");
        std::cout << "\n";
    }

    // -----------------------------------------------------------------
    // 8. compute_vectors = true: the unified interface delivers the
    //    ground-state vector(s) through `GroundStateResult.eigenvectors`.
    // -----------------------------------------------------------------
    {
        std::cout << "[8] compute_vectors = true: retrieve the ground state.\n";

        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N)};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.num_eigs        = 1;
        opts.method          = ed::SolveMethod::Lanczos;
        opts.tolerance       = 1e-12;
        opts.compute_vectors = true;
        auto r = ed::workflows::solve(*op, opts);

        // Verify ||psi||_2 = 1 and report the largest-amplitude
        // basis-state components (the all-down / all-up states have
        // zero amplitude in the Sz=0 ground state, so showing
        // |psi[0..3]| would just print zeros).
        if (r.eigenvectors && !r.eigenvectors->host.empty()) {
            std::cout << std::fixed << std::setprecision(8);
            const auto& psi = r.eigenvectors->host.front();
            double norm2 = 0.0;
            std::size_t arg_max = 0;
            double amp_max = -1.0;
            for (std::size_t k = 0; k < psi.size(); ++k) {
                const double a2 = std::norm(psi[k]);
                norm2 += a2;
                if (a2 > amp_max) { amp_max = a2; arg_max = k; }
            }
            std::cout << "    E0 = " << r.eigenvalues[0]
                      << "    ||psi||_2 = " << std::sqrt(norm2) << "\n";
            std::cout << "    max-amplitude basis index = " << arg_max
                      << "  |psi[arg_max]|^2 = " << amp_max << "\n\n";
        } else {
            std::cout << "    (eigenvector not produced -- HDF5-only lane)\n\n";
        }
    }

    // -----------------------------------------------------------------
    // 9. BackendConstraints: explicit CPU pin. This is the way to
    //    force the runtime onto a specific lane regardless of what
    //    backends the build offers.
    // -----------------------------------------------------------------
    {
        std::cout << "[9] BackendConstraints: pin the CPU lane explicitly.\n";

        ed::OperatorSpec spec;
        spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N)};
        spec.num_sites = N;
        auto op = ed::make_operator(std::move(spec));

        ed::SolveOptions opts;
        opts.num_eigs           = 1;
        opts.method             = ed::SolveMethod::Lanczos;
        opts.backend.allow_gpu  = false;
        opts.backend.allow_mpi  = false;
        auto r = ed::workflows::solve(*op, opts);
        std::cout << std::fixed << std::setprecision(8);
        std::cout << "    E0 = " << r.eigenvalues[0]
                  << "    backend = " << r.backend.lane
                  << "  (wall = " << r.backend.wall_seconds << " s)\n\n";
    }

    std::cout << "===============================================\n";
    std::cout << "  Done. The nine blocks above cover every C++\n";
    std::cout << "  ED workflow reduced to the same three-line shape:\n";
    std::cout << "     OperatorSpec -> make_operator -> workflows::*\n";
    std::cout << "===============================================\n";
    return 0;
}
