// =============================================================================
// examples/solve/lanczos/cpu_save.cpp
//
// solve | LANCZOS | CPU (OpenMP) | full Hilbert (no symmetry)
// + persistent eigenvector save (HDF5)
//
// Demonstrates the May 2026 eigenvector-save contract:
//
//     opts.compute_eigenvectors = true;
//     opts.output_dir           = "<dir>";
//
// causes `ed::api::solve` (== `ed::workflows::solve`) to write the
// eigenvalues + eigenvectors to `<dir>/ed_results.h5` and surface the
// path via `GroundStateResult::hdf5_path` (mirrored into Python's
// `EDResults.eigenvectors_path`). Works for every solver method
// (LANCZOS / BLOCK_LANCZOS / KRYLOV_SCHUR / FULL) on the CPU /
// single-rank GPU lanes -- to switch method, just change `opts.solver`
// below.
//
// Twin: examples/solve/lanczos/cpu_save.py
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

int main(int argc, char** argv) {
    auto guard = ed_example::mpi_guard(argc, argv);
    (void)guard;
    constexpr std::uint64_t N = 8;

    auto op   = ed_example::heisenberg_chain(N, /*pbc=*/true);
    auto spec = ed_example::in_memory_spec(std::move(op), N);

    const std::string outdir = "ed_save_demo";
    std::filesystem::remove_all(outdir);

    ed::api::SolveOptions opts;
    opts.num_eigenvalues      = 1;
    opts.solver               = "LANCZOS";
    opts.device               = "cpu";
    opts.tolerance            = 1e-12;
    opts.compute_eigenvectors = true;
    opts.output_dir           = outdir;

    auto result = ed::api::solve(std::move(spec), opts);

    std::cout << std::setprecision(10);
    ed_example::rank0_print("E[0]      = ", result.eigenvalues.front(), "\n");
    ed_example::rank0_print("hdf5_path = ", result.hdf5_path, "\n");

    // Round-trip: load the eigenvalues + ground-state vector back from
    // disk and check the in-memory ground state agrees up to a global
    // phase.
    auto e_loaded   = HDF5IO::loadEigenvalues(result.hdf5_path);
    auto psi_loaded = HDF5IO::loadEigenvector(result.hdf5_path, /*index=*/0);
    ed_example::rank0_print("loaded #eigenvalues  = ", e_loaded.size(), "\n");
    ed_example::rank0_print("loaded |psi|         = ", psi_loaded.size(), "\n");

    using Complex = std::complex<double>;
    Complex acc{0.0, 0.0};
    const auto& psi_mem = result.eigenvectors->host[0];
    for (std::size_t i = 0; i < psi_mem.size(); ++i) {
        acc += std::conj(psi_loaded[i]) * psi_mem[i];
    }
    const double overlap = std::abs(acc);
    ed_example::rank0_print("|<psi_disk|psi_mem>| = ",
                             std::setprecision(10), overlap, "\n");

    // === Expected output (deterministic; captured on the CI reference runner) ===
    // E[0]      = -3.651093409
    // hdf5_path = ed_save_demo/ed_results.h5
    // loaded #eigenvalues  = 1
    // loaded |psi|         = 256
    // |<psi_disk|psi_mem>| = 1
    // ===========================================================================
    return 0;
}
