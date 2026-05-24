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
// On the first invocation the example builds an in-memory N=8
// Heisenberg chain Operator, then walks through every axis and
// workflow type. No HDF5, no CLI integration, no environment
// dependencies --- just `Operator -> make_operator (no-op for the
// in-memory case) -> workflows::*` ---  exactly the shape every
// downstream caller (CLI, examples, Python) should adopt.
// =============================================================================

#include <ed/core/make_operator.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <cmath>
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

    std::cout << "===============================================\n";
    std::cout << "  Done. The four blocks above show every C++ ED\n";
    std::cout << "  workflow reduced to the same three-line shape:\n";
    std::cout << "     OperatorSpec -> make_operator -> workflows::*\n";
    std::cout << "===============================================\n";
    return 0;
}
