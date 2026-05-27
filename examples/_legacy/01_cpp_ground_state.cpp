// =============================================================================
// examples/01_cpp_ground_state.cpp
//
// Single-process CPU ground state of a Heisenberg chain via the unified
// ED interface (Full Unified-Interface Collapse, May 2026):
//
//     OperatorSpec  ->  ed::make_operator(spec)  ->  ed::workflows::solve(*op, opts)
//
// This is the smallest possible end-to-end program that uses the toolkit.
//
// Build (from the top-level build directory):
//
//     cmake --build build -j --target ex01_cpp_ground_state
//
// Run:
//
//     ./build/examples/ex01_cpp_ground_state            # default N=12 PBC
//     ./build/examples/ex01_cpp_ground_state 16 0       # N=16 OBC
// =============================================================================

#include <ed/core/make_operator.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

using Complex = std::complex<double>;

// Heisenberg J=1 spin-1/2 chain. `periodic == 1` for PBC, 0 for OBC.
std::unique_ptr<Operator>
build_heisenberg_chain(std::uint64_t N, bool periodic) {
    auto op = std::make_unique<Operator>(N, /*spin=*/0.5f);
    const Complex Jz(1.0, 0.0);
    const Complex Jpm(0.5, 0.0);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op->addTwoBodyTerm(2, i, 2, j, Jz);
        op->addTwoBodyTerm(0, i, 1, j, Jpm);
        op->addTwoBodyTerm(1, i, 0, j, Jpm);
    }
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t N        = (argc > 1) ? std::stoull(argv[1]) : 12;
    const bool          periodic = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;

    const std::uint64_t dim = 1ULL << N;
    std::cout << "Heisenberg chain  N=" << N
              << "  PBC=" << (periodic ? 1 : 0)
              << "  dim=" << dim << "\n";

    // 1. Describe the operator. `InMemoryOperator` forwards the
    //    already-built Operator straight through `make_operator`.
    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_heisenberg_chain(N, periodic)};
    spec.num_sites = N;
    spec.spin_l    = 0.5f;

    // 2. Materialize the LinearOperator (no-op forward in this case).
    auto op = ed::make_operator(std::move(spec));

    // 3. Run the orchestrator. The backend (CPU here) is auto-selected.
    ed::SolveOptions opts;
    opts.num_eigs        = 3;
    opts.method          = ed::SolveMethod::Lanczos;
    opts.tolerance       = 1e-10;
    opts.compute_vectors = false;

    auto result = ed::workflows::solve(*op, opts);

    std::cout << "Lowest 3 eigenvalues:\n";
    for (std::size_t k = 0; k < result.eigenvalues.size() && k < 3; ++k) {
        std::cout << "  E[" << k << "] = " << result.eigenvalues[k] << "\n";
    }
    std::cout << "  backend = " << result.backend.lane
              << "  (wall = " << result.backend.wall_seconds << " s)\n";
    return 0;
}
