// =============================================================================
// examples/02_cpp_full_spectrum.cpp
//
// Dense LAPACK eigendecomposition of a J1-J2 Heisenberg chain, via the
// unified ED interface (Full Unified-Interface Collapse, May 2026):
//
//     OperatorSpec  ->  ed::make_operator(spec)
//     ed::SolveOptions{ .method = FullDiag }  ->  ed::workflows::solve(*op)
//
// J1-J2 Hamiltonian:
//   H = J1 sum_<ij> S_i . S_j  +  J2 sum_<<ij>> S_i . S_j
//
// Run:
//     ./build/examples/ex02_cpp_full_spectrum             # default J2=0.0
//     ./build/examples/ex02_cpp_full_spectrum 8 0.5       # N=8, J2/J1=0.5
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

void add_heisenberg_bond(Operator& op, std::uint64_t i, std::uint64_t j,
                         double J) {
    op.addTwoBodyTerm(2, i, 2, j, Complex(J, 0.0));
    op.addTwoBodyTerm(0, i, 1, j, Complex(0.5 * J, 0.0));
    op.addTwoBodyTerm(1, i, 0, j, Complex(0.5 * J, 0.0));
}

std::unique_ptr<Operator>
build_j1j2_chain(std::uint64_t N, double J2) {
    auto op = std::make_unique<Operator>(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i + 1 < N; ++i) {
        add_heisenberg_bond(*op, i, i + 1, 1.0);
    }
    for (std::uint64_t i = 0; i + 2 < N; ++i) {
        add_heisenberg_bond(*op, i, i + 2, J2);
    }
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t N  = (argc > 1) ? std::stoull(argv[1]) : 8;
    const double        J2 = (argc > 2) ? std::stod(argv[2]) : 0.0;

    if (N > 12) {
        std::cerr << "Refusing to dense-diagonalize N=" << N
                  << " (would need " << (1ULL << N) << "^2 doubles).\n";
        return 1;
    }

    const std::uint64_t dim = 1ULL << N;
    std::cout << "J1-J2 chain N=" << N << " J2/J1=" << J2
              << " dim=" << dim << "\n";

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_j1j2_chain(N, J2)};
    spec.num_sites = N;
    spec.spin_l    = 0.5f;

    auto op = ed::make_operator(std::move(spec));

    ed::SolveOptions opts;
    opts.num_eigs        = 10;
    opts.method          = ed::SolveMethod::FullDiag;
    opts.compute_vectors = false;

    auto result = ed::workflows::solve(*op, opts);

    std::cout << "Lowest 10 eigenvalues:\n";
    for (std::size_t k = 0; k < result.eigenvalues.size() && k < 10; ++k) {
        std::cout << "  E[" << k << "] = " << result.eigenvalues[k] << "\n";
    }
    return 0;
}
