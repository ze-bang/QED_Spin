// =============================================================================
// examples/01_cpp_ground_state.cpp
//
// Single-process CPU ground state of a Heisenberg chain via the matrix-free
// `Operator` API + `lanczos()`. This is the smallest possible end-to-end
// program that uses the toolkit.
//
// Build (from the examples/ directory, after the main project is built):
//
//     cmake -B build -DCMAKE_PREFIX_PATH="$PWD/../build" .
//     cmake --build build -j --target ex01_cpp_ground_state
//
// Run:
//
//     ./build/ex01_cpp_ground_state            # default N=12 PBC
//     ./build/ex01_cpp_ground_state 16 0       # N=16 OBC
// =============================================================================

#include <ed/core/construct_ham.h>
#include <ed/solvers/lanczos.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

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

        Operator::TransformData zz;
        zz.op_type = 2; zz.site_index = i;
        zz.op_type_2 = 2; zz.site_index_2 = j;
        zz.coefficient = Jz; zz.is_two_body = true;
        op->transform_data_.push_back(zz);

        Operator::TransformData pm;
        pm.op_type = 0; pm.site_index = i;
        pm.op_type_2 = 1; pm.site_index_2 = j;
        pm.coefficient = Jpm; pm.is_two_body = true;
        op->transform_data_.push_back(pm);

        Operator::TransformData mp;
        mp.op_type = 1; mp.site_index = i;
        mp.op_type_2 = 0; mp.site_index_2 = j;
        mp.coefficient = Jpm; mp.is_two_body = true;
        op->transform_data_.push_back(mp);
    }
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t N        = (argc > 1) ? std::stoull(argv[1]) : 12;
    const bool          periodic = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;

    const std::uint64_t dim = 1ULL << N;
    auto op = build_heisenberg_chain(N, periodic);

    std::cout << "Heisenberg chain  N=" << N
              << "  PBC=" << (periodic ? 1 : 0)
              << "  dim=" << dim << "\n";

    std::vector<double> eigenvalues;
    lanczos(
        // matrix-vector callback bound to op->apply
        [&op](const Complex* in, Complex* out, int n) {
            op->apply(in, out, static_cast<std::size_t>(n));
        },
        /*N=*/dim,
        /*max_iter=*/200,
        /*exct=*/3,           // keep the lowest 3 Ritz values
        /*tol=*/1e-10,
        eigenvalues,
        /*dir=*/"",           // empty = no on-disk basis
        /*eigenvectors=*/false);

    std::cout << "Lowest 3 eigenvalues:\n";
    for (std::size_t k = 0; k < eigenvalues.size() && k < 3; ++k) {
        std::cout << "  E[" << k << "] = " << eigenvalues[k] << "\n";
    }
    return 0;
}
