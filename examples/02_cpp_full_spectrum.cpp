// =============================================================================
// examples/02_cpp_full_spectrum.cpp
//
// Dense LAPACK eigendecomposition of an 8-site Heisenberg J1-J2 chain.
// Prints the lowest 10 eigenvalues. Useful as a small reference / sanity
// check against Lanczos on the same system.
//
// J1-J2 Hamiltonian:
//   H = J1 sum_<ij> S_i . S_j  +  J2 sum_<<ij>> S_i . S_j
//
// Run:
//     ./build/ex02_cpp_full_spectrum             # default J2=0.0 (pure Heisenberg)
//     ./build/ex02_cpp_full_spectrum 8 0.5       # N=8, J2/J1=0.5
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

void add_heisenberg_bond(Operator& op, std::uint64_t i, std::uint64_t j, double J) {
    const Complex Jz(J, 0.0);
    const Complex Jpm(0.5 * J, 0.0);

    Operator::TransformData zz;
    zz.op_type = 2; zz.site_index = i;
    zz.op_type_2 = 2; zz.site_index_2 = j;
    zz.coefficient = Jz; zz.is_two_body = true;
    op.transform_data_.push_back(zz);

    Operator::TransformData pm;
    pm.op_type = 0; pm.site_index = i;
    pm.op_type_2 = 1; pm.site_index_2 = j;
    pm.coefficient = Jpm; pm.is_two_body = true;
    op.transform_data_.push_back(pm);

    Operator::TransformData mp;
    mp.op_type = 1; mp.site_index = i;
    mp.op_type_2 = 0; mp.site_index_2 = j;
    mp.coefficient = Jpm; mp.is_two_body = true;
    op.transform_data_.push_back(mp);
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

    Operator op(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i + 1 < N; ++i)        add_heisenberg_bond(op, i,     i + 1, 1.0);
    for (std::uint64_t i = 0; i + 2 < N; ++i)        add_heisenberg_bond(op, i,     i + 2, J2 );

    const std::uint64_t dim = 1ULL << N;
    std::cout << "J1-J2 chain N=" << N << " J2/J1=" << J2 << " dim=" << dim << "\n";

    std::vector<double> eigenvalues;
    full_diagonalization(
        [&op](const Complex* in, Complex* out, int n) {
            op.apply(in, out, static_cast<std::size_t>(n));
        },
        /*N=*/dim,
        /*num_eigs=*/10,
        eigenvalues,
        /*dir=*/"",
        /*compute_eigenvectors=*/false);

    std::cout << "Lowest 10 eigenvalues:\n";
    for (std::size_t k = 0; k < eigenvalues.size() && k < 10; ++k) {
        std::cout << "  E[" << k << "] = " << eigenvalues[k] << "\n";
    }
    return 0;
}
