// =============================================================================
// examples/03_cpp_ftlm_thermal.cpp
//
// Finite-Temperature Lanczos Method (Jaklic-Prelovsek, PRB 49, 5065 (1994)).
// Computes the canonical thermodynamic observables E(T), C(T), F(T), S(T)
// of a 12-site Heisenberg PBC chain on a 50-point log-spaced T grid.
//
// Run:
//     ./build/ex03_cpp_ftlm_thermal              # default N=12 PBC, R=20 samples
// =============================================================================

#include <ed/core/construct_ham.h>
#include <ed/solvers/ftlm.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using Complex = std::complex<double>;

static void add_bond(Operator& op, std::uint64_t i, std::uint64_t j, double J) {
    Operator::TransformData zz;
    zz.op_type = 2; zz.site_index = i; zz.op_type_2 = 2; zz.site_index_2 = j;
    zz.coefficient = Complex(J, 0.0); zz.is_two_body = true;
    op.transform_data_.push_back(zz);

    Operator::TransformData pm;
    pm.op_type = 0; pm.site_index = i; pm.op_type_2 = 1; pm.site_index_2 = j;
    pm.coefficient = Complex(0.5 * J, 0.0); pm.is_two_body = true;
    op.transform_data_.push_back(pm);

    Operator::TransformData mp;
    mp.op_type = 1; mp.site_index = i; mp.op_type_2 = 0; mp.site_index_2 = j;
    mp.coefficient = Complex(0.5 * J, 0.0); mp.is_two_body = true;
    op.transform_data_.push_back(mp);
}

int main(int argc, char** argv) {
    const std::uint64_t N         = (argc > 1) ? std::stoull(argv[1]) : 12;
    const std::uint64_t n_samples = (argc > 2) ? std::stoull(argv[2]) : 20;

    Operator op(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i < N; ++i) add_bond(op, i, (i + 1) % N, 1.0);

    const std::uint64_t dim = 1ULL << N;

    FTLMParameters params;
    params.krylov_dim          = 100;
    params.num_samples         = n_samples;
    params.max_iterations      = 1000;
    params.tolerance           = 1e-10;
    params.full_reorthogonalization = true;
    params.random_seed         = 12345;
    params.compute_error_bars  = true;

    auto results = finite_temperature_lanczos(
        [&op](const Complex* in, Complex* out, int n) {
            op.apply(in, out, static_cast<std::size_t>(n));
        },
        /*N=*/dim,
        params,
        /*temp_min=*/0.05,
        /*temp_max=*/10.0,
        /*num_temp_bins=*/50);

    const auto& T = results.thermo_data.temperatures;
    const auto& E = results.thermo_data.energy;
    const auto& Cv = results.thermo_data.specific_heat;
    const auto& F  = results.thermo_data.free_energy;
    const auto& S  = results.thermo_data.entropy;

    std::cout << "Heisenberg PBC chain N=" << N
              << " dim=" << dim
              << " R=" << n_samples << " FTLM samples\n";
    std::cout << "  T            E/N            Cv             F/N            S/N\n";
    for (std::size_t i = 0; i < T.size(); ++i) {
        std::cout
            << "  " << T[i]
            << "    " << E[i]  / static_cast<double>(N)
            << "    " << Cv[i]
            << "    " << F[i]  / static_cast<double>(N)
            << "    " << S[i]  / static_cast<double>(N)
            << "\n";
    }
    std::cout << "Ground-state estimate (lowest Ritz across samples) = "
              << results.ground_state_estimate << "\n";
    return 0;
}
