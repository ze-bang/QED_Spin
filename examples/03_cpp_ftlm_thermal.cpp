// =============================================================================
// examples/03_cpp_ftlm_thermal.cpp
//
// Finite-Temperature Lanczos Method (Jaklic-Prelovsek, PRB 49, 5065 (1994)).
// Runs via the unified ED interface (Full Unified-Interface Collapse,
// May 2026):
//
//     OperatorSpec  ->  ed::make_operator(spec)
//     ed::ThermalOptions{ .method = FTLM }  ->  ed::workflows::thermal(*op)
//
// Computes E(T), C(T), F(T), S(T) of a 12-site Heisenberg PBC chain on a
// 50-point log-spaced T grid.
//
// Run:
//     ./build/examples/ex03_cpp_ftlm_thermal             # default N=12, R=20 samples
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

void add_bond(Operator& op, std::uint64_t i, std::uint64_t j, double J) {
    op.addTwoBodyTerm(2, i, 2, j, Complex(J, 0.0));
    op.addTwoBodyTerm(0, i, 1, j, Complex(0.5 * J, 0.0));
    op.addTwoBodyTerm(1, i, 0, j, Complex(0.5 * J, 0.0));
}

std::unique_ptr<Operator>
build_heisenberg_pbc(std::uint64_t N) {
    auto op = std::make_unique<Operator>(N, /*spin=*/0.5f);
    for (std::uint64_t i = 0; i < N; ++i) {
        add_bond(*op, i, (i + 1) % N, 1.0);
    }
    return op;
}

}  // namespace

int main(int argc, char** argv) {
    const std::uint64_t N         = (argc > 1) ? std::stoull(argv[1]) : 12;
    const std::uint64_t n_samples = (argc > 2) ? std::stoull(argv[2]) : 20;

    const std::uint64_t dim = 1ULL << N;

    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{build_heisenberg_pbc(N)};
    spec.num_sites = N;
    spec.spin_l    = 0.5f;

    auto op = ed::make_operator(std::move(spec));

    ed::ThermalOptions opts;
    opts.method        = ed::ThermalOptions::Method::FTLM;
    opts.num_samples   = n_samples;
    opts.krylov_dim    = 100;
    opts.random_seed   = 12345;
    opts.temp_min      = 0.05;
    opts.temp_max      = 10.0;
    opts.num_temp_bins = 50;

    auto results = ed::workflows::thermal(*op, opts);

    const auto& T  = results.thermo.temperatures;
    const auto& E  = results.thermo.energy;
    const auto& Cv = results.thermo.specific_heat;
    const auto& F  = results.thermo.free_energy;
    const auto& S  = results.thermo.entropy;

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
    std::cout << "Ground-state estimate = "
              << results.ground_state_energy << "\n";
    std::cout << "Backend lane = " << results.backend.lane
              << "  (wall = " << results.backend.wall_seconds << " s)\n";
    return 0;
}
