// =============================================================================
// tests/unit/test_kernel_facades.cpp
//
// Lockdown for the Backend-templated algorithm kernels (Lanczos / FTLM / mTPQ /
// Krylov-Schur). The tests prove the headers compile, link, and produce correct
// numbers on small Heisenberg chains.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/matvec/cpu_backend.h>
#include <ed/matvec/linear_operator.h>

#include <ed/krylov/lanczos.h>
#include <ed/krylov/krylov_schur.h>
#include <ed/thermal/ftlm.h>
#include <ed/thermal/mtpq.h>
#include <ed/thermal/tpq_thermo.h>


#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using Complex = std::complex<double>;

namespace {

// Wrap a `LinearOperator` reference as a `void(in,out,n)` callable
// suitable for the kernel facades.
struct MatvecCallable {
    const ed::LinearOperator* op;
    void operator()(const Complex* in, Complex* out, std::size_t n) const {
        op->apply(in, out, n);
    }
};

}  // namespace

TEST_CASE("krylov::krylov_schur_kernel == dense lowest-k WITH multiplicity",
          "[kernel-facade][krylov-schur][degeneracy]") {
    // The Heisenberg ring has SU(2)-degenerate levels: a Krylov space grown from one
    // start vector holds one copy of each, so the k lowest eigenvalues *counting
    // multiplicity* need the fresh starts after locking and the degeneracy probe.
    constexpr std::uint64_t N   = 6;
    constexpr std::size_t   dim = std::size_t{1} << N;
    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);

    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    // Dense reference: materialize H by applying it to each unit column.
    Eigen::MatrixXcd M(dim, dim);
    std::vector<Complex> e(dim), col(dim);
    for (std::size_t c = 0; c < dim; ++c) {
        std::fill(e.begin(), e.end(), Complex(0.0, 0.0));
        e[c] = Complex(1.0, 0.0);
        apply(e.data(), col.data(), dim);
        for (std::size_t r = 0; r < dim; ++r) M(static_cast<long>(r), static_cast<long>(c)) = col[r];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    const auto ref = es.eigenvalues();   // ascending, with multiplicity

    std::mt19937_64 gen(0x51ED0B70ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Complex> v0(dim);
    for (auto& z : v0) z = Complex(nd(gen), nd(gen));
    ed::krylov::KrylovSchurOptions opts;
    opts.num_eigs     = 6;
    opts.max_iter     = 40;
    opts.tolerance    = 1e-10;
    opts.max_restarts = 200;
    auto res = ed::krylov::krylov_schur_kernel(backend, apply, dim, v0.data(), opts);

    REQUIRE(res.converged);
    REQUIRE(res.eigenvalues.size() == opts.num_eigs);
    for (std::size_t i = 0; i < opts.num_eigs; ++i)
        REQUIRE(std::abs(res.eigenvalues[i] - ref[static_cast<long>(i)]) < 1e-7);
}

TEST_CASE("krylov::krylov_schur_kernel degeneracy probe recovers skipped copies",
          "[kernel-facade][krylov-schur][degeneracy]") {
    // Diagonal H = diag(0, 1, 1, 2, 2, 2, 3, 4, ...) and a start vector with no weight on
    // e_2, e_4, e_5: those components stay exactly zero through every matvec and
    // reorthogonalisation, so the restarted single-vector method finds 0, 1, 2, 3, 4, 5 and
    // skips the second 1 and two of the 2s. The probe must restore the multiplicities.
    constexpr std::size_t dim = 200;
    std::vector<double> d(dim);
    const double head[] = {0.0, 1.0, 1.0, 2.0, 2.0, 2.0};
    for (std::size_t i = 0; i < dim; ++i) d[i] = i < 6 ? head[i] : static_cast<double>(i) - 3.0;
    auto apply = [&d](const Complex* in, Complex* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) out[i] = d[i] * in[i];
    };
    ed::matvec::CpuBackend backend;
    std::mt19937_64 gen(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Complex> v0(dim);
    for (auto& z : v0) z = Complex(nd(gen), nd(gen));
    v0[2] = v0[4] = v0[5] = Complex(0.0, 0.0);

    ed::krylov::KrylovSchurOptions opts;
    opts.num_eigs     = 6;
    opts.max_iter     = 40;
    opts.tolerance    = 1e-10;
    opts.max_restarts = 200;
    opts.compute_vectors = true;
    const auto res = ed::krylov::krylov_schur_kernel(backend, apply, dim, v0.data(), opts);
    REQUIRE(res.converged);
    REQUIRE(res.eigenvalues.size() == 6);
    for (std::size_t i = 0; i < 6; ++i) REQUIRE(std::abs(res.eigenvalues[i] - head[i]) < 1e-8);
    // The recovered copies are genuine, mutually orthogonal eigenvectors.
    for (std::size_t a = 0; a < 6; ++a)
        for (std::size_t b = 0; b <= a; ++b) {
            const Complex ov = backend.dot(res.eigenvectors[a].get(), res.eigenvectors[b].get(), dim);
            REQUIRE(std::abs(ov - (a == b ? 1.0 : 0.0)) < 1e-8);
        }

    opts.probe_degeneracy = false;   // the failure the probe exists for
    const auto bare = ed::krylov::krylov_schur_kernel(backend, apply, dim, v0.data(), opts);
    REQUIRE(bare.eigenvalues.size() == 6);
    REQUIRE(std::abs(bare.eigenvalues.back() - 5.0) < 1e-8);
}

TEST_CASE("krylov::krylov_subspace_dim is predictable (floor / grow / memory cap)",
          "[kernel-facade][subspace]") {
    using ed::krylov::krylov_subspace_dim;
    // floor = 2k+20
    REQUIRE(krylov_subspace_dim(1, 0, 0, 0)   == 22);
    REQUIRE(krylov_subspace_dim(4, 0, 0, 0)   == 28);
    // grows with the requested (iteration budget)
    REQUIRE(krylov_subspace_dim(1, 200, 0, 0) == 200);
    // the MEMORY cap is the predictable upper bound (cannot OOM)
    REQUIRE(krylov_subspace_dim(1, 200, 0, 50) == 50);
    // the dimension of the space caps it too
    REQUIRE(krylov_subspace_dim(1, 200, 30, 0) == 30);
    // never below nev+1
    REQUIRE(krylov_subspace_dim(5, 1, 0, 2)   == 6);
}

TEST_CASE("krylov::krylov_schur_kernel returns sane Heisenberg eigenvalues",
          "[kernel-facade][krylov-schur][phase6]") {
    constexpr std::uint64_t N   = 6;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);

    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    ed::krylov::KrylovSchurOptions opts;
    opts.num_eigs  = 3;
    opts.max_iter  = 40;
    opts.tolerance = 1e-10;

    std::vector<std::complex<double>> seed(dim);
    {
        std::mt19937_64 gen(0xC0FFEEULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        double sumsq = 0.0;
        for (auto& z : seed) {
            const double a = nd(gen), b = nd(gen);
            z = std::complex<double>(a, b);
            sumsq += a * a + b * b;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (auto& z : seed) z *= inv;
    }
    auto res = ed::krylov::krylov_schur_kernel(
        backend, apply, dim, seed.data(), opts);

    REQUIRE(res.eigenvalues.size() >= opts.num_eigs);
    for (std::size_t i = 1; i < res.eigenvalues.size(); ++i) {
        REQUIRE(res.eigenvalues[i - 1] <= res.eigenvalues[i] + 1e-9);
    }
}

TEST_CASE("thermal::ftlm_kernel returns thermodynamic data over a beta grid",
          "[kernel-facade][ftlm][phase6]") {
    constexpr std::uint64_t N   = 4;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);

    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    ed::thermal::FtlmOptions opts;
    opts.num_samples = 3;
    opts.krylov_dim  = 16;
    opts.betas       = {0.1, 0.5, 1.0, 2.0};
    opts.random_seed = 42;

    auto res = ed::thermal::ftlm_kernel(
        backend, apply, dim, opts);

    REQUIRE(res.curves.E.size() == opts.betas.size());
    REQUIRE(res.curves.V.size() == opts.betas.size());
    REQUIRE(res.curves.lnZ.size() == opts.betas.size());
}

TEST_CASE("thermal::mtpq_kernel runs end-to-end on a small Heisenberg chain",
          "[kernel-facade][mtpq][phase6]") {
    constexpr std::uint64_t N   = 4;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);

    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    ed::thermal::MtpqOptions opts;
    opts.num_samples  = 1;
    opts.max_iter     = 50;
    opts.large_value  = 50.0;

    auto res = ed::thermal::mtpq_kernel(
        backend, apply, dim, opts);

    REQUIRE_FALSE(res.energies.empty());
}

TEST_CASE("thermal::mtpq_canonical_thermo reproduces its start vector's canonical average exactly",
          "[kernel-facade][mtpq]") {
    // One sample's moments mu_j = <psi0|(L - H)^j|psi0> sum to S_0(beta) = e^{beta L}
    // <psi0|e^{-beta H}|psi0>, so the estimator's ln Z, E and C are those of the ensemble
    // weighted by |<n|psi0>|^2 times D -- checked against the dense spectral decomposition of
    // the same start vector, at any beta the trajectory reaches.
    constexpr std::uint64_t N   = 6;
    constexpr std::size_t   dim = std::size_t{1} << N;
    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);
    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    Eigen::MatrixXcd Hd(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
    std::vector<Complex> unit(dim), col(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        std::fill(unit.begin(), unit.end(), Complex(0, 0));
        unit[j] = Complex(1, 0);
        apply(unit.data(), col.data(), dim);
        for (std::size_t i = 0; i < dim; ++i)
            Hd(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hd);
    const Eigen::VectorXd ev = es.eigenvalues();
    const double L = ev(static_cast<Eigen::Index>(dim) - 1) + 0.5;

    ed::thermal::MtpqOptions opts;
    opts.num_samples = 1;
    opts.random_seed = 99;
    opts.large_value = L;
    opts.max_iter    = 300;
    const auto res = ed::thermal::mtpq_kernel(backend, apply, dim, opts);
    REQUIRE(res.sample_energies.size() == 1);
    REQUIRE(res.sample_energies[0].size() == 301);
    REQUIRE(res.sample_log_norms[0].size() == 300);

    // The kernel's start vector: sample 0 of base seed 99.
    std::mt19937 eng = ed::thermal::sample_engine(99, 0);
    const auto psi0 = ed::thermal::gaussian_vector(dim, eng);
    Eigen::VectorXcd p(static_cast<Eigen::Index>(dim));
    for (std::size_t i = 0; i < dim; ++i) p(static_cast<Eigen::Index>(i)) = psi0[i];
    const Eigen::VectorXd w = (es.eigenvectors().adjoint() * p).cwiseAbs2();

    const std::vector<double> Ts = {0.25, 0.5, 1.0, 4.0};
    std::vector<double> betas;
    for (double T : Ts) betas.push_back(1.0 / T);
    const auto mt = ed::thermal::mtpq_canonical_thermo(res.sample_energies, res.sample_log_norms, L, betas,
                                                       static_cast<double>(dim));
    REQUIRE(mt.unconverged.empty());
    for (std::size_t t = 0; t < Ts.size(); ++t) {
        const double beta = betas[t];
        double z = 0.0, ez = 0.0;
        for (Eigen::Index i = 0; i < ev.size(); ++i) {
            const double b = w(i) * std::exp(-beta * (ev(i) - ev(0)));
            z += b; ez += b * ev(i);
        }
        const double E = ez / z;
        double v = 0.0;
        for (Eigen::Index i = 0; i < ev.size(); ++i)
            v += w(i) * std::exp(-beta * (ev(i) - ev(0))) * (ev(i) - E) * (ev(i) - E);
        v /= z;
        const double lnZ = std::log(static_cast<double>(dim)) + std::log(z) - beta * ev(0);
        REQUIRE(std::abs(mt.curves.E[t] - E) < 1e-10);
        REQUIRE(std::abs(mt.curves.lnZ[t] - lnZ) < 1e-10);
        REQUIRE(std::abs(beta * beta * (mt.curves.V[t] - v)) < 1e-8);
    }
    // A target colder than 300 steps reach is reported as such, never clamped.
    const auto cold = ed::thermal::mtpq_canonical_thermo(res.sample_energies, res.sample_log_norms, L, {200.0},
                                                         static_cast<double>(dim));
    REQUIRE(cold.unconverged.size() == 1);
}
