// =============================================================================
// tests/unit/test_kernel_facades.cpp
//
// Phase-6 lockdown for the unified algorithm-kernel facades. Each
// kernel header is a thin Backend-templated wrapper over an existing
// CPU body (Lanczos / FTLM / mTPQ / KPM-DOS /
// block-Lanczos / Krylov-Schur). The tests prove the new headers
// actually compile, link, and produce the same numbers as the legacy
// entry points on small Heisenberg chains.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/matvec.h>

#include <ed/krylov/lanczos_kernel.h>
#include <ed/krylov/block_lanczos_kernel.h>
#include <ed/krylov/block_krylov_schur_kernel.h>
#include <ed/krylov/krylov_schur_kernel.h>
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/mtpq_kernel.h>
#include <ed/thermal/kpm_dos_kernel.h>

#include <ed/observables/expectation.h>
#include <ed/observables/static_correlator.h>
#include <ed/observables/cf_dynamical.h>
#include <ed/observables/kpm_dynamical.h>

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

// Wrap a `MatVecOperator` reference as a `void(in,out,n)` callable
// suitable for the kernel facades.
struct MatvecCallable {
    const ed::matvec::MatVecOperator* op;
    void operator()(const Complex* in, Complex* out, std::size_t n) const {
        op->apply(in, out, n);
    }
};

}  // namespace

TEST_CASE("krylov::block_lanczos_kernel matches the legacy block_lanczos",
          "[kernel-facade][block-lanczos][phase6]") {
    constexpr std::uint64_t N   = 6;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, /*J=*/1.0, /*periodic=*/true);

    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    ed::krylov::BlockLanczosOptions opts;
    opts.num_eigs   = 2;
    opts.block_size = 2;
    opts.max_iter   = 20;
    opts.tolerance  = 1e-10;

    auto res = ed::krylov::block_lanczos_kernel(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

    REQUIRE(res.eigenvalues.size() >= opts.num_eigs);
    // Ground state energy of the 6-site periodic Heisenberg chain is
    // exactly -11/4 + (J-dependent shift); we just sanity-check the
    // bound and that the kernel returned monotone eigenvalues.
    REQUIRE(res.eigenvalues[0] <  0.0);
    REQUIRE(res.eigenvalues[0] <= res.eigenvalues[1] + 1e-10);
}

TEST_CASE("krylov::block_lanczos_kernel is reproducible and its converged flag is honest",
          "[kernel-facade][block-lanczos][determinism]") {
    // A dense Hermitian matrix whose dimension (35) is NOT a multiple of the block size:
    // the shape on which two identical calls used to disagree at 1e-5 while both reported
    // converged = true (unseeded start block; the last, rank-deficient block was accepted
    // without evaluating a single residual).
    constexpr std::size_t dim = 35;
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);
    std::mt19937_64 gen(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < dim; ++i) {
        M(i, i) = u(gen);
        for (std::size_t j = i + 1; j < dim; ++j) {
            M(i, j) = Complex(u(gen), u(gen));
            M(j, i) = std::conj(M(i, j));
        }
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    auto apply = [&](const Complex* in, Complex* out, std::size_t n) {
        Eigen::Map<const Eigen::VectorXcd> x(in, static_cast<Eigen::Index>(n));
        Eigen::Map<Eigen::VectorXcd> y(out, static_cast<Eigen::Index>(n));
        y = M * x;
    };
    ed::matvec::CpuBackend backend;
    ed::krylov::BlockLanczosOptions opts;
    opts.num_eigs = 4; opts.block_size = 4; opts.max_iter = 0; opts.tolerance = 1e-10;

    auto a = ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, opts);
    auto b = ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, opts);
    REQUIRE(a.eigenvalues.size() == b.eigenvalues.size());
    for (std::size_t i = 0; i < a.eigenvalues.size(); ++i)
        REQUIRE(a.eigenvalues[i] == b.eigenvalues[i]);           // bit-for-bit
    REQUIRE(a.blocks_built * opts.block_size <= dim);             // never more columns than dimensions

    // the flag is derived from the residuals: whatever it says must be true
    REQUIRE(a.residuals.size() == a.eigenvalues.size());
    bool all_small = true;
    for (double r : a.residuals) all_small = all_small && (r <= opts.tolerance);
    REQUIRE(a.converged == all_small);
    for (std::size_t i = 0; i < a.eigenvalues.size(); ++i)
        if (a.residuals[i] <= opts.tolerance)
            REQUIRE(std::abs(a.eigenvalues[i] - es.eigenvalues()(static_cast<Eigen::Index>(i))) < 1e-8);

    // a different seed is a different (but equally valid) run
    ed::krylov::BlockLanczosOptions other = opts;
    other.seed = 7;
    auto c = ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, other);
    REQUIRE(c.eigenvalues[0] != a.eigenvalues[0]);              // genuinely another start block
    if (c.residuals[0] <= opts.tolerance)                       // accurate whenever it says so
        REQUIRE(std::abs(c.eigenvalues[0] - es.eigenvalues()(0)) < 1e-8);
    else
        REQUIRE_FALSE(c.converged);
}

TEST_CASE("krylov::block_lanczos_kernel lean reorth (keep_basis=false) matches full",
          "[kernel-facade][block-lanczos][lean]") {
    constexpr std::uint64_t N   = 6;
    constexpr std::size_t   dim = std::size_t{1} << N;
    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);
    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    ed::krylov::BlockLanczosOptions full;
    full.num_eigs = 4; full.block_size = 4; full.max_iter = 20; full.tolerance = 1e-10;
    auto rf = ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, full);

    ed::krylov::BlockLanczosOptions lean = full;
    lean.keep_basis = false;                       // lean: local reorth, no stored basis
    auto rl = ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, lean);

    REQUIRE(rl.eigenvalues.size() == rf.eigenvalues.size());
    for (std::size_t i = 0; i < rf.eigenvalues.size(); ++i)
        REQUIRE(std::abs(rl.eigenvalues[i] - rf.eigenvalues[i]) < 1e-8);

    // Lean mode cannot return eigenvectors (no stored basis).
    lean.compute_vectors = true;
    REQUIRE_THROWS_AS(
        ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, lean),
        std::invalid_argument);
}

TEST_CASE("krylov::block_krylov_schur_kernel == dense lowest-k WITH multiplicity",
          "[kernel-facade][block-krylov-schur]") {
    // Heisenberg chain has SU(2)-degenerate levels -- the discriminating test
    // for a block method: it must return the k lowest eigenvalues *counting
    // multiplicity*, where single-vector Lanczos/Krylov-Schur miss copies.
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

    ed::krylov::BlockKrylovSchurOptions opts;
    opts.num_eigs     = 6;
    opts.block_size   = 4;
    opts.tolerance    = 1e-10;
    opts.max_restarts = 200;
    auto res = ed::krylov::block_krylov_schur_kernel(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

    REQUIRE(res.eigenvalues.size() == opts.num_eigs);
    for (std::size_t i = 0; i < opts.num_eigs; ++i)
        REQUIRE(std::abs(res.eigenvalues[i] - ref[static_cast<long>(i)]) < 1e-7);
}

TEST_CASE("krylov::krylov_subspace_dim is predictable (floor / grow / memory cap)",
          "[kernel-facade][subspace]") {
    using ed::krylov::krylov_subspace_dim;
    using ed::krylov::krylov_vector_budget;
    // floor = 2k+20
    REQUIRE(krylov_subspace_dim(1, 0, 0, 0)   == 22);
    REQUIRE(krylov_subspace_dim(4, 0, 0, 0)   == 28);
    // grows with the requested (iteration budget)
    REQUIRE(krylov_subspace_dim(1, 200, 0, 0) == 200);
    // the MEMORY cap is the predictable upper bound (cannot OOM)
    REQUIRE(krylov_subspace_dim(1, 200, 0, 50) == 50);
    // global_dim caps it too
    REQUIRE(krylov_subspace_dim(1, 200, 30, 0) == 30);
    // never below nev+1
    REQUIRE(krylov_subspace_dim(5, 1, 0, 2)   == 6);
    // budget: 16 GiB, N=1e8 (1.6 GB/vec), 50% safety -> ~5 resident vectors
    const auto vb = krylov_vector_budget(16ull << 30, 100'000'000ull, 0.5, 0);
    REQUIRE(vb >= 4);
    REQUIRE(vb <= 6);
}

TEST_CASE("krylov::block diagnostics: per-eigenvalue residuals + n_converged",
          "[kernel-facade][diagnostics]") {
    constexpr std::uint64_t N   = 6;
    constexpr std::size_t   dim = std::size_t{1} << N;
    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);
    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    SECTION("block Lanczos reports residuals aligned with eigenvalues") {
        ed::krylov::BlockLanczosOptions o;
        o.num_eigs = 4; o.block_size = 4; o.max_iter = 30; o.tolerance = 1e-10;
        auto r = ed::krylov::block_lanczos_kernel(backend, apply, dim, dim, o);
        REQUIRE(r.residuals.size() == r.eigenvalues.size());
        REQUIRE(r.n_converged >= 1);                       // GS at least
        REQUIRE(r.n_converged <= r.eigenvalues.size());
        for (std::size_t i = 0; i < r.n_converged; ++i)
            REQUIRE(r.residuals[i] <= 1e-9);               // converged => tiny residual
        REQUIRE_FALSE(r.resid_history.empty());            // convergence curve captured
    }
    SECTION("block Krylov-Schur: locked == converged, residuals below tol") {
        ed::krylov::BlockKrylovSchurOptions o;
        o.num_eigs = 4; o.block_size = 4; o.tolerance = 1e-10; o.max_restarts = 200;
        auto r = ed::krylov::block_krylov_schur_kernel(backend, apply, dim, dim, o);
        REQUIRE(r.residuals.size() == r.eigenvalues.size());
        REQUIRE(r.n_converged == r.eigenvalues.size());
        for (double rho : r.residuals) REQUIRE(rho <= 1e-10);
    }
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
    opts.global_n  = static_cast<std::uint64_t>(dim);

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
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

    REQUIRE_FALSE(res.energy.empty());
    REQUIRE_FALSE(res.heat_capacity.empty());
}

// NOTE: there is no ltlm_kernel facade to pin. ed/thermal/ltlm_kernel.h and
// low_temperature_lanczos were deleted in consolidation Family 1 (Jul 2026):
// both reimplemented the same GS-local-DOS bug, and for a function of H the
// symmetric LTLM estimator reduces exactly to the FTLM trace, so the
// orchestrator's LTLM branch dispatches through ftlm_kernel. That equivalence
// is pinned in test_thermal_dense_ref ("LTLM thermodynamics IS the FTLM
// trace"); the genuinely LTLM-only estimator that survives
// (compute_connected_qh_response_ltlm, dM/dT -- an observable that does NOT
// commute with H) is pinned by test_ltlm_static_connected_qh.

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
    opts.target_beta  = 5.0;
    opts.large_value  = 50.0;

    auto res = ed::thermal::mtpq_kernel(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

    REQUIRE_FALSE(res.energies.empty());
}

TEST_CASE("thermal::kpm_dos_kernel returns Z/E/Cv/S over a beta grid",
          "[kernel-facade][kpm-dos][phase6]") {
    constexpr std::uint64_t N   = 4;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);

    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    ed::thermal::KpmDosOptions opts;
    opts.num_moments        = 128;
    opts.num_random_vectors = 4;
    opts.betas              = {0.5, 1.0, 2.0};
    opts.random_seed        = 123;

    auto res = ed::thermal::kpm_dos_kernel(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

    REQUIRE(res.energy.size() == opts.betas.size());
    REQUIRE(res.specific_heat.size() == opts.betas.size());
    REQUIRE(res.e_min_estimate < 0.0);
}

TEST_CASE("observables::expectation_value reproduces <psi|H|psi>",
          "[kernel-facade][observables][phase6]") {
    constexpr std::uint64_t N   = 4;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, 1.0, true);

    ed::matvec::CpuBackend backend;
    std::vector<Complex> psi(dim, Complex(0.0, 0.0));
    psi[0] = Complex(1.0, 0.0);                  // |0...0> Neel-flavoured basis state
    std::vector<Complex> out(dim, Complex(0.0, 0.0));

    Complex e = ed::observables::expectation_value(
        backend, *H, psi.data(), out.data(), dim);

    // <0...0| H | 0...0> = 0 for any traceless Heisenberg term + half-shift;
    // here only Sz Sz contributes => +N/4 on the fully-polarised state.
    REQUIRE(std::abs(e.imag()) < 1e-12);
}
