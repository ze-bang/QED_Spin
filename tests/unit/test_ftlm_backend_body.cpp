// =============================================================================
// tests/unit/test_ftlm_backend_body.cpp
//
// WP10: feature parity of the Backend-templated FTLM body
// (``ed::thermal::detail::ftlm_kernel_via_backend``) with the retired Gen-1
// CPU driver. The body is exercised directly on ``CpuBackend``:
//   * full reorthogonalisation on and off both give finite curves that
//     agree with each other at low temperature;
//   * ``ground_state_estimate`` is filled with the lowest Ritz value and
//     matches the dense ground energy;
//   * the thread-budget scope restores the caller's OpenMP team size.
//
// A failed sample (tridiagonal solve returning no Ritz values) cannot be
// provoked deterministically through the public matvec interface, so the
// skip path is not exercised here.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/matvec.h>
#include <ed/thermal/ftlm_kernel.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Complex = std::complex<double>;

namespace {

struct MatvecCallable {
    const ed::matvec::MatVecOperator* op;
    void operator()(const Complex* in, Complex* out, std::size_t n) const {
        op->apply(in, out, n);
    }
};

double dense_ground_energy(const MatvecCallable& apply, std::size_t dim) {
    Eigen::MatrixXcd Hd(dim, dim);
    std::vector<Complex> e(dim), col(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        std::fill(e.begin(), e.end(), Complex(0.0, 0.0));
        e[j] = Complex(1.0, 0.0);
        apply(e.data(), col.data(), dim);
        for (std::size_t i = 0; i < dim; ++i) Hd(i, j) = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(
        Hd, Eigen::EigenvaluesOnly);
    return es.eigenvalues()(0);
}

void require_finite(const ed::thermal::FtlmResult& r, std::size_t nb) {
    REQUIRE(r.energy.size() == nb);
    REQUIRE(r.heat_capacity.size() == nb);
    REQUIRE(r.entropy.size() == nb);
    REQUIRE(r.free_energy.size() == nb);
    REQUIRE(r.partition_function.size() == nb);
    for (std::size_t t = 0; t < nb; ++t) {
        CHECK(std::isfinite(r.energy[t]));
        CHECK(std::isfinite(r.heat_capacity[t]));
        CHECK(std::isfinite(r.entropy[t]));
        CHECK(std::isfinite(r.free_energy[t]));
        CHECK(std::isfinite(r.partition_function[t]));
    }
}

}  // namespace

TEST_CASE("ftlm_kernel_via_backend: full reorth on/off, ground-state "
          "estimate, thread scope",
          "[ftlm][thermal][wp10]") {
    constexpr std::uint64_t N   = 8;
    constexpr std::size_t   dim = std::size_t{1} << N;

    auto H = ed_tests::build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    ed::matvec::CpuBackend backend;
    MatvecCallable apply{H.get()};

    const double e0 = dense_ground_energy(apply, dim);

    ed::thermal::FtlmOptions opts;
    opts.num_samples = 4;
    opts.krylov_dim  = 60;
    opts.betas       = {0.1, 0.5, 1.0, 2.0, 20.0};
    opts.random_seed = 12345;

#ifdef _OPENMP
    const int omp_before = omp_get_max_threads();
#endif

    opts.full_reorthogonalization = false;
    const auto local = ed::thermal::detail::ftlm_kernel_via_backend(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

    opts.full_reorthogonalization = true;
    const auto full = ed::thermal::detail::ftlm_kernel_via_backend(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);

#ifdef _OPENMP
    CHECK(omp_get_max_threads() == omp_before);
#endif

    const std::size_t nb = opts.betas.size();
    require_finite(local, nb);
    require_finite(full, nb);

    // Lowest Ritz value of a 60-step Krylov space on a 256-dim ring is
    // converged to machine precision.
    REQUIRE(std::isfinite(local.ground_state_estimate));
    REQUIRE(std::isfinite(full.ground_state_estimate));
    CHECK(std::abs(local.ground_state_estimate - e0) < 1e-8);
    CHECK(std::abs(full.ground_state_estimate - e0) < 1e-8);

    // Deep in the gapped regime both policies see the ground state only.
    CHECK(std::abs(local.energy.back() - e0) < 1e-4);
    CHECK(std::abs(full.energy.back() - e0) < 1e-4);

    // Same random vectors, different reorth policy: the thermodynamics
    // agree to well within the stochastic error of the trace.
    for (std::size_t t = 0; t < nb; ++t) {
        CHECK(std::abs(local.energy[t] - full.energy[t])
              < 1e-6 * (1.0 + std::abs(full.energy[t])));
    }
}
