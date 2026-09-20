// =============================================================================
// tests/unit/test_ftlm_sample_seed.cpp
//
// WP10 C3/C6: the FTLM body (``ed::thermal::detail::ftlm_kernel_via_backend``)
// draws sample ``s`` as
//     generateGaussianRandomVector(N, sample_engine(resolve_base_seed(seed), s))
// (the recipe the retired Gen-1 CPU driver used, shared with OFTLM through
// ``sample_seed.h``). Until C6 this was checked by comparing against that
// driver; with the driver deleted the same protection is kept by
//   * recording every draw through a pass-through ``seed_transform`` and
//     requiring it to equal the recipe bit for bit;
//   * requiring two runs with the same options to be bit-identical;
//   * requiring the CpuBackend front door ``ftlm_kernel`` to return exactly
//     the body's curves on the T = 1/beta grid, and the caller's grid
//     verbatim when ``opts.temperatures`` is set.
// Checked on an 8-site Heisenberg ring (dim 256) and a 14-site ring
// (dim 16384, above the 8192 thread-budget threshold of
// ``auto_threads_for_dim``), full reorthogonalisation off and on.
// Reorth on/off agreement and the dense ground energy are covered by
// test_ftlm_backend_body; the curves themselves are pinned by the goldens.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <algorithm>
#include <cmath>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/matvec.h>
#include <ed/solvers/lanczos.h>
#include <ed/thermal/ftlm_kernel.h>
#include <ed/thermal/sample_seed.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using Complex = std::complex<double>;

namespace {

struct MatvecCallable {
    const ed::matvec::MatVecOperator* op;
    void operator()(const Complex* in, Complex* out, std::size_t n) const {
        op->apply(in, out, n);
    }
};

void require_identical(const std::string& name,
                       const std::vector<double>& a,
                       const std::vector<double>& b) {
    INFO("curve " << name);
    REQUIRE(a.size() == b.size());
    for (std::size_t t = 0; t < a.size(); ++t) {
        INFO("t index " << t << ": " << a[t] << " vs " << b[t]);
        CHECK(a[t] == b[t]);
    }
}

void require_identical(const ed::thermal::FtlmResult& a,
                       const ed::thermal::FtlmResult& b) {
    require_identical("energy",             a.energy,             b.energy);
    require_identical("specific_heat",      a.heat_capacity,      b.heat_capacity);
    require_identical("entropy",            a.entropy,            b.entropy);
    require_identical("free_energy",        a.free_energy,        b.free_energy);
    require_identical("partition_function", a.partition_function, b.partition_function);
    CHECK(a.ground_state_estimate == b.ground_state_estimate);
}

void check_seed_contract(std::uint64_t n_sites, std::size_t samples,
                         std::size_t krylov, bool full_reorth) {
    const std::size_t dim = std::size_t{1} << n_sites;
    auto H = ed_tests::build_heisenberg_chain(n_sites, 1.0, /*periodic=*/true);
    MatvecCallable apply{H.get()};
    const auto& backend = ed::matvec::default_cpu_backend();

    const std::vector<double> betas = {0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0};
    std::vector<double> temperatures;
    for (double b : betas) temperatures.push_back(1.0 / b);

    ed::thermal::FtlmOptions opts;
    opts.num_samples              = samples;
    opts.krylov_dim               = krylov;
    opts.betas                    = betas;
    opts.random_seed              = 20260919;
    opts.full_reorthogonalization = full_reorth;

    INFO("n_sites " << n_sites << ", full_reorth " << full_reorth);

    // 1. Sample s starts from the shared recipe, bit for bit.
    {
        std::vector<std::vector<Complex>> drawn;
        ed::thermal::FtlmOptions rec = opts;
        rec.seed_transform = [&drawn](Complex* v, std::size_t n) {
            drawn.emplace_back(v, v + n);
        };
        (void)ed::thermal::detail::ftlm_kernel_via_backend(
            backend, apply, dim, static_cast<std::uint64_t>(dim), rec);
        REQUIRE(drawn.size() == samples);
        const std::uint64_t base = ed::thermal::resolve_base_seed(opts.random_seed);
        REQUIRE(base == opts.random_seed);
        for (std::size_t s = 0; s < samples; ++s) {
            INFO("sample " << s);
            std::mt19937 eng = ed::thermal::sample_engine(base, s);
            const ComplexVector ref =
                generateGaussianRandomVector(static_cast<int>(dim), eng);
            REQUIRE(ref.size() == dim);
            REQUIRE(drawn[s].size() == dim);
            // The draw is normalised with dznrm2, a threaded reduction above ~8192 entries,
            // and the kernel runs it under its own thread budget: above that size every
            // entry can differ from this reference in the last bits of the common scale
            // factor (measured 4.6e-16 at 14 sites, job 60564039). A different random
            // STREAM would differ by O(1), which is what this pins. Below the threaded
            // size the two draws are bit-identical.
            double worst = 0.0;
            for (std::size_t i = 0; i < dim; ++i)
                worst = std::max(worst,
                                 std::abs(drawn[s][i] - ref[i]) / std::max(std::abs(ref[i]), 1e-300));
            INFO("worst relative element difference " << worst);
            if (dim <= 8192) CHECK(worst == 0.0);
            else             CHECK(worst < 1e-14);
        }
    }

    // 2. Same options twice -> bit-identical curves.
    const auto first = ed::thermal::detail::ftlm_kernel_via_backend(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);
    const auto second = ed::thermal::detail::ftlm_kernel_via_backend(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);
    REQUIRE(first.energy.size() == betas.size());
    require_identical(first, second);

    // 3. The CpuBackend front door runs this body on the 1/beta grid.
    const auto front = ed::thermal::ftlm_kernel(
        backend, apply, dim, static_cast<std::uint64_t>(dim), opts);
    CHECK(front.betas == betas);
    CHECK(front.temperatures == temperatures);
    require_identical(front, first);

    // 4. An exact temperature grid is reported verbatim.
    ed::thermal::FtlmOptions exact = opts;
    exact.betas.clear();
    exact.temperatures = temperatures;
    const auto front_t = ed::thermal::ftlm_kernel(
        backend, apply, dim, static_cast<std::uint64_t>(dim), exact);
    CHECK(front_t.temperatures == temperatures);
    REQUIRE(front_t.energy.size() == temperatures.size());
}

}  // namespace

TEST_CASE("ftlm_kernel draws the shared per-sample seed: 8-site ring",
          "[ftlm][thermal][wp10]") {
    SECTION("local reorthogonalisation") { check_seed_contract(8, 4, 40, false); }
    SECTION("full reorthogonalisation")  { check_seed_contract(8, 4, 40, true); }
}

TEST_CASE("ftlm_kernel draws the shared per-sample seed: "
          "14-site ring (dim > 8192)",
          "[ftlm][thermal][wp10]") {
    SECTION("local reorthogonalisation") { check_seed_contract(14, 2, 25, false); }
    SECTION("full reorthogonalisation")  { check_seed_contract(14, 2, 25, true); }
}
