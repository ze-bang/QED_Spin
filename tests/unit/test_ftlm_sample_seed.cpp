// =============================================================================
// tests/unit/test_ftlm_sample_seed.cpp
//
// The FTLM body (``ed::thermal::ftlm_kernel``)
// draws sample ``s`` as
//     generateGaussianRandomVector(N, sample_engine(resolve_base_seed(seed), s))
// (the recipe shared with OFTLM through ``sample_seed.h``). Pinned by
//   * recording every draw through a pass-through ``seed_transform`` and
//     requiring it to equal the recipe bit for bit;
//   * requiring two runs with the same options to give bit-identical ln Z, E and V.
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
#include <ed/core/linear_operator.h>
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
    const ed::LinearOperator* op;
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
    require_identical("lnZ", a.curves.lnZ, b.curves.lnZ);
    require_identical("E",   a.curves.E,   b.curves.E);
    require_identical("V",   a.curves.V,   b.curves.V);
    CHECK(a.ground_state_estimate == b.ground_state_estimate);
}

void check_seed_contract(std::uint64_t n_sites, std::size_t samples,
                         std::size_t krylov, bool full_reorth) {
    const std::size_t dim = std::size_t{1} << n_sites;
    auto H = ed_tests::build_heisenberg_chain(n_sites, 1.0, /*periodic=*/true);
    MatvecCallable apply{H.get()};
    const auto& backend = ed::matvec::default_cpu_backend();

    const std::vector<double> betas = {0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0};

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
        (void)ed::thermal::ftlm_kernel(
            backend, apply, dim, rec);
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
    const auto first = ed::thermal::ftlm_kernel(
        backend, apply, dim, opts);
    const auto second = ed::thermal::ftlm_kernel(
        backend, apply, dim, opts);
    REQUIRE(first.curves.E.size() == betas.size());
    require_identical(first, second);
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
