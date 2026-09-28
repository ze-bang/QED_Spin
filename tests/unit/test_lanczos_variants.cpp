// =============================================================================
// test_lanczos_variants (Catch2 v3)
//
// Drives the Lanczos solver exported from `ed/solvers/lanczos.h` on
// a small Heisenberg chain (N=8, dim=256) and checks that the lowest few
// eigenvalues match a dense Eigen reference.
//
// The solver runs in its own scratch directory (the ED_LANCZOS_DISK=1 variant
// of this binary exercises the on-disk basis path).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/io/lanczos_basis_buffer.h>
#include <ed/solvers/lanczos.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;

namespace {

struct Fixture {
    std::unique_ptr<Operator> op;
    uint64_t dim;
    std::vector<double> dense_eigs;
    std::function<void(const Complex*, Complex*, int)> Hv;
};

Fixture make_fixture(uint64_t N) {
    Fixture f;
    f.op = build_heisenberg_chain(N, 1.0, /*periodic=*/true);
    f.dim = 1ULL << N;
    auto ref = reference_from_operator(*f.op, f.dim);
    f.dense_eigs = ref.eigs;
    auto* op_ptr = f.op.get();
    f.Hv = [op_ptr](const Complex* in, Complex* out, int n) {
        op_ptr->apply(in, out, static_cast<size_t>(n));
    };
    return f;
}

void check_ground_state(const std::string& name, const Fixture& f,
                        std::vector<double> eigs, double tol) {
    INFO("solver=" << name << " ground-state");
    REQUIRE_FALSE(eigs.empty());
    std::sort(eigs.begin(), eigs.end());
    INFO("got=" << eigs.front()
         << " want=" << f.dense_eigs.front()
         << " |Δ|=" << std::abs(eigs.front() - f.dense_eigs.front())
         << " tol=" << tol);
    REQUIRE(std::abs(eigs.front() - f.dense_eigs.front()) <= tol);
}

void check_all_ritz_valid(const std::string& name, const Fixture& f,
                          const std::vector<double>& eigs, double tol) {
    double worst = 0.0;
    int worst_idx = -1;
    for (size_t i = 0; i < eigs.size(); ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (double e : f.dense_eigs) {
            double d = std::abs(eigs[i] - e);
            if (d < best) best = d;
        }
        if (best > worst) { worst = best; worst_idx = static_cast<int>(i); }
    }
    INFO("solver=" << name << " worst Ritz |Δ|=" << worst
         << " at idx=" << worst_idx << " tol=" << tol);
    REQUIRE(worst <= tol);
}

void run_solver(const std::string& name, const Fixture& f,
                double gs_tol, double ritz_tol,
                const std::function<void(const std::string&,
                                         std::vector<double>&)>& solver) {
    std::string outdir = make_scratch_dir("lanczos_" + name);
    std::vector<double> eigs;
    REQUIRE_NOTHROW(solver(outdir, eigs));
    check_ground_state(name, f, eigs, gs_tol);
    check_all_ritz_valid(name, f, eigs, ritz_tol);
}

} // namespace

TEST_CASE("Lanczos-family solvers reproduce dense spectrum (N=8)",
          "[lanczos][variants]") {
    // Problem sizing rationale: N=8, dim=256, max_iter=30 → 12% Krylov ratio,
    // well inside the regime where the ground state converges to ~1e-10 for
    // every variant, independent of the random starting vector.
    const uint64_t N = 8;
    const uint64_t max_iter = 30;
    Fixture f = make_fixture(N);

    SECTION("lanczos") {
        run_solver("lanczos", f, 1e-8, 1e-6,
                   [&](const std::string& dir, std::vector<double>& e) {
                       lanczos(f.Hv, f.dim, max_iter, 3, 1e-12, e, dir, false);
                   });
    }
}
