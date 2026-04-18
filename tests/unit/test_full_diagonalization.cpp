// =============================================================================
// test_full_diagonalization
//
// Drives the CPU `full_diagonalization()` entry point end-to-end on a small
// Heisenberg chain (N=4, dim=16) and cross-checks its spectrum with an
// independent Eigen SelfAdjointEigenSolver on the same operator.
// =============================================================================

#include "common/test_harness.h"

#include <ed/solvers/lanczos.h>

#include <memory>
#include <vector>

using namespace ed_tests;

static void run_full_diag_for_N(TestContext& ctx, uint64_t N, double tol) {
    auto op = build_heisenberg_chain(N, 1.0);
    const uint64_t dim = 1ULL << N;
    auto ref = reference_from_operator(*op, dim);

    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };

    std::string outdir = make_scratch_dir("full_diag_N" + std::to_string(N));

    std::vector<double> eigs;
    full_diagonalization(Hv, dim, /*num_eigs=*/dim, eigs, outdir,
                         /*compute_eigenvectors=*/false);

    check_eigs_close(ctx, eigs, ref.eigs, ref.eigs.size(), tol,
                     "full_diagonalization matches dense reference (N=" +
                         std::to_string(N) + ")");
}

int main() {
    TestContext ctx("test_full_diagonalization");
    run_full_diag_for_N(ctx, 4, 1e-9);
    run_full_diag_for_N(ctx, 6, 1e-8);
    return ctx.summary_exit_code();
}
