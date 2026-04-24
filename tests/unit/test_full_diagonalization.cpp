// =============================================================================
// test_full_diagonalization (Catch2 v3, P1.8 / audit Q12)
//
// Drives the CPU `full_diagonalization()` entry point end-to-end on a small
// Heisenberg chain (N=4, dim=16; N=6, dim=64) and cross-checks its spectrum
// with an independent Eigen SelfAdjointEigenSolver on the same operator.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/solvers/lanczos.h>

#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;

namespace {

void run_full_diag_for_N(uint64_t N, double tol) {
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

    require_eigs_close(eigs, ref.eigs, ref.eigs.size(), tol,
                       "full_diagonalization N=" + std::to_string(N));
}

} // namespace

TEST_CASE("full_diagonalization matches dense reference for N=4",
          "[full_diag][N4]") {
    run_full_diag_for_N(4, 1e-9);
}

TEST_CASE("full_diagonalization matches dense reference for N=6",
          "[full_diag][N6]") {
    run_full_diag_for_N(6, 1e-8);
}
