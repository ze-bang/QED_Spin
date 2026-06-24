// =============================================================================
// tests/unit/test_symmetry_adapted.cpp
//
// End-to-end NON-ABELIAN symmetry-adapted ED: 4-site Heisenberg square with its
// full point group C4v / D4 (order 8, includes a 2-D irrep). Build the
// symmetry-adapted basis for every irrep, project H into each Γ-block, and
// recombine the eigenvalues (each ×d_Γ). Assert:
//   * completeness  Σ_Γ d_Γ · (#SAB vectors of Γ) == 2^N,
//   * the recombined spectrum matches brute-force full diagonalization to 1e-9,
//   * the 2-D irrep block actually appears (genuinely non-abelian reduction),
//     i.e. blocks shrink by the full |G|, not just an abelian subgroup.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/basis_utils.h>          // applyPermutation
#include <ed/symmetry/group.h>
#include <ed/symmetry/irreps.h>
#include <ed/symmetry/symmetry_adapted.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <complex>
#include <vector>

using ed::sym::generate_group;
using ed::sym::Permutation;
using ed::symmetry::build_sab_partition0;
using ed::symmetry::decompose_irreps;
using Complex = std::complex<double>;

namespace {

// Dense Heisenberg H = Σ_bonds S_i·S_j on the periodic 4-site chain (= square).
Eigen::MatrixXcd heisenberg_square(int N) {
    const std::uint64_t dim = std::uint64_t{1} << N;
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t b = 0; b < dim; ++b) {
        for (int i = 0; i < N; ++i) {
            const int j = (i + 1) % N;
            const double si = ((b >> i) & 1) ? 0.5 : -0.5;
            const double sj = ((b >> j) & 1) ? 0.5 : -0.5;
            H(b, b) += si * sj;                                   // Sz Sz
            if (((b >> i) & 1) != ((b >> j) & 1)) {               // (S+S- + S-S+)/2
                const std::uint64_t bp = b ^ ((1ull << i) | (1ull << j));
                H(bp, b) += 0.5;
            }
        }
    }
    return H;
}

}  // namespace

TEST_CASE("non-abelian SAB: C4v Heisenberg square == brute force (with d_Γ degeneracy)",
          "[symmetry_adapted][nonabelian]") {
    const int N = 4;
    const Permutation r{1, 2, 3, 0};   // 4-fold rotation
    const Permutation s{0, 3, 2, 1};   // reflection
    auto Gp = generate_group({r, s});
    REQUIRE(Gp.size() == 8);

    auto gi = decompose_irreps(Gp, N);
    REQUIRE_FALSE(gi.is_abelian());    // exercises the 2-D irrep path

    const Eigen::MatrixXcd H = heisenberg_square(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ref(H);
    std::vector<double> brute(ref.eigenvalues().data(),
                              ref.eigenvalues().data() + ref.eigenvalues().size());

    std::vector<double> recombined;
    long long completeness = 0;
    bool saw_2d_block = false;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        const auto sab = build_sab_partition0(gi, Gp, static_cast<int>(g), N);
        if (sab.empty()) continue;
        const int nb = static_cast<int>(sab.size());
        completeness += static_cast<long long>(d) * nb;
        if (d == 2) saw_2d_block = true;

        // Φ: (2^N × nb) orthonormal columns = the partner-0 SAB vectors.
        Eigen::MatrixXcd Phi = Eigen::MatrixXcd::Zero(std::uint64_t{1} << N, nb);
        for (int j = 0; j < nb; ++j)
            for (std::size_t k = 0; k < sab[static_cast<std::size_t>(j)].states.size(); ++k)
                Phi(sab[static_cast<std::size_t>(j)].states[k], j) =
                    sab[static_cast<std::size_t>(j)].coeffs[k];

        const Eigen::MatrixXcd Hg = Phi.adjoint() * H * Phi;        // Γ-block
        // Hermiticity sanity (H commutes with the group).
        REQUIRE((Hg - Hg.adjoint()).norm() < 1e-9);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hg);
        for (int e = 0; e < es.eigenvalues().size(); ++e)
            for (int rep = 0; rep < d; ++rep)                       // physical d_Γ degeneracy
                recombined.push_back(es.eigenvalues()(e));
    }

    REQUIRE(saw_2d_block);
    REQUIRE(completeness == (1LL << N));                            // completeness

    std::sort(recombined.begin(), recombined.end());
    REQUIRE(recombined.size() == brute.size());
    for (std::size_t i = 0; i < brute.size(); ++i)
        REQUIRE(std::abs(recombined[i] - brute[i]) < 1e-9);        // spectrum match
}
