// =============================================================================
// src/symmetry/symmetry_adapted.cpp  --  see symmetry_adapted.h
// =============================================================================

#include <ed/symmetry/symmetry_adapted.h>

#include <ed/core/basis_utils.h>   // applyPermutation

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <map>
#include <vector>

namespace ed::symmetry {

using Complex = std::complex<double>;

std::vector<SABVector>
build_sab_partition0(const GroupIrreps&                   gi,
                     const std::vector<std::vector<int>>& max_clique,
                     int                                  irrep_index,
                     int                                  n_sites)
{
    const IrrepData& ir = gi.irreps[static_cast<std::size_t>(irrep_index)];
    const int        d  = ir.dim;
    const int        G  = gi.order;

    // Partner-0 diagonal matrix element D^Γ_{00}(g) for every group element.
    std::vector<Complex> D00(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        D00[static_cast<std::size_t>(g)] =
            ir.matrices[static_cast<std::size_t>(g)][0];  // (0,0) of a d×d row-major

    const std::uint64_t full = std::uint64_t{1} << n_sites;
    std::vector<SABVector> out;

    for (std::uint64_t s = 0; s < full; ++s) {
        // Process orbit representatives only (orbit minimum).
        std::uint64_t rep = s;
        for (int g = 0; g < G; ++g)
            rep = std::min(rep, applyPermutation(s, max_clique[static_cast<std::size_t>(g)]));
        if (rep != s) continue;

        // Distinct orbit states + coordinate index.
        std::map<std::uint64_t, int> coord;
        std::vector<std::uint64_t>   orbit;
        for (int g = 0; g < G; ++g) {
            const std::uint64_t t = applyPermutation(s, max_clique[static_cast<std::size_t>(g)]);
            if (coord.emplace(t, static_cast<int>(orbit.size())).second) orbit.push_back(t);
        }
        const int M = static_cast<int>(orbit.size());

        // Candidate matrix: column k = P^Γ_{00} |orbit[k]>  over orbit coords.
        //   P^Γ_{00}|x> = Σ_g conj(D00(g)) |perm_g(x)>   (constant d_Γ/|G| dropped;
        //   the SVD column basis is gauge/scale invariant).
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(M, M);
        for (int k = 0; k < M; ++k) {
            for (int g = 0; g < G; ++g) {
                const std::uint64_t t =
                    applyPermutation(orbit[static_cast<std::size_t>(k)],
                                     max_clique[static_cast<std::size_t>(g)]);
                A(coord[t], k) += std::conj(D00[static_cast<std::size_t>(g)]);
            }
        }

        // Orthonormal column-space basis = the m_Γ(O) partner-0 SAB vectors.
        // A = (|G|/d_Γ)·P^Γ_{00} with P an orthogonal projector, so the genuine
        // singular values are |G|/d_Γ = O(1) while an ABSENT irrep gives A≈0
        // (~1e-14 noise). Use an ABSOLUTE threshold (genuine σ_min ≥ |G|/d_max ≥
        // √|G| ≫ tol) so absent irreps contribute zero vectors.
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU);
        const auto& sv = svd.singularValues();
        const double tol = 1e-8 * static_cast<double>(G);
        for (int i = 0; i < sv.size(); ++i) {
            if (sv(i) <= tol) continue;
            SABVector v;
            const Eigen::VectorXcd u = svd.matrixU().col(i);  // unit-norm
            for (int c = 0; c < M; ++c) {
                if (std::abs(u(c)) > 1e-12) {
                    v.states.push_back(orbit[static_cast<std::size_t>(c)]);
                    v.coeffs.push_back(u(c));
                }
            }
            out.push_back(std::move(v));
        }
    }

    (void)d;  // d only enters the (dropped) projector prefactor + degeneracy bookkeeping
    return out;
}

}  // namespace ed::symmetry
