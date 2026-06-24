// =============================================================================
// src/symmetry/symmetry_adapted.cpp  --  see symmetry_adapted.h
// =============================================================================

#include <ed/symmetry/symmetry_adapted.h>

#include <ed/core/basis_utils.h>   // applyPermutation
#include <ed/symmetry/gosper.h>    // next_bit_permutation (fixed-popcount enumeration)

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ed::symmetry {

using Complex = std::complex<double>;

std::vector<SABVector>
build_sab_partition0(const GroupIrreps&                   gi,
                     const std::vector<std::vector<int>>& max_clique,
                     int                                  irrep_index,
                     int                                  n_sites,
                     int                                  n_up)
{
    const IrrepData& ir = gi.irreps[static_cast<std::size_t>(irrep_index)];
    const int        G  = gi.order;

    // Partner-0 diagonal matrix element D^Γ_{00}(g) for every group element.
    std::vector<Complex> D00(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        D00[static_cast<std::size_t>(g)] =
            ir.matrices[static_cast<std::size_t>(g)][0];  // (0,0) of a d×d row-major

    std::vector<SABVector> out;

    // Per-orbit-representative SAB construction (independent of which subspace
    // the rep was enumerated from — orbits of fixed-popcount states stay in the
    // Sz sector because permutations preserve popcount).
    auto process_rep = [&](std::uint64_t s) {
        std::uint64_t rep = s;
        for (int g = 0; g < G; ++g)
            rep = std::min(rep, applyPermutation(s, max_clique[static_cast<std::size_t>(g)]));
        if (rep != s) return;  // only orbit minima

        std::map<std::uint64_t, int> coord;
        std::vector<std::uint64_t>   orbit;
        for (int g = 0; g < G; ++g) {
            const std::uint64_t t = applyPermutation(s, max_clique[static_cast<std::size_t>(g)]);
            if (coord.emplace(t, static_cast<int>(orbit.size())).second) orbit.push_back(t);
        }
        const int M = static_cast<int>(orbit.size());

        // Column k = P^Γ_{00}|orbit[k]> = Σ_g conj(D00(g)) |perm_g(orbit[k])>
        // (constant d_Γ/|G| dropped; SVD column basis is gauge/scale invariant).
        Eigen::MatrixXcd A = Eigen::MatrixXcd::Zero(M, M);
        for (int k = 0; k < M; ++k)
            for (int g = 0; g < G; ++g) {
                const std::uint64_t t =
                    applyPermutation(orbit[static_cast<std::size_t>(k)],
                                     max_clique[static_cast<std::size_t>(g)]);
                A(coord[t], k) += std::conj(D00[static_cast<std::size_t>(g)]);
            }

        // Orthonormal column space = the m_Γ(O) partner-0 SAB vectors. Genuine
        // singular values are |G|/d_Γ = O(1); an ABSENT irrep gives A≈0 (~1e-14),
        // so an ABSOLUTE threshold contributes zero spurious vectors.
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(A, Eigen::ComputeThinU);
        const auto& sv = svd.singularValues();
        const double tol = 1e-8 * static_cast<double>(G);
        for (int i = 0; i < sv.size(); ++i) {
            if (sv(i) <= tol) continue;
            SABVector v;
            const Eigen::VectorXcd u = svd.matrixU().col(i);  // unit-norm
            for (int c = 0; c < M; ++c)
                if (std::abs(u(c)) > 1e-12) {
                    v.states.push_back(orbit[static_cast<std::size_t>(c)]);
                    v.coeffs.push_back(u(c));
                }
            out.push_back(std::move(v));
        }
    };

    if (n_up < 0) {
        const std::uint64_t full = std::uint64_t{1} << n_sites;
        for (std::uint64_t s = 0; s < full; ++s) process_rep(s);
    } else if (n_up <= n_sites) {
        // Fixed-Sz: enumerate states of popcount n_up via Gosper's hack.
        if (n_up == 0) {
            process_rep(0);
        } else {
            const std::uint64_t lim =
                (n_sites < 64) ? (std::uint64_t{1} << n_sites)
                               : std::numeric_limits<std::uint64_t>::max();
            std::uint64_t s = (std::uint64_t{1} << n_up) - 1;
            while (s < lim) {
                process_rep(s);
                const std::uint64_t nxt = next_bit_permutation(s);
                if (nxt <= s) break;  // wrap-around / end of sequence
                s = nxt;
            }
        }
    }
    return out;
}

SymAdaptedSpectrum symmetry_adapted_spectrum(
    const std::function<void(const Complex*, Complex*, std::uint64_t)>& H_full,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  n_up)
{
    const std::uint64_t full = std::uint64_t{1} << n_sites;
    SymAdaptedSpectrum out;

    std::vector<Complex> embed(full), happ(full);
    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        const auto sab = build_sab_partition0(gi, max_clique, static_cast<int>(g), n_sites, n_up);
        if (sab.empty()) continue;
        const int nb = static_cast<int>(sab.size());

        // H_Γ[k][j] = <φ_k| H |φ_j>, built one column at a time via the matvec.
        Eigen::MatrixXcd Hg(nb, nb);
        for (int j = 0; j < nb; ++j) {
            std::fill(embed.begin(), embed.end(), Complex(0.0, 0.0));
            const auto& vj = sab[static_cast<std::size_t>(j)];
            for (std::size_t t = 0; t < vj.states.size(); ++t)
                embed[vj.states[t]] = vj.coeffs[t];
            H_full(embed.data(), happ.data(), full);
            for (int k = 0; k < nb; ++k) {
                const auto& vk = sab[static_cast<std::size_t>(k)];
                Complex acc(0.0, 0.0);
                for (std::size_t t = 0; t < vk.states.size(); ++t)
                    acc += std::conj(vk.coeffs[t]) * happ[vk.states[t]];
                Hg(k, j) = acc;
            }
        }

        if ((Hg - Hg.adjoint()).norm() > 1e-8 * std::max(1.0, Hg.norm()))
            throw std::runtime_error(
                "symmetry_adapted_spectrum: H_Γ not Hermitian — the Hamiltonian "
                "does not commute with the supplied symmetry group");

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hg);
        for (int e = 0; e < es.eigenvalues().size(); ++e)
            for (int rep = 0; rep < d; ++rep)            // physical d_Γ degeneracy
                out.eigenvalues.push_back(es.eigenvalues()(e));
        out.block_irrep_dim.push_back(d);
        out.block_size.push_back(nb);
    }

    std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
    return out;
}

SymAdaptedSpectrum symmetry_adapted_spectrum_terms(
    const ConnectFn&                     connect,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  n_up)
{
    SymAdaptedSpectrum out;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        const auto sab = build_sab_partition0(gi, max_clique, static_cast<int>(g), n_sites, n_up);
        if (sab.empty()) continue;
        const int nb = static_cast<int>(sab.size());

        // state s' -> [(basis index k, coeff c^k_{s'})] over THIS irrep's SAB.
        std::unordered_map<std::uint64_t, std::vector<std::pair<int, Complex>>> bystate;
        for (int k = 0; k < nb; ++k) {
            const auto& vk = sab[static_cast<std::size_t>(k)];
            for (std::size_t t = 0; t < vk.states.size(); ++t)
                bystate[vk.states[t]].emplace_back(k, vk.coeffs[t]);
        }

        // H_Γ[k][j] = Σ_{s∈φ_j} c^j_s Σ_{s': <s'|H|s>} h Σ_{k: s'∈φ_k} conj(c^k_{s'})
        // -- apply H term-by-term over the orbit support only (no 2^n vector).
        Eigen::MatrixXcd Hg = Eigen::MatrixXcd::Zero(nb, nb);
        for (int j = 0; j < nb; ++j) {
            const auto& vj = sab[static_cast<std::size_t>(j)];
            for (std::size_t t = 0; t < vj.states.size(); ++t) {
                const Complex cjs = vj.coeffs[t];
                connect(vj.states[t], [&](std::uint64_t sprime, Complex h) {
                    auto it = bystate.find(sprime);
                    if (it == bystate.end()) return;
                    for (const auto& kc : it->second)
                        Hg(kc.first, j) += std::conj(kc.second) * h * cjs;
                });
            }
        }

        if ((Hg - Hg.adjoint()).norm() > 1e-8 * std::max(1.0, Hg.norm()))
            throw std::runtime_error(
                "symmetry_adapted_spectrum_terms: H_Γ not Hermitian — the "
                "Hamiltonian does not commute with the supplied symmetry group");

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hg);
        for (int e = 0; e < es.eigenvalues().size(); ++e)
            for (int rep = 0; rep < d; ++rep)
                out.eigenvalues.push_back(es.eigenvalues()(e));
        out.block_irrep_dim.push_back(d);
        out.block_size.push_back(nb);
    }

    std::sort(out.eigenvalues.begin(), out.eigenvalues.end());
    return out;
}

}  // namespace ed::symmetry
