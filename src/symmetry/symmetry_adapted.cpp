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
#include <cmath>
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
                     int                                  n_up,
                     int                                  partner)
{
    const IrrepData& ir = gi.irreps[static_cast<std::size_t>(irrep_index)];
    const int        G  = gi.order;
    const int        d  = ir.dim;

    // Partner-`partner` diagonal element D^Γ_{pp}(g) for every group element.
    // Summing eigenstates over ALL partners p = 0..d-1 spans the full isotypic
    // space (needed for complete matrix-element sums, e.g. DSSF); a single
    // partner suffices for eigenvalues (GS / thermodynamics).
    std::vector<Complex> D00(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        D00[static_cast<std::size_t>(g)] =
            ir.matrices[static_cast<std::size_t>(g)]
                       [static_cast<std::size_t>(partner) * d + partner];

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

::SymmetrySector build_symmetry_adapted_sector(
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  irrep_index,
    int                                  n_sites,
    int                                  n_up,
    int                                  partner)
{
    ::SymmetrySector sec;
    sec.sector_id        = static_cast<std::uint64_t>(irrep_index);
    sec.quantum_numbers  = {irrep_index};
    const auto sab = build_sab_partition0(gi, max_clique, irrep_index, n_sites, n_up, partner);
    sec.basis_states.reserve(sab.size());
    for (const auto& v : sab) {
        ::SymBasisState bs;
        bs.orbit_elements     = v.states;
        bs.orbit_coefficients = v.coeffs;
        bs.quantum_numbers    = {irrep_index};
        // SAB vectors are orthonormal: norm = 1, and the matvec uses group_norm = 1
        // (no 1/|G|), so SymmetryBasisPolicy's coeff_modifier reduces to conj(c_{s'}).
        bs.norm     = 1.0;
        bs.inv_norm = 1.0;
        bs.orbit_rep = v.states.empty()
            ? 0ULL
            : *std::min_element(v.states.begin(), v.states.end());
        bs.sortOrbit();   // sort orbit_elements (+ parallel coeffs) for findCoeff; refreshes inv_norm
        sec.basis_states.push_back(std::move(bs));
    }
    return sec;
}


// Exact canonical thermodynamics from a full eigenvalue list (multiplicities
// already folded in). Z(β)=Σ e^{-βE}; reference-shifted by E0 for stability.
// Public so the GPU consumer reuses the identical reduction.
ThermodynamicData
canonical_thermo_from_eigs(const std::vector<double>& eigs,
                           const std::vector<double>& T) {
    ThermodynamicData td;
    td.temperatures = T;
    if (eigs.empty()) return td;
    const double E0 = *std::min_element(eigs.begin(), eigs.end());
    td.e_min = E0;
    for (double t : T) {
        const double beta = 1.0 / t;
        double Z = 0.0, E = 0.0, E2 = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - E0));
            Z += w; E += w * e; E2 += w * e * e;
        }
        const double Eavg  = E / Z;
        const double E2avg = E2 / Z;
        const double Cv    = beta * beta * (E2avg - Eavg * Eavg);
        const double F     = E0 - t * std::log(Z);   // -T ln Z_full
        const double S     = (Eavg - F) / t;
        td.energy.push_back(Eavg);
        td.specific_heat.push_back(Cv);
        td.free_energy.push_back(F);
        td.entropy.push_back(S);
    }
    return td;
}


}  // namespace ed::symmetry
