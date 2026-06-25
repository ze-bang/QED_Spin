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

namespace {

// Build the dense block Hamiltonian H_Γ = Φ† H Φ over a set of SAB vectors,
// applying H term-by-term over the orbit support via `connect` (no 2^N vector).
// Shared by the spectrum / thermodynamics / DSSF consumers.
[[nodiscard]] Eigen::MatrixXcd
block_hamiltonian(const ConnectFn& connect, const std::vector<SABVector>& sab) {
    const int nb = static_cast<int>(sab.size());
    std::unordered_map<std::uint64_t, std::vector<std::pair<int, Complex>>> bystate;
    for (int k = 0; k < nb; ++k)
        for (std::size_t t = 0; t < sab[static_cast<std::size_t>(k)].states.size(); ++t)
            bystate[sab[static_cast<std::size_t>(k)].states[t]]
                .emplace_back(k, sab[static_cast<std::size_t>(k)].coeffs[t]);

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
    return Hg;
}

}  // namespace

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

SymBlocksPacked build_symmetry_blocks_packed(
    const ConnectFn&                     connect,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    int                                  n_up)
{
    SymBlocksPacked P;
    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const auto sab = build_sab_partition0(gi, max_clique, static_cast<int>(g), n_sites, n_up);
        if (sab.empty()) continue;
        const Eigen::MatrixXcd Hg = block_hamiltonian(connect, sab);
        // Same guard as the CPU spectrum path: a non-Hermitian block means H does
        // not commute with the supplied group. Catch it HERE (host) so the GPU
        // eigensolver never silently diagonalises a wrong (one-triangle) matrix.
        if ((Hg - Hg.adjoint()).norm() > 1e-8 * std::max(1.0, Hg.norm()))
            throw std::runtime_error(
                "build_symmetry_blocks_packed: H_Γ not Hermitian — the "
                "Hamiltonian does not commute with the supplied symmetry group");
        const int nb = static_cast<int>(sab.size());
        P.offset.push_back(P.data.size());
        P.block_dim.push_back(nb);
        P.block_irrep_dim.push_back(gi.irreps[g].dim);
        // Column-major pack (cuSOLVER / LAPACK convention).
        for (int col = 0; col < nb; ++col)
            for (int row = 0; row < nb; ++row)
                P.data.push_back(Hg(row, col));
    }
    return P;
}

ThermodynamicData symmetry_adapted_thermodynamics(
    const ConnectFn&                     connect,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    const std::vector<double>&           temperatures,
    int                                  n_up)
{
    // Reuse the block diagonalisation: the full eigenvalue list (with d_Γ
    // multiplicities) IS the symmetry-reduced spectrum; canonical thermo is exact.
    const auto spec = symmetry_adapted_spectrum_terms(connect, gi, max_clique, n_sites, n_up);
    return canonical_thermo_from_eigs(spec.eigenvalues, temperatures);
}

SymDSSFResult symmetry_adapted_ground_state_dssf(
    const ConnectFn&                     h_connect,
    const ConnectFn&                     o_connect,
    const GroupIrreps&                   gi,
    const std::vector<std::vector<int>>& max_clique,
    int                                  n_sites,
    double                               omega_min,
    double                               omega_max,
    int                                  n_omega,
    double                               broadening)
{
    // All eigenstates over the FULL Hilbert space (every irrep), so a final
    // state |n> is captured even when O changes the irrep/Sz.
    struct EigState { double energy; std::map<std::uint64_t, Complex> amp; };
    std::vector<EigState> states;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        // Sum over ALL d_Γ partners so the eigenbasis spans the full isotypic
        // space — required for a complete Σ_n |<n|O|0>|² (partner-0 alone is the
        // partner-0 SLICE and undercounts the spectral weight for d≥2 irreps).
        for (int partner = 0; partner < d; ++partner) {
            const auto sab = build_sab_partition0(
                gi, max_clique, static_cast<int>(g), n_sites, -1, partner);
            if (sab.empty()) continue;
            const Eigen::MatrixXcd Hg = block_hamiltonian(h_connect, sab);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hg);
            const int nb = static_cast<int>(sab.size());
            for (int e = 0; e < es.eigenvalues().size(); ++e) {
                EigState st;
                st.energy = es.eigenvalues()(e);
                const Eigen::VectorXcd v = es.eigenvectors().col(e);  // SAB-basis coeffs
                for (int j = 0; j < nb; ++j) {
                    if (std::abs(v(j)) < 1e-300) continue;
                    const auto& sj = sab[static_cast<std::size_t>(j)];
                    for (std::size_t t = 0; t < sj.states.size(); ++t)
                        st.amp[sj.states[t]] += v(j) * sj.coeffs[t];   // expand to comp basis
                }
                states.push_back(std::move(st));
            }
        }
    }

    SymDSSFResult R;
    R.omega.resize(static_cast<std::size_t>(n_omega));
    R.spectral.assign(static_cast<std::size_t>(n_omega), 0.0);
    if (states.empty()) return R;

    // Ground state |0> = global minimum.
    std::size_t gs = 0;
    for (std::size_t i = 1; i < states.size(); ++i)
        if (states[i].energy < states[gs].energy) gs = i;
    const double E0 = states[gs].energy;
    R.ground_energy = E0;

    // |chi> = O|0> in the computational basis.
    std::map<std::uint64_t, Complex> chi;
    for (const auto& kv : states[gs].amp) {
        const Complex a0 = kv.second;
        o_connect(kv.first, [&](std::uint64_t sprime, Complex o) {
            chi[sprime] += a0 * o;
        });
    }

    // omega grid + Lehmann sum  S(ω) = Σ_n |<n|chi>|² · (η/π)/((ω-(E_n-E0))²+η²).
    const double dw = (n_omega > 1) ? (omega_max - omega_min) / (n_omega - 1) : 0.0;
    for (int i = 0; i < n_omega; ++i) R.omega[static_cast<std::size_t>(i)] = omega_min + dw * i;
    const double eta = broadening, inv_pi_eta = eta / M_PI;

    for (const auto& st : states) {
        Complex ov(0.0, 0.0);
        // <n|chi> = Σ_s conj(n_s) chi_s  (iterate the smaller support)
        if (st.amp.size() <= chi.size()) {
            for (const auto& kv : st.amp) {
                auto it = chi.find(kv.first);
                if (it != chi.end()) ov += std::conj(kv.second) * it->second;
            }
        } else {
            for (const auto& kv : chi) {
                auto it = st.amp.find(kv.first);
                if (it != st.amp.end()) ov += std::conj(it->second) * kv.second;
            }
        }
        const double w = std::norm(ov);
        if (w < 1e-300) continue;
        R.total_weight += w;
        const double de = st.energy - E0;
        for (int i = 0; i < n_omega; ++i) {
            const double x = R.omega[static_cast<std::size_t>(i)] - de;
            R.spectral[static_cast<std::size_t>(i)] += w * inv_pi_eta / (x * x + eta * eta);
        }
    }
    return R;
}

}  // namespace ed::symmetry
