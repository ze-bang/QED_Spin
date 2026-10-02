// =============================================================================
// src/basis/irreps.cpp  --  numerical irrep decomposition (see irreps.h).
// =============================================================================

#include <ed/basis/irreps.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ed::symmetry {

using Complex = std::complex<double>;

std::vector<Complex> IrrepData::partner_diagonal(int n) const {
    std::vector<Complex> out;
    out.reserve(matrices.size());
    for (const auto& M : matrices) {
        out.push_back(M[static_cast<std::size_t>(n) * dim + n]);
    }
    return out;
}

namespace {

// Group product consistent with the STATE action U(g)|s> = |applyPermutation(s,
// perm_g)>: U(g)U(h) = U(g·h) with (g·h)[i] = perm_h[perm_g[i]]. Using this
// convention makes the extracted D^Γ(g) usable directly in the matvec.
[[nodiscard]] std::vector<int>
compose(const std::vector<int>& pg, const std::vector<int>& ph) {
    const std::size_t n = pg.size();
    std::vector<int> c(n);
    for (std::size_t i = 0; i < n; ++i) c[i] = ph[static_cast<std::size_t>(pg[i])];
    return c;
}

// One decomposition attempt with a given RNG seed. Returns false (and leaves
// `out` untouched) if the random Hermitian commutant element failed to separate
// the irreps cleanly (caller retries with a new seed). Works purely from the
// (mult, inverse) tables -- the group can be abstract (little
// co-groups have no faithful site-permutation realisation).
[[nodiscard]] bool try_decompose(const std::vector<int>& inverse,
                                 const std::vector<std::vector<int>>& mult,
                                 std::uint64_t seed,
                                 std::vector<IrrepData>& out) {
    const int n = static_cast<int>(mult.size());

    // ---- Random Hermitian commutant element M = Σ_h c_h R(h) -----------------
    // R(h) e_x = e_{x·h^{-1}} = e_{mult[x][inverse[h]]}. Hermiticity needs
    // c_{inverse[h]} = conj(c_h).
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Complex> c(static_cast<std::size_t>(n));
    std::vector<char> set(static_cast<std::size_t>(n), 0);
    for (int h = 0; h < n; ++h) {
        if (set[static_cast<std::size_t>(h)]) continue;
        const int hi = inverse[static_cast<std::size_t>(h)];
        if (hi == h) {
            c[static_cast<std::size_t>(h)] = Complex(nd(gen), 0.0);  // involution -> real
        } else {
            const Complex z(nd(gen), nd(gen));
            c[static_cast<std::size_t>(h)]  = z;
            c[static_cast<std::size_t>(hi)] = std::conj(z);
            set[static_cast<std::size_t>(hi)] = 1;
        }
        set[static_cast<std::size_t>(h)] = 1;
    }

    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(n, n);
    for (int h = 0; h < n; ++h) {
        const int ih = inverse[static_cast<std::size_t>(h)];
        for (int x = 0; x < n; ++x) {
            const int row = mult[static_cast<std::size_t>(x)][static_cast<std::size_t>(ih)];
            M(row, x) += c[static_cast<std::size_t>(h)];
        }
    }
    // Symmetrise away round-off so the solver sees an exactly-Hermitian matrix.
    M = 0.5 * (M + M.adjoint());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    if (es.info() != Eigen::Success) return false;
    const Eigen::VectorXd evals = es.eigenvalues();
    const Eigen::MatrixXcd evecs = es.eigenvectors();

    // ---- Cluster eigenvalues into eigenspaces (each = one irreducible copy) --
    const double span = std::max(1.0, evals(n - 1) - evals(0));
    const double tol  = 1e-6 * span;
    std::vector<std::pair<int, int>> clusters;  // [begin, end)
    int b = 0;
    for (int i = 1; i <= n; ++i) {
        if (i == n || (evals(i) - evals(i - 1)) > tol) {
            clusters.emplace_back(b, i);
            b = i;
        }
    }

    // ---- Per eigenspace: D(g)_{mn} = <v_m| L(g) |v_n>, character = tr --------
    // L(g) e_x = e_{mult[g][x]} : (L(g) v)[mult[g][x]] = v[x].
    struct Cand { int dim; std::vector<std::vector<Complex>> mats;
                  std::vector<Complex> chi; };
    std::vector<Cand> cands;
    for (const auto& cl : clusters) {
        const int d = cl.second - cl.first;
        Eigen::MatrixXcd V = evecs.block(0, cl.first, n, d);  // n x d orthonormal
        Cand cand;
        cand.dim = d;
        cand.mats.resize(static_cast<std::size_t>(n));
        cand.chi.assign(static_cast<std::size_t>(n), Complex(0.0, 0.0));
        Eigen::MatrixXcd Lv(n, d);
        for (int g = 0; g < n; ++g) {
            // Lv = L(g) V : permute rows.
            for (int x = 0; x < n; ++x)
                Lv.row(mult[static_cast<std::size_t>(g)][static_cast<std::size_t>(x)]) = V.row(x);
            const Eigen::MatrixXcd D = V.adjoint() * Lv;  // d x d = D^Γ(g)
            auto& flat = cand.mats[static_cast<std::size_t>(g)];
            flat.resize(static_cast<std::size_t>(d) * d);
            for (int m = 0; m < d; ++m)
                for (int nn = 0; nn < d; ++nn)
                    flat[static_cast<std::size_t>(m) * d + nn] = D(m, nn);
            cand.chi[static_cast<std::size_t>(g)] = D.trace();
        }
        cands.push_back(std::move(cand));
    }

    // ---- Deduplicate by character; one IrrepData per distinct irrep ---------
    std::vector<IrrepData> irreps;
    auto same_char = [&](const std::vector<Complex>& a,
                         const std::vector<Complex>& bb) {
        for (int g = 0; g < n; ++g)
            if (std::abs(a[static_cast<std::size_t>(g)] - bb[static_cast<std::size_t>(g)]) > 1e-5)
                return false;
        return true;
    };
    long long sum_d2 = 0;
    for (auto& cand : cands) {
        bool seen = false;
        for (const auto& ir : irreps)
            if (ir.dim == cand.dim && same_char(ir.character, cand.chi)) { seen = true; break; }
        if (seen) continue;
        IrrepData ir;
        ir.dim = cand.dim;
        ir.character = cand.chi;
        ir.matrices = std::move(cand.mats);
        sum_d2 += static_cast<long long>(ir.dim) * ir.dim;
        irreps.push_back(std::move(ir));
    }

    if (sum_d2 != static_cast<long long>(n)) return false;  // bad draw -> retry

    out = std::move(irreps);
    return true;
}

// e^{2 pi i p / L}, exact at the quarter turns.
[[nodiscard]] Complex root_of_unity(long long p, long long L) {
    p %= L;
    if ((4 * p) % L == 0) {
        switch ((4 * p) / L) {
            case 0: return {1.0, 0.0};
            case 1: return {0.0, 1.0};
            case 2: return {-1.0, 0.0};
            default: return {0.0, -1.0};
        }
    }
    const double th = 6.283185307179586476925286766559 * static_cast<double>(p) / static_cast<double>(L);
    return {std::cos(th), std::sin(th)};
}

// The characters of an ABELIAN group, exactly: chi(x) = e^{2 pi i phase(x) / L} with L the
// exponent (the lcm of the element orders) and integer phases. Built one generator at a time:
// with H the subgroup found so far and g outside it, let m be the least power with g^m in H;
// every element of <H, g> is h g^k (h in H, 0 <= k < m) exactly once, and each character chi
// of H extends in exactly m ways, chi'(g) = e^{2 pi i b / L} with m b = phase_chi(g^m) mod L,
// i.e. b = phase_chi(g^m) / m + j L / m. The trivial character comes first.
[[nodiscard]] std::vector<IrrepData> abelian_characters(const std::vector<std::vector<int>>& mult, int e) {
    const int n = static_cast<int>(mult.size());
    const auto mul = [&mult](int a, int b) {
        return mult[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)];
    };
    long long L = 1;
    for (int a = 0; a < n; ++a) {
        long long ord = 1;
        for (int x = a; x != e; x = mul(x, a)) ++ord;
        L = std::lcm(L, ord);
    }
    std::vector<int> elems{e};                                  // H, in construction order
    std::vector<char> in_H(static_cast<std::size_t>(n), 0);
    in_H[static_cast<std::size_t>(e)] = 1;
    std::vector<std::vector<long long>> phase{std::vector<long long>(static_cast<std::size_t>(n), 0)};
    for (int g = 0; g < n; ++g) {
        if (in_H[static_cast<std::size_t>(g)]) continue;
        std::vector<int> gk{e};                                  // g^0 .. g^(m-1)
        int x = g;
        while (!in_H[static_cast<std::size_t>(x)]) { gk.push_back(x); x = mul(x, g); }
        const int gm = x;                                        // g^m, in H
        const long long m = static_cast<long long>(gk.size());
        std::vector<int> grown;
        grown.reserve(elems.size() * gk.size());
        for (std::size_t k = 0; k < gk.size(); ++k)
            for (int h : elems) grown.push_back(mul(h, gk[k]));
        std::vector<std::vector<long long>> next;
        next.reserve(phase.size() * static_cast<std::size_t>(m));
        for (const auto& chi : phase) {
            const long long a = chi[static_cast<std::size_t>(gm)];
            if (a % m != 0) throw std::logic_error("abelian_characters: a character does not extend");
            for (long long j = 0; j < m; ++j) {
                const long long b = (a / m + j * (L / m)) % L;
                std::vector<long long> ext(static_cast<std::size_t>(n), 0);
                for (std::size_t k = 0; k < gk.size(); ++k)
                    for (std::size_t i = 0; i < elems.size(); ++i)
                        ext[static_cast<std::size_t>(grown[k * elems.size() + i])] =
                            (chi[static_cast<std::size_t>(elems[i])] + static_cast<long long>(k) * b) % L;
                next.push_back(std::move(ext));
            }
        }
        phase = std::move(next);
        elems = std::move(grown);
        for (int y : elems) in_H[static_cast<std::size_t>(y)] = 1;
    }
    std::vector<IrrepData> out;
    out.reserve(phase.size());
    for (const auto& chi : phase) {
        IrrepData ir;
        ir.dim = 1;
        ir.character.resize(static_cast<std::size_t>(n));
        ir.matrices.resize(static_cast<std::size_t>(n));
        for (int a = 0; a < n; ++a) {
            const Complex c = root_of_unity(chi[static_cast<std::size_t>(a)], L);
            ir.character[static_cast<std::size_t>(a)] = c;
            ir.matrices[static_cast<std::size_t>(a)] = {c};
        }
        out.push_back(std::move(ir));
    }
    return out;
}


// One decomposition of the omega-twisted group algebra (decompose_projective_irreps): the right
// regular action R(h) e_x = omega(x, h) e_{xh} commutes with the left one L(g) e_x = omega(g, x)
// e_{gx} (associativity of the twisted algebra), R(h)^dagger is a phase times R(h^-1), so
// M = M0 + M0^dagger with M0 = sum_h c_h R(h) is a generic Hermitian element of the right algebra;
// its eigenspaces are single irreducible left modules, and D(g) = V^dagger L(g) V on each.
[[nodiscard]] bool try_decompose_twisted(const std::vector<std::vector<int>>& mult,
                                         const std::vector<std::vector<Complex>>& omega,
                                         std::uint64_t seed, std::vector<IrrepData>& out) {
    const int n = static_cast<int>(mult.size());
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXcd M0 = Eigen::MatrixXcd::Zero(n, n);
    for (int h = 0; h < n; ++h) {
        const Complex c(nd(gen), nd(gen));
        for (int x = 0; x < n; ++x)
            M0(mult[static_cast<std::size_t>(x)][static_cast<std::size_t>(h)], x) +=
                c * omega[static_cast<std::size_t>(x)][static_cast<std::size_t>(h)];
    }
    const Eigen::MatrixXcd M = M0 + M0.adjoint();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(M);
    if (es.info() != Eigen::Success) return false;
    const Eigen::VectorXd evals = es.eigenvalues();
    const Eigen::MatrixXcd evecs = es.eigenvectors();
    const double span = std::max(1.0, evals(n - 1) - evals(0));
    const double tol  = 1e-6 * span;
    std::vector<IrrepData> irreps;
    long long sum_d2 = 0;
    for (int b = 0, i = 1; i <= n; ++i) {
        if (i < n && evals(i) - evals(i - 1) <= tol) continue;
        const int d = i - b;
        const Eigen::MatrixXcd V = evecs.block(0, b, n, d);
        b = i;
        IrrepData ir;
        ir.dim = d;
        ir.matrices.resize(static_cast<std::size_t>(n));
        ir.character.assign(static_cast<std::size_t>(n), Complex(0.0, 0.0));
        Eigen::MatrixXcd Lv(n, d);
        for (int g = 0; g < n; ++g) {
            for (int x = 0; x < n; ++x)
                Lv.row(mult[static_cast<std::size_t>(g)][static_cast<std::size_t>(x)]) =
                    omega[static_cast<std::size_t>(g)][static_cast<std::size_t>(x)] * V.row(x);
            const Eigen::MatrixXcd D = V.adjoint() * Lv;
            auto& flat = ir.matrices[static_cast<std::size_t>(g)];
            flat.resize(static_cast<std::size_t>(d) * d);
            for (int r = 0; r < d; ++r)
                for (int c = 0; c < d; ++c) flat[static_cast<std::size_t>(r) * d + c] = D(r, c);
            ir.character[static_cast<std::size_t>(g)] = D.trace();
        }
        bool seen = false;
        for (const auto& other : irreps) {
            if (other.dim != d) continue;
            bool same = true;
            for (int g = 0; g < n && same; ++g)
                same = std::abs(other.character[static_cast<std::size_t>(g)] - ir.character[static_cast<std::size_t>(g)]) <= 1e-5;
            if (same) { seen = true; break; }
        }
        if (seen) continue;
        sum_d2 += static_cast<long long>(d) * d;
        irreps.push_back(std::move(ir));
    }
    if (sum_d2 != static_cast<long long>(n)) return false;   // a degenerate draw: retry
    out = std::move(irreps);
    return true;
}

}  // namespace

GroupIrreps decompose_irreps(const std::vector<std::vector<int>>& max_clique,
                             int /*n_sites*/) {
    const int n = static_cast<int>(max_clique.size());
    if (n == 0) throw std::runtime_error("decompose_irreps: empty group");

    // Index every permutation for table lookups.
    std::map<std::vector<int>, int> idx;
    for (int a = 0; a < n; ++a) idx[max_clique[static_cast<std::size_t>(a)]] = a;

    auto lookup = [&](const std::vector<int>& p) -> int {
        auto it = idx.find(p);
        if (it == idx.end())
            throw std::runtime_error(
                "decompose_irreps: group is not closed under composition");
        return it->second;
    };

    // Multiplication table; the rest (inverse, classes, numerical
    // decomposition) is abstract.
    std::vector<std::vector<int>> mult(
        static_cast<std::size_t>(n), std::vector<int>(static_cast<std::size_t>(n)));
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            mult[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] =
                lookup(compose(max_clique[static_cast<std::size_t>(a)],
                               max_clique[static_cast<std::size_t>(b)]));
    return decompose_irreps_tables(mult);
}

GroupIrreps decompose_irreps_tables(const std::vector<std::vector<int>>& mult) {
    const int n = static_cast<int>(mult.size());
    if (n == 0) throw std::runtime_error("decompose_irreps_tables: empty group");

    GroupIrreps gi;
    gi.order = n;
    gi.mult  = mult;

    // Identity + inverse from the table alone: e is the unique element with
    // mult[e][b] == b for every b; inverse[a] solves mult[a][x] == e.
    int e = -1;
    for (int a = 0; a < n && e < 0; ++a) {
        bool is_e = true;
        for (int b = 0; b < n; ++b)
            if (gi.mult[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] != b) {
                is_e = false;
                break;
            }
        if (is_e) e = a;
    }
    if (e < 0)
        throw std::runtime_error(
            "decompose_irreps_tables: multiplication table has no identity");
    gi.inverse.assign(static_cast<std::size_t>(n), -1);
    for (int a = 0; a < n; ++a) {
        for (int x = 0; x < n; ++x) {
            if (gi.mult[static_cast<std::size_t>(a)][static_cast<std::size_t>(x)] == e) {
                gi.inverse[static_cast<std::size_t>(a)] = x;
                break;
            }
        }
        if (gi.inverse[static_cast<std::size_t>(a)] < 0)
            throw std::runtime_error(
                "decompose_irreps_tables: element without inverse (not a group)");
    }

    // Conjugacy classes: a ~ x·a·x^{-1}.
    gi.class_of.assign(static_cast<std::size_t>(n), -1);
    int nclasses = 0;
    for (int a = 0; a < n; ++a) {
        if (gi.class_of[static_cast<std::size_t>(a)] >= 0) continue;
        for (int x = 0; x < n; ++x) {
            const int xa  = gi.mult[static_cast<std::size_t>(x)][static_cast<std::size_t>(a)];
            const int xax = gi.mult[static_cast<std::size_t>(xa)]
                                   [static_cast<std::size_t>(gi.inverse[static_cast<std::size_t>(x)])];
            gi.class_of[static_cast<std::size_t>(xax)] = nclasses;
        }
        ++nclasses;
    }
    gi.num_classes = nclasses;

    // Abelian (every class one element): the characters in closed form.
    if (nclasses == n) {
        gi.irreps = abelian_characters(gi.mult, e);
        return gi;
    }

    // Numerical decomposition; retry a few seeds if a draw is degenerate.
    std::vector<IrrepData> irreps;
    bool ok = false;
    for (std::uint64_t seed = 1; seed <= 16 && !ok; ++seed)
        ok = try_decompose(gi.inverse, gi.mult, 0x9E3779B97F4A7C15ull * seed, irreps);
    if (!ok)
        throw std::runtime_error(
            "decompose_irreps: numerical decomposition failed (Σ d_Γ² != |G|) "
            "after 16 attempts");
    if (static_cast<int>(irreps.size()) != nclasses)
        throw std::runtime_error(
            "decompose_irreps: #irreps != #conjugacy-classes");

    // Order irreps by ascending dimension (1-D first), stable.
    std::stable_sort(irreps.begin(), irreps.end(),
                     [](const IrrepData& a, const IrrepData& b) { return a.dim < b.dim; });
    gi.irreps = std::move(irreps);
    return gi;
}


GroupIrreps decompose_projective_irreps(const std::vector<std::vector<int>>& mult,
                                        const std::vector<std::vector<Complex>>& omega) {
    const std::size_t n = mult.size();
    if (omega.size() != n)
        throw std::invalid_argument("decompose_projective_irreps: omega must be |G| x |G|");
    bool trivial = true;
    for (std::size_t a = 0; a < n; ++a) {
        if (omega[a].size() != n)
            throw std::invalid_argument("decompose_projective_irreps: omega must be |G| x |G|");
        for (std::size_t b = 0; b < n; ++b) {
            // scale-free: unit-modulus phases (group data)
            if (std::abs(std::abs(omega[a][b]) - 1.0) > 1e-8)
                throw std::invalid_argument("decompose_projective_irreps: omega is not unit-modulus");
            // scale-free: unit-modulus phases (group data)
            trivial = trivial && std::abs(omega[a][b] - Complex(1.0, 0.0)) <= 1e-12;
        }
    }
    GroupIrreps gi = decompose_irreps_tables(mult);   // the tables; and the irreps when omega = 1
    if (trivial) return gi;
    // A 2-cocycle, normalised at the identity: omega(x, y) omega(xy, z) = omega(y, z) omega(x, yz).
    int e = 0;   // the identity: e x = x for one x already
    while (gi.mult[static_cast<std::size_t>(e)][0] != 0) ++e;
    for (std::size_t x = 0; x < n; ++x) {
        // scale-free: unit-modulus phases (group data)
        if (std::abs(omega[static_cast<std::size_t>(e)][x] - 1.0) > 1e-8 || std::abs(omega[x][static_cast<std::size_t>(e)] - 1.0) > 1e-8)
            throw std::invalid_argument("decompose_projective_irreps: omega is not normalised at the identity");
        for (std::size_t y = 0; y < n; ++y)
            for (std::size_t z = 0; z < n; ++z) {
                const auto xy = static_cast<std::size_t>(mult[x][y]), yz = static_cast<std::size_t>(mult[y][z]);
                // scale-free: unit-modulus phases (group data)
                if (std::abs(omega[x][y] * omega[xy][z] - omega[y][z] * omega[x][yz]) > 1e-8)
                    throw std::invalid_argument("decompose_projective_irreps: omega is not a 2-cocycle");
            }
    }
    std::vector<IrrepData> irreps;
    bool ok = false;
    for (std::uint64_t seed = 1; seed <= 16 && !ok; ++seed)
        ok = try_decompose_twisted(mult, omega, 0x9E3779B97F4A7C15ull * seed, irreps);
    if (!ok)
        throw std::runtime_error("decompose_projective_irreps: numerical decomposition failed "
                                 "(sum d^2 != |G|) after 16 attempts");
    std::stable_sort(irreps.begin(), irreps.end(),
                     [](const IrrepData& a, const IrrepData& b) { return a.dim < b.dim; });
    gi.irreps = std::move(irreps);
    return gi;
}

}  // namespace ed::symmetry
