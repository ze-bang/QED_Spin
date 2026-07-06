// =============================================================================
// src/symmetry/irreps.cpp  --  numerical irrep decomposition (see irreps.h).
// =============================================================================

#include <ed/symmetry/irreps.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <complex>
#include <map>
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
// (mult, inverse) tables -- the group can be abstract (Stage 7 little
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

}  // namespace ed::symmetry
