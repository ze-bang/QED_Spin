// =============================================================================
// src/ops/invariance.cpp -- the records-to-terms adapter and the symmetry verdicts.
// =============================================================================
#include <ed/ops/invariance.h>
#include <ed/core/errors.h>
#include <ed/ops/operator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

namespace ed::ops {

namespace {

char op_char(std::uint8_t t) {
    switch (t) {
        case 0: return '+';
        case 1: return '-';
        case 2: return 'z';
        default:
            throw std::invalid_argument("Operator: op_type " + std::to_string(t)
                                        + " is not 0 (S+), 1 (S-) or 2 (Sz)");
    }
}

int site_of(std::uint64_t s, int n) {
    if (s >= static_cast<std::uint64_t>(n))
        throw std::invalid_argument("Operator: site " + std::to_string(s) + " is outside [0, "
                                    + std::to_string(n) + ")");
    return static_cast<int>(s);
}

MaskedOperator total(int n, char a) {   // S^a_tot
    MaskedOperator S(n);
    for (int i = 0; i < n; ++i) S.add(MaskedOperator::product(n, std::string(1, a), {i}));
    return S;
}

}  // namespace

MaskedOperator masked(const ::Operator& op) {
    const int n = static_cast<int>(op.getNumBits());
    MaskedOperator m(n);
    for (const auto& t : op.records()) {
        if (t.is_two_body)
            m.add(MaskedOperator::product(n, {op_char(t.op_type), op_char(t.op_type_2)},
                                          {site_of(t.site_index, n), site_of(t.site_index_2, n)},
                                          t.coefficient));
        else
            m.add(MaskedOperator::product(n, std::string(1, op_char(t.op_type)),
                                          {site_of(t.site_index, n)}, t.coefficient));
    }
    for (const auto& t : op.three_body_records())   // O1 acts first: the last factor of a product
        m.add(MaskedOperator::product(n, {op_char(t.op_type_3), op_char(t.op_type_2), op_char(t.op_type_1)},
                                      {site_of(t.site_index_3, n), site_of(t.site_index_2, n),
                                       site_of(t.site_index_1, n)},
                                      t.coefficient));
    for (const auto& t : op.extra_terms()) m.add_term(t);
    return m;
}

std::vector<ProductTerm> product_terms(const MaskedOperator& m) {
    const int n = m.n_sites();
    const double zf = kSetBitIsDown ? 2.0 : -2.0;   // Z = (-1)^bit = zf S^z
    std::vector<ProductTerm> out;
    for (const auto& t : m.terms()) {
        ProductTerm p{t.coeff, {}, {}};
        for (int i = 0; i < n; ++i) {
            const std::uint64_t b = 1ULL << i;
            if (t.flip_mask & b) {          // S+ lifts a down spin, S- lowers an up one
                p.ops.push_back((t.cond_val & b) == down_bits(b) ? '+' : '-');
            } else if (t.sign_mask & b) {
                p.ops.push_back('z');
                p.coeff *= zf;
            } else {
                continue;
            }
            p.sites.push_back(i);
        }
        out.push_back(std::move(p));
    }
    return out;
}

::Operator to_operator(const MaskedOperator& m) {
    ::Operator op(static_cast<std::uint64_t>(m.n_sites()), 0.5f);
    auto code = [](char c) -> std::uint8_t { return c == '+' ? 0 : (c == '-' ? 1 : 2); };
    const auto terms = m.terms();
    const auto products = product_terms(m);   // the same terms, in the same (key) order
    for (std::size_t i = 0; i < products.size(); ++i) {
        const auto& p = products[i];
        const auto& o = p.ops;
        auto s = [&p](std::size_t k) { return static_cast<std::uint64_t>(p.sites[k]); };
        switch (o.size()) {
            case 0: op.addTwoBodyTerm(2, 0, 2, 0, 4.0 * p.coeff); break;   // Sz_0 Sz_0 = 1/4
            case 1: op.addOneBodyTerm(code(o[0]), s(0), p.coeff); break;
            case 2: op.addTwoBodyTerm(code(o[0]), s(0), code(o[1]), s(1), p.coeff); break;
            case 3: op.addThreeBodyTerm(code(o[0]), s(0), code(o[1]), s(1), code(o[2]), s(2), p.coeff); break;
            default: op.add_extra_term(terms[i]); break;   // four or more sites: no record holds it
        }
    }
    return op;
}

void require_permutation(const std::vector<int>& perm, int n) {
    if (static_cast<int>(perm.size()) != n)
        throw std::invalid_argument("permutation: " + std::to_string(perm.size()) + " entries for "
                                    + std::to_string(n) + " sites");
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    for (int p : perm) {
        if (p < 0 || p >= n || seen[static_cast<std::size_t>(p)])
            throw std::invalid_argument("permutation: not a permutation of 0.." + std::to_string(n - 1));
        seen[static_cast<std::size_t>(p)] = 1;
    }
}

MaskedOperator keep_sz_changes(const MaskedOperator& O, SzKeep keep) {
    if (keep == SzKeep::All) return O;
    MaskedOperator out(O.n_sites());
    for (const auto& t : O.terms()) {
        const int d = masked_delta_set_bits(t);
        if (keep == SzKeep::Zero ? d != 0 : (d % 2 != 0)) continue;
        out.add_term(t);
    }
    return out;
}

MaskedOperator group_average(const MaskedOperator& O, const std::vector<std::vector<int>>& G, bool flip) {
    const int n = O.n_sites();
    const std::uint64_t all = (n == 64) ? ~0ULL : ((1ULL << n) - 1ULL);
    const double w = 1.0 / static_cast<double>(G.size() * (flip ? 2 : 1));
    MaskedOperator out(n);
    for (const auto& g : G) {
        require_permutation(g, n);
        out.add(O.image(g.data(), 0), w);
        if (flip) out.add(O.image(g.data(), all), w);
    }
    return out;
}

bool commutes_with_permutation(const MaskedOperator& H, const std::vector<int>& perm, double rtol) {
    require_permutation(perm, H.n_sites());
    return invariant(H, H.image(perm.data(), 0), rtol);
}

SzContent sz_content(const MaskedOperator& H, double rtol) {
    bool u1 = true, parity = true;
    for (const auto& t : H.terms(rtol * H.max_abs())) {
        const int d = masked_delta_set_bits(t);   // the change of either spin count, up to sign
        if (d != 0) u1 = false;
        if (d % 2 != 0) parity = false;
    }
    return u1 ? SzContent::U1 : (parity ? SzContent::Parity : SzContent::None);
}

bool su2_invariant(const MaskedOperator& H, double rtol) {
    const double cut = rtol * H.max_abs();
    for (char a : {'+', '-', 'z'})
        if (commutator(H, total(H.n_sites(), a)).max_abs() > cut) return false;
    return true;
}

}  // namespace ed::ops

namespace ed::ops {

namespace {

using Rot = std::array<double, 9>;   // row-major 3 x 3

Rot rotation(std::array<double, 3> a, double angle) {   // Rodrigues, about the unit axis a
    const double n = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    for (double& x : a) x /= n;
    const double c = std::cos(angle), s = std::sin(angle), t = 1.0 - c;
    return {t * a[0] * a[0] + c,        t * a[0] * a[1] - s * a[2], t * a[0] * a[2] + s * a[1],
            t * a[0] * a[1] + s * a[2], t * a[1] * a[1] + c,        t * a[1] * a[2] - s * a[0],
            t * a[0] * a[2] - s * a[1], t * a[1] * a[2] + s * a[0], t * a[2] * a[2] + c};
}

Rot multiply(const Rot& x, const Rot& y) {
    Rot z{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) z[3 * i + j] += x[3 * i + k] * y[3 * k + j];
    return z;
}

// The 60 rotations of the icosahedron with vertices (0, +-1, +-phi) and their cyclic
// permutations, closed from the five-fold rotation about a vertex and the three-fold rotation
// about a face centre.
const std::vector<Rot>& icosahedral_rotations() {
    static const std::vector<Rot> group = [] {
        const double phi = (1.0 + std::sqrt(5.0)) / 2.0, pi = 3.14159265358979323846;
        const std::vector<Rot> gens = {rotation({0.0, 1.0, phi}, 2.0 * pi / 5.0),
                                       rotation({1.0, 1.0, 1.0}, 2.0 * pi / 3.0)};
        std::vector<Rot> g = {{1, 0, 0, 0, 1, 0, 0, 0, 1}};
        auto known = [&g](const Rot& r) {
            for (const Rot& h : g) {
                double d = 0.0;
                for (int i = 0; i < 9; ++i) d = std::max(d, std::abs(h[i] - r[i]));
                // scale-free: entries of rotation matrices
                if (d < 1e-9) return true;
            }
            return false;
        };
        for (std::size_t head = 0; head < g.size(); ++head)
            for (const Rot& x : gens) {
                const Rot r = multiply(x, g[head]);
                if (!known(r)) g.push_back(r);
            }
        if (g.size() != 60) throw std::logic_error("icosahedral_rotations: closed to " + std::to_string(g.size()));
        return g;
    }();
    return group;
}

}  // namespace

MaskedOperator su2_scalar_part(const MaskedOperator& O) {
    using C = std::complex<double>;
    if (su2_invariant(O)) return O;
    const auto& G = icosahedral_rotations();
    const C I(0.0, 1.0);
    MaskedOperator out(O.n_sites());
    for (const ProductTerm& t : product_terms(O)) {
        const std::size_t n = t.ops.size();
        if (n == 0) { out.add(MaskedOperator::product(O.n_sites(), "", {}, t.coeff)); continue; }
        if (n > 5)
            throw ed::Unsupported("su2_scalar_part: a term on " + std::to_string(n) + " sites; the rotation "
                                  "average is exact on at most 5");
        // A factor S+ = S^x + i S^y, S- = S^x - i S^y, S^z as its Cartesian vector v (op = v.S).
        std::vector<std::array<C, 3>> v(n);
        for (std::size_t k = 0; k < n; ++k)
            v[k] = t.ops[k] == '+' ? std::array<C, 3>{1.0, I, 0.0}
                 : t.ops[k] == '-' ? std::array<C, 3>{1.0, -I, 0.0} : std::array<C, 3>{0.0, 0.0, 1.0};
        std::size_t combos = 1;
        for (std::size_t k = 0; k < n; ++k) combos *= 3;
        std::string ops(n, ' ');
        for (const Rot& R : G) {
            // U_R (v.S) U_R^dagger = (R v).S, written back over S+, S-, S^z:
            // w.S = (w_x - i w_y)/2 S+ + (w_x + i w_y)/2 S- + w_z S^z.
            std::vector<std::array<C, 3>> f(n);
            for (std::size_t k = 0; k < n; ++k) {
                C w[3];
                for (int a = 0; a < 3; ++a) w[a] = R[3 * a] * v[k][0] + R[3 * a + 1] * v[k][1] + R[3 * a + 2] * v[k][2];
                f[k] = {0.5 * (w[0] - I * w[1]), 0.5 * (w[0] + I * w[1]), w[2]};
            }
            for (std::size_t idx = 0; idx < combos; ++idx) {
                C c = t.coeff / static_cast<double>(G.size());
                std::size_t r = idx;
                for (std::size_t k = 0; k < n; ++k, r /= 3) {
                    ops[k] = "+-z"[r % 3];
                    c *= f[k][r % 3];
                }
                if (c != C(0.0, 0.0)) out.add(MaskedOperator::product(O.n_sites(), ops, t.sites, c));
            }
        }
    }
    return out;
}

}  // namespace ed::ops
