// =============================================================================
// src/ops/algebra.cpp -- MaskedOperator: canonical terms, products, adjoints, group images.
// =============================================================================
#include <ed/ops/algebra.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ed::ops {

namespace {

inline int popc(std::uint64_t x) { return __builtin_popcountll(x); }
inline std::uint64_t lowest_bit(std::uint64_t x) { return x & (~x + 1); }

}  // namespace

MaskedOperator::MaskedOperator(int n_sites) : n_(n_sites) {
    if (n_sites < 1 || n_sites > 64)
        throw std::invalid_argument("MaskedOperator: n_sites must be in [1, 64]");
}

void MaskedOperator::add_canonical(std::uint64_t flip, std::uint64_t val,
                                   std::uint64_t sign, Complex c) {
    accumulate(Key{flip, val, sign}, c);
}

void MaskedOperator::accumulate(const Key& k, Complex c) {
    if (c == Complex(0.0, 0.0)) return;
    const auto it = t_.try_emplace(k, 0.0).first;
    it->second += c;
    if (it->second == Complex(0.0, 0.0)) t_.erase(it);   // exact cancellation: no term
}

void MaskedOperator::add_term(const MaskedTerm& in) {
    // Iterative expansion to canonical form (cond_mask == flip_mask, sign disjoint).
    struct Item { std::uint64_t C, V, F, S; Complex c; };
    std::vector<Item> stack{{in.cond_mask, in.cond_val & in.cond_mask, in.flip_mask,
                             in.sign_mask, in.coeff}};
    const std::uint64_t all = (n_ == 64) ? ~0ULL : ((1ULL << n_) - 1ULL);
    if (((in.cond_mask | in.flip_mask | in.sign_mask) & ~all) != 0)
        throw std::invalid_argument("MaskedOperator::add_term: mask beyond n_sites");
    while (!stack.empty()) {
        Item it = stack.back();
        stack.pop_back();
        it.V &= it.C;
        // Z on a conditioned site is a known sign.
        const std::uint64_t known = it.S & it.C;
        if (known) {
            if (popc(known & it.V) & 1) it.c = -it.c;
            it.S &= ~it.C;
        }
        if (it.c == Complex(0.0, 0.0)) continue;
        if (const std::uint64_t free_flip = it.F & ~it.C) {
            // an unconditioned flip is S+ + S- on that site: split on the bit's value
            const std::uint64_t b = lowest_bit(free_flip);
            stack.push_back({it.C | b, it.V, it.F, it.S, it.c});
            stack.push_back({it.C | b, it.V | b, it.F, it.S, it.c});
            continue;
        }
        if (const std::uint64_t proj = it.C & ~it.F) {
            // a condition on an unflipped site is a projector (1 +- Z)/2
            const std::uint64_t b = lowest_bit(proj);
            const double s = (it.V & b) ? -1.0 : 1.0;   // bit set: (1 - Z)/2
            stack.push_back({it.C & ~b, it.V & ~b, it.F, it.S, 0.5 * it.c});
            stack.push_back({it.C & ~b, it.V & ~b, it.F, it.S | b, 0.5 * s * it.c});
            continue;
        }
        add_canonical(it.F, it.V, it.S, it.c);
    }
}

MaskedOperator MaskedOperator::product(int n_sites, const std::string& ops,
                                       const std::vector<int>& sites, Complex coeff) {
    if (ops.size() != sites.size())
        throw std::invalid_argument("MaskedOperator::product: ops and sites differ in length");
    MaskedOperator r(n_sites);
    r.add_canonical(0, 0, 0, coeff);
    for (std::size_t k = 0; k < ops.size(); ++k) {
        const int i = sites[k];
        if (i < 0 || i >= n_sites)
            throw std::invalid_argument("MaskedOperator::product: site out of range");
        const std::uint64_t b = 1ULL << i;
        const std::uint64_t up = up_bits(b), dn = down_bits(b);
        // Z = (-1)^bit is +1 on a clear bit: S^z = Z / 2 when a clear bit is up.
        const double zsign = kSetBitIsDown ? 1.0 : -1.0;
        MaskedOperator f(n_sites);
        switch (ops[k]) {
            case '+': f.add_term({b, dn, b, 0, 1.0}); break;          // S+: down -> up
            case '-': f.add_term({b, up, b, 0, 1.0}); break;          // S-: up -> down
            case 'z': f.add_term({0, 0, 0, b, zsign * 0.5}); break;   // S^z = +-Z / 2
            case 'x': f.add_term({b, dn, b, 0, 0.5}); f.add_term({b, up, b, 0, 0.5}); break;
            case 'y': f.add_term({b, dn, b, 0, Complex(0.0, -0.5)});  // (S+ - S-) / 2i
                      f.add_term({b, up, b, 0, Complex(0.0, 0.5)}); break;
            case 'u': f.add_term({b, up, 0, 0, 1.0}); break;          // |up><up|
            case 'd': f.add_term({b, dn, 0, 0, 1.0}); break;          // |dn><dn|
            case 'I': f.add_canonical(0, 0, 0, 1.0); break;
            default:
                throw std::invalid_argument(std::string("MaskedOperator::product: unknown op '")
                                            + ops[k] + "'");
        }
        r = r * f;
    }
    return r;
}

MaskedOperator& MaskedOperator::add(const MaskedOperator& o, Complex scale) {
    if (o.n_ != n_) throw std::invalid_argument("MaskedOperator::add: n_sites differ");
    if (&o == this) { *this = o.scaled(1.0 + scale); return *this; }
    for (const auto& [k, c] : o.t_) accumulate(k, scale * c);
    return *this;
}

MaskedOperator MaskedOperator::operator+(const MaskedOperator& o) const {
    MaskedOperator r = *this;
    r.add(o);
    return r;
}

MaskedOperator MaskedOperator::operator-(const MaskedOperator& o) const {
    MaskedOperator r = *this;
    r.add(o, -1.0);
    return r;
}

MaskedOperator MaskedOperator::operator-() const { return scaled(-1.0); }

MaskedOperator commutator(const MaskedOperator& a, const MaskedOperator& b) { return a * b - b * a; }

MaskedOperator MaskedOperator::scaled(Complex s) const {
    MaskedOperator r(n_);
    for (const auto& [k, c] : t_) r.accumulate(k, s * c);
    return r;
}

MaskedOperator MaskedOperator::operator*(const MaskedOperator& o) const {
    if (o.n_ != n_) throw std::invalid_argument("MaskedOperator::operator*: n_sites differ");
    MaskedOperator r(n_);
    for (const auto& [ka, ca] : t_) {
        const auto [FA, VA, SA] = ka;              // A acts second, C_A = F_A
        for (const auto& [kb, cb] : o.t_) {
            const auto [FB, VB, SB] = kb;          // B acts first,  C_B = F_B
            const std::uint64_t VA_in = VA ^ (FB & FA);    // A's condition seen on the input
            const std::uint64_t overlap = FA & FB;
            if (((VB ^ VA_in) & overlap) != 0) continue;    // contradictory conditions: zero
            Complex c = ca * cb;
            if (popc(FB & SA) & 1) c = -c;         // A's Z evaluated after B's flip
            r.add_term({FA | FB, VB | VA_in, FA ^ FB, SA ^ SB, c});
        }
    }
    return r;
}

MaskedOperator MaskedOperator::dagger() const {
    MaskedOperator r(n_);
    for (const auto& [k, c] : t_) {
        const auto [F, V, S] = k;
        r.add_canonical(F, V ^ F, S, std::conj(c));  // S+ <-> S-, Z commutes (disjoint sites)
    }
    return r;
}

MaskedOperator MaskedOperator::image(const int* perm, std::uint64_t m) const {
    MaskedOperator r(n_);
    for (const auto& [k, c] : t_) {
        const auto [F, V, S] = k;
        const std::uint64_t Fp = permute_mask(F, perm, n_);
        const std::uint64_t Vp = permute_mask(V, perm, n_) ^ (m & Fp);
        const std::uint64_t Sp = permute_mask(S, perm, n_);
        r.add_canonical(Fp, Vp, Sp, (popc(m & Sp) & 1) ? -c : c);
    }
    return r;
}

MaskedOperator MaskedOperator::image(Map g) const {
    // Per term T = c (-1)^{s.S} |s^F><s| on (s & F) == V:
    //   F:  the condition sees the flipped bits, V ^= F, and (-1)^{s.S} gains (-1)^{|S|};
    //   Dz: (-1)^{popc(s) + popc(s ^ F)} = (-1)^{|F|} (the global sign of either convention cancels);
    //   K:  c -> c*;   Theta = (prod i sigma^y) K = Dz F K: all three.
    MaskedOperator r(n_);
    for (const auto& [k, c] : t_) {
        const auto [F, V, S] = k;
        const bool flip = g == Map::F || g == Map::Theta;
        int parity = 0;
        if (flip) parity += popc(S);
        if (g == Map::Dz || g == Map::Theta) parity += popc(F);
        Complex v = (g == Map::K || g == Map::Theta) ? std::conj(c) : c;
        if (parity & 1) v = -v;
        r.add_canonical(F, flip ? (V ^ F) : V, S, v);
    }
    return r;
}

double MaskedOperator::max_abs() const {
    double m = 0.0;
    for (const auto& [k, c] : t_) m = std::max(m, std::abs(c));
    return m;
}

bool MaskedOperator::equals(const MaskedOperator& o, double rtol) const {
    if (o.n_ != n_) return false;
    const double scale = std::max(max_abs(), o.max_abs());
    for (const auto& [k, c] : t_) {
        const auto it = o.t_.find(k);
        if (std::abs(c - (it == o.t_.end() ? Complex(0.0, 0.0) : it->second)) > rtol * scale) return false;
    }
    for (const auto& [k, c] : o.t_)
        if (t_.find(k) == t_.end() && std::abs(c) > rtol * scale) return false;
    return true;
}

bool MaskedOperator::is_hermitian(double tol) const { return equals(dagger(), tol); }

int MaskedOperator::delta_set_bits() const {
    bool first = true;
    int delta = 0;
    for (const auto& [k, c] : t_) {
        if (c == Complex(0.0, 0.0)) continue;
        const auto [F, V, S] = k;
        const int d = popc(F & ~V) - popc(F & V);   // clear -> set adds a set bit
        if (first) { delta = d; first = false; }
        else if (d != delta)
            throw std::invalid_argument(
                "MaskedOperator::delta_set_bits: terms change S^z by different amounts");
    }
    return delta;
}

std::vector<MaskedTerm> MaskedOperator::terms(double drop) const {
    std::vector<MaskedTerm> out;
    out.reserve(t_.size());
    for (const auto& [k, c] : t_) {
        if (!(std::abs(c) > drop)) continue;
        const auto [F, V, S] = k;
        out.push_back({F, V, F, S, c});
    }
    return out;
}

std::vector<MaskedOperator::Complex> MaskedOperator::to_dense() const {
    if (n_ > 12) throw std::invalid_argument("MaskedOperator::to_dense: n_sites > 12");
    const std::uint64_t dim = 1ULL << n_;
    std::vector<Complex> M(dim * dim, Complex(0.0, 0.0));
    for (const auto& t : terms())
        for (std::uint64_t s = 0; s < dim; ++s) {
            std::uint64_t tt;
            double sg;
            if (masked_apply(t, s, tt, sg)) M[tt * dim + s] += sg * t.coeff;
        }
    return M;
}

}  // namespace ed::ops
