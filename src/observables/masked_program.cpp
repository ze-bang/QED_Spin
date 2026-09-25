// =============================================================================
// src/observables/masked_program.cpp -- MaskedOperator algebra and compile_program().
// See include/ed/observables/masked_program.h for the conventions.
// =============================================================================
#include <ed/observables/masked_program.h>

#include <ed/symmetry/rep_sector_data.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ed::observables {

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
    if (c == Complex(0.0, 0.0)) return;
    t_[Key{flip, val, sign}] += c;
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
            const double s = (it.V & b) ? -1.0 : 1.0;   // bit set (down): (1 - Z)/2
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
        MaskedOperator f(n_sites);
        switch (ops[k]) {
            case '+': f.add_term({b, b, b, 0, 1.0}); break;           // S+: down -> up
            case '-': f.add_term({b, 0, b, 0, 1.0}); break;           // S-: up -> down
            case 'z': f.add_term({0, 0, 0, b, 0.5}); break;           // S^z = Z / 2
            case 'x': f.add_term({b, b, b, 0, 0.5}); f.add_term({b, 0, b, 0, 0.5}); break;
            case 'y': f.add_term({b, b, b, 0, Complex(0.0, -0.5)});   // (S+ - S-) / 2i
                      f.add_term({b, 0, b, 0, Complex(0.0, 0.5)}); break;
            case 'u': f.add_term({b, 0, 0, 0, 1.0}); break;           // |up><up|
            case 'd': f.add_term({b, b, 0, 0, 1.0}); break;           // |dn><dn|
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
    for (const auto& [k, c] : o.t_) t_[k] += scale * c;
    return *this;
}

MaskedOperator MaskedOperator::operator+(const MaskedOperator& o) const {
    MaskedOperator r = *this;
    r.add(o);
    return r;
}

MaskedOperator MaskedOperator::scaled(Complex s) const {
    MaskedOperator r(n_);
    for (const auto& [k, c] : t_) r.t_[k] = s * c;
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

bool MaskedOperator::is_hermitian(double tol) const {
    const MaskedOperator d = dagger();
    double scale = 1.0;
    for (const auto& [k, c] : t_) scale = std::max(scale, std::abs(c));
    for (const auto& [k, c] : t_) {
        const auto it = d.t_.find(k);
        if (std::abs(c - (it == d.t_.end() ? Complex(0.0, 0.0) : it->second)) > tol * scale)
            return false;
    }
    for (const auto& [k, c] : d.t_)
        if (t_.find(k) == t_.end() && std::abs(c) > tol * scale) return false;
    return true;
}

int MaskedOperator::delta_set_bits() const {
    bool first = true;
    int delta = 0;
    for (const auto& [k, c] : t_) {
        if (c == Complex(0.0, 0.0)) continue;
        const auto [F, V, S] = k;
        const int d = popc(F & ~V) - popc(F & V);   // clear->set adds a down spin
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

// -----------------------------------------------------------------------------
// compile_program
// -----------------------------------------------------------------------------
MaskedProgram compile_program(const std::vector<MaskedOperator>& ops,
                              const ed::symmetry::RepSectorData& src,
                              const ed::symmetry::RepSectorData& tgt,
                              const CompileOptions& opt) {
    const int G = src.group_size;
    const int n = src.n_sites;
    if (tgt.group_size != G || tgt.n_sites != n || tgt.perms_flat != src.perms_flat)
        throw std::invalid_argument("compile_program: the two sectors come from different groups");
    const auto flip_of = [](const ed::symmetry::RepSectorData& rd, int g) -> std::uint64_t {
        return rd.flip_masks.empty() ? 0ULL : rd.flip_masks[static_cast<std::size_t>(g)];
    };
    for (int g = 0; g < G; ++g)
        if (flip_of(src, g) != flip_of(tgt, g))
            throw std::invalid_argument("compile_program: the two sectors differ in flip masks");
    if (static_cast<int>(src.characters.size()) != G || static_cast<int>(tgt.characters.size()) != G)
        throw std::invalid_argument("compile_program: character tables have the wrong length");
    if (src.n_up < 0 || tgt.n_up < 0)
        throw std::invalid_argument("compile_program: fixed-S^z sectors required");
    const int delta_req = tgt.n_up - src.n_up;

    // weights conj(lambda(g)) / |G|, lambda = chi_bra * conj(chi_ket)
    std::vector<std::complex<double>> w(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        w[static_cast<std::size_t>(g)] =
            std::conj(tgt.characters[static_cast<std::size_t>(g)]
                      * std::conj(src.characters[static_cast<std::size_t>(g)]))
            / static_cast<double>(G);

    // flip -> value -> terms
    struct T { std::uint64_t sign; std::complex<double> c; std::uint32_t obs; };
    std::map<std::uint64_t, std::map<std::uint64_t, std::vector<T>>> tree;
    MaskedProgram P;
    P.n_obs = static_cast<int>(ops.size());
    P.delta_set_bits = delta_req;
    P.terms_per_obs.assign(ops.size(), 0);
    for (std::size_t a = 0; a < ops.size(); ++a) {
        const MaskedOperator& O = ops[a];
        if (O.n_sites() != n)
            throw std::invalid_argument("compile_program: observable " + std::to_string(a)
                                        + " acts on a different number of sites");
        MaskedOperator Ol(n);
        if (opt.project) {
            for (int g = 0; g < G; ++g) {
                const auto wg = w[static_cast<std::size_t>(g)];
                if (std::abs(wg) < 1e-15) continue;
                Ol.add(O.image(src.perms_flat.data() + static_cast<std::size_t>(g) * n,
                               flip_of(src, g)), wg);
            }
        } else {
            Ol = O;
        }
        double scale = 0.0;
        for (const auto& t : Ol.terms()) scale = std::max(scale, std::abs(t.coeff));
        for (const auto& t : Ol.terms(opt.drop * scale)) {
            if (masked_delta_set_bits(t) != delta_req) continue;   // cannot connect the sectors
            tree[t.flip_mask][t.cond_val].push_back(T{t.sign_mask, t.coeff, static_cast<std::uint32_t>(a)});
            ++P.terms_per_obs[a];
        }
    }
    P.group_vbegin.push_back(0);
    P.vsub_tbegin.push_back(0);
    for (const auto& [F, vmap] : tree) {
        P.group_flip.push_back(F);
        P.group_setbits.push_back(popc(vmap.begin()->first));
        for (const auto& [V, ts] : vmap) {
            P.vsub_val.push_back(V);
            for (const auto& t : ts) {
                P.term_sign.push_back(t.sign);
                P.term_coeff.push_back(t.c);
                P.term_obs.push_back(t.obs);
            }
            P.vsub_tbegin.push_back(static_cast<std::uint32_t>(P.term_sign.size()));
        }
        P.group_vbegin.push_back(static_cast<std::uint32_t>(P.vsub_val.size()));
    }
    if (P.n_groups() > 100000 || P.n_terms() > 1000000)
        std::fprintf(stderr, "compile_program: large program (%zu groups, %zu terms)\n",
                     P.n_groups(), P.n_terms());
    return P;
}

}  // namespace ed::observables
