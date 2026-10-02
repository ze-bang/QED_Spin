// =============================================================================
// src/ops/program.cpp -- compile_program and the CPU sweep behind rep_matrix_elements().
// See include/ed/ops/program.h for the formula and conventions.
// =============================================================================
#include <ed/ops/program.h>

#include <ed/basis/rep_sector.h>
#include <ed/core/log.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::ops {

namespace {
inline int popc(std::uint64_t x) { return __builtin_popcountll(x); }

// flip -> value -> terms, packed into the program arrays
struct T { std::uint64_t sign; std::complex<double> c; std::uint32_t obs; };
using Tree = std::map<std::uint64_t, std::map<std::uint64_t, std::vector<T>>>;

void pack(const Tree& tree, MaskedProgram& P) {
    P.group_vbegin.push_back(0);
    P.vsub_tbegin.push_back(0);
    for (const auto& [F, vmap] : tree) {
        P.group_flip.push_back(F);
        int setbits = popc(vmap.begin()->first);
        for (const auto& [V, ts] : vmap)
            if (popc(V) != setbits) setbits = -1;
        P.group_setbits.push_back(setbits);
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
}
}  // namespace

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
    if ((src.n_up < 0) != (tgt.n_up < 0))
        throw std::invalid_argument("compile_program: a fixed-S^z sector paired with a full or parity sector");
    const bool fixed_sz = src.n_up >= 0;   // full / parity sectors: no S^z selection rule
    const int delta_req = fixed_sz ? tgt.n_up - src.n_up : 0;

    // weights conj(lambda(g)) / |G|, lambda = chi_bra * conj(chi_ket)
    std::vector<std::complex<double>> w(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g)
        w[static_cast<std::size_t>(g)] =
            std::conj(tgt.characters[static_cast<std::size_t>(g)]
                      * std::conj(src.characters[static_cast<std::size_t>(g)]))
            / static_cast<double>(G);

    Tree tree;
    MaskedProgram P;
    P.n_obs = static_cast<int>(ops.size());
    P.delta_set_bits = delta_req;
    P.src_characters = src.characters;
    P.tgt_characters = tgt.characters;
    P.src_n_up = src.n_up;
    P.tgt_n_up = tgt.n_up;
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
        // relative to the UNPROJECTED operator, so a selection-rule zero (a projection
        // that is pure roundoff) compiles to no terms
        double scale = 0.0;
        for (const auto& t : O.terms()) scale = std::max(scale, std::abs(t.coeff));
        for (const auto& t : Ol.terms(opt.drop * scale)) {
            if (fixed_sz && masked_delta_set_bits(t) != delta_req) continue;   // cannot connect them
            tree[t.flip_mask][t.cond_val].push_back(T{t.sign_mask, t.coeff, static_cast<std::uint32_t>(a)});
            ++P.terms_per_obs[a];
        }
    }
    pack(tree, P);
    if (P.n_groups() > 100000 || P.n_terms() > 1000000)
        ED_LOG(Info, "compile_program: large program (%zu groups, %zu terms)", P.n_groups(), P.n_terms());
    return P;
}

MaskedProgram compile_operator(const MaskedOperator& O) {
    Tree tree;
    for (const auto& t : O.terms())   // key order (flip, value, sign)
        tree[t.flip_mask][t.cond_val].push_back(T{t.sign_mask, t.coeff, 0});
    MaskedProgram P;
    P.n_obs = 1;
    P.terms_per_obs.assign(1, O.size());
    pack(tree, P);
    return P;
}

MaskedOperator program_operator(const MaskedProgram& P, int n_sites) {
    MaskedOperator O(n_sites);
    for (std::size_t g = 0; g < P.n_groups(); ++g)
        for (std::uint32_t vi = P.group_vbegin[g]; vi < P.group_vbegin[g + 1]; ++vi)
            for (std::uint32_t k = P.vsub_tbegin[vi]; k < P.vsub_tbegin[vi + 1]; ++k)
                O.add_term({P.group_flip[g], P.vsub_val[vi], P.group_flip[g], P.term_sign[k], P.term_coeff[k]});
    return O;
}

using Complex = std::complex<double>;

// true if every mask holds exactly half of its bits set (Q_h = 0 on every hexagon)
inline bool is_balanced(std::uint64_t s, const std::vector<std::uint64_t>& masks) {
    for (std::uint64_t m : masks)
        if (2 * masked_popcount(s & m) != masked_popcount(m)) return false;
    return true;
}

void check_balanced_masks(const ed::symmetry::RepSectorData& rd, const std::vector<std::uint64_t>& masks) {
    if (masks.empty()) return;
    for (std::uint64_t m : masks)
        if (masked_popcount(m) % 2)
            throw std::invalid_argument("rep_matrix_elements: balanced mask with an odd number of sites");
    const auto pol = rd.make_policy();
    std::vector<std::uint64_t> sorted = masks;
    std::sort(sorted.begin(), sorted.end());
    for (int g = 0; g < rd.group_size; ++g)
        for (std::uint64_t m : masks) {
            // image of the SITE SET: apply_perm also XORs the element's flip mask; undo it
            const std::uint64_t flip = rd.flip_masks.empty() ? 0ULL : rd.flip_masks[static_cast<std::size_t>(g)];
            const std::uint64_t site_img = pol.apply_perm(m, g) ^ flip;
            if (!std::binary_search(sorted.begin(), sorted.end(), site_img))
                throw std::invalid_argument("rep_matrix_elements: balanced masks are not permuted "
                                            "among themselves by the group (constraint would break the symmetry)");
        }
}

std::vector<Complex>
rep_matrix_elements(const ed::symmetry::RepSectorData& src,
                    const ed::symmetry::RepSectorData& tgt,
                    const MaskedProgram& prog,
                    const std::vector<RepVectorView>& kets,
                    const std::vector<RepVectorView>& bras,
                    const std::vector<std::pair<int, int>>& pairs,
                    const RepMEOptions& opt) {
    if (prog.src_characters != src.characters || prog.tgt_characters != tgt.characters
        || prog.src_n_up != src.n_up || prog.tgt_n_up != tgt.n_up)
        throw std::invalid_argument("rep_matrix_elements: program was compiled for other sectors");
    if (src.group_size != tgt.group_size || src.n_sites != tgt.n_sites)
        throw std::invalid_argument("rep_matrix_elements: the two sectors come from different groups");
    for (const auto& v : kets)
        if (v.size != src.dim() || (v.size > 0 && v.data == nullptr))
            throw std::invalid_argument("rep_matrix_elements: ket length != source sector dim");
    for (const auto& v : bras)
        if (v.size != tgt.dim() || (v.size > 0 && v.data == nullptr))
            throw std::invalid_argument("rep_matrix_elements: bra length != target sector dim");
    for (const auto& [b, k] : pairs)
        if (b < 0 || k < 0 || b >= static_cast<int>(bras.size()) || k >= static_cast<int>(kets.size()))
            throw std::invalid_argument("rep_matrix_elements: pair (" + std::to_string(b) + ", "
                                        + std::to_string(k) + ") out of range");

    const std::size_t n_obs = static_cast<std::size_t>(prog.n_obs);
    const std::size_t n_pairs = pairs.size();
    std::vector<Complex> out(n_pairs * n_obs, Complex(0.0, 0.0));
    if (n_pairs == 0 || n_obs == 0 || prog.n_terms() == 0) return out;
    check_balanced_masks(src, opt.balanced_masks);
    if (opt.use_gpu) return rep_matrix_elements_gpu(src, tgt, prog, kets, bras, pairs, opt);

    const auto tpol = tgt.make_policy();
    const bool same_sector = (&src == &tgt);
    const long long dim = static_cast<long long>(src.dim());
    const std::size_t n_groups = prog.n_groups();

    std::vector<const Complex*> ket_of(n_pairs), bra_of(n_pairs);
    for (std::size_t p = 0; p < n_pairs; ++p) {
        bra_of[p] = bras[static_cast<std::size_t>(pairs[p].first)].data;
        ket_of[p] = kets[static_cast<std::size_t>(pairs[p].second)].data;
    }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<Complex>> partial(static_cast<std::size_t>(n_threads));

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        std::vector<Complex> acc(n_pairs * n_obs, Complex(0.0, 0.0));
        std::vector<Complex> bk(n_pairs), y(n_pairs);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long long ri = 0; ri < dim; ++ri) {
            const std::size_t r = static_cast<std::size_t>(ri);
            bool any = false;
            for (std::size_t p = 0; p < n_pairs; ++p) {
                bk[p] = ket_of[p][r];
                any = any || (bk[p] != Complex(0.0, 0.0));
            }
            if (!any) continue;
            const std::uint64_t s = src.reps[r];
            if (!is_balanced(s, opt.balanced_masks)) continue;       // P on the ket
            const double w = src.inv_norms[r];
            for (std::size_t gi = 0; gi < n_groups; ++gi) {
                const std::uint64_t F = prog.group_flip[gi];
                const std::uint64_t v = s & F;
                if (prog.group_setbits[gi] >= 0 && masked_popcount(v) != prog.group_setbits[gi]) continue;
                const auto vb = prog.vsub_val.begin() + prog.group_vbegin[gi];
                const auto ve = prog.vsub_val.begin() + prog.group_vbegin[gi + 1];
                const auto it = std::lower_bound(vb, ve, v);
                if (it == ve || *it != v) continue;
                const std::size_t vi = static_cast<std::size_t>(it - prog.vsub_val.begin());

                Complex proj;
                std::int64_t j;
                if (same_sector && F == 0) {        // diagonal term inside one sector
                    j = static_cast<std::int64_t>(r);
                    proj = Complex(1.0 / w, 0.0);
                } else {
                    if (!is_balanced(s ^ F, opt.balanced_masks)) continue;   // P on the bra
                    j = tpol.index_and_projection(s ^ F, proj);
                    if (j < 0) continue;            // target orbit cancelled in this sector
                }
                const Complex base = proj * w;
                const std::size_t jj = static_cast<std::size_t>(j);
                for (std::size_t p = 0; p < n_pairs; ++p)
                    y[p] = base * bk[p] * std::conj(bra_of[p][jj]);
                for (std::uint32_t k = prog.vsub_tbegin[vi]; k < prog.vsub_tbegin[vi + 1]; ++k) {
                    Complex c = prog.term_coeff[k];
                    if (masked_popcount(s & prog.term_sign[k]) & 1) c = -c;
                    const std::size_t o = prog.term_obs[k];
                    for (std::size_t p = 0; p < n_pairs; ++p) acc[p * n_obs + o] += c * y[p];
                }
            }
        }
        partial[static_cast<std::size_t>(tid)] = std::move(acc);
    }
    for (const auto& a : partial) {
        if (a.empty()) continue;
        for (std::size_t i = 0; i < out.size(); ++i) out[i] += a[i];
    }
    return out;
}

}  // namespace ed::ops


#ifndef WITH_CUDA
namespace ed::ops {
std::vector<Complex>
rep_matrix_elements_gpu(const ed::symmetry::RepSectorData&, const ed::symmetry::RepSectorData&,
                        const MaskedProgram&, const std::vector<RepVectorView>&,
                        const std::vector<RepVectorView>&, const std::vector<std::pair<int, int>>&,
                        const RepMEOptions&) {
    throw std::runtime_error("rep_matrix_elements: use_gpu requires a WITH_CUDA build");
}
bool rep_matrix_elements_gpu_available() { return false; }
}  // namespace ed::ops
#endif
