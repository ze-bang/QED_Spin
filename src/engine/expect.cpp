// =============================================================================
// src/engine/expect.cpp -- expectation values and matrix elements
// over the sector-resolved eigenpairs (include/ed/sectors/expect.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "validate.h"
#include "walk.h"

#include <ed/sectors/expect.h>

#include <array>
#include <map>
#include <set>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;


std::vector<std::vector<Complex>> expect(const EigsResult& r, const Spec& s,
                                         const std::vector<const ::Operator*>& ops) {
    const int n_sites = r.n_sites;
    detail::validate_environment("expect");
    for (std::size_t i = 0; i < ops.size(); ++i) detail::validate_observable(ops[i], n_sites, "expect", i);
    // Every level's vector lives in its own sector basis; the operators averaged over the
    // symmetry group (and the flip where the level folds or projects by it) are invariant, so
    // <v|Obar|v> is one rep_matrix_elements sweep per (basis, flip, keep) over all operators --
    // and their conjugates, for a time-reversal-folded level: <K v|A|K v> = conj(<v|A^K|v>) --
    // and all the levels sharing them. No matvec, no CSR, a fixed order.
    detail::Averager avg(s, n_sites, s.two_S >= 0 && !r.levels.empty() && detail::members(r.levels.front()) > 1);
    const std::size_t n_ops = ops.size();
    std::vector<std::vector<Complex>> out(r.levels.size(), std::vector<Complex>(n_ops));
    // A level paired by an antiunitary map A (a time-reversal-folded star, a Theta mirror) also
    // averages <A v|O|A v> = conj(<v|A^-1 O A|v>); one basis has one such map.
    struct Group {
        std::vector<std::size_t> levels;
        Antiunitary image = Antiunitary::None;
    };
    std::map<std::tuple<const void*, bool, int>, Group> groups;
    using detail::Keep;
    for (std::size_t li = 0; li < r.levels.size(); ++li) {
        const Level& L = r.levels[li];
        if (L.vector < 0) throw std::invalid_argument("expect: a level has no vector (solve with vectors)");
        const BlockVector& v = r.vectors[static_cast<std::size_t>(L.vector)];
        const bool flip = (L.mirror == 2 && !detail::theta_mirror(L)) || L.tag.flip_parity >= 0 || v.basis->has_flips();
        const Keep keep = v.basis->n_up >= 0 ? Keep::Zero : (L.tag.sz_parity >= 0 ? Keep::Even : Keep::All);
        Group& g = groups[{v.basis.get(), flip, static_cast<int>(keep)}];
        g.levels.push_back(li);
        if (L.fold != Antiunitary::None) g.image = L.fold;
    }
    for (const auto& [key, g] : groups) {
        const auto& basis_ptr = r.vectors[static_cast<std::size_t>(r.levels[g.levels.front()].vector)].basis;
        const auto& basis = *basis_ptr;
        const bool flip = std::get<1>(key);
        const auto keep = static_cast<Keep>(std::get<2>(key));
        if (basis.irrep_dim > 1) {
            // A sector of an irrep of dimension > 1: each averaged operator acts within it as H does
            // (rep_sector.h), so <v|Obar|v> comes from one apply per level and operator.
            const std::size_t n = basis.states();
            std::vector<Complex> w(n);
            auto me = [&](const ::Operator& O, Antiunitary image, const std::vector<Complex>& v) {
                const ed::solvers::lg_detail::RepSectorMatVec op(avg.program(O, flip, keep, image), basis_ptr);
                op.apply(v.data(), w.data(), n);
                Complex acc(0.0, 0.0);
                for (std::size_t k = 0; k < n; ++k) acc += std::conj(v[k]) * w[k];
                return acc;
            };
            for (const std::size_t li : g.levels) {
                const Level& L = r.levels[li];
                const auto& v = r.vectors[static_cast<std::size_t>(L.vector)].amplitudes;
                for (std::size_t o = 0; o < n_ops; ++o) {
                    const Complex a = me(*ops[o], Antiunitary::None, v);
                    out[li][o] = L.fold != Antiunitary::None ? 0.5 * (a + std::conj(me(*ops[o], g.image, v))) : a;
                }
            }
            continue;
        }
        std::vector<ed::ops::MaskedOperator> avgs;
        avgs.reserve(2 * ops.size());
        for (const ::Operator* O : ops) avgs.push_back(avg.average(*O, flip, keep));
        if (g.image != Antiunitary::None)
            for (const ::Operator* O : ops) avgs.push_back(avg.average(*O, flip, keep, g.image));
        ed::ops::CompileOptions copt;
        copt.project = false;   // already invariant under the sector's group
        const auto prog = ed::ops::compile_program(avgs, basis, basis, copt);
        std::vector<ed::ops::RepVectorView> vecs;
        std::vector<std::pair<int, int>> pairs;
        for (std::size_t i = 0; i < g.levels.size(); ++i) {
            const auto& amp = r.vectors[static_cast<std::size_t>(r.levels[g.levels[i]].vector)].amplitudes;
            vecs.push_back({amp.data(), amp.size()});
            pairs.emplace_back(static_cast<int>(i), static_cast<int>(i));
        }
        const auto me = ed::ops::rep_matrix_elements(basis, basis, prog, vecs, vecs, pairs);
        const std::size_t width = avgs.size();
        for (std::size_t i = 0; i < g.levels.size(); ++i) {
            const Level& L = r.levels[g.levels[i]];
            for (std::size_t o = 0; o < n_ops; ++o) {
                const Complex a = me[i * width + o];
                out[g.levels[i]][o] =
                    L.fold != Antiunitary::None ? 0.5 * (a + std::conj(me[i * width + n_ops + o])) : a;
            }
        }
    }
    return out;
}

Complex matrix_element(const EigsResult& r, const ::Operator& O, std::size_t i, std::size_t j) {
    if (i >= r.levels.size() || j >= r.levels.size()) throw std::out_of_range("matrix_element: level index");
    if (static_cast<int>(O.getNumBits()) != r.n_sites)
        throw ed::InvalidRequest("matrix_element: the operator acts on " + std::to_string(O.getNumBits())
                                 + " sites, the levels on " + std::to_string(r.n_sites));
    const Level& Li = r.levels[i];
    const Level& Lj = r.levels[j];
    if (Li.vector < 0 || Lj.vector < 0) throw std::invalid_argument("matrix_element: a level has no vector");
    const BlockVector& bra = r.vectors[static_cast<std::size_t>(Li.vector)];
    const BlockVector& ket = r.vectors[static_cast<std::size_t>(Lj.vector)];
    const auto& src = *ket.basis;
    const auto& tgt = *bra.basis;
    // Two sectors of one group: the lambda-projected program, one sweep over the ket's
    // representatives. Otherwise (a group sector and a momentum sector, two stars' group
    // sectors, ...) the ket's orbit is walked explicitly.
    if (ed::ops::same_group(src, tgt) && (src.n_up < 0) == (tgt.n_up < 0) && src.irrep_dim == 1 && tgt.irrep_dim == 1) {
        const auto prog = ed::ops::compile_program({O.canonical()}, src, tgt);
        const ed::ops::RepVectorView k{ket.amplitudes.data(), ket.amplitudes.size()};
        const ed::ops::RepVectorView b{bra.amplitudes.data(), bra.amplitudes.size()};
        return ed::ops::rep_matrix_elements(src, tgt, prog, {k}, {b}, {{0, 0}})[0];
    }
    return ed::ops::orbit_matrix_element(ed::ops::compile_operator(O.canonical()), src, tgt, ket.amplitudes,
                                         bra.amplitudes);
}

}  // namespace ed::sectors
