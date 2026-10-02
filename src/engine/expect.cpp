// =============================================================================
// src/engine/expect.cpp -- expectation values and matrix elements
// over the sector-resolved eigenpairs (include/ed/sectors/expect.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "walk.h"

#include <ed/dynamics/cross_sector.h>
#include <ed/sectors/expect.h>

#include <array>
#include <map>
#include <set>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

Complex dot(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    Complex d(0, 0);
    for (std::size_t i = 0; i < a.size(); ++i) d += std::conj(a[i]) * b[i];
    return d;
}

}  // namespace


std::vector<std::vector<Complex>>
expect(const EigsResult& r, const Spec& s, const std::vector<const ::Operator*>& ops) {
    const int n_sites = r.n_sites;
    for (const ::Operator* O : ops)
        if (s.two_S >= 0 && !ed::ops::su2_invariant(O->canonical()))
            throw std::invalid_argument("expect: with a total-spin restriction every operator must be SU(2) "
                                        "invariant (a level holds one member of each spin multiplet)");
    // Every level's vector lives in its own sector basis; the operators averaged over the
    // symmetry group (and the flip where the level folds or projects by it) are invariant, so
    // <v|Obar|v> is one rep_matrix_elements sweep per (basis, flip, keep) over all operators --
    // and their conjugates, for a time-reversal-folded level: <K v|A|K v> = conj(<v|A^K|v>) --
    // and all the levels sharing them. No matvec, no CSR, a fixed order.
    detail::Averager avg(s, n_sites);
    const std::size_t n_ops = ops.size();
    std::vector<std::vector<Complex>> out(r.levels.size(), std::vector<Complex>(n_ops));
    struct Group { std::vector<std::size_t> levels; bool folded = false; };
    std::map<std::tuple<const void*, bool, int>, Group> groups;
    using detail::Keep;
    for (std::size_t li = 0; li < r.levels.size(); ++li) {
        const Level& L = r.levels[li];
        if (L.vector < 0) throw std::invalid_argument("expect: a level has no vector (solve with vectors)");
        const BlockVector& v = r.vectors[static_cast<std::size_t>(L.vector)];
        const bool flip = L.mirror == 2 || L.tag.flip_parity >= 0 || v.basis->has_flips();
        const Keep keep = v.basis->n_up >= 0 ? Keep::Zero : (L.tag.sz_parity >= 0 ? Keep::Even : Keep::All);
        Group& g = groups[{v.basis.get(), flip, static_cast<int>(keep)}];
        g.levels.push_back(li);
        g.folded = g.folded || L.tag.tr_folded;
    }
    for (const auto& [key, g] : groups) {
        const auto& basis = *r.vectors[static_cast<std::size_t>(r.levels[g.levels.front()].vector)].basis;
        const bool flip = std::get<1>(key);
        const auto keep = static_cast<Keep>(std::get<2>(key));
        std::vector<ed::ops::MaskedOperator> avgs;
        for (const ::Operator* O : ops) avgs.push_back(avg.average(*O, flip, keep, false));
        if (g.folded)
            for (const ::Operator* O : ops) avgs.push_back(avg.average(*O, flip, keep, true));
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
                out[g.levels[i]][o] = L.tag.tr_folded ? 0.5 * (a + std::conj(me[i * width + n_ops + o])) : a;
            }
        }
    }
    return out;
}

Complex matrix_element(const EigsResult& r, const ::Operator& O, std::size_t i, std::size_t j) {
    if (O.has_extra_terms())
        throw ed::Unsupported("terms on four or more sites are not supported in an observable yet (expect, thermal "
                              "observables, dynamics, matrix_element); they work in the Hamiltonian");
    const int n_sites = r.n_sites;
    if (i >= r.levels.size() || j >= r.levels.size()) throw std::out_of_range("matrix_element: level index");
    if (!O.three_body_records().empty())
        throw std::invalid_argument("matrix_element: three-body terms are not supported between sectors");
    const Level& Li = r.levels[i];
    const Level& Lj = r.levels[j];
    if (Li.vector < 0 || Lj.vector < 0) throw std::invalid_argument("matrix_element: a level has no vector");
    const BlockVector& vi = r.vectors[static_cast<std::size_t>(Li.vector)];
    const BlockVector& vj = r.vectors[static_cast<std::size_t>(Lj.vector)];
    using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;
    const auto n = static_cast<std::uint64_t>(n_sites);
    ed::dssf::CrossSectorOrbitObservable obs(Ref::from_rep(*vj.basis, n), 0, Ref::from_rep(*vi.basis, n), 0,
                                             O.records(), O.getSpin());
    std::vector<Complex> y(vi.amplitudes.size());
    obs.apply(vj.amplitudes.data(), y.data(), y.size());
    return dot(vi.amplitudes, y);
}

}  // namespace ed::sectors
