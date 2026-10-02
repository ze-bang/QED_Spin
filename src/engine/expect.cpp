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
        if (s.two_S >= 0 && !ed::ops::su2_invariant(ed::ops::masked(*O)))
            throw std::invalid_argument("expect: with a total-spin restriction every operator must be SU(2) "
                                        "invariant (a level holds one member of each spin multiplet)");
    // One averaged operator per (op, flip, keep); one matvec per (averaged op, basis).
    detail::Averager avg(s, n_sites);
    std::map<std::pair<const ::Operator*, const void*>, std::shared_ptr<RepSectorMatVec>> matvecs;
    auto average_of = [&](std::size_t o, bool flip, detail::Keep keep, bool conj) {
        return avg.get(*ops[o], flip, keep, conj);
    };
    auto value = [&](const ::Operator& A, const BlockVector& v) {
        auto& mv = matvecs[{&A, v.basis.get()}];
        if (!mv) mv = std::make_shared<RepSectorMatVec>(A, v.basis);
        std::vector<Complex> y(v.amplitudes.size());
        mv->apply(v.amplitudes.data(), y.data(), y.size());
        return dot(v.amplitudes, y);
    };

    std::vector<std::vector<Complex>> out;
    out.reserve(r.levels.size());
    for (const Level& L : r.levels) {
        if (L.vector < 0) throw std::invalid_argument("expect: a level has no vector (solve with vectors)");
        const BlockVector& v = r.vectors[static_cast<std::size_t>(L.vector)];
        const bool flip = L.mirror == 2 || L.tag.flip_parity >= 0 || v.basis->has_flips();
        using detail::Keep;
        const Keep keep = v.basis->n_up >= 0 ? Keep::Sz : (L.tag.sz_parity >= 0 ? Keep::Parity : Keep::All);
        std::vector<Complex> row;
        for (std::size_t o = 0; o < ops.size(); ++o) {
            const Complex a = value(*average_of(o, flip, keep, false), v);
            // The time-reversed partner K psi: <K psi|A|K psi> = conj(<psi|A*|psi>).
            row.push_back(L.tag.tr_folded ? 0.5 * (a + std::conj(value(*average_of(o, flip, keep, true), v))) : a);
        }
        out.push_back(std::move(row));
    }
    return out;
}

Complex matrix_element(const EigsResult& r, const ::Operator& O, std::size_t i, std::size_t j) {
    const int n_sites = r.n_sites;
    if (i >= r.levels.size() || j >= r.levels.size()) throw std::out_of_range("matrix_element: level index");
    if (!O.three_body_data_.empty())
        throw std::invalid_argument("matrix_element: three-body terms are not supported between sectors");
    const Level& Li = r.levels[i];
    const Level& Lj = r.levels[j];
    if (Li.vector < 0 || Lj.vector < 0) throw std::invalid_argument("matrix_element: a level has no vector");
    const BlockVector& vi = r.vectors[static_cast<std::size_t>(Li.vector)];
    const BlockVector& vj = r.vectors[static_cast<std::size_t>(Lj.vector)];
    using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;
    const auto n = static_cast<std::uint64_t>(n_sites);
    ed::dssf::CrossSectorOrbitObservable obs(Ref::from_rep(*vj.basis, n), 0, Ref::from_rep(*vi.basis, n), 0,
                                             O.transform_data_, O.getSpin());
    std::vector<Complex> y(vi.amplitudes.size());
    obs.apply(vj.amplitudes.data(), y.data(), y.size());
    return dot(vi.amplitudes, y);
}

}  // namespace ed::sectors
