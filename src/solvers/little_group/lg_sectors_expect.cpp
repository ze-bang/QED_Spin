// =============================================================================
// src/solvers/little_group/lg_sectors_expect.cpp -- expectation values and matrix elements
// over the sector-resolved eigenpairs (include/ed/sectors/expect.h).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_walk.h"

#include <ed/dssf/cross_sector_orbit_observable.h>
#include <ed/sectors/expect.h>

#include <array>
#include <map>
#include <set>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

// Every element of the group the permutations generate.
std::vector<Perm> close_group(const std::vector<Perm>& gens, int n) {
    Perm id(static_cast<std::size_t>(n));
    std::iota(id.begin(), id.end(), 0);
    std::set<Perm> seen{id};
    std::vector<Perm> out{id};
    for (std::size_t head = 0; head < out.size(); ++head)
        for (const Perm& g : gens) {
            Perm y(id.size());
            for (std::size_t i = 0; i < y.size(); ++i) y[i] = g[static_cast<std::size_t>(out[head][i])];
            if (seen.insert(y).second) out.push_back(std::move(y));
            if (out.size() > 1'000'000) throw std::runtime_error("expect: symmetry group too large to average over");
        }
    return out;
}

int sz_shift(int op) { return op == 0 ? -1 : (op == 1 ? 1 : 0); }

// One product of single-site operators; the factors of distinct sites commute, so they are
// kept in site order and equal products merge.
struct Term {
    std::array<std::uint8_t, 3>  op{};
    std::array<std::uint64_t, 3> site{};
    int n = 0;
};

enum class Keep { All, Sz, Parity };

// O averaged over the site permutations `G` (and the Sz flip), without the terms that change
// what `keep` conserves. Equal terms are merged, so the result is no larger than it must be.
::Operator averaged(const ::Operator& O, const std::vector<Perm>& G, bool flip, Keep keep) {
    std::vector<std::pair<Term, Complex>> terms;
    for (const auto& t : O.transform_data_) {
        Term x;
        x.n = t.is_two_body ? 2 : 1;
        x.op   = {t.op_type, t.op_type_2, 0};
        x.site = {t.site_index, t.site_index_2, 0};
        terms.push_back({x, t.coefficient});
    }
    for (const auto& t : O.three_body_data_) {
        Term x;
        x.n = 3;
        x.op   = {t.op_type_1, t.op_type_2, t.op_type_3};
        x.site = {t.site_index_1, t.site_index_2, t.site_index_3};
        terms.push_back({x, t.coefficient});
    }
    const double w = 1.0 / static_cast<double>(G.size() * (flip ? 2 : 1));
    std::map<std::array<std::uint64_t, 7>, Complex> acc;
    for (const auto& [t, c] : terms) {
        int shift = 0;
        for (int f = 0; f < t.n; ++f) shift += sz_shift(t.op[static_cast<std::size_t>(f)]);
        if ((keep == Keep::Sz && shift != 0) || (keep == Keep::Parity && shift % 2 != 0)) continue;
        for (int fl = 0; fl < (flip ? 2 : 1); ++fl)
            for (const Perm& g : G) {
                std::array<std::pair<std::uint64_t, std::uint8_t>, 3> f{};
                Complex coef = c * w;
                for (int k = 0; k < t.n; ++k) {
                    std::uint8_t op = t.op[static_cast<std::size_t>(k)];
                    if (fl) {                       // S+ <-> S-, Sz -> -Sz
                        if (op == 2) coef = -coef;
                        else op = static_cast<std::uint8_t>(1 - op);
                    }
                    f[static_cast<std::size_t>(k)] = {static_cast<std::uint64_t>(g[t.site[static_cast<std::size_t>(k)]]), op};
                }
                bool distinct = true;
                for (int a = 0; a < t.n; ++a)
                    for (int b = a + 1; b < t.n; ++b)
                        if (f[static_cast<std::size_t>(a)].first == f[static_cast<std::size_t>(b)].first) distinct = false;
                if (distinct) std::sort(f.begin(), f.begin() + t.n);
                std::array<std::uint64_t, 7> key{static_cast<std::uint64_t>(t.n)};
                for (int k = 0; k < t.n; ++k) {
                    key[1 + 2 * static_cast<std::size_t>(k)] = f[static_cast<std::size_t>(k)].first;
                    key[2 + 2 * static_cast<std::size_t>(k)] = f[static_cast<std::size_t>(k)].second;
                }
                acc[key] += coef;
            }
    }
    ::Operator out(O.getNumBits(), O.getSpin());
    for (const auto& [k, c] : acc) {
        if (std::abs(c) < 1e-15) continue;
        const auto op = [&](int i) { return static_cast<std::uint8_t>(k[2 + 2 * static_cast<std::size_t>(i)]); };
        const auto st = [&](int i) { return k[1 + 2 * static_cast<std::size_t>(i)]; };
        if (k[0] == 1)      out.addOneBodyTerm(op(0), st(0), c);
        else if (k[0] == 2) out.addTwoBodyTerm(op(0), st(0), op(1), st(1), c);
        else                out.addThreeBodyTerm(op(0), st(0), op(1), st(1), op(2), st(2), c);
    }
    return out;
}

::Operator conjugated(const ::Operator& O) {
    ::Operator out(O.getNumBits(), O.getSpin());
    out.copyTermsFrom(O);
    for (auto& t : out.transform_data_) t.coefficient = std::conj(t.coefficient);
    for (auto& t : out.three_body_data_) t.coefficient = std::conj(t.coefficient);
    return out;
}

Complex dot(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    Complex d(0, 0);
    for (std::size_t i = 0; i < a.size(); ++i) d += std::conj(a[i]) * b[i];
    return d;
}

}  // namespace

std::vector<std::vector<Complex>>
expect(const EigsResult& r, const Spec& s, int n_sites, const std::vector<const ::Operator*>& ops) {
    for (const ::Operator* O : ops)
        if (s.two_S >= 0 && !ed::symmetry::hamiltonian_is_su2_symmetric(term_soa(*O)))
            throw std::invalid_argument("expect: with a total-spin restriction every operator must be SU(2) "
                                        "invariant (a level holds one member of each spin multiplet)");
    std::vector<Perm> gens = detail::abelian_or_identity(s, n_sites);
    gens.insert(gens.end(), s.residues.begin(), s.residues.end());
    const std::vector<Perm> G = close_group(gens, n_sites);

    // One averaged operator per (op, flip, keep); one matvec per (averaged op, basis).
    std::map<std::tuple<std::size_t, bool, int, bool>, std::shared_ptr<::Operator>> averaged_ops;
    std::map<std::pair<const ::Operator*, const void*>, std::shared_ptr<RepSectorMatVec>> matvecs;
    auto average_of = [&](std::size_t o, bool flip, Keep keep, bool conj) {
        auto& slot = averaged_ops[{o, flip, static_cast<int>(keep), conj}];
        if (!slot) {
            ::Operator a = averaged(*ops[o], G, flip, keep);
            slot = std::make_shared<::Operator>(conj ? conjugated(a) : std::move(a));
        }
        return slot;
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

Complex matrix_element(const EigsResult& r, int n_sites, const ::Operator& O, std::size_t i, std::size_t j) {
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
