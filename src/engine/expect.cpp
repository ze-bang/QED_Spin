// =============================================================================
// src/engine/expect.cpp -- expectation values, pair correlations and matrix elements
// over the sector-resolved eigenpairs (include/ed/sectors/expect.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "validate.h"
#include "walk.h"

#include <ed/parallel/numa.h>   // pin_omp_threads_once
#include <ed/sectors/expect.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <stdexcept>
#include <tuple>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

using ed::ops::MaskedOperator;

// The deduplication of AveragedOperators: a key from the terms that matter, then a proportionality
// check against the operators already kept under that key.
struct Unique {
    std::vector<MaskedOperator>& ops;
    std::map<std::vector<std::uint64_t>, std::vector<std::size_t>> by_shape;   // term keys -> candidates

    std::pair<std::size_t, Complex> add(MaskedOperator a) {
        if (a.empty()) return {detail::AveragedOperators::kNone, Complex(0.0, 0.0)};
        // The shape ignores roundoff-level terms (a cancellation left at 1e-17 in one member of
        // an orbit and exactly 0 in another), and the factor comes from the largest term.
        const auto terms = a.terms(ed::numerics::kAveragedTermRel * a.max_abs());
        std::vector<std::uint64_t> shape;
        shape.reserve(3 * terms.size());
        std::size_t top = 0;
        for (std::size_t k = 0; k < terms.size(); ++k) {
            shape.push_back(terms[k].flip_mask);
            shape.push_back(terms[k].cond_val);
            shape.push_back(terms[k].sign_mask);
            if (std::abs(terms[k].coeff) > std::abs(terms[top].coeff)) top = k;
        }
        auto& cands = by_shape[shape];
        for (const std::size_t c : cands) {
            const auto ref = ops[c].terms(ed::numerics::kAveragedTermRel * ops[c].max_abs());
            const Complex f = terms[top].coeff / ref[top].coeff;
            if (a.equals(ops[c].scaled(f))) return {c, f};
        }
        cands.push_back(ops.size());
        ops.push_back(std::move(a));
        return {ops.size() - 1, Complex(1.0, 0.0)};
    }
};

// f(i) for every i < n on the thread team; the first exception is rethrown after the loop (an
// exception must not leave an OpenMP region).
template <class F>
void parallel_for(std::size_t n, F&& f) {
    std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic, 16)
    for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(n); ++x) {
        try {
            f(static_cast<std::size_t>(x));
        } catch (...) {
#pragma omp critical(qed_expect_failure)
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
}

}  // namespace

namespace detail {

std::shared_ptr<const AveragedOperators> average_operators(const Averager& avg, const std::vector<MaskedOperator>& xs,
                                                           bool flip, Keep keep, Antiunitary image) {
    auto out = std::make_shared<AveragedOperators>();
    out->n_x = xs.size();
    out->images = image != Antiunitary::None;
    if (xs.empty()) return out;
    const std::size_t n = out->images ? 2 * xs.size() : xs.size();
    std::vector<MaskedOperator> averaged(n, MaskedOperator(xs.front().n_sites()));
    parallel_for(n, [&](std::size_t i) {
        averaged[i] = avg.average_of(xs[i % out->n_x], flip, keep, i < out->n_x ? Antiunitary::None : image);
    });
    Unique uniq{out->unique, {}};
    out->slot.reserve(n);
    for (auto& a : averaged) out->slot.push_back(uniq.add(std::move(a)));
    ED_LOG(Info, "[measure] %zu operator(s), %zu after averaging (flip %d, keep %d, image %d)", n, out->unique.size(),
           static_cast<int>(flip), static_cast<int>(keep), static_cast<int>(image));
    return out;
}

SectorObservables::SectorObservables(std::shared_ptr<const AveragedOperators> ops,
                                     std::shared_ptr<const ed::symmetry::RepSectorData> basis)
    : ops_(std::move(ops)), basis_(std::move(basis)) {
    if (ops_->unique.empty()) return;
    if (basis_->irrep_dim > 1) {
        // A sector of an irrep of dimension > 1: each averaged operator acts within it as H does
        // (rep_sector.h), so <v|Xbar|v> comes from one apply per vector and unique operator.
        mv_.reserve(ops_->unique.size());
        for (const auto& u : ops_->unique)
            mv_.push_back(std::make_unique<RepSectorMatVec>(
                std::make_shared<const ed::ops::MaskedProgram>(ed::ops::compile_operator(u.dagger())), basis_));
    } else {
        ed::ops::CompileOptions copt;
        copt.project = false;   // already invariant under the sector's group
        prog_ = std::make_shared<const ed::ops::MaskedProgram>(
            ed::ops::compile_program(ops_->unique, *basis_, *basis_, copt));
    }
}

SectorObservables::Values SectorObservables::values(const std::vector<const Complex*>& vecs) const {
    std::vector<std::pair<int, int>> diag;
    diag.reserve(vecs.size());
    for (std::size_t i = 0; i < vecs.size(); ++i) diag.emplace_back(static_cast<int>(i), static_cast<int>(i));
    return values(vecs, diag);
}

SectorObservables::Values SectorObservables::values(const std::vector<const Complex*>& vecs,
                                                    const std::vector<std::pair<int, int>>& pairs) const {
    const std::size_t np = pairs.size(), width = ops_->unique.size(), nx = ops_->n_x;
    const std::size_t n = basis_->states();
    // vals[pair][u] = <bra| unique[u] |ket>: one sweep for every operator and pair.
    std::vector<std::vector<Complex>> vals(np, std::vector<Complex>(width, Complex(0.0, 0.0)));
    if (width > 0 && np > 0) {
        if (!mv_.empty()) {
            const std::lock_guard<std::mutex> lock(mv_mutex_);
            std::vector<Complex> w(n);
            for (std::size_t u = 0; u < width; ++u)
                for (std::size_t p = 0; p < np; ++p) {
                    const Complex* bra = vecs.at(static_cast<std::size_t>(pairs[p].first));
                    mv_[u]->apply(vecs.at(static_cast<std::size_t>(pairs[p].second)), w.data(), n);
                    Complex acc(0.0, 0.0);
                    for (std::size_t k = 0; k < n; ++k) acc += std::conj(bra[k]) * w[k];
                    vals[p][u] = acc;
                }
        } else {
            std::vector<ed::ops::RepVectorView> views;
            views.reserve(vecs.size());
            for (const Complex* v : vecs) views.push_back({v, n});
            const auto me = ed::ops::rep_matrix_elements(*basis_, *basis_, *prog_, views, views, pairs);
            for (std::size_t p = 0; p < np; ++p)
                for (std::size_t u = 0; u < width; ++u) vals[p][u] = me[p * width + u];
        }
    }
    auto value = [&](std::size_t p, std::size_t sl) {
        const auto& [u, f] = ops_->slot[sl];
        return u == AveragedOperators::kNone ? Complex(0.0, 0.0) : f * vals[p][u];
    };
    Values out;
    out.direct.assign(np, std::vector<Complex>(nx));
    if (ops_->images) out.image.assign(np, std::vector<Complex>(nx));
    for (std::size_t p = 0; p < np; ++p)
        for (std::size_t x = 0; x < nx; ++x) {
            out.direct[p][x] = value(p, x);
            if (ops_->images) out.image[p][x] = value(p, nx + x);
        }
    return out;
}

std::vector<std::vector<Complex>> SectorObservables::folded(const std::vector<const Complex*>& vecs) const {
    std::vector<std::pair<int, int>> diag;
    diag.reserve(vecs.size());
    for (std::size_t i = 0; i < vecs.size(); ++i) diag.emplace_back(static_cast<int>(i), static_cast<int>(i));
    return folded(vecs, diag);
}

std::vector<std::vector<Complex>> SectorObservables::folded(const std::vector<const Complex*>& vecs,
                                                            const std::vector<std::pair<int, int>>& pairs) const {
    Values v = values(vecs, pairs);
    if (ops_->images)
        for (std::size_t p = 0; p < v.direct.size(); ++p)
            for (std::size_t x = 0; x < v.direct[p].size(); ++x)
                v.direct[p][x] = 0.5 * (v.direct[p][x] + std::conj(v.image[p][x]));
    return std::move(v.direct);
}

std::vector<MaskedOperator> requested_operators(const std::vector<const ::Operator*>& singles,
                                                const std::vector<PairRequest>& pairs, int n_sites) {
    std::size_t total = singles.size();
    for (const auto& p : pairs) total += p.A.size() * p.B.size();
    std::vector<MaskedOperator> xs(total, MaskedOperator(n_sites));
    for (std::size_t i = 0; i < singles.size(); ++i) xs[i] = singles[i]->canonical();
    std::size_t at = singles.size();
    for (const auto& p : pairs) {
        std::vector<MaskedOperator> Ad;
        Ad.reserve(p.A.size());
        for (const ::Operator* a : p.A) Ad.push_back(a->canonical().dagger());
        std::vector<MaskedOperator> Bc;
        Bc.reserve(p.B.size());
        for (const ::Operator* b : p.B) Bc.push_back(b->canonical());
        const std::size_t nb = Bc.size();
        parallel_for(Ad.size() * nb, [&](std::size_t i) { xs[at + i] = Ad[i / nb] * Bc[i % nb]; });
        at += Ad.size() * nb;
    }
    return xs;
}

}  // namespace detail

std::vector<std::vector<Complex>> averaged_values(const EigsResult& r, const Spec& s,
                                                  const std::vector<MaskedOperator>& xs) {
    // Every level's vector lives in its own sector basis; the operators averaged over the
    // symmetry group (and the flip where the level folds or projects by it) are invariant, so
    // <v|Xbar|v> is one sweep per (basis, flip, keep) over all of them -- and their antiunitary
    // images, for a folded level: <K v|A|K v> = conj(<v|A^K|v>) -- and all the levels sharing them.
    // The averages are formed once per (flip, keep, image) and deduplicated: the members of one
    // orbit cost one operator.
    const int n_sites = r.n_sites;
    const detail::Averager avg(s, n_sites, s.two_S >= 0 && !r.levels.empty() && detail::members(r.levels.front()) > 1);
    const std::size_t n_x = xs.size();
    std::vector<std::vector<Complex>> out(r.levels.size(), std::vector<Complex>(n_x, Complex(0.0, 0.0)));
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
        if (L.vector < 0) throw ed::InvalidRequest("expect: a level has no vector (solve with vectors)");
        const BlockVector& v = r.vectors[static_cast<std::size_t>(L.vector)];
        const bool flip = (L.mirror == 2 && !detail::theta_mirror(L)) || L.tag.flip_parity >= 0 || v.basis->has_flips();
        const Keep keep = v.basis->n_up >= 0 ? Keep::Zero : (L.tag.sz_parity >= 0 ? Keep::Even : Keep::All);
        Group& g = groups[{v.basis.get(), flip, static_cast<int>(keep)}];
        g.levels.push_back(li);
        if (L.fold != Antiunitary::None) g.image = L.fold;
    }
    std::map<std::tuple<bool, int, int>, std::shared_ptr<const detail::AveragedOperators>> averaged;
    for (const auto& [key, g] : groups) {
        const bool flip = std::get<1>(key);
        const auto keep = static_cast<Keep>(std::get<2>(key));
        auto& ops = averaged[{flip, static_cast<int>(keep), static_cast<int>(g.image)}];
        if (!ops) ops = detail::average_operators(avg, xs, flip, keep, g.image);
        const detail::SectorObservables so(
            ops, r.vectors[static_cast<std::size_t>(r.levels[g.levels.front()].vector)].basis);
        std::vector<const Complex*> vecs;
        for (const std::size_t li : g.levels)
            vecs.push_back(r.vectors[static_cast<std::size_t>(r.levels[li].vector)].amplitudes.data());
        const auto v = so.values(vecs);
        for (std::size_t i = 0; i < g.levels.size(); ++i) {
            const bool fold = r.levels[g.levels[i]].fold != Antiunitary::None;
            for (std::size_t x = 0; x < n_x; ++x)
                out[g.levels[i]][x] = fold ? 0.5 * (v.direct[i][x] + std::conj(v.image[i][x])) : v.direct[i][x];
        }
    }
    return out;
}

std::vector<std::vector<Complex>> expect(const EigsResult& r, const Spec& s,
                                         const std::vector<const ::Operator*>& ops) {
    detail::validate_environment("expect");
    for (std::size_t i = 0; i < ops.size(); ++i) detail::validate_observable(ops[i], r.n_sites, "expect", i);
    ed::parallel::pin_omp_threads_once();   // ED_NUMA_PIN_THREADS, as every verb
    return averaged_values(r, s, detail::requested_operators(ops, {}, r.n_sites));
}

std::vector<std::vector<Complex>> evaluate(const EigsResult& r, const Spec& s,
                                           const std::vector<const ::Operator*>& singles,
                                           const std::vector<PairRequest>& pair_requests) {
    detail::validate_environment("evaluate");
    std::size_t index = 0;
    for (const ::Operator* O : singles) detail::validate_observable(O, r.n_sites, "evaluate", index++);
    for (const auto& p : pair_requests) {
        for (const ::Operator* O : p.A) detail::validate_observable(O, r.n_sites, "evaluate", index++);
        for (const ::Operator* O : p.B) detail::validate_observable(O, r.n_sites, "evaluate", index++);
    }
    ed::parallel::pin_omp_threads_once();
    return averaged_values(r, s, detail::requested_operators(singles, pair_requests, r.n_sites));
}

Complex matrix_element(const EigsResult& r, const ::Operator& O, std::size_t i, std::size_t j) {
    if (i >= r.levels.size() || j >= r.levels.size()) throw std::out_of_range("matrix_element: level index");
    if (static_cast<int>(O.getNumBits()) != r.n_sites)
        throw ed::InvalidRequest("matrix_element: the operator acts on " + std::to_string(O.getNumBits())
                                 + " sites, the levels on " + std::to_string(r.n_sites));
    if (!std::isfinite(O.canonical().l1_norm()))
        throw ed::InvalidRequest("matrix_element: the operator has a coefficient that is not finite (NaN or inf)");
    const Level& Li = r.levels[i];
    const Level& Lj = r.levels[j];
    if (Li.vector < 0 || Lj.vector < 0) throw ed::InvalidRequest("matrix_element: a level has no vector");
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

TransitionAmplitudes transition_amplitudes(const EigsResult& ri, const Spec& si,
                                           const std::vector<std::size_t>& initial, const EigsResult& rf,
                                           const Spec& sf, const std::vector<std::size_t>& final,
                                           const std::vector<const ::Operator*>& ops) {
    detail::validate_environment("transitions");
    const int n_sites = ri.n_sites;
    if (rf.n_sites != n_sites)
        throw ed::InvalidRequest("transitions: the initial levels are on " + std::to_string(n_sites)
                                 + " sites, the final ones on " + std::to_string(rf.n_sites));
    for (std::size_t i = 0; i < ops.size(); ++i) detail::validate_observable(ops[i], n_sites, "transitions", i);
    if (si.two_S >= 0 || sf.two_S >= 0)
        throw ed::Unsupported("transitions: levels of a total-spin restriction (their Sz members are not "
                              "in the result); solve without total_spin");
    ed::parallel::pin_omp_threads_once();
    // Both multiplets go into momentum sectors of one abelian group, where one program per
    // (source, target) sector pair carries every operator.
    auto sorted_group = [&](const Spec& s) {
        auto g = detail::abelian_or_identity(s, n_sites);
        std::sort(g.begin(), g.end());
        return g;
    };
    const auto A = sorted_group(si);
    if (A != sorted_group(sf))
        throw ed::InvalidRequest("transitions: the initial and final levels were solved over different "
                                 "abelian groups (use one spatial symmetry for both)");
    const ::Operator zero(static_cast<std::uint64_t>(n_sites), 0.5f);   // the sectors need no H
    detail::MemberSectors ms{zero, A, n_sites, {}};
    TransitionAmplitudes out;
    auto expand = [&](const EigsResult& r, const Spec& s, const std::vector<std::size_t>& idx,
                      std::vector<std::size_t>& offsets, std::vector<detail::Member>& members) {
        offsets.push_back(0);
        for (const std::size_t i : idx) {
            if (i >= r.levels.size()) throw std::out_of_range("transitions: level index");
            const Level& L = r.levels[i];
            if (L.vector < 0) throw ed::InvalidRequest("transitions: a level has no vector (solve with vectors)");
            auto m = detail::members_of(L, r.vectors[static_cast<std::size_t>(L.vector)], L.multiplicity, s, ms);
            for (auto& x : m) members.push_back(std::move(x));
            offsets.push_back(members.size());
        }
    };
    std::vector<detail::Member> mi, mf;
    expand(ri, si, initial, out.initial_offsets, mi);
    expand(rf, sf, final, out.final_offsets, mf);
    const std::size_t n_ops = ops.size(), ni = mi.size(), nf = mf.size();
    out.n_ops = n_ops;
    out.amplitudes.assign(n_ops * nf * ni, Complex(0.0, 0.0));   // [op][final member][initial member]
    std::vector<MaskedOperator> xs;
    xs.reserve(n_ops);
    for (const ::Operator* O : ops) xs.push_back(O->canonical());
    // Members by sector.
    std::map<const void*, std::vector<std::size_t>> src, tgt;
    for (std::size_t n = 0; n < ni; ++n) src[mi[n].v.basis.get()].push_back(n);
    for (std::size_t m = 0; m < nf; ++m) tgt[mf[m].v.basis.get()].push_back(m);
    for (const auto& [sp, kets_idx] : src) {
        const auto& sb = *mi[kets_idx.front()].v.basis;
        for (const auto& [tp, bras_idx] : tgt) {
            const auto& tb = *mf[bras_idx.front()].v.basis;
            // The lambda projection keeps what connects the two momenta (O_q maps k to k - q) and the
            // S^z change between their subspaces; a program with no term is an exact zero.
            const auto prog = ed::ops::compile_program(xs, sb, tb);
            if (prog.n_groups() == 0) continue;
            std::vector<ed::ops::RepVectorView> kets, bras;
            for (const std::size_t n : kets_idx) kets.push_back({mi[n].v.amplitudes.data(), mi[n].v.amplitudes.size()});
            for (const std::size_t m : bras_idx) bras.push_back({mf[m].v.amplitudes.data(), mf[m].v.amplitudes.size()});
            std::vector<std::pair<int, int>> pairs;   // (bra, ket)
            for (std::size_t b = 0; b < bras_idx.size(); ++b)
                for (std::size_t k = 0; k < kets_idx.size(); ++k)
                    pairs.emplace_back(static_cast<int>(b), static_cast<int>(k));
            const auto me = ed::ops::rep_matrix_elements(sb, tb, prog, kets, bras, pairs);
            for (std::size_t p = 0; p < pairs.size(); ++p) {
                const std::size_t m = bras_idx[static_cast<std::size_t>(pairs[p].first)];
                const std::size_t n = kets_idx[static_cast<std::size_t>(pairs[p].second)];
                for (std::size_t o = 0; o < n_ops; ++o) out.amplitudes[(o * nf + m) * ni + n] = me[p * n_ops + o];
            }
        }
    }
    out.n_initial = ni;
    out.n_final = nf;
    return out;
}

}  // namespace ed::sectors
