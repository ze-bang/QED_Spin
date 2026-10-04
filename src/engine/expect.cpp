// =============================================================================
// src/engine/expect.cpp -- expectation values, pair correlations and matrix elements
// over the sector-resolved eigenpairs (include/ed/sectors/expect.h).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "validate.h"
#include "walk.h"

#include <ed/parallel/numa.h>   // pin_omp_threads_once
#include <ed/sectors/expect.h>

#include <cmath>
#include <exception>
#include <map>
#include <tuple>

namespace ed::sectors {

using namespace ed::solvers;
using namespace ed::solvers::lg_detail;

namespace {

using ed::ops::MaskedOperator;

// Deduplicated averaged operators: each distinct one once, each input as (index, factor) with
// input = factor * unique[index] (kNone: the average vanished). Operators in one orbit of the
// group average to the same operator, so a family's pairs collapse to one per orbit here.
struct Unique {
    static constexpr std::size_t kNone = ~std::size_t{0};
    std::vector<MaskedOperator> ops;
    std::map<std::vector<std::uint64_t>, std::vector<std::size_t>> by_shape;   // term keys -> candidates

    std::pair<std::size_t, Complex> add(MaskedOperator a) {
        if (a.empty()) return {kNone, Complex(0.0, 0.0)};
        // The shape ignores roundoff-level terms (a cancellation left at 1e-17 in one member of
        // an orbit and exactly 0 in another), and the factor comes from the largest term.
        const auto terms = a.terms(1e-13 * a.max_abs());
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
            const auto ref = ops[c].terms(1e-13 * ops[c].max_abs());
            const Complex f = terms[top].coeff / ref[top].coeff;
            if (a.equals(ops[c].scaled(f))) return {c, f};
        }
        cands.push_back(ops.size());
        ops.push_back(std::move(a));
        return {ops.size() - 1, Complex(1.0, 0.0)};
    }
};

}  // namespace

std::vector<std::vector<Complex>> averaged_values(const EigsResult& r, const Spec& s,
                                                  const std::vector<MaskedOperator>& xs) {
    // Every level's vector lives in its own sector basis; the operators averaged over the
    // symmetry group (and the flip where the level folds or projects by it) are invariant, so
    // <v|Xbar|v> is one rep_matrix_elements sweep per (basis, flip, keep) over all of them --
    // and their antiunitary images, for a folded level: <K v|A|K v> = conj(<v|A^K|v>) -- and all
    // the levels sharing them. The averages are deduplicated first: the members of one orbit
    // cost one operator.
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
    for (const auto& entry : groups) {
        const auto& key = entry.first;
        const Group& g = entry.second;   // a plain reference: it is used inside OpenMP regions
        const auto& basis_ptr = r.vectors[static_cast<std::size_t>(r.levels[g.levels.front()].vector)].basis;
        const auto& basis = *basis_ptr;
        const bool flip = std::get<1>(key);
        const auto keep = static_cast<Keep>(std::get<2>(key));
        const bool images = g.image != Antiunitary::None;
        // Average every operator (and its antiunitary image) in parallel, then deduplicate in order.
        std::vector<MaskedOperator> averaged(images ? 2 * n_x : n_x, MaskedOperator(n_sites));
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic, 16)
        for (std::ptrdiff_t x = 0; x < static_cast<std::ptrdiff_t>(averaged.size()); ++x) {
            const auto i = static_cast<std::size_t>(x);
            try {
                averaged[i] = avg.average_of(xs[i % n_x], flip, keep, i < n_x ? Antiunitary::None : g.image);
            } catch (...) {
#pragma omp critical(qed_expect_failure)
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
        Unique uniq;
        std::vector<std::pair<std::size_t, Complex>> slot;
        slot.reserve(averaged.size());
        for (auto& a : averaged) slot.push_back(uniq.add(std::move(a)));
        const std::size_t width = uniq.ops.size();
        ED_LOG(Info, "[expect] %zu level(s) in a sector of dim %zu: %zu operator(s), %zu after averaging", g.levels.size(),
               basis.states(), averaged.size(), width);
        // vals[level i of the group][u] = <v_i| unique[u] |v_i>
        std::vector<std::vector<Complex>> vals(g.levels.size(), std::vector<Complex>(width, Complex(0.0, 0.0)));
        if (width > 0 && basis.irrep_dim > 1) {
            // A sector of an irrep of dimension > 1: each averaged operator acts within it as H does
            // (rep_sector.h), so <v|Xbar|v> comes from one apply per level and unique operator.
            const std::size_t n = basis.states();
            std::vector<Complex> w(n);
            for (std::size_t u = 0; u < width; ++u) {
                const RepSectorMatVec op(std::make_shared<const ed::ops::MaskedProgram>(
                                             ed::ops::compile_operator(uniq.ops[u].dagger())),
                                         basis_ptr);
                for (std::size_t i = 0; i < g.levels.size(); ++i) {
                    const auto& v = r.vectors[static_cast<std::size_t>(r.levels[g.levels[i]].vector)].amplitudes;
                    op.apply(v.data(), w.data(), n);
                    Complex acc(0.0, 0.0);
                    for (std::size_t k = 0; k < n; ++k) acc += std::conj(v[k]) * w[k];
                    vals[i][u] = acc;
                }
            }
        } else if (width > 0) {
            ed::ops::CompileOptions copt;
            copt.project = false;   // already invariant under the sector's group
            const auto prog = ed::ops::compile_program(uniq.ops, basis, basis, copt);
            std::vector<ed::ops::RepVectorView> vecs;
            std::vector<std::pair<int, int>> pairs;
            for (std::size_t i = 0; i < g.levels.size(); ++i) {
                const auto& amp = r.vectors[static_cast<std::size_t>(r.levels[g.levels[i]].vector)].amplitudes;
                vecs.push_back({amp.data(), amp.size()});
                pairs.emplace_back(static_cast<int>(i), static_cast<int>(i));
            }
            const auto me = ed::ops::rep_matrix_elements(basis, basis, prog, vecs, vecs, pairs);
            for (std::size_t i = 0; i < g.levels.size(); ++i)
                for (std::size_t u = 0; u < width; ++u) vals[i][u] = me[i * width + u];
        }
        auto value = [&](std::size_t i, std::size_t sl) {
            const auto& [u, f] = slot[sl];
            return u == Unique::kNone ? Complex(0.0, 0.0) : f * vals[i][u];
        };
        for (std::size_t i = 0; i < g.levels.size(); ++i) {
            const Level& L = r.levels[g.levels[i]];
            for (std::size_t x = 0; x < n_x; ++x) {
                const Complex a = value(i, x);
                out[g.levels[i]][x] = L.fold != Antiunitary::None ? 0.5 * (a + std::conj(value(i, n_x + x))) : a;
            }
        }
    }
    return out;
}

std::vector<std::vector<Complex>> expect(const EigsResult& r, const Spec& s,
                                         const std::vector<const ::Operator*>& ops) {
    detail::validate_environment("expect");
    for (std::size_t i = 0; i < ops.size(); ++i) detail::validate_observable(ops[i], r.n_sites, "expect", i);
    ed::parallel::pin_omp_threads_once();   // ED_NUMA_PIN_THREADS, as every verb
    std::vector<MaskedOperator> xs;
    xs.reserve(ops.size());
    for (const ::Operator* O : ops) xs.push_back(O->canonical());
    return averaged_values(r, s, xs);
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
    std::size_t total = singles.size();
    for (const auto& p : pair_requests) total += p.A.size() * p.B.size();
    std::vector<MaskedOperator> xs(total, MaskedOperator(r.n_sites));
    for (std::size_t i = 0; i < singles.size(); ++i) xs[i] = singles[i]->canonical();
    // The products A_a^dag B_b (B_b acts first), exact in the spin-1/2 algebra.
    std::size_t at = singles.size();
    for (const auto& p : pair_requests) {
        std::vector<MaskedOperator> Ad;
        Ad.reserve(p.A.size());
        for (const ::Operator* a : p.A) Ad.push_back(a->canonical().dagger());
        std::vector<MaskedOperator> Bc;
        Bc.reserve(p.B.size());
        for (const ::Operator* b : p.B) Bc.push_back(b->canonical());
        const std::size_t nb = Bc.size();
        const auto n_pairs = static_cast<std::ptrdiff_t>(Ad.size() * nb);
        std::exception_ptr failure;
#pragma omp parallel for schedule(dynamic, 16)
        for (std::ptrdiff_t x = 0; x < n_pairs; ++x) {
            const auto i = static_cast<std::size_t>(x);
            try {
                xs[at + i] = Ad[i / nb] * Bc[i % nb];
            } catch (...) {
#pragma omp critical(qed_expect_failure)
                if (!failure) failure = std::current_exception();
            }
        }
        if (failure) std::rethrow_exception(failure);
        at += Ad.size() * nb;
    }
    return averaged_values(r, s, xs);
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


}  // namespace ed::sectors
