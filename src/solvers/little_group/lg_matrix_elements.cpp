// =============================================================================
// src/solvers/little_group/lg_matrix_elements.cpp -- matrix elements of general
// (MaskedOperator) observables between block eigenstates, in the representative basis.
// Part of the little-group engine; see lg_internal.h for the file map and
// little_group_blocks.h (little_group_block_observables) for the contract.
// =============================================================================

#include "lg_internal.h"

#include <ed/observables/rep_matrix_elements.h>

#include <limits>

namespace ed::solvers {

using namespace lg_detail;

namespace {

// One solved star: its sector basis and the states that live in it.
struct SolvedSector {
    std::shared_ptr<const ed::symmetry::RepSectorData> rd;
    int k_raw = -1;
    std::vector<int> states;                       // indices into result.states
    std::vector<std::vector<Complex>> vecs;        // rep-basis vectors, parallel
};

}  // namespace

LittleGroupMEResult little_group_block_observables(
    const ::Operator&                                  op,
    const std::vector<ed::observables::MaskedOperator>& ops,
    const std::vector<std::vector<int>>&               abelian_group,
    const std::vector<std::vector<int>>&               residue_perms,
    int                                                n_sites,
    const LittleGroupOptions&                          opt,
    const LittleGroupMEOptions&                        me)
{
    if (me.levels < 1)
        throw std::invalid_argument("little_group_block_observables: levels must be >= 1");
    for (std::size_t i = 0; i < ops.size(); ++i)
        if (ops[i].n_sites() != n_sites)
            throw std::invalid_argument("little_group_block_observables: observable "
                                        + std::to_string(i) + " acts on a different number of sites");

    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt, cx, tr_on);
    const auto stars = star_partition(cx, tr_on);
    bool ignore_plan = false;
    std::set<int> only_k0(opt.only_k0.begin(), opt.only_k0.end());
    parse_only_k0_env(only_k0, ignore_plan);

    LittleGroupMEResult out;
    out.flip_engaged = cx.flip_half;
    out.tr_engaged   = tr_on;
    for (int kk = 0; kk < cx.n_irr_raw; ++kk)
        out.irrep_characters.push_back(cx.giA.irreps[static_cast<std::size_t>(kk)].character);

    // ---- solve star by star; keep only rep data + lifted vectors ------------------
    std::vector<SolvedSector> sectors;
    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(op, cx, tr_on, k0, members, opt,
                                         /*plan_print=*/false, nullptr, nullptr, nullptr);
        if (!sb.hk) { out.stars.push_back(sb.info); continue; }
        const int star_k0 = k0;   // structured bindings cannot be captured in C++17
        SolvedSector S;
        S.rd = sb.hk->rep_data_ptr();
        const std::size_t nrep = sb.hk->dim();
        std::vector<Complex> hu(nrep);

        const auto add_state = [&](std::vector<Complex> u, double e, const LittleGroupBlockTag& tag,
                                   int level, int partner, bool conv) {
            double nrm = 0.0;
            for (const auto& c : u) nrm += std::norm(c);
            nrm = std::sqrt(nrm);
            if (!(nrm > 0.0))
                throw std::runtime_error("little_group_block_observables: zero lifted vector");
            for (auto& c : u) c /= nrm;
            sb.hk->apply(u.data(), hu.data(), nrep);
            double res = 0.0;
            for (std::size_t i = 0; i < nrep; ++i) res += std::norm(hu[i] - e * u[i]);
            LittleGroupMEState st;
            st.energy = e;
            st.label.k_raw = tag.k_raw;
            st.label.flip_parity = tag.flip_parity;
            st.label.irrep = tag.irrep;
            st.label.irrep_dim = tag.irrep_dim;
            st.label.converged = conv;
            st.star_k0 = star_k0;
            st.level = level;
            st.partner = partner;
            st.multiplicity = tag.multiplicity;
            st.residual = std::sqrt(res);
            S.states.push_back(static_cast<int>(out.states.size()));
            out.states.push_back(st);
            S.vecs.push_back(std::move(u));
        };

        for (const auto& impl : sb.blocks) {
            LittleGroupBlock block(impl);
            const auto& tag = block.tag();
            S.k_raw = tag.k_raw;
            bool conv = true;
            auto [ev, vv] = solve_block_eigenpairs(block.op(), me.levels, opt.dense_max_dim,
                                                   opt.block_size, &conv);
            if (!conv) ++out.unconverged_blocks;
            for (std::size_t j = 0; j < ev.size() && j < vv.size(); ++j) {
                std::vector<Complex> u = block.lift_to_rep(vv[j].data());
                std::vector<std::vector<Complex>> partners;
                if (me.partners && tag.irrep_dim > 1) partners = block.degenerate_partners(u);
                add_state(std::move(u), ev[j], tag, static_cast<int>(j), 0, conv);
                for (std::size_t q = 0; q < partners.size(); ++q)
                    add_state(std::move(partners[q]), ev[j], tag, static_cast<int>(j),
                              static_cast<int>(q + 1), conv);
            }
        }
        sb.info.gpu_engaged = sb.hk->gpu_engaged();
        sb.info.csr_engaged = sb.hk->csr_engaged();
        out.stars.push_back(sb.info);
        if (!S.states.empty()) sectors.push_back(std::move(S));
        // sb (H_k0 and its device mirror) is released here
    }

    // ---- one sweep per (ket sector, bra sector) of equal momentum -----------------
    ed::observables::RepMEOptions ro;
    ro.use_gpu = me.use_gpu;
    const bool diagonal = (me.pairs == LittleGroupMEOptions::Pairs::diagonal);
    // Off-diagonal pairs carry only me.pair_observables when that list is given (a
    // multi-thousand observable set on every pair would multiply the sweeps); the
    // entries not evaluated are NaN, never a silent zero.
    std::vector<ed::observables::MaskedOperator> off_ops;
    std::vector<std::size_t> off_index;
    if (!me.pair_observables.empty()) {
        for (int o : me.pair_observables) {
            if (o < 0 || o >= static_cast<int>(ops.size()))
                throw std::invalid_argument("little_group_block_observables: pair_observables index "
                                            + std::to_string(o) + " out of range");
            off_ops.push_back(ops[static_cast<std::size_t>(o)]);
            off_index.push_back(static_cast<std::size_t>(o));
        }
    }
    const bool restrict_off = !me.pair_observables.empty();
    const Complex nan_c(std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::quiet_NaN());
    const auto run = [&](const std::vector<ed::observables::MaskedOperator>& set,
                         const std::vector<std::size_t>* index, std::size_t a, std::size_t b,
                         const std::vector<std::pair<int, int>>& pr) {
        if (pr.empty()) return;
        const auto& src = *sectors[a].rd;
        const auto& tgt = *sectors[b].rd;   // same object when a == b
        const auto prog = ed::observables::compile_program(set, src, tgt);
        std::vector<ed::observables::RepVectorView> kv, bv;
        for (const auto& v : sectors[a].vecs) kv.push_back({v.data(), v.size()});
        for (const auto& v : sectors[b].vecs) bv.push_back({v.data(), v.size()});
        const auto M = ed::observables::rep_matrix_elements(src, tgt, prog, kv, bv, pr, ro);
        for (std::size_t p = 0; p < pr.size(); ++p) {
            out.pairs.emplace_back(sectors[b].states[static_cast<std::size_t>(pr[p].first)],
                                   sectors[a].states[static_cast<std::size_t>(pr[p].second)]);
            std::vector<Complex> row(ops.size(), index ? nan_c : Complex(0.0, 0.0));
            for (std::size_t o = 0; o < set.size(); ++o)
                row[index ? (*index)[o] : o] = M[p * set.size() + o];
            out.values.push_back(std::move(row));
        }
    };
    for (std::size_t a = 0; a < sectors.size(); ++a)
        for (std::size_t b = 0; b < sectors.size(); ++b) {
            if (diagonal && a != b) continue;
            if (sectors[a].k_raw != sectors[b].k_raw) continue;
            const int nk = static_cast<int>(sectors[a].vecs.size());
            const int nb = static_cast<int>(sectors[b].vecs.size());
            std::vector<std::pair<int, int>> diag_pr, off_pr;
            for (int bi = 0; bi < nb; ++bi)
                for (int ki = 0; ki < nk; ++ki) {
                    if (a == b && bi == ki) diag_pr.emplace_back(bi, ki);
                    else if (!diagonal) off_pr.emplace_back(bi, ki);
                }
            if (!restrict_off) {
                std::vector<std::pair<int, int>> all = diag_pr;
                all.insert(all.end(), off_pr.begin(), off_pr.end());
                run(ops, nullptr, a, b, all);
            } else {
                run(ops, nullptr, a, b, diag_pr);
                run(off_ops, &off_index, a, b, off_pr);
            }
        }
    return out;
}

}  // namespace ed::solvers
