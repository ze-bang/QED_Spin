// =============================================================================
// src/solvers/little_group/lg_observables.cpp -- expectation values <n|O|n> of block
// eigenstates in the representative basis (no 2^N expansion).
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

#include <ed/symmetry/commute_check.h>

namespace ed::solvers {

using namespace lg_detail;

namespace {

// The lowest `want` eigenpairs of one block, in block coordinates. Dense below the
// lowest-k crossover (exact); one level through the certified ground-state solver
// (memory-light two-pass lane at frontier dimensions); several levels through
// (block) Krylov-Schur with vectors. `converged` is false when the block could not
// certify the requested window; the certified prefix is still returned.
[[nodiscard]] std::pair<std::vector<double>, std::vector<std::vector<Complex>>>
solve_block_eigenpairs(const ed::matvec::MatVecOperator& mv, int want,
                       int dense_max_dim, int block_size, bool* converged) {
    *converged = true;
    const std::size_t nb = mv.dim();
    std::vector<double> ev;
    std::vector<std::vector<Complex>> vv;
    if (nb == 0) return {ev, vv};
    const std::size_t k = std::min<std::size_t>(std::max(want, 1), nb);
    if (nb <= lowest_dense_floor(k, dense_max_dim) || nb <= 2) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(materialize(mv));
        if (es.info() != Eigen::Success)
            throw std::runtime_error("little_group expectations: dense block eigensolve failed");
        for (std::size_t j = 0; j < k; ++j) {
            ev.push_back(es.eigenvalues()(static_cast<Eigen::Index>(j)));
            std::vector<Complex> v(nb);
            for (std::size_t i = 0; i < nb; ++i)
                v[i] = es.eigenvectors()(static_cast<Eigen::Index>(i),
                                         static_cast<Eigen::Index>(j));
            vv.push_back(std::move(v));
        }
        return {ev, vv};
    }
    if (k == 1 && block_size <= 1) {
        try {
            auto [e0, v] = solve_gs_vector(mv, dense_max_dim);
            ev.push_back(e0);
            vv.push_back(std::move(v));
        } catch (const std::runtime_error&) {
            *converged = false;            // residual guard failed: certify nothing
        }
        return {ev, vv};
    }
    ev = solve_block_lowest_krylov_schur(mv, k, block_size, converged, &vv);
    return {ev, vv};
}

// The operator must be representable in the sector basis H's symmetries define.
void require_compatible(const ::Operator& O, std::size_t index, const EngineContext& cx,
                        bool tr_on, const LittleGroupOptions& opt) {
    const std::string who = "little_group_block_expectations: observable "
                            + std::to_string(index);
    for (std::size_t a = 0; a < cx.A.size(); ++a)
        if (!ed::symmetry::hamiltonian_commutes_with_permutation(
                O.transform_data_, O.three_body_data_, cx.A[a]))
            throw std::invalid_argument(
                who + " does not commute with abelian element " + std::to_string(a)
                + " (a translation): it mixes momentum sectors and has no matrix in "
                "one sector basis.");
    for (std::size_t r = 0; r < cx.residues.size(); ++r)
        if (!ed::symmetry::hamiltonian_commutes_with_permutation(
                O.transform_data_, O.three_body_data_, cx.residues[r]))
            throw std::invalid_argument(
                who + " does not commute with point-group residue " + std::to_string(r)
                + ". Drop that residue from residue_perms (the blocks then carry the "
                "smaller group) or symmetrise the observable.");
    const auto soa = term_soa(O);
    const auto axis = ed::symmetry::sz_axis_of(soa);
    if (opt.n_up >= 0 && axis != ed::symmetry::SzAxis::U1)
        throw std::invalid_argument(who + " does not conserve Sz, but n_up is fixed.");
    if (opt.sz_parity >= 0 && axis == ed::symmetry::SzAxis::None)
        throw std::invalid_argument(who + " does not conserve the Sz parity.");
    if (cx.flip_half && !ed::symmetry::hamiltonian_is_spin_flip_symmetric(soa))
        throw std::invalid_argument(
            who + " is not symmetric under the global spin flip, which the blocks use "
            "(pass spin_flip=0 to drop it).");
    if (tr_on && !ed::symmetry::hamiltonian_is_real(soa))
        throw std::invalid_argument(
            who + " is complex, but the blocks fold time reversal (pass time_reversal=0).");
}

}  // namespace

LittleGroupExpectations little_group_block_expectations(
    const ::Operator&                     op,
    const std::vector<const ::Operator*>& observables,
    const std::vector<std::vector<int>>&  abelian_group,
    const std::vector<std::vector<int>>&  residue_perms,
    int                                   n_sites,
    int                                   k,
    const LittleGroupOptions&             opt)
{
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, residue_perms, n_sites, opt, cx, tr_on);
    for (std::size_t i = 0; i < observables.size(); ++i) {
        if (observables[i] == nullptr)
            throw std::invalid_argument("little_group_block_expectations: null observable");
        if (static_cast<int>(observables[i]->getNumBits()) != n_sites)
            throw std::invalid_argument(
                "little_group_block_expectations: observable " + std::to_string(i)
                + " acts on a different number of sites");
        require_compatible(*observables[i], i, cx, tr_on, opt);
    }
    const auto stars = star_partition(cx, tr_on);
    bool ignore_plan = false;
    std::set<int> only_k0(opt.only_k0.begin(), opt.only_k0.end());
    parse_only_k0_env(only_k0, ignore_plan);

    LittleGroupExpectations out;
    out.flip_engaged = cx.flip_half;
    out.tr_engaged   = tr_on;
    for (int kk = 0; kk < cx.n_irr_raw; ++kk)
        out.irrep_characters.push_back(cx.giA.irreps[static_cast<std::size_t>(kk)].character);

    for (const auto& [k0, members] : stars) {
        if (!only_k0.empty() && only_k0.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(op, cx, tr_on, k0, members, opt,
                                         /*plan_print=*/false, nullptr, nullptr, nullptr);
        if (!sb.hk) { out.stars.push_back(sb.info); continue; }
        // One sector basis per star, shared by H and every observable.
        std::vector<std::unique_ptr<RepSectorMatVec>> Ok;
        Ok.reserve(observables.size());
        for (const auto* O : observables)
            Ok.push_back(std::make_unique<RepSectorMatVec>(*O, sb.hk->rep_data_ptr()));
        const std::size_t nrep = sb.hk->dim();
        std::vector<Complex> hu(nrep), ou(nrep);

        for (const auto& impl : sb.blocks) {
            LittleGroupBlock block(impl);
            const auto& tag = block.tag();
            bool conv = true;
            auto [ev, vv] = solve_block_eigenpairs(block.op(), k, opt.dense_max_dim,
                                                   opt.block_size, &conv);
            if (!conv) ++out.unconverged_blocks;
            for (std::size_t j = 0; j < ev.size() && j < vv.size(); ++j) {
                std::vector<Complex> u = block.lift_to_rep(vv[j].data());
                double nrm = 0.0;
                for (const auto& c : u) nrm += std::norm(c);
                nrm = std::sqrt(nrm);
                if (!(nrm > 0.0))
                    throw std::runtime_error("little_group_block_expectations: zero lifted vector");
                for (auto& c : u) c /= nrm;
                sb.hk->apply(u.data(), hu.data(), nrep);
                double res = 0.0;
                for (std::size_t i = 0; i < nrep; ++i) res += std::norm(hu[i] - ev[j] * u[i]);
                std::vector<double> vals;
                vals.reserve(Ok.size());
                for (const auto& O : Ok) {
                    O->apply(u.data(), ou.data(), nrep);
                    Complex s(0.0, 0.0);
                    for (std::size_t i = 0; i < nrep; ++i) s += std::conj(u[i]) * ou[i];
                    vals.push_back(s.real());
                }
                LittleGroupLabel lab;
                lab.k_raw       = tag.k_raw;
                lab.flip_parity = tag.flip_parity;
                lab.irrep       = tag.irrep;
                lab.irrep_dim   = tag.irrep_dim;
                lab.converged   = conv;
                out.energies.push_back(ev[j]);
                out.labels.push_back(lab);
                out.level.push_back(static_cast<int>(j));
                out.multiplicity.push_back(static_cast<int>(tag.multiplicity));
                out.values.push_back(std::move(vals));
                out.residuals.push_back(std::sqrt(res));
            }
        }
        sb.info.gpu_engaged = sb.hk->gpu_engaged();
        sb.info.csr_engaged = sb.hk->csr_engaged();
        out.stars.push_back(sb.info);
    }
    return out;
}

}  // namespace ed::solvers
