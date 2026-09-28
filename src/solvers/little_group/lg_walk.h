#pragma once
// =============================================================================
// src/solvers/little_group/lg_walk.h -- the block walk shared by the sector-resolved
// verbs (lg_sectors*.cpp): engine options for one subspace, and the star-by-star walk.
// Private to the little-group engine.
// =============================================================================

#include "lg_internal.h"

#include <ed/sectors/sectors.h>
#include <ed/operators/casimir.h>
#include <ed/symmetry/casimir_projector.h>
#include <ed/symmetry/su2.h>

#include <numeric>
#include <set>

namespace ed::sectors::detail {

inline std::vector<Perm> abelian_or_identity(const Spec& s, int n_sites) {
    if (!s.abelian.empty()) return s.abelian;
    Perm id(static_cast<std::size_t>(n_sites));
    std::iota(id.begin(), id.end(), 0);
    return {id};
}

inline ed::solvers::LittleGroupOptions
engine_options(const Spec& s, const Subspace& sub, int dense_max_dim, int block_size) {
    ed::solvers::LittleGroupOptions o;
    o.n_up          = sub.n_up;
    o.sz_parity     = sub.sz_parity;
    // subspaces() already enforced 'require' against H. A subspace the flip maps onto a
    // different one gets the symmetry through the mirror fold, so inside it the engine
    // may only engage the flip where the subspace is its own image.
    o.spin_flip     = (s.spin_flip == 1 && sub.mirror == 2) ? -1 : s.spin_flip;
    o.time_reversal = s.time_reversal;
    o.only_k0       = s.only_k0;
    o.only_irrep    = s.only_irrep;
    o.dense_max_dim = dense_max_dim;
    o.block_size    = block_size;
    return o;
}

/// The operator one block is solved with. With a total-spin restriction it is the block's
/// H wrapped in the Lowdin projector onto the spin-S tower (S^2 built on the same basis):
/// H on the tower and `ghost` on the rest, which callers drop. `multiplicity` includes the
/// 2S + 1 members of each multiplet. A null `op` means the block holds no
/// state of the requested spin.
struct BlockOp {
    std::shared_ptr<const ed::matvec::MatVecOperator> op;
    double        ghost        = std::numeric_limits<double>::infinity();
    std::uint64_t multiplicity = 1;
    [[nodiscard]] bool is_ghost(double e) const {
        return std::isfinite(ghost) && e > ghost - 1e-6 * std::max(1.0, std::abs(ghost));
    }
};

inline BlockOp block_operator(const Spec& s, int n_sites, const Subspace& sub,
                              const ed::solvers::lg_detail::StarBuild& sb,
                              const std::shared_ptr<ed::solvers::LittleGroupBlock::Impl>& bi,
                              const std::shared_ptr<::Operator>& s2_carrier) {
    using namespace ed::solvers::lg_detail;
    BlockOp b;
    b.op = std::shared_ptr<const ed::matvec::MatVecOperator>(bi, &block_mv(*bi));
    b.multiplicity = bi->tag.multiplicity * static_cast<std::uint64_t>(sub.mirror);
    if (s.two_S < 0) return b;
    std::shared_ptr<const ed::matvec::MatVecOperator> s2;
    if (bi->gop) {
        s2 = std::make_shared<RepSectorMatVec>(*s2_carrier, bi->gsec);
    } else {
        auto s2k = std::make_shared<RepSectorMatVec>(*s2_carrier, sb.hk->rep_data_ptr());
        if (bi->W) s2 = std::make_shared<ProjectedBlockOp>(s2k, bi->W);
        else       s2 = s2k;
    }
    const auto towers = ed::symmetry::allowed_two_S_in_block(n_sites, sub.n_up, sub.sz_parity,
                                                             bi->tag.flip_parity);
    if (std::find(towers.begin(), towers.end(), s.two_S) == towers.end()) {
        b.op.reset();                   // this flip-parity block holds no spin-S state
        return b;
    }
    auto proj = std::make_shared<ed::symmetry::LowdinS2Projector>(s2, s.two_S, towers);
    auto wrapped = std::make_shared<ed::symmetry::CasimirProjectedOperator>(b.op, proj, 1);
    b.ghost = wrapped->ghost_shift();
    b.op = wrapped;
    b.multiplicity *= static_cast<std::uint64_t>(s.two_S + 1);
    return b;
}

/// The S^2 operator a total-spin restriction needs (null without one).
inline std::shared_ptr<::Operator> s2_carrier_for(const Spec& s, int n_sites) {
    return s.two_S >= 0 ? ed::ops::make_S2_carrier(static_cast<std::uint64_t>(n_sites)) : nullptr;
}

/// fn(cx, tr_on, star) for every star of one subspace, one star resident at a time.
template <class Fn>
void walk(const ::Operator& H, int n_sites, const Spec& s, const ed::solvers::LittleGroupOptions& opt,
          Fn&& fn) {
    using namespace ed::solvers::lg_detail;
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(H, abelian_or_identity(s, n_sites), s.residues, n_sites, opt, cx, tr_on);
    const std::set<int> only(s.only_k0.begin(), s.only_k0.end());
    for (const auto& [k0, members] : star_partition(cx, tr_on)) {
        if (!only.empty() && only.count(k0) == 0) continue;
        StarBuild sb = build_star_blocks(H, cx, tr_on, k0, members, opt, false,
                                         nullptr, nullptr, nullptr);
        fn(cx, tr_on, sb);
    }
}

}  // namespace ed::sectors::detail
