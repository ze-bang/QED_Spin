#pragma once
// =============================================================================
// src/solvers/little_group/lg_walk.h -- the block walk shared by the sector-resolved
// verbs (lg_sectors*.cpp): engine options for one subspace, and the star-by-star walk.
// Private to the little-group engine.
// =============================================================================

#include "lg_internal.h"

#include <ed/sectors/sectors.h>

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
