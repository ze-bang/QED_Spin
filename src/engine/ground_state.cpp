// =============================================================================
// src/engine/ground_state.cpp -- the streamed raw momentum sectors and
// shared sector data (the certified ground-state vector lives in block_solve.cpp).
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "internal.h"

namespace ed::solvers {

using namespace lg_detail;

void little_group_k_sectors_stream(const ::Operator& op, const std::vector<std::vector<int>>& abelian_group,
                                   int n_sites, int n_up, int sz_parity,
                                   const std::function<void(ed::symmetry::RepSectorData&)>& fn) {
    // Build ONE raw momentum
    // sector at a time, hand it to ``fn``, then free it before building the
    // next. Holding every k-sector resident
    // costs ~15-20 GB/sector at N=36 half-filling -- 12 sectors
    // OOMs a 128 GB node. This keeps the resident set at one destination
    // sector for the factorized static/dynamical structure-factor loops.
    LittleGroupOptions o;
    o.n_up = n_up;
    o.sz_parity = sz_parity;
    o.spin_flip = 0;      // destination sectors are RAW
    o.time_reversal = 0;
    EngineContext cx;
    bool tr_on = false;
    make_engine_context(op, abelian_group, {}, n_sites, o, cx, tr_on);
    for (int k = 0; k < cx.n_irr_raw; ++k) {
        auto rd = build_k_sector(cx, k, n_up);
        if (!rd.reps.empty()) fn(rd);
    }
}

std::shared_ptr<const ed::symmetry::RepSectorData> share_rep_sector(ed::symmetry::RepSectorData rd) {
    auto p = std::make_shared<ed::symmetry::RepSectorData>(std::move(rd));
    p->build_perm_lut();
    p->build_buckets();
    return p;
}

}  // namespace ed::solvers
