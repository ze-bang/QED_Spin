// =============================================================================
// src/engine/blocks.cpp -- lift_to_rep: a block vector in its momentum sector's rep basis
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "internal.h"

namespace ed::solvers::lg_detail {

std::vector<std::complex<double>> lift_to_rep(const BlockData& b, const std::complex<double>* v) {
    if (!b.hk)       // a star solved on the group-sector path never builds its k-sector
        throw std::logic_error("lift_to_rep: this group-sector block has no momentum sector; "
                               "its vectors live in the group sector");
    const std::size_t nrep = b.hk->dim();
    if (b.gop)       // group-sector block: re-express in the k-sector (a subgroup of G_k), norm kept
        return lift_group_vector(b.gop->rep_data(), b.hk->rep_data(), v);
    if (!b.W) {   // plain block: block coords ARE the rep basis
        return std::vector<std::complex<double>>(v, v + nrep);
    }
    std::vector<std::complex<double>> u(nrep, {0.0, 0.0});
    const auto& cols = b.W->cols;
    for (std::size_t c = 0; c < cols.size(); ++c)
        for (const auto& [i, w] : cols[c])
            u[static_cast<std::size_t>(i)] += w * v[c];
    return u;
}

}  // namespace ed::solvers::lg_detail
