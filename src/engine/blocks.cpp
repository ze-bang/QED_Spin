// =============================================================================
// src/engine/blocks.cpp -- the LittleGroupBlock handle over one (star, irrep) block
// Part of the little-group engine; see internal.h for the file map.
// =============================================================================

#include "internal.h"

namespace ed::solvers {

using namespace lg_detail;

LittleGroupBlock::LittleGroupBlock(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
LittleGroupBlock::~LittleGroupBlock() = default;
LittleGroupBlock::LittleGroupBlock(const LittleGroupBlock&) = default;
LittleGroupBlock& LittleGroupBlock::operator=(const LittleGroupBlock&) = default;
LittleGroupBlock::LittleGroupBlock(LittleGroupBlock&&) noexcept = default;
LittleGroupBlock& LittleGroupBlock::operator=(LittleGroupBlock&&) noexcept
    = default;

std::vector<std::complex<double>>
LittleGroupBlock::lift_to_rep(const std::complex<double>* v) const {
    if (!impl_->hk)       // a star solved on the group-sector path never builds its k-sector
        throw std::logic_error("lift_to_rep: this group-sector block has no momentum sector; "
                               "its vectors live in the group sector");
    const std::size_t nrep = impl_->hk->dim();
    if (impl_->gop)       // group-sector block: re-express in the k-sector (a subgroup of G_k), norm kept
        return lift_group_vector(impl_->gop->rep_data(), impl_->hk->rep_data(), v);
    if (!impl_->W) {   // plain block: block coords ARE the rep basis
        return std::vector<std::complex<double>>(v, v + nrep);
    }
    std::vector<std::complex<double>> u(nrep, {0.0, 0.0});
    const auto& cols = impl_->W->cols;
    for (std::size_t c = 0; c < cols.size(); ++c)
        for (const auto& [i, w] : cols[c])
            u[static_cast<std::size_t>(i)] += w * v[c];
    return u;
}

}  // namespace ed::solvers
