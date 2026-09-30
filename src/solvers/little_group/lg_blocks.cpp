// =============================================================================
// src/solvers/little_group/lg_blocks.cpp -- the LittleGroupBlock handle over one (star, irrep) block
// Part of the little-group engine; see lg_internal.h for the file map.
// =============================================================================

#include "lg_internal.h"

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

const LittleGroupBlockTag& LittleGroupBlock::tag() const noexcept {
    return impl_->tag;
}
ed::LinearOperator& LittleGroupBlock::op() const noexcept {
    if (impl_->gop) return static_cast<ed::LinearOperator&>(*impl_->gop);   // group-sector block
    return impl_->pop ? static_cast<ed::LinearOperator&>(*impl_->pop)
                      : static_cast<ed::LinearOperator&>(*impl_->hk);
}
const ed::symmetry::RepSectorData& LittleGroupBlock::rep_data() const noexcept {
    return impl_->hk->rep_data();
}
bool LittleGroupBlock::projected() const noexcept {
    return impl_->W != nullptr || impl_->gop != nullptr;
}
bool LittleGroupBlock::gpu_engaged() const noexcept {
    if (impl_->gop) return impl_->gop->gpu_engaged();
    return impl_->hk != nullptr && impl_->hk->gpu_engaged();
}
std::vector<std::complex<double>>
LittleGroupBlock::lift_to_rep(const std::complex<double>* v) const {
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
