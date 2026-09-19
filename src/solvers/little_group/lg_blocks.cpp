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
    return impl_->pop ? static_cast<ed::LinearOperator&>(*impl_->pop)
                      : static_cast<ed::LinearOperator&>(*impl_->hk);
}
const ed::symmetry::RepSectorData& LittleGroupBlock::rep_data() const noexcept {
    return impl_->hk->rep_data();
}
bool LittleGroupBlock::projected() const noexcept {
    return impl_->W != nullptr;
}
bool LittleGroupBlock::gpu_engaged() const noexcept {
    return impl_->hk != nullptr && impl_->hk->gpu_engaged();
}
std::vector<std::vector<std::complex<double>>>
LittleGroupBlock::degenerate_partners(
    const std::vector<std::complex<double>>& u_rep) const {
    std::vector<std::vector<std::complex<double>>> out;
    const int d = impl_->tag.irrep_dim;
    if (d <= 1 || !impl_->M || impl_->Dmats.empty()) return out;
    const auto& M = *impl_->M;
    const std::size_t n = u_rep.size();
    // cache M_p u once per element
    std::vector<std::vector<Complex>> Mu(M.size(),
                                         std::vector<Complex>(n, {0, 0}));
    for (std::size_t p = 0; p < M.size(); ++p)
        for (std::size_t i = 0; i < n; ++i)
            Mu[p][static_cast<std::size_t>(M[p].to[i])] =
                M[p].phase[i] * u_rep[i];
    const double pref = static_cast<double>(d)
                      / static_cast<double>(M.size());
    for (int j = 1; j < d; ++j) {
        std::vector<Complex> pj(n, Complex(0, 0));
        for (std::size_t p = 0; p < M.size(); ++p) {
            const Complex c = std::conj(
                impl_->Dmats[p][static_cast<std::size_t>(j) *
                                static_cast<std::size_t>(d)]);  // D_{j0}
            if (c == Complex(0, 0)) continue;
            for (std::size_t i = 0; i < n; ++i) pj[i] += c * Mu[p][i];
        }
        double n2 = 0.0;
        for (auto& c : pj) { c *= pref; }
        for (const auto& c : pj) n2 += std::norm(c);
        if (n2 < 1e-12)
            throw std::runtime_error(
                "degenerate_partners: shift projector annihilated the "
                "vector (row convention mismatch?)");
        const double inv = 1.0 / std::sqrt(n2);
        for (auto& c : pj) c *= inv;
        out.push_back(std::move(pj));
    }
    return out;
}

std::vector<std::complex<double>>
LittleGroupBlock::lift_to_rep(const std::complex<double>* v) const {
    const std::size_t nrep = impl_->hk->dim();
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
