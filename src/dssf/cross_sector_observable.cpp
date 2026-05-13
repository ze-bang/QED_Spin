// =============================================================================
// src/dssf/cross_sector_observable.cpp
//
// Implementation of ed::dssf::CrossSectorObservable. See the header for
// the mathematical setup. The hot loop mirrors the matrix-free path of
// FixedSzOperator::apply but writes into a destination basis whose Lin
// O(1) lookup table is built for a *different* n_up sector. This unlocks
// the legitimate single-sector-lift transverse channels (S+/-, Sx/y) that
// the audit-#1-partial guard rail in src/cli/workflows.cpp currently
// drops to prevent silent zeros.
// =============================================================================

#include <ed/dssf/cross_sector_observable.h>

#include <ed/core/basis_utils.h>   // popcount

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::dssf {

namespace {
inline std::int64_t infer_n_up(const FixedSzOperator& op) {
    const auto& states = op.getBasisStates();
    if (states.empty()) {
        throw std::invalid_argument(
            "CrossSectorObservable: FixedSzOperator has empty basis");
    }
    return static_cast<std::int64_t>(popcount(states.front()));
}
}  // namespace

CrossSectorObservable::CrossSectorObservable(
    std::shared_ptr<FixedSzOperator> src,
    std::shared_ptr<FixedSzOperator> dst,
    std::vector<TransformData> transforms,
    float spin_length)
  : src_(std::move(src)),
    dst_(std::move(dst)),
    transforms_(std::move(transforms)),
    spin_l_(spin_length) {
    if (!src_ || !dst_) {
        throw std::invalid_argument(
            "CrossSectorObservable: src/dst must be non-null");
    }
    if (src_->getNumBits() != dst_->getNumBits()) {
        throw std::invalid_argument(
            "CrossSectorObservable: src/dst num_sites mismatch (" +
            std::to_string(src_->getNumBits()) + " vs " +
            std::to_string(dst_->getNumBits()) + ")");
    }
    if (transforms_.empty()) {
        throw std::invalid_argument(
            "CrossSectorObservable: transforms is empty");
    }
    n_bits_ = src_->getNumBits();
    n_up_src_ = infer_n_up(*src_);
    n_up_dst_ = infer_n_up(*dst_);
}

void CrossSectorObservable::apply(const Complex* in, Complex* out,
                                  std::size_t dst_size) const {
    const std::uint64_t dim_dst_ = dst_->getFixedSzDim();
    if (dst_size != static_cast<std::size_t>(dim_dst_)) {
        throw std::invalid_argument(
            "CrossSectorObservable::apply: dst_size mismatch (got " +
            std::to_string(dst_size) + ", expected " +
            std::to_string(dim_dst_) + ")");
    }

    const std::uint64_t dim_src_ = src_->getFixedSzDim();
    const auto& src_basis = src_->getBasisStates();
    const std::int64_t target_n_up = n_up_dst_;
    const Complex zero(0.0, 0.0);

    std::fill(out, out + dim_dst_, zero);

    const double S = static_cast<double>(spin_l_);
    const double S_sq = S * S;

    // Cache-blocked scatter pattern with atomic accumulation, mirroring
    // FixedSzOperator::apply but targeting the dst lookup table.
    constexpr std::uint64_t kBlock = 4096;
    const std::uint64_t num_blocks = (dim_src_ + kBlock - 1) / kBlock;

    #pragma omp parallel for schedule(dynamic, 1) if(dim_src_ > 8192)
    for (std::uint64_t blk = 0; blk < num_blocks; ++blk) {
        const std::uint64_t b0 = blk * kBlock;
        const std::uint64_t b1 = std::min<std::uint64_t>(b0 + kBlock, dim_src_);
        for (std::uint64_t i = b0; i < b1; ++i) {
            const Complex coeff = in[i];
            if (std::abs(coeff) < 1e-15) continue;
            const std::uint64_t basis_i = src_basis[i];

            for (const auto& t : transforms_) {
                std::uint64_t new_basis = basis_i;
                Complex scalar = t.coefficient;
                bool valid = true;

                if (!t.is_two_body) {
                    if (t.op_type == 2) {
                        // Sz: diagonal, delta_n_up = 0
                        const double sign =
                            ((basis_i >> t.site_index) & 1ULL) ? -1.0 : 1.0;
                        scalar *= S * sign;
                    } else {
                        // S+ (op=0) or S- (op=1): off-diagonal, delta_n_up = +-1
                        const std::uint64_t bit =
                            (basis_i >> t.site_index) & 1ULL;
                        if (bit != t.op_type) {
                            new_basis ^= (1ULL << t.site_index);
                        } else {
                            valid = false;
                        }
                    }
                } else {
                    // Two-body. Apply first then second op.
                    const std::uint64_t bit_1 =
                        (basis_i >> t.site_index) & 1ULL;
                    const std::uint64_t bit_2 =
                        (basis_i >> t.site_index_2) & 1ULL;

                    if (t.op_type == 2 && t.op_type_2 == 2) {
                        const double s1 = bit_1 ? -1.0 : 1.0;
                        const double s2 = bit_2 ? -1.0 : 1.0;
                        scalar *= S_sq * s1 * s2;
                    } else {
                        if (t.op_type != 2) {
                            if (bit_1 != t.op_type) {
                                new_basis ^= (1ULL << t.site_index);
                            } else {
                                valid = false;
                            }
                        } else {
                            const double s1 = bit_1 ? -1.0 : 1.0;
                            scalar *= S * s1;
                        }
                        if (valid && t.op_type_2 != 2) {
                            const std::uint64_t new_bit_j =
                                (new_basis >> t.site_index_2) & 1ULL;
                            if (new_bit_j != t.op_type_2) {
                                new_basis ^= (1ULL << t.site_index_2);
                            } else {
                                valid = false;
                            }
                        } else if (valid) {
                            const double s2 = bit_2 ? -1.0 : 1.0;
                            scalar *= S * s2;
                        }
                    }
                }

                if (!valid || std::abs(scalar) < 1e-15) continue;
                if (static_cast<std::int64_t>(popcount(new_basis)) != target_n_up) {
                    // Sector mismatch (e.g., a Sz-only term in a list with
                    // delta_n_up = +-1, or two ops adding to a different
                    // shift). Skip; this is the sole legitimate filter.
                    continue;
                }
                const std::int64_t j = dst_->lookupState(new_basis);
                if (j < 0) continue;

                const Complex contrib = scalar * coeff;
                double* out_ptr =
                    reinterpret_cast<double*>(&out[static_cast<std::size_t>(j)]);
                #pragma omp atomic
                out_ptr[0] += contrib.real();
                #pragma omp atomic
                out_ptr[1] += contrib.imag();
            }
        }
    }
}

}  // namespace ed::dssf
