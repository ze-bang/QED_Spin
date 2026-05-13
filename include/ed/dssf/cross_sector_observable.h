#pragma once
// =============================================================================
// include/ed/dssf/cross_sector_observable.h
//
// Audit item #1 (full): rectangular operator that maps between two
// fixed-Sz sectors of the same lattice. Used to compute the legitimate
// cross-sector spectral functions
//
//     S_{O1, O2}(omega) = -1/pi  Im  <0|  O1†  (omega + E0 - H + i eta)^{-1}  O2  |0>
//
// in the fixed-Sz workflow. Today the workflow's (Operator value type)
// route silently drops these contributions (see audit item #1 partial
// guard rail in src/cli/workflows.cpp); this header introduces the
// missing rectangular operator so a follow-on dispatch in the workflow
// can build the inner-sector Hamiltonian and route to a cross-sector
// continued-fraction kernel.
//
// Mathematical structure
// ----------------------
// Given a parent FixedSzOperator src_op at sector (N, n_up_src) and a
// target FixedSzOperator dst_op at sector (N, n_up_dst), with
// n_up_dst = n_up_src + delta_n_up, a single-site spin operator
//
//     T = sum_i  c_i * S^op_i           (op in {0, 1, 2 = Sz})
//
// induces a uniform delta_n_up shift PER op_type. NOTE on the bit
// convention used by Operator / FixedSzOperator: bit=1 carries the
// popcount, so n_up = popcount(basis), and op_type=0 (legacy "S+")
// flips bit=1 -> 0 -> *decreases* popcount by 1, while op_type=1
// (legacy "S-") flips bit=0 -> 1 and *increases* popcount by 1.
// Concretely:
//    op_type=0 -> delta_n_up = -1
//    op_type=1 -> delta_n_up = +1
//    op_type=2 -> delta_n_up =  0
// A linear combination of such terms must therefore have a uniform
// delta_n_up on every TransformData entry, otherwise the matrix
// elements vanish by sector orthogonality (these contributions are
// silently filtered out by the popcount check at apply() time).
//
// CrossSectorObservable holds:
//   - shared pointers to two FixedSzOperator instances (basis +
//     Lin index lookup tables);
//   - a vector of Operator::TransformData entries describing the
//     coefficients to apply (typically copied verbatim from a
//     FixedSzSumOperator / FixedSzSumOperatorXYZ / sublattice variant);
//   - a single delta_n_up scalar set by the constructor.
//
// apply(in, out) walks the src basis and, for each transform t, finds
// the destination basis state by bit-flipping (S+/S-) or returning the
// same state with sign (Sz). The destination index is looked up in
// dst_op via its Lin O(1) table; entries whose final popcount differs
// from n_up_dst are skipped (this should be impossible for a uniform-
// delta-n_up transform list, but the check preserves the same defensive
// pattern as FixedSzOperator::apply).
//
// Threading: the implementation uses an OpenMP-parallel loop over the
// source basis. Output writes go through atomic real/imag updates
// because two distinct (i, t) pairs can scatter to the same dst index;
// this is the same atomic pattern as FixedSzOperator::apply.
//
// Build cost: zero -- the constructor just stores the pointers and
// transform list. apply() is O(dim_src * |transforms|) per call, identical
// to the matrix-free FixedSzOperator::apply on the source sector.
// =============================================================================

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>

#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::dssf {

class CrossSectorObservable {
public:
    using Complex = std::complex<double>;
    using TransformData = Operator::TransformData;

    /**
     * @param src   FixedSzOperator providing the source basis (input vector).
     * @param dst   FixedSzOperator providing the destination basis (output).
     * @param transforms  TransformData entries (may be one-body S+/S-/Sz
     *              and/or two-body terms). For one-body the per-entry
     *              op_type must produce delta_n_up = (n_up_dst - n_up_src);
     *              for two-body the COMBINED op_type + op_type_2 shift must
     *              equal delta_n_up. Mixed-shift entries are silently
     *              filtered out by the popcount check at apply() time.
     * @param spin_length   Spin S (0.5 for spin-1/2). Used for Sz eigenvalues.
     *
     * Throws std::invalid_argument if src/dst are null, sit on different
     * site counts, or transforms is empty.
     */
    CrossSectorObservable(std::shared_ptr<FixedSzOperator> src,
                          std::shared_ptr<FixedSzOperator> dst,
                          std::vector<TransformData> transforms,
                          float spin_length);

    std::uint64_t dim_src() const { return src_->getFixedSzDim(); }
    std::uint64_t dim_dst() const { return dst_->getFixedSzDim(); }
    std::int64_t  n_up_src() const { return n_up_src_; }
    std::int64_t  n_up_dst() const { return n_up_dst_; }
    std::int64_t  delta_n_up() const { return n_up_dst_ - n_up_src_; }

    /**
     * Apply the rectangular operator to a source-sector vector.
     *
     * @param in   pointer to dim_src() complex doubles
     * @param out  pointer to dim_dst() complex doubles (zeroed by callee)
     * @param dst_size  must equal dim_dst() (defensive size check)
     */
    void apply(const Complex* in, Complex* out, std::size_t dst_size) const;

    /// Convenience std::function adapter for callers that pass operator
    /// applications as lambdas (the standard FTLM matvec API).
    auto as_apply_function() const {
        return [this](const Complex* in, Complex* out, std::size_t n) {
            this->apply(in, out, n);
        };
    }

    const std::shared_ptr<FixedSzOperator>& src_operator() const { return src_; }
    const std::shared_ptr<FixedSzOperator>& dst_operator() const { return dst_; }
    const std::vector<TransformData>& transforms() const { return transforms_; }

private:
    std::shared_ptr<FixedSzOperator> src_;
    std::shared_ptr<FixedSzOperator> dst_;
    std::vector<TransformData> transforms_;
    float spin_l_ = 0.5f;
    std::uint64_t n_bits_ = 0;
    std::int64_t  n_up_src_ = 0;
    std::int64_t  n_up_dst_ = 0;
};

} // namespace ed::dssf
