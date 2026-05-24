#pragma once
// =============================================================================
// include/ed/matvec/term_kernels_gather.h
//
// GATHER-form term kernel. The dual of ``term_kernels.h``'s SCATTER form.
//
//   SCATTER: for each basis row b in [0, dim), for each term, find the
//            output basis c and ADD <c|H|b> v[b] to y[c].
//
//   GATHER : for each output row r in [0, dim), for each term, find the
//            input basis c (= r XOR flip-pattern) and ACCUMULATE
//            <r|H|c> v[c] into y[r].
//
// GATHER is the natural shape when the output row r is "owned" by the
// current process / thread / GPU block and the input vector v is accessed
// via a generic callable (e.g. MPI lookup table, NCCL get_v, page-fault-
// triggered remote read). It keeps writes scalar (one accumulator per r)
// at the cost of more conditional reads.
//
// The bit-flip / Sz / popcount semantics MUST stay byte-identical with
// the SCATTER kernel in ``term_kernels.h`` -- this header is the single
// source of truth for the GATHER direction, and it's tested against the
// SCATTER path via the distributed-equivalence unit tests.
//
// Op-type encoding (unchanged across both kernels):
//   0 = S+ (raise), 1 = S- (lower), 2 = Sz (diagonal)
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>

#include <ed/matvec/term_storage.h>

namespace ed::matvec::kernel {

/**
 * @brief GATHER one row's worth of H * v at output index r.
 *
 * @tparam GetV     Callable ``Complex(std::uint64_t c)`` returning v[c].
 *                  The caller resolves local vs. off-rank lookup.
 * @param r              Global output basis index.
 * @param v_local_at_r   v[r] (passed by value for diagonal-only terms,
 *                       saving one GetV call per diag term).
 * @param terms          Source-of-truth SoA term storage (caller has
 *                       already called ``commitPendingTransforms()`` on
 *                       the owning Operator).
 * @param spin_l         Spin length (1/2 for spin-1/2).
 * @param get_v          Closure resolving v[c] for any visited column.
 * @return              Accumulated <r|H|v> contribution.
 *
 * Complexity: O(|terms_in_each_bin|) -- one pass per SoA bin.
 *
 * The three-body branch now mirrors the complex SCATTER path in
 * ``term_kernels.h`` byte-for-byte (the imaginary part of the coupling
 * is no longer dropped). The distributed CPU SpMV reaches this kernel
 * via ``DistributedOperator::apply``; complex three-body couplings now
 * give the same result as the serial CPU path.
 */
template <class GetV>
inline std::complex<double>
gather_row(std::uint64_t r,
           std::complex<double> v_local_at_r,
           const TermStorage& terms,
           double spin_l,
           GetV&& get_v) noexcept
{
    using Complex = std::complex<double>;
    const double spin    = spin_l;
    const double spin_sq = spin * spin;

    Complex acc(0.0, 0.0);

    // ---- Diagonal one-body (Sz): GATHER(r) = always, c = r ----
    for (const auto& t : terms.diag_one_body) {
        const double sign = ((r >> t.site_index) & 1) ? -1.0 : 1.0;
        acc += t.coefficient * (spin * sign) * v_local_at_r;
    }

    // ---- Off-diagonal one-body (S+/S-): c = r XOR (1<<i),
    //      GATHER condition flips: ((r>>i)&1) != op_type ----
    for (const auto& t : terms.offdiag_one_body) {
        const std::uint64_t bit = (r >> t.site_index) & 1ULL;
        if (bit != t.op_type) {
            const std::uint64_t c = r ^ (1ULL << t.site_index);
            acc += t.coefficient * get_v(c);
        }
    }

    // ---- Diagonal two-body (Sz Sz): GATHER(r) = always, c = r ----
    for (const auto& t : terms.diag_two_body) {
        const double si = ((r >> t.site_index_1) & 1) ? -1.0 : 1.0;
        const double sj = ((r >> t.site_index_2) & 1) ? -1.0 : 1.0;
        acc += t.coefficient * (spin_sq * si * sj) * v_local_at_r;
    }

    // ---- Mixed two-body (Sz at sz_site, S+/- at flip_site) ----
    //  c = r XOR (1<<flip_site); GATHER: ((r>>flip)&1) != flip_op_type
    //  sz_sign evaluated at b = c (since sz_site != flip_site by
    //  addInteractAll invariant). Note: b at sz_site == r at sz_site
    //  whenever sz_site != flip_site, so we read from r directly when
    //  that invariant holds; here we conservatively use b.
    for (const auto& t : terms.mixed_two_body) {
        const std::uint64_t flip_bit = (r >> t.flip_site) & 1ULL;
        if (flip_bit != t.flip_op_type) {
            const std::uint64_t b = r ^ (1ULL << t.flip_site);
            const double sz_sign = ((b >> t.sz_site) & 1) ? -1.0 : 1.0;
            acc += t.coefficient * (spin * sz_sign) * get_v(b);
        }
    }

    // ---- Off-diagonal two-body (S+/- S+/-) ----
    //  c = r XOR (1<<i) XOR (1<<j)
    //  GATHER: both bits-on-r conditions flipped
    for (const auto& t : terms.offdiag_two_body) {
        const std::uint64_t bit_1 = (r >> t.site_index_1) & 1ULL;
        const std::uint64_t bit_2 = (r >> t.site_index_2) & 1ULL;
        if (bit_1 != t.op_type_1 && bit_2 != t.op_type_2) {
            const std::uint64_t c =
                r ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
            acc += t.coefficient * get_v(c);
        }
    }

    // ---- Three-body (full complex coupling, mirrors SCATTER) ---------
    // Reconstruct the source basis b = r XOR flip_xor where flip_xor is
    // the XOR of (1<<site_k) for each S+/- operator. Then walk the same
    // gating sequence the SCATTER kernel uses (with `b` as the source);
    // we require the final walked state to equal r (self-consistency).
    // ``scalar`` starts as the full complex coupling -- the previous
    // version of this kernel projected to .real() which silently dropped
    // the imaginary part on every distributed run with complex couplings.
    for (const auto& tdata : terms.three_body) {
        bool valid = true;
        std::uint64_t flip_xor = 0;
        if (tdata.op_type_1 != 2) flip_xor ^= (1ULL << tdata.site_index_1);
        if (tdata.op_type_2 != 2) flip_xor ^= (1ULL << tdata.site_index_2);
        if (tdata.op_type_3 != 2) flip_xor ^= (1ULL << tdata.site_index_3);
        const std::uint64_t b = r ^ flip_xor;

        std::uint64_t walking = b;
        Complex scalar = tdata.coefficient;
        auto step = [&](std::uint8_t op_type, std::uint64_t site) {
            if (!valid) return;
            if (op_type == 2) {
                const std::uint64_t bit = (walking >> site) & 1ULL;
                scalar *= spin * (bit ? -1.0 : 1.0);
            } else {
                const std::uint64_t bit = (walking >> site) & 1ULL;
                if (bit != op_type) {
                    walking ^= (1ULL << site);
                } else {
                    valid = false;
                }
            }
        };
        step(tdata.op_type_1, tdata.site_index_1);
        step(tdata.op_type_2, tdata.site_index_2);
        step(tdata.op_type_3, tdata.site_index_3);

        if (valid && walking == r && std::abs(scalar) > 1e-15) {
            acc += scalar * get_v(b);
        }
    }

    return acc;
}

} // namespace ed::matvec::kernel
