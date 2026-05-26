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
#include <type_traits>

#include <ed/matvec/term_storage.h>

namespace ed::matvec::kernel {

/**
 * @brief GATHER one row's worth of H * v at output index r.
 *
 * Legacy (FullBasisPolicy-only) signature kept as a thin wrapper around
 * the generic ``gather_row<BasisPolicy>`` below: the bitstring r IS the
 * global array index in the full-Hilbert basis, so this overload simply
 * threads through ``c = r XOR pattern`` directly.
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

// ===========================================================================
// gather_row<BasisPolicy, Scalar=Complex>:
//
// Wave 2 of the "Unify all 16 matvec cells under apply_terms<BasisPolicy,
// Scalar, Backend>" plan (May 2026). Generalises the GATHER kernel so the
// CPU+MPI lane covers cells 1C, 2C, 3C, 4C uniformly:
//
//   * cell 1C (Full)         -- BasisPolicy = FullBasisPolicy
//   * cell 2C (FixedSz)      -- BasisPolicy = FixedSzBasisPolicy
//   * cell 3C (Symm)         -- BasisPolicy = SymmetryBasisPolicy
//   * cell 4C (FixedSz+Symm) -- BasisPolicy = SymmetryBasisPolicy
//                                (orbit walk via iter_orbit; symmetry
//                                policies plug coeff_modifier into the
//                                accumulator before emission)
//
// For symmetry policies, the caller must wrap the outer-row loop in
// ``basis.iter_orbit(r_idx, ...)`` and the closure must apply the
// symmetry weighting before / after this kernel runs -- the gather
// version stays simple (a single row, single state).
//
// @tparam BasisPolicy   compile-time basis description (see basis_policy.h)
// @tparam GetV          callable Complex(std::uint64_t c_global_idx)
//                       returning v[c]. Caller resolves local vs.
//                       off-rank lookup. The argument is a GLOBAL ARRAY
//                       INDEX (not a bitstring) because the gather
//                       kernel needs to drive the halo lookup table.
//
// @param r_idx          Global ARRAY-index of the output row.
// @param v_local_at_r   v[r_idx], passed by value (saves one get_v call
//                       per diagonal term).
// @param basis          BasisPolicy view -- bitstring <-> index mapping.
// @param terms          SoA term storage (caller already committed).
// @param spin_l         Spin length.
// @param get_v          Closure resolving v[c] for any visited column.
// @return               Accumulated <r|H|v> contribution.
//
// Off-diagonal terms read the row bitstring via ``basis.state_of``,
// XOR the flip pattern, and resolve the destination ARRAY-index via
// ``basis.index_of``. When ``index_of`` returns -1 the column lies
// outside the basis (Fixed-Sz popcount mismatch / symmetry off-orbit)
// and the term is skipped, exactly as in the SCATTER kernel.
//
// For trivial policies (Full / FixedSz) the additional ``index_of``
// call is the same O(1) (Full) / O(1) hash (FixedSz) the rest of the
// stack already pays.
// ===========================================================================
template <class BasisPolicy, class Scalar, class GetV>
inline Scalar
gather_row_basis(std::uint64_t r_idx,
                 Scalar v_local_at_r,
                 BasisPolicy basis,
                 const TermStorage& terms,
                 double spin_l,
                 GetV&& get_v) noexcept
{
    using Cplx = std::complex<double>;
    const double spin    = spin_l;
    const double spin_sq = spin * spin;

    Scalar acc = Scalar(0);

    auto to_scalar = [](const Cplx& c) -> Scalar {
        if constexpr (std::is_same_v<Scalar, Cplx>) {
            return c;
        } else {
            static_assert(std::is_same_v<Scalar, double>,
                          "gather_row_basis: Scalar must be Complex or double");
            return c.real();
        }
    };

    // Get the bitstring of the row. For FullBasisPolicy this is the
    // identity; for FixedSzBasisPolicy it reads basis_states[r_idx].
    const std::uint64_t r_state = basis.state_of(r_idx);

    // ---- Diagonal one-body (Sz) ---------------------------------------
    for (const auto& t : terms.diag_one_body) {
        const double sign = ((r_state >> t.site_index) & 1) ? -1.0 : 1.0;
        acc += to_scalar(t.coefficient) * (spin * sign) * v_local_at_r;
    }

    // ---- Off-diagonal one-body (S+/S-) --------------------------------
    for (const auto& t : terms.offdiag_one_body) {
        const std::uint64_t bit = (r_state >> t.site_index) & 1ULL;
        if (bit == t.op_type) continue;
        const std::uint64_t c_state = r_state ^ (1ULL << t.site_index);
        if constexpr (BasisPolicy::may_leave_basis) {
            const std::int64_t c_idx = basis.index_of(c_state);
            if (c_idx < 0) continue;
            acc += to_scalar(t.coefficient) * get_v(static_cast<std::uint64_t>(c_idx));
        } else {
            acc += to_scalar(t.coefficient) * get_v(c_state);
        }
    }

    // ---- Diagonal two-body (Sz Sz) ------------------------------------
    for (const auto& t : terms.diag_two_body) {
        const double si = ((r_state >> t.site_index_1) & 1) ? -1.0 : 1.0;
        const double sj = ((r_state >> t.site_index_2) & 1) ? -1.0 : 1.0;
        acc += to_scalar(t.coefficient) * (spin_sq * si * sj) * v_local_at_r;
    }

    // ---- Mixed two-body (Sz S+/-) -------------------------------------
    for (const auto& t : terms.mixed_two_body) {
        const std::uint64_t flip_bit = (r_state >> t.flip_site) & 1ULL;
        if (flip_bit == t.flip_op_type) continue;
        const std::uint64_t b_state = r_state ^ (1ULL << t.flip_site);
        const double sz_sign = ((b_state >> t.sz_site) & 1) ? -1.0 : 1.0;
        if constexpr (BasisPolicy::may_leave_basis) {
            const std::int64_t c_idx = basis.index_of(b_state);
            if (c_idx < 0) continue;
            acc += to_scalar(t.coefficient) * (spin * sz_sign)
                 * get_v(static_cast<std::uint64_t>(c_idx));
        } else {
            acc += to_scalar(t.coefficient) * (spin * sz_sign)
                 * get_v(b_state);
        }
    }

    // ---- Off-diagonal two-body (S+/- S+/-) ----------------------------
    for (const auto& t : terms.offdiag_two_body) {
        const std::uint64_t bit_1 = (r_state >> t.site_index_1) & 1ULL;
        const std::uint64_t bit_2 = (r_state >> t.site_index_2) & 1ULL;
        if (bit_1 != t.op_type_1 && bit_2 != t.op_type_2) {
            const std::uint64_t c_state =
                r_state ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
            if constexpr (BasisPolicy::may_leave_basis) {
                const std::int64_t c_idx = basis.index_of(c_state);
                if (c_idx < 0) continue;
                acc += to_scalar(t.coefficient)
                     * get_v(static_cast<std::uint64_t>(c_idx));
            } else {
                acc += to_scalar(t.coefficient) * get_v(c_state);
            }
        }
    }

    // ---- Three-body (full complex coupling) ---------------------------
    for (const auto& tdata : terms.three_body) {
        bool valid = true;
        std::uint64_t flip_xor = 0;
        if (tdata.op_type_1 != 2) flip_xor ^= (1ULL << tdata.site_index_1);
        if (tdata.op_type_2 != 2) flip_xor ^= (1ULL << tdata.site_index_2);
        if (tdata.op_type_3 != 2) flip_xor ^= (1ULL << tdata.site_index_3);
        const std::uint64_t b_state = r_state ^ flip_xor;

        std::uint64_t walking = b_state;
        Cplx scalar = tdata.coefficient;
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

        if (valid && walking == r_state && std::abs(scalar) > 1e-15) {
            if constexpr (BasisPolicy::may_leave_basis) {
                const std::int64_t c_idx = basis.index_of(b_state);
                if (c_idx >= 0) {
                    acc += to_scalar(scalar)
                         * get_v(static_cast<std::uint64_t>(c_idx));
                }
            } else {
                acc += to_scalar(scalar) * get_v(b_state);
            }
        }
    }

    return acc;
}

} // namespace ed::matvec::kernel
