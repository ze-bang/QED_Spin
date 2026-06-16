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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#  include <omp.h>
#endif

#include <ed/matvec/term_storage.h>

namespace ed::matvec::kernel {

// Forward declaration: the duck-typed SoA GATHER row kernel that
// ``gather_row_basis`` (below) and ``apply_terms_gather`` both delegate
// to. Defined further down; declared here so ``gather_row_basis`` can
// forward to it without reordering the file.
template <class BasisPolicy, class Scalar,
          class DiagOneBodyVec, class OffDiagOneBodyVec, class DiagTwoBodyVec,
          class MixedTwoBodyVec, class OffDiagTwoBodyVec, class ThreeBodyVec,
          class GetV>
inline Scalar gather_row_terms(std::uint64_t r_idx,
                               Scalar v_local_at_r,
                               BasisPolicy basis,
                               double spin_l,
                               const DiagOneBodyVec&    diag_one_body,
                               const OffDiagOneBodyVec& offdiag_one_body,
                               const DiagTwoBodyVec&    diag_two_body,
                               const MixedTwoBodyVec&   mixed_two_body,
                               const OffDiagTwoBodyVec& offdiag_two_body,
                               const ThreeBodyVec&      three_body,
                               GetV&&                   get_v) noexcept;

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

    // ---- Off-diagonal one-body (S+/S-): c = r XOR (1<<i) ----
    //  GATHER existence: the term S^{op} maps column c -> row r, so the
    //  ROW bit after the operator acted equals ``op_type``
    //  (S+ (op 0) leaves the flipped site up=0; S- (op 1) leaves it
    //  down=1). The column bit is the complement -- exactly the SCATTER
    //  gate transposed. (The earlier ``!= op_type`` form computed the
    //  Hermitian-conjugate element; for real symmetric couplings it
    //  coincides, but it dropped/swapped asymmetric & complex weights.)
    for (const auto& t : terms.offdiag_one_body) {
        const std::uint64_t bit = (r >> t.site_index) & 1ULL;
        if (bit == t.op_type) {
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
    //  c = r XOR (1<<flip_site); GATHER existence: row flip bit == op.
    //  sz_sign evaluated at the column b (== r at sz_site since
    //  sz_site != flip_site).
    for (const auto& t : terms.mixed_two_body) {
        const std::uint64_t flip_bit = (r >> t.flip_site) & 1ULL;
        if (flip_bit == t.flip_op_type) {
            const std::uint64_t b = r ^ (1ULL << t.flip_site);
            const double sz_sign = ((b >> t.sz_site) & 1) ? -1.0 : 1.0;
            acc += t.coefficient * (spin * sz_sign) * get_v(b);
        }
    }

    // ---- Off-diagonal two-body (S+/- S+/-) ----
    //  c = r XOR (1<<i) XOR (1<<j); GATHER existence: both row bits == op.
    for (const auto& t : terms.offdiag_two_body) {
        const std::uint64_t bit_1 = (r >> t.site_index_1) & 1ULL;
        const std::uint64_t bit_2 = (r >> t.site_index_2) & 1ULL;
        if (bit_1 == t.op_type_1 && bit_2 == t.op_type_2) {
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
    // Thin delegator: ``gather_row_terms`` (below) is the single source of
    // truth for the GATHER row math. Passing the SoA bins of ``terms`` keeps
    // the MPI / distributed callers (TermStorage-based) bit-identical with the
    // shared-memory ``apply_terms_gather`` driver.
    return gather_row_terms<BasisPolicy, Scalar>(
        r_idx, v_local_at_r, basis, spin_l,
        terms.diag_one_body, terms.offdiag_one_body,
        terms.diag_two_body, terms.mixed_two_body, terms.offdiag_two_body,
        terms.three_body, std::forward<GetV>(get_v));
}

// ===========================================================================
// gather_row_terms<BasisPolicy, Scalar, ...SoA vecs>:
//
// The single source of truth for the GATHER direction, duck-typed on the six
// SoA term bins (same field-name contract as ``apply_terms`` in
// ``term_kernels.h``). ``gather_row_basis`` (TermStorage) and the OpenMP
// ``apply_terms_gather`` driver both delegate here so the bit-flip / Sz /
// popcount semantics cannot drift between the SCATTER and GATHER kernels or
// between the shared-memory and distributed lanes.
//
// Diagonal note: callers that precompute the Hamiltonian diagonal (the
// ``apply_terms_gather`` fast path) pass EMPTY ``diag_one_body`` /
// ``diag_two_body`` bins here and add ``diag[r] * v[r]`` themselves, so the
// per-row term loop only walks the off-diagonal bins.
// ===========================================================================
template <class BasisPolicy, class Scalar,
          class DiagOneBodyVec, class OffDiagOneBodyVec, class DiagTwoBodyVec,
          class MixedTwoBodyVec, class OffDiagTwoBodyVec, class ThreeBodyVec,
          class GetV>
inline Scalar
gather_row_terms(std::uint64_t r_idx,
                 Scalar v_local_at_r,
                 BasisPolicy basis,
                 double spin_l,
                 const DiagOneBodyVec&    diag_one_body,
                 const OffDiagOneBodyVec& offdiag_one_body,
                 const DiagTwoBodyVec&    diag_two_body,
                 const MixedTwoBodyVec&   mixed_two_body,
                 const OffDiagTwoBodyVec& offdiag_two_body,
                 const ThreeBodyVec&      three_body,
                 GetV&&                   get_v) noexcept
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
                          "gather_row_terms: Scalar must be Complex or double");
            return c.real();
        }
    };

    // Get the bitstring of the row. For FullBasisPolicy this is the
    // identity; for FixedSzBasisPolicy it reads basis_states[r_idx].
    const std::uint64_t r_state = basis.state_of(r_idx);

    // ---- Diagonal one-body (Sz) ---------------------------------------
    for (const auto& t : diag_one_body) {
        const double sign = ((r_state >> t.site_index) & 1) ? -1.0 : 1.0;
        acc += to_scalar(t.coefficient) * (spin * sign) * v_local_at_r;
    }

    // ---- Off-diagonal one-body (S+/S-) --------------------------------
    //  GATHER existence: the ROW bit after S^{op} acted equals op_type
    //  (transpose of the SCATTER column gate). The column is c = r ^ flip.
    for (const auto& t : offdiag_one_body) {
        const std::uint64_t bit = (r_state >> t.site_index) & 1ULL;
        if (bit != t.op_type) continue;
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
    for (const auto& t : diag_two_body) {
        const double si = ((r_state >> t.site_index_1) & 1) ? -1.0 : 1.0;
        const double sj = ((r_state >> t.site_index_2) & 1) ? -1.0 : 1.0;
        acc += to_scalar(t.coefficient) * (spin_sq * si * sj) * v_local_at_r;
    }

    // ---- Mixed two-body (Sz S+/-) -------------------------------------
    //  GATHER existence: row flip bit == flip_op_type (column = r ^ flip).
    for (const auto& t : mixed_two_body) {
        const std::uint64_t flip_bit = (r_state >> t.flip_site) & 1ULL;
        if (flip_bit != t.flip_op_type) continue;
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
    //  GATHER existence: both row bits == their op_type (column = r^flips).
    for (const auto& t : offdiag_two_body) {
        const std::uint64_t bit_1 = (r_state >> t.site_index_1) & 1ULL;
        const std::uint64_t bit_2 = (r_state >> t.site_index_2) & 1ULL;
        if (bit_1 == t.op_type_1 && bit_2 == t.op_type_2) {
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
    for (const auto& tdata : three_body) {
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

// ===========================================================================
// apply_terms_gather<BasisPolicy, Scalar, ...SoA vecs>:
//
// The SOTA lock-free, row-owned GATHER SpMV driver for the shared-memory CPU
// lane. ``out = H * in`` computed one output row at a time:
//
//   * Each thread owns a disjoint contiguous range of output rows
//     (``schedule(static)``), so writes to ``out[r]`` are contention-free --
//     NO atomics, NO radix sort, NO thread-local contribution buffer (the
//     three costs the SCATTER ``apply_terms`` pays).
//   * ``out`` is fully overwritten (every row in [0, dim) is assigned), so
//     the caller does NOT need to pre-zero it.
//
// Diagonal fast path: when ``diag_cache != nullptr`` the precomputed
// Hamiltonian diagonal is applied as ``out[r] = diag_cache[r] * in[r] + (off-
// diagonal gather)``; the ``diag_one_body`` / ``diag_two_body`` bins are then
// skipped inside ``gather_row_terms`` (we pass empty bins), removing the
// per-row Sz sign-product recomputation that dominates XXZ/Heisenberg.
//
// Equivalence with the SCATTER kernel (``apply_terms``) is pinned bit-for-bit
// by the matvec unit tests; the GATHER form is a pure transpose of the
// iteration and requires no Hermiticity assumption.
// ===========================================================================
template <
    class BasisPolicy,
    class Scalar,
    class DiagOneBodyVec,
    class OffDiagOneBodyVec,
    class DiagTwoBodyVec,
    class MixedTwoBodyVec,
    class OffDiagTwoBodyVec,
    class ThreeBodyVec>
inline void apply_terms_gather(
    BasisPolicy              basis,
    double                   spin_l,
    const DiagOneBodyVec&    diag_one_body,
    const OffDiagOneBodyVec& offdiag_one_body,
    const DiagTwoBodyVec&    diag_two_body,
    const MixedTwoBodyVec&   mixed_two_body,
    const OffDiagTwoBodyVec& offdiag_two_body,
    const ThreeBodyVec&      three_body,
    const Scalar* __restrict__ in,
    Scalar*       __restrict__ out,
    const Scalar* __restrict__ diag_cache = nullptr)
{
    const std::uint64_t dim = basis.dim();

    // When a precomputed diagonal is supplied, skip the diagonal bins inside
    // the row kernel by handing it empty bins (constructed once, hold no
    // allocation) and add ``diag_cache[r] * in[r]`` in the driver instead.
    const bool use_cache = (diag_cache != nullptr);
    const DiagOneBodyVec empty_d1{};
    const DiagTwoBodyVec empty_d2{};
    const DiagOneBodyVec& d1 = use_cache ? empty_d1 : diag_one_body;
    const DiagTwoBodyVec& d2 = use_cache ? empty_d2 : diag_two_body;

#ifdef _OPENMP
    const std::uint64_t par_threshold =
        static_cast<std::uint64_t>(omp_get_max_threads()) * 1024ULL;
#else
    const std::uint64_t par_threshold = std::numeric_limits<std::uint64_t>::max();
#endif

    auto get_v = [in](std::uint64_t c) noexcept -> Scalar { return in[c]; };

    #pragma omp parallel for schedule(static) if(dim > par_threshold)
    for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
        const std::uint64_t r = static_cast<std::uint64_t>(ir);
        const Scalar v_r = in[r];
        Scalar acc = gather_row_terms<BasisPolicy, Scalar>(
            r, v_r, basis, spin_l,
            d1, offdiag_one_body, d2, mixed_two_body, offdiag_two_body,
            three_body, get_v);
        if (use_cache) acc += diag_cache[r] * v_r;
        out[r] = acc;
    }
}

} // namespace ed::matvec::kernel
