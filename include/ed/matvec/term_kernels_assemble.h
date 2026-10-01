#pragma once
// =============================================================================
// include/ed/matvec/term_kernels_assemble.h
//
// CSR assembly kernel. The third form of the unified term-kernel decision tree:
//
//   SCATTER (term_kernels.h)         : y[c] += <c|H|b> v[b]
//   GATHER  (term_kernels_gather.h)  : y[r] += <r|H|c> v[c]
//   ASSEMBLE (this header)           : CSR rows of <r|H|c>
//
// All three share byte-identical op_type / bit-flip / popcount semantics
// (op_type encoding: 0=S+, 1=S-, 2=Sz). Every sparse-matrix assembly (the
// CpuMatVecBackend CSR cache) goes through this header, so the bit-flip
// semantics are shared by construction.
//
// BasisPolicy contract (identical to term_kernels.h):
//   * BasisPolicy::dim() -> uint64_t
//   * BasisPolicy::state_of(idx) -> uint64_t      (idx -> bitstring)
//   * BasisPolicy::index_of(state) -> int64_t     (bitstring -> idx, or -1)
//   * static constexpr bool BasisPolicy::may_leave_basis
//        If false (e.g. full Hilbert space), index_of(state) == state and
//        we skip the leave-basis check at compile time.
// =============================================================================

#include <Eigen/SparseCore>

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <ed/matvec/term_storage.h>
#include <ed/matvec/term_kernels_gather.h>  // for_each_row_state

#include <stdexcept>
#include <utility>

namespace ed::matvec::kernel {

// ===========================================================================
// Direct two-pass CSR assembly in GATHER form.
//
// The Eigen triplet route costs 24 B/nnz of temporaries plus a serial
// ``setFromTriplets`` sort (measured ~47 s at N = 24). Here the
// row kernel's gates are walked twice -- once to count each row's merged
// entries, once to fill -- so the peak memory is the final CSR and every
// stage is parallel over rows. Entry order and duplicate handling match
// ``setFromTriplets`` (columns ascending, duplicates summed), and the row
// math mirrors ``gather_row_terms_state`` so the CSR and matrix-free lanes
// compute the same <r|H|c>.
// ===========================================================================
template <class Scalar>
struct OwnedCsr {
    std::vector<std::int64_t>  row_ptr;   ///< dim + 1
    std::vector<std::uint32_t> col;       ///< nnz (dim < 2^32)
    std::vector<Scalar>        val;       ///< nnz
    [[nodiscard]] std::size_t nnz()   const noexcept { return val.size(); }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return row_ptr.size() * sizeof(std::int64_t)
             + col.size() * sizeof(std::uint32_t)
             + val.size() * sizeof(Scalar);
    }
    void clear() {
        std::vector<std::int64_t>().swap(row_ptr);
        std::vector<std::uint32_t>().swap(col);
        std::vector<Scalar>().swap(val);
    }
};

namespace detail {

template <class Scalar>
[[nodiscard]] inline Scalar coerce_to(const std::complex<double>& c) noexcept {
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) return c;
    else                                                         return c.real();
}

// Row entries of <r|H|.> for row bitstring r_state. Emits (col_idx, value)
// through ``sink``; diagonal first, then the off-diagonal bins in the same
// gate convention as ``gather_row_terms_state``.
template <class BasisPolicy, class Scalar, class Sink>
inline void row_entries_gather(std::uint64_t r_idx,
                               std::uint64_t r_state,
                               const BasisPolicy& basis,
                               double spin,
                               const std::vector<DiagOneBody>&    diag_one_body,
                               const std::vector<OffDiagOneBody>& offdiag_one_body,
                               const std::vector<DiagTwoBody>&    diag_two_body,
                               const std::vector<MixedTwoBody>&   mixed_two_body,
                               const std::vector<OffDiagTwoBody>& offdiag_two_body,
                               const std::vector<ThreeBodyTerm>&  three_body,
                               Sink&& sink)
{
    const double spin2 = spin * spin;
    auto resolve = [&basis](std::uint64_t c_state) -> std::int64_t {
        if constexpr (BasisPolicy::may_leave_basis) {
            return basis.index_of(c_state);
        } else {
            (void)basis;
            return static_cast<std::int64_t>(c_state);
        }
    };

    Scalar diag = Scalar(0);
    for (const auto& d : diag_one_body) {
        const double sign = ((r_state >> d.site_index) & 1) ? -1.0 : 1.0;
        diag += coerce_to<Scalar>(d.coefficient) * (spin * sign);
    }
    for (const auto& d : diag_two_body) {
        const double si = ((r_state >> d.site_index_1) & 1) ? -1.0 : 1.0;
        const double sj = ((r_state >> d.site_index_2) & 1) ? -1.0 : 1.0;
        diag += coerce_to<Scalar>(d.coefficient) * (spin2 * si * sj);
    }
    if (std::abs(diag) > 1e-15) sink(r_idx, diag);

    for (const auto& t : offdiag_one_body) {
        const std::uint64_t bit = (r_state >> t.site_index) & 1ULL;
        if (bit != t.op_type) continue;
        const std::int64_t c = resolve(r_state ^ (1ULL << t.site_index));
        if (c < 0) continue;
        const Scalar v = coerce_to<Scalar>(t.coefficient);
        if (std::abs(v) > 1e-15) sink(static_cast<std::uint64_t>(c), v);
    }
    for (const auto& t : mixed_two_body) {
        const std::uint64_t flip_bit = (r_state >> t.flip_site) & 1ULL;
        if (flip_bit != t.flip_op_type) continue;
        const std::uint64_t b_state = r_state ^ (1ULL << t.flip_site);
        const double sz_sign = ((b_state >> t.sz_site) & 1) ? -1.0 : 1.0;
        const std::int64_t c = resolve(b_state);
        if (c < 0) continue;
        const Scalar v = coerce_to<Scalar>(t.coefficient) * (spin * sz_sign);
        if (std::abs(v) > 1e-15) sink(static_cast<std::uint64_t>(c), v);
    }
    for (const auto& t : offdiag_two_body) {
        const std::uint64_t bit_1 = (r_state >> t.site_index_1) & 1ULL;
        const std::uint64_t bit_2 = (r_state >> t.site_index_2) & 1ULL;
        if (bit_1 != t.op_type_1 || bit_2 != t.op_type_2) continue;
        const std::int64_t c = resolve(
            r_state ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2));
        if (c < 0) continue;
        const Scalar v = coerce_to<Scalar>(t.coefficient);
        if (std::abs(v) > 1e-15) sink(static_cast<std::uint64_t>(c), v);
    }
    for (const auto& tdata : three_body) {
        bool valid = true;
        std::uint64_t flip_xor = 0;
        if (tdata.op_type_1 != 2) flip_xor ^= (1ULL << tdata.site_index_1);
        if (tdata.op_type_2 != 2) flip_xor ^= (1ULL << tdata.site_index_2);
        if (tdata.op_type_3 != 2) flip_xor ^= (1ULL << tdata.site_index_3);
        const std::uint64_t b_state = r_state ^ flip_xor;
        std::uint64_t walking = b_state;
        std::complex<double> scalar = tdata.coefficient;
        auto step = [&](std::uint8_t op_type, std::uint64_t site) {
            if (!valid) return;
            const std::uint64_t bit = (walking >> site) & 1ULL;
            if (op_type == 2) {
                scalar *= spin * (bit ? -1.0 : 1.0);
            } else if (bit != op_type) {
                walking ^= (1ULL << site);
            } else {
                valid = false;
            }
        };
        step(tdata.op_type_1, tdata.site_index_1);
        step(tdata.op_type_2, tdata.site_index_2);
        step(tdata.op_type_3, tdata.site_index_3);
        if (!(valid && walking == r_state && std::abs(scalar) > 1e-15)) continue;
        const std::int64_t c = resolve(b_state);
        if (c < 0) continue;
        sink(static_cast<std::uint64_t>(c), coerce_to<Scalar>(scalar));
    }
}

// Sort a row's (col, val) entries by column and sum duplicates in place.
// Returns the merged count. Insertion sort for the short rows of local
// Hamiltonians; std::sort beyond 32 entries (long-range / all-to-all).
template <class Scalar>
inline std::size_t sort_merge_row(std::pair<std::uint32_t, Scalar>* e,
                                  std::size_t n) noexcept
{
    if (n <= 32) {
        for (std::size_t i = 1; i < n; ++i) {
            const auto key = e[i];
            std::size_t j = i;
            while (j > 0 && e[j - 1].first > key.first) { e[j] = e[j - 1]; --j; }
            e[j] = key;
        }
    } else {
        std::sort(e, e + n, [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    std::size_t w = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (w > 0 && e[w - 1].first == e[i].first) e[w - 1].second += e[i].second;
        else e[w++] = e[i];
    }
    return w;
}

} // namespace detail

template <class BasisPolicy, class Scalar>
inline void build_csr_gather(const BasisPolicy& basis,
                             double spin_l,
                             const std::vector<DiagOneBody>&    diag_one_body,
                             const std::vector<OffDiagOneBody>& offdiag_one_body,
                             const std::vector<DiagTwoBody>&    diag_two_body,
                             const std::vector<MixedTwoBody>&   mixed_two_body,
                             const std::vector<OffDiagTwoBody>& offdiag_two_body,
                             const std::vector<ThreeBodyTerm>&  three_body,
                             OwnedCsr<Scalar>& csr)
{
    const std::uint64_t dim = basis.dim();
    if (dim > 0xFFFFFFFFull) {
        throw std::runtime_error(
            "build_csr_gather: dim > 2^32-1 does not fit 32-bit column indices");
    }
    const std::size_t cap = 1 + offdiag_one_body.size() + mixed_two_body.size()
                          + offdiag_two_body.size() + three_body.size();
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    std::vector<std::vector<std::pair<std::uint32_t, Scalar>>> scratch(
        static_cast<std::size_t>(nthreads));
    for (auto& s : scratch) s.resize(cap);
    auto collect = [&](std::uint64_t r, std::uint64_t r_state,
                       std::pair<std::uint32_t, Scalar>* e) -> std::size_t {
        std::size_t n = 0;
        detail::row_entries_gather<BasisPolicy, Scalar>(
            r, r_state, basis, spin_l,
            diag_one_body, offdiag_one_body, diag_two_body,
            mixed_two_body, offdiag_two_body, three_body,
            [&](std::uint64_t c, Scalar v) {
                e[n++] = std::pair<std::uint32_t, Scalar>(static_cast<std::uint32_t>(c), v);
            });
        return detail::sort_merge_row(e, n);
    };
    auto my_scratch = [&]() -> std::pair<std::uint32_t, Scalar>* {
#ifdef _OPENMP
        return scratch[static_cast<std::size_t>(omp_get_thread_num())].data();
#else
        return scratch[0].data();
#endif
    };

    // Pass 1: merged entry count per row.
    csr.row_ptr.assign(static_cast<std::size_t>(dim) + 1, 0);
    for_each_row_state(basis, [&](std::uint64_t r, std::uint64_t r_state) {
        csr.row_ptr[r + 1] = static_cast<std::int64_t>(collect(r, r_state, my_scratch()));
    });
    for (std::uint64_t r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
    const std::size_t nnz = static_cast<std::size_t>(csr.row_ptr[dim]);
    csr.col.resize(nnz);
    csr.val.resize(nnz);

    // Pass 2: fill.
    for_each_row_state(basis, [&](std::uint64_t r, std::uint64_t r_state) {
        auto* e = my_scratch();
        const std::size_t n = collect(r, r_state, e);
        const std::int64_t base = csr.row_ptr[r];
        for (std::size_t i = 0; i < n; ++i) {
            csr.col[static_cast<std::size_t>(base) + i] = e[i].first;
            csr.val[static_cast<std::size_t>(base) + i] = e[i].second;
        }
    });
}

} // namespace ed::matvec::kernel
