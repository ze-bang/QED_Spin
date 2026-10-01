#pragma once
// =============================================================================
// include/ed/matvec/reduced_symmetry_csr.h
//
// Reduced-CSR "connectivity skeleton" for the representative symmetry SpMV.
//
// The matrix-free rep walk (apply_terms_rep_symmetry_gather) regenerates, for
// EVERY connected state and EVERY matvec, the representative + projection via
// an O(|G|) group scan (index_and_projection). On a large sector that O(|G|)
// factor dominates and is repeated every Lanczos/FTLM/... iteration. This
// builds the reduced sparse matrix ONCE -- running the IDENTICAL row
// enumeration as the gather kernel, but RECORDING (col, value) instead of
// accumulating value*in[col] -- so every subsequent apply is a plain
// O(1)-per-nonzero CSR SpMV.
//
// Memory: O(dim * avg_nnz_per_row) per sector (the materialized reduced matrix) --
// the planner gates this lane vs the matrix-free rep walk by the probed budget, so
// the rep walk remains the at-scale (O(#reps) memory) fallback for the very large
// sectors where even the reduced matrix does not fit.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ed/matvec/term_kernels.h>   // conj_scalar, coerce_coeff, kOpSz, kOpPlus

namespace ed::matvec {

// Allocator whose value-less construct() default-initialises: resize() then leaves trivial elements untouched instead
// of zero-filling them on the calling thread. Linux places a page on the NUMA node of the thread that first writes it,
// so a serial zero fill would put a whole reduced CSR (tens to hundreds of GB) on ONE node and every spmv would stream
// it across the interconnect (measured on Fir: ~80 GB/s from one domain vs ~8x that interleaved). The builders below
// instead first-touch every row's slots from the thread that owns the row in spmv's static partition.
template <class T, class A = std::allocator<T>>
struct DefaultInitAllocator : A {
    using A::A;
    template <class U>
    struct rebind { using other = DefaultInitAllocator<U, typename std::allocator_traits<A>::template rebind_alloc<U>>; };
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new (static_cast<void*>(p)) U; }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) {
        std::allocator_traits<A>::construct(static_cast<A&>(*this), p, std::forward<Args>(args)...);
    }
};

template <class T>
using NumaVector = std::vector<T, DefaultInitAllocator<T>>;

template <class Scalar>
struct ReducedSymmetryCsr {
    NumaVector<std::uint64_t> row_ptr;    // size dim+1
    NumaVector<std::uint32_t> col_idx;    // size nnz
    NumaVector<Scalar>        val;        // size nnz
    std::uint64_t             dim = 0;

    // Size the arrays for the prefix-summed row_ptr and first-touch them row by row in spmv's static partition.
    void allocate_first_touch() {
        const std::uint64_t total = row_ptr[dim];
        col_idx.resize(total);
        val.resize(total);
        #pragma omp parallel for schedule(static) if(dim > (1ULL << 16))
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            const std::uint64_t r = static_cast<std::uint64_t>(ir);
            for (std::uint64_t e = row_ptr[r]; e < row_ptr[r + 1]; ++e) { col_idx[e] = 0; val[e] = Scalar(0); }
        }
    }

    [[nodiscard]] std::uint64_t nnz() const noexcept { return val.size(); }
    [[nodiscard]] bool          built() const noexcept { return dim > 0 && !row_ptr.empty(); }

    // out = A * in   (A is the FULL reduced sector matrix, diagonal included).
    inline void spmv(const Scalar* __restrict__ in, Scalar* __restrict__ out) const {
#ifdef _OPENMP
        const std::uint64_t par = static_cast<std::uint64_t>(omp_get_max_threads()) * 1024ULL;
#else
        const std::uint64_t par = std::numeric_limits<std::uint64_t>::max();
#endif
        #pragma omp parallel for schedule(static) if(dim > par)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            const std::uint64_t r = static_cast<std::uint64_t>(ir);
            Scalar acc = Scalar(0);
            const std::uint64_t e0 = row_ptr[r], e1 = row_ptr[r + 1];
            for (std::uint64_t e = e0; e < e1; ++e) acc += val[e] * in[col_idx[e]];
            out[r] = acc;
        }
    }
};

// ---------------------------------------------------------------------------
// build_reduced_symmetry_csr_rep: assemble the reduced sector matrix from the
// CSR-FREE representative policy (``RepSymmetryBasisPolicy``), so the default
// ``RepReducedCsr`` CPU lane never materializes a per-sector orbit CSR
// (O(dim_Sz) images + coefficients, ~24 GiB/sector at N=32).
//
// Element math mirrors ``apply_terms_rep_symmetry_gather`` verbatim (the
// single source of the rep-walk matrix element): row r applies H to the one
// representative ``rep_r = state_of(r)``; each connected computational
// state ``s'`` resolves to its source column ``j`` + projection ``proj``
// via ``index_and_projection``, contributing
//
//     A[r, j] += inv_norm[r] * conj(h(s') * proj(s'))
//
// (all six term bins enumerated, so the diagonal lands in the CSR
// naturally). CSR * v therefore equals the rep-walk gather to machine
// precision -- pinned by tests/unit/test_rep_symmetry_backend.cpp.
// ---------------------------------------------------------------------------
namespace detail {

// Row r of the reduced sector matrix: (column, value) pairs sorted by column, zeros dropped.
template <class RepPolicy, class Scalar,
          class D1, class O1, class D2, class M2, class O2, class T3>
inline void reduced_symmetry_row(const RepPolicy& basis, double spin_l,
                                 const D1& diag_one_body, const O1& offdiag_one_body,
                                 const D2& diag_two_body, const M2& mixed_two_body,
                                 const O2& offdiag_two_body, const T3& three_body,
                                 std::uint64_t r, std::vector<std::pair<std::uint32_t, Scalar>>& dstrow) {
    const std::uint64_t rep_r = basis.state_of(r);
    const Scalar inv_norm_r = kernel::coerce_coeff<Scalar>(
        std::complex<double>(basis.inv_norm_of(r), 0.0));
    std::unordered_map<std::uint32_t, Scalar> acc;
    kernel::apply_term_to_state<Scalar>(
        rep_r, spin_l,
        diag_one_body, offdiag_one_body, diag_two_body,
        mixed_two_body, offdiag_two_body, three_body,
        [&](std::uint64_t s_prime, const Scalar& h) {
            std::complex<double> proj;
            const std::int64_t j = basis.index_and_projection(s_prime, proj);
            if (j < 0) return;
            acc[static_cast<std::uint32_t>(j)] +=
                inv_norm_r * kernel::conj_scalar<Scalar>(
                    h * kernel::coerce_coeff<Scalar>(proj));
        });
    dstrow.clear();
    dstrow.reserve(acc.size());
    for (const auto& kv : acc)
        if (std::abs(kv.second) > 0.0) dstrow.emplace_back(kv.first, kv.second);
    std::sort(dstrow.begin(), dstrow.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
}

}  // namespace detail

// Mean row length of the reduced sector matrix from `samples` evenly spaced rows -- the
// real fill, for budgeting a build whose term-count bound (every term on every row) is
// several times too high.
template <class RepPolicy, class Scalar,
          class D1, class O1, class D2, class M2, class O2, class T3>
[[nodiscard]] inline double sampled_reduced_symmetry_row_length(
    const RepPolicy& basis, double spin_l,
    const D1& diag_one_body, const O1& offdiag_one_body,
    const D2& diag_two_body, const M2& mixed_two_body,
    const O2& offdiag_two_body, const T3& three_body, std::uint64_t samples = 4096)
{
    const std::uint64_t dim = basis.dim();
    if (dim == 0) return 0.0;
    const std::uint64_t n = std::min(samples, dim);
    std::uint64_t total = 0;
    #pragma omp parallel reduction(+ : total)
    {
        std::vector<std::pair<std::uint32_t, Scalar>> row;
        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const std::uint64_t r = static_cast<std::uint64_t>(i) * dim / n;
            detail::reduced_symmetry_row<RepPolicy, Scalar>(basis, spin_l, diag_one_body, offdiag_one_body,
                                                            diag_two_body, mixed_two_body, offdiag_two_body,
                                                            three_body, r, row);
            total += row.size();
        }
    }
    return static_cast<double>(total) / static_cast<double>(n);
}

template <class RepPolicy, class Scalar,
          class D1, class O1, class D2, class M2, class O2, class T3>
[[nodiscard]] inline ReducedSymmetryCsr<Scalar> build_reduced_symmetry_csr_rep(
    RepPolicy basis, double spin_l,
    const D1& diag_one_body, const O1& offdiag_one_body,
    const D2& diag_two_body, const M2& mixed_two_body,
    const O2& offdiag_two_body, const T3& three_body)
{
    const std::uint64_t dim = basis.dim();

    ReducedSymmetryCsr<Scalar> csr;
    csr.dim = dim;
    csr.row_ptr.assign(dim + 1, 0);

    // Two passes over the rows, both parallel: (1) count the nonzeros of every row, (2) after one allocation of the
    // final arrays, recompute each row and write it straight into its slot. Both passes run the same deterministic
    // row builder, so the counts match the writes exactly. Peak memory is the final CSR itself (~290 GB at N=36:
    // 3.8e8 rows, 1.5e10 nonzeros) rather than one heap vector per row plus a flatten, and nothing runs serially
    // except the prefix sum.
#ifdef _OPENMP
    const std::uint64_t par = static_cast<std::uint64_t>(omp_get_max_threads()) * 256ULL;
#else
    const std::uint64_t par = std::numeric_limits<std::uint64_t>::max();
#endif
    auto build_row = [&](std::uint64_t r, std::vector<std::pair<std::uint32_t, Scalar>>& dstrow) {
        detail::reduced_symmetry_row<RepPolicy, Scalar>(basis, spin_l, diag_one_body, offdiag_one_body,
                                                        diag_two_body, mixed_two_body, offdiag_two_body,
                                                        three_body, r, dstrow);
    };

    // pass 1: row lengths (row_ptr[r + 1] holds the length of row r until the prefix sum)
    #pragma omp parallel if(dim > par)
    {
        std::vector<std::pair<std::uint32_t, Scalar>> row;
        #pragma omp for schedule(dynamic, 256)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            build_row(static_cast<std::uint64_t>(ir), row);
            csr.row_ptr[static_cast<std::uint64_t>(ir) + 1] = row.size();
        }
    }
    for (std::uint64_t r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
    // Pages first-touched in spmv's static partition (one cheap bandwidth-bound sweep), so pass 2 keeps its dynamic
    // load balance without deciding where the memory lives.
    csr.allocate_first_touch();
    // pass 2: fill in place
    #pragma omp parallel if(dim > par)
    {
        std::vector<std::pair<std::uint32_t, Scalar>> row;
        #pragma omp for schedule(dynamic, 256)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            const std::uint64_t r = static_cast<std::uint64_t>(ir);
            build_row(r, row);
            std::uint64_t e = csr.row_ptr[r];
            for (const auto& cv : row) { csr.col_idx[e] = cv.first; csr.val[e] = cv.second; ++e; }
        }
    }
    return csr;
}

}  // namespace ed::matvec
