#pragma once
// =============================================================================
// include/ed/matvec/reduced_csr.h -- the CSR an operator's rows are assembled into once
// (sector_rows.h build_sector_csr, Operator's full-space lane), so every later apply is a
// plain O(1)-per-nonzero SpMV instead of a row walk with its O(|G|) lookups.
//
// Memory: O(dim * avg_nnz_per_row); the planner gates this against the walk by the probed
// budget, and the walk (O(#reps) memory) serves the sectors where it does not fit.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::matvec {

// Allocator whose value-less construct() default-initialises: resize() then leaves trivial elements untouched instead
// of zero-filling them on the calling thread. Linux places a page on the NUMA node of the thread that first writes it,
// so a serial zero fill would put a whole reduced CSR (tens to hundreds of GB) on ONE node and every spmv would stream
// it across the interconnect (measured on Fir: ~80 GB/s from one domain vs ~8x that interleaved). The builders below
// (allocate_first_touch) instead first-touch every row's slots from the thread that owns the row in spmv's static
// partition.
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

}  // namespace ed::matvec
