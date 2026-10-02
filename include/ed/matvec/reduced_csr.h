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

#include <ed/core/numa_vector.h>

namespace ed::matvec {

// The CSR arrays are NumaVectors (core/numa_vector.h): the builders below (allocate_first_touch)
// first-touch every row's slots from the thread that owns the row in spmv's static partition.
using ed::core::DefaultInitAllocator;
using ed::core::NumaVector;

template <class Scalar>
struct ReducedSymmetryCsr {
    NumaVector<std::uint64_t> row_ptr;    // size dim+1
    NumaVector<std::uint32_t> col_idx;    // size nnz
    /// The values: in `val`, or -- when the matrix holds few distinct values, as a symmetry sector
    /// of a clean lattice does -- as ids into `dict` (uint8 up to 256 values, else uint16), the
    /// same bits: 5-6 bytes an entry instead of 20.
    NumaVector<Scalar>        val;        // size nnz, or empty
    NumaVector<std::uint8_t>  id8;        // size nnz, or empty
    NumaVector<std::uint16_t> id16;       // size nnz, or empty
    std::vector<Scalar>       dict;
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

    [[nodiscard]] std::uint64_t nnz() const noexcept { return col_idx.size(); }
    [[nodiscard]] bool          built() const noexcept { return dim > 0 && !row_ptr.empty(); }
    [[nodiscard]] bool          dictionary() const noexcept { return !dict.empty(); }
    /// Entry e's value, whichever way it is stored.
    [[nodiscard]] Scalar value(std::uint64_t e) const noexcept {
        return !id8.empty() ? dict[id8[e]] : !id16.empty() ? dict[id16[e]] : val[e];
    }
    [[nodiscard]] std::uint64_t bytes() const noexcept {
        return row_ptr.size() * sizeof(std::uint64_t) + col_idx.size() * sizeof(std::uint32_t)
               + val.size() * sizeof(Scalar) + id8.size() + id16.size() * sizeof(std::uint16_t)
               + dict.size() * sizeof(Scalar);
    }

    // out = A * in   (A is the FULL reduced sector matrix, diagonal included).
    inline void spmv(const Scalar* __restrict__ in, Scalar* __restrict__ out) const {
        if (!id8.empty()) spmv_with([this](std::uint64_t e) { return dict[id8[e]]; }, in, out);
        else if (!id16.empty()) spmv_with([this](std::uint64_t e) { return dict[id16[e]]; }, in, out);
        else spmv_with([this](std::uint64_t e) { return val[e]; }, in, out);
    }

private:
    template <class Value>
    inline void spmv_with(Value value, const Scalar* __restrict__ in, Scalar* __restrict__ out) const {
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
            for (std::uint64_t e = e0; e < e1; ++e) acc += value(e) * in[col_idx[e]];
            out[r] = acc;
        }
    }
};

}  // namespace ed::matvec
