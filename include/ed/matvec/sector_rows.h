// =============================================================================
// include/ed/matvec/sector_rows.h -- an operator's rows in a symmetry sector, from the row walk.
//
// Row r of O in the sector a rep policy describes (rep_symmetry_basis_policy.h) comes from
// the walk (row_walk.h) of the program of O^dagger from the representative s_r: each group
// that acts on s_r gives a target t and h = <t|O^dagger|s_r>, and the entry
//
//     O[r][j(t)] += inv_norm[r] * conj(h * proj(t)),     (j, proj) = index_and_projection(t),
//
// exact for any O. The diagonal group (t == s_r) needs no lookup: inv_norm[r] conj(proj(s_r))
// is 1 for every 1-dim irrep (the stabiliser sum of chi is |Stab| on a surviving orbit), so
// it adds conj(h) at column r. Several targets can share an orbit, so a row can name a column
// more than once: the CSR merges them in emission order, the gather simply accumulates.
// =============================================================================
#pragma once

#include <ed/matvec/reduced_csr.h>   // ReducedSymmetryCsr
#include <ed/ops/program.h>
#include <ed/ops/row_walk.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <utility>
#include <vector>

namespace ed::matvec {

using SectorComplex = std::complex<double>;

/// emit(j, value) for every entry of row r (columns may repeat; see the header comment).
template <class Policy, class Emit>
inline void for_each_sector_entry(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol,
                                  std::uint64_t r, Emit&& emit) {
    const std::uint64_t s = pol.state_of(r);
    const double w = pol.inv_norm_of(r);
    ed::ops::for_each_connection(P, s, [&](std::uint64_t t, const SectorComplex& h) {
        if (t == s) { emit(r, std::conj(h)); return; }
        SectorComplex proj;
        const std::int64_t j = pol.index_and_projection(t, proj);
        if (j < 0) return;                       // the target's orbit cancels in this sector
        emit(static_cast<std::uint64_t>(j), w * std::conj(h * proj));
    });
}

/// out = O in on the sector, row by row (no CSR).
template <class Policy>
inline void sector_gather(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol, std::uint64_t dim,
                          const SectorComplex* in, SectorComplex* out) {
    #pragma omp parallel for schedule(static)
    for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
        SectorComplex acc(0.0, 0.0);
        for_each_sector_entry(P, pol, static_cast<std::uint64_t>(ir),
                              [&](std::uint64_t j, const SectorComplex& v) { acc += v * in[j]; });
        out[ir] = acc;
    }
}

namespace detail {
// Row r's entries merged by column: sorted by column, equal columns summed in emission order,
// exact zeros dropped.
template <class Policy>
inline void sector_row(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol, std::uint64_t r,
                       std::vector<std::pair<std::uint64_t, SectorComplex>>& row) {
    row.clear();
    for_each_sector_entry(P, pol, r, [&](std::uint64_t j, const SectorComplex& v) { row.emplace_back(j, v); });
    std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::size_t out = 0;
    for (std::size_t i = 0; i < row.size();) {
        SectorComplex sum = row[i].second;
        std::size_t k = i + 1;
        for (; k < row.size() && row[k].first == row[i].first; ++k) sum += row[k].second;
        if (sum != SectorComplex(0.0, 0.0)) row[out++] = {row[i].first, sum};
        i = k;
    }
    row.resize(out);
}
}  // namespace detail

/// O on the sector as a CSR (columns ascending within a row). Two passes over the rows.
template <class Policy>
inline ReducedSymmetryCsr<SectorComplex> build_sector_csr(const ed::ops::ProgramView<SectorComplex>& P,
                                                          const Policy& pol, std::uint64_t dim) {
    ReducedSymmetryCsr<SectorComplex> csr;
    csr.dim = dim;
    csr.row_ptr.assign(dim + 1, 0);
    #pragma omp parallel
    {
        std::vector<std::pair<std::uint64_t, SectorComplex>> row;
        #pragma omp for schedule(dynamic, 256)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            detail::sector_row(P, pol, static_cast<std::uint64_t>(ir), row);
            csr.row_ptr[static_cast<std::size_t>(ir) + 1] = row.size();
        }
    }
    for (std::uint64_t r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
    csr.allocate_first_touch();
    #pragma omp parallel
    {
        std::vector<std::pair<std::uint64_t, SectorComplex>> row;
        #pragma omp for schedule(dynamic, 256)
        for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
            detail::sector_row(P, pol, static_cast<std::uint64_t>(ir), row);
            std::uint64_t e = csr.row_ptr[static_cast<std::size_t>(ir)];
            for (const auto& [j, v] : row) {
                csr.col_idx[e] = static_cast<std::uint32_t>(j);
                csr.val[e] = v;
                ++e;
            }
        }
    }
    return csr;
}

/// The mean number of stored entries per row over up to `samples` evenly spaced rows.
template <class Policy>
inline double sampled_sector_row_length(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol,
                                        std::uint64_t dim, std::uint64_t samples = 4096) {
    if (dim == 0) return 0.0;
    const std::uint64_t n = std::min(dim, samples);
    std::vector<std::pair<std::uint64_t, SectorComplex>> row;
    double total = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        detail::sector_row(P, pol, i * dim / n, row);
        total += static_cast<double>(row.size());
    }
    return total / static_cast<double>(n);
}

}  // namespace ed::matvec
