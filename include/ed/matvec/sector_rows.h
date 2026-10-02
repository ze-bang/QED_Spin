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
//
// Between two sectors of one group (rows in sector R, columns in sector C) the same walk gives
// O[j][r] = <R;j|O|C;r> from the program of (O_lambda)^dagger compiled with R as the ket and C
// as the bra -- compile_program({O.dagger()}, R, C) -- each target looked up in C. The diagonal
// shortcut holds only when R and C are the same sector.
// =============================================================================
#pragma once

#include <ed/matvec/reduced_csr.h>   // ReducedSymmetryCsr
#include <ed/ops/program.h>
#include <ed/ops/row_walk.h>

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ed::matvec {

using SectorComplex = std::complex<double>;

/// emit(c, value) for every entry of row j of O from column sector `col` to row sector `row`
/// (`same`: they are one sector, which allows the diagonal shortcut). Columns may repeat.
template <class RowPolicy, class ColPolicy, class Emit>
inline void for_each_cross_entry(const ed::ops::ProgramView<SectorComplex>& P, const RowPolicy& row,
                                 const ColPolicy& col, bool same, std::uint64_t j, Emit&& emit) {
    const std::uint64_t s = row.state_of(j);
    const double w = row.inv_norm_of(j);
    ed::ops::for_each_connection(P, s, [&](std::uint64_t t, const SectorComplex& h) {
        if (same && t == s) { emit(j, std::conj(h)); return; }
        SectorComplex proj;
        const std::int64_t c = col.index_and_projection(t, proj);
        if (c < 0) return;                       // the target's orbit cancels in the column sector
        emit(static_cast<std::uint64_t>(c), w * std::conj(h * proj));
    });
}

/// emit(j, value) for every entry of row r within one sector (columns may repeat).
template <class Policy, class Emit>
inline void for_each_sector_entry(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol,
                                  std::uint64_t r, Emit&& emit) {
    for_each_cross_entry(P, pol, pol, true, r, std::forward<Emit>(emit));
}

/// The largest irrep dimension a sector may carry (P6.3: point groups need 3, little co-groups' projective irreps 4).
inline constexpr int kMaxIrrepDim = 8;

/// d > 1 (rep_sector.h, P6.3): emit(row_state, col_state, value) for every entry of the rows of
/// row-sector representative r (its rank states) from column sector `col` (`same`: one sector).
/// A target t of the walk gives the block conj(h) C_r^dag A(t) C_c, A(t) = sum_{g: g t = rep_c} D(g)^T;
/// the diagonal group (t == s_r) gives conj(h) I, as C_r^dag (sum_{s in Stab} D(s)^T) C_r = I.
template <class RowPolicy, class ColPolicy, class Emit>
inline void for_each_cross_block(const ed::ops::ProgramView<SectorComplex>& P, const RowPolicy& row,
                                 const ColPolicy& col, bool same, std::uint64_t r, Emit&& emit) {
    const int d = row.irrep_dim;
    const std::uint64_t s = row.state_of(r);
    const int rr = row.rank_of(r);
    const SectorComplex* Cr = row.C_of(r);
    const std::uint64_t o_r = row.first_state_of(r);
    SectorComplex A[kMaxIrrepDim * kMaxIrrepDim], T[kMaxIrrepDim * kMaxIrrepDim];
    ed::ops::for_each_connection(P, s, [&](std::uint64_t t, const SectorComplex& h) {
        const SectorComplex ch = std::conj(h);
        if (same && t == s) {
            for (int a = 0; a < rr; ++a) emit(o_r + static_cast<std::uint64_t>(a), o_r + static_cast<std::uint64_t>(a), ch);
            return;
        }
        const std::int64_t c = col.index_and_matrix(t, A);
        if (c < 0) return;                       // the target's orbit holds no state of the irrep
        const int rc = col.rank_of(static_cast<std::uint64_t>(c));
        const SectorComplex* Cc = col.C_of(static_cast<std::uint64_t>(c));
        const std::uint64_t o_c = col.first_state_of(static_cast<std::uint64_t>(c));
        for (int i = 0; i < d; ++i)              // T = A C_c (d x rc)
            for (int b = 0; b < rc; ++b) {
                SectorComplex acc(0.0, 0.0);
                for (int j = 0; j < d; ++j) acc += A[i * d + j] * Cc[j * d + b];
                T[i * d + b] = acc;
            }
        for (int a = 0; a < rr; ++a)             // conj(h) C_r^dag T (rr x rc)
            for (int b = 0; b < rc; ++b) {
                SectorComplex acc(0.0, 0.0);
                for (int i = 0; i < d; ++i) acc += std::conj(Cr[i * d + a]) * T[i * d + b];
                emit(o_r + static_cast<std::uint64_t>(a), o_c + static_cast<std::uint64_t>(b), ch * acc);
            }
    });
}

/// out = O in from the column sector to the row sector, row by row (no CSR); a d > 1 sector by
/// representative, its rank rows at once (`rows` counts the states).
template <class RowPolicy, class ColPolicy>
inline void cross_gather(const ed::ops::ProgramView<SectorComplex>& P, const RowPolicy& row, const ColPolicy& col,
                         bool same, std::uint64_t rows, const SectorComplex* in, SectorComplex* out) {
    if (row.irrep_dim > 1) {
        #pragma omp parallel for schedule(static)
        for (long long ir = 0; ir < static_cast<long long>(row.dim()); ++ir) {
            const std::uint64_t r = static_cast<std::uint64_t>(ir), o = row.first_state_of(r);
            SectorComplex acc[kMaxIrrepDim];
            for (int a = 0; a < kMaxIrrepDim; ++a) acc[a] = SectorComplex(0.0, 0.0);
            for_each_cross_block(P, row, col, same, r,
                                 [&](std::uint64_t i, std::uint64_t c, const SectorComplex& v) { acc[i - o] += v * in[c]; });
            for (int a = 0; a < row.rank_of(r); ++a) out[o + static_cast<std::uint64_t>(a)] = acc[a];
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (long long ir = 0; ir < static_cast<long long>(rows); ++ir) {
        SectorComplex acc(0.0, 0.0);
        for_each_cross_entry(P, row, col, same, static_cast<std::uint64_t>(ir),
                             [&](std::uint64_t c, const SectorComplex& v) { acc += v * in[c]; });
        out[ir] = acc;
    }
}

/// out = O in on one sector, row by row (no CSR).
template <class Policy>
inline void sector_gather(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol, std::uint64_t dim,
                          const SectorComplex* in, SectorComplex* out) {
    cross_gather(P, pol, pol, true, dim, in, out);
}

namespace detail {
using Row = std::vector<std::pair<std::uint64_t, SectorComplex>>;

// A row's entries merged by column: sorted by column, equal columns summed in emission order,
// exact zeros dropped.
inline void merge_row(Row& row) {
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

// Row j's entries, merged.
template <class RowPolicy, class ColPolicy>
inline void cross_row(const ed::ops::ProgramView<SectorComplex>& P, const RowPolicy& rowp, const ColPolicy& colp,
                      bool same, std::uint64_t j, Row& row) {
    row.clear();
    for_each_cross_entry(P, rowp, colp, same, j, [&](std::uint64_t c, const SectorComplex& v) { row.emplace_back(c, v); });
    merge_row(row);
}

// The rows of unit u, merged: row u of a d = 1 sector, or the rank rows of representative u of a
// d > 1 one (rep_sector.h).
template <class RowPolicy, class ColPolicy>
inline void unit_rows(const ed::ops::ProgramView<SectorComplex>& P, const RowPolicy& rowp, const ColPolicy& colp,
                      bool same, std::uint64_t u, std::vector<Row>& rows) {
    if (rowp.irrep_dim == 1) {
        rows.resize(1);
        cross_row(P, rowp, colp, same, u, rows[0]);
        return;
    }
    const std::uint64_t o = rowp.first_state_of(u);
    rows.resize(static_cast<std::size_t>(rowp.rank_of(u)));
    for (auto& row : rows) row.clear();
    for_each_cross_block(P, rowp, colp, same, u, [&](std::uint64_t i, std::uint64_t c, const SectorComplex& v) {
        rows[static_cast<std::size_t>(i - o)].emplace_back(c, v);
    });
    for (auto& row : rows) merge_row(row);
}
}  // namespace detail

/// The exact bits of a value, as a dictionary key.
struct ValueBits {
    std::uint64_t re, im;
    bool operator==(const ValueBits& o) const noexcept { return re == o.re && im == o.im; }
};
struct ValueBitsHash {
    std::size_t operator()(const ValueBits& k) const noexcept {
        return static_cast<std::size_t>(k.re * 0x9E3779B97F4A7C15ULL ^ (k.im + 0x7F4A7C159E3779B9ULL + (k.re << 6)));
    }
};
inline ValueBits bits_of(const SectorComplex& v) noexcept {
    ValueBits k{};
    const double re = v.real(), im = v.imag();
    std::memcpy(&k.re, &re, sizeof re);
    std::memcpy(&k.im, &im, sizeof im);
    return k;
}

/// The most distinct values a CSR keeps in a dictionary (uint16 ids).
inline constexpr std::size_t kCsrDictMax = 65536;

/// O from the column sector to the row sector as a CSR (columns ascending within a row), its values
/// in a dictionary (reduced_csr.h) when it holds at most kCsrDictMax distinct ones -- a symmetry
/// sector of a clean lattice holds hundreds. One pass over the rows: chunks of rows go to threads
/// dynamically, each keeping its merged rows as (column, chunk-local value id) with the chunk's own
/// values; the row lengths give row_ptr; the chunks' values merge into one dictionary in chunk
/// order; each chunk is copied into the first-touched arrays and freed. The build peaks near 14
/// bytes an entry. Too many values for one dictionary (disordered couplings) and the rows are
/// computed twice instead (count, then fill) into full values, which never holds more than the
/// CSR. Either way the entries are those of the two-pass build bit for bit. The rows come in units: a row
/// of a d = 1 sector, or the rank rows of one representative of a d > 1 sector (`dim` counts rows).
/// `max_full_bytes` caps the full-value fallback: when its exact size (20 bytes an entry, 8 a row)
/// exceeds it, nothing is built (dim 0) and the caller takes the walk.
template <class RowPolicy, class ColPolicy>
inline ReducedSymmetryCsr<SectorComplex> build_cross_csr(const ed::ops::ProgramView<SectorComplex>& P,
                                                         const RowPolicy& rowp, const ColPolicy& colp, bool same,
                                                         std::uint64_t dim,
                                                         std::uint64_t max_full_bytes = ~std::uint64_t{0}) {
    ReducedSymmetryCsr<SectorComplex> csr;
    csr.dim = dim;
    csr.row_ptr.assign(dim + 1, 0);
    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    const bool blocks = rowp.irrep_dim > 1;
    const std::uint64_t units = blocks ? rowp.dim() : dim;
    const auto row_of_unit = [&rowp, blocks, units, dim](std::uint64_t u) {
        return !blocks ? u : (u == units ? dim : rowp.first_state_of(u));
    };
    const std::uint64_t n_chunks =
        std::max<std::uint64_t>(1, std::min<std::uint64_t>(units, 64ull * static_cast<std::uint64_t>(T)));
    const std::uint64_t q = units / n_chunks, rem = units % n_chunks;
    const auto first_unit = [q, rem](std::uint64_t c) { return c * q + std::min(c, rem); };
    const auto first_row = [&](std::uint64_t c) { return row_of_unit(first_unit(c)); };
    struct Slab {
        std::vector<std::uint32_t> col;       // per entry: column,
        std::vector<std::uint16_t> id;        // and chunk-local value id (a chunk keeps <= kCsrDictMax)
        std::vector<SectorComplex> values;    // the chunk's distinct values, by local id
        std::unordered_map<ValueBits, std::uint32_t, ValueBitsHash> index;
    };
    std::vector<Slab> slabs(static_cast<std::size_t>(n_chunks));
    std::atomic<bool> overflow{false};
    #pragma omp parallel
    {
        std::vector<detail::Row> rows;
        #pragma omp for schedule(dynamic, 1)
        for (long long c = 0; c < static_cast<long long>(n_chunks); ++c) {
            if (overflow.load(std::memory_order_relaxed)) continue;
            Slab& slab = slabs[static_cast<std::size_t>(c)];
            for (std::uint64_t u = first_unit(static_cast<std::uint64_t>(c)); u < first_unit(static_cast<std::uint64_t>(c) + 1);
                 ++u) {
                detail::unit_rows(P, rowp, colp, same, u, rows);
                const std::uint64_t r0 = row_of_unit(u);
                bool full = false;
                for (std::size_t a = 0; a < rows.size() && !full; ++a) {
                    csr.row_ptr[r0 + a + 1] = rows[a].size();
                    for (const auto& [j, v] : rows[a]) {
                        const auto [it, fresh] = slab.index.try_emplace(bits_of(v), static_cast<std::uint32_t>(slab.values.size()));
                        if (fresh) {
                            if (slab.values.size() == kCsrDictMax) { full = true; break; }   // ids stay below 2^16
                            slab.values.push_back(v);
                        }
                        slab.col.push_back(static_cast<std::uint32_t>(j));
                        slab.id.push_back(static_cast<std::uint16_t>(it->second));
                    }
                }
                if (full) {
                    overflow.store(true, std::memory_order_relaxed);
                    break;
                }
            }
            slab.index = {};
        }
    }
    // One dictionary for the whole matrix, numbered in chunk order.
    std::vector<std::vector<std::uint32_t>> to_global(static_cast<std::size_t>(n_chunks));
    if (!overflow) {
        std::unordered_map<ValueBits, std::uint32_t, ValueBitsHash> global;
        for (std::size_t c = 0; c < slabs.size() && csr.dict.size() <= kCsrDictMax; ++c)
            for (const auto& v : slabs[c].values) {
                const auto [it, fresh] = global.try_emplace(bits_of(v), static_cast<std::uint32_t>(csr.dict.size()));
                if (fresh) csr.dict.push_back(v);
                to_global[c].push_back(it->second);
            }
    }
    if (overflow || csr.dict.size() > kCsrDictMax) {
        slabs.clear();
        slabs.shrink_to_fit();
        csr.dict.clear();
        #pragma omp parallel
        {
            std::vector<detail::Row> rows;
            #pragma omp for schedule(dynamic, 256)
            for (long long iu = 0; iu < static_cast<long long>(units); ++iu) {
                detail::unit_rows(P, rowp, colp, same, static_cast<std::uint64_t>(iu), rows);
                const std::uint64_t r0 = row_of_unit(static_cast<std::uint64_t>(iu));
                for (std::size_t a = 0; a < rows.size(); ++a) csr.row_ptr[r0 + a + 1] = rows[a].size();
            }
        }
        for (std::uint64_t r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
        const std::uint64_t full_bytes = csr.row_ptr[dim] * (sizeof(SectorComplex) + sizeof(std::uint32_t))
                                         + (dim + 1) * sizeof(std::uint64_t);
        if (full_bytes > max_full_bytes) return ReducedSymmetryCsr<SectorComplex>{};
        csr.allocate_first_touch();
        #pragma omp parallel
        {
            std::vector<detail::Row> rows;
            #pragma omp for schedule(dynamic, 256)
            for (long long iu = 0; iu < static_cast<long long>(units); ++iu) {
                detail::unit_rows(P, rowp, colp, same, static_cast<std::uint64_t>(iu), rows);
                const std::uint64_t r0 = row_of_unit(static_cast<std::uint64_t>(iu));
                for (std::size_t a = 0; a < rows.size(); ++a) {
                    std::uint64_t e = csr.row_ptr[r0 + a];
                    for (const auto& [j, v] : rows[a]) {
                        csr.col_idx[e] = static_cast<std::uint32_t>(j);
                        csr.val[e] = v;
                        ++e;
                    }
                }
            }
        }
        return csr;
    }
    for (std::uint64_t r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
    const std::uint64_t nnz = csr.row_ptr[dim];
    const bool narrow = csr.dict.size() <= 256;
    // The dictionary when it is the smaller form (a tiny sector stores its few values directly).
    const bool full = csr.dict.size() * sizeof(SectorComplex) + nnz * (narrow ? 1 : 2) >= nnz * sizeof(SectorComplex);
    if (full) csr.dict.clear();
    // Allocated, not touched (NumaVector default-initialises): the copy below is the first touch,
    // so a page is committed only as the slabs that fill it are released -- the slabs (6 B an
    // entry) and the CSR (5-6 B) are never resident together (a 1.2e9-entry sector: 14 GB at the
    // peak when the CSR was touched first). The static chunk schedule puts each chunk's pages on
    // the thread that owns its rows in spmv's static partition.
    csr.col_idx.resize(nnz);
    if (full)        csr.val.resize(nnz);
    else if (narrow) csr.id8.resize(nnz);
    else             csr.id16.resize(nnz);
    #pragma omp parallel for schedule(static)
    for (long long c = 0; c < static_cast<long long>(n_chunks); ++c) {
        Slab& slab = slabs[static_cast<std::size_t>(c)];
        const auto& map = to_global[static_cast<std::size_t>(c)];
        const std::uint64_t e0 = csr.row_ptr[first_row(static_cast<std::uint64_t>(c))];
        std::copy(slab.col.begin(), slab.col.end(), csr.col_idx.begin() + static_cast<std::ptrdiff_t>(e0));
        for (std::size_t i = 0; i < slab.id.size(); ++i) {
            if (full)        csr.val[e0 + i] = slab.values[slab.id[i]];
            else if (narrow) csr.id8[e0 + i] = static_cast<std::uint8_t>(map[slab.id[i]]);
            else             csr.id16[e0 + i] = static_cast<std::uint16_t>(map[slab.id[i]]);
        }
        slab = Slab{};
    }
    return csr;
}

/// O on one sector as a CSR (max_full_bytes: build_cross_csr).
template <class Policy>
inline ReducedSymmetryCsr<SectorComplex> build_sector_csr(const ed::ops::ProgramView<SectorComplex>& P,
                                                          const Policy& pol, std::uint64_t dim,
                                                          std::uint64_t max_full_bytes = ~std::uint64_t{0}) {
    return build_cross_csr(P, pol, pol, true, dim, max_full_bytes);
}

/// The mean number of stored entries per row over up to `samples` evenly spaced rows.
template <class Policy>
inline double sampled_sector_row_length(const ed::ops::ProgramView<SectorComplex>& P, const Policy& pol,
                                        std::uint64_t dim, std::uint64_t samples = 4096) {
    if (dim == 0) return 0.0;
    if (pol.irrep_dim > 1) {   // by representative: its rows over its states (dim counts states)
        const std::uint64_t reps = pol.dim(), n = std::min(reps, samples);
        std::vector<detail::Row> rows;
        double total = 0.0, states = 0.0;
        for (std::uint64_t i = 0; i < n; ++i) {
            detail::unit_rows(P, pol, pol, true, i * reps / n, rows);
            for (const auto& r : rows) total += static_cast<double>(r.size());
            states += static_cast<double>(rows.size());
        }
        return states > 0.0 ? total / states : 0.0;
    }
    const std::uint64_t n = std::min(dim, samples);
    std::vector<std::pair<std::uint64_t, SectorComplex>> row;
    double total = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        detail::cross_row(P, pol, pol, true, i * dim / n, row);
        total += static_cast<double>(row.size());
    }
    return total / static_cast<double>(n);
}

}  // namespace ed::matvec
