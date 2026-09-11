#pragma once
// =============================================================================
// include/ed/matvec/term_kernels_assemble.h
//
// CSR-triplet emission kernel. The third (and final) form of the unified
// term-kernel decision tree:
//
//   SCATTER (term_kernels.h)         : y[c] += <c|H|b> v[b]
//   GATHER  (term_kernels_gather.h)  : y[r] += <r|H|c> v[c]
//   ASSEMBLE (this header)           : triplets.emplace(c, b, <c|H|b>)
//
// All three share byte-identical op_type / bit-flip / popcount semantics
// (op_type encoding: 0=S+, 1=S-, 2=Sz). The SCATTER kernel is the canonical
// reference; this header replicates that decision tree but emits sparse
// matrix triplets instead of a dense write to y[].
//
// Historically the assembly logic was duplicated three times: once inside
// ed::matvec::CpuMatVecBackend (private ``emit_triplets_``), once in
// ed::Operator::buildSparseMatrix (legacy std::function-only path, which
// silently produced INCOMPLETE matrices when callers populated the typed
// AoS via addOneBodyTerm/addTwoBodyTerm — a latent correctness bug), and
// once in ed::FixedSzOperator::buildFixedSzMatrix (via unit-vector
// probing). Factoring the assembly out here makes the bit-flip semantics
// shared by construction.
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
#include <ed/matvec/term_kernels_gather.h>  // for_each_row_state (audit F1/F4)

#include <stdexcept>
#include <utility>

namespace ed::matvec::kernel {

namespace detail {

template <class Scalar>
[[nodiscard]] inline Scalar coerce_to(const std::complex<double>& c) noexcept {
    if constexpr (std::is_same_v<Scalar, std::complex<double>>) return c;
    else                                                         return c.real();
}

} // namespace detail

/**
 * @brief Walk the SoA term bins and emit Eigen sparse-matrix triplets.
 *
 * Triplets are emitted as ``(row, col, value)`` with
 * ``row = index_of(<c|)``, ``col = index_of(|b>)``, ``value = <c|H|b>``.
 * The walk loops over basis indices (cols); for each column, every term
 * is applied and any non-zero matrix element produces a triplet. Output
 * order is unspecified.
 *
 * Parallelism: OpenMP-parallelised over the basis index. Thread-local
 * triplet vectors are merged at the end.
 *
 * @tparam BasisPolicy   Compile-time basis (full Hilbert / fixed-Sz / ...).
 * @tparam Scalar        Triplet value type. ``Complex`` for general
 *                       Hamiltonians, ``double`` for real ones (caller is
 *                       responsible for verifying ``TermStorage::is_real()``).
 *
 * The signature mirrors ``ed::matvec::kernel::apply_terms`` so callers
 * holding a ``term_view`` (six pointers to vectors) can plug straight in
 * without constructing an owning ``TermStorage``.
 */
template <class BasisPolicy, class Scalar>
inline void emit_term_triplets(const BasisPolicy& basis,
                               double spin_l,
                               const std::vector<DiagOneBody>&    diag_one_body,
                               const std::vector<OffDiagOneBody>& offdiag_one_body,
                               const std::vector<DiagTwoBody>&    diag_two_body,
                               const std::vector<MixedTwoBody>&   mixed_two_body,
                               const std::vector<OffDiagTwoBody>& offdiag_two_body,
                               const std::vector<ThreeBodyTerm>&  three_body,
                               std::vector<Eigen::Triplet<Scalar>>& triplets)
{
    const std::uint64_t dim   = basis.dim();
    const double        spin  = spin_l;
    const double        spin2 = spin * spin;

    // Per-row term count (used for a generous triplet-reserve heuristic).
    const std::size_t terms_per_row =
          diag_one_body.size()
        + offdiag_one_body.size()
        + diag_two_body.size()
        + 2 * mixed_two_body.size()
        + offdiag_two_body.size()
        + three_body.size();

#ifdef _OPENMP
    const int num_threads = omp_get_max_threads();
#else
    const int num_threads = 1;
#endif
    std::vector<std::vector<Eigen::Triplet<Scalar>>> tls(num_threads);
    for (auto& v : tls) {
        v.reserve(static_cast<std::size_t>(dim) / num_threads
                  * std::max<std::size_t>(1, terms_per_row));
    }

    // ------------------------------------------------------------------
    // emit helpers (mirror CpuMatVecBackend's private emit_ /
    // emit_if_in_basis_).
    // ------------------------------------------------------------------
    auto emit_diag = [](auto& local, std::uint64_t i, Scalar value) {
        if (std::abs(value) > 1e-15) {
            local.emplace_back(static_cast<int>(i), static_cast<int>(i), value);
        }
    };

    auto emit_offdiag = [&basis](auto& local, std::uint64_t new_state,
                                 int col, Scalar value) {
        if constexpr (BasisPolicy::may_leave_basis) {
            const std::int64_t j = basis.index_of(new_state);
            if (j >= 0 && std::abs(value) > 1e-15) {
                local.emplace_back(static_cast<int>(j), col, value);
            }
        } else {
            (void)basis;
            if (std::abs(value) > 1e-15) {
                local.emplace_back(static_cast<int>(new_state), col, value);
            }
        }
    };

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& local = tls[tid];

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (long long ii = 0; ii < static_cast<long long>(dim); ++ii) {
            const std::uint64_t i      = static_cast<std::uint64_t>(ii);
            const std::uint64_t basis_state = basis.state_of(i);

            // --- diag one-body (Sz) ---
            for (const auto& d : diag_one_body) {
                const double sign = ((basis_state >> d.site_index) & 1) ? -1.0 : 1.0;
                emit_diag(local, i,
                          detail::coerce_to<Scalar>(d.coefficient) * (spin * sign));
            }

            // --- off-diag one-body (S+ / S-) ---
            for (const auto& o : offdiag_one_body) {
                const std::uint64_t bit = (basis_state >> o.site_index) & 1ULL;
                if (bit != o.op_type) {
                    const std::uint64_t new_basis = basis_state ^ (1ULL << o.site_index);
                    emit_offdiag(local, new_basis, static_cast<int>(i),
                                 detail::coerce_to<Scalar>(o.coefficient));
                }
            }

            // --- diag two-body (Sz Sz) ---
            for (const auto& d : diag_two_body) {
                const double sign_i = ((basis_state >> d.site_index_1) & 1) ? -1.0 : 1.0;
                const double sign_j = ((basis_state >> d.site_index_2) & 1) ? -1.0 : 1.0;
                emit_diag(local, i,
                          detail::coerce_to<Scalar>(d.coefficient) * (spin2 * sign_i * sign_j));
            }

            // --- mixed two-body (Sz * S+/-) ---
            for (const auto& m : mixed_two_body) {
                const std::uint64_t flip_bit = (basis_state >> m.flip_site) & 1ULL;
                if (flip_bit != m.flip_op_type) {
                    const double sz_sign =
                        ((basis_state >> m.sz_site) & 1) ? -1.0 : 1.0;
                    const std::uint64_t new_basis =
                        basis_state ^ (1ULL << m.flip_site);
                    emit_offdiag(local, new_basis, static_cast<int>(i),
                                 detail::coerce_to<Scalar>(m.coefficient)
                                     * (spin * sz_sign));
                }
            }

            // --- off-diag two-body (S+/- S+/-) ---
            for (const auto& o : offdiag_two_body) {
                const std::uint64_t bit_1 = (basis_state >> o.site_index_1) & 1ULL;
                const std::uint64_t bit_2 = (basis_state >> o.site_index_2) & 1ULL;
                if (bit_1 != o.op_type_1 && bit_2 != o.op_type_2) {
                    const std::uint64_t new_basis =
                        basis_state ^ (1ULL << o.site_index_1)
                                    ^ (1ULL << o.site_index_2);
                    emit_offdiag(local, new_basis, static_cast<int>(i),
                                 detail::coerce_to<Scalar>(o.coefficient));
                }
            }

            // --- three-body ---
            for (const auto& th : three_body) {
                std::uint64_t new_basis = basis_state;
                std::complex<double> scalar = th.coefficient;
                bool valid = true;

                auto apply_one = [&](std::uint8_t op_type, std::uint64_t site) {
                    if (!valid) return;
                    if (op_type == 2) {
                        scalar *= spin *
                            (((new_basis >> site) & 1) ? -1.0 : 1.0);
                    } else {
                        const std::uint64_t b = (new_basis >> site) & 1ULL;
                        if (b != op_type) new_basis ^= (1ULL << site);
                        else              valid = false;
                    }
                };
                apply_one(th.op_type_1, th.site_index_1);
                apply_one(th.op_type_2, th.site_index_2);
                apply_one(th.op_type_3, th.site_index_3);

                if (valid) {
                    emit_offdiag(local, new_basis, static_cast<int>(i),
                                 detail::coerce_to<Scalar>(scalar));
                }
            }
        }
    }

    std::size_t total = 0;
    for (const auto& v : tls) total += v.size();
    triplets.clear();
    triplets.reserve(total);
    for (auto& v : tls) {
        triplets.insert(triplets.end(),
                        std::make_move_iterator(v.begin()),
                        std::make_move_iterator(v.end()));
        std::vector<Eigen::Triplet<Scalar>>().swap(v);
    }
}

// ===========================================================================
// Audit F4 (2026-09): direct two-pass CSR assembly in GATHER form.
//
// The Eigen triplet route costs 24 B/nnz of temporaries plus a serial
// ``setFromTriplets`` sort (measured ~47 s at N = 24 in the audit). Here the
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
