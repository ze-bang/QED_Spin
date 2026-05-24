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

} // namespace ed::matvec::kernel
