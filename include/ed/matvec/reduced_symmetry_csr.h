#pragma once
// =============================================================================
// include/ed/matvec/reduced_symmetry_csr.h
//
// Reduced-CSR "connectivity skeleton" for the orbit-walk symmetry SpMV.
//
// The matrix-free rep walk (apply_terms_gather_symmetry) regenerates, for EVERY
// connected state and EVERY matvec, the representative via an O(|G|) image scan
// (basis.index_of). On a large sector that O(|G|) factor dominates and is repeated
// every Lanczos/FTLM/... iteration. This builds the reduced sparse matrix ONCE --
// running the IDENTICAL row enumeration + coeff_modifier as the gather kernel, but
// RECORDING (col, value) instead of accumulating value*in[col] -- so every
// subsequent apply is a plain O(1)-per-nonzero CSR SpMV (no |G| scan, no index_of).
//
// Correctness is by construction: the per-element value `conj(base_contrib * mod)`
// is byte-for-byte the quantity the gather kernel accumulates (see the parity test
// in tests/unit/test_reduced_symmetry_csr.cpp, which checks CSR*v == gather*v).
//
// Memory: O(dim * avg_nnz_per_row) per sector (the materialized reduced matrix) --
// the planner gates this lane vs the matrix-free rep walk by the probed budget, so
// the rep walk remains the at-scale (O(#reps) memory) fallback for the very large
// sectors where even the reduced matrix does not fit.
// =============================================================================

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <ed/matvec/term_kernels.h>   // conj_scalar, coerce_coeff, kOpSz, kOpPlus

namespace ed::matvec {

template <class Scalar>
struct ReducedSymmetryCsr {
    std::vector<std::uint64_t> row_ptr;   // size dim+1
    std::vector<std::uint32_t> col_idx;   // size nnz
    std::vector<Scalar>        val;       // size nnz
    std::uint64_t              dim = 0;

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

namespace detail {

// Enumerate row `r`'s reduced-matrix elements via the SAME orbit walk + term
// emission + coeff_modifier the gather kernel uses, calling
// ``on_element(dst_idx, conj(base_contrib*mod))`` per connection. Mirrors
// ``apply_terms_gather_symmetry``'s per-row body verbatim (the single source of
// the symmetry-adapted matrix element); only the sink differs.
template <class BasisPolicy, class Scalar,
          class D1, class O1, class D2, class M2, class O2, class T3,
          class OnElement>
inline void rep_symmetry_row_for_each(
    std::uint64_t r, BasisPolicy basis, double spin_l, double spin_sq,
    const D1& diag_one_body, const O1& offdiag_one_body,
    const D2& diag_two_body, const M2& mixed_two_body,
    const O2& offdiag_two_body, const T3& three_body,
    OnElement&& on_element)
{
    auto gather_emit = [&](std::uint64_t dst_idx, std::uint64_t s_prime,
                           std::uint64_t basis_state, const Scalar& base_contrib) {
        const Scalar mod = basis.template coeff_modifier<Scalar>(
            basis_state, s_prime, r, dst_idx);
        on_element(dst_idx, kernel::conj_scalar<Scalar>(base_contrib * mod));
    };

    auto process_source = [&](std::uint64_t basis_state, std::complex<double> pre_phase) {
        const Scalar coeff = kernel::coerce_coeff<Scalar>(pre_phase);
        if (std::abs(coeff) < 1e-15) return;

        for (const auto& t : diag_one_body) {
            const double sign = ((basis_state >> t.site_index) & 1) ? -1.0 : 1.0;
            const Scalar contrib =
                kernel::coerce_coeff<Scalar>(t.coefficient) * spin_l * sign * coeff;
            gather_emit(r, basis_state, basis_state, contrib);
        }
        for (const auto& t : offdiag_one_body) {
            const std::uint64_t bit = (basis_state >> t.site_index) & 1;
            if (bit == t.op_type) continue;
            const std::uint64_t new_state = basis_state ^ (1ULL << t.site_index);
            const Scalar contrib = kernel::coerce_coeff<Scalar>(t.coefficient) * coeff;
            const std::int64_t j = basis.index_of(new_state);
            if (j < 0) continue;
            gather_emit(static_cast<std::uint64_t>(j), new_state, basis_state, contrib);
        }
        for (const auto& t : diag_two_body) {
            const double sa = ((basis_state >> t.site_index_1) & 1) ? -1.0 : 1.0;
            const double sb = ((basis_state >> t.site_index_2) & 1) ? -1.0 : 1.0;
            const Scalar contrib =
                kernel::coerce_coeff<Scalar>(t.coefficient) * spin_sq * sa * sb * coeff;
            gather_emit(r, basis_state, basis_state, contrib);
        }
        for (const auto& t : mixed_two_body) {
            const std::uint64_t flip_bit = (basis_state >> t.flip_site) & 1;
            if (flip_bit == t.flip_op_type) continue;
            const double sz_sign = ((basis_state >> t.sz_site) & 1) ? -1.0 : 1.0;
            const std::uint64_t new_state = basis_state ^ (1ULL << t.flip_site);
            const Scalar contrib =
                kernel::coerce_coeff<Scalar>(t.coefficient) * spin_l * sz_sign * coeff;
            const std::int64_t j = basis.index_of(new_state);
            if (j < 0) continue;
            gather_emit(static_cast<std::uint64_t>(j), new_state, basis_state, contrib);
        }
        for (const auto& t : offdiag_two_body) {
            const std::uint64_t b1 = (basis_state >> t.site_index_1) & 1;
            const std::uint64_t b2 = (basis_state >> t.site_index_2) & 1;
            if (b1 == t.op_type_1 || b2 == t.op_type_2) continue;
            const std::uint64_t new_state =
                basis_state ^ (1ULL << t.site_index_1) ^ (1ULL << t.site_index_2);
            const Scalar contrib = kernel::coerce_coeff<Scalar>(t.coefficient) * coeff;
            const std::int64_t j = basis.index_of(new_state);
            if (j < 0) continue;
            gather_emit(static_cast<std::uint64_t>(j), new_state, basis_state, contrib);
        }
        for (const auto& t : three_body) {
            std::uint64_t cur_state = basis_state;
            Scalar        scalar    = kernel::coerce_coeff<Scalar>(t.coefficient);
            bool          valid     = true;
            auto gate = [&](std::uint8_t op_type, std::uint64_t site) {
                if (!valid) return;
                if (op_type == kernel::kOpSz) {
                    const double sg = ((cur_state >> site) & 1) ? -1.0 : 1.0;
                    scalar *= spin_l * sg;
                } else {
                    const std::uint64_t bb = (cur_state >> site) & 1;
                    if (bb != op_type) cur_state ^= (1ULL << site);
                    else               valid = false;
                }
            };
            gate(t.op_type_1, t.site_index_1);
            gate(t.op_type_2, t.site_index_2);
            gate(t.op_type_3, t.site_index_3);
            if (!valid) continue;
            if (std::abs(scalar) < 1e-15) continue;
            const Scalar contrib = scalar * coeff;
            const std::int64_t j = basis.index_of(cur_state);
            if (j < 0) continue;
            gather_emit(static_cast<std::uint64_t>(j), cur_state, basis_state, contrib);
        }
    };

    basis.iter_orbit(r, process_source);
}

}  // namespace detail

// Build the reduced CSR for one orbit-walk symmetry sector. Runs the full O(|G|)
// row enumeration ONCE per row (in parallel); the result is reused by every later
// apply as an O(1)-per-nnz SpMV.
template <class BasisPolicy, class Scalar,
          class D1, class O1, class D2, class M2, class O2, class T3>
[[nodiscard]] inline ReducedSymmetryCsr<Scalar> build_reduced_symmetry_csr(
    BasisPolicy basis, double spin_l,
    const D1& diag_one_body, const O1& offdiag_one_body,
    const D2& diag_two_body, const M2& mixed_two_body,
    const O2& offdiag_two_body, const T3& three_body)
{
    const std::uint64_t dim = basis.dim();
    const double spin_sq = spin_l * spin_l;

    ReducedSymmetryCsr<Scalar> csr;
    csr.dim = dim;
    csr.row_ptr.assign(dim + 1, 0);

    // Per-row build (parallel): accumulate dst -> value, then flatten sorted.
    std::vector<std::vector<std::pair<std::uint32_t, Scalar>>> rows(dim);
#ifdef _OPENMP
    const std::uint64_t par = static_cast<std::uint64_t>(omp_get_max_threads()) * 256ULL;
#else
    const std::uint64_t par = std::numeric_limits<std::uint64_t>::max();
#endif
    #pragma omp parallel for schedule(dynamic, 256) if(dim > par)
    for (long long ir = 0; ir < static_cast<long long>(dim); ++ir) {
        const std::uint64_t r = static_cast<std::uint64_t>(ir);
        std::unordered_map<std::uint32_t, Scalar> acc;
        detail::rep_symmetry_row_for_each<BasisPolicy, Scalar>(
            r, basis, spin_l, spin_sq,
            diag_one_body, offdiag_one_body, diag_two_body,
            mixed_two_body, offdiag_two_body, three_body,
            [&](std::uint64_t dst, const Scalar& v) {
                acc[static_cast<std::uint32_t>(dst)] += v;
            });
        auto& dstrow = rows[r];
        dstrow.reserve(acc.size());
        for (const auto& kv : acc)
            if (std::abs(kv.second) > 0.0) dstrow.emplace_back(kv.first, kv.second);
        std::sort(dstrow.begin(), dstrow.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    for (std::uint64_t r = 0; r < dim; ++r)
        csr.row_ptr[r + 1] = csr.row_ptr[r] + rows[r].size();
    const std::uint64_t total = csr.row_ptr[dim];
    csr.col_idx.resize(total);
    csr.val.resize(total);
    for (std::uint64_t r = 0; r < dim; ++r) {
        std::uint64_t e = csr.row_ptr[r];
        for (const auto& cv : rows[r]) { csr.col_idx[e] = cv.first; csr.val[e] = cv.second; ++e; }
    }
    return csr;
}

}  // namespace ed::matvec
