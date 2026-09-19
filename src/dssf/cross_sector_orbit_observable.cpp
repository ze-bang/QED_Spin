// =============================================================================
// src/dssf/cross_sector_orbit_observable.cpp
//
// Implementation of ed::dssf::CrossSectorOrbitObservable. See the
// header for the math; this file is the inner hot loop.
//
// The algorithm mirrors `StreamingSymmetryOperator::applyHamiltonianTermsFullSpace`
// (the same-sector matvec inner loop in
// `include/ed/core/streaming_symmetry.h`) but routes the projection
// through the *destination* sector's orbit basis instead of the
// source sector's:
//
//   for each input orbit-basis index alpha in [0, dim_src):
//     c_alpha = in[alpha]
//     for each (s, alpha_s) in orbit_alpha (source sector):
//       weighted = c_alpha * alpha_s / norm_alpha
//       for each transform T in transforms_:
//         (s', h) = apply T to s        // bit-flip / Sz-sign action
//         k = dst.lookupBasisIndex(dst_sector, s')
//         if k == kNotFound: skip
//         beta_s' = dst.sector(dst_sector).basis_states[k].findCoeff(s')
//         out[k] += weighted * h * conj(beta_s') * group_norm / norm_k
//
// This is the SOTA spectral-function building block that closes the
// cross-irrep gap in docs/architecture/SYMMETRY.md Section 3.
// =============================================================================

#include <ed/dssf/cross_sector_orbit_observable.h>

#include <ed/config/env_registry.h>
#include <ed/core/basis_utils.h>          // popcount (defensive, mirrors CrossSectorObservable)
#include <ed/core/sorted_uint64_index.h>  // SortedUint64Index::kNotFound
#include <ed/matvec/symmetry_matvec_backend.h>  // rep_policy_from (Stage 8d)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::dssf {

CrossSectorOrbitObservable::CrossSectorOrbitObservable(
    OperatorRef                 src,
    std::size_t                 src_sector,
    OperatorRef                 dst,
    std::size_t                 dst_sector,
    std::vector<TransformData>  transforms,
    float                       spin_l)
  : src_(src),
    src_sector_(src_sector),
    dst_(dst),
    dst_sector_(dst_sector),
    transforms_(std::move(transforms)),
    spin_l_(spin_l) {
    if (!src_.valid()) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable: src operator handle is empty.");
    }
    if (!dst_.valid()) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable: dst operator handle is empty.");
    }
    if (src_.num_bits() != dst_.num_bits()) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable: src/dst num_bits mismatch.");
    }
    if (src_sector_ >= src_.num_sectors()) {
        throw std::out_of_range(
            "CrossSectorOrbitObservable: src_sector out of range.");
    }
    if (dst_sector_ >= dst_.num_sectors()) {
        throw std::out_of_range(
            "CrossSectorOrbitObservable: dst_sector out of range.");
    }
    if (transforms_.empty()) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable: transforms is empty.");
    }
    n_bits_     = src_.num_bits();
    dim_src_    = src_.dim();
    dim_dst_    = dst_.dim();
    // group_norm = 1 / |G|, matches the same-sector matvec at
    // streaming_symmetry.h:453.
    const std::uint64_t G = src_.group_size();
    if (G == 0) {
        throw std::runtime_error(
            "CrossSectorOrbitObservable: src group_size is zero "
            "(streaming-symmetry operator missing automorphism metadata?).");
    }
    // U2b-r1: src and dst may come from DIFFERENT group extensions -- a
    // flip-extended (k, +/-) source (|G'| = 2|A|) scattering into raw
    // destination sectors (|A|). The normalization convention stores
    // per-rep norms of sqrt(|Stab|) (rep_projection.h closed form), i.e.
    // each side's basis vector is short a sqrt(|G_side|); the bra-ket
    // therefore carries 1/sqrt(G_src * G_dst), which reduces to the
    // historical 1/G when the groups match (every existing pin
    // unchanged). The destination group must still divide or extend the
    // same automorphism content -- that is the caller's contract (both
    // handles built from one engine context).
    const std::uint64_t Gd = dst_.group_size();
    if (Gd == 0) {
        throw std::runtime_error(
            "CrossSectorOrbitObservable: dst group_size is zero.");
    }
    group_norm_ = 1.0 / std::sqrt(static_cast<double>(G)
                                  * static_cast<double>(Gd));
    // Stage 8d: build the POD policy views once. The rep-lane inner loops
    // regenerate the orbit / projection arithmetically through these.
    if (src_.is_rep()) src_pol_ = ed::matvec::rep_policy_from(*src_.rd);
    if (dst_.is_rep()) dst_pol_ = ed::matvec::rep_policy_from(*dst_.rd);
    if ((src_.is_rep() && G > 256) || (dst_.is_rep() && G > 256)) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable: rep-lane refs support group_size "
            "<= 256 (index_and_projection stack buffer).");
    }
}

namespace {

// Apply one transform term to a computational-basis state ``s``.
// Returns whether the action is non-zero ("valid"), the resulting
// state ``s_prime`` (potentially with one or two bits flipped), and
// the accumulated matrix-element factor ``amp`` (includes the
// transform coefficient and the spin-S diagonal/off-diagonal
// prefactors). Mirrors the term-walking switch in
// streaming_symmetry.h:1290-1352 and the equivalent block in
// cross_sector_observable.cpp.
struct TermResult {
    bool                          valid     = true;
    std::uint64_t                 s_prime   = 0;
    std::complex<double>          amp       = 0.0;
};

inline TermResult applyOneTerm(
        std::uint64_t                              s,
        const Operator::TransformData&             t,
        double                                     S) {
    TermResult r;
    r.s_prime = s;
    r.amp     = t.coefficient;

    if (!t.is_two_body) {
        // One-body: S^op_i
        if (t.op_type == 2) {
            // Sz_i: diagonal, sign = +1 if bit=0 else -1
            double sign = ((s >> t.site_index) & 1ULL) ? -1.0 : 1.0;
            r.amp *= S * sign;
        } else {
            // S+ (op_type=0) / S- (op_type=1): flip bit if it matches
            // the source state of the operator.
            std::uint64_t bit = (s >> t.site_index) & 1ULL;
            if (bit != t.op_type) {
                r.s_prime ^= (1ULL << t.site_index);
            } else {
                r.valid = false;
            }
        }
        return r;
    }

    // Two-body: S^op_i S^op2_j
    std::uint64_t bit_i = (s >> t.site_index)   & 1ULL;
    std::uint64_t bit_j = (s >> t.site_index_2) & 1ULL;

    if (t.op_type == 2 && t.op_type_2 == 2) {
        double sign_i = bit_i ? -1.0 : 1.0;
        double sign_j = bit_j ? -1.0 : 1.0;
        r.amp *= S * S * sign_i * sign_j;
        return r;
    }

    // Mixed terms: walk sequentially through site_i then site_j on
    // the partially-updated state.
    if (t.op_type != 2) {
        if (bit_i != t.op_type) {
            r.s_prime ^= (1ULL << t.site_index);
        } else {
            r.valid = false;
            return r;
        }
    } else {
        double sign_i = bit_i ? -1.0 : 1.0;
        r.amp *= S * sign_i;
    }

    if (t.op_type_2 != 2) {
        std::uint64_t new_bit_j = (r.s_prime >> t.site_index_2) & 1ULL;
        if (new_bit_j != t.op_type_2) {
            r.s_prime ^= (1ULL << t.site_index_2);
        } else {
            r.valid = false;
        }
    } else {
        std::uint64_t new_bit_j = (r.s_prime >> t.site_index_2) & 1ULL;
        double sign_j = new_bit_j ? -1.0 : 1.0;
        r.amp *= S * sign_j;
    }
    return r;
}

}  // namespace

// ---------------------------------------------------------------------------
// The walk. ``emit(alpha, k, value)`` receives every non-zero reduced matrix
// element A[k, alpha] (value EXCLUDES the input coefficient in[alpha]); it is
// called from inside an OpenMP team, so the emitter must be thread-safe (the
// callers below use per-thread buffers).
// ---------------------------------------------------------------------------
template <class Emit>
void CrossSectorOrbitObservable::walk_columns_(Emit&& emit) const {
    const SymmetrySector* src_sec =
        src_.is_rep() ? nullptr : &src_.sector(src_sector_);
    const SymmetrySector* dst_sec =
        dst_.is_rep() ? nullptr : &dst_.sector(dst_sector_);
    const double S = static_cast<double>(spin_l_);
    const bool src_rep = src_.is_rep();
    const bool dst_rep = dst_.is_rep();
    const int  G_src   = static_cast<int>(src_.group_size());

    #pragma omp parallel
    {
        // Destination projection, shared by both source lanes. Orbit lane:
        // sorted-index lookup + findCoeff; rep lane: index_and_projection
        // (identical arithmetic, pinned by test_rep_cross_sector.cpp).
        auto scatter_dst = [&](std::uint64_t alpha, Complex weighted, std::uint64_t s_prime) {
            if (dst_rep) {
                Complex proj;
                const std::int64_t k =
                    dst_pol_.index_and_projection(s_prime, proj);
                if (k < 0) return;
                emit(alpha, static_cast<std::size_t>(k), weighted * proj * group_norm_);
                return;
            }
            const std::size_t k = dst_.lookupBasisIndex(dst_sector_, s_prime);
            if (k == ed::core::SortedUint64Index::kNotFound) return;
            const auto& state_k = dst_sec->basis_states[k];
            if (!(state_k.norm > 0.0)) return;
            const Complex beta_s_prime = state_k.findCoeff(s_prime);
            // Same projection formula as applyHamiltonianTermsFullSpace
            // (streaming_symmetry.h:1288) with the destination orbit
            // basis providing beta_s_prime / norm_k.
            emit(alpha, k, weighted * std::conj(beta_s_prime) * group_norm_ / state_k.norm);
        };

        #pragma omp for schedule(dynamic, 64)
        for (std::int64_t ia = 0; ia < static_cast<std::int64_t>(dim_src_); ++ia) {
            const std::uint64_t alpha = static_cast<std::uint64_t>(ia);
            if (src_rep) {
                // Rep lane: regenerate the source orbit per group element.
                // The per-state expansion coefficient alpha_s of the orbit
                // basis is sum_{g: g(rep)=s} conj(chi(g)); summing per-g is
                // the same sum without the dedup.
                const std::uint64_t rep = src_pol_.state_of(alpha);
                const double inv_norm_alpha = src_pol_.inv_norm_of(alpha);
                for (int g = 0; g < G_src; ++g) {
                    const std::uint64_t s = src_pol_.apply_perm(rep, g);
                    const Complex chi_g   = src_pol_.characters[g];
                    const Complex weighted = std::conj(chi_g) * inv_norm_alpha;
                    for (const auto& t : transforms_) {
                        const TermResult r = applyOneTerm(s, t, S);
                        if (r.valid) scatter_dst(alpha, weighted * r.amp, r.s_prime);
                    }
                }
                continue;
            }

            const auto& state_alpha = src_sec->basis_states[alpha];
            const double norm_alpha = state_alpha.norm;
            if (!(norm_alpha > 0.0)) continue;

            // Walk the orbit of the source basis state, mirroring
            // streaming_symmetry.h:467-477 (same expansion that the
            // same-sector matvec uses).
            const std::size_t orbit_sz = state_alpha.orbit_elements.size();
            for (std::size_t orbit_idx = 0; orbit_idx < orbit_sz; ++orbit_idx) {
                const std::uint64_t s        = state_alpha.orbit_elements[orbit_idx];
                const Complex       alpha_s  = state_alpha.orbit_coefficients[orbit_idx];
                if (std::abs(alpha_s) < 1e-15) continue;
                const Complex weighted = alpha_s / norm_alpha;
                for (const auto& t : transforms_) {
                    const TermResult r = applyOneTerm(s, t, S);
                    if (r.valid) scatter_dst(alpha, weighted * r.amp, r.s_prime);
                }
            }
        }
    }  // end omp parallel
}

// Fallback (no cache): scatter walk with per-thread output accumulators.
void CrossSectorOrbitObservable::apply_walk_(const Complex* in, Complex* out) const {
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<std::vector<Complex>> tls(
        max_threads, std::vector<Complex>(dim_dst_, Complex(0.0, 0.0)));
    walk_columns_([&](std::uint64_t alpha, std::size_t k, Complex v) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        tls[tid][k] += in[alpha] * v;
    });
    for (int t = 0; t < max_threads; ++t) {
        const auto& src = tls[t];
        for (std::size_t k = 0; k < dim_dst_; ++k) out[k] += src[k];
    }
}

void CrossSectorOrbitObservable::build_csr_() const {
    // Budget on the pre-merge triplet stream (upper bound: every group
    // element x term of every source row emits one entry).
    const double est = static_cast<double>(dim_src_)
                     * static_cast<double>(std::max<std::uint64_t>(src_.group_size(), 1))
                     * static_cast<double>(std::max<std::size_t>(transforms_.size(), 1))
                     * 24.0;
    double budget_gib = 4.0;
    if (const char* v = ed::env::raw("ED_XSEC_CSR_BUDGET_GIB")) {
        const double b = std::atof(v);
        if (b > 0.0) budget_gib = b;
    }
    if (est > budget_gib * 1073741824.0 || dim_src_ > 0xFFFFFFFFull) {
        csr_refused_ = true;
        return;
    }
    struct Entry { std::uint32_t row; std::uint32_t col; Complex val; };
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<std::vector<Entry>> tls(static_cast<std::size_t>(max_threads));
    walk_columns_([&](std::uint64_t alpha, std::size_t k, Complex v) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        tls[static_cast<std::size_t>(tid)].push_back(
            Entry{static_cast<std::uint32_t>(k), static_cast<std::uint32_t>(alpha), v});
    });
    std::size_t total = 0;
    for (const auto& v : tls) total += v.size();
    std::vector<Entry> all;
    all.reserve(total);
    for (auto& v : tls) {
        all.insert(all.end(), v.begin(), v.end());
        std::vector<Entry>().swap(v);
    }
    std::sort(all.begin(), all.end(), [](const Entry& a, const Entry& b) {
        return (a.row != b.row) ? a.row < b.row : a.col < b.col;
    });
    csr_row_ptr_.assign(dim_dst_ + 1, 0);
    csr_col_.clear();
    csr_val_.clear();
    csr_col_.reserve(all.size());
    csr_val_.reserve(all.size());
    for (std::size_t i = 0; i < all.size();) {
        std::size_t j = i + 1;
        Complex acc = all[i].val;
        while (j < all.size() && all[j].row == all[i].row && all[j].col == all[i].col) {
            acc += all[j].val;
            ++j;
        }
        if (std::abs(acc) > 1e-15) {
            csr_col_.push_back(all[i].col);
            csr_val_.push_back(acc);
            csr_row_ptr_[all[i].row + 1] += 1;
        }
        i = j;
    }
    for (std::size_t r = 0; r < dim_dst_; ++r) csr_row_ptr_[r + 1] += csr_row_ptr_[r];
    csr_built_ = true;
}

void CrossSectorOrbitObservable::apply(const Complex* in,
                                       Complex*       out,
                                       std::size_t    dst_size) const {
    if (dst_size != dim_dst_) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable::apply: dst_size mismatch ("
            "got " + std::to_string(dst_size) +
            ", expected " + std::to_string(dim_dst_) + ").");
    }

    std::fill(out, out + dim_dst_, Complex(0.0, 0.0));
    if (dim_src_ == 0 || dim_dst_ == 0) return;

    if (!csr_built_ && !csr_refused_) {
        std::lock_guard<std::mutex> lock(csr_mutex_);
        if (!csr_built_ && !csr_refused_) build_csr_();
    }
    if (!csr_built_) {
        apply_walk_(in, out);
        return;
    }
    const std::int64_t*  rp  = csr_row_ptr_.data();
    const std::uint32_t* col = csr_col_.data();
    const Complex*       val = csr_val_.data();
    const std::int64_t   R   = static_cast<std::int64_t>(dim_dst_);
    #pragma omp parallel for schedule(static) if(R > 4096)
    for (std::int64_t k = 0; k < R; ++k) {
        Complex acc(0.0, 0.0);
        for (std::int64_t q = rp[k]; q < rp[k + 1]; ++q) acc += val[q] * in[col[q]];
        out[k] = acc;
    }
}

}  // namespace ed::dssf
