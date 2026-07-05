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

#include <ed/core/basis_utils.h>          // popcount (defensive, mirrors CrossSectorObservable)
#include <ed/core/sorted_uint64_index.h>  // SortedUint64Index::kNotFound
#include <ed/matvec/symmetry_matvec_backend.h>  // rep_policy_from (Stage 8d)

#include <algorithm>
#include <cstdint>
#include <stdexcept>

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
    if (dst_.group_size() != G) {
        throw std::invalid_argument(
            "CrossSectorOrbitObservable: src/dst group_size mismatch ("
            + std::to_string(G) + " vs "
            + std::to_string(dst_.group_size()) + ") -- the two sectors "
            "must come from the same (possibly flip-extended) group.");
    }
    group_norm_ = 1.0 / static_cast<double>(G);
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

    const SymmetrySector* src_sec =
        src_.is_rep() ? nullptr : &src_.sector(src_sector_);
    const SymmetrySector* dst_sec =
        dst_.is_rep() ? nullptr : &dst_.sector(dst_sector_);
    const double S = static_cast<double>(spin_l_);
    const bool src_rep = src_.is_rep();
    const bool dst_rep = dst_.is_rep();
    const int  G_src   = static_cast<int>(src_.group_size());

    // OpenMP-parallel walk over source orbit-basis indices, with
    // per-thread output accumulators. Avoids the per-update atomic
    // contention on dst entries; the merge at the end is O(dim_dst).
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    std::vector<std::vector<Complex>> tls(
        max_threads, std::vector<Complex>(dim_dst_, Complex(0.0, 0.0)));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        auto& local_out = tls[tid];

        // Destination projection, shared by both source lanes. Orbit lane:
        // sorted-index lookup + findCoeff; rep lane: index_and_projection
        // (identical arithmetic, pinned by test_rep_cross_sector.cpp).
        auto scatter_dst = [&](Complex weighted, std::uint64_t s_prime) {
            if (dst_rep) {
                Complex proj;
                const std::int64_t k =
                    dst_pol_.index_and_projection(s_prime, proj);
                if (k < 0) return;
                local_out[static_cast<std::size_t>(k)] +=
                    weighted * proj * group_norm_;
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
            local_out[k] += weighted * std::conj(beta_s_prime)
                          * group_norm_ / state_k.norm;
        };

        #pragma omp for schedule(dynamic, 64)
        for (std::int64_t alpha = 0;
             alpha < static_cast<std::int64_t>(dim_src_); ++alpha) {
            const Complex c_alpha = in[alpha];
            if (std::abs(c_alpha) < 1e-15) continue;

            if (src_rep) {
                // Rep lane: regenerate the source orbit per group element.
                // The per-state expansion coefficient alpha_s of the orbit
                // basis is sum_{g: g(rep)=s} conj(chi(g)); summing per-g is
                // the same sum without the dedup.
                const std::uint64_t rep = src_pol_.state_of(
                    static_cast<std::uint64_t>(alpha));
                const double inv_norm_alpha = src_pol_.inv_norm_of(
                    static_cast<std::uint64_t>(alpha));
                for (int g = 0; g < G_src; ++g) {
                    const std::uint64_t s = src_pol_.apply_perm(rep, g);
                    const Complex chi_g   = src_pol_.characters[g];
                    const Complex weighted =
                        c_alpha * std::conj(chi_g) * inv_norm_alpha;
                    for (const auto& t : transforms_) {
                        const TermResult r = applyOneTerm(s, t, S);
                        if (r.valid) scatter_dst(weighted * r.amp, r.s_prime);
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

                const Complex weighted = c_alpha * alpha_s / norm_alpha;

                for (const auto& t : transforms_) {
                    const TermResult r = applyOneTerm(s, t, S);
                    if (r.valid) scatter_dst(weighted * r.amp, r.s_prime);
                }
            }
        }
    }  // end omp parallel

    // Merge the per-thread accumulators into out[].
    for (int t = 0; t < max_threads; ++t) {
        const auto& src = tls[t];
        for (std::size_t k = 0; k < dim_dst_; ++k) {
            out[k] += src[k];
        }
    }
}

}  // namespace ed::dssf
