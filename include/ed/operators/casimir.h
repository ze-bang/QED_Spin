#pragma once
// =============================================================================
// include/ed/operators/casimir.h
//
// The SU(2) Casimir S^2_tot = (S_tot)^2 in the canonical TermStorage schema,
// plus the label helpers (<S^2> expectation, S(S+1) snapping) used by every
// workflow that reports total-spin quantum numbers.
//
// Stage 12a of the SU(2) rollout (docs/architecture/SYMMETRY_V2_DESIGN.md).
//
// Operator identity (spin-1/2, N sites):
//
//   S^2 = sum_i S_i^2 + 2 sum_{i<j} S_i . S_j
//       = (3N/4) Id + sum_{i<j} [ 2 Sz_i Sz_j + S+_i S-_j + S-_i S+_j ]
//
// Every piece is expressible in the EXISTING term ABI -- that is the whole
// point of Route A (Casimir/projection): the same CPU/GPU/MPI term kernels
// that apply H in any symmetry-adapted basis (fixed-Sz combinadic, rep/orbit
// per momentum x irrep, flip-projected, little-group isotypic via lift) apply
// S^2 there too, because [S^2, g] = 0 for every site permutation g, every
// flip mask, and Sz (S^2 conserves popcount term-by-term).
//
// The (3N/4) Id shift needs NO new kernel: a diag_two_body term with
// site_1 == site_2 evaluates through diag_two_body_factor (term_gate_math.h)
// to spin_sq * sign^2 = 1/4 for EVERY basis state, on the CPU and GPU gate
// math alike. N such terms with coefficient 3.0 are exactly (3N/4) Id.
//
// Cost model: S^2 carries ~1.5*N^2 terms vs ~3*z*N for a short-range H, so
// one S^2 matvec costs about (N/2z) H-matvecs. Labeling one Ritz vector is
// one S^2 matvec (+1 for the certification residual).
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <ed/core/operator.h>
#include <ed/matvec/matvec.h>
#include <ed/matvec/term_storage.h>

namespace ed::ops {

// ---------------------------------------------------------------------------
// Term emission
// ---------------------------------------------------------------------------

/// Append the terms of S^2_tot for `n_sites` spin-1/2 sites to `t`.
/// Emits N diag(i,i) identity-shift terms + N(N-1)/2 * 3 pair terms.
inline void append_S2_total(ed::matvec::TermStorage& t, std::uint64_t n_sites) {
    using Cx = std::complex<double>;
    if (n_sites == 0 || n_sites >= 64) {
        throw std::invalid_argument(
            "append_S2_total: n_sites must be in [1, 63]");
    }
    // (3N/4) Id via the diag(i,i) trick: spin_sq * (+-1)^2 = 1/4 per state.
    for (std::uint64_t i = 0; i < n_sites; ++i) {
        t.add_diag_two_body(i, i, Cx(3.0, 0.0));
    }
    // 2 Sz_i Sz_j + S+_i S-_j + S-_i S+_j  per unordered pair.
    for (std::uint64_t i = 0; i < n_sites; ++i) {
        for (std::uint64_t j = i + 1; j < n_sites; ++j) {
            t.add_diag_two_body(i, j, Cx(2.0, 0.0));
            t.add_offdiag_two_body(i, j, /*S+*/ 0, /*S-*/ 1, Cx(1.0, 0.0));
            t.add_offdiag_two_body(i, j, /*S-*/ 1, /*S+*/ 0, Cx(1.0, 0.0));
        }
    }
}

/// S^2_tot as a full `::Operator` carrier (AoS terms, spin-1/2), consumable
/// by every per-sector factory that restricts H itself: the rep-basis
/// `make_rep_sector_matvec(op, RepSectorData)` (little_group_solve.h), the
/// fixed-Sz `SubspaceOperator` builders, and the dense assembly paths.
[[nodiscard]] inline std::shared_ptr<::Operator>
make_S2_carrier(std::uint64_t n_sites) {
    using Cx = std::complex<double>;
    auto op = std::make_shared<::Operator>(n_sites, 0.5f);
    for (std::uint64_t i = 0; i < n_sites; ++i) {
        op->addTwoBodyTerm(2, i, 2, i, Cx(3.0, 0.0));
    }
    for (std::uint64_t i = 0; i < n_sites; ++i) {
        for (std::uint64_t j = i + 1; j < n_sites; ++j) {
            op->addTwoBodyTerm(2, i, 2, j, Cx(2.0, 0.0));
            op->addTwoBodyTerm(0, i, 1, j, Cx(1.0, 0.0));
            op->addTwoBodyTerm(1, i, 0, j, Cx(1.0, 0.0));
        }
    }
    return op;
}

// ---------------------------------------------------------------------------
// Label helpers
// ---------------------------------------------------------------------------

/// S(S+1) for two_S = 2S (kept in doubled-integer form so odd-N
/// half-integer spins stay exact).
[[nodiscard]] inline double s2_eigenvalue_of_two_S(int two_S) noexcept {
    return 0.25 * static_cast<double>(two_S) *
           static_cast<double>(two_S + 2);
}

/// Snap a measured <S^2> to the nearest ALLOWED S(S+1) and return two_S,
/// or -1 when no allowed value lies within `tol` (absolute; adjacent
/// allowed eigenvalues are >= 2 apart so 1e-6 is comfortably safe).
///
/// Allowed set:
///   * two_S has the parity of n_sites and two_S <= n_sites;
///   * two_S >= |2*n_up - n_sites| when a fixed-Sz sector is active
///     (S >= |Sz|);
///   * when a spin-flip parity block is active at half filling,
///     X|S, m=0> = (-1)^{N/2 - S} |S, m=0>, so only
///     ((n_sites - two_S)/2) % 2 == flip_parity survives (0 = +, 1 = -).
[[nodiscard]] inline int snap_two_S(double s2_exp, int n_sites,
                                    int n_up = -1, int flip_parity = -1,
                                    double tol = 1e-6) noexcept {
    int lo = (n_up >= 0) ? std::abs(2 * n_up - n_sites) : (n_sites % 2);
    int best = -1;
    double best_d = std::numeric_limits<double>::infinity();
    for (int ts = lo; ts <= n_sites; ts += 2) {
        if (flip_parity >= 0 &&
            (((n_sites - ts) / 2) % 2) != flip_parity) {
            continue;
        }
        const double d = std::abs(s2_exp - s2_eigenvalue_of_two_S(ts));
        if (d < best_d) {
            best_d = d;
            best = ts;
        }
    }
    return (best >= 0 && best_d <= tol) ? best : -1;
}

/// <v|S^2|v> / <v|v> through any sector-restricted S^2 matvec. When
/// `out_residual` is non-null it also reports the certification residual
///     ||S^2 v - <S^2> v|| / ||v||
/// (costs nothing extra: reuses the same S^2 v buffer). A label should be
/// trusted only when the residual is small (<= 1e-8): an accidental
/// energy degeneracy can mix different-S eigenvectors, in which case the
/// expectation is not an S(S+1) point and must not be snapped.
[[nodiscard]] inline double
s2_expectation(const ed::matvec::MatVecOperator& s2,
               const std::complex<double>* v, std::uint64_t dim,
               double* out_residual = nullptr) {
    using Cx = std::complex<double>;
    if (dim == 0 || dim != s2.dim()) {
        throw std::invalid_argument(
            "s2_expectation: dim mismatch with the S^2 operator");
    }
    std::vector<Cx> w(dim);
    s2.apply(v, w.data(), dim);
    double norm2 = 0.0;
    Cx dot(0.0, 0.0);
    for (std::uint64_t i = 0; i < dim; ++i) {
        norm2 += std::norm(v[i]);
        dot += std::conj(v[i]) * w[i];
    }
    if (norm2 <= 0.0) {
        throw std::invalid_argument("s2_expectation: zero vector");
    }
    const double exp_val = dot.real() / norm2;
    if (out_residual != nullptr) {
        double r2 = 0.0;
        for (std::uint64_t i = 0; i < dim; ++i) {
            r2 += std::norm(w[i] - exp_val * v[i]);
        }
        *out_residual = std::sqrt(r2 / norm2);
    }
    return exp_val;
}

/// Residual threshold below which a snapped label is *certified* (see
/// s2_expectation). Shared by the workflow labelers so C++ and Python
/// agree on when to report spin = None.
inline constexpr double kS2CertifyTol = 1e-8;

}  // namespace ed::ops
