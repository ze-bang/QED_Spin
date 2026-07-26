#pragma once
// =============================================================================
// include/ed/symmetry/casimir_projector.h
//
// Stage 12d of the SU(2) rollout (docs/architecture/SYMMETRY_V2_DESIGN.md):
// the Lowdin total-spin projector and the Krylov-targeting operator wrapper.
// This is the `CasimirProjector` seat reserved in projector.h -- realised at
// the OPERATOR level (a polynomial in S^2 acting on vectors), NOT as a
// Subspace: S-eigenspaces are not spanned by computational basis states, so
// there is nothing for an `index_of` to filter. The matvec ABI is untouched.
//
// Lowdin projector onto the spin-S eigenspace of S^2:
//
//   P_S = prod_{S' != S, S' in block} (S^2 - S'(S'+1)) / (S(S+1) - S'(S'+1))
//
// a polynomial in S^2 of degree = |excluded set|. Within a fixed-Sz block
// only S' >= |Sz| appear; within a flip-parity block at half filling only
// every OTHER S' appears (X|S, m=0> = (-1)^{N/2-S}|S, m=0>), halving the
// degree again. `allowed_two_S_in_block` is the single source of that set.
//
// Numerical policy (see the SU(2) plan):
//   * factors are applied NUMERATOR-ONLY, farthest eigenvalue first (the
//     dominant unwanted components are annihilated while intermediate
//     norms stay O(1));
//   * the vector is renormalised after every factor; the exact overall
//     multiplier (product of intermediate norms / product of denominators)
//     is accumulated in log space and restored at the end, so `project`
//     computes EXACTLY P_S v, not a rescaled cousin;
//   * cost: one factor = one S^2 matvec (~1.5 N^2 terms, i.e. ~(N/2z) of a
//     short-range H matvec). Full projection ~ degree * that. Applied at
//     the start vector plus every `reproject_freq`-th operator apply.
//
// Krylov targeting: since [H, S^2] = 0 exactly for an SU(2)-invariant H,
// a start vector inside the S-eigenspace keeps its whole Krylov space
// there; P_S is the IDENTITY on the exact Krylov space and only scrubs
// floating-point drift. `CasimirProjectedOperator` therefore wraps any
// sector matvec without changing its spectrum on the targeted tower --
// Lanczos / Krylov-Schur / FTLM / mTPQ consume it via the ordinary
// `LinearOperator` surface. Host memory space only (the device-resident
// wrapper is the Stage-12h audit follow-up).
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/core/linear_operator.h>
#include <ed/matvec/matvec.h>
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/su2_dims.h>

namespace ed::symmetry {

/// Reprojection cadence for the targeting wrapper. Default 1 (project
/// every apply): eigenvalue-only Lanczos lanes run bounded (local)
/// reorthogonalisation, where an off-tower roundoff component is
/// AMPLIFIED toward the global extremal eigenvalue and converges as a
/// ghost within ~30 iterations if left unscrubbed -- correctness first.
/// Raise via ED_SYM_SU2_REPROJECT_FREQ (k = project every k-th apply;
/// 0 = seed projection only) when the projection cost dominates and the
/// spectrum is known to be drift-benign. Read per call so tests can
/// toggle from Python.
[[nodiscard]] inline int su2_reproject_freq() noexcept {
    const char* v = std::getenv("ED_SYM_SU2_REPROJECT_FREQ");
    if (v == nullptr || *v == '\0') return 1;
    return std::atoi(v);
}

/// The set of total-spin labels (as two_S) present in a symmetry block,
/// the single source consumed by the Lowdin excluded-set and by the dense
/// S-resolution (Stage 12e).
///   * n_up >= 0     : fixed-Sz block, S >= |Sz|;
///   * sz_parity >= 0: Sz-parity half; only the S = 0 tower is parity-
///     selective (its lone member sits at n_up = N/2), every S >= 1
///     multiplet spans both halves;
///   * flip_parity >= 0 (0 = +, 1 = -): flip-projected block; admissible
///     only where `flip_subspace_admissible` says so, and the surviving
///     towers obey ((N - 2S)/2) % 2 == flip_parity via the Sz = 0 member.
[[nodiscard]] inline std::vector<int>
allowed_two_S_in_block(int n_sites, int n_up = -1, int sz_parity = -1,
                       int flip_parity = -1) {
    if (n_sites <= 0 || n_sites >= 64) {
        throw std::invalid_argument(
            "allowed_two_S_in_block: n_sites must be in [1, 63]");
    }
    if (flip_parity >= 0 &&
        !flip_subspace_admissible(n_up, sz_parity, n_sites)) {
        throw std::invalid_argument(
            "allowed_two_S_in_block: flip parity requested in a block the "
            "global spin flip does not preserve (n_up=" +
            std::to_string(n_up) + ", sz_parity=" +
            std::to_string(sz_parity) + ", N=" + std::to_string(n_sites) +
            ")");
    }
    const int lo =
        (n_up >= 0) ? std::abs(2 * n_up - n_sites) : (n_sites % 2);
    std::vector<int> out;
    for (int ts = lo; ts <= n_sites; ts += 2) {
        if (flip_parity >= 0 &&
            (((n_sites - ts) / 2) % 2) != flip_parity) {
            continue;
        }
        if (ts == 0 && n_up < 0 && sz_parity >= 0 &&
            ((n_sites / 2) % 2) != sz_parity) {
            continue;  // lone S=0 member sits in the other parity half
        }
        out.push_back(ts);
    }
    return out;
}

// ---------------------------------------------------------------------------
// LowdinS2Projector
// ---------------------------------------------------------------------------
class LowdinS2Projector {
public:
    /// `s2` must be S^2 restricted to the SAME basis the projected vectors
    /// live in (full, fixed-Sz, rep/orbit, flip-projected, ...).
    /// `two_S_present` is the block's tower content (allowed_two_S_in_block
    /// or the S-resolved-dims survey); the target must be a member.
    LowdinS2Projector(std::shared_ptr<const ed::matvec::MatVecOperator> s2,
                      int two_S_target, std::vector<int> two_S_present)
        : s2_(std::move(s2)), two_S_(two_S_target) {
        if (!s2_) {
            throw std::invalid_argument("LowdinS2Projector: null S^2");
        }
        bool found = false;
        const double lam_t = 0.25 * two_S_ * (two_S_ + 2);
        for (int ts : two_S_present) {
            if (ts == two_S_) {
                found = true;
                continue;
            }
            excluded_.push_back(0.25 * ts * (ts + 2));
        }
        if (!found) {
            throw std::invalid_argument(
                "LowdinS2Projector: target two_S = " +
                std::to_string(two_S_) +
                " is not in the block's tower set");
        }
        // Farthest-first: annihilate the dominant unwanted towers early.
        std::sort(excluded_.begin(), excluded_.end(),
                  [lam_t](double a, double b) {
                      return std::abs(a - lam_t) > std::abs(b - lam_t);
                  });
    }

    [[nodiscard]] int degree() const noexcept {
        return static_cast<int>(excluded_.size());
    }
    [[nodiscard]] int two_S() const noexcept { return two_S_; }

    /// In-place EXACT P_S v (per-factor renormalisation is undone through
    /// the log-space multiplier). `v` and the scratch live on the host.
    void project(std::complex<double>* v, std::uint64_t dim) const {
        const double norm_in = norm_of(v, dim);
        if (norm_in == 0.0) return;
        const double scale = project_normalized(v, dim);
        for (std::uint64_t i = 0; i < dim; ++i) v[i] *= scale;
    }

    /// In-place projection leaving a UNIT vector; returns ||P_S v|| (the
    /// exact multiplier that `project` would have restored). A return of
    /// ~0 means v had (numerically) no weight in the target tower -- the
    /// caller should redraw its random seed.
    double project_normalized(std::complex<double>* v,
                              std::uint64_t dim) const {
        using Cx = std::complex<double>;
        if (dim != s2_->dim()) {
            throw std::invalid_argument(
                "LowdinS2Projector: dim mismatch with the S^2 operator");
        }
        const double lam_t = 0.25 * two_S_ * (two_S_ + 2);
        double log_scale = 0.0;
        double sign = 1.0;
        {
            const double n0 = norm_of(v, dim);
            if (n0 == 0.0) return 0.0;
            const double inv = 1.0 / n0;
            for (std::uint64_t i = 0; i < dim; ++i) v[i] *= inv;
            log_scale += std::log(n0);
        }
        std::vector<Cx> w(dim);
        for (const double lam : excluded_) {
            s2_->apply(v, w.data(), dim);
            for (std::uint64_t i = 0; i < dim; ++i) {
                v[i] = w[i] - lam * v[i];
            }
            const double n = norm_of(v, dim);
            const double denom = lam_t - lam;
            if (n == 0.0) return 0.0;  // annihilated: no target weight
            const double inv = 1.0 / n;
            for (std::uint64_t i = 0; i < dim; ++i) v[i] *= inv;
            log_scale += std::log(n) - std::log(std::abs(denom));
            if (denom < 0.0) sign = -sign;
        }
        if (sign < 0.0) {
            for (std::uint64_t i = 0; i < dim; ++i) v[i] = -v[i];
        }
        return std::exp(log_scale);
    }

private:
    static double norm_of(const std::complex<double>* v,
                          std::uint64_t dim) noexcept {
        double n2 = 0.0;
        for (std::uint64_t i = 0; i < dim; ++i) n2 += std::norm(v[i]);
        return std::sqrt(n2);
    }

    std::shared_ptr<const ed::matvec::MatVecOperator> s2_;
    int two_S_;
    std::vector<double> excluded_;  // S'(S'+1), farthest-first
};

// ---------------------------------------------------------------------------
// CasimirProjectedOperator
// ---------------------------------------------------------------------------
/// Wraps any host sector matvec H with periodic P_S drift scrubbing.
/// Spectrally H and the wrapper agree on the targeted tower ([H, P_S] = 0
/// and P_S == Id there); off-tower roundoff components introduced by
/// finite arithmetic are annihilated every `reproject_freq`-th apply.
class CasimirProjectedOperator : public ed::LinearOperator {
public:
    CasimirProjectedOperator(
        std::shared_ptr<const ed::matvec::MatVecOperator> h,
        std::shared_ptr<const LowdinS2Projector> projector,
        int reproject_freq = -1)  // -1 = env default
        : h_(std::move(h)),
          projector_(std::move(projector)),
          freq_(reproject_freq >= 0 ? reproject_freq
                                    : su2_reproject_freq()) {
        if (!h_ || !projector_) {
            throw std::invalid_argument(
                "CasimirProjectedOperator: null operator/projector");
        }
        if (h_->memory_space() != ed::matvec::MemorySpace::Host) {
            throw std::invalid_argument(
                "CasimirProjectedOperator: host memory space only "
                "(device-resident targeting is the Stage-12h follow-up)");
        }
    }

    void apply(const Complex* in, Complex* out,
               std::size_t size) const override {
        h_->apply(in, out, size);
        if (freq_ > 0 && (++apply_count_ % freq_) == 0) {
            projector_->project(out, size);
        }
    }

    [[nodiscard]] std::size_t dim() const override { return h_->dim(); }
    [[nodiscard]] std::size_t global_dim() const override {
        return h_->global_dim();
    }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return h_->memory_space();
    }
    [[nodiscard]] bool is_hermitian() const override {
        return h_->is_hermitian();
    }

    /// Project a (random) seed into the target tower and normalise it.
    /// Returns ||P_S seed|| -- ~0 tells the caller to redraw.
    double prepare_start_vector(Complex* v, std::uint64_t dim) const {
        return projector_->project_normalized(v, dim);
    }

    [[nodiscard]] const LowdinS2Projector& projector() const noexcept {
        return *projector_;
    }

private:
    std::shared_ptr<const ed::matvec::MatVecOperator> h_;
    std::shared_ptr<const LowdinS2Projector> projector_;
    int freq_;
    mutable std::uint64_t apply_count_ = 0;
};

}  // namespace ed::symmetry
