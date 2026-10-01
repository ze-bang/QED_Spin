#pragma once
// =============================================================================
// include/ed/krylov/lanczos_kernel.h
//
// THE single Lanczos algorithm.
//
// One template-function body drives both deployment targets in this
// codebase (CPU and single GPU). It is parameterised on:
//
//   1. A `Backend` (from `ed/matvec/backend.h`) that provides the
//      linear-algebra plane: alloc, dot/axpy/scale/nrm2 and batched
//      dot_many/axpy_many for CGS2 reorth.
//   2. A `MatvecFn` callable (`void(const Complex*, Complex*, size_t)`)
//      that knows how to apply H to a vector. The callable hides any
//      cuBLAS handle / stream wiring from the kernel.
//
// Algorithmic features:
//
//   * Three-term Lanczos recurrence with swap-rotated working vectors
//     (no per-iter memcpy).
//   * Optional full reorthogonalisation via classical Gram-Schmidt 2
//     (CGS2). CGS2 has the same numerical quality as modified
//     Gram-Schmidt but batches the M projections of each pass into one
//     dot_many / axpy_many call.
//   * Breakdown detection via beta < tol.
//   * Optional basis-vector retention for downstream FTLM / 
//     observable-projection consumers.
//
// Pointer-convention notes:
//
//   * Every `Complex*` in this file is in the backend's memory space
//     (host RAM for `CpuBackend`, device memory for `CudaBackend`).
//     Callers must NOT mix host pointers into a GPU backend, nor vice
//     versa --- there is no defensive copy_to/from_host in here. The
//     `Backend::make_zero_vector` helper produces a properly-spaced
//     pointer.
//   * `local_n` is the vector length (the problem dimension on the CPU
//     and single-GPU backends).
//
// This header is host-only (no CUDA-only types). The GPU backend
// implements its Backend in a .cuh file; this header is shared by all
// backends.
// =============================================================================

#include <ed/config/env_registry.h>
#include <ed/core/log.h>
#include <algorithm>
#include <chrono>     // ED_LANCZOS_KERNEL_PROFILE wallclock timers
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>    // getenv
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ed/matvec/backend.h>

namespace ed::krylov {

using Complex = std::complex<double>;

/// Reorthogonalisation policy.
enum class ReorthPolicy : std::uint8_t {
    /// No reorth: pure three-term recurrence. Cheap but accumulates
    /// loss of orthogonality for M > ~30; suitable only for very
    /// short Lanczos sequences or eigenvalues-only runs where
    /// "ghost" Ritz pairs are acceptable.
    None,
    /// Full reorth against every stored basis vector via classical
    /// Gram-Schmidt 2 (CGS2). Requires `keep_basis = true` and
    /// O(M^2 * local_n) work (two batched passes per step).
    FullCGS2,
    /// Local DGKS-3 reorth: ring buffer of up to 3 most-recent basis
    /// vectors, project `w` against each via thresholded `axpy`. Does
    /// NOT require `keep_basis = true` (the kernel owns the ring
    /// internally). Cheap (constant per-iter cost; never grows) but
    /// only counteracts the immediate-neighbor loss-of-orthogonality
    /// that three-term recurrence accrues -- accumulated drift past
    /// ~30 iterations still appears.
    LocalDGKS3,
};

/// Options for `lanczos_kernel`.
struct LanczosKernelOptions {
    std::size_t max_iter = 100;          ///< Krylov dimension cap.

    /// Genuine-invariant-subspace breakdown threshold on beta_{j+1}.
    /// Default is the smallest value for which `1/bnext` is still
    /// representable (so we only break when w is mathematically zero,
    /// not just numerically small). Ritz convergence is a separate
    /// test, handled by the `convergence_check` callback below;
    /// `breakdown_tol` (this field) is *only* the invariant-subspace
    /// detection threshold. Callers that want the bare recurrence to
    /// stop on a small residual (||w|| < tol) set it explicitly.
    double      breakdown_tol = 1e-300;

    ReorthPolicy reorth  = ReorthPolicy::FullCGS2;
    bool keep_basis      = true;         ///< Retain orthonormal basis (req'd for reorth ≠ None).

    /// Optional Ritz-convergence early-exit callback. Called every
    /// `convergence_check_interval` iterations (after the kernel has
    /// pushed `alpha[j]` and `beta[j+1]`, AFTER any reorthogonalisation,
    /// and BEFORE the swap-rotate that prepares iteration j+1). If the
    /// callback returns true, the kernel terminates: the current step
    /// IS counted in `iters_done` and the alpha/beta/basis arrays are
    /// returned in their post-iteration state.
    ///
    /// `alpha.size() == j + 1` and `beta.size() == j + 2` (with the
    /// sentinel `beta[0] == 0.0`) on entry to the callback.
    ///
    /// Default `nullptr` (no early exit) -- the kernel runs to the
    /// `cap = min(max_iter, local_n)` bound or until breakdown.
    ///
    /// Typical use: solve the running tridiagonal (`tridiag_eig`) and test
    /// the Paige bound |beta_m z_{m,j}| of the wanted Ritz values, as the
    /// block lanes' k = 1 scan does (lg_block_solve.cpp).
    std::function<bool(const std::vector<double>& alpha,
                       const std::vector<double>& beta)>
        convergence_check;

    /// Stride (in iterations) between `convergence_check` invocations.
    /// 0 disables the check entirely (the kernel doesn't call the
    /// callback). Default 0. Typical values: 1 (check every step --
    /// expensive but most responsive), 10 (cheap stability check at the
    /// cost of a few extra Krylov iterations near convergence).
    std::size_t convergence_check_interval = 0;

    /// Optional per-iteration hook. Invoked AFTER the swap-rotate has
    /// produced V_{j+1}:
    ///
    ///   * `iteration_count == j + 1` (number of completed iterations).
    ///   * `alpha.size() == j + 1`, `beta.size() == j + 2`.
    ///   * `v_curr` points to V_{j+1} (the next vector that will be
    ///     used in iteration j+1) in backend memory; `v_prev` points
    ///     to V_j. Both buffers MUST be treated as read-only by the
    ///     hook --- the kernel reuses them in the next iteration.
    ///
    /// Used by the two-pass eigenvector reconstruction in the solve lane
    /// (the replayed recurrence streams V_j past the hook). On the LAST
    /// iteration (j + 1 == cap) the rotation happens but no further
    /// matvec is done; the hook still fires.
    ///
    /// Default `nullptr` (no hook called).
    std::function<void(std::size_t iteration_count,
                       const std::vector<double>& alpha,
                       const std::vector<double>& beta,
                       const Complex* v_curr,
                       const Complex* v_prev,
                       std::size_t local_n)>
        on_step;

    /// Stride (in iterations) between `on_step` invocations. 0 disables
    /// the hook entirely (the kernel never calls the callback even when
    /// it is non-null). Default 0.
    std::size_t on_step_interval = 0;

    /// Optional FIXED set of vectors (in backend memory) that every
    /// reorthogonalisation pass also projects out of `w`. The set is
    /// closed at construction time (the kernel does not append to it)
    /// and the kernel does not take ownership; the pointers must
    /// remain valid for the duration of the kernel call.
    ///
    /// Use case: thick-restart / Krylov-Schur, where the per-cycle
    /// Lanczos body must be orthogonal to the previously LOCKED Ritz
    /// vectors in addition to the current cycle's basis. Setting this
    /// non-empty AND `reorth != ReorthPolicy::None` causes the kernel
    /// to project against `aux_ortho_ptrs` ∪ basis_ptrs on every CGS2
    /// pass via the same batched `dot_many` / `axpy_many` primitives;
    /// the per-step batched call count is unchanged (still 2 per step in
    /// the CGS2 case — one per pass over the combined set).
    ///
    /// Contract on `v0_local`: the kernel takes v0 AS-IS as the first
    /// basis vector V_0 (only re-normalises). It does NOT project v0
    /// against `aux_ortho_ptrs`. The caller is responsible for handing
    /// the kernel a v0 that is already orthogonal to every entry of
    /// `aux_ortho_ptrs`; the per-step CGS2 then keeps V_1, V_2, ...
    /// orthogonal to both `aux_ortho_ptrs` and to V_0. The Krylov-Schur
    /// kernel pre-orthogonalises its restart seed against the locked
    /// Ritz set before entering the per-cycle Lanczos.
    ///
    /// Default empty (the kernel only orthogonalises against its own
    /// growing basis). NOT consulted when `reorth == None`.
    std::vector<const Complex*> aux_ortho_ptrs;

    /// Number of recent basis vectors the LocalDGKS3 ring buffer
    /// retains. Range: 1..N. Only consulted when
    /// ``reorth == LocalDGKS3``.
    ///
    /// Default 1: K=1 local DGKS is the optimum
    /// for our Krylov dimensions on real-Hermitian / well-conditioned
    /// spectra (validated to 1e-9 against xdiag). Set this field
    /// explicitly when constructing options manually for a wider
    /// ring. Raise it for problems with
    /// near-degenerate ground states where loss of orthogonality
    /// across a small window is observable.
    std::size_t local_ring_size = 1;

    /// Threshold below which a LocalDGKS3 projection is skipped (the
    /// resulting correction sits below the round-off floor). Default
    /// sqrt(eps) ~= 1.49e-8.
    double local_ortho_threshold = 1.49011611938476562e-08;
};

/// Output of `lanczos_kernel`. `basis` is populated iff
/// `opts.keep_basis == true`; ownership transfers to the caller.
struct LanczosKernelResult {
    /// Diagonal of the tridiagonal matrix, size = `iters_done`.
    std::vector<double>              alpha;
    /// Sub-diagonal of the tridiagonal matrix, size = `iters_done + 1`.
    /// `beta[0]` is unused, so `beta[j]` couples `alpha[j-1]` and `alpha[j]`.
    std::vector<double>              beta;
    /// Orthonormal Krylov basis in backend memory. Each vector is
    /// dimension `local_n`. Empty iff `opts.keep_basis == false`.
    std::vector<ed::matvec::Backend::UniqueVec> basis;
    /// Number of iterations actually completed (may be less than
    /// `opts.max_iter` if Lanczos broke down via beta < tol).
    std::size_t                      iters_done = 0;
};

// ---------------------------------------------------------------------------
// THE KERNEL.
// ---------------------------------------------------------------------------

/// Run a Lanczos iteration on `H` starting from `v0_local` (dimension
/// `local_n`). The matvec callable signature is
///
///     void matvec(const Complex* in, Complex* out, std::size_t local_n);
///
/// where pointers are in `be`'s memory space.
///
/// Throws `std::invalid_argument` if `v0_local` has zero norm or if a
/// reorth policy other than `None` is requested without `keep_basis`.
/// DGKS "twice is enough" threshold for the CGS2 reprojection: a second
/// pass is performed only if the first removed more than a fraction
/// 1 - kappa of the norm (kappa = 1/sqrt(2) is the classical choice).
inline constexpr double kDgksKappa = 0.7071067811865476;

template <typename MatvecFn>
LanczosKernelResult lanczos_kernel(
    const ed::matvec::Backend& be,
    MatvecFn&& matvec,
    std::size_t local_n,
    const Complex* v0_local,
    const LanczosKernelOptions& opts)
{
    using ed::matvec::Backend;

    // ------------------------------------------------------------------
    // Per-bucket microsecond timers, opt-in via env
    // ``ED_LANCZOS_KERNEL_PROFILE=1``. Zero cost when the env is unset
    // (the gate is a single getenv at kernel entry, the per-iter cost
    // is just `if (profile_on) accumulate`).
    // ------------------------------------------------------------------
    const bool profile_on = []() {
        return ed::env::flag("ED_LANCZOS_KERNEL_PROFILE", false);
    }();
    auto now_us = [] {
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t_apply_us = 0.0, t_recur_us = 0.0, t_reorth_us = 0.0;
    double t_norm_us  = 0.0, t_ring_us  = 0.0, t_check_us  = 0.0;
    std::size_t iters_profiled = 0;
    const double kernel_t0 = profile_on ? now_us() : 0.0;

    // FullCGS2 requires keep_basis (we project against the growing basis).
    // LocalDGKS3 owns its own ring buffer and does NOT require keep_basis.
    const bool needs_kept_basis = opts.reorth == ReorthPolicy::FullCGS2;
    if (needs_kept_basis && !opts.keep_basis) {
        throw std::invalid_argument(
            "lanczos_kernel: FullCGS2 requires keep_basis = true");
    }
    if (opts.max_iter == 0) {
        return LanczosKernelResult{};
    }
    // NB: `local_n == 0` does not short-circuit here. All host BLAS-1
    // locals below operate trivially on a length-0 buffer; the backend's
    // `make_zero_vector` is expected to handle `n == 0` (CpuBackend does).

    LanczosKernelResult R;
    R.alpha.reserve(opts.max_iter);
    R.beta.reserve(opts.max_iter + 1);

    // Working vectors. Three are mandatory; w doubles as the SpMV
    // output target. v0 is copied into a fresh v_curr.
    auto v_prev = be.make_zero_vector(local_n);
    auto v_curr = be.make_zero_vector(local_n);
    auto w = be.make_zero_vector(local_n);

    // Ring buffer for LocalDGKS3 reorth with K >= 3. Implemented as a
    // vector of backend-owned vectors plus an explicit (head, count)
    // pair to avoid the cost of an erase()/insert() on the hot path.
    // Sized to opts.local_ring_size when LocalDGKS3 is active.
    std::vector<Backend::UniqueVec> ring;
    std::size_t                     ring_head  = 0;
    std::size_t                     ring_count = 0;

    // Optional basis storage. We keep TWO parallel structures:
    //
    //   `basis`      : ownership (UniqueVec per vector),
    //   `basis_ptrs` : raw pointers, used by dot_many / axpy_many.
    //
    // Keeping them in sync requires that we never reallocate `basis`
    // while `basis_ptrs` is being read --- so we reserve up front.
    std::vector<Backend::UniqueVec> basis;
    std::vector<const Complex*>     basis_ptrs;

    const std::size_t expected_total = opts.max_iter;

    R.beta.push_back(0.0);  // beta[0] unused (kept for index alignment).

    // Normalise the initial vector in-place (we own a fresh copy).
    be.copy(v0_local, v_curr.get(), local_n);
    const double v0_norm = be.nrm2(v_curr.get(), local_n);
    if (!(v0_norm > 0.0)) {
        throw std::invalid_argument(
            "lanczos_kernel: initial vector has non-positive norm");
    }
    be.scale(Complex(1.0 / v0_norm, 0.0), v_curr.get(), local_n);

    if (opts.keep_basis) {
        basis.reserve(expected_total);
        basis_ptrs.reserve(expected_total);
        auto first = be.make_zero_vector(local_n);
        be.copy(v_curr.get(), first.get(), local_n);
        basis_ptrs.push_back(first.get());
        basis.emplace_back(std::move(first));
    }
    const bool ring_needed =
        (opts.reorth == ReorthPolicy::LocalDGKS3 && opts.local_ring_size > 2);
    if (ring_needed) {
        ring.reserve(opts.local_ring_size);
        auto first = be.make_zero_vector(local_n);
        be.copy(v_curr.get(), first.get(), local_n);
        ring.emplace_back(std::move(first));
        ring_count = 1;
        ring_head  = 0;
    }

    // Reorth pointer set. When `aux_ortho_ptrs` is non-empty (the
    // Krylov-Schur thick-restart case), each CGS2 pass projects `w`
    // against `aux_ortho_ptrs` ∪ basis_ptrs in a single batched
    // dot_many / axpy_many. We materialise that union once
    // (`ortho_ptrs`) and append the new basis pointer to it on each
    // iteration; that way the batched primitives stay one call
    // per pass regardless of how the caller splits the work between
    // aux and basis.
    const std::size_t n_aux = opts.aux_ortho_ptrs.size();
    std::vector<const Complex*> ortho_ptrs;
    if (needs_kept_basis) {
        ortho_ptrs.reserve(n_aux + expected_total);
        for (const Complex* p : opts.aux_ortho_ptrs) ortho_ptrs.push_back(p);
        for (auto* p : basis_ptrs) ortho_ptrs.push_back(p);
    }

    // Scratch for CGS2 coefficients.
    std::vector<Complex> coeffs;
    if (needs_kept_basis) {
        coeffs.reserve(n_aux + expected_total);
    }

    // A Krylov space never exceeds the dimension of the space.
    const std::size_t cap = std::min<std::size_t>(opts.max_iter, local_n);

    for (std::size_t j = 0; j < cap; ++j) {
        const double t0 = profile_on ? now_us() : 0.0;
        // w = H * v_curr (matvec is opaque to this kernel --- it may
        // be a host term-matvec, a cuBLAS-backed SpMV, etc.)
        matvec(v_curr.get(), w.get(), local_n);
        const double t1 = profile_on ? now_us() : 0.0;
        if (profile_on) t_apply_us += (t1 - t0);

        // Fused recurrence: the four
        // BLAS-1 sweeps {w -= beta v_prev; alpha = <v_curr,w>; w -= alpha
        // v_curr; overlap = <v_curr,w>} become two single-pass calls, and
        // for LocalDGKS3 with K <= 2 the projection onto v_curr is folded
        // into the norm pass below (three fused regions per iteration).
        const Complex aj = (j > 0)
            ? be.axpy_dot(Complex(-R.beta[j], 0.0), v_prev.get(), w.get(),
                          v_curr.get(), local_n)
            : be.dot(v_curr.get(), w.get(), local_n);
        R.alpha.push_back(aj.real());

        const bool fuse_local_k12 =
            (opts.reorth == ReorthPolicy::LocalDGKS3) &&
            (opts.local_ring_size <= 2) && (j > 0);
        Complex overlap_curr{0.0, 0.0};
        if (fuse_local_k12) {
            // w -= alpha[j] * v_curr, and the LocalDGKS3 overlap <v_curr, w>
            overlap_curr = be.axpy_dot(Complex(-R.alpha[j], 0.0), v_curr.get(),
                                       w.get(), v_curr.get(), local_n);
        } else {
            // w -= alpha[j] * v_curr
            be.axpy(Complex(-R.alpha[j], 0.0),
                    v_curr.get(), w.get(), local_n);
        }
        // Deferred projection coefficient folded into the norm pass (K == 1).
        bool    defer_axpy = false;
        Complex defer_coef{0.0, 0.0};
        const double t2 = profile_on ? now_us() : 0.0;
        if (profile_on) t_recur_us += (t2 - t1);

        // Reorthogonalisation. Three policies share the same
        // dispatch site:
        //
        //   FullCGS2: CGS2 against `aux_ortho_ptrs ∪
        //     basis_ptrs` (two passes; two batched dot_many +
        //     axpy_many per step).
        //
        //   LocalDGKS3: pointwise `dot`/`axpy` against the ring
        //     buffer of up to local_ring_size most-recent basis
        //     vectors. Below threshold the axpy is skipped.
        //
        //   None: nothing.
        const bool do_cgs2 = opts.reorth == ReorthPolicy::FullCGS2;
        if (do_cgs2 && !ortho_ptrs.empty()) {
            coeffs.resize(ortho_ptrs.size());

            // ----- CGS2 pass 1 -----
            const double n_before = be.nrm2(w.get(), local_n);
            be.dot_many(ortho_ptrs.data(), ortho_ptrs.size(),
                        w.get(), local_n, coeffs.data());
            for (auto& c : coeffs) c = -c;
            be.axpy_many(coeffs.data(), ortho_ptrs.data(),
                         ortho_ptrs.size(), w.get(), local_n);

            // ----- CGS2 pass 2 (reprojection), DGKS-gated -----
            // "Twice is enough" (Daniel-Gragg-Kaufman-Stewart): a second
            // projection is only needed when the first one removed a
            // substantial part of w. Two norms (two sweeps over w) are
            // cheaper than an unconditional 2m-sweep second pass.
            const double n_after = be.nrm2(w.get(), local_n);
            if (n_after < kDgksKappa * n_before) {
                be.dot_many(ortho_ptrs.data(), ortho_ptrs.size(),
                            w.get(), local_n, coeffs.data());
                for (auto& c : coeffs) c = -c;
                be.axpy_many(coeffs.data(), ortho_ptrs.data(),
                             ortho_ptrs.size(), w.get(), local_n);
            }
        } else if (opts.reorth == ReorthPolicy::LocalDGKS3) {
            // For the common cases K=1 and K=2 the kernel already holds
            // the basis vectors we need to project against in
            // ``v_curr`` (= V_j) and ``v_prev`` (= V_{j-1}). Use them
            // directly and skip the ring buffer entirely, avoiding the
            // per-iter O(n) ``be.copy(v_curr -> ring[slot])``. For K>=3
            // we still need V_{j-2} and beyond, so use the ring storage.
            const std::size_t K = opts.local_ring_size;
            if (K <= 2) {
                // K>=1: project against V_j (= v_curr). The overlap was
                // computed in the fused recurrence pass above; for K == 1
                // the projection itself is folded into the norm pass.
                if (j > 0) {
                    const Complex overlap = fuse_local_k12
                        ? overlap_curr
                        : be.dot(v_curr.get(), w.get(), local_n);
                    if (std::abs(overlap) > opts.local_ortho_threshold) {
                        if (K == 1) {
                            defer_axpy = true;
                            defer_coef = -overlap;
                        } else {
                            be.axpy(-overlap, v_curr.get(), w.get(), local_n);
                        }
                    }
                }
                // K==2: also project against V_{j-1} (= v_prev) when
                // available. At j == 0 there is no prior basis vector.
                if (K == 2 && j > 0) {
                    const Complex overlap =
                        be.dot(v_prev.get(), w.get(), local_n);
                    if (std::abs(overlap) > opts.local_ortho_threshold) {
                        be.axpy(-overlap, v_prev.get(), w.get(), local_n);
                    }
                }
            } else if (ring_count > 0) {
                // K>=3: fall back to the ring buffer (still updated
                // below). Walks most-recent-first, threshold-gated.
                const std::size_t cap_ring = ring.size();
                const std::size_t k_max    = std::min<std::size_t>(
                    ring_count, opts.local_ring_size);
                for (std::size_t k = 0; k < k_max; ++k) {
                    const std::size_t slot =
                        (ring_head + ring_count - 1 - k) % cap_ring;
                    const Complex overlap =
                        be.dot(ring[slot].get(), w.get(), local_n);
                    if (std::abs(overlap) > opts.local_ortho_threshold) {
                        be.axpy(-overlap, ring[slot].get(), w.get(), local_n);
                    }
                }
            }
        }

        const double t3 = profile_on ? now_us() : 0.0;
        if (profile_on) t_reorth_us += (t3 - t2);

        // beta[j+1] = ||w||, fused with the deferred K == 1 projection when there is one.
        const double bnext = defer_axpy
            ? be.axpy_nrm2(defer_coef, v_curr.get(), w.get(), local_n)
            : be.nrm2(w.get(), local_n);
        R.beta.push_back(bnext);
        const double t4 = profile_on ? now_us() : 0.0;
        if (profile_on) t_norm_us += (t4 - t3);

        // Lanczos breakdown: an invariant subspace has been found.
        // We use `breakdown_tol` (default ~1e-300, i.e. essentially
        // "exact zero") rather than the user-facing Ritz `tol`,
        // because at full Krylov dimension the residual collapses to
        // O(eps * ||H||) but the Krylov subspace is still valid (and
        // downstream consumers like FTLM rely on the full tridiag).
        // See the LanczosKernelOptions::breakdown_tol comment.
        if (bnext < opts.breakdown_tol) break;

        // Compute V_{j+1} = w / beta_{j+1} and rotate, BEFORE the
        // on_step / convergence_check hooks. After the rotation:
        //   v_prev = V_j, v_curr = V_{j+1}, w = scratch.
        be.scale(Complex(1.0 / bnext, 0.0), w.get(), local_n);
        std::swap(v_prev, v_curr);
        std::swap(v_curr, w);

        // Update the LocalDGKS3 ring buffer with V_{j+1}. Only
        // for K >= 3 -- the K <= 2 reorth path reads v_curr/v_prev directly.
        if (ring_needed) {
            const std::size_t cap_ring = opts.local_ring_size;
            if (ring_count < cap_ring) {
                auto slot = be.make_zero_vector(local_n);
                be.copy(v_curr.get(), slot.get(), local_n);
                ring.emplace_back(std::move(slot));
                ++ring_count;
            } else {
                be.copy(v_curr.get(), ring[ring_head].get(), local_n);
                ring_head = (ring_head + 1) % cap_ring;
            }
        }

        // Per-iteration hook. Fires AFTER rotation so the hook sees
        // v_curr = V_{j+1}, v_prev = V_j; `iteration_count` is `j + 1`.
        if (opts.on_step &&
            opts.on_step_interval > 0 &&
            ((j + 1) % opts.on_step_interval == 0))
        {
            opts.on_step(j + 1, R.alpha, R.beta,
                         v_curr.get(), v_prev.get(), local_n);
        }

        const double t5 = profile_on ? now_us() : 0.0;
        if (profile_on) t_ring_us += (t5 - t4);

        // Optional Ritz-convergence early-exit. We run the callback
        // AFTER on_step (so the hook sees every completed step, including
        // the one that triggered the break) but BEFORE the next
        // matvec. See the `LanczosKernelOptions::convergence_check`
        // comment for the exact alpha/beta sizes on entry.
        if (opts.convergence_check &&
            opts.convergence_check_interval > 0 &&
            ((j + 1) % opts.convergence_check_interval == 0))
        {
            const double tc0 = profile_on ? now_us() : 0.0;
            const bool converged = opts.convergence_check(R.alpha, R.beta);
            if (profile_on) t_check_us += (now_us() - tc0);
            if (converged) break;
        }

        if (profile_on) ++iters_profiled;

        if (j + 1 >= cap) break;

        // Append V_{j+1} to the kept basis, but only AFTER we've
        // committed to running iteration j+1 (i.e., neither
        // convergence_check nor cap-hit broke out above). This keeps
        // the invariant `basis.size() == iters_done` so downstream
        // consumers (Krylov-Schur restart, Ritz reconstruction,
        // FTLM tridiag) see a basis of the right length.
        if (opts.keep_basis) {
            auto next = be.make_zero_vector(local_n);
            be.copy(v_curr.get(), next.get(), local_n);
            basis_ptrs.push_back(next.get());
            basis.emplace_back(std::move(next));
            if (needs_kept_basis) {
                ortho_ptrs.push_back(basis_ptrs.back());
            }
        }
    }

    R.iters_done = R.alpha.size();
    if (opts.keep_basis) R.basis = std::move(basis);

    if (profile_on) {
        const double t_total = now_us() - kernel_t0;
        const double t_other = std::max(0.0,
            t_total - t_apply_us - t_recur_us - t_reorth_us
                    - t_norm_us  - t_ring_us  - t_check_us);
        const std::size_t iters = R.iters_done;
        const double inv_iters = (iters > 0)
            ? 1.0 / static_cast<double>(iters) : 0.0;
        const auto pct = [&](double x) -> double {
            return (t_total > 0.0) ? 100.0 * x / t_total : 0.0;
        };
        ED_LOG(Info,
            "[lanczos_kernel] iters=%zu total=%.2f ms = "
            "apply %.1f%% (%.1f us/it) "
            "recur %.1f%% (%.1f us/it) "
            "reorth %.1f%% (%.1f us/it) "
            "norm %.1f%% (%.1f us/it) "
            "ring %.1f%% (%.1f us/it) "
            "check %.1f%% (%.1f us/it) "
            "other %.1f%%",
            iters, t_total / 1000.0,
            pct(t_apply_us),  t_apply_us  * inv_iters,
            pct(t_recur_us),  t_recur_us  * inv_iters,
            pct(t_reorth_us), t_reorth_us * inv_iters,
            pct(t_norm_us),   t_norm_us   * inv_iters,
            pct(t_ring_us),   t_ring_us   * inv_iters,
            pct(t_check_us),  t_check_us  * inv_iters,
            pct(t_other));
        (void)iters_profiled;  // available for callers that want it
    }

    return R;
}

} // namespace ed::krylov
