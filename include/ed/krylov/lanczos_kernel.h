#pragma once
// =============================================================================
// include/ed/krylov/lanczos_kernel.h
//
// THE single Lanczos algorithm.
//
// One template-function body drives all four deployment targets in this
// codebase (CPU, single GPU, CPU+MPI, GPU+MPI). It is parameterised on:
//
//   1. A `Backend` (from `ed/matvec/backend.h`) that provides the
//      linear-algebra plane: alloc, dot/axpy/scale/nrm2, batched
//      dot_many/axpy_many for CGS2 reorth, and the no-op-or-Allreduce
//      reductions that turn a single-process kernel into a distributed
//      kernel without touching the algorithm body.
//   2. A `MatvecFn` callable (`void(const Complex*, Complex*, size_t)`)
//      that knows how to apply H to a rank-local slab. The callable
//      hides any halo exchange / cuBLAS handle / NCCL stream wiring
//      from the kernel.
//
// Algorithmic features (matching the previous serial canonical body
// `build_lanczos_tridiagonal_with_basis` in src/solvers/cpu/lanczos.cpp,
// modulo the upgrade from sequential MGS to batched CGS2):
//
//   * Three-term Lanczos recurrence with swap-rotated working vectors
//     (no per-iter memcpy).
//   * Optional full reorthogonalisation via classical Gram-Schmidt 2
//     (CGS2). CGS2 has the same numerical quality as modified
//     Gram-Schmidt but uses ONE batched Allreduce per pass instead of
//     M sequential Allreduces --- the headline performance win for
//     CPU-MPI and (in Phase B) GPU-MPI Lanczos.
//   * Optional periodic reorthogonalisation (every `reorth_freq`
//     iterations) on the same CGS2 path.
//   * Breakdown detection via beta < tol.
//   * Optional basis-vector retention for downstream FTLM / LTLM /
//     observable-projection consumers.
//
// Pointer-convention notes:
//
//   * Every `Complex*` in this file is in the backend's memory space
//     (host RAM for `CpuBackend` / `MpiBackend`, device memory for the
//     future `CudaBackend` / `MpiCudaBackend`). Callers must NOT mix
//     host pointers into a GPU backend, nor vice versa --- there is no
//     defensive copy_to/from_host in here. The `Backend::make_zero_vector`
//     helper produces a properly-spaced pointer.
//   * `local_n` is the rank-local slab dimension. Global dimension N
//     only enters as a convergence/breakdown sanity check (not used here).
//
// This header is host-only (no CUDA-only types). The GPU backends will
// implement their own Backend specialisations in cuh files; this file
// remains a single compilation unit shared by everyone.
// =============================================================================

#include <algorithm>
#include <chrono>     // Wave 5.1: ED_LANCZOS_KERNEL_PROFILE wallclock timers
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>     // Wave 5.1: profile summary to stderr
#include <cstdlib>    // Wave 5.1: getenv
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
    /// O(M^2 * local_n) work + one Allreduce per pass per step in
    /// the distributed case (2 Allreduces per step for CGS2).
    FullCGS2,
    /// Periodic reorth (every `reorth_freq` steps) using the CGS2
    /// machinery. Same arithmetic as Full but skipped on most steps.
    PeriodicCGS2,
    /// Local DGKS-3 reorth: ring buffer of up to 3 most-recent basis
    /// vectors, project `w` against each via thresholded `axpy`. Does
    /// NOT require `keep_basis = true` (the kernel owns the ring
    /// internally). Cheap (constant per-iter cost; never grows) but
    /// only counteracts the immediate-neighbor loss-of-orthogonality
    /// that three-term recurrence accrues -- accumulated drift past
    /// ~30 iterations still appears. Matches the legacy main-body
    /// `lanczos()` CPU code path that this kernel replaces.
    LocalDGKS3,
};

/// In-memory resume state for re-entering `lanczos_kernel` after a
/// previous run was interrupted (or for chaining two runs with
/// different `max_iter`s). Mirrors the on-disk
/// `ed::io::lanczos_io::LanczosCheckpoint` schema but lives entirely
/// in backend memory so the kernel doesn't have to round-trip through
/// the I/O layer on every resume.
///
/// Bridge functions in `ed/io/lanczos_checkpoint.h` convert between
/// the host-side `LanczosCheckpoint` (with `ComplexVector` payload)
/// and a `LanczosResumeState` (with `Backend::UniqueVec` payload).
///
/// On entry to the kernel:
///   * `alpha.size() == j_start`
///   * `beta.size()  == j_start + 1` (beta[0] is the legacy sentinel)
///   * `v_curr`, `v_prev` are dimension-`local_n` backend vectors
///   * `ring_vectors` holds up to 3 most-recent basis vectors when the
///     reorth policy is `LocalDGKS3`; ignored otherwise.
///   * `basis` holds every basis vector V_0..V_{j_start - 1} when the
///     resume run is replaying a FullCGS2 / PeriodicCGS2 run; the
///     kernel will own these going forward. May be empty for
///     LocalDGKS3 / None policies (the kernel doesn't need them).
struct LanczosResumeState {
    std::size_t                                  j_start = 0;
    std::vector<double>                          alpha;
    std::vector<double>                          beta;
    ed::matvec::Backend::UniqueVec               v_curr;
    ed::matvec::Backend::UniqueVec               v_prev;
    std::vector<ed::matvec::Backend::UniqueVec>  ring_vectors;
    std::vector<ed::matvec::Backend::UniqueVec>  basis;
};

/// Options for `lanczos_kernel`.
struct LanczosKernelOptions {
    std::size_t max_iter = 100;          ///< Krylov dimension cap.

    /// Genuine-invariant-subspace breakdown threshold on beta_{j+1}.
    /// Default is the smallest value for which `1/bnext` is still
    /// representable (so we only break when w is mathematically zero,
    /// not just numerically small).
    ///
    /// Rationale: the *user-facing* `tol` in legacy entry points
    /// (`build_lanczos_tridiagonal_with_basis(..., tol, ...)`) is a
    /// Ritz-convergence parameter, NOT a breakdown threshold. The
    /// legacy MGS-once code conflated the two: it broke when
    /// `||w|| < tol`, but at full Krylov dimension the MGS noise
    /// floor (~1e-10) sat above the typical tol=1e-12, so it never
    /// triggered. CGS2 has a much lower noise floor (~1e-14), so the
    /// same threshold would break early -- and the resulting Ritz
    /// truncation would corrupt downstream observables (LTLM static-
    /// connected-Q-H test). The kernel decouples the two: Ritz
    /// convergence is handled by the `convergence_check` callback
    /// below; `breakdown_tol` (this field) is *only* the
    /// invariant-subspace detection threshold.
    double      breakdown_tol = 1e-300;

    ReorthPolicy reorth  = ReorthPolicy::FullCGS2;
    std::size_t reorth_freq = 10;        ///< Used iff reorth == PeriodicCGS2.
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
    /// Typical use: solve the small running tridiagonal and check
    /// `|Δλ_smallest| / max(|λ_smallest|, 1e-300) < ritz_tol` against
    /// the prior invocation. See `ed/krylov/ritz_convergence.h` for a
    /// stateful predicate factory that does exactly that.
    std::function<bool(const std::vector<double>& alpha,
                       const std::vector<double>& beta)>
        convergence_check;

    /// Stride (in iterations) between `convergence_check` invocations.
    /// 0 disables the check entirely (the kernel doesn't call the
    /// callback). Default 0. Typical values: 1 (check every step --
    /// expensive but most responsive), 5 (the historical
    /// `distributed_lanczos` cadence), 10 (cheap stability check at the
    /// cost of a few extra Krylov iterations near convergence).
    std::size_t convergence_check_interval = 0;

    /// Optional per-iteration checkpoint hook. Invoked AFTER the swap-
    /// rotate has produced V_{j+1}, so the hook sees the same state
    /// the legacy `lanczos.cpp` checkpoint stored:
    ///
    ///   * `iteration_count == j + 1` (number of completed iterations,
    ///     matching the legacy `cp.iteration` field).
    ///   * `alpha.size() == j + 1`, `beta.size() == j + 2`.
    ///   * `v_curr` points to V_{j+1} (the next vector that will be
    ///     used in iteration j+1) in backend memory; `v_prev` points
    ///     to V_j. Both buffers MUST be treated as read-only by the
    ///     hook --- the kernel reuses them in the next iteration.
    ///
    /// Designed for `lanczos.cpp`-style checkpoint / restart: the hook
    /// can serialise (alpha, beta, V_{j+1}, V_j, RNG state) to disk
    /// every `on_step_interval` steps and the calling context can
    /// resume from a kernel re-entry whose `LanczosResumeState` has
    /// `j_start = iteration_count`, `v_curr = V_{iteration_count}`,
    /// `v_prev = V_{iteration_count - 1}`.
    ///
    /// Note: on the LAST iteration (j + 1 == cap), the rotation
    /// happens but no further matvec is done; the hook still fires
    /// so a checkpoint includes the final V_{cap}.
    ///
    /// The `ring_view` argument is a pointer to the current LocalDGKS3
    /// ring buffer, in chronological order (oldest first), or nullptr
    /// when the reorth policy is not LocalDGKS3. Lifetime: only valid
    /// for the duration of the callback. The hook can use this view
    /// to persist the ring state into a checkpoint so a future resume
    /// starts with full local-reorth quality from iter 0.
    ///
    /// Default `nullptr` (no hook called).
    std::function<void(std::size_t iteration_count,
                       const std::vector<double>& alpha,
                       const std::vector<double>& beta,
                       const Complex* v_curr,
                       const Complex* v_prev,
                       std::size_t local_n,
                       const std::vector<const Complex*>* ring_view)>
        on_step;

    /// Stride (in iterations) between `on_step` invocations. 0 disables
    /// the hook entirely (the kernel never calls the callback even when
    /// it is non-null). Default 0.
    std::size_t on_step_interval = 0;

    /// Upper bound on the Krylov subspace dimension imposed by the
    /// problem geometry. The kernel's effective iteration cap is
    ///
    ///     cap = min(max_iter, dim_cap > 0 ? dim_cap : local_n)
    ///
    /// Default 0 means "use `local_n`", which is the correct serial /
    /// single-GPU value (`local_n == global_dim` in those backends).
    /// Distributed backends MUST set this to the global problem
    /// dimension, otherwise on a rank with a small slab (e.g. global
    /// dim 6 split across 4 ranks gives `local_n ∈ {1,2}`) the kernel
    /// would silently terminate after a single iteration. The legacy
    /// `distributed_lanczos` body had no such cap; it relied on
    /// `max_iter` plus the breakdown check, and we preserve that
    /// semantics by letting MPI callers pass the global dim here.
    std::size_t dim_cap = 0;

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
    /// the per-step Allreduce count is unchanged (still 2 per step in
    /// the CGS2 case — one per pass over the combined set).
    ///
    /// Contract on `v0_local`: the kernel takes v0 AS-IS as the first
    /// basis vector V_0 (only re-normalises). It does NOT project v0
    /// against `aux_ortho_ptrs`. The caller is responsible for handing
    /// the kernel a v0 that is already orthogonal to every entry of
    /// `aux_ortho_ptrs`; the per-step CGS2 then keeps V_1, V_2, ...
    /// orthogonal to both `aux_ortho_ptrs` and to V_0. This matches
    /// the historical thick-restart KS body, which pre-orthogonalises
    /// its restart seed against the locked Ritz set before entering
    /// the per-cycle Lanczos.
    ///
    /// Default empty (the kernel only orthogonalises against its own
    /// growing basis). NOT consulted when `reorth == None`.
    std::vector<const Complex*> aux_ortho_ptrs;

    /// Number of recent basis vectors the LocalDGKS3 ring buffer
    /// retains. Range: 1..N. Only consulted when
    /// ``reorth == LocalDGKS3``.
    ///
    /// Default 1: K=1 local DGKS matches the legacy `lanczos_real`
    /// fast path (`src/solvers/cpu/lanczos.cpp:1146-1154`) and is
    /// the optimum for our Krylov dimensions on real-Hermitian /
    /// well-conditioned spectra (validated to 1e-9 by the Apr 25
    /// xdiag bake-off + the SOTA-symmetry suite). K=3 was the
    /// pre-Wave-2.1 default and remains reachable via env
    /// ``ED_LANCZOS_REORTH_K`` (read by `lanczos()` /
    /// orchestrator) or by setting this field explicitly when
    /// constructing options manually. Raise it for problems with
    /// near-degenerate ground states where loss of orthogonality
    /// across a small window is observable.
    std::size_t local_ring_size = 1;

    /// Threshold below which a LocalDGKS3 projection is skipped (the
    /// resulting correction sits below the round-off floor). Default
    /// sqrt(eps) ~= 1.49e-8, matching the legacy lanczos.cpp constant.
    double local_ortho_threshold = 1.49011611938476562e-08;

    /// Optional in-memory resume state. When non-null, the kernel
    /// skips the v0 init / first-iteration setup and continues from
    /// `state->j_start` with the supplied alpha/beta/v_curr/v_prev
    /// (and ring buffer for LocalDGKS3 / basis for FullCGS2).
    /// Ownership of the state transfers to the kernel on entry; the
    /// kernel populates `R.basis` (and a separate `R.ring_vectors`
    /// after the run, accessible via the `on_step` hook) with the
    /// state at termination.
    ///
    /// The pointer is intentionally raw (not unique_ptr) so the
    /// caller can build the state on the stack and let it die after
    /// the call. The kernel does NOT take ownership of the pointer
    /// itself; only of the unique_ptr-owned vectors inside.
    LanczosResumeState* resume_state = nullptr;
};

/// Output of `lanczos_kernel`. `basis` is populated iff
/// `opts.keep_basis == true`; ownership transfers to the caller.
struct LanczosKernelResult {
    /// Diagonal of the tridiagonal matrix, size = `iters_done`.
    std::vector<double>              alpha;
    /// Sub-diagonal of the tridiagonal matrix, size = `iters_done + 1`.
    /// `beta[0]` is unused (kept for legacy index alignment with the
    /// classic `build_lanczos_tridiagonal_with_basis` ABI).
    std::vector<double>              beta;
    /// Orthonormal Krylov basis in backend memory. Each vector is
    /// dimension `local_n`. Empty iff `opts.keep_basis == false`.
    std::vector<ed::matvec::Backend::UniqueVec> basis;
    /// Final v_curr / v_prev / ring buffer. Owned by the caller. Used
    /// by checkpoint writers so a resumed run can pick up exactly where
    /// this one left off. Only populated when `opts.resume_state` is
    /// non-null OR `opts.on_step` is configured AND the run terminates
    /// before `max_iter` (e.g. via `convergence_check`). The full set
    /// is always emitted when the run terminates by hitting `cap`,
    /// regardless.
    ed::matvec::Backend::UniqueVec   v_curr;
    ed::matvec::Backend::UniqueVec   v_prev;
    std::vector<ed::matvec::Backend::UniqueVec> ring_vectors;
    /// Number of iterations actually completed (may be less than
    /// `opts.max_iter` if Lanczos broke down via beta < tol).
    std::size_t                      iters_done = 0;
};

// ---------------------------------------------------------------------------
// THE KERNEL.
// ---------------------------------------------------------------------------

/// Run a Lanczos iteration on `H` starting from `v0_local` (rank-local
/// dimension `local_n`). The matvec callable signature is
///
///     void matvec(const Complex* in, Complex* out, std::size_t local_n);
///
/// where pointers are in `be`'s memory space.
///
/// Throws `std::invalid_argument` if `v0_local` has zero norm or if a
/// reorth policy other than `None` is requested without `keep_basis`.
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
    // Wave 5.1 of the SOTA Performance rollout (May 2026): per-bucket
    // microsecond timers, opt-in via env ``ED_LANCZOS_KERNEL_PROFILE=1``.
    // Mirrors the same gauges in ``lanczos_real`` so we can A/B the two
    // engines on identical workloads. Zero cost when the env is unset
    // (the gate is a single getenv at kernel entry, the per-iter cost
    // is just `if (profile_on) accumulate`).
    // ------------------------------------------------------------------
    const bool profile_on = []() {
        const char* env = std::getenv("ED_LANCZOS_KERNEL_PROFILE");
        return env && env[0] == '1';
    }();
    auto now_us = [] {
        return std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double t_apply_us = 0.0, t_recur_us = 0.0, t_reorth_us = 0.0;
    double t_norm_us  = 0.0, t_ring_us  = 0.0, t_check_us  = 0.0;
    std::size_t iters_profiled = 0;
    const double kernel_t0 = profile_on ? now_us() : 0.0;

    // FullCGS2/PeriodicCGS2 require keep_basis (we project against the
    // growing basis). LocalDGKS3 owns its own ring buffer and does NOT
    // require keep_basis.
    const bool needs_kept_basis =
        (opts.reorth == ReorthPolicy::FullCGS2) ||
        (opts.reorth == ReorthPolicy::PeriodicCGS2);
    if (needs_kept_basis && !opts.keep_basis) {
        throw std::invalid_argument(
            "lanczos_kernel: FullCGS2 / PeriodicCGS2 require keep_basis = true");
    }
    if (opts.max_iter == 0) {
        return LanczosKernelResult{};
    }
    // NB: `local_n == 0` MUST NOT short-circuit here. In distributed runs
    // it's legitimate for a rank to receive an empty slab (e.g. LPT-balanced
    // orbit partition on a small symmetry sector with global_dim < n_ranks).
    // Such ranks still have to participate in every collective inside the
    // kernel (Allreduce for dot/nrm2, batched Allreduces in CGS2), so the
    // loop body has to run on them. All host BLAS-1 locals below short
    // operate trivially on a length-0 buffer; the backend's `make_zero_vector`
    // is expected to handle `n == 0` (CpuBackend / MpiBackend do).

    LanczosKernelResult R;
    R.alpha.reserve(opts.max_iter);
    R.beta.reserve(opts.max_iter + 1);

    const bool          resuming = (opts.resume_state != nullptr);
    const std::size_t   j_start  = resuming ? opts.resume_state->j_start : 0;

    // Working vectors. Three are mandatory; w doubles as the SpMV
    // output target. On a resume, v_curr / v_prev are handed to us;
    // otherwise we allocate fresh and copy v0 into v_curr.
    Backend::UniqueVec v_prev;
    Backend::UniqueVec v_curr;
    auto w = be.make_zero_vector(local_n);

    // Ring buffer for LocalDGKS3 reorth. Implemented as a vector of
    // backend-owned vectors plus an explicit (head, count) pair to
    // avoid the cost of an erase()/insert() on the hot path. Sized
    // to opts.local_ring_size when LocalDGKS3 is active.
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

    if (resuming) {
        // Adopt the resume state. Vectors are moved out of the state
        // (they live in backend memory already; no extra allocation).
        R.alpha = std::move(opts.resume_state->alpha);
        R.beta  = std::move(opts.resume_state->beta);
        if (R.beta.empty()) R.beta.push_back(0.0);
        v_curr  = std::move(opts.resume_state->v_curr);
        v_prev  = std::move(opts.resume_state->v_prev);
        if (!v_curr || !v_prev) {
            throw std::invalid_argument(
                "lanczos_kernel: resume_state missing v_curr/v_prev");
        }
        if (R.alpha.size() != j_start) {
            throw std::invalid_argument(
                "lanczos_kernel: resume_state alpha.size() != j_start");
        }
        if (R.beta.size() != j_start + 1) {
            throw std::invalid_argument(
                "lanczos_kernel: resume_state beta.size() != j_start + 1");
        }
        if (opts.reorth == ReorthPolicy::LocalDGKS3) {
            ring        = std::move(opts.resume_state->ring_vectors);
            ring_count  = ring.size();
            ring_head   = 0;  // resume convention: head = 0 after the move
        }
        if (opts.keep_basis) {
            basis = std::move(opts.resume_state->basis);
            basis_ptrs.reserve(std::max(basis.size(), expected_total));
            for (auto& bv : basis) basis_ptrs.push_back(bv.get());
        }
    } else {
        v_prev = be.make_zero_vector(local_n);
        v_curr = be.make_zero_vector(local_n);
        R.beta.push_back(0.0);  // beta[0] unused, matches legacy ABI.

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
        if (opts.reorth == ReorthPolicy::LocalDGKS3) {
            ring.reserve(std::max<std::size_t>(opts.local_ring_size, 1));
            auto first = be.make_zero_vector(local_n);
            be.copy(v_curr.get(), first.get(), local_n);
            ring.emplace_back(std::move(first));
            ring_count = 1;
            ring_head  = 0;
        }
    }

    // Reorth pointer set. When `aux_ortho_ptrs` is non-empty (the
    // Krylov-Schur thick-restart case), each CGS2 pass projects `w`
    // against `aux_ortho_ptrs` ∪ basis_ptrs in a single batched
    // dot_many / axpy_many. We materialise that union once
    // (`ortho_ptrs`) and append the new basis pointer to it on each
    // iteration; that way the batched primitives stay one Allreduce
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

    const std::size_t dim_bound =
        (opts.dim_cap > 0) ? opts.dim_cap : local_n;
    const std::size_t cap = std::min<std::size_t>(opts.max_iter, dim_bound);

    for (std::size_t j = j_start; j < cap; ++j) {
        const double t0 = profile_on ? now_us() : 0.0;
        // w = H * v_curr (matvec is opaque to this kernel --- it may
        // be a halo-aware distributed apply, a cuBLAS-backed SpMV, etc.)
        matvec(v_curr.get(), w.get(), local_n);
        const double t1 = profile_on ? now_us() : 0.0;
        if (profile_on) t_apply_us += (t1 - t0);

        // w -= beta[j] * v_prev   (skipped for j == 0; beta[0] is the
        // sentinel zero pushed above).
        if (j > 0) {
            be.axpy(Complex(-R.beta[j], 0.0),
                    v_prev.get(), w.get(), local_n);
        }

        // alpha[j] = <v_curr, w>  (already-reduced for distributed backends)
        const Complex aj = be.dot(v_curr.get(), w.get(), local_n);
        R.alpha.push_back(aj.real());

        // w -= alpha[j] * v_curr
        be.axpy(Complex(-R.alpha[j], 0.0),
                v_curr.get(), w.get(), local_n);
        const double t2 = profile_on ? now_us() : 0.0;
        if (profile_on) t_recur_us += (t2 - t1);

        // Reorthogonalisation. Three policies share the same
        // dispatch site:
        //
        //   FullCGS2 / PeriodicCGS2: CGS2 against `aux_ortho_ptrs ∪
        //     basis_ptrs` (two passes; two batched dot_many +
        //     axpy_many = two Allreduces per step).
        //
        //   LocalDGKS3: pointwise `dot`/`axpy` against the ring
        //     buffer of up to local_ring_size most-recent basis
        //     vectors. Below threshold the axpy is skipped.
        //
        //   None: nothing.
        const bool do_cgs2 =
            (opts.reorth == ReorthPolicy::FullCGS2) ||
            (opts.reorth == ReorthPolicy::PeriodicCGS2 &&
             opts.reorth_freq > 0 && ((j + 1) % opts.reorth_freq == 0));
        if (do_cgs2 && !ortho_ptrs.empty()) {
            coeffs.resize(ortho_ptrs.size());

            // ----- CGS2 pass 1 -----
            be.dot_many(ortho_ptrs.data(), ortho_ptrs.size(),
                        w.get(), local_n, coeffs.data());
            for (auto& c : coeffs) c = -c;
            be.axpy_many(coeffs.data(), ortho_ptrs.data(),
                         ortho_ptrs.size(), w.get(), local_n);

            // ----- CGS2 pass 2 (reprojection) -----
            be.dot_many(ortho_ptrs.data(), ortho_ptrs.size(),
                        w.get(), local_n, coeffs.data());
            for (auto& c : coeffs) c = -c;
            be.axpy_many(coeffs.data(), ortho_ptrs.data(),
                         ortho_ptrs.size(), w.get(), local_n);
        } else if (opts.reorth == ReorthPolicy::LocalDGKS3) {
            // Wave 2.3 of the SOTA Performance rollout (May 2026):
            // for the common cases K=1 and K=2 (the new defaults
            // post-Wave 2.1) the kernel already holds the basis
            // vectors we need to project against in ``v_curr`` (= V_j)
            // and ``v_prev`` (= V_{j-1}). Use them directly and skip
            // the ring buffer entirely. For K>=3 we still need V_{j-2}
            // and beyond, so fall back to the ring storage path.
            //
            // This matches the `lanczos_real` zero-copy reorth in
            // `src/solvers/cpu/lanczos.cpp:1146-1154` and eliminates
            // the per-iter O(n) ``be.copy(v_curr -> ring[slot])``
            // dominant for large-N small-state-space workloads.
            const std::size_t K = opts.local_ring_size;
            if (K <= 2) {
                // K>=1: project against V_j (= v_curr).
                if (j > 0 || j_start > 0) {
                    const Complex overlap =
                        be.dot(v_curr.get(), w.get(), local_n);
                    if (std::abs(overlap) > opts.local_ortho_threshold) {
                        be.axpy(-overlap, v_curr.get(), w.get(), local_n);
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

        // beta[j+1] = ||w||  (already-reduced for distributed backends)
        const double bnext = be.nrm2(w.get(), local_n);
        R.beta.push_back(bnext);
        const double t4 = profile_on ? now_us() : 0.0;
        if (profile_on) t_norm_us += (t4 - t3);

        // Lanczos breakdown: an invariant subspace has been found.
        // We use `breakdown_tol` (default ~1e-300, i.e. essentially
        // "exact zero") rather than the user-facing Ritz `tol`,
        // because at full Krylov dimension the residual collapses to
        // O(eps * ||H||) but the Krylov subspace is still valid (and
        // downstream consumers like LTLM rely on the full tridiag).
        // See the LanczosKernelOptions::breakdown_tol comment.
        if (bnext < opts.breakdown_tol) break;

        // Compute V_{j+1} = w / beta_{j+1} and rotate, BEFORE the
        // on_step / convergence_check hooks. After the rotation:
        //   v_prev = V_j, v_curr = V_{j+1}, w = scratch.
        // This way the on_step hook sees the same state the legacy
        // `lanczos.cpp` body wrote to checkpoint:
        //   cp.iteration  = j + 1
        //   cp.v_current  = V_{j+1}
        //   cp.v_prev     = V_j
        // (matching the legacy convention so checkpoints round-trip
        //  bit-for-bit semantically across the migration).
        be.scale(Complex(1.0 / bnext, 0.0), w.get(), local_n);
        std::swap(v_prev, v_curr);
        std::swap(v_curr, w);

        // Update the LocalDGKS3 ring buffer with V_{j+1} so the
        // on_step hook can snapshot it for resume. Ring is internal
        // state and unaffected by basis-keep semantics.
        //
        // Wave 2.3: skip the ring update for K<=2 -- the reorth path
        // reads v_curr/v_prev directly in that regime. The only
        // remaining reason to maintain the ring is the on_step hook
        // emitting ``ring_view`` for checkpointing; we still populate
        // it when an on_step hook is registered.
        const bool ring_needed_for_reorth =
            (opts.reorth == ReorthPolicy::LocalDGKS3 &&
             opts.local_ring_size > 2);
        const bool ring_needed_for_checkpoint =
            (opts.reorth == ReorthPolicy::LocalDGKS3 &&
             opts.on_step != nullptr);
        if (ring_needed_for_reorth || ring_needed_for_checkpoint) {
            const std::size_t cap_ring =
                std::max<std::size_t>(opts.local_ring_size, 1);
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

        // Per-iteration checkpoint hook. Fires AFTER rotation so the
        // hook sees v_curr = V_{j+1}, v_prev = V_j (the same vectors
        // the legacy `lanczos.cpp` checkpoint stored). The
        // `iteration_count` passed to the hook is `j + 1`, matching
        // the legacy `cp.iteration` semantic.
        if (opts.on_step &&
            opts.on_step_interval > 0 &&
            ((j + 1) % opts.on_step_interval == 0))
        {
            // Materialise the chronological ring-buffer view as raw
            // pointers (oldest first). Only populated when the
            // LocalDGKS3 policy is active.
            std::vector<const Complex*> ring_view;
            const std::vector<const Complex*>* ring_view_ptr = nullptr;
            if (opts.reorth == ReorthPolicy::LocalDGKS3 && ring_count > 0) {
                const std::size_t cap_ring = ring.size();
                ring_view.reserve(ring_count);
                for (std::size_t k = 0; k < ring_count; ++k) {
                    const std::size_t slot = (ring_head + k) % cap_ring;
                    ring_view.push_back(ring[slot].get());
                }
                ring_view_ptr = &ring_view;
            }
            opts.on_step(j + 1, R.alpha, R.beta,
                         v_curr.get(), v_prev.get(), local_n, ring_view_ptr);
        }

        const double t5 = profile_on ? now_us() : 0.0;
        if (profile_on) t_ring_us += (t5 - t4);

        // Optional Ritz-convergence early-exit. We run the callback
        // AFTER on_step (so a checkpoint always gets written for the
        // iteration that triggered the break) but BEFORE the next
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
        std::fprintf(stderr,
            "[lanczos_kernel] iters=%zu total=%.2f ms = "
            "apply %.1f%% (%.1f us/it) "
            "recur %.1f%% (%.1f us/it) "
            "reorth %.1f%% (%.1f us/it) "
            "norm %.1f%% (%.1f us/it) "
            "ring %.1f%% (%.1f us/it) "
            "check %.1f%% (%.1f us/it) "
            "other %.1f%%\n",
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

    // Emit final v_curr / v_prev / ring buffer so callers can checkpoint
    // a resume-ready snapshot. We move out the unique_ptrs we own.
    R.v_curr = std::move(v_curr);
    R.v_prev = std::move(v_prev);
    if (opts.reorth == ReorthPolicy::LocalDGKS3) {
        // Linearise the ring buffer in chronological order (oldest first)
        // so consumers don't have to know about the (head, count) layout.
        R.ring_vectors.reserve(ring_count);
        const std::size_t cap_ring = ring.size();
        if (cap_ring > 0) {
            for (std::size_t k = 0; k < ring_count; ++k) {
                const std::size_t slot = (ring_head + k) % cap_ring;
                R.ring_vectors.emplace_back(std::move(ring[slot]));
            }
        }
    }
    return R;
}

} // namespace ed::krylov
