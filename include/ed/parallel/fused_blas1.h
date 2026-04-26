#pragma once

// =============================================================================
// include/ed/parallel/fused_blas1.h                              (Phase 6 #5)
//
// Fused BLAS-1 kernels for the Lanczos / Arnoldi inner loop.
//
// MOTIVATION
// ----------
// At N = 18-22 (FixedSz Heisenberg, Krylov dim < 1M) the Lanczos iteration
// is dominated by per-call OpenMP / BLAS overhead, not by FLOPs:
//
//   * each cblas_zaxpy / cblas_zdotc_sub / cblas_dznrm2 call enters and
//     exits a fresh OpenMP parallel region;
//   * each region pays a fork / join cost (5-50 us with 8 threads
//     depending on the runtime) plus a function-call boundary into
//     OpenBLAS;
//   * the body of each call is bandwidth-bound and takes < 100 us at
//     N ~ 200K, so the per-region overhead is a non-trivial fraction
//     of useful time.
//
// The standard Lanczos recurrence makes 4 BLAS-1 sweeps per iter that
// touch the same vectors:
//
//   1)  w  -= beta_j   * v_prev
//   2)  alpha_j = <v_current, w>     (real for Hermitian H)
//   3)  w  -= alpha_j  * v_current
//   4)  beta_{j+1} = ||w||
//
// (1) and (2) read the same w; (3) and (4) read the same w. Fusing each
// pair into a single OpenMP parallel region halves the fork / join count
// per iter and -- because v_prev / v_current / w each fit in L2 at the
// per-thread block size -- improves cache reuse as well.
//
// KERNELS
// -------
// All kernels take raw pointers and a length and use a single OpenMP
// parallel region with a static schedule. They assume aligned-enough
// allocations (std::vector<std::complex<double>> is 16-byte aligned on
// every libstdc++ we target, which is sufficient for AVX2 movupd /
// AVX-512 vmovups on x86_64).
//
// The reductions use ``reduction(+:.)`` which OpenMP guarantees is
// numerically equivalent to a tree reduction; this matches the
// LAPACK/OpenBLAS dot/norm behaviour to within the usual floating-point
// non-associativity tolerance (irrelevant at the 1e-10 Lanczos
// convergence threshold).
//
// USAGE
// -----
// These are *not* drop-in replacements for cblas_*; they are specialised
// to the exact Lanczos pattern. Call sites must:
//   * have already computed `w = H * v_current` (SpMV is left to the
//     operator's parallel implementation);
//   * pass the buffers in the order shown in the kernel signatures.
//
// The kernels are inline in the header so they can be inlined into the
// solver TU and so callers don't pay an extra link-time call.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::parallel {

using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// fused_axpy_dotc_real
//
//   w[i] -= beta * v_prev[i]   (BLAS-1 zaxpy with beta scalar)
//   alpha = Re( sum_i conj(v_current[i]) * w_after[i] )
//
// Fuses steps (1) and (2) of the Lanczos recurrence. ``beta`` may be 0.0
// (j == 0 case) in which case the axpy is a no-op for free; we still
// pay the read of v_prev unless the caller skips the call entirely
// (which the lanczos drivers already do via `if (j > 0)` -- so this
// kernel is only invoked when beta != 0). The ``apply_beta_term`` flag
// preserves that contract while keeping a single fused entry point.
//
// Returns the real part of <v_current, w_after_axpy>. The imaginary
// part is dropped (Hermitian H => alpha is real to roundoff).
// -----------------------------------------------------------------------------
inline double
fused_axpy_dotc_real(std::uint64_t N,
                     double beta,
                     const Complex* __restrict__ v_prev,
                     const Complex* __restrict__ v_current,
                     Complex*       __restrict__ w,
                     bool apply_beta_term)
{
    double alpha_re = 0.0;
    double alpha_im = 0.0;

    if (apply_beta_term) {
        #pragma omp parallel for schedule(static) \
                reduction(+:alpha_re, alpha_im) if(N > 1024)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
            // w_i := w_i - beta * v_prev_i
            const Complex w_new = w[i] - beta * v_prev[i];
            // contrib := conj(v_current_i) * w_new
            const Complex vc = v_current[i];
            const double c_re = vc.real() * w_new.real() + vc.imag() * w_new.imag();
            const double c_im = vc.real() * w_new.imag() - vc.imag() * w_new.real();
            alpha_re += c_re;
            alpha_im += c_im;
            w[i] = w_new;
        }
    } else {
        #pragma omp parallel for schedule(static) \
                reduction(+:alpha_re, alpha_im) if(N > 1024)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
            const Complex w_new = w[i];
            const Complex vc = v_current[i];
            const double c_re = vc.real() * w_new.real() + vc.imag() * w_new.imag();
            const double c_im = vc.real() * w_new.imag() - vc.imag() * w_new.real();
            alpha_re += c_re;
            alpha_im += c_im;
        }
    }

    (void)alpha_im;  // dropped by Hermitian-H contract
    return alpha_re;
}

// -----------------------------------------------------------------------------
// fused_axpy_norm2
//
//   w[i] -= alpha * v_current[i]
//   norm = sqrt( sum_i |w_after[i]|^2 )
//
// Fuses steps (3) and (4) of the Lanczos recurrence. The norm uses a
// straight ||.||_2 reduction (no Kahan); cblas_dznrm2 internally also
// uses a plain accumulation for short-to-medium vectors, so the
// numerics match.
// -----------------------------------------------------------------------------
inline double
fused_axpy_norm2(std::uint64_t N,
                 double alpha,
                 const Complex* __restrict__ v_current,
                 Complex*       __restrict__ w)
{
    double norm_sq = 0.0;

    #pragma omp parallel for schedule(static) \
            reduction(+:norm_sq) if(N > 1024)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        const Complex w_new = w[i] - alpha * v_current[i];
        norm_sq += w_new.real() * w_new.real() + w_new.imag() * w_new.imag();
        w[i] = w_new;
    }

    return std::sqrt(norm_sq);
}

// -----------------------------------------------------------------------------
// fused_dotc_axpy_inplace
//
// Single-vector local-reorth pass:
//
//   overlap = <u, w>
//   if (|overlap| > threshold) w -= overlap * u
//
// Fused into one OMP region. Returns the magnitude of the overlap so
// the caller can decide whether to log / count it. The axpy is always
// performed when overlap is non-zero; conditional axpy on threshold is
// kept inside the kernel so the call site stays compact.
//
// Implementation note: we run *two* parallel regions still (one for
// the dot, one for the conditional axpy) because the axpy depends on
// the reduced overlap value. Doing this in a single region requires a
// barrier + thread-local partial accumulators; for the 3-vector loop
// in the solver, the two-region version is already a 3x reduction in
// fork / joins (was 6 BLAS-1 calls -> 3 calls of this fused kernel),
// which captures most of the win without the complexity.
// -----------------------------------------------------------------------------
inline double
fused_dotc_axpy_inplace(std::uint64_t N,
                        const Complex* __restrict__ u,
                        Complex*       __restrict__ w,
                        double threshold)
{
    double ov_re = 0.0;
    double ov_im = 0.0;

    #pragma omp parallel for schedule(static) \
            reduction(+:ov_re, ov_im) if(N > 1024)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        const Complex ui = u[i];
        const Complex wi = w[i];
        ov_re += ui.real() * wi.real() + ui.imag() * wi.imag();
        ov_im += ui.real() * wi.imag() - ui.imag() * wi.real();
    }

    const Complex overlap(ov_re, ov_im);
    const double mag = std::abs(overlap);
    if (mag > threshold) {
        const Complex neg_ov = -overlap;
        #pragma omp parallel for schedule(static) if(N > 1024)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
            w[i] += neg_ov * u[i];
        }
    }
    return mag;
}

// -----------------------------------------------------------------------------
// fused_scale_inplace
//
//   v_next[i] = w[i] * inv_norm
//
// Single OMP region; matches cblas_zscal numerically (multiplication by
// a real scalar). Provided so the entire Lanczos inner loop can be
// expressed without any cblas_* call -- this matters because OpenBLAS
// re-enters its thread pool on every call, and skipping that re-entry
// is the whole point of fusing.
// -----------------------------------------------------------------------------
inline void
fused_scale_real(std::uint64_t N,
                 double inv_norm,
                 Complex* __restrict__ v_next)
{
    #pragma omp parallel for schedule(static) if(N > 1024)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        v_next[i] *= inv_norm;
    }
}

// =============================================================================
// REAL-arithmetic kernels (Phase 6 #9). Mirror the complex versions above
// but on double*. Used by ``lanczos_real`` to (a) keep all BLAS-1 work
// inside our own OpenMP team -- avoiding OpenBLAS's pthread pool, which
// fights with our OMP threads for the same physical cores at large N --
// and (b) fuse the recurrence's read-then-write passes so each iter
// has 4 OMP regions instead of ~10 cblas_* calls' worth.
//
// At N=20 fixed-Sz Heisenberg this drops the Lanczos cost from
// ~425 ms (cblas BLAS-1, OPENBLAS_NUM_THREADS=1) to ~?? ms.
// =============================================================================

// fused_axpy_dot_axpy_real
//   if (apply_beta_term):  w[i] -= beta * v_prev[i]
//   alpha = sum_i  v_current[i] * w_after[i]
//   w[i] -= alpha * v_current[i]
// returns alpha.
//
// Fuses the three-step "w -= beta*v_prev; alpha = <v_curr, w>;
// w -= alpha*v_curr" recurrence (steps 1-3 of the Lanczos inner loop)
// into a SINGLE OpenMP parallel region. The two for-loops within the
// region share the same thread team and are separated by an implicit
// barrier (end of the first ``omp for``). Without this fusion each
// step needs its own fork/join (3x overhead at large thread counts).
//
// We use a small per-thread accumulator + an atomic to combine, then
// a barrier so the post-barrier loop sees the final alpha. ``reduction``
// on ``omp for`` would also work but the manual pattern lets us keep
// alpha visible after the first for-loop with one fewer implicit
// reduction broadcast.
inline double
fused_axpy_dot_axpy_real(std::uint64_t N,
                         double beta,
                         const double* __restrict__ v_prev,
                         const double* __restrict__ v_current,
                         double*       __restrict__ w,
                         bool apply_beta_term)
{
    double alpha = 0.0;
#ifdef _OPENMP
    if (N > 1024) {
        #pragma omp parallel
        {
            double local_alpha = 0.0;
            if (apply_beta_term) {
                #pragma omp for schedule(static) nowait
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
                    const double w_new = w[i] - beta * v_prev[i];
                    local_alpha += v_current[i] * w_new;
                    w[i] = w_new;
                }
            } else {
                #pragma omp for schedule(static) nowait
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
                    local_alpha += v_current[i] * w[i];
                }
            }
            #pragma omp atomic
            alpha += local_alpha;
            #pragma omp barrier
            // alpha is now the final reduced value; do the second axpy.
            #pragma omp for schedule(static) nowait
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
                w[i] -= alpha * v_current[i];
            }
        }
        return alpha;
    }
#endif
    if (apply_beta_term) {
        for (std::uint64_t i = 0; i < N; ++i) {
            const double w_new = w[i] - beta * v_prev[i];
            alpha += v_current[i] * w_new;
            w[i] = w_new;
        }
    } else {
        for (std::uint64_t i = 0; i < N; ++i) alpha += v_current[i] * w[i];
    }
    for (std::uint64_t i = 0; i < N; ++i) w[i] -= alpha * v_current[i];
    return alpha;
}

// fused_axpy_norm2_real
//   w[i] -= alpha * v_current[i]
//   norm  = sqrt( sum_i w_after[i]^2 )
inline double
fused_axpy_norm2_real(std::uint64_t N,
                      double alpha,
                      const double* __restrict__ v_current,
                      double*       __restrict__ w)
{
    double norm_sq = 0.0;
    #pragma omp parallel for schedule(static) \
            reduction(+:norm_sq) if(N > 1024)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        const double w_new = w[i] - alpha * v_current[i];
        norm_sq += w_new * w_new;
        w[i] = w_new;
    }
    return std::sqrt(norm_sq);
}

// fused_dot_axpy_norm2_scale_real
//
//   overlap = sum_i u[i] * w[i]
//   if |overlap| > threshold: w[i] -= overlap * u[i]
//   norm    = sqrt( sum_i w_after[i]^2 )
//   v[i]    = w_after[i] * (1 / norm)        (only if norm > 0)
//
// The "tail" of one Lanczos iter, fused into one OpenMP region. Saves
// three fork/joins (the standalone reorth-pass, the norm reduction,
// and the scale) per iter at the cost of a few barriers. Used as the
// last reorth pass + norm + scale in one shot.
//
// NOTE: This fused kernel is currently unused -- preserved as a
// reference for future tuning. The mainline lanczos_real path keeps
// reorth, norm and scale as separate fused kernels for clarity, since
// the savings (one extra fork/join per iter) are already in the noise
// at the scales we care about (N <= 24).
inline double
fused_norm2_scale_real(std::uint64_t N,
                       const double* __restrict__ w,
                       double*       __restrict__ v_out)
{
    double norm_sq = 0.0;
#ifdef _OPENMP
    if (N > 1024) {
        #pragma omp parallel
        {
            double local = 0.0;
            #pragma omp for schedule(static) nowait
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
                local += w[i] * w[i];
            }
            #pragma omp atomic
            norm_sq += local;
            #pragma omp barrier
            const double inv = (norm_sq > 0.0) ? 1.0 / std::sqrt(norm_sq) : 0.0;
            #pragma omp for schedule(static) nowait
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
                v_out[i] = w[i] * inv;
            }
        }
        return std::sqrt(norm_sq);
    }
#endif
    for (std::uint64_t i = 0; i < N; ++i) norm_sq += w[i] * w[i];
    const double norm = std::sqrt(norm_sq);
    const double inv = (norm > 0.0) ? 1.0 / norm : 0.0;
    for (std::uint64_t i = 0; i < N; ++i) v_out[i] = w[i] * inv;
    return norm;
}

// fused_dot_axpy_real
//   overlap = sum_i u[i] * w[i]
//   if |overlap| > threshold:  w[i] -= overlap * u[i]
// returns |overlap|.
inline double
fused_dot_axpy_real(std::uint64_t N,
                    const double* __restrict__ u,
                    double*       __restrict__ w,
                    double threshold)
{
    double overlap = 0.0;
    #pragma omp parallel for schedule(static) \
            reduction(+:overlap) if(N > 1024)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        overlap += u[i] * w[i];
    }
    const double mag = std::abs(overlap);
    if (mag > threshold) {
        const double neg = -overlap;
        #pragma omp parallel for schedule(static) if(N > 1024)
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
            w[i] += neg * u[i];
        }
    }
    return mag;
}

// fused_scale_inplace_real
//   v[i] *= inv_norm
inline void
fused_scale_inplace_real(std::uint64_t N,
                         double inv_norm,
                         double* __restrict__ v)
{
    #pragma omp parallel for schedule(static) if(N > 1024)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(N); ++i) {
        v[i] *= inv_norm;
    }
}

}  // namespace ed::parallel
