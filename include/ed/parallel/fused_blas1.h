#pragma once

// =============================================================================
// include/ed/parallel/fused_blas1.h                              (Phase 6 #5)
//
// Fused real BLAS-1 kernels for the ``lanczos_real`` inner loop.
//
// MOTIVATION
// ----------
// At N = 18-22 (Krylov dim < 1M) the Lanczos iteration is dominated by
// per-call OpenMP / BLAS overhead, not by FLOPs: each cblas_* call enters
// and exits a fresh parallel region (5-50 us fork / join with 8 threads)
// around a bandwidth-bound body of < 100 us at N ~ 200K. The recurrence
// makes several BLAS-1 sweeps per iteration over the same vectors:
//
//   1)  w  -= beta_j   * v_prev
//   2)  alpha_j = <v_current, w>
//   3)  w  -= alpha_j  * v_current
//   4)  beta_{j+1} = ||w||,  v_{j+1} = w / beta_{j+1}
//
// Fusing them keeps all BLAS-1 work inside our own OpenMP team (avoiding
// OpenBLAS's pthread pool, which fights our threads for the same cores)
// and cuts the fork / join count per iteration.
//
// KERNELS
// -------
// All kernels take raw pointers and a length and use a single OpenMP
// parallel region with a static schedule. The reductions match the
// LAPACK/OpenBLAS dot/norm behaviour to within the usual floating-point
// non-associativity tolerance (irrelevant at the 1e-10 Lanczos
// convergence threshold). They are specialised to the Lanczos pattern,
// not drop-in replacements for cblas_*.
// =============================================================================

#include <cstddef>
#include <cstdint>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::parallel {

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

// fused_norm2_scale_real
//   norm   = sqrt( sum_i w[i]^2 )
//   v_out[i] = w[i] / norm          (zero when norm == 0)
// returns norm.
//
// Step (4) of the recurrence in one OpenMP region: the new Lanczos vector
// is written straight into its destination slot (no extra copy).
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

}  // namespace ed::parallel
