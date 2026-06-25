#pragma once
// =============================================================================
// include/ed/krylov/block_lanczos_kernel.h
//
// Block-Lanczos kernel --- `template<Backend, MatvecFn>`. Builds the
// block-tridiagonal Lanczos basis using the Backend's BLAS-3 surface
// (`gemm`, `qr_thin`) introduced in Phase 1 of the Minimalist ED
// Collapse (May 2026). Two specialisations land in tree:
//
//     CpuBackend   --- LAPACKE + cBLAS path, dense gemm + Householder
//                       QR via LAPACKE_zgeqrf / _zungqr.
//     CudaBackend  --- cuBLAS + cuSolver path, device gemm + cuSolver
//                       QR. Random initial block and final eigensolve
//                       still happen on host (the projected matrix is
//                       small: m_blocks * b square).
//
// MpiBackend / MpiCudaBackend are NOT yet supported here. The block
// Gram update needs an all-reduce across ranks for the (b x b) block;
// a TSQR-based replacement for the local `qr_thin` is similarly
// pending. The static_assert below is the documented fallback. The
// distributed block-Lanczos lives in `src/distributed/distributed_block_lanczos*`
// (a separate path that doesn't go through this kernel). When the
// distributed `qr_thin` (CholeskyQR2-based, already in Backend::qr_thin
// for MpiBackend) is wired through, the static_assert can be relaxed
// to also accept MpiBackend / MpiCudaBackend.
//
// Algorithm (textbook block tridiagonal Lanczos, all on Backend):
//   1. Random N x b block on host -> backend -> `qr_thin` -> V0.
//   2. for j = 0..m-1:
//        W = H * V_j                                        (per-column matvec)
//        A_j = V_j^H * W                                    (gemm)
//        Hermitianize A_j (host trip --- A_j is b x b).
//        W -= V_j * A_j                                     (gemm)
//        if j > 0: W -= V_{j-1} * B_{j-1}^H                 (gemm)
//        periodic full CGS2 against all stored basis blocks (gemm + gemm)
//        QR(W) -> V_{j+1}, B_j                              (qr_thin)
//        push V_j into the basis storage (Backend memory)
//   3. Build the (m*b) x (m*b) block-tridiag T on host, run LAPACKE zheevd.
//   4. Reconstruct eigenvectors as V * Y on Backend (gemm), copy to host.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>

#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
#endif

namespace ed::krylov {

using Complex = std::complex<double>;

struct BlockLanczosOptions {
    std::size_t num_eigs        = 1;
    std::size_t max_iter        = 100;
    std::size_t block_size      = 4;
    double      tolerance       = 1e-10;
    bool        compute_vectors = false;
    std::string output_dir;
    /// Optional global dimension (for sanity; not used internally).
    std::uint64_t global_n      = 0;
    /// Periodic full reorthogonalisation interval (every K iterations).
    std::size_t reorth_period   = 3;
    /// Reorthogonalisation strategy / memory profile (mirrors single-vector
    /// Lanczos's keep_basis). When true (default) the kernel stores EVERY block
    /// (O(max_iter * block_size * N)) for periodic FULL reorth + one-pass
    /// eigenvector reconstruction. When false ("lean"), only a rolling window of
    /// blocks is kept (O(block_size * N)) and reorth is LOCAL-only: eigenvalues
    /// come from the tiny block-tridiagonal so they stay bounded, but accuracy
    /// degrades past ~30 blocks (ghosts) and EIGENVECTORS are unavailable
    /// (set keep_basis=true, or use Block Krylov-Schur for bounded-memory
    /// eigenvectors). The planner flips this off for large-N eigenvalue runs
    /// whose full basis would not fit the memory budget.
    bool        keep_basis      = true;
};

struct BlockLanczosResult {
    std::vector<double>                              eigenvalues;
    /// Eigenvectors in backend memory, one per requested eigenvalue.
    /// Only populated when `opts.compute_vectors == true`.
    std::vector<ed::matvec::Backend::UniqueVec>      eigenvectors;
    std::size_t                                      blocks_built = 0;
    bool                                             converged    = false;
};

namespace detail {

// ----------------------------------------------------------------------------
// Hermitianise an (b x b) column-major block in place. Cheap (b is small),
// stays on host for simplicity --- all backends only need a tiny round-trip.
// ----------------------------------------------------------------------------
inline void hermitianize_inplace(Complex* A, std::size_t b) {
    for (std::size_t c = 0; c < b; ++c) {
        A[c + c * b] = Complex(std::real(A[c + c * b]), 0.0);
        for (std::size_t r = c + 1; r < b; ++r) {
            const Complex avg = 0.5 * (A[r + c * b] + std::conj(A[c + r * b]));
            A[r + c * b] = avg;
            A[c + r * b] = std::conj(avg);
        }
    }
}

// ----------------------------------------------------------------------------
// Build the (m*b) x (m*b) block-tridiag projected matrix on host from the
// alpha / beta blocks. Used for the convergence check and the final eigensolve.
// ----------------------------------------------------------------------------
inline void build_projected_matrix(
        const std::vector<std::vector<Complex>>& alpha_blocks,
        const std::vector<std::vector<Complex>>& beta_blocks,
        std::size_t b,
        std::vector<Complex>& matrix /*out*/) {
    const std::size_t m = alpha_blocks.size();
    const std::size_t total = m * b;
    matrix.assign(total * total, Complex(0.0, 0.0));
    for (std::size_t blk = 0; blk < m; ++blk) {
        const auto& A = alpha_blocks[blk];
        const std::size_t off = blk * b;
        for (std::size_t c = 0; c < b; ++c) {
            for (std::size_t r = 0; r < b; ++r) {
                matrix[(off + r) + (off + c) * total] = A[r + c * b];
            }
        }
    }
    for (std::size_t blk = 0; blk + 1 < m; ++blk) {
        const auto& B = beta_blocks[blk];
        const std::size_t off = blk * b;
        for (std::size_t c = 0; c < b; ++c) {
            for (std::size_t r = 0; r < b; ++r) {
                matrix[(off + b + r) + (off + c) * total]   = B[r + c * b];
                matrix[(off + r) + (off + b + c) * total]   = std::conj(B[c + r * b]);
            }
        }
    }
}

}  // namespace detail

// ----------------------------------------------------------------------------
// The kernel itself.
//   `backend`     : non-const Backend& (allocation / device handles).
//   `apply_H`     : callable (const Complex* x, Complex* y, size_t n).
//                   Applies H to a single rank-local vector. Block matvec
//                   is realised by looping over the `b` columns.
//   `local_n`     : rank-local dimension (= N for non-distributed).
//   `seed_block`  : N*b host-side initial block (row-major-by-column,
//                   i.e. column-major in BLAS terms). Caller provides
//                   to keep the kernel deterministic across processes;
//                   pass nullptr to let the kernel generate a random
//                   block from `opts.output_dir` hashing (TODO: not yet
//                   used --- random_device for now).
// ----------------------------------------------------------------------------
template <typename Backend, typename MatvecFn>
BlockLanczosResult block_lanczos_kernel(Backend&                  backend,
                                        MatvecFn&&                apply_H,
                                        std::size_t               local_n,
                                        std::uint64_t             /*global_n*/,
                                        const BlockLanczosOptions& opts)
{
    static_assert(
        std::is_base_of_v<ed::matvec::CpuBackend, std::decay_t<Backend>>
#ifdef WITH_CUDA
            || std::is_base_of_v<ed::matvec::CudaBackend, std::decay_t<Backend>>
#endif
        ,
        "block_lanczos_kernel currently supports CpuBackend / CudaBackend "
        "(and MpiBackend via CpuBackend inheritance, in the single-rank "
        "build). MpiBackend / MpiCudaBackend with a true distributed "
        "TSQR pathway is a documented Phase 2.3 deferral --- the "
        "distributed block-Lanczos still uses `src/distributed/"
        "distributed_block_lanczos*` directly.");

    const std::size_t N = local_n;
    if (N == 0) {
        throw std::invalid_argument("block_lanczos_kernel: local_n == 0");
    }

    const std::size_t b           = std::max<std::size_t>(1, std::min<std::size_t>(opts.block_size, N));
    const std::size_t target_eigs = std::max<std::size_t>(1, std::min<std::size_t>(opts.num_eigs, N));
    const std::size_t max_blocks  = (opts.max_iter == 0)
        ? (N + b - 1) / b
        : std::min<std::size_t>(opts.max_iter, (N + b - 1) / b);
    const double conv_tol      = (opts.tolerance <= 0.0) ? 1e-12 : opts.tolerance;
    constexpr double breakdown = 1e-12;

    // -------------------------------------------------------------------
    // Random initial block on host (uniform [-1,1]^2, complex), then a
    // round-trip into Backend memory + qr_thin for the orthonormal V_0.
    // -------------------------------------------------------------------
    std::vector<Complex> host_block(N * b);
    {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& z : host_block) z = Complex(dist(gen), dist(gen));
    }
    auto V_curr = backend.make_zero_vector(N * b);
    backend.copy_from_host(host_block.data(), V_curr.get(), N * b);
    {
        std::vector<Complex> R0(b * b);
        backend.qr_thin(V_curr.get(), N, b, R0.data());
    }
    auto V_prev = backend.make_zero_vector(N * b);
    auto V_next = backend.make_zero_vector(N * b);
    auto W      = backend.make_zero_vector(N * b);
    auto AB     = backend.make_zero_vector(b * b);  // shared bbxb scratch

    // Stored basis blocks (Backend memory). One element per Lanczos block.
    std::vector<ed::matvec::Backend::UniqueVec> basis;
    basis.reserve(max_blocks);

    std::vector<std::vector<Complex>> alpha_blocks;
    std::vector<std::vector<Complex>> beta_blocks;
    alpha_blocks.reserve(max_blocks);
    beta_blocks.reserve(max_blocks);

    std::vector<Complex> Aj_host(b * b);
    std::vector<Complex> Bj_host(b * b);
    std::vector<Complex> Bprev_host(b * b, Complex(0.0, 0.0));

    const Complex one(1.0, 0.0), zero(0.0, 0.0), neg_one(-1.0, 0.0);
    bool converged = false;

    // Lean mode: no stored basis -> local reorth only, eigenvalues only.
    const bool keep_basis = opts.keep_basis;
    if (!keep_basis && opts.compute_vectors) {
        throw std::invalid_argument(
            "block_lanczos_kernel: keep_basis=false (lean reorth) cannot return "
            "eigenvectors -- the N-dimensional basis is not stored. Set "
            "keep_basis=true, or use block_krylov_schur_kernel for "
            "bounded-memory eigenvectors.");
    }

    for (std::size_t j = 0; j < max_blocks; ++j) {
        // Snapshot V_curr into the basis storage (only when we keep it: full
        // reorth + eigenvector reconstruction need it; lean mode does not).
        if (keep_basis) {
            auto V_j_keep = backend.make_zero_vector(N * b);
            backend.copy(V_curr.get(), V_j_keep.get(), N * b);
            basis.emplace_back(std::move(V_j_keep));
        }

        // W = H * V_curr  (column-by-column).
        for (std::size_t c = 0; c < b; ++c) {
            apply_H(V_curr.get() + c * N, W.get() + c * N, N);
        }

        // A_j = V_curr^H * W   (local; for distributed backends this is
        // the rank-local Gram block and `all_reduce_sum_vec` consolidates
        // it. The static_assert above currently keeps us out of that path.)
        backend.gemm('C', 'N', b, b, N,
                     one, V_curr.get(), N, W.get(), N,
                     zero, AB.get(), b);
        backend.copy_to_host(AB.get(), Aj_host.data(), b * b);
        backend.all_reduce_sum_vec(Aj_host.data(), b * b);
        detail::hermitianize_inplace(Aj_host.data(), b);
        backend.copy_from_host(Aj_host.data(), AB.get(), b * b);

        // W -= V_curr * A_j.
        backend.gemm('N', 'N', N, b, b,
                     neg_one, V_curr.get(), N, AB.get(), b,
                     one, W.get(), N);

        if (j > 0) {
            // W -= V_prev * B_prev^H.
            backend.copy_from_host(Bprev_host.data(), AB.get(), b * b);
            backend.gemm('N', 'C', N, b, b,
                         neg_one, V_prev.get(), N, AB.get(), b,
                         one, W.get(), N);
        }

        // Periodic full CGS2 against all stored basis blocks
        // (gem) every `reorth_period` iterations. Cheap on GPU because the
        // gemms stay on device.
        const bool do_full_reorth =
            keep_basis && (opts.reorth_period > 0) &&
            (j > 0) && (j % opts.reorth_period == 0);
        if (do_full_reorth) {
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t blk = 0; blk <= j; ++blk) {
                    const Complex* Vptr = basis[blk].get();
                    backend.gemm('C', 'N', b, b, N,
                                 one, Vptr, N, W.get(), N,
                                 zero, AB.get(), b);
                    {
                        std::vector<Complex> tmp(b * b);
                        backend.copy_to_host(AB.get(), tmp.data(), b * b);
                        backend.all_reduce_sum_vec(tmp.data(), b * b);
                        backend.copy_from_host(tmp.data(), AB.get(), b * b);
                    }
                    backend.gemm('N', 'N', N, b, b,
                                 neg_one, Vptr, N, AB.get(), b,
                                 one, W.get(), N);
                }
            }
        } else {
            // Local CGS against V_curr (single pass).
            backend.gemm('C', 'N', b, b, N,
                         one, V_curr.get(), N, W.get(), N,
                         zero, AB.get(), b);
            {
                std::vector<Complex> tmp(b * b);
                backend.copy_to_host(AB.get(), tmp.data(), b * b);
                backend.all_reduce_sum_vec(tmp.data(), b * b);
                backend.copy_from_host(tmp.data(), AB.get(), b * b);
            }
            backend.gemm('N', 'N', N, b, b,
                         neg_one, V_curr.get(), N, AB.get(), b,
                         one, W.get(), N);
            if (j > 0) {
                backend.gemm('C', 'N', b, b, N,
                             one, V_prev.get(), N, W.get(), N,
                             zero, AB.get(), b);
                {
                    std::vector<Complex> tmp(b * b);
                    backend.copy_to_host(AB.get(), tmp.data(), b * b);
                    backend.all_reduce_sum_vec(tmp.data(), b * b);
                    backend.copy_from_host(tmp.data(), AB.get(), b * b);
                }
                backend.gemm('N', 'N', N, b, b,
                             neg_one, V_prev.get(), N, AB.get(), b,
                             one, W.get(), N);
            }
        }

        alpha_blocks.push_back(Aj_host);
        if (j > 0) beta_blocks.push_back(Bprev_host);

        // QR(W) -> V_next, B_j. `qr_thin` overwrites W with Q.
        backend.qr_thin(W.get(), N, b, Bj_host.data());

        // Breakdown check on min |diag(B_j)|.
        double min_diag = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < b; ++i) {
            min_diag = std::min(min_diag, std::abs(Bj_host[i + i * b]));
        }

        // Convergence check (uses block-tridiag eigensolve on host).
        const std::size_t total_blocks = alpha_blocks.size();
        const std::size_t total_dim    = total_blocks * b;
        if (total_dim >= target_eigs) {
            std::vector<Complex> T_mat;
            detail::build_projected_matrix(alpha_blocks, beta_blocks, b, T_mat);
            std::vector<double> evals(total_dim);
            const auto info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U',
                static_cast<lapack_int>(total_dim),
                reinterpret_cast<lapack_complex_double*>(T_mat.data()),
                static_cast<lapack_int>(total_dim),
                evals.data());
            if (info == 0) {
                const bool have_next_block = (min_diag > breakdown)
                                          && (j + 1 < max_blocks);
                std::size_t ok = 0;
                std::vector<Complex> res_block(b);
                for (std::size_t kk = 0; kk < target_eigs; ++kk) {
                    if (!have_next_block) { ok = target_eigs; break; }
                    // residual ≈ ||B_j * y_last||
                    cblas_zgemv(CblasColMajor, CblasNoTrans,
                                b, b, &one,
                                Bj_host.data(), b,
                                &T_mat[(total_blocks - 1) * b + kk * total_dim], 1,
                                &zero, res_block.data(), 1);
                    if (cblas_dznrm2(b, res_block.data(), 1) <= conv_tol) ++ok;
                }
                if (ok >= target_eigs) {
                    converged = true;
                    break;
                }
            }
        }

        if (min_diag < breakdown) {
            break;
        }

        // Rotate V_prev <- V_curr, V_curr <- W (= Q), B_prev <- B_j.
        std::swap(V_prev, V_curr);
        backend.copy(W.get(), V_curr.get(), N * b);
        Bprev_host = Bj_host;
    }

    // ----- final eigensolve on host -----------------------------------
    BlockLanczosResult out;
    if (alpha_blocks.empty()) return out;

    const std::size_t m_built = alpha_blocks.size();
    const std::size_t total   = m_built * b;
    std::vector<Complex> T_mat;
    detail::build_projected_matrix(alpha_blocks, beta_blocks, b, T_mat);

    std::vector<double> evals(total);
    const char jobz = opts.compute_vectors ? 'V' : 'N';
    const auto info = LAPACKE_zheevd(LAPACK_COL_MAJOR, jobz, 'U',
        static_cast<lapack_int>(total),
        reinterpret_cast<lapack_complex_double*>(T_mat.data()),
        static_cast<lapack_int>(total),
        evals.data());
    if (info != 0) {
        return out;
    }

    const std::size_t output_eigs = std::min<std::size_t>(target_eigs, total);
    out.eigenvalues.assign(evals.begin(), evals.begin() + output_eigs);
    out.blocks_built = m_built;
    out.converged    = converged;

    if (opts.compute_vectors) {
        // Each eigenvector v_k = sum_{blk} V_{blk} * Y_{blk,k}, where
        // Y is a (b x output_eigs) slab of the m*b x m*b eigenvector
        // matrix. We hold Y_blk in Backend scratch (b * output_eigs)
        // and accumulate into a single (N x output_eigs) backend buffer.
        auto evec_storage = backend.make_zero_vector(N * output_eigs);
        auto Yblk         = backend.make_zero_vector(b * output_eigs);
        std::vector<Complex> Yblk_host(b * output_eigs);

        for (std::size_t blk = 0; blk < m_built; ++blk) {
            for (std::size_t c = 0; c < output_eigs; ++c) {
                for (std::size_t r = 0; r < b; ++r) {
                    Yblk_host[r + c * b] =
                        T_mat[(blk * b + r) + c * total];
                }
            }
            backend.copy_from_host(Yblk_host.data(), Yblk.get(),
                                   b * output_eigs);
            // evec_storage += V_blk * Y_blk    (N x output_eigs)
            backend.gemm('N', 'N', N, output_eigs, b,
                         one, basis[blk].get(), N,
                         Yblk.get(), b,
                         one, evec_storage.get(), N);
        }

        // Normalise & hand out one UniqueVec per eigenvector.
        out.eigenvectors.reserve(output_eigs);
        for (std::size_t k = 0; k < output_eigs; ++k) {
            auto v = backend.make_zero_vector(N);
            backend.copy(evec_storage.get() + k * N, v.get(), N);
            const double nrm = backend.nrm2(v.get(), N);
            if (nrm > 0.0) backend.scale(Complex(1.0 / nrm, 0.0), v.get(), N);
            out.eigenvectors.emplace_back(std::move(v));
        }
    }

    return out;
}

}  // namespace ed::krylov
