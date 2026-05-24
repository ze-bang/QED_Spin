#pragma once
// =============================================================================
// include/ed/matvec/backends/mpi_backend.h
//
// MpiBackend: host-memory + OpenMP for local BLAS-1, plus MPI_Allreduce
// for the cross-rank reductions inside `dot`, `nrm2`, `all_reduce_sum`,
// and (most importantly for Krylov) the batched `dot_many` used by CGS2.
//
// Phase A of the Krylov-kernel unification roll-out:
//   * one Lanczos kernel (`ed/krylov/lanczos_kernel.h`) drives all four
//     deployment targets (CPU / CUDA / MPI / MPI+CUDA),
//   * the kernel sees only the abstract `Backend` interface,
//   * `MpiBackend` is the CPU+MPI realisation. The single biggest win
//     here is `dot_many`: instead of M MPI_Allreduce round-trips per
//     Lanczos step (one per stored basis vector), we issue exactly one
//     Allreduce over a 2*M-double buffer. At M=100 that's a ~100x
//     reduction in collective latency.
//
// Pointer convention follows the Backend contract: all `Complex*` are
// rank-local slabs of dimension `n`; the matvec callback handed to
// the Lanczos kernel is responsible for any halo exchange.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <ed/matvec/backend.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <ed/matvec/memory_space.h>

namespace ed::matvec {

/// CPU + MPI backend. Inherits CpuBackend's host alloc / BLAS-1 locals
/// and overrides the reduction-bearing primitives to MPI-Allreduce.
class MpiBackend final : public CpuBackend {
public:
    /// Phase 6.2 of the Krylov-unification gap-fill (May 2026 day 12+):
    /// duplicate the caller-supplied communicator so the backend owns an
    /// independent communicator handle for the duration of its lifetime.
    /// Rationale (audit item D6): if a downstream app subsequently calls
    /// `MPI_Comm_split` / `MPI_Comm_free` on the original communicator,
    /// our backend's stored handle would dangle into freed MPI state.
    /// `MPI_Comm_dup` is collective on `parent`; constructing an
    /// `MpiBackend` on `MPI_COMM_WORLD` is therefore collective on every
    /// rank of `MPI_COMM_WORLD` (the convention every existing solver
    /// already follows by virtue of constructing one backend per
    /// collective Lanczos / FTLM entry point).
    explicit MpiBackend(MPI_Comm parent = MPI_COMM_WORLD) {
        if (parent == MPI_COMM_NULL) {
            throw std::invalid_argument(
                "MpiBackend: parent communicator must not be MPI_COMM_NULL");
        }
        const int rc = MPI_Comm_dup(parent, &comm_);
        if (rc != MPI_SUCCESS) {
            throw std::runtime_error(
                "MpiBackend: MPI_Comm_dup failed (return code " +
                std::to_string(rc) + ")");
        }
    }

    ~MpiBackend() override {
        if (comm_ != MPI_COMM_NULL) {
            // Best-effort cleanup: a destructor must not throw, and
            // MPI_Comm_free at process-shutdown time can racily return
            // MPI_ERR_OTHER if MPI_Finalize has already been called.
            // Swallow any non-success here; the only practical
            // observable on success is the released communicator slot.
            int flag = 0;
            MPI_Finalized(&flag);
            if (!flag) {
                MPI_Comm_free(&comm_);
            }
        }
    }

    MpiBackend(const MpiBackend&)            = delete;
    MpiBackend& operator=(const MpiBackend&) = delete;
    MpiBackend(MpiBackend&&)                 = delete;
    MpiBackend& operator=(MpiBackend&&)      = delete;

    [[nodiscard]] MemorySpace memory_space() const override {
        return MemorySpace::DistributedHost;
    }
    [[nodiscard]] std::string description() const override {
        int sz = 1;
        MPI_Comm_size(comm_, &sz);
        return "MpiBackend(OpenMP+MPI, np=" + std::to_string(sz) + ")";
    }

    [[nodiscard]] MPI_Comm comm() const noexcept { return comm_; }

    // ------------------------------------------------------------------
    // Scalar reductions.
    // ------------------------------------------------------------------
    [[nodiscard]] Complex all_reduce_sum(Complex v) const override {
        double pair[2] = {v.real(), v.imag()};
        MPI_Allreduce(MPI_IN_PLACE, pair, 2, MPI_DOUBLE, MPI_SUM, comm_);
        return Complex(pair[0], pair[1]);
    }
    [[nodiscard]] double all_reduce_sum(double v) const override {
        MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, comm_);
        return v;
    }
    void all_reduce_sum_vec(Complex* buf, std::size_t count) const override {
        if (count == 0) return;
        MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<double*>(buf),
                      static_cast<int>(2 * count),
                      MPI_DOUBLE, MPI_SUM, comm_);
    }
    void all_reduce_sum_vec(double* buf, std::size_t count) const override {
        if (count == 0) return;
        MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(count),
                      MPI_DOUBLE, MPI_SUM, comm_);
    }

    // ------------------------------------------------------------------
    // BLAS-1 overrides: compute the local term via CpuBackend::*, then
    // Allreduce. We deliberately do not call the single-pair `dot` from
    // `dot_many` so we can batch the Allreduce.
    // ------------------------------------------------------------------
    [[nodiscard]] Complex dot(const Complex* x, const Complex* y,
                              std::size_t n) const override {
        const Complex local = CpuBackend::dot(x, y, n);
        return all_reduce_sum(local);
    }

    [[nodiscard]] double nrm2(const Complex* x, std::size_t n) const override {
        // Compute the local ||x||^2 directly (re-using CpuBackend::nrm2
        // and squaring would lose precision and force an extra sqrt+sq).
        double sq = 0.0;
        if (n > 0) {
#ifdef _OPENMP
            #pragma omp parallel for reduction(+:sq) schedule(static) if(n > 8192)
#endif
            for (long long i = 0; i < static_cast<long long>(n); ++i) {
                const Complex v = x[i];
                sq += v.real() * v.real() + v.imag() * v.imag();
            }
        }
        sq = all_reduce_sum(sq);
        return std::sqrt(sq);
    }

    // ------------------------------------------------------------------
    // Batched dot_many: local partial products via CpuBackend::dot_many,
    // followed by ONE Allreduce over the entire k-element coefficient
    // buffer. This is the headline win of this backend: at M=100 it
    // replaces ~100 sequential Allreduces per Lanczos step with one.
    // ------------------------------------------------------------------
    void dot_many(const Complex* const* basis,
                  std::size_t           num_basis,
                  const Complex*        v,
                  std::size_t           n,
                  Complex*              coeffs_out) const override {
        if (num_basis == 0) return;
        // Local partial products into coeffs_out (CpuBackend version
        // does NOT reduce -- it's just the host-side multi-dot).
        CpuBackend::dot_many(basis, num_basis, v, n, coeffs_out);
        // One batched Allreduce over 2*num_basis doubles.
        MPI_Allreduce(MPI_IN_PLACE,
                      reinterpret_cast<double*>(coeffs_out),
                      static_cast<int>(2 * num_basis),
                      MPI_DOUBLE, MPI_SUM, comm_);
    }

    // axpy_many is purely local; the inherited CpuBackend implementation
    // already does the right thing (no override needed).

    // ------------------------------------------------------------------
    // Distributed tall-skinny QR via CholeskyQR2 (Phase 1 of the
    // Minimalist ED Collapse).
    //
    // CholeskyQR2: stable thin-QR for tall blocks where m_local >> b.
    // Each pass computes G = B^H B (b x b), MPI_Allreduces it, takes
    // a host upper Cholesky G = R^H R, then forms Q = B R^{-1} via
    // trsm. One pass is the cheap Cholesky-QR; two passes recover
    // full orthogonality to roundoff for any block whose initial
    // singular spectrum is bounded above ~eps^{-1/2}. ED block sizes
    // are tiny (b <= 8 in practice), so the host Cholesky is free
    // and the dominant cost is the local gemm + the b x b Allreduce.
    //
    // The local gemm/trsm use the inherited CpuBackend BLAS-3
    // overrides; only the b x b Gram reduction is added here.
    // ------------------------------------------------------------------
    void qr_thin(Complex* A, std::size_t m_local, std::size_t b,
                 Complex* R_host) const override {
        if (b == 0) return;
        std::vector<Complex> G(b * b);
        std::vector<Complex> R1(b * b);

        // ---- Pass 1 ---------------------------------------------------
        cpu_gemm_chc_(A, m_local, b, G.data());
        all_reduce_sum_vec(G.data(), b * b);
        host_cholesky_upper_(G.data(), b, R1.data());
        // A <- A * R1^{-1}
        if (m_local > 0) {
            CpuBackend::trsm('R', 'U', 'N', 'N', m_local, b,
                Complex{1.0, 0.0}, R1.data(), b, A, m_local);
        }

        // ---- Pass 2 (reorth) -----------------------------------------
        std::vector<Complex> R2(b * b);
        cpu_gemm_chc_(A, m_local, b, G.data());
        all_reduce_sum_vec(G.data(), b * b);
        host_cholesky_upper_(G.data(), b, R2.data());
        if (m_local > 0) {
            CpuBackend::trsm('R', 'U', 'N', 'N', m_local, b,
                Complex{1.0, 0.0}, R2.data(), b, A, m_local);
        }

        // R = R2 * R1 (b x b, host)
        std::fill_n(R_host, b * b, Complex{0.0, 0.0});
        CpuBackend::gemm('N', 'N', b, b, b,
            Complex{1.0, 0.0},
            R2.data(), b, R1.data(), b,
            Complex{0.0, 0.0}, R_host, b);
    }

private:
    /// Local-only Gram block via CpuBackend::gemm: G = A^H A (b x b).
    /// On a rank with m_local == 0, fills G with zeros (so the
    /// subsequent allreduce sees the right shape).
    void cpu_gemm_chc_(const Complex* A, std::size_t m_local,
                       std::size_t b, Complex* G_local) const {
        std::fill_n(G_local, b * b, Complex{0.0, 0.0});
        if (m_local == 0) return;
        CpuBackend::gemm('C', 'N', b, b, m_local,
            Complex{1.0, 0.0}, A, m_local, A, m_local,
            Complex{0.0, 0.0}, G_local, b);
    }

    /// Host upper-triangular Cholesky G = R^H R; writes R (b x b)
    /// column-major with strict-lower set to zero. Uses LAPACKE_zpotrf.
    /// Defensive: if `zpotrf` fails (G not SPD due to a near-linearly-
    /// dependent block), we throw -- the caller is expected to choose
    /// a non-degenerate initial block.
    void host_cholesky_upper_(Complex* G, std::size_t b, Complex* R) const {
        int info = LAPACKE_zpotrf(LAPACK_COL_MAJOR, 'U',
                                  static_cast<int>(b),
                                  reinterpret_cast<lapack_complex_double*>(G),
                                  static_cast<int>(b));
        if (info != 0) {
            throw std::runtime_error(
                "MpiBackend::qr_thin: CholeskyQR factorization failed (info=" +
                std::to_string(info) +
                "). Initial block may be near-rank-deficient.");
        }
        for (std::size_t j = 0; j < b; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                R[i + j * b] = G[i + j * b];
            }
            for (std::size_t i = j + 1; i < b; ++i) {
                R[i + j * b] = Complex{0.0, 0.0};
            }
        }
    }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
};

} // namespace ed::matvec
