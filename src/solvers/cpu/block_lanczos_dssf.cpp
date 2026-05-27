// =============================================================================
// src/solvers/cpu/block_lanczos_dssf.cpp
//
// Audit item #6: implementation of block / band Lanczos for ground-state
// dynamical structure factor with multiple starting vectors.
//
// See ed/solvers/block_lanczos_dssf.h for the mathematical setup. The
// implementation closely follows the block-Lanczos eigensolver in
// src/solvers/cpu/lanczos.cpp (BLAS/LAPACK calls, column-major dense
// blocks, modified Gram-Schmidt periodic full reorth) but:
//   * keeps the entire basis in memory (no disk paging), since DSSF
//     krylov_dim is typically O(50-200) and N is moderate;
//   * after the recurrence completes, builds the full block tridiagonal
//     T explicitly and diagonalises with LAPACKE_zheevd to obtain Ritz
//     pairs;
//   * computes per-channel spectral weights from the top P-block of each
//     Ritz vector, then evaluates Lorentzian-broadened spectra at every
//     frequency in `params.{omega_min, omega_max, num_omega_points}`.
// =============================================================================

#include <ed/solvers/block_lanczos_dssf.h>

#include <ed/core/blas_lapack_wrapper.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ed::block_lanczos_dssf {
namespace {

constexpr Complex kOne(1.0, 0.0);
constexpr Complex kZero(0.0, 0.0);
constexpr Complex kNegOne(-1.0, 0.0);

// In-place block QR via LAPACK zgeqrf + zungqr. On entry `M` is the
// N-by-P column-major matrix; on exit `M` holds the orthonormal Q and
// `R` (size P*P column-major) holds the upper-triangular factor.
//
// Returns true on success. On rank deficiency the matrix is still
// orthonormalised (LAPACK fills the deficient columns with arbitrary
// orthonormal vectors), and the deficient diagonals of R are returned
// near-zero -- callers can detect this via `R[k + k*P]` magnitude.
bool block_qr(Complex* Mptr, std::size_t N, std::size_t P,
              Complex* Rptr, std::vector<Complex>& tau_scratch) {
    tau_scratch.assign(P, kZero);
    const lapack_int info = LAPACKE_zgeqrf(
        LAPACK_COL_MAJOR, static_cast<lapack_int>(N), static_cast<lapack_int>(P),
        reinterpret_cast<lapack_complex_double*>(Mptr), static_cast<lapack_int>(N),
        reinterpret_cast<lapack_complex_double*>(tau_scratch.data()));
    if (info != 0) return false;

    // Extract R (upper triangular) into Rptr (P x P column-major).
    std::fill(Rptr, Rptr + P * P, kZero);
    for (std::size_t col = 0; col < P; ++col) {
        for (std::size_t row = 0; row <= col; ++row) {
            Rptr[row + col * P] = Mptr[row + col * N];
        }
    }

    const lapack_int info2 = LAPACKE_zungqr(
        LAPACK_COL_MAJOR, static_cast<lapack_int>(N), static_cast<lapack_int>(P),
        static_cast<lapack_int>(P),
        reinterpret_cast<lapack_complex_double*>(Mptr), static_cast<lapack_int>(N),
        reinterpret_cast<lapack_complex_double*>(tau_scratch.data()));
    return info2 == 0;
}

// Symmetrise a P x P column-major Hermitian block in place: enforce
// A = (A + A^dag) / 2 with real diagonal. Mirrors the same trick used
// in the eigensolver block Lanczos.
void hermitise(Complex* A, std::size_t P) {
    for (std::size_t c = 0; c < P; ++c) {
        A[c + c * P] = Complex(std::real(A[c + c * P]), 0.0);
        for (std::size_t r = c + 1; r < P; ++r) {
            const Complex avg = 0.5 * (A[r + c * P] + std::conj(A[c + r * P]));
            A[r + c * P] = avg;
            A[c + r * P] = std::conj(avg);
        }
    }
}

} // namespace

MultiChannelDSSFResults compute_ground_state_block_dssf(
    const std::function<void(const Complex*, Complex*, std::size_t)>& H,
    const std::vector<std::function<void(const Complex*, Complex*, std::size_t)>>& O_list,
    const ComplexVector& ground_state,
    double ground_state_energy,
    std::size_t N,
    const BlockLanczosDSSFParameters& params) {

    if (O_list.empty()) {
        throw std::invalid_argument(
            "compute_ground_state_block_dssf: O_list is empty");
    }
    if (ground_state.size() != N) {
        throw std::invalid_argument(
            "compute_ground_state_block_dssf: ground_state size != N");
    }
    if (params.krylov_dim == 0) {
        throw std::invalid_argument(
            "compute_ground_state_block_dssf: krylov_dim must be > 0");
    }
    if (params.num_omega_points == 0) {
        throw std::invalid_argument(
            "compute_ground_state_block_dssf: num_omega_points must be > 0");
    }

    const std::size_t P = O_list.size();
    const std::size_t M = params.krylov_dim;

    if (params.verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Block Lanczos Multi-channel Ground State DSSF\n";
        std::cout << "==========================================\n";
        std::cout << "  Hilbert dim N      = " << N << "\n";
        std::cout << "  Channels P         = " << P << "\n";
        std::cout << "  Block krylov dim M = " << M << "\n";
        std::cout << "  Total Krylov MP    = " << (M * P) << "\n";
        std::cout << "  Broadening eta     = " << params.broadening << "\n";
        std::cout << "  Omega range        = [" << params.omega_min
                  << ", " << params.omega_max << "]\n";
        std::cout << "  Omega points       = " << params.num_omega_points << "\n";
        std::cout << std::flush;
    }

    MultiChannelDSSFResults results;
    results.block_size = P;
    results.frequencies.resize(params.num_omega_points);
    results.spectral.assign(P, std::vector<double>(params.num_omega_points, 0.0));
    results.channel_norm_sq.assign(P, 0.0);

    {
        const double dw = (params.num_omega_points > 1)
            ? (params.omega_max - params.omega_min) /
              static_cast<double>(params.num_omega_points - 1)
            : 0.0;
        for (std::size_t i = 0; i < params.num_omega_points; ++i) {
            results.frequencies[i] = params.omega_min + static_cast<double>(i) * dw;
        }
    }

    // ---------------------------------------------------------------------
    // Step 1: build the initial block Phi = [O_1|0>, ..., O_P|0>] as an
    // N x P column-major dense matrix.
    // ---------------------------------------------------------------------
    std::vector<Complex> V_curr(N * P, kZero);
    for (std::size_t p = 0; p < P; ++p) {
        O_list[p](ground_state.data(), V_curr.data() + p * N, N);
        const double nrm = cblas_dznrm2(static_cast<lapack_int>(N),
                                        V_curr.data() + p * N, 1);
        results.channel_norm_sq[p] = nrm * nrm;
    }

    // ---------------------------------------------------------------------
    // Step 2: QR factorise Phi in place. Phi <- V_0 (orthonormal),
    // R0 holds the upper triangular factor.
    // ---------------------------------------------------------------------
    std::vector<Complex> R0(P * P, kZero);
    std::vector<Complex> tau_scratch;
    if (!block_qr(V_curr.data(), N, P, R0.data(), tau_scratch)) {
        throw std::runtime_error(
            "compute_ground_state_block_dssf: initial block QR failed "
            "(zgeqrf/zungqr error)");
    }

    // ---------------------------------------------------------------------
    // Allocate workspace: previous block, work block, A, B blocks.
    // The full basis is kept in memory: V_all[0..M-1] each N*P complex.
    // Memory footprint = (M+2) * N * P * 16 bytes.
    // ---------------------------------------------------------------------
    std::vector<Complex> V_prev(N * P, kZero);
    std::vector<Complex> V_next(N * P, kZero);
    std::vector<Complex> W(N * P, kZero);
    std::vector<Complex> A(P * P, kZero);
    std::vector<Complex> B_curr(P * P, kZero);
    std::vector<Complex> B_prev(P * P, kZero);
    std::vector<Complex> correction(P * P, kZero);

    std::vector<std::vector<Complex>> A_blocks;  // size M
    std::vector<std::vector<Complex>> B_blocks;  // size up to M-1
    A_blocks.reserve(M);
    B_blocks.reserve(M);

    std::vector<std::vector<Complex>> V_all;  // size up to M, each N*P
    V_all.reserve(M);
    V_all.emplace_back(V_curr);  // V_0

    // ---------------------------------------------------------------------
    // Step 3: block Lanczos recurrence.
    // ---------------------------------------------------------------------
    std::size_t iters_done = 0;
    for (std::size_t j = 0; j < M; ++j) {
        // W = H * V_curr  (column by column to keep H matvec API scalar)
        for (std::size_t p = 0; p < P; ++p) {
            H(V_curr.data() + p * N, W.data() + p * N, N);
        }

        // A = V_curr^dag * W   (P x P)
        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                    static_cast<lapack_int>(P), static_cast<lapack_int>(P),
                    static_cast<lapack_int>(N),
                    &kOne, V_curr.data(), static_cast<lapack_int>(N),
                    W.data(), static_cast<lapack_int>(N),
                    &kZero, A.data(), static_cast<lapack_int>(P));
        hermitise(A.data(), P);

        // W = W - V_curr * A
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<lapack_int>(N), static_cast<lapack_int>(P),
                    static_cast<lapack_int>(P),
                    &kNegOne, V_curr.data(), static_cast<lapack_int>(N),
                    A.data(), static_cast<lapack_int>(P),
                    &kOne, W.data(), static_cast<lapack_int>(N));

        // W = W - V_prev * B_prev^dag   (skip on j == 0)
        if (j > 0) {
            cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                        static_cast<lapack_int>(N), static_cast<lapack_int>(P),
                        static_cast<lapack_int>(P),
                        &kNegOne, V_prev.data(), static_cast<lapack_int>(N),
                        B_prev.data(), static_cast<lapack_int>(P),
                        &kOne, W.data(), static_cast<lapack_int>(N));
        }

        // Periodic full reorthogonalization against all stored basis
        // blocks (CGS-2 to recover loss of orthogonality at finite
        // precision -- mirrors the eigensolver block Lanczos).
        const bool do_full_reorth =
            params.reorth_interval > 0 &&
            j > 0 &&
            (j % params.reorth_interval == 0);
        const int passes = do_full_reorth ? 2 : 1;
        for (int pass = 0; pass < passes; ++pass) {
            const std::size_t kmax = do_full_reorth ? V_all.size() : 1;
            for (std::size_t k = 0; k < kmax; ++k) {
                const Complex* V_ptr = do_full_reorth
                    ? V_all[k].data()
                    : V_curr.data();  // local: against current block only
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            static_cast<lapack_int>(P), static_cast<lapack_int>(P),
                            static_cast<lapack_int>(N),
                            &kOne, V_ptr, static_cast<lapack_int>(N),
                            W.data(), static_cast<lapack_int>(N),
                            &kZero, correction.data(), static_cast<lapack_int>(P));
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<lapack_int>(N), static_cast<lapack_int>(P),
                            static_cast<lapack_int>(P),
                            &kNegOne, V_ptr, static_cast<lapack_int>(N),
                            correction.data(), static_cast<lapack_int>(P),
                            &kOne, W.data(), static_cast<lapack_int>(N));
            }
            // Also reorth against V_prev locally if we're not in full mode.
            if (!do_full_reorth && j > 0) {
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            static_cast<lapack_int>(P), static_cast<lapack_int>(P),
                            static_cast<lapack_int>(N),
                            &kOne, V_prev.data(), static_cast<lapack_int>(N),
                            W.data(), static_cast<lapack_int>(N),
                            &kZero, correction.data(), static_cast<lapack_int>(P));
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            static_cast<lapack_int>(N), static_cast<lapack_int>(P),
                            static_cast<lapack_int>(P),
                            &kNegOne, V_prev.data(), static_cast<lapack_int>(N),
                            correction.data(), static_cast<lapack_int>(P),
                            &kOne, W.data(), static_cast<lapack_int>(N));
            }
        }

        A_blocks.push_back(A);
        iters_done = j + 1;

        // Last iteration: don't need to compute the next block.
        if (j + 1 == M) break;

        // QR factorise W -> V_next * B_curr.
        std::memcpy(V_next.data(), W.data(), N * P * sizeof(Complex));
        if (!block_qr(V_next.data(), N, P, B_curr.data(), tau_scratch)) {
            if (params.verbose) {
                std::cout << "  Block QR failed at iteration " << j
                          << "; truncating Krylov subspace.\n";
            }
            break;
        }

        // Detect deflation: if all diagonals of B_curr are below tol,
        // the block is fully deflated -- stop. Partial deflation is
        // handled implicitly by the QR returning near-zero diagonals
        // (the corresponding column of V_next is then orthogonal noise
        // and contributes negligibly to the projected matrix).
        bool fully_deflated = true;
        for (std::size_t p = 0; p < P; ++p) {
            if (std::abs(B_curr[p + p * P]) > params.deflation_tol) {
                fully_deflated = false;
                break;
            }
        }
        if (fully_deflated) {
            if (params.verbose) {
                std::cout << "  Full block deflation at iteration " << j
                          << "; truncating Krylov subspace.\n";
            }
            break;
        }

        B_blocks.push_back(B_curr);

        // Cycle: V_prev <- V_curr, V_curr <- V_next, B_prev <- B_curr.
        V_prev.swap(V_curr);
        V_curr.swap(V_next);
        B_prev = B_curr;
        V_all.emplace_back(V_curr);
    }

    results.iterations_completed = iters_done;

    // ---------------------------------------------------------------------
    // Step 4: assemble the block-tridiagonal projected matrix T (size
    // iters_done * P) and diagonalise.
    // ---------------------------------------------------------------------
    const std::size_t total_dim = iters_done * P;
    if (total_dim == 0) {
        if (params.verbose) {
            std::cout << "  Zero Krylov dimension; returning empty spectra.\n";
        }
        return results;
    }

    std::vector<Complex> T(total_dim * total_dim, kZero);
    for (std::size_t blk = 0; blk < iters_done; ++blk) {
        const auto& Ablk = A_blocks[blk];
        const std::size_t off = blk * P;
        for (std::size_t c = 0; c < P; ++c) {
            for (std::size_t r = 0; r < P; ++r) {
                T[(off + r) + (off + c) * total_dim] = Ablk[r + c * P];
            }
        }
    }
    for (std::size_t blk = 0; blk + 1 < iters_done; ++blk) {
        if (blk >= B_blocks.size()) break;
        const auto& Bblk = B_blocks[blk];
        const std::size_t off = blk * P;
        for (std::size_t c = 0; c < P; ++c) {
            for (std::size_t r = 0; r < P; ++r) {
                // Lower block: B
                T[(off + P + r) + (off + c) * total_dim] = Bblk[r + c * P];
                // Upper block: B^dag
                T[(off + r) + (off + P + c) * total_dim] = std::conj(Bblk[c + r * P]);
            }
        }
    }

    std::vector<double> eig(total_dim, 0.0);
    {
        const lapack_int info = LAPACKE_zheevd(
            LAPACK_COL_MAJOR, 'V', 'U',
            static_cast<lapack_int>(total_dim),
            reinterpret_cast<lapack_complex_double*>(T.data()),
            static_cast<lapack_int>(total_dim),
            eig.data());
        if (info != 0) {
            throw std::runtime_error(
                "compute_ground_state_block_dssf: LAPACKE_zheevd failed with info="
                + std::to_string(info));
        }
    }
    // After zheevd: T columns hold the eigenvectors u_n, eig[n] = mu_n.

    // ---------------------------------------------------------------------
    // Step 5: per-channel weights w_{n, p} = (top-P block of u_n)^dag * R0[:, p]
    // For each Ritz eigenvalue mu_n we have one column u_n of T (size MP).
    // Take its first P entries u_n[0..P-1], conjugate, and contract with
    // R0(:, p) which lives in C^P. The squared modulus is the spectral
    // weight at (mu_n, channel p).
    // ---------------------------------------------------------------------
    const double inv_pi = 1.0 / M_PI;
    const double eta = params.broadening;

    // Precompute weights in an n_eig x P matrix (column-major n_eig fast).
    std::vector<double> weights(total_dim * P, 0.0);
    for (std::size_t n = 0; n < total_dim; ++n) {
        const Complex* u_top = T.data() + n * total_dim;  // first P entries
        for (std::size_t p = 0; p < P; ++p) {
            Complex w(0.0, 0.0);
            for (std::size_t k = 0; k < P; ++k) {
                w += std::conj(u_top[k]) * R0[k + p * P];
            }
            weights[n + p * total_dim] = std::norm(w);  // |w|^2
        }
    }

    // Lorentzian-broadened spectra. Pillar 2 of the "Save and DSSF
    // Upgrades" plan (May 2026): drop the ``P * num_omega > 4096``
    // gate. The serial fall-through was a tiny-grid micro-optimisation
    // that hid the user's expectation that "DSSF is omega-parallel"
    // -- the OpenMP overhead at the typical production grids
    // (P ~ 4, num_omega >= 200) is negligible.
    #pragma omp parallel for schedule(static) collapse(2)
    for (std::size_t p = 0; p < P; ++p) {
        for (std::size_t iw = 0; iw < params.num_omega_points; ++iw) {
            const double omega = results.frequencies[iw];
            double s = 0.0;
            for (std::size_t n = 0; n < total_dim; ++n) {
                const double d = omega - (eig[n] - ground_state_energy);
                s += weights[n + p * total_dim] * (eta * inv_pi) /
                     (d * d + eta * eta);
            }
            results.spectral[p][iw] = s;
        }
    }

    // Sum-rule diagnostic per channel (rank 0 / verbose only).
    if (params.verbose) {
        for (std::size_t p = 0; p < P; ++p) {
            double integral = 0.0;
            for (std::size_t i = 1; i < params.num_omega_points; ++i) {
                const double dw = results.frequencies[i] - results.frequencies[i - 1];
                integral += 0.5 * (results.spectral[p][i] + results.spectral[p][i - 1]) * dw;
            }
            const double target = results.channel_norm_sq[p];
            std::cout << std::setprecision(6)
                      << "  Channel " << p << ": integral = " << integral
                      << " ; ||O_" << p << "|0>||^2 = " << target
                      << " ; ratio = " << (target > 1e-14 ? integral / target : 0.0)
                      << "\n";
        }
    }

    return results;
}

} // namespace ed::block_lanczos_dssf
