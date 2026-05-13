#pragma once
// =============================================================================
// include/ed/solvers/block_lanczos_dssf.h
//
// Audit item #6: Block / band Lanczos for ground-state DSSF with multiple
// starting vectors (e.g. P momentum points or P spin-channel pairs).
//
// Mathematical setup
// ------------------
// Given a Hamiltonian H, a normalized ground state |0> with energy E_0,
// and P operators {O_p}_{p=1..P}, the standard scalar Lanczos approach
// builds one Krylov chain per p:
//
//     phi_p = O_p |0> ;    S_p(omega) = -1/pi Im <phi_p | (omega + E_0 - H + i eta)^{-1} | phi_p>
//
// That requires P independent matrix-free Hamiltonian builds and P
// Lanczos chains. Block Lanczos starts with a *block* of P vectors at
// once and runs a single block recurrence:
//
//     V_0 R_0 = QR(Phi),   Phi = [phi_1 | ... | phi_P]    (column block)
//     W_j = H V_j  -  V_{j-1} B_j^dag             (j >= 1, B_0 := 0)
//     A_j = V_j^dag W_j                            (P x P, Hermitian)
//     W_j = W_j - V_j A_j
//     V_{j+1} B_{j+1} = QR(W_j)                    (block QR factorization)
//
// After M block iterations the projected matrix
//
//                | A_0    B_1^dag                   |
//                | B_1    A_1     B_2^dag           |
//        T  =    |        B_2     A_2     B_3^dag   |        (M*P square)
//                |             ...     ...     ...  |
//                |               B_{M-1}    A_{M-1} |
//
// is block tridiagonal of size MP x MP. Diagonalising T (with LAPACK
// zheevd) gives Ritz pairs (mu_n, u_n). The spectral function for the
// p-th channel is then
//
//     S_p(omega) = sum_n |w_{n,p}|^2 * Lorentzian(omega - (mu_n - E_0); eta)
//
// where w_{n,p} = (top-P block of u_n)^dag * R_0[:,p] is the Krylov
// projection of phi_p onto Ritz state n. (Equivalently this is the
// generalisation of |<u_n|phi>|^2 that appears in the scalar case.)
//
// Practical advantages over P scalar Lanczos chains:
//   - One H-apply call per block step instead of P scalar calls (the cost
//     is identical, but the SpMV pipeline runs over a contiguous N*P
//     buffer rather than P separate vectors -- hits BLAS-3 territory in
//     the V_curr * Aj / V_prev * B_prev^dag corrections);
//   - Ritz pairs of T capture cross-channel coupling for free (off-block
//     eigenvectors), so peaks shared across channels are reconstructed
//     more accurately at the same MP than from M iterations of P scalar
//     chains;
//   - Memory peaks at (M+2) * N * P complex doubles for the basis blocks;
//     for typical (N, M, P) = (50000, 80, 12) that's ~770 MB which fits
//     on a single workstation and avoids the disk-paged variant used by
//     the eigensolver block Lanczos in src/solvers/cpu/lanczos.cpp.
//
// API mirrors the scalar `compute_ground_state_dssf` in ftlm.h to keep
// downstream wiring trivial: the workflow constructs a list of
// per-channel matvec lambdas instead of calling `compute_ground_state_dssf`
// in a loop.
// =============================================================================

#include <complex>
#include <functional>
#include <string>
#include <vector>

namespace ed::block_lanczos_dssf {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

struct BlockLanczosDSSFParameters {
    /// Number of block Lanczos iterations (M). Total Krylov dim is M*P.
    /// Mirrors `GroundStateDSSFParameters::krylov_dim` semantics.
    std::size_t krylov_dim = 100;

    /// Lorentzian broadening eta (>= 0).
    double broadening = 0.05;

    /// Frequency grid.
    double omega_min = -10.0;
    double omega_max =  10.0;
    std::size_t num_omega_points = 1024;

    /// Tolerance for QR rank deficiency (column norm of W after correction).
    /// If the j-th diagonal of B_{j+1} drops below this we *deflate* the
    /// affected column to zero rather than aborting; the projected
    /// matrix is then block-tridiagonal with a degenerate B and the
    /// continued-fraction weight for the deflated channel terminates
    /// naturally at iteration j.
    double deflation_tol = 1e-12;

    /// Full reorthogonalization against all stored basis blocks every
    /// `reorth_interval` iterations. Single-block (V_curr, V_prev) local
    /// reorth is always done. Set to 0 to disable periodic full reorth
    /// (only safe for very small krylov_dim).
    std::size_t reorth_interval = 3;

    /// Verbose progress printing (rank-0 only when MPI is enabled).
    bool verbose = false;
};

struct MultiChannelDSSFResults {
    /// Frequency grid (size num_omega_points).
    std::vector<double> frequencies;

    /// Spectral functions, one per starting channel. spectral[p] has size
    /// num_omega_points and integrates (trapezoid) to ||O_p|0>||^2 in the
    /// limit of converged Krylov dimension.
    std::vector<std::vector<double>> spectral;

    /// Per-channel norms ||O_p|0>||^2. Useful for sum-rule diagnostics.
    std::vector<double> channel_norm_sq;

    /// Effective block Lanczos iterations completed (M). May be less than
    /// `params.krylov_dim` if every column of W deflated.
    std::size_t iterations_completed = 0;

    /// Number of channels P.
    std::size_t block_size = 0;
};

/**
 * @brief Compute T=0 multi-channel dynamical structure factor via block Lanczos.
 *
 * @param H            Matrix-free Hamiltonian, signature (in, out, dim).
 * @param O_list       P operator matvec callables, each (in, out, dim).
 *                     Channel p starting vector is O_list[p](|0>).
 * @param ground_state Normalized ground state |0> (size N).
 * @param ground_state_energy E_0.
 * @param N            Hilbert space dimension.
 * @param params       Algorithm parameters.
 * @return MultiChannelDSSFResults with one spectral function per channel.
 *
 * Throws std::invalid_argument if O_list is empty or N != ground_state.size().
 */
MultiChannelDSSFResults compute_ground_state_block_dssf(
    const std::function<void(const Complex*, Complex*, std::size_t)>& H,
    const std::vector<std::function<void(const Complex*, Complex*, std::size_t)>>& O_list,
    const ComplexVector& ground_state,
    double ground_state_energy,
    std::size_t N,
    const BlockLanczosDSSFParameters& params);

} // namespace ed::block_lanczos_dssf
