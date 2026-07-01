#pragma once
// =============================================================================
// include/ed/thermal/oftlm_kernel.h
//
// OFTLM -- Orthogonalized Finite-Temperature Lanczos Method
// (Morita & Tohyama, Phys. Rev. Research 2, 013205 (2020)).
//
// OFTLM = FTLM with the N_V lowest eigenstates treated EXACTLY and the random
// starting vectors orthogonalized against those N_V eigenvectors. This removes
// the well-known low-temperature bias of plain FTLM (where the stochastic trace
// under-resolves the few dominant low-energy states). As N_V -> 0 it reduces to
// FTLM.
//
// Estimator (with a common ground-state shift e_min = eps_0):
//
//   Z(beta)   = sum_{i<N_V} e^{-beta (eps_i - e_min)}
//             + (D - N_V)/R * sum_r sum_j |<r~|psi_j^r>|^2 e^{-beta (eps~_j^r - e_min)}
//   <E> Z     = sum_{i<N_V} eps_i e^{-beta(...)} + (D-N_V)/R * sum_r sum_j eps~_j |..|^2 e^{-beta(...)}
//   <E^2> Z   = ... eps~_j^2 ...
//   Cv        = beta^2 (<E^2> - <E>^2),   S = ln Z + beta(<E> - e_min)
//
// where |r~> is a random vector orthogonalized against the N_V exact
// eigenvectors and renormalized, D is the (sector) Hilbert dimension, R the
// number of random samples. Because the (D - N_V) factor already scales the
// random part to the (D - N_V)-dimensional complement, Z is the FULL (shifted)
// trace -- there is no separate ln(D) term as in plain FTLM.
//
// CPU-only lane (matches LTLM). Consumes the host term-matvec + the shared host
// Lanczos helpers in ed/solvers/lanczos.h.
// =============================================================================

#include <complex>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <ed/thermal/ftlm_kernel.h>   // FtlmResult (reused as the return type)

namespace ed::thermal {

struct OftlmOptions {
    std::size_t   num_samples  = 20;   ///< R: random samples for the stochastic part
    std::size_t   krylov_dim   = 100;  ///< M: Lanczos steps per random sample
    std::size_t   num_exact    = 8;    ///< N_V: low-lying states treated exactly
    std::size_t   exact_krylov = 0;    ///< Lanczos steps for the exact eigenpairs (0 -> auto)
    std::vector<double> betas;         ///< inverse-temperature grid (strictly positive)
    std::uint64_t random_seed  = 0;
    std::string   output_dir;
};

/// Orthogonalized FTLM on a single (CPU, single-rank) sector.
/// @param apply_H  host term-matvec: out = H * in, length N.
/// @param N        sector Hilbert dimension (local == global on CPU).
FtlmResult oftlm_cpu(
    const std::function<void(const std::complex<double>*,
                             std::complex<double>*, int)>& apply_H,
    std::uint64_t          N,
    const OftlmOptions&    opts);

}  // namespace ed::thermal
