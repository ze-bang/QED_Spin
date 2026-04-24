// =============================================================================
// include/ed/bfg/spin_structure_factor.h
//
// Site-resolved spin structure factor S(q) for the BFG order-parameter
// pipeline (P2.1 spin-structure-factor slice).
//
// Lifted out of `compute_bfg_order_parameters.cpp` so the GPU driver, the
// Python bindings, and the CPU driver all share the same authoritative
// implementation. Acts on the precomputed two-body correlation tables
// produced by `ed_bfg::compute_smsp_correlations` and
// `ed_bfg::compute_szsz_correlations`; no wavefunction passes through this
// module.
//
// Phase convention: minimum-image displacement r_{ij} from
// `Cluster::minimum_image_displacement(i, j)` so the Fourier sum
// respects PBC.
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <array>
#include <complex>
#include <vector>

#include "ed/bfg/cluster.h"

namespace ed::bfg {

using Complex = std::complex<double>;

/**
 * Output of `compute_spin_structure_factor`.
 *
 * `s_q[ik]` is the full Heisenberg structure factor at `cluster.k_points[ik]`
 * decomposed into the longitudinal (`s_q_szsz`) and transverse
 * (`s_q_smsp`) channels:
 *
 *   S(q)        = (1/N) sum_{i,j} e^{i q . r_{ij}} <S_i . S_j>
 *               = S^zz(q) + (1/2)(S^{-+}(q) + S^{+-}(q))
 *               = S^zz(q) + Re S^{-+}(q)            (real lattices)
 *
 * `q_max_idx` / `q_max` / `s_q_max` flag the abscissa of the maximum
 * |S(q)|, and `m_translation = sqrt(max |S(q)| / N)` is the canonical
 * BFG translation order parameter.
 */
struct StructureFactorResult {
    std::vector<Complex> s_q;
    std::vector<Complex> s_q_smsp;
    std::vector<Complex> s_q_szsz;
    int q_max_idx = 0;
    Complex s_q_max{0.0, 0.0};
    std::array<double, 2> q_max{0.0, 0.0};
    double m_translation = 0.0;
};

/**
 * Compute S(q) at every k-point in `cluster.k_points` from a precomputed
 * `<S^-_i S^+_j>` matrix and `<S^z_i S^z_j>` matrix. Both inputs are
 * site-by-site `n_sites x n_sites` tables; no symmetrisation is assumed.
 *
 * The function emits a single line of progress to stdout (matching the
 * historical CPU-driver behaviour) so headless workflows that compare
 * stdout transcripts continue to pass; downstream wrappers can redirect
 * the stream when running in a notebook.
 */
StructureFactorResult compute_spin_structure_factor(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const std::vector<std::vector<double>>& szsz_corr,
    const Cluster& cluster
);

}  // namespace ed::bfg
