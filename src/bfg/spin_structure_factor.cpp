// =============================================================================
// src/bfg/spin_structure_factor.cpp
//
// Lifted from compute_bfg_order_parameters.cpp without behaviour changes:
//   * same per-k-point OpenMP loop,
//   * same per-pair minimum-image displacement,
//   * same SzSz / S-S+ accumulator types (double / Complex),
//   * same `S(q) = SzSz(q) + Re S^{-+}(q)` reduction,
//   * same q_max search and m_translation = sqrt(max / N).
//
// The only deltas vs. the original CPU-driver implementation are scoping
// (anonymous namespace for I, namespace ed::bfg for the rest) and
// surfacing the `[ms]` timing through std::cout exactly as before.
// =============================================================================

#include "ed/bfg/spin_structure_factor.h"

#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>

namespace ed::bfg {

namespace {
constexpr Complex I{0.0, 1.0};
}

StructureFactorResult compute_spin_structure_factor(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const std::vector<std::vector<double>>& szsz_corr,
    const Cluster& cluster
) {
    StructureFactorResult result;
    const int n_k = static_cast<int>(cluster.k_points.size());
    const int n_sites = cluster.n_sites;

    result.s_q.resize(n_k, Complex(0.0, 0.0));
    result.s_q_smsp.resize(n_k, Complex(0.0, 0.0));
    result.s_q_szsz.resize(n_k, Complex(0.0, 0.0));

    std::cout << "Computing S(q) at " << n_k
              << " k-points (full Heisenberg)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int ik = 0; ik < n_k; ++ik) {
        const auto& q = cluster.k_points[ik];
        Complex s_q_smsp(0.0, 0.0);
        double  s_q_szsz = 0.0;

        for (int i = 0; i < n_sites; ++i) {
            for (int j = 0; j < n_sites; ++j) {
                // Minimum-image displacement keeps the Fourier phases
                // PBC-correct on small clusters.
                const auto dr = cluster.minimum_image_displacement(i, j);
                const double phase_arg = q[0] * dr[0] + q[1] * dr[1];
                const Complex phase = std::exp(I * phase_arg);

                s_q_smsp += smsp_corr[i][j] * phase;
                s_q_szsz += szsz_corr[i][j] * std::real(phase);
            }
        }

        // S(q) = SzSz(q) + (1/2)(S-S+(q) + S+S-(q)).
        // <S+_i S-_j> = <S-_j S+_i>* so on real lattices with inversion
        // the half-sum collapses to Re S^{-+}(q); the historical CPU
        // driver applied that reduction here, and we preserve it.
        const double inv_n = 1.0 / static_cast<double>(n_sites);
        result.s_q_smsp[ik] = s_q_smsp * inv_n;
        result.s_q_szsz[ik] = Complex(s_q_szsz * inv_n, 0.0);
        result.s_q[ik] =
            result.s_q_szsz[ik] + std::real(result.s_q_smsp[ik]);
    }

    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        const double val = std::abs(result.s_q[ik]);
        if (val > max_val) {
            max_val = val;
            result.q_max_idx = ik;
            result.s_q_max = result.s_q[ik];
            result.q_max = cluster.k_points[ik];
        }
    }

    result.m_translation = std::sqrt(max_val / n_sites);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;

    return result;
}

}  // namespace ed::bfg
