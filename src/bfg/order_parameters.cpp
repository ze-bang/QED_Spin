// =============================================================================
// src/bfg/order_parameters.cpp
//
// Implementation of the BFG order-parameter physics kernels (P2.1
// order-parameters slice). Lifted verbatim from
// `compute_bfg_order_parameters.cpp` so the CPU driver, the GPU driver, and
// downstream Python bindings share one authoritative implementation.
//
// All numerical and console-progress behaviour is preserved. The kernels
// rely only on other `ed_bfg` library functions (correlations, structure
// factor, ring observables, spin structure factor) and on `Cluster`, so
// this TU has no dependency on the argv driver.
//
// Audit ref: P2.1.
// =============================================================================

#include "ed/bfg/order_parameters.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <utility>

#include "ed/bfg/correlations.h"
#include "ed/bfg/ring_observables.h"
#include "ed/bfg/spin_structure_factor.h"
#include "ed/bfg/structure_factor.h"
#include "ed/bfg/topology.h"

namespace ed::bfg {

namespace {

constexpr double kPI = 3.14159265358979323846;
constexpr Complex kI{0.0, 1.0};

// Map (alpha, beta) bond-orientation pair (0..2 each) to the canonical
// 6-element flat index used throughout the per-orientation tables:
//   0 = (0,0), 1 = (0,1), 2 = (0,2), 3 = (1,1), 4 = (1,2), 5 = (2,2)
inline int orient_pair_idx(int alpha, int beta) {
    if (alpha > beta) std::swap(alpha, beta);
    if (alpha == 0) return beta;
    if (alpha == 1) return 2 + beta;
    return 5;
}

}  // namespace

// -----------------------------------------------------------------------------
// Generic nematic order from complex bond expectations
// -----------------------------------------------------------------------------
NematicResult compute_nematic_order(
    const std::map<std::pair<int, int>, Complex>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type
) {
    NematicResult result;
    result.bond_type = bond_type;
    std::array<Complex, 3> sum_by_orient = {0.0, 0.0, 0.0};
    std::array<int, 3> count_by_orient = {0, 0, 0};

    for (const auto& [edge, exp_val] : bond_exp) {
        int alpha = cluster.bond_orientation.at(edge);
        sum_by_orient[alpha] += exp_val;
        count_by_orient[alpha]++;
    }

    for (int alpha = 0; alpha < 3; ++alpha) {
        if (count_by_orient[alpha] > 0) {
            result.O_bar[alpha] = sum_by_orient[alpha] /
                                  static_cast<double>(count_by_orient[alpha]);
        }
    }

    // psi_nem = sum_alpha omega^alpha O_bar_alpha, omega = exp(2 pi i / 3)
    Complex omega = std::exp(2.0 * kPI * kI / 3.0);
    result.psi_nem = result.O_bar[0]
                   + omega * result.O_bar[1]
                   + omega * omega * result.O_bar[2];
    result.m_nem = std::abs(result.psi_nem);

    std::array<double, 3> mags = {std::abs(result.O_bar[0]),
                                  std::abs(result.O_bar[1]),
                                  std::abs(result.O_bar[2])};
    double max_mag = *std::max_element(mags.begin(), mags.end());
    double min_mag = *std::min_element(mags.begin(), mags.end());
    result.anisotropy = (max_mag > 1e-10) ? (max_mag - min_mag) / max_mag : 0.0;

    std::cout << "Nematic order (" << bond_type << "): m_nem = " << result.m_nem
              << ", anisotropy = " << result.anisotropy << std::endl;

    return result;
}

NematicResult compute_nematic_order_real(
    const std::map<std::pair<int, int>, double>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type
) {
    std::map<std::pair<int, int>, Complex> bond_exp_complex;
    for (const auto& [edge, val] : bond_exp) {
        bond_exp_complex[edge] = Complex(val, 0.0);
    }
    return compute_nematic_order(bond_exp_complex, cluster, bond_type);
}

// -----------------------------------------------------------------------------
// Compute VBS (Valence Bond Solid) order with 4-site dimer correlations
// -----------------------------------------------------------------------------
VBSResult compute_vbs_order(
    const std::vector<Complex>& psi,
    const std::map<std::pair<int, int>, Complex>& xy_bond_exp,
    const std::map<std::pair<int, int>, double>& heisenberg_bond_exp,
    const Cluster& cluster,
    int n_q_grid
) {
    VBSResult result;
    result.n_q_grid = n_q_grid;
    int n_bonds = cluster.edges_nn.size();
    result.n_bonds = n_bonds;
    int n_k = cluster.k_points.size();

    if (n_bonds == 0) {
        result.m_vbs_xy = 0.0;
        result.m_vbs_heis = 0.0;
        return result;
    }

    std::cout << "Computing VBS order using efficient Fourier-space method..."
              << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(),
                                           cluster.edges_nn.end());

    std::array<std::vector<std::pair<int, int>>, 3> edges_by_orient;
    std::array<std::vector<std::array<double, 2>>, 3> centers_by_orient;
    std::vector<std::array<double, 2>> all_bond_centers(n_bonds);

    for (int b = 0; b < n_bonds; ++b) {
        int i = edges[b].first;
        int j = edges[b].second;
        auto center = cluster.bond_center_pbc(i, j);
        all_bond_centers[b] = center;

        int orient = cluster.bond_orientation.at(edges[b]);
        edges_by_orient[orient].push_back(edges[b]);
        centers_by_orient[orient].push_back(center);
    }

    result.n_bonds_per_orientation = {
        static_cast<int>(edges_by_orient[0].size()),
        static_cast<int>(edges_by_orient[1].size()),
        static_cast<int>(edges_by_orient[2].size())
    };

    std::cout << "  Bonds per orientation: "
              << result.n_bonds_per_orientation[0] << ", "
              << result.n_bonds_per_orientation[1] << ", "
              << result.n_bonds_per_orientation[2] << std::endl;

    // -------------------------------------------------------------------------
    // Mean bond values <D_alpha> for each orientation
    // -------------------------------------------------------------------------
    std::array<Complex, 3> D_mean_xy_orient = {};
    std::array<double, 3> D_mean_heis_orient = {};

    double sum_xy = 0.0, sum_heis = 0.0;
    for (int b = 0; b < n_bonds; ++b) {
        sum_xy += std::real(xy_bond_exp.at(edges[b]));
        sum_heis += heisenberg_bond_exp.at(edges[b]);
    }
    result.D_mean_xy = sum_xy / n_bonds;
    result.D_mean_heis = sum_heis / n_bonds;

    for (int alpha = 0; alpha < 3; ++alpha) {
        Complex sum_a = 0.0;
        double sum_h = 0.0;
        for (const auto& edge : edges_by_orient[alpha]) {
            sum_a += xy_bond_exp.at(edge);
            sum_h += heisenberg_bond_exp.at(edge);
        }
        if (!edges_by_orient[alpha].empty()) {
            D_mean_xy_orient[alpha] =
                sum_a / static_cast<double>(edges_by_orient[alpha].size());
            D_mean_heis_orient[alpha] =
                sum_h / edges_by_orient[alpha].size();
        }
    }

    result.S_d_xy.resize(n_k, 0.0);
    result.S_d_heis.resize(n_k, 0.0);
    result.S_d_xy_oriented.resize(n_k);
    result.S_d_heis_oriented.resize(n_k);

    std::cout << "  Computing S_D(q) at " << n_k
              << " k-points using Fourier method..." << std::flush;

    if (memory_efficient_mode_enabled()) {
        std::cout << "\n  [Memory-efficient mode: using direct SF computation]"
                  << std::endl;

        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];

            auto sf_xy = compute_dimer_sf_direct(psi, edges, all_bond_centers, q);
            Complex s_d_total_xy = sf_xy.overlap - std::norm(sf_xy.expect_q1);
            result.S_d_xy[ik] = s_d_total_xy / static_cast<double>(n_bonds);

            auto sf_heis = compute_heisenberg_sf_direct(psi, edges,
                                                        all_bond_centers, q);
            Complex s_d_total_heis = sf_heis.overlap -
                                     std::norm(sf_heis.expect_q1);
            result.S_d_heis[ik] = s_d_total_heis.real() / n_bonds;

            std::array<Complex, 6> s_d_xy_orient = {};
            std::array<double, 6> s_d_heis_orient = {};
            result.S_d_xy_oriented[ik] = s_d_xy_orient;
            result.S_d_heis_oriented[ik] = s_d_heis_orient;

            if ((ik + 1) % 5 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_D(q): " << (ik + 1) << "/"
                          << n_k << " k-points..." << std::flush;
            }
        }
    } else {
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];

            std::array<std::vector<Complex>, 3> D_q_psi_xy;
            std::array<Complex, 3> D_q_expect_xy = {};

            for (int alpha = 0; alpha < 3; ++alpha) {
                if (edges_by_orient[alpha].empty()) {
                    D_q_psi_xy[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                D_q_psi_xy[alpha] = apply_dimer_fourier(
                    psi, edges_by_orient[alpha],
                    centers_by_orient[alpha], q);
                for (size_t b = 0; b < edges_by_orient[alpha].size(); ++b) {
                    double phase_arg = q[0] * centers_by_orient[alpha][b][0] +
                                       q[1] * centers_by_orient[alpha][b][1];
                    D_q_expect_xy[alpha] +=
                        std::exp(kI * phase_arg) *
                        xy_bond_exp.at(edges_by_orient[alpha][b]);
                }
            }

            std::array<Complex, 6> s_d_xy_orient = {};
            Complex s_d_total = 0.0;

            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_bonds_per_orientation[alpha];
                    int N_b = result.n_bonds_per_orientation[beta];

                    if (N_a == 0 || N_b == 0) continue;

                    // OpenMP reduction over the (potentially huge)
                    // many-body Hilbert sum. Splitting into real+imag
                    // accumulators because OpenMP cannot reduce on
                    // `std::complex` directly. Same pattern below for
                    // the Heisenberg overlap.
                    double ovl_re = 0.0;
                    double ovl_im = 0.0;
                    const Complex* a_ptr = D_q_psi_xy[alpha].data();
                    const Complex* b_ptr = D_q_psi_xy[beta].data();
                    const std::size_t Ns = psi.size();
                    #pragma omp parallel for reduction(+:ovl_re, ovl_im) schedule(static)
                    for (std::size_t s = 0; s < Ns; ++s) {
                        const Complex c = std::conj(a_ptr[s]) * b_ptr[s];
                        ovl_re += c.real();
                        ovl_im += c.imag();
                    }
                    Complex overlap(ovl_re, ovl_im);

                    Complex connected = overlap -
                        std::conj(D_q_expect_xy[alpha]) * D_q_expect_xy[beta];

                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_d_xy_orient[idx] = connected / norm;

                    if (alpha == beta) {
                        s_d_total += connected;
                    } else {
                        s_d_total += 2.0 * connected;
                    }
                }
            }

            result.S_d_xy_oriented[ik] = s_d_xy_orient;
            result.S_d_xy[ik] = s_d_total / static_cast<double>(n_bonds);

            // Heisenberg via Fourier-applied dimer kernel. Applied per
            // orientation below; the previous all-bonds invocation that
            // sat here was dead -- its (D_q_psi_heis_all, D_q_expect_heis_all)
            // outputs were never read, but the call itself is O(n_bonds × N)
            // and is the most expensive primitive in this loop.
            std::array<double, 6> s_d_heis_orient = {};
            double s_d_heis_total = 0.0;

            std::array<std::vector<Complex>, 3> D_q_psi_heis;
            std::array<Complex, 3> D_q_expect_heis = {};

            for (int alpha = 0; alpha < 3; ++alpha) {
                if (edges_by_orient[alpha].empty()) {
                    D_q_psi_heis[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                auto [dpsi, dexp] = apply_heisenberg_dimer_fourier(
                    psi, edges_by_orient[alpha],
                    centers_by_orient[alpha], q);
                D_q_psi_heis[alpha] = std::move(dpsi);
                D_q_expect_heis[alpha] = dexp;
            }

            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_bonds_per_orientation[alpha];
                    int N_b = result.n_bonds_per_orientation[beta];

                    if (N_a == 0 || N_b == 0) continue;

                    double ovl_re = 0.0;
                    double ovl_im = 0.0;
                    const Complex* a_ptr = D_q_psi_heis[alpha].data();
                    const Complex* b_ptr = D_q_psi_heis[beta].data();
                    const std::size_t Ns = psi.size();
                    #pragma omp parallel for reduction(+:ovl_re, ovl_im) schedule(static)
                    for (std::size_t s = 0; s < Ns; ++s) {
                        const Complex c = std::conj(a_ptr[s]) * b_ptr[s];
                        ovl_re += c.real();
                        ovl_im += c.imag();
                    }
                    Complex overlap(ovl_re, ovl_im);

                    Complex connected = overlap -
                        std::conj(D_q_expect_heis[alpha]) * D_q_expect_heis[beta];
                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_d_heis_orient[idx] = connected.real() / norm;

                    if (alpha == beta) {
                        s_d_heis_total += connected.real();
                    } else {
                        s_d_heis_total += 2.0 * connected.real();
                    }
                }
            }

            result.S_d_heis_oriented[ik] = s_d_heis_orient;
            result.S_d_heis[ik] = s_d_heis_total / n_bonds;

            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_D(q): " << (ik + 1) << "/"
                          << n_k << " k-points..." << std::flush;
            }
        }
    }
    std::cout << " done" << std::endl;

    double max_val_xy = 0.0, max_val_heis = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val_xy = std::abs(result.S_d_xy[ik]);
        if (val_xy > max_val_xy) {
            max_val_xy = val_xy;
            result.q_max_idx_xy = ik;
            result.s_d_max_xy = result.S_d_xy[ik];
            result.q_max_xy = cluster.k_points[ik];
        }

        double val_heis = std::abs(result.S_d_heis[ik]);
        if (val_heis > max_val_heis) {
            max_val_heis = val_heis;
            result.q_max_idx_heis = ik;
            result.s_d_max_heis = result.S_d_heis[ik];
            result.q_max_heis = cluster.k_points[ik];
        }
    }
    result.m_vbs_xy = std::sqrt(max_val_xy / n_bonds);
    result.m_vbs_heis = std::sqrt(max_val_heis / n_bonds);

    // -------------------------------------------------------------------------
    // Dense 2D q-grid for visualisation (skipped in memory-efficient mode for
    // large grids).
    // -------------------------------------------------------------------------
    if (memory_efficient_mode_enabled() && n_q_grid > 10) {
        std::cout << "  [Memory-efficient mode: skipping dense 2D VBS grid "
                     "(use --n-q-grid 10 to enable)]" << std::endl;
        result.S_d_xy_2d.resize(0);
        result.S_d_heis_2d.resize(0);
    } else {
        result.S_d_xy_2d.resize(n_q_grid,
                                std::vector<Complex>(n_q_grid, 0.0));
        result.S_d_heis_2d.resize(n_q_grid,
                                  std::vector<double>(n_q_grid, 0.0));

        std::cout << "  Computing S_D(q) on " << n_q_grid << "x" << n_q_grid
                  << " grid..." << std::flush;

        for (int i1 = 0; i1 < n_q_grid; ++i1) {
            for (int i2 = 0; i2 < n_q_grid; ++i2) {
                double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
                double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
                std::array<double, 2> qvec = {
                    q1 * cluster.b1[0] + q2 * cluster.b2[0],
                    q1 * cluster.b1[1] + q2 * cluster.b2[1]
                };

                if (memory_efficient_mode_enabled()) {
                    auto sf_xy = compute_dimer_sf_direct(psi, edges,
                                                         all_bond_centers,
                                                         qvec);
                    result.S_d_xy_2d[i1][i2] =
                        (sf_xy.overlap - std::norm(sf_xy.expect_q1)) /
                        static_cast<double>(n_bonds);

                    auto sf_heis = compute_heisenberg_sf_direct(
                        psi, edges, all_bond_centers, qvec);
                    result.S_d_heis_2d[i1][i2] =
                        (sf_heis.overlap -
                         std::norm(sf_heis.expect_q1)).real() / n_bonds;
                } else {
                    auto D_q_psi = apply_dimer_fourier(
                        psi, edges, all_bond_centers, qvec);
                    Complex D_q_expect = 0.0;
                    for (int b = 0; b < n_bonds; ++b) {
                        double phase_arg = qvec[0] * all_bond_centers[b][0] +
                                           qvec[1] * all_bond_centers[b][1];
                        D_q_expect += std::exp(kI * phase_arg) *
                                      xy_bond_exp.at(edges[b]);
                    }
                    // |D_q_psi|^2 reduction over the full Hilbert space.
                    // Real-only since |c|^2 = c.real()^2 + c.imag()^2.
                    double ovl_re = 0.0;
                    {
                        const Complex* p = D_q_psi.data();
                        const std::size_t Ns = psi.size();
                        #pragma omp parallel for reduction(+:ovl_re) schedule(static)
                        for (std::size_t s = 0; s < Ns; ++s) {
                            const double re = p[s].real();
                            const double im = p[s].imag();
                            ovl_re += re * re + im * im;
                        }
                    }
                    Complex overlap(ovl_re, 0.0);
                    result.S_d_xy_2d[i1][i2] =
                        (overlap - std::norm(D_q_expect)) /
                        static_cast<double>(n_bonds);

                    auto [D_q_psi_h, D_q_expect_h] =
                        apply_heisenberg_dimer_fourier(
                            psi, edges, all_bond_centers, qvec);
                    double ovl_re_h = 0.0;
                    {
                        const Complex* p = D_q_psi_h.data();
                        const std::size_t Ns = psi.size();
                        #pragma omp parallel for reduction(+:ovl_re_h) schedule(static)
                        for (std::size_t s = 0; s < Ns; ++s) {
                            const double re = p[s].real();
                            const double im = p[s].imag();
                            ovl_re_h += re * re + im * im;
                        }
                    }
                    Complex overlap_h(ovl_re_h, 0.0);
                    result.S_d_heis_2d[i1][i2] =
                        (overlap_h - std::norm(D_q_expect_h)).real() /
                        n_bonds;
                }
            }

            if ((i1 + 1) % 10 == 0 || i1 == n_q_grid - 1) {
                std::cout << "\r  Computing S_D(q) 2D grid: " << (i1 + 1)
                          << "/" << n_q_grid << " rows..." << std::flush;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done" << std::endl;
    std::cout << "  VBS order (XY):         m_vbs = " << result.m_vbs_xy
              << " at q = (" << result.q_max_xy[0] << ", "
              << result.q_max_xy[1] << ")" << std::endl;
    std::cout << "  VBS order (Heisenberg): m_vbs = " << result.m_vbs_heis
              << " at q = (" << result.q_max_heis[0] << ", "
              << result.q_max_heis[1] << ")"
              << " [" << duration.count() << " ms]" << std::endl;

    return result;
}

// -----------------------------------------------------------------------------
// Plaquette / bowtie resonance order
// -----------------------------------------------------------------------------
PlaquetteResult compute_plaquette_order(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    int n_q_grid
) {
    PlaquetteResult result;
    int n_k = cluster.k_points.size();

    std::cout << "Computing plaquette/bowtie resonance order using efficient "
                 "Fourier method..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto triangles = find_triangles(cluster);
    result.n_triangles = triangles.size();
    std::cout << "  Found " << result.n_triangles
              << " triangular plaquettes" << std::endl;

    auto bowties = find_bowties(cluster);
    result.n_plaquettes = bowties.size();
    std::cout << "  Found " << result.n_plaquettes
              << " bowtie plaquettes (5-site, 2 triangles sharing corner)"
              << std::endl;

    if (result.n_plaquettes == 0) {
        result.m_plaquette = 0.0;
        result.P_mean = 0.0;
        result.resonance_strength = 0.0;
        return result;
    }

    std::array<std::vector<Bowtie>, 3> bowties_by_orient;
    const std::vector<Bowtie>& all_bowties = bowties;

    result.P_r.resize(result.n_plaquettes);
    result.centers.resize(result.n_plaquettes);
    result.orientations.resize(result.n_plaquettes);

    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        const auto& bt = bowties[idx];
        result.centers[idx] = bt.center;
        result.orientations[idx] = bt.orientation;
        bowties_by_orient[bt.orientation].push_back(bt);
    }

    result.n_plaquettes_per_orientation = {
        static_cast<int>(bowties_by_orient[0].size()),
        static_cast<int>(bowties_by_orient[1].size()),
        static_cast<int>(bowties_by_orient[2].size())
    };

    std::cout << "  Bowties per orientation: "
              << result.n_plaquettes_per_orientation[0] << ", "
              << result.n_plaquettes_per_orientation[1] << ", "
              << result.n_plaquettes_per_orientation[2] << std::endl;

    std::cout << "  Computing individual bowtie expectations..." << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        const auto& bt = bowties[idx];
        result.P_r[idx] = compute_bowtie_resonance(psi, bt.s1, bt.s2,
                                                   bt.s3, bt.s4);
    }
    std::cout << " done" << std::endl;

    std::array<Complex, 3> P_mean_orient = {};
    double sum_P = 0.0, sum_abs_P = 0.0;

    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        sum_P += result.P_r[idx].real();
        sum_abs_P += std::abs(result.P_r[idx]);
        P_mean_orient[result.orientations[idx]] += result.P_r[idx];
    }
    result.P_mean = sum_P / result.n_plaquettes;
    result.resonance_strength = sum_abs_P / result.n_plaquettes;

    for (int alpha = 0; alpha < 3; ++alpha) {
        if (result.n_plaquettes_per_orientation[alpha] > 0) {
            P_mean_orient[alpha] /=
                static_cast<double>(result.n_plaquettes_per_orientation[alpha]);
        }
    }

    result.S_p.resize(n_k, 0.0);
    result.S_p_oriented.resize(n_k);

    std::cout << "  Computing S_P(q) at " << n_k
              << " k-points using Fourier method..." << std::flush;

    if (memory_efficient_mode_enabled()) {
        std::cout << "\n  [Memory-efficient mode: using simplified plaquette "
                     "SF computation]" << std::endl;

        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];

            Complex P_q_expect = 0.0;
            Complex P_q_expect_sq = 0.0;

            for (int p = 0; p < result.n_plaquettes; ++p) {
                double phase_arg = q[0] * result.centers[p][0] +
                                   q[1] * result.centers[p][1];
                Complex phase = std::exp(kI * phase_arg);
                P_q_expect += phase * result.P_r[p];
                P_q_expect_sq += std::norm(result.P_r[p]);
            }

            result.S_p[ik] = (P_q_expect_sq -
                              std::norm(P_q_expect) / result.n_plaquettes) /
                             static_cast<double>(result.n_plaquettes);

            result.S_p_oriented[ik] = {};

            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_P(q): " << (ik + 1) << "/"
                          << n_k << " k-points..." << std::flush;
            }
        }
    } else {
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];

            std::array<std::vector<Complex>, 3> P_q_psi;
            std::array<Complex, 3> P_q_expect = {};

            for (int alpha = 0; alpha < 3; ++alpha) {
                if (bowties_by_orient[alpha].empty()) {
                    P_q_psi[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                P_q_psi[alpha] =
                    apply_bowtie_fourier(bowties_by_orient[alpha], psi, q);

                for (int p = 0; p < result.n_plaquettes; ++p) {
                    if (result.orientations[p] == alpha) {
                        double phase_arg = q[0] * result.centers[p][0] +
                                           q[1] * result.centers[p][1];
                        P_q_expect[alpha] +=
                            std::exp(kI * phase_arg) * result.P_r[p];
                    }
                }
            }

            std::array<Complex, 6> s_p_orient = {};
            Complex s_p_total = 0.0;

            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_plaquettes_per_orientation[alpha];
                    int N_b = result.n_plaquettes_per_orientation[beta];

                    if (N_a == 0 || N_b == 0) continue;

                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap +=
                            std::conj(P_q_psi[alpha][s]) * P_q_psi[beta][s];
                    }

                    Complex connected = overlap -
                        std::conj(P_q_expect[alpha]) * P_q_expect[beta];

                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_p_orient[idx] = connected / norm;

                    if (alpha == beta) {
                        s_p_total += connected;
                    } else {
                        s_p_total += 2.0 * connected;
                    }
                }
            }

            result.S_p_oriented[ik] = s_p_orient;
            result.S_p[ik] = s_p_total /
                             static_cast<double>(result.n_plaquettes);

            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_P(q): " << (ik + 1) << "/"
                          << n_k << " k-points..." << std::flush;
            }
        }
    }
    std::cout << " done" << std::endl;

    double max_val = 0.0;
    for (int ik = 0; ik < n_k; ++ik) {
        double val = std::abs(result.S_p[ik]);
        if (val > max_val) {
            max_val = val;
            result.q_max_idx = ik;
            result.s_p_max = result.S_p[ik];
            result.q_max = cluster.k_points[ik];
        }
    }
    result.m_plaquette = std::sqrt(max_val / result.n_plaquettes);

    if (memory_efficient_mode_enabled() && n_q_grid > 10) {
        std::cout << "  [Memory-efficient mode: skipping dense 2D plaquette grid]"
                  << std::endl;
        result.S_p_2d.resize(0);
    } else {
        result.S_p_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));

        std::cout << "  Computing S_P(q) on " << n_q_grid << "x" << n_q_grid
                  << " grid..." << std::flush;

        for (int i1 = 0; i1 < n_q_grid; ++i1) {
            for (int i2 = 0; i2 < n_q_grid; ++i2) {
                double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
                double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
                std::array<double, 2> qvec = {
                    q1 * cluster.b1[0] + q2 * cluster.b2[0],
                    q1 * cluster.b1[1] + q2 * cluster.b2[1]
                };

                auto P_q_psi = apply_bowtie_fourier(all_bowties, psi, qvec);
                Complex P_q_expect = 0.0;
                for (int p = 0; p < result.n_plaquettes; ++p) {
                    double phase_arg = qvec[0] * result.centers[p][0] +
                                       qvec[1] * result.centers[p][1];
                    P_q_expect += std::exp(kI * phase_arg) * result.P_r[p];
                }

                Complex overlap = 0.0;
                for (size_t s = 0; s < psi.size(); ++s) {
                    overlap += std::conj(P_q_psi[s]) * P_q_psi[s];
                }
                result.S_p_2d[i1][i2] =
                    (overlap - std::norm(P_q_expect)) /
                    static_cast<double>(result.n_plaquettes);
            }

            if ((i1 + 1) % 10 == 0 || i1 == n_q_grid - 1) {
                std::cout << "\r  Computing S_P(q) 2D grid: " << (i1 + 1)
                          << "/" << n_q_grid << " rows..." << std::flush;
            }
        }
    }

    result.chi_r.resize(result.n_triangles);

    std::cout << " done" << std::endl;
    std::cout << "  Computing triangle chiral expectations..." << std::flush;

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < result.n_triangles; ++idx) {
        const auto& tri = triangles[idx];
        result.chi_r[idx] = compute_triangle_chiral(psi, tri[0], tri[1], tri[2]);
    }

    double sum_chi = 0.0;
    for (int idx = 0; idx < result.n_triangles; ++idx) {
        sum_chi += result.chi_r[idx].real();
    }
    result.chi_mean = sum_chi / result.n_triangles;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << " done" << std::endl;
    std::cout << "  Plaquette order: m_plaquette = " << result.m_plaquette
              << " at q = (" << result.q_max[0] << ", " << result.q_max[1]
              << ")" << std::endl;
    std::cout << "  Mean resonance: <P> = " << result.P_mean
              << ", strength = " << result.resonance_strength << std::endl;
    std::cout << "  Triangle chiral: <chi> = " << result.chi_mean
              << " [" << duration.count() << " ms]" << std::endl;

    return result;
}

// -----------------------------------------------------------------------------
// 2D q-grid spin structure factor
// -----------------------------------------------------------------------------
Sq2DGridResult compute_sq_2d_grid(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const std::vector<std::vector<double>>& szsz_corr,
    const Cluster& cluster,
    int n_q_grid
) {
    int n_sites = cluster.n_sites;
    Sq2DGridResult result;
    result.s_q_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
    result.s_q_smsp_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
    result.s_q_szsz_2d.resize(n_q_grid, std::vector<double>(n_q_grid, 0.0));

    std::cout << "Computing S(q) on " << n_q_grid << "x" << n_q_grid
              << " grid (full Heisenberg)..." << std::flush;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < n_q_grid; ++i1) {
        for (int i2 = 0; i2 < n_q_grid; ++i2) {
            double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
            double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
            double qx = q1 * cluster.b1[0] + q2 * cluster.b2[0];
            double qy = q1 * cluster.b1[1] + q2 * cluster.b2[1];

            Complex s_q_smsp = 0.0;
            double s_q_szsz = 0.0;
            for (int i = 0; i < n_sites; ++i) {
                for (int j = 0; j < n_sites; ++j) {
                    auto dr = cluster.minimum_image_displacement(i, j);
                    double phase_arg = qx * dr[0] + qy * dr[1];
                    Complex phase = std::exp(kI * phase_arg);

                    s_q_smsp += smsp_corr[i][j] * phase;
                    s_q_szsz += szsz_corr[i][j] * std::real(phase);
                }
            }
            result.s_q_smsp_2d[i1][i2] = s_q_smsp / static_cast<double>(n_sites);
            result.s_q_szsz_2d[i1][i2] = s_q_szsz / static_cast<double>(n_sites);
            result.s_q_2d[i1][i2] = result.s_q_szsz_2d[i1][i2] +
                                    std::real(result.s_q_smsp_2d[i1][i2]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;

    return result;
}

// -----------------------------------------------------------------------------
// Compute all order parameters (scalar aggregator)
// -----------------------------------------------------------------------------
OrderParameterResults compute_all_order_parameters(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    double jpm_value
) {
    OrderParameterResults results;
    results.jpm = jpm_value;

    auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
    auto szsz_corr = compute_szsz_correlations(psi, cluster.n_sites);

    auto sf_result = compute_spin_structure_factor(smsp_corr, szsz_corr, cluster);
    results.m_translation = sf_result.m_translation;

    auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
    auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
    auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
    auto heisenberg_bond_exp =
        compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);

    auto nem_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
    auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
    auto nem_szsz_result =
        compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
    auto nem_heis_result =
        compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");

    results.m_nematic = nem_result.m_nem;
    results.anisotropy = nem_result.anisotropy;
    results.m_nematic_spsm = nem_spsm_result.m_nem;
    results.m_nematic_szsz = nem_szsz_result.m_nem;
    results.m_nematic_heisenberg = nem_heis_result.m_nem;

    auto vbs_result =
        compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster);
    results.m_vbs = vbs_result.m_vbs_xy;
    results.m_vbs_xy = vbs_result.m_vbs_xy;
    results.m_vbs_heis = vbs_result.m_vbs_heis;
    results.D_mean = vbs_result.D_mean_xy;
    results.D_mean_xy = vbs_result.D_mean_xy;
    results.D_mean_heis = vbs_result.D_mean_heis;

    auto plaq_result = compute_plaquette_order(psi, cluster);
    results.m_plaquette = plaq_result.m_plaquette;
    results.P_mean = plaq_result.P_mean;
    results.resonance_strength = plaq_result.resonance_strength;
    results.chi_mean = plaq_result.chi_mean;
    results.n_plaquettes = plaq_result.n_plaquettes;
    results.n_triangles = plaq_result.n_triangles;

    return results;
}

}  // namespace ed::bfg
