/**
 * @file compute_bfg_order_parameters.cpp
 * @brief Fast C++ computation of BFG order parameters from wavefunctions
 * 
 * Computes:
 * 1. S(q) - Spin structure factor using S^-S^+ correlations at ALL k-points + 2D grid
 * 2. Nematic order - Bond orientation anisotropy (C3 → C1 breaking)
 *    - Variants: XY (S+S- + S-S+), S+S-, SzSz, Heisenberg (S·S)
 * 3. VBS (Valence Bond Solid) order - S_D(q) with proper 4-site correlations
 *    - Computes actual ⟨D_b D_b'⟩ dimer-dimer correlations
 * 
 * Output includes:
 * - Order parameters at special k-points (Γ, K, M, etc.)
 * - Full 2D structure factor grids for visualization
 * - Per-bond expectations for spatial visualization
 * - Detailed HDF5 output compatible with Python plotting
 * 
 * Usage:
 *   ./compute_bfg_order_parameters <wavefunction.h5> <cluster_dir> [output.h5]
 * 
 * Compile with:
 *   g++ -O3 -march=native -fopenmp -std=c++17 compute_bfg_order_parameters.cpp -o compute_bfg_order_parameters -lhdf5 -lhdf5_cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <limits>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <regex>
#include <mutex>
#include <atomic>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <H5Cpp.h>

#include "ed/bfg/cluster.h"
#include "ed/bfg/correlations.h"
#include "ed/bfg/results_io.h"
#include "ed/bfg/ring_observables.h"
#include "ed/bfg/spin_structure_factor.h"
#include "ed/bfg/structure_factor.h"
#include "ed/bfg/topology.h"
#include "ed/bfg/wavefunction_io.h"

namespace fs = std::filesystem;

using Complex = std::complex<double>;
const double PI = 3.14159265358979323846;
const Complex I(0.0, 1.0);

// P2.1: Cluster, load_cluster, find_triangles, find_bowties, Bowtie, and the
// six two-body correlation / bond-expectation kernels live in the ed_bfg
// static library so the GPU driver, the future Python bindings, and the
// in-progress library split below the main() driver all share the same
// authoritative implementation. Pull the names into the local TU so the
// existing call sites keep working without an explicit `ed::bfg::` prefix.
using ed::bfg::Bowtie;
using ed::bfg::Cluster;
using ed::bfg::compute_heisenberg_bond_expectations;
using ed::bfg::compute_smsp_correlations;
using ed::bfg::compute_spsm_bond_expectations;
using ed::bfg::compute_szsz_bond_expectations;
using ed::bfg::compute_szsz_correlations;
using ed::bfg::compute_xy_bond_expectations;
using ed::bfg::find_bowties;
using ed::bfg::find_triangles;
using ed::bfg::load_all_tpq_states;
using ed::bfg::load_cluster;
using ed::bfg::load_tpq_state;
using ed::bfg::load_wavefunction;
using ed::bfg::TPQState;
// P2.1 (4th slice): bond-bilinear structure factors / Fourier-applied dimer
// kernels also live in ed_bfg now.
using ed::bfg::apply_dimer_fourier;
using ed::bfg::apply_heisenberg_dimer_fourier;
using ed::bfg::compute_dimer_dimer_correlation;
using ed::bfg::compute_dimer_sf_direct;
using ed::bfg::compute_heisenberg_dimer_dimer_correlation;
using ed::bfg::compute_heisenberg_sf_direct;
using ed::bfg::DimerSFResult;
using ed::bfg::memory_efficient_mode_enabled;
using ed::bfg::set_memory_efficient_mode;
// P2.1 (5th slice): bowtie ring-flip + triangle ring-exchange kernels live
// in ed_bfg::ring_observables. The Bowtie POD that was passed to
// `apply_bowtie_fourier` from this TU is now the canonical
// `ed::bfg::Bowtie` (already used by find_bowties) -- the file-local
// `BowtieData` has been retired.
using ed::bfg::apply_bowtie_fourier;
using ed::bfg::compute_bowtie_resonance;
using ed::bfg::compute_triangle_chiral;
// P2.1 (6th slice): spin structure factor S(q) over precomputed
// two-body correlations now lives in ed_bfg::spin_structure_factor.
using ed::bfg::compute_spin_structure_factor;
using ed::bfg::StructureFactorResult;
// P2.1 (7th slice): the per-cluster scalar-summary structs and the HDF5
// results writers (`save_results`, `save_temperature_scan_results`,
// `save_scan_results`) moved into ed_bfg::results_io. The struct
// definitions used to live in this TU; the kernels (`compute_nematic_order`,
// `compute_vbs_order`, `compute_plaquette_order`, `compute_sq_2d_grid`,
// `compute_all_order_parameters`) still build them in-place but now consume
// the canonical library-side type definitions.
using ed::bfg::NematicResult;
using ed::bfg::OrderParameterResults;
using ed::bfg::PlaquetteResult;
using ed::bfg::save_results;
using ed::bfg::save_scan_results;
using ed::bfg::save_temperature_scan_results;
using ed::bfg::Sq2DGridResult;
using ed::bfg::VBSResult;

// -----------------------------------------------------------------------------
// Bit manipulation helpers (inlined for speed)
// -----------------------------------------------------------------------------

inline int get_bit(uint64_t state, int site) {
    return (state >> site) & 1;
}

inline uint64_t set_bit(uint64_t state, int site, int value) {
    if (value) {
        return state | (1ULL << site);
    } else {
        return state & ~(1ULL << site);
    }
}

inline uint64_t flip_bit(uint64_t state, int site) {
    return state ^ (1ULL << site);
}

// -----------------------------------------------------------------------------
// Spin operators on basis states
// Returns: (new_state, coefficient) or (-1, 0) if annihilated
// -----------------------------------------------------------------------------

// S^+ raises spin: |↓⟩ → |↑⟩, |↑⟩ → 0
// ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
inline std::pair<int64_t, double> apply_sp(uint64_t state, int site) {
    if (get_bit(state, site) == 1) {  // spin down (bit=1 in ED convention)
        return {static_cast<int64_t>(state & ~(1ULL << site)), 1.0};  // flip to 0 (up)
    }
    return {-1, 0.0};
}

// S^- lowers spin: |↑⟩ → |↓⟩, |↓⟩ → 0
// ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
inline std::pair<int64_t, double> apply_sm(uint64_t state, int site) {
    if (get_bit(state, site) == 0) {  // spin up (bit=0 in ED convention)
        return {static_cast<int64_t>(state | (1ULL << site)), 1.0};  // flip to 1 (down)
    }
    return {-1, 0.0};
}

// S^z eigenvalue: |↑⟩ → +1/2, |↓⟩ → -1/2
// ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
inline double sz_value(uint64_t state, int site) {
    return get_bit(state, site) ? -0.5 : 0.5;
}

// -----------------------------------------------------------------------------
// `Cluster` and `load_cluster` moved into the new `ed_bfg` static library
// (see include/ed/bfg/cluster.h, src/bfg/cluster.cpp). Pulled in here via a
// using-declaration in the file header so the rest of this TU keeps using
// the unqualified names. P2.1.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// P2.1 (third slice): the wavefunction / TPQ-state HDF5 loaders moved to
// `include/ed/bfg/wavefunction_io.h` + `src/bfg/wavefunction_io.cpp` so the
// CPU driver, the GPU driver (`compute_bfg_order_parameters_gpu.cu`), and
// the future Python bindings share one authoritative implementation. Pulled
// in via the using-declarations at the top of this TU so the existing call
// sites continue to work without an explicit `ed::bfg::` prefix.
// -----------------------------------------------------------------------------

// P2.1: compute_smsp_correlations / compute_szsz_correlations / the four
// compute_*_bond_expectations kernels moved into the ed_bfg static library
// (`include/ed/bfg/correlations.h`) so the GPU driver and Python bindings
// share the same authoritative implementation. The console progress prints
// the file-local versions used to emit are intentionally dropped -- the
// library version stays quiet, and process_all_temperatures / single-file
// mode below already log start/stop separately.

// -----------------------------------------------------------------------------
// P2.1 (4th slice): the bond-bilinear structure factor / Fourier-applied
// dimer kernels (DimerSFResult, compute_*_sf_direct, apply_*_fourier,
// compute_*_dimer_dimer_correlation, set_memory_efficient_mode) moved into
// the ed_bfg static library (`include/ed/bfg/structure_factor.h`,
// `src/bfg/structure_factor.cpp`). The using-declarations at the top of
// this TU keep the existing call sites unchanged.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// P2.1 (5th slice): apply_bowtie_fourier moved to ed_bfg::ring_observables.
// The file-local `BowtieData {s1,s2,s3,s4,center}` POD was a strict subset
// of `ed::bfg::Bowtie` (which carries `s0` and `orientation` on top of the
// same five fields), so the kernel now consumes `Bowtie` directly and the
// caller-side rebuild loop drops away (see the `find_bowties` consumer
// below).
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// P2.1 (4th slice): the real-space dimer-dimer correlations
// (compute_dimer_dimer_correlation, compute_heisenberg_dimer_dimer_correlation)
// also moved into ed_bfg::structure_factor; using-declarations at the top of
// this TU pull them in unchanged.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// P2.1 (6th slice): the spin structure factor S(q) over precomputed
// two-body correlation tables (StructureFactorResult,
// compute_spin_structure_factor) moved into the ed_bfg static library
// (`include/ed/bfg/spin_structure_factor.h`,
// `src/bfg/spin_structure_factor.cpp`). The using-declarations at the top
// of this TU keep the existing call sites unchanged.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Compute nematic order
// -----------------------------------------------------------------------------

// `NematicResult` now lives in `ed/bfg/results_io.h`. Pulled in via the
// using-declaration at the top of this TU; see P2.1 (7th slice).

// Generic nematic order from complex bond expectations
NematicResult compute_nematic_order(
    const std::map<std::pair<int, int>, Complex>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type = "xy"
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
            result.O_bar[alpha] = sum_by_orient[alpha] / static_cast<double>(count_by_orient[alpha]);
        }
    }
    
    // ψ_nem = Σ ω^α O̅_α, where ω = exp(2πi/3)
    Complex omega = std::exp(2.0 * PI * I / 3.0);
    result.psi_nem = result.O_bar[0] + omega * result.O_bar[1] + omega * omega * result.O_bar[2];
    result.m_nem = std::abs(result.psi_nem);
    
    // Anisotropy
    std::array<double, 3> mags = {std::abs(result.O_bar[0]), std::abs(result.O_bar[1]), std::abs(result.O_bar[2])};
    double max_mag = *std::max_element(mags.begin(), mags.end());
    double min_mag = *std::min_element(mags.begin(), mags.end());
    result.anisotropy = (max_mag > 1e-10) ? (max_mag - min_mag) / max_mag : 0.0;
    
    std::cout << "Nematic order (" << bond_type << "): m_nem = " << result.m_nem 
              << ", anisotropy = " << result.anisotropy << std::endl;
    
    return result;
}

// Nematic order from real-valued bond expectations (SzSz, Heisenberg)
NematicResult compute_nematic_order_real(
    const std::map<std::pair<int, int>, double>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type = "szsz"
) {
    // Convert to complex map and use the generic function
    std::map<std::pair<int, int>, Complex> bond_exp_complex;
    for (const auto& [edge, val] : bond_exp) {
        bond_exp_complex[edge] = Complex(val, 0.0);
    }
    return compute_nematic_order(bond_exp_complex, cluster, bond_type);
}

// -----------------------------------------------------------------------------
// Compute VBS (Valence Bond Solid) order with PROPER 4-site correlations
// S_D(q) = (1/N_b) Σ_{b,b'} exp(iq·(r_b - r_{b'})) ⟨δD_b δD_{b'}⟩_connected
// where D_b = S^+_i S^-_j + S^-_i S^+_j (XY dimer operator)
//    or D_b = S_i · S_j = SzSz + (1/2)(S+S- + S-S+) (Heisenberg dimer)
// 
// The connected correlator is: ⟨D_b D_{b'}⟩ - ⟨D_b⟩⟨D_{b'}⟩
// This requires computing ACTUAL 4-site spin correlations!
// -----------------------------------------------------------------------------

// `VBSResult` now lives in `ed/bfg/results_io.h` (the backward-compatible
// accessors used to live on the struct itself; the historical CPU driver
// only ever read `m_vbs_xy` / `D_mean_xy` directly, so the accessors were
// dead code and have been dropped from the library type). Pulled in via
// the using-declaration at the top of this TU; see P2.1 (7th slice).

VBSResult compute_vbs_order(
    const std::vector<Complex>& psi,
    const std::map<std::pair<int, int>, Complex>& xy_bond_exp,
    const std::map<std::pair<int, int>, double>& heisenberg_bond_exp,
    const Cluster& cluster,
    int n_q_grid = 50
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
    
    std::cout << "Computing VBS order using efficient Fourier-space method..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Convert edges to vector for indexed access
    std::vector<std::pair<int, int>> edges(cluster.edges_nn.begin(), cluster.edges_nn.end());
    
    // Group bonds by orientation and compute bond centers
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
    
    std::cout << "  Bonds per orientation: " << result.n_bonds_per_orientation[0] << ", "
              << result.n_bonds_per_orientation[1] << ", "
              << result.n_bonds_per_orientation[2] << std::endl;
    
    // =========================================================================
    // Mean bond values ⟨D_α⟩ for each orientation
    // =========================================================================
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
            D_mean_xy_orient[alpha] = sum_a / static_cast<double>(edges_by_orient[alpha].size());
            D_mean_heis_orient[alpha] = sum_h / edges_by_orient[alpha].size();
        }
    }
    
    // =========================================================================
    // Efficient Fourier-space computation of S_D^{αβ}(q):
    // D_α(q)|ψ⟩ = Σ_{b∈α} exp(iq·r_b) D_b|ψ⟩
    // S_D^{αβ}(q) = (1/√(N_α N_β)) [⟨D_α(q)ψ|D_β(q)ψ⟩ - ⟨D_α(q)⟩*⟨D_β(q)⟩]
    //
    // This is O(3 * n_k * N_bonds * Hilbert) instead of O(N_bonds² * Hilbert)
    // =========================================================================
    
    result.S_d_xy.resize(n_k, 0.0);
    result.S_d_heis.resize(n_k, 0.0);
    result.S_d_xy_oriented.resize(n_k);
    result.S_d_heis_oriented.resize(n_k);
    
    // Map from (α, β) pair to index in the 6-element array
    auto orient_pair_idx = [](int alpha, int beta) -> int {
        if (alpha > beta) std::swap(alpha, beta);
        if (alpha == 0) return beta;
        if (alpha == 1) return 2 + beta;
        return 5;
    };
    
    std::cout << "  Computing S_D(q) at " << n_k << " k-points using Fourier method..." << std::flush;
    
    if (memory_efficient_mode_enabled()) {
        // Memory-efficient mode: use direct structure factor computation
        // This computes ⟨D†(q)D(q)⟩ without storing O(Hilbert) intermediate vectors
        std::cout << "\n  [Memory-efficient mode: using direct SF computation]" << std::endl;
        
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            
            // Compute XY dimer structure factor directly
            auto sf_xy = compute_dimer_sf_direct(psi, edges, all_bond_centers, q);
            Complex s_d_total_xy = sf_xy.overlap - std::norm(sf_xy.expect_q1);
            result.S_d_xy[ik] = s_d_total_xy / static_cast<double>(n_bonds);
            
            // Compute Heisenberg dimer structure factor directly
            auto sf_heis = compute_heisenberg_sf_direct(psi, edges, all_bond_centers, q);
            Complex s_d_total_heis = sf_heis.overlap - std::norm(sf_heis.expect_q1);
            result.S_d_heis[ik] = s_d_total_heis.real() / n_bonds;
            
            // For orientation-resolved, compute each pair separately
            std::array<Complex, 6> s_d_xy_orient = {};
            std::array<double, 6> s_d_heis_orient = {};
            
            // Skip detailed orientation-resolved computation in memory-efficient mode
            // (would require O(9 * n_bonds² * Hilbert) which is expensive)
            result.S_d_xy_oriented[ik] = s_d_xy_orient;
            result.S_d_heis_oriented[ik] = s_d_heis_orient;
            
            if ((ik + 1) % 5 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_D(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    } else {
        // Original mode: use Fourier-transformed operators (faster but more memory)
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            std::array<double, 2> neg_q = {-q[0], -q[1]};
            
            // Apply D_α(q)|ψ⟩ for each orientation (XY dimer)
            std::array<std::vector<Complex>, 3> D_q_psi_xy;
            std::array<Complex, 3> D_q_expect_xy = {};
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                if (edges_by_orient[alpha].empty()) {
                    D_q_psi_xy[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                D_q_psi_xy[alpha] = apply_dimer_fourier(psi, edges_by_orient[alpha], 
                                                        centers_by_orient[alpha], q);
                // ⟨D_α(q)⟩ = Σ_b exp(iq·r_b) ⟨D_b⟩
                for (size_t b = 0; b < edges_by_orient[alpha].size(); ++b) {
                    double phase_arg = q[0] * centers_by_orient[alpha][b][0] + 
                                       q[1] * centers_by_orient[alpha][b][1];
                    D_q_expect_xy[alpha] += std::exp(I * phase_arg) * 
                                            xy_bond_exp.at(edges_by_orient[alpha][b]);
                }
            }
            
            // Compute S_D^{αβ}(q) for all orientation pairs
            std::array<Complex, 6> s_d_xy_orient = {};
            Complex s_d_total = 0.0;
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_bonds_per_orientation[alpha];
                    int N_b = result.n_bonds_per_orientation[beta];
                    
                    if (N_a == 0 || N_b == 0) continue;
                    
                    // ⟨D_α(q)ψ|D_β(q)ψ⟩
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(D_q_psi_xy[alpha][s]) * D_q_psi_xy[beta][s];
                    }
                    
                    // Connected: subtract ⟨D_α(q)⟩*⟨D_β(q)⟩
                    Complex connected = overlap - std::conj(D_q_expect_xy[alpha]) * D_q_expect_xy[beta];
                    
                    // Normalize by sqrt(N_α * N_β)
                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_d_xy_orient[idx] = connected / norm;
                    
                    // Contribution to total (accounting for symmetry α↔β)
                    if (alpha == beta) {
                        s_d_total += connected;
                    } else {
                        s_d_total += 2.0 * connected;  // α,β and β,α
                    }
                }
            }
            
            result.S_d_xy_oriented[ik] = s_d_xy_orient;
            result.S_d_xy[ik] = s_d_total / static_cast<double>(n_bonds);
            
            // For Heisenberg, use the Heisenberg dimer Fourier transform
            auto [D_q_psi_heis_all, D_q_expect_heis_all] = apply_heisenberg_dimer_fourier(
                psi, edges, all_bond_centers, q);
            
            // For orientation-resolved Heisenberg, we need separate applications
            std::array<double, 6> s_d_heis_orient = {};
            double s_d_heis_total = 0.0;
            
            // Compute per-orientation Heisenberg structure factors
            std::array<std::vector<Complex>, 3> D_q_psi_heis;
            std::array<Complex, 3> D_q_expect_heis = {};
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                if (edges_by_orient[alpha].empty()) {
                    D_q_psi_heis[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                auto [dpsi, dexp] = apply_heisenberg_dimer_fourier(
                    psi, edges_by_orient[alpha], centers_by_orient[alpha], q);
                D_q_psi_heis[alpha] = std::move(dpsi);
                D_q_expect_heis[alpha] = dexp;
            }
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_bonds_per_orientation[alpha];
                    int N_b = result.n_bonds_per_orientation[beta];
                    
                    if (N_a == 0 || N_b == 0) continue;
                    
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(D_q_psi_heis[alpha][s]) * D_q_psi_heis[beta][s];
                    }
                    
                    Complex connected = overlap - std::conj(D_q_expect_heis[alpha]) * D_q_expect_heis[beta];
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
                std::cout << "\r  Computing S_D(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    }
    std::cout << " done" << std::endl;
    
    // Find maxima
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
    
    // =========================================================================
    // Also compute on dense 2D grid for visualization (total S_D only)
    // Skip in memory-efficient mode for very large systems
    // =========================================================================
    if (memory_efficient_mode_enabled() && n_q_grid > 10) {
        std::cout << "  [Memory-efficient mode: skipping dense 2D VBS grid (use --n-q-grid 10 to enable)]" << std::endl;
        // Just initialize empty grids
        result.S_d_xy_2d.resize(0);
        result.S_d_heis_2d.resize(0);
    } else {
        result.S_d_xy_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
        result.S_d_heis_2d.resize(n_q_grid, std::vector<double>(n_q_grid, 0.0));
        
        std::cout << "  Computing S_D(q) on " << n_q_grid << "x" << n_q_grid << " grid..." << std::flush;
        
        // For 2D grid, use total bonds (not orientation-resolved for speed)
        for (int i1 = 0; i1 < n_q_grid; ++i1) {
            for (int i2 = 0; i2 < n_q_grid; ++i2) {
                double q1 = -1.0 + 2.0 * i1 / (n_q_grid - 1);
                double q2 = -1.0 + 2.0 * i2 / (n_q_grid - 1);
                std::array<double, 2> qvec = {
                    q1 * cluster.b1[0] + q2 * cluster.b2[0],
                    q1 * cluster.b1[1] + q2 * cluster.b2[1]
                };
                
                if (memory_efficient_mode_enabled()) {
                    // Use direct structure factor computation
                    auto sf_xy = compute_dimer_sf_direct(psi, edges, all_bond_centers, qvec);
                    result.S_d_xy_2d[i1][i2] = (sf_xy.overlap - std::norm(sf_xy.expect_q1)) / static_cast<double>(n_bonds);
                    
                    auto sf_heis = compute_heisenberg_sf_direct(psi, edges, all_bond_centers, qvec);
                    result.S_d_heis_2d[i1][i2] = (sf_heis.overlap - std::norm(sf_heis.expect_q1)).real() / n_bonds;
                } else {
                    // XY
                    auto D_q_psi = apply_dimer_fourier(psi, edges, all_bond_centers, qvec);
                    Complex D_q_expect = 0.0;
                    for (int b = 0; b < n_bonds; ++b) {
                        double phase_arg = qvec[0] * all_bond_centers[b][0] + qvec[1] * all_bond_centers[b][1];
                        D_q_expect += std::exp(I * phase_arg) * xy_bond_exp.at(edges[b]);
                    }
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(D_q_psi[s]) * D_q_psi[s];
                    }
                    result.S_d_xy_2d[i1][i2] = (overlap - std::norm(D_q_expect)) / static_cast<double>(n_bonds);
                    
                    // Heisenberg
                    auto [D_q_psi_h, D_q_expect_h] = apply_heisenberg_dimer_fourier(psi, edges, all_bond_centers, qvec);
                    Complex overlap_h = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap_h += std::conj(D_q_psi_h[s]) * D_q_psi_h[s];
                    }
                    result.S_d_heis_2d[i1][i2] = (overlap_h - std::norm(D_q_expect_h)).real() / n_bonds;
                }
            }
            
            if ((i1 + 1) % 10 == 0 || i1 == n_q_grid - 1) {
                std::cout << "\r  Computing S_D(q) 2D grid: " << (i1 + 1) << "/" << n_q_grid << " rows..." << std::flush;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done" << std::endl;
    std::cout << "  VBS order (XY):         m_vbs = " << result.m_vbs_xy 
              << " at q = (" << result.q_max_xy[0] << ", " << result.q_max_xy[1] << ")" << std::endl;
    std::cout << "  VBS order (Heisenberg): m_vbs = " << result.m_vbs_heis
              << " at q = (" << result.q_max_heis[0] << ", " << result.q_max_heis[1] << ")"
              << " [" << duration.count() << " ms]" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Find all NN triangles in the kagome lattice
// A triangle is 3 sites where all pairs are NN connected.
// Returns list of (s1, s2, s3) tuples with s1 < s2 < s3.
// -----------------------------------------------------------------------------

// P2.1: find_triangles, find_bowties, and the Bowtie struct moved into the
// ed_bfg static library (`include/ed/bfg/topology.h`); the local
// `using ed::bfg::*` aliases at the top of this file keep call sites
// unchanged.

// -----------------------------------------------------------------------------
// P2.1 (5th slice): compute_bowtie_resonance and compute_triangle_chiral
// moved to ed_bfg::ring_observables. Pulled in via the using-declarations at
// the top of this TU so call sites stay unchanged.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Plaquette order parameter result structure
// -----------------------------------------------------------------------------

// `PlaquetteResult` now lives in `ed/bfg/results_io.h`. Pulled in via the
// using-declaration at the top of this TU; see P2.1 (7th slice).

// -----------------------------------------------------------------------------
// Compute plaquette/bowtie resonance order
// S_P(q) = (1/N_bt) Σ_{bt,bt'} exp(iq·(R_bt - R_{bt'})) ⟨δP_bt δP_{bt'}⟩
// where δP_bt = P_bt - ⟨P⟩
// -----------------------------------------------------------------------------

PlaquetteResult compute_plaquette_order(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    int n_q_grid = 50
) {
    PlaquetteResult result;
    int n_k = cluster.k_points.size();
    
    std::cout << "Computing plaquette/bowtie resonance order using efficient Fourier method..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Find all triangles (for chiral order)
    auto triangles = find_triangles(cluster);
    result.n_triangles = triangles.size();
    std::cout << "  Found " << result.n_triangles << " triangular plaquettes" << std::endl;
    
    // Find all bowties (for resonance order)
    auto bowties = find_bowties(cluster);
    result.n_plaquettes = bowties.size();
    std::cout << "  Found " << result.n_plaquettes << " bowtie plaquettes (5-site, 2 triangles sharing corner)" << std::endl;
    
    if (result.n_plaquettes == 0) {
        result.m_plaquette = 0.0;
        result.P_mean = 0.0;
        result.resonance_strength = 0.0;
        return result;
    }
    
    // P2.1 (5th slice): apply_bowtie_fourier now takes `ed::bfg::Bowtie`
    // directly (it only reads s1..s4 + center; s0/orientation are ignored),
    // so the file-local BowtieData rebuild from previous revisions is gone.
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
    
    std::cout << "  Bowties per orientation: " << result.n_plaquettes_per_orientation[0] << ", "
              << result.n_plaquettes_per_orientation[1] << ", "
              << result.n_plaquettes_per_orientation[2] << std::endl;
    
    // Compute single-plaquette expectations ⟨P_bt⟩ for each bowtie (needed for means)
    std::cout << "  Computing individual bowtie expectations..." << std::flush;
    
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < result.n_plaquettes; ++idx) {
        const auto& bt = bowties[idx];
        result.P_r[idx] = compute_bowtie_resonance(psi, bt.s1, bt.s2, bt.s3, bt.s4);
    }
    std::cout << " done" << std::endl;
    
    // Mean plaquette value and resonance strength per orientation
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
            P_mean_orient[alpha] /= static_cast<double>(result.n_plaquettes_per_orientation[alpha]);
        }
    }
    
    // =========================================================================
    // Efficient Fourier-space computation of S_P^{αβ}(q):
    // P_α(q)|ψ⟩ = Σ_{p∈α} exp(iq·r_p) P_p|ψ⟩
    // S_P^{αβ}(q) = (1/√(N_α N_β)) [⟨P_α(q)ψ|P_β(q)ψ⟩ - ⟨P_α(q)⟩*⟨P_β(q)⟩]
    // =========================================================================
    
    result.S_p.resize(n_k, 0.0);
    result.S_p_oriented.resize(n_k);
    
    auto orient_pair_idx = [](int alpha, int beta) -> int {
        if (alpha > beta) std::swap(alpha, beta);
        if (alpha == 0) return beta;
        if (alpha == 1) return 2 + beta;
        return 5;
    };
    
    std::cout << "  Computing S_P(q) at " << n_k << " k-points using Fourier method..." << std::flush;
    
    if (memory_efficient_mode_enabled()) {
        // Memory-efficient mode: compute structure factor without storing full O(Hilbert) vectors
        // For plaquette/bowtie, this is more complex - we skip detailed orientation resolution
        std::cout << "\n  [Memory-efficient mode: using simplified plaquette SF computation]" << std::endl;
        
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            
            // For memory efficiency, compute ⟨P†(q)P(q)⟩ directly
            // P(q)|ψ⟩ = Σ_p exp(iq·r_p) P_p|ψ⟩
            // We need ⟨ψ|P†(q)P(q)|ψ⟩ = Σ_{p,p'} exp(i q·(r_{p'} - r_p)) ⟨ψ|P_p P_{p'}|ψ⟩
            // For large systems, approximate by using individual expectations
            
            Complex P_q_expect = 0.0;
            Complex P_q_expect_sq = 0.0;  // For variance approximation
            
            for (int p = 0; p < result.n_plaquettes; ++p) {
                double phase_arg = q[0] * result.centers[p][0] + q[1] * result.centers[p][1];
                Complex phase = std::exp(I * phase_arg);
                P_q_expect += phase * result.P_r[p];
                P_q_expect_sq += std::norm(result.P_r[p]);  // |⟨P_p⟩|²
            }
            
            // Approximate S_P(q) using ⟨P⟩ values (underestimates true structure factor)
            // True value would require ⟨P_p P_{p'}⟩ four-point functions
            result.S_p[ik] = (P_q_expect_sq - std::norm(P_q_expect) / result.n_plaquettes) / 
                            static_cast<double>(result.n_plaquettes);
            
            result.S_p_oriented[ik] = {};  // Skip orientation-resolved in memory mode
            
            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_P(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    } else {
        // Original mode with full vector storage
        for (int ik = 0; ik < n_k; ++ik) {
            const auto& q = cluster.k_points[ik];
            
            // Apply P_α(q)|ψ⟩ for each orientation
            std::array<std::vector<Complex>, 3> P_q_psi;
            std::array<Complex, 3> P_q_expect = {};
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                if (bowties_by_orient[alpha].empty()) {
                    P_q_psi[alpha].resize(psi.size(), 0.0);
                    continue;
                }
                P_q_psi[alpha] = apply_bowtie_fourier(bowties_by_orient[alpha], psi, q);
                
                // ⟨P_α(q)⟩ = Σ_p exp(iq·r_p) ⟨P_p⟩
                int local_idx = 0;
                for (int p = 0; p < result.n_plaquettes; ++p) {
                    if (result.orientations[p] == alpha) {
                        double phase_arg = q[0] * result.centers[p][0] + q[1] * result.centers[p][1];
                        P_q_expect[alpha] += std::exp(I * phase_arg) * result.P_r[p];
                    }
                }
            }
            
            // Compute S_P^{αβ}(q) for all orientation pairs
            std::array<Complex, 6> s_p_orient = {};
            Complex s_p_total = 0.0;
            
            for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = alpha; beta < 3; ++beta) {
                    int idx = orient_pair_idx(alpha, beta);
                    int N_a = result.n_plaquettes_per_orientation[alpha];
                    int N_b = result.n_plaquettes_per_orientation[beta];
                    
                    if (N_a == 0 || N_b == 0) continue;
                    
                    // ⟨P_α(q)ψ|P_β(q)ψ⟩
                    Complex overlap = 0.0;
                    for (size_t s = 0; s < psi.size(); ++s) {
                        overlap += std::conj(P_q_psi[alpha][s]) * P_q_psi[beta][s];
                    }
                    
                    // Connected: subtract ⟨P_α(q)⟩*⟨P_β(q)⟩
                    Complex connected = overlap - std::conj(P_q_expect[alpha]) * P_q_expect[beta];
                    
                    // Normalize by sqrt(N_α * N_β)
                    double norm = std::sqrt(static_cast<double>(N_a) * N_b);
                    s_p_orient[idx] = connected / norm;
                    
                    // Contribution to total
                    if (alpha == beta) {
                        s_p_total += connected;
                    } else {
                        s_p_total += 2.0 * connected;
                    }
                }
            }
            
            result.S_p_oriented[ik] = s_p_orient;
            result.S_p[ik] = s_p_total / static_cast<double>(result.n_plaquettes);
            
            if ((ik + 1) % 10 == 0 || ik == n_k - 1) {
                std::cout << "\r  Computing S_P(q): " << (ik + 1) << "/" << n_k << " k-points..." << std::flush;
            }
        }
    }
    std::cout << " done" << std::endl;
    
    // Find maximum
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
    
    // =========================================================================
    // Also compute on dense 2D grid for visualization
    // Skip in memory-efficient mode for large systems
    // =========================================================================
    if (memory_efficient_mode_enabled() && n_q_grid > 10) {
        std::cout << "  [Memory-efficient mode: skipping dense 2D plaquette grid]" << std::endl;
        result.S_p_2d.resize(0);
    } else {
        result.S_p_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
        
        std::cout << "  Computing S_P(q) on " << n_q_grid << "x" << n_q_grid << " grid..." << std::flush;
        
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
                    double phase_arg = qvec[0] * result.centers[p][0] + qvec[1] * result.centers[p][1];
                    P_q_expect += std::exp(I * phase_arg) * result.P_r[p];
                }
                
                Complex overlap = 0.0;
                for (size_t s = 0; s < psi.size(); ++s) {
                    overlap += std::conj(P_q_psi[s]) * P_q_psi[s];
                }
                result.S_p_2d[i1][i2] = (overlap - std::norm(P_q_expect)) / static_cast<double>(result.n_plaquettes);
            }
            
            if ((i1 + 1) % 10 == 0 || i1 == n_q_grid - 1) {
                std::cout << "\r  Computing S_P(q) 2D grid: " << (i1 + 1) << "/" << n_q_grid << " rows..." << std::flush;
            }
        }
    }
    
    // =========================================================================
    // Compute triangle chiral order
    // =========================================================================
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
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << " done" << std::endl;
    std::cout << "  Plaquette order: m_plaquette = " << result.m_plaquette 
              << " at q = (" << result.q_max[0] << ", " << result.q_max[1] << ")" << std::endl;
    std::cout << "  Mean resonance: <P> = " << result.P_mean 
              << ", strength = " << result.resonance_strength << std::endl;
    std::cout << "  Triangle chiral: <χ> = " << result.chi_mean
              << " [" << duration.count() << " ms]" << std::endl;
    
    return result;
}

// -----------------------------------------------------------------------------
// Compute 2D S(q) grid for visualization
// Returns: {full S(q), S-S+ component, SzSz component}
// Full S(q) = SzSz + real(S-S+) for Heisenberg
// -----------------------------------------------------------------------------

// `Sq2DGridResult` now lives in `ed/bfg/results_io.h`. Pulled in via the
// using-declaration at the top of this TU; see P2.1 (7th slice).
Sq2DGridResult compute_sq_2d_grid(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const std::vector<std::vector<double>>& szsz_corr,
    const Cluster& cluster,
    int n_q_grid = 50
) {
    int n_sites = cluster.n_sites;
    Sq2DGridResult result;
    result.s_q_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
    result.s_q_smsp_2d.resize(n_q_grid, std::vector<Complex>(n_q_grid, 0.0));
    result.s_q_szsz_2d.resize(n_q_grid, std::vector<double>(n_q_grid, 0.0));
    
    std::cout << "Computing S(q) on " << n_q_grid << "x" << n_q_grid << " grid (full Heisenberg)..." << std::flush;
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
                    // Use minimum-image displacement for PBC-correct phases
                    auto dr = cluster.minimum_image_displacement(i, j);
                    double phase_arg = qx * dr[0] + qy * dr[1];
                    Complex phase = std::exp(I * phase_arg);
                    
                    // S^-S^+ contribution
                    s_q_smsp += smsp_corr[i][j] * phase;
                    
                    // S^zS^z contribution
                    s_q_szsz += szsz_corr[i][j] * std::real(phase);
                }
            }
            result.s_q_smsp_2d[i1][i2] = s_q_smsp / static_cast<double>(n_sites);
            result.s_q_szsz_2d[i1][i2] = s_q_szsz / static_cast<double>(n_sites);
            // Full S(q) = SzSz + real(S-S+)
            result.s_q_2d[i1][i2] = result.s_q_szsz_2d[i1][i2] + std::real(result.s_q_smsp_2d[i1][i2]);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " done (" << duration.count() << " ms)" << std::endl;
    
    return result;
}

// `save_results` (full per-cluster bundle) now lives in `ed_bfg::results_io`.
// Pulled in via the using-declaration at the top of this TU. P2.1 (7th slice).

// `OrderParameterResults` now lives in `ed/bfg/results_io.h`. Pulled in via
// the using-declaration at the top of this TU. P2.1 (7th slice).

// -----------------------------------------------------------------------------
// Compute all order parameters (for scan mode)
// -----------------------------------------------------------------------------

OrderParameterResults compute_all_order_parameters(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    double jpm_value
) {
    OrderParameterResults results;
    results.jpm = jpm_value;
    
    // Compute correlations
    auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
    auto szsz_corr = compute_szsz_correlations(psi, cluster.n_sites);

    // Structure factor (full Heisenberg)
    auto sf_result = compute_spin_structure_factor(smsp_corr, szsz_corr, cluster);
    results.m_translation = sf_result.m_translation;
    
    // Bond expectations
    auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
    auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
    auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
    auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
    
    // Nematic (all variants)
    auto nem_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
    auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
    auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
    auto nem_heis_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
    
    results.m_nematic = nem_result.m_nem;
    results.anisotropy = nem_result.anisotropy;
    results.m_nematic_spsm = nem_spsm_result.m_nem;
    results.m_nematic_szsz = nem_szsz_result.m_nem;
    results.m_nematic_heisenberg = nem_heis_result.m_nem;
    
    // VBS order (proper 4-site correlations, both XY and Heisenberg)
    auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster);
    results.m_vbs = vbs_result.m_vbs_xy;
    results.m_vbs_xy = vbs_result.m_vbs_xy;
    results.m_vbs_heis = vbs_result.m_vbs_heis;
    results.D_mean = vbs_result.D_mean_xy;
    results.D_mean_xy = vbs_result.D_mean_xy;
    results.D_mean_heis = vbs_result.D_mean_heis;
    
    // Plaquette/bowtie resonance order
    auto plaq_result = compute_plaquette_order(psi, cluster);
    results.m_plaquette = plaq_result.m_plaquette;
    results.P_mean = plaq_result.P_mean;
    results.resonance_strength = plaq_result.resonance_strength;
    results.chi_mean = plaq_result.chi_mean;
    results.n_plaquettes = plaq_result.n_plaquettes;
    results.n_triangles = plaq_result.n_triangles;
    
    return results;
}

// -----------------------------------------------------------------------------
// Process all temperatures for a single Jpm directory (for tpq_all_temps mode)
// Returns a vector of results, one per temperature
// -----------------------------------------------------------------------------

std::vector<OrderParameterResults> process_all_temperatures(
    const std::string& wf_file,
    const Cluster& cluster,
    double jpm,
    int n_q_grid,
    bool save_full
) {
    std::vector<OrderParameterResults> results_list;
    
    // Load all TPQ states (all temperatures)
    auto tpq_states = load_all_tpq_states(wf_file);
    
    if (tpq_states.empty()) {
        std::cerr << "No TPQ states found in " << wf_file << std::endl;
        return results_list;
    }
    
    std::cout << "  Processing " << tpq_states.size() << " temperatures for Jpm=" << jpm << std::endl;
    
    // Process each temperature (using quick scalar computation; full mode not yet implemented for T-scan)
    (void)n_q_grid;
    (void)save_full;
    
    for (size_t t_idx = 0; t_idx < tpq_states.size(); ++t_idx) {
        const auto& tpq_state = tpq_states[t_idx];
        
        OrderParameterResults results = compute_all_order_parameters(tpq_state.psi, cluster, jpm);
        results.temperature = tpq_state.temperature;
        
        results_list.push_back(results);
    }
    
    return results_list;
}

// `save_temperature_scan_results` now lives in `ed_bfg::results_io`. The
// using-declaration at the top of this TU keeps the call site below
// unchanged. P2.1 (7th slice).

// -----------------------------------------------------------------------------
// Scan directory mode
// -----------------------------------------------------------------------------

std::vector<OrderParameterResults> scan_jpm_directories(
    const std::string& scan_dir,
    const std::string& output_dir,
    int n_workers,
    int n_q_grid,
    bool save_full,
    bool use_tpq,
    bool tpq_all_temps = false
) {
    // Get MPI rank and size
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    
    // Find all Jpm=* directories (all ranks do this)
    std::vector<std::pair<double, std::string>> jpm_dirs;
    std::regex jpm_regex("Jpm=([+-]?[0-9]*\\.?[0-9]+)");
    
    for (const auto& entry : fs::directory_iterator(scan_dir)) {
        if (!entry.is_directory()) continue;
        std::string dirname = entry.path().filename().string();
        std::smatch match;
        if (std::regex_search(dirname, match, jpm_regex)) {
            double jpm = std::stod(match[1].str());
            jpm_dirs.push_back({jpm, entry.path().string()});
        }
    }
    
    std::sort(jpm_dirs.begin(), jpm_dirs.end());
    
    if (mpi_rank == 0) {
        std::cout << "Found " << jpm_dirs.size() << " Jpm directories" << std::endl;
#ifdef USE_MPI
        std::cout << "MPI enabled: " << mpi_size << " processes" << std::endl;
#endif
        if (save_full) {
            std::cout << "Full output mode: saving S(q) and S_D(q) 2D grids per directory" << std::endl;
        }
        if (tpq_all_temps) {
            std::cout << "TPQ all-temps mode: processing ALL temperatures for each Jpm" << std::endl;
        }
    }
    
    if (jpm_dirs.empty()) {
        return {};
    }
    
    // Load cluster from first directory
    Cluster cluster = load_cluster(jpm_dirs[0].second);
    if (mpi_rank == 0) {
        std::cout << "Cluster: " << cluster.n_sites << " sites, " 
                  << cluster.edges_nn.size() << " bonds" << std::endl;
    }
    
    // Process directories - each MPI rank processes a subset
    std::vector<OrderParameterResults> all_results(jpm_dirs.size());
    std::mutex print_mutex;
    std::atomic<int> completed(0);
    
    // Distribute work: rank processes indices where (i % mpi_size == mpi_rank)
    #pragma omp parallel for schedule(dynamic) num_threads(n_workers)
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        // MPI work distribution: only process if this index belongs to this rank
#ifdef USE_MPI
        if (static_cast<int>(i % mpi_size) != mpi_rank) {
            continue;
        }
#endif
        double jpm = jpm_dirs[i].first;
        const std::string& dir = jpm_dirs[i].second;
        
        try {
            // Find wavefunction file
            std::string wf_file;
            
            // First check output/ subdirectory for ed_results.h5
            std::string output_subdir = dir + "/output";
            if (fs::exists(output_subdir)) {
                std::string ed_results = output_subdir + "/ed_results.h5";
                if (fs::exists(ed_results)) {
                    wf_file = ed_results;
                } else {
                    // Search for any .h5 file in output/
                    for (const auto& entry : fs::directory_iterator(output_subdir)) {
                        if (entry.path().extension() == ".h5") {
                            wf_file = entry.path().string();
                            break;
                        }
                    }
                }
            }
            
            // If not found in output/, try main directory
            if (wf_file.empty()) {
                for (const auto& entry : fs::directory_iterator(dir)) {
                    std::string fname = entry.path().filename().string();
                    if (fname.find(".h5") != std::string::npos && 
                        fname.find("eigenvector") != std::string::npos) {
                        wf_file = entry.path().string();
                        break;
                    }
                }
            }
            
            if (wf_file.empty()) {
                // Try other patterns in main directory
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".h5") {
                        wf_file = entry.path().string();
                        break;
                    }
                }
            }
            
            if (wf_file.empty()) {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cerr << "No wavefunction file found in " << dir << std::endl;
                continue;
            }
            
            // ============================================================
            // TPQ ALL TEMPERATURES MODE: process all beta values and save
            // per-Jpm temperature-dependent results
            // ============================================================
            if (tpq_all_temps) {
                auto temp_results = process_all_temperatures(wf_file, cluster, jpm, n_q_grid, save_full);
                
                if (temp_results.empty()) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "No TPQ states found in " << wf_file << std::endl;
                    continue;
                }
                
                // Save per-Jpm temperature scan file
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << jpm;
                std::string out_file = output_dir + "/order_params_Jpm=" + oss.str() + "_T_scan.h5";
                save_temperature_scan_results(temp_results, out_file, jpm);
                
                // Use lowest temperature result for summary
                all_results[i] = temp_results.back();  // Sorted by T descending, so last is lowest
                
                int done = ++completed;
                {
                    std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
                    std::cout << "[Rank " << mpi_rank << "] ";
#endif
                    std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                              << "Jpm=" << std::fixed << std::setprecision(4) << jpm
                              << " | " << temp_results.size() << " temperatures processed"
                              << " | T_range=[" << temp_results.back().temperature 
                              << ", " << temp_results.front().temperature << "]"
                              << std::endl;
                }
                continue;  // Skip the normal single-temperature processing
            }
            
            // Load wavefunction (ground state or TPQ lowest temperature)
            std::vector<Complex> psi;
            double temperature = 0.0;
            
            if (use_tpq) {
                try {
                    auto [tpq_psi, T] = load_tpq_state(wf_file);
                    psi = std::move(tpq_psi);
                    temperature = T;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cerr << "TPQ load failed for " << dir << ": " << e.what() 
                              << ", falling back to ground state" << std::endl;
                    psi = load_wavefunction(wf_file);
                }
            } else {
                psi = load_wavefunction(wf_file);
            }
            
            // Set memory-efficient mode on first file processed by this thread
            // (all files should have same Hilbert space size)
            #pragma omp critical(set_memory_mode)
            {
                static bool memory_mode_set = false;
                if (!memory_mode_set) {
                    set_memory_efficient_mode(psi.size());
                    memory_mode_set = true;
                }
            }
            
            OrderParameterResults results;
            results.jpm = jpm;
            results.temperature = temperature;
            
            if (save_full) {
                // Full computation with 2D grids
                auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
                auto szsz_corr = compute_szsz_correlations(psi, cluster.n_sites);
                auto sf_result = compute_spin_structure_factor(smsp_corr, szsz_corr, cluster);
                auto s_q_2d = compute_sq_2d_grid(smsp_corr, szsz_corr, cluster, n_q_grid);
                
                auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
                auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
                auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
                auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
                
                auto nem_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
                auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
                auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
                auto nem_heisenberg_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
                
                auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster, n_q_grid);
                auto plaq_result = compute_plaquette_order(psi, cluster, n_q_grid);
                
                // Save full results to per-Jpm file
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << jpm;
                std::string out_file = output_dir + "/order_params_Jpm=" + oss.str() + ".h5";
                save_results(out_file, sf_result, nem_result, nem_spsm_result, nem_szsz_result,
                            nem_heisenberg_result, vbs_result, plaq_result,
                            cluster, s_q_2d, n_q_grid,
                            spsm_bond_exp, szsz_bond_exp, heisenberg_bond_exp);
                
                // Fill scalar results for summary
                results.m_translation = sf_result.m_translation;
                results.m_nematic = nem_result.m_nem;
                results.m_nematic_spsm = nem_spsm_result.m_nem;
                results.m_nematic_szsz = nem_szsz_result.m_nem;
                results.m_nematic_heisenberg = nem_heisenberg_result.m_nem;
                results.anisotropy = nem_result.anisotropy;
                results.m_vbs = vbs_result.m_vbs_xy;
                results.m_vbs_xy = vbs_result.m_vbs_xy;
                results.m_vbs_heis = vbs_result.m_vbs_heis;
                results.D_mean = vbs_result.D_mean_xy;
                results.D_mean_xy = vbs_result.D_mean_xy;
                results.D_mean_heis = vbs_result.D_mean_heis;
                results.m_plaquette = plaq_result.m_plaquette;
                results.P_mean = plaq_result.P_mean;
                results.resonance_strength = plaq_result.resonance_strength;
                results.chi_mean = plaq_result.chi_mean;
                results.n_plaquettes = plaq_result.n_plaquettes;
                results.n_triangles = plaq_result.n_triangles;
                results.D_mean_xy = vbs_result.D_mean_xy;
                results.D_mean_heis = vbs_result.D_mean_heis;
            } else {
                // Quick scalar-only computation
                results = compute_all_order_parameters(psi, cluster, jpm);
                results.temperature = temperature;  // Preserve temperature from TPQ
            }
            
            all_results[i] = results;
            
            int done = ++completed;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
                std::cout << "[Rank " << mpi_rank << "] ";
#endif
                std::cout << "[" << done << "/" << jpm_dirs.size() << "] "
                          << "Jpm=" << std::fixed << std::setprecision(4) << jpm;
                if (use_tpq) {
                    std::cout << " T=" << std::setprecision(6) << results.temperature;
                }
                std::cout << " | m_trans=" << std::setprecision(6) << results.m_translation
                          << " | m_nem=" << results.m_nematic
                          << " | m_vbs_xy=" << results.m_vbs_xy
                          << " | m_vbs_heis=" << results.m_vbs_heis
                          << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(print_mutex);
#ifdef USE_MPI
            std::cerr << "[Rank " << mpi_rank << "] ";
#endif
            std::cerr << "Error processing " << dir << ": " << e.what() << std::endl;
        }
    }
    
#ifdef USE_MPI
    // Gather results from all ranks to rank 0
    if (mpi_rank == 0) {
        std::cout << "\nGathering results from all MPI ranks..." << std::endl;
    }
    
    // Create a flat buffer for MPI communication (just the scalar values)
    std::vector<double> local_buffer;
    std::vector<int> local_indices;
    
    for (size_t i = 0; i < jpm_dirs.size(); ++i) {
        if (static_cast<int>(i % mpi_size) == mpi_rank && all_results[i].is_valid()) {
            local_indices.push_back(i);
            local_buffer.push_back(all_results[i].jpm);
            local_buffer.push_back(all_results[i].temperature);
            local_buffer.push_back(all_results[i].m_translation);
            local_buffer.push_back(all_results[i].m_nematic);
            local_buffer.push_back(all_results[i].m_nematic_spsm);
            local_buffer.push_back(all_results[i].m_nematic_szsz);
            local_buffer.push_back(all_results[i].m_nematic_heisenberg);
            local_buffer.push_back(all_results[i].m_vbs);
            local_buffer.push_back(all_results[i].m_vbs_xy);
            local_buffer.push_back(all_results[i].m_vbs_heis);
            local_buffer.push_back(all_results[i].anisotropy);
            local_buffer.push_back(all_results[i].D_mean);
            local_buffer.push_back(all_results[i].D_mean_xy);
            local_buffer.push_back(all_results[i].D_mean_heis);
        }
    }
    
    // Gather all results to rank 0
    std::vector<int> recv_counts(mpi_size);
    int local_count = local_buffer.size();
    MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> displs;
    std::vector<double> recv_buffer;
    if (mpi_rank == 0) {
        displs.resize(mpi_size);
        int total = 0;
        for (int i = 0; i < mpi_size; ++i) {
            displs[i] = total;
            total += recv_counts[i];
        }
        recv_buffer.resize(total);
    }
    
    MPI_Gatherv(local_buffer.data(), local_count, MPI_DOUBLE,
                recv_buffer.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // Gather indices
    std::vector<int> all_indices;
    std::vector<int> index_counts(mpi_size);
    int local_idx_count = local_indices.size();
    MPI_Gather(&local_idx_count, 1, MPI_INT, index_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    std::vector<int> idx_displs;
    if (mpi_rank == 0) {
        idx_displs.resize(mpi_size);
        int total = 0;
        for (int i = 0; i < mpi_size; ++i) {
            idx_displs[i] = total;
            total += index_counts[i];
        }
        all_indices.resize(total);
    }
    
    MPI_Gatherv(local_indices.data(), local_idx_count, MPI_INT,
                all_indices.data(), index_counts.data(), idx_displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    
    // Rank 0 unpacks the results
    if (mpi_rank == 0) {
        const int vals_per_result = 14;
        for (size_t i = 0; i < all_indices.size(); ++i) {
            int idx = all_indices[i];
            size_t offset = i * vals_per_result;
            all_results[idx].jpm = recv_buffer[offset + 0];
            all_results[idx].temperature = recv_buffer[offset + 1];
            all_results[idx].m_translation = recv_buffer[offset + 2];
            all_results[idx].m_nematic = recv_buffer[offset + 3];
            all_results[idx].m_nematic_spsm = recv_buffer[offset + 4];
            all_results[idx].m_nematic_szsz = recv_buffer[offset + 5];
            all_results[idx].m_nematic_heisenberg = recv_buffer[offset + 6];
            all_results[idx].m_vbs = recv_buffer[offset + 7];
            all_results[idx].m_vbs_xy = recv_buffer[offset + 8];
            all_results[idx].m_vbs_heis = recv_buffer[offset + 9];
            all_results[idx].anisotropy = recv_buffer[offset + 10];
            all_results[idx].D_mean = recv_buffer[offset + 11];
            all_results[idx].D_mean_xy = recv_buffer[offset + 12];
            all_results[idx].D_mean_heis = recv_buffer[offset + 13];
        }
    }
#endif
    
    return all_results;
}

// `save_scan_results` now lives in `ed_bfg::results_io`. The
// using-declaration at the top of this TU keeps the call site below
// unchanged. P2.1 (7th slice).

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  Single file mode:\n"
              << "    " << prog << " <wavefunction.h5> <cluster_dir> [output.h5]\n\n"
              << "  Scan directory mode:\n"
              << "    " << prog << " --scan-dir <dir> [options]\n\n"
              << "Options:\n"
              << "  --scan-dir <dir>     Directory containing Jpm=* subdirectories\n"
              << "  --output-dir <dir>   Output directory for results\n"
              << "  --n-workers <n>      Number of parallel workers (default: 4)\n"
              << "  --n-q-grid <n>       2D q-grid size for visualization (default: 50)\n"
              << "  --save-full          Save full S(q), S_D(q), S_P(q) 2D grids and bond-resolved data per Jpm\n"
              << "  --tpq                Use TPQ states (lowest temperature) instead of ground state\n"
              << "  --tpq-all-temps      Process ALL temperatures from TPQ data (creates T-dependent output)\n"
              << "\nComputes BFG order parameters from ground state or TPQ wavefunction:\n"
              << "  1. S(q) - Spin structure factor (translation order)\n"
              << "  2. Nematic order - Bond orientation anisotropy (C3 breaking)\n"
              << "     - Variants: XY, S+S-, SzSz, Heisenberg\n"
              << "  3. VBS order - Valence bond solid with proper 4-site dimer correlations\n"
              << "     - Variants: XY dimer (S+S- + S-S+), Heisenberg dimer (S·S)\n"
              << "     - Bond-resolved: full dimer-dimer correlation matrices\n"
              << "  4. Plaquette/bowtie resonance order - BFG-native 4-spin ring flip\n"
              << "     - S_P(q) structure factor for bowtie plaquettes\n"
              << "     - Triangle chiral order\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#else
    int mpi_rank = 0;
#endif
    
    if (argc < 2) {
        if (mpi_rank == 0) {
            print_usage(argv[0]);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    std::string scan_dir, output_dir;
    std::string wf_file, cluster_dir, output_file = "bfg_order_parameters.h5";
    int n_workers = 4;
    int n_q_grid = 50;
    bool scan_mode = false;
    bool save_full = false;
    bool use_tpq = false;
    bool tpq_all_temps = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--scan-dir" && i + 1 < argc) {
            scan_dir = argv[++i];
            scan_mode = true;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--n-workers" && i + 1 < argc) {
            n_workers = std::stoi(argv[++i]);
        } else if (arg == "--n-q-grid" && i + 1 < argc) {
            n_q_grid = std::stoi(argv[++i]);
        } else if (arg == "--save-full") {
            save_full = true;
        } else if (arg == "--tpq") {
            use_tpq = true;
        } else if (arg == "--tpq-all-temps") {
            use_tpq = true;
            tpq_all_temps = true;
        } else if (arg == "-h" || arg == "--help") {
            if (mpi_rank == 0) {
                print_usage(argv[0]);
            }
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else if (wf_file.empty()) {
            wf_file = arg;
        } else if (cluster_dir.empty()) {
            cluster_dir = arg;
        } else {
            output_file = arg;
        }
    }
    
    #ifdef _OPENMP
    if (mpi_rank == 0) {
        std::cout << "OpenMP enabled with " << omp_get_max_threads() << " max threads" << std::endl;
    }
    #endif
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (scan_mode) {
            // Scan directory mode
            if (scan_dir.empty()) {
                if (mpi_rank == 0) {
                    std::cerr << "Error: --scan-dir required in scan mode" << std::endl;
                }
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            if (output_dir.empty()) {
                output_dir = scan_dir + "/order_parameter_results";
            }
            
            // Only rank 0 creates output directory
            if (mpi_rank == 0) {
                fs::create_directories(output_dir);
                
                std::string mode_str;
                if (tpq_all_temps) {
                    mode_str = "TPQ (all temperatures)";
                } else if (use_tpq) {
                    mode_str = "TPQ (lowest temperature)";
                } else {
                    mode_str = "Ground state";
                }
                
                std::cout << "========================================\n"
                          << "BFG ORDER PARAMETER SCAN (CPU)\n"
                          << "========================================\n"
                          << "Scan directory: " << scan_dir << "\n"
                          << "Output directory: " << output_dir << "\n"
                          << "Workers: " << n_workers << "\n"
                          << "Mode: " << mode_str << "\n"
                          << "Save full: " << (save_full ? "yes (2D grids)" : "no (scalars only)") << "\n"
                          << "========================================" << std::endl;
            }
            
#ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);  // Wait for directory creation
#endif
            
            auto results = scan_jpm_directories(scan_dir, output_dir, n_workers, n_q_grid, save_full, use_tpq, tpq_all_temps);
            
            // Only rank 0 saves the combined results
            if (mpi_rank == 0 && !results.empty()) {
                std::string results_file = output_dir + "/scan_results.h5";
                save_scan_results(results, results_file);
            }
            
        } else {
            // Single file mode - only run on rank 0
#ifdef USE_MPI
            if (mpi_rank != 0) {
                MPI_Finalize();
                return 0;  // Other ranks exit
            }
#endif
            
            if (wf_file.empty() || cluster_dir.empty()) {
                print_usage(argv[0]);
#ifdef USE_MPI
                MPI_Finalize();
#endif
                return 1;
            }
            
            std::cout << "========================================\n"
                      << "BFG ORDER PARAMETER COMPUTATION (CPU)\n"
                      << "========================================" << std::endl;
            
            Cluster cluster = load_cluster(cluster_dir);
            
            std::vector<Complex> psi;
            double temperature = 0.0;
            
            if (use_tpq) {
                auto [tpq_psi, tpq_temp] = load_tpq_state(wf_file);
                psi = std::move(tpq_psi);
                temperature = tpq_temp;
                std::cout << "Using TPQ state (T=" << temperature << ")" << std::endl;
            } else {
                psi = load_wavefunction(wf_file);
                std::cout << "Using ground state wavefunction" << std::endl;
            }
            
            // Set memory-efficient mode based on Hilbert space size
            set_memory_efficient_mode(psi.size());
            
            // Verify size
            uint64_t expected_size = 1ULL << cluster.n_sites;
            if (psi.size() != expected_size) {
                std::cerr << "Warning: wavefunction size " << psi.size() 
                          << " != expected 2^" << cluster.n_sites << " = " << expected_size << std::endl;
            }
            
            // Compute correlations
            auto smsp_corr = compute_smsp_correlations(psi, cluster.n_sites);
            auto szsz_corr = compute_szsz_correlations(psi, cluster.n_sites);
            
            // Compute S(q) at k-points (full Heisenberg = SzSz + (1/2)(S+S- + S-S+))
            auto sf_result = compute_spin_structure_factor(smsp_corr, szsz_corr, cluster);
            
            // Compute 2D S(q) grid for visualization
            auto s_q_2d = compute_sq_2d_grid(smsp_corr, szsz_corr, cluster, n_q_grid);
            
            // Compute bond expectations for nematic and VBS
            auto xy_bond_exp = compute_xy_bond_expectations(psi, cluster);
            
            // Compute additional bond operators for visualization
            auto spsm_bond_exp = compute_spsm_bond_expectations(psi, cluster);
            auto szsz_bond_exp = compute_szsz_bond_expectations(psi, cluster);
            auto heisenberg_bond_exp = compute_heisenberg_bond_expectations(szsz_bond_exp, xy_bond_exp);
            
            // Compute nematic order for all three bond types
            auto nem_xy_result = compute_nematic_order(xy_bond_exp, cluster, "xy");
            auto nem_spsm_result = compute_nematic_order(spsm_bond_exp, cluster, "spsm");
            auto nem_szsz_result = compute_nematic_order_real(szsz_bond_exp, cluster, "szsz");
            auto nem_heisenberg_result = compute_nematic_order_real(heisenberg_bond_exp, cluster, "heisenberg");
            
            // Use XY nematic as primary (for backwards compatibility)
            auto nem_result = nem_xy_result;
            
            // VBS order with proper 4-site correlations (with 2D grid)
            auto vbs_result = compute_vbs_order(psi, xy_bond_exp, heisenberg_bond_exp, cluster, n_q_grid);
            
            // Plaquette/bowtie resonance order
            auto plaq_result = compute_plaquette_order(psi, cluster, n_q_grid);
            
            // Save results with full 2D grids and bond data
            save_results(output_file, sf_result, nem_result, nem_spsm_result, nem_szsz_result, 
                        nem_heisenberg_result, vbs_result, plaq_result,
                        cluster, s_q_2d, n_q_grid,
                        spsm_bond_exp, szsz_bond_exp, heisenberg_bond_exp);
            
            // Print summary
            std::cout << "\n========== ORDER PARAMETER SUMMARY ==========" << std::endl;
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Translation order:  m = " << sf_result.m_translation 
                      << " at q = (" << sf_result.q_max[0] << ", " << sf_result.q_max[1] << ")" << std::endl;
            std::cout << "Nematic order (XY):         m = " << nem_result.m_nem 
                      << ", anisotropy = " << nem_result.anisotropy << std::endl;
            std::cout << "Nematic order (S+S-):       m = " << nem_spsm_result.m_nem 
                      << ", anisotropy = " << nem_spsm_result.anisotropy << std::endl;
            std::cout << "Nematic order (SzSz):       m = " << nem_szsz_result.m_nem 
                      << ", anisotropy = " << nem_szsz_result.anisotropy << std::endl;
            std::cout << "Nematic order (Heisenberg): m = " << nem_heisenberg_result.m_nem 
                      << ", anisotropy = " << nem_heisenberg_result.anisotropy << std::endl;
            std::cout << "VBS order (XY, 4-site):         m = " << vbs_result.m_vbs_xy 
                      << " at q = (" << vbs_result.q_max_xy[0] << ", " << vbs_result.q_max_xy[1] << ")" << std::endl;
            std::cout << "VBS order (Heisenberg, 4-site): m = " << vbs_result.m_vbs_heis 
                      << " at q = (" << vbs_result.q_max_heis[0] << ", " << vbs_result.q_max_heis[1] << ")" << std::endl;
            std::cout << "Plaquette order (bowtie):   m = " << plaq_result.m_plaquette 
                      << " at q = (" << plaq_result.q_max[0] << ", " << plaq_result.q_max[1] << ")" << std::endl;
            std::cout << "  Resonance: <P> = " << plaq_result.P_mean 
                      << ", strength = " << plaq_result.resonance_strength << std::endl;
            std::cout << "  Triangle chiral: <χ> = " << plaq_result.chi_mean << std::endl;
            std::cout << "==============================================" << std::endl;
        }
        
    } catch (const std::exception& e) {
        if (mpi_rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return 1;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    if (mpi_rank == 0) {
        std::cout << "\nTotal runtime: " << total_duration.count() << " seconds" << std::endl;
    }
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    return 0;
}
