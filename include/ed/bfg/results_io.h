// =============================================================================
// include/ed/bfg/results_io.h
//
// HDF5 results-writers + scalar-summary structs for the BFG order-parameter
// pipeline (P2.1 results-IO slice).
//
// Lifted out of `compute_bfg_order_parameters.cpp` so the GPU driver, the
// Python bindings, and any future test harnesses share the same authoritative
// on-disk layout. Two complementary surfaces live here:
//
//   * Per-Jpm / per-temperature scalar aggregates -- `OrderParameterResults`,
//     `save_temperature_scan_results`, `save_scan_results`.
//   * Full per-cluster results bundle -- `NematicResult`, `VBSResult`,
//     `PlaquetteResult`, `Sq2DGridResult`, plus `save_results`.
//
// Phase / index conventions follow the historical CPU-driver behaviour
// (HDF5 dataset names, attribute names, orientation index map
// `0=(0,0), 1=(0,1), 2=(0,2), 3=(1,1), 4=(1,2), 5=(2,2)`).
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <array>
#include <complex>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ed/bfg/cluster.h"
#include "ed/bfg/spin_structure_factor.h"

namespace ed::bfg {

using Complex = std::complex<double>;

/**
 * Scalar nematic-order summary (one orientation channel).
 *
 * `psi_nem = O_bar[0] + ω O_bar[1] + ω^2 O_bar[2]` with `ω = exp(2πi/3)`.
 * `m_nem = |psi_nem|`. `bond_type` is the free-form tag the kernel was
 * called with ("xy", "spsm", "szsz", "heisenberg").
 */
struct NematicResult {
    Complex psi_nem{0.0, 0.0};
    double m_nem{0.0};
    std::array<Complex, 3> O_bar{Complex{0.0, 0.0},
                                 Complex{0.0, 0.0},
                                 Complex{0.0, 0.0}};
    double anisotropy{0.0};
    std::string bond_type;
};

/**
 * Valence-bond-solid order-parameter bundle. The struct carries both the
 * XY (S+S- + S-S+) and Heisenberg (S·S) variants together with their
 * orientation-resolved, bond-resolved, and 2D grid representations. All
 * vector members default to empty so `save_results` / unit tests can
 * skip the "if not empty" branches by simply leaving them unset.
 */
struct VBSResult {
    std::vector<Complex> S_d_xy;
    std::vector<std::vector<Complex>> S_d_xy_2d;
    std::vector<std::vector<Complex>> dimer_corr_xy;
    std::vector<std::vector<Complex>> connected_corr_xy;
    int q_max_idx_xy{0};
    Complex s_d_max_xy{0.0, 0.0};
    std::array<double, 2> q_max_xy{0.0, 0.0};
    double m_vbs_xy{0.0};
    double D_mean_xy{0.0};

    std::vector<double> S_d_heis;
    std::vector<std::vector<double>> S_d_heis_2d;
    std::vector<std::vector<double>> dimer_corr_heis;
    std::vector<std::vector<double>> connected_corr_heis;
    int q_max_idx_heis{0};
    double s_d_max_heis{0.0};
    std::array<double, 2> q_max_heis{0.0, 0.0};
    double m_vbs_heis{0.0};
    double D_mean_heis{0.0};

    std::vector<std::array<Complex, 6>> S_d_xy_oriented;
    std::vector<std::array<double, 6>> S_d_heis_oriented;
    std::array<int, 3> n_bonds_per_orientation{0, 0, 0};

    int n_q_grid{0};
    int n_bonds{0};
};

/**
 * Bowtie / triangle plaquette resonance bundle.
 */
struct PlaquetteResult {
    std::vector<Complex> S_p;
    std::vector<std::vector<Complex>> S_p_2d;
    int q_max_idx{0};
    Complex s_p_max{0.0, 0.0};
    std::array<double, 2> q_max{0.0, 0.0};
    double m_plaquette{0.0};

    std::vector<std::array<Complex, 6>> S_p_oriented;
    std::array<int, 3> n_plaquettes_per_orientation{0, 0, 0};

    int n_plaquettes{0};
    double P_mean{0.0};
    double resonance_strength{0.0};
    std::vector<Complex> P_r;
    std::vector<std::array<double, 2>> centers;
    std::vector<int> orientations;

    int n_triangles{0};
    double chi_mean{0.0};
    std::vector<Complex> chi_r;
};

/**
 * 2D q-grid spin structure factor (Heisenberg + SmSp + SzSz).
 *
 * The grid spans the dimensionless reduced coordinates
 * `q1, q2 ∈ [-1, 1]` with `n_q_grid` points per axis; the physical
 * Cartesian wavevector is `q = q1 b1 + q2 b2`.
 */
struct Sq2DGridResult {
    std::vector<std::vector<Complex>> s_q_2d;
    std::vector<std::vector<Complex>> s_q_smsp_2d;
    std::vector<std::vector<double>> s_q_szsz_2d;
};

/**
 * Per-cluster scalar summary used by scan / temperature-scan modes.
 *
 * `is_valid()` returns false when `jpm` is left at its sentinel NaN -- the
 * default-constructed value -- so MPI gather paths can recognise un-touched
 * slots without an explicit `bool` flag.
 */
struct OrderParameterResults {
    double jpm{std::numeric_limits<double>::quiet_NaN()};
    double temperature{0.0};
    double m_translation{0.0};
    double m_nematic{0.0};
    double m_nematic_spsm{0.0};
    double m_nematic_szsz{0.0};
    double m_nematic_heisenberg{0.0};
    double m_vbs{0.0};
    double m_vbs_xy{0.0};
    double m_vbs_heis{0.0};
    double anisotropy{0.0};
    double D_mean{0.0};
    double D_mean_xy{0.0};
    double D_mean_heis{0.0};
    double m_plaquette{0.0};
    double P_mean{0.0};
    double resonance_strength{0.0};
    double chi_mean{0.0};
    int n_plaquettes{0};
    int n_triangles{0};

    bool is_valid() const { return !std::isnan(jpm); }
};

/**
 * Write the full per-cluster result bundle to `filename` (HDF5, truncate).
 *
 * Empty `bonds` / `vbs.S_d_*` / `plaq.S_p` members are skipped silently to
 * keep the on-disk schema parity with the historical CPU driver -- writers
 * only emit datasets when their input vector is non-empty. `n_q_grid`
 * controls the size of the 2D grid datasets *and* the abscissa
 * `q_grid_vals = linspace(-1, 1, n_q_grid)` written alongside.
 */
void save_results(
    const std::string& filename,
    const StructureFactorResult& sf,
    const NematicResult& nem,
    const NematicResult& nem_spsm,
    const NematicResult& nem_szsz,
    const NematicResult& nem_heisenberg,
    const VBSResult& vbs,
    const PlaquetteResult& plaq,
    const Cluster& cluster,
    const Sq2DGridResult& s_q_2d,
    int n_q_grid,
    const std::map<std::pair<int, int>, Complex>& spsm_bonds = {},
    const std::map<std::pair<int, int>, double>& szsz_bonds = {},
    const std::map<std::pair<int, int>, double>& heisenberg_bonds = {}
);

/**
 * Per-Jpm temperature-scan output: one HDF5 file per Jpm directory with
 * length-N temperature-indexed scalar columns. `jpm` is also stored as a
 * root-level attribute. No-op when `results` is empty.
 */
void save_temperature_scan_results(
    const std::vector<OrderParameterResults>& results,
    const std::string& output_file,
    double jpm
);

/**
 * Aggregated Jpm-scan output: a single HDF5 file with length-N
 * Jpm-indexed scalar columns (one entry per directory in the scan).
 */
void save_scan_results(
    const std::vector<OrderParameterResults>& results,
    const std::string& output_file
);

}  // namespace ed::bfg
