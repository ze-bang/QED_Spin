// =============================================================================
// include/ed/bfg/cluster.h
//
// `ed::bfg::Cluster` and `load_cluster(...)` -- the canonical, library-level
// representation of a kagome / pyrochlore-superlattice cluster used by the
// BFG order-parameter pipeline.
//
// Before this refactor (P2.1), the same struct + loader lived in
// `src/apps/compute_bfg_order_parameters.cpp` (CPU, ~330 LOC) and a *drifted*
// copy in `src/apps/compute_bfg_order_parameters_gpu.cu` (GPU, simpler). By
// promoting the CPU version to a library type the two binaries can share
// the same loader and Python bindings can construct the cluster directly.
//
// Audit ref: P2.1 (BFG library extraction).
// =============================================================================

#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ed::bfg {

/**
 * Geometry + connectivity of a kagome / pyrochlore-superlattice BFG cluster.
 *
 * Loaded from a `cluster_dir` containing:
 *   * `<prefix>_(pbc|obc)_lattice_parameters.dat`
 *   * `positions.dat`
 *   * `<prefix>_(pbc|obc)_nn_list.dat` (optional; falls back to distance cut)
 *
 * Layout follows the `kagome_bfg_*` and `pyrochlore_super_*` data files
 * already in the repo's `lattice_data/` tree.
 */
struct Cluster {
    int n_sites{0};
    std::vector<std::array<double, 2>> positions;
    std::vector<int>                    sublattice;
    std::vector<std::pair<int, int>>    edges_nn;
    std::map<int, std::vector<int>>     nn_list;

    /// Real-space lattice vectors (a1, a2).
    std::array<double, 2> a1{0.0, 0.0};
    std::array<double, 2> a2{0.0, 0.0};
    /// Reciprocal lattice vectors (b1, b2).
    std::array<double, 2> b1{0.0, 0.0};
    std::array<double, 2> b2{0.0, 0.0};

    std::vector<std::array<double, 2>> k_points;

    /// Grid dimensions, e.g. 3x3 for a kagome 27-site cluster.
    int n_cells_x{0};
    int n_cells_y{0};

    /// Bond orientation (0=horizontal, 1=~60°, 2=~120°) for each NN edge.
    std::map<std::pair<int, int>, int> bond_orientation;

    /// Unit cell coordinates (n1, n2) for each site -- used by
    /// `minimum_image_displacement` to compute PBC-correct vectors.
    std::vector<std::array<int, 2>>    cell_coords;
    std::vector<std::array<double, 2>> sublattice_offsets;
    int sites_per_cell{3};   // Kagome default; set to 4 for pyrochlore.

    /**
     * Compute the minimum-image displacement r_j - r_i in the chosen PBC.
     *
     * Falls back to the naïve (positions[j] - positions[i]) if the cluster
     * was loaded without grid metadata.
     */
    std::array<double, 2> minimum_image_displacement(int i, int j) const;

    /**
     * Bond midpoint using the minimum-image displacement; correct even for
     * bonds that wrap around the periodic boundary.
     */
    std::array<double, 2> bond_center_pbc(int i, int j) const;
};

/**
 * Load a Cluster from `cluster_dir`.
 *
 * @throws std::runtime_error if `cluster_dir/positions.dat` cannot be read.
 *
 * On success, all geometry, connectivity, k-point grid, bond orientations,
 * and unit-cell coordinates are populated. Diagnostics (file paths,
 * site/edge counts) are written to `std::cout` for parity with the
 * pre-refactor behavior.
 */
Cluster load_cluster(const std::string& cluster_dir);

}  // namespace ed::bfg
