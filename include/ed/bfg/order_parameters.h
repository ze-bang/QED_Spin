// =============================================================================
// include/ed/bfg/order_parameters.h
//
// BFG order-parameter physics kernels (P2.1 order-parameters slice).
//
// Lifted out of `compute_bfg_order_parameters.cpp` so the GPU driver, the
// Python bindings, and any future test harnesses share the same authoritative
// implementation. The kernels here build on the lower-level slices already
// living in `ed_bfg`:
//
//   * cluster.h               -- `Cluster` lattice container
//   * correlations.h          -- two-body spin correlations + bond expectations
//   * topology.h              -- triangle / bowtie enumeration
//   * structure_factor.h      -- bond-bilinear SF + Fourier-applied dimer ops
//   * spin_structure_factor.h -- site-resolved S(q) over correlations
//   * ring_observables.h      -- bowtie / triangle ring kernels
//   * results_io.h            -- `NematicResult`, `VBSResult`, `PlaquetteResult`,
//                                `Sq2DGridResult`, `OrderParameterResults` POD
//                                types (plus the HDF5 writers)
//
// Surface:
//
//   * compute_nematic_order(...)         -- complex-valued bond expectations
//                                           (XY / S+S- variants)
//   * compute_nematic_order_real(...)    -- real-valued bond expectations
//                                           (SzSz / Heisenberg variants)
//   * compute_vbs_order(...)             -- valence-bond solid S_D(q) over
//                                           XY and Heisenberg dimer operators
//                                           with proper 4-site connected
//                                           correlations
//   * compute_plaquette_order(...)       -- bowtie resonance + triangle chiral
//                                           order parameters
//   * compute_sq_2d_grid(...)            -- dense 2D q-grid spin structure
//                                           factor (full Heisenberg + SmSp /
//                                           SzSz components separately)
//   * compute_all_order_parameters(...)  -- one-shot scalar aggregator that
//                                           builds the per-(Jpm, T)
//                                           `OrderParameterResults` summary
//                                           consumed by scan-mode HDF5 writers
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <complex>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ed/bfg/cluster.h"
#include "ed/bfg/results_io.h"

namespace ed::bfg {

using Complex = std::complex<double>;

NematicResult compute_nematic_order(
    const std::map<std::pair<int, int>, Complex>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type = "xy"
);

NematicResult compute_nematic_order_real(
    const std::map<std::pair<int, int>, double>& bond_exp,
    const Cluster& cluster,
    const std::string& bond_type = "szsz"
);

VBSResult compute_vbs_order(
    const std::vector<Complex>& psi,
    const std::map<std::pair<int, int>, Complex>& xy_bond_exp,
    const std::map<std::pair<int, int>, double>& heisenberg_bond_exp,
    const Cluster& cluster,
    int n_q_grid = 50
);

PlaquetteResult compute_plaquette_order(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    int n_q_grid = 50
);

Sq2DGridResult compute_sq_2d_grid(
    const std::vector<std::vector<Complex>>& smsp_corr,
    const std::vector<std::vector<double>>& szsz_corr,
    const Cluster& cluster,
    int n_q_grid = 50
);

OrderParameterResults compute_all_order_parameters(
    const std::vector<Complex>& psi,
    const Cluster& cluster,
    double jpm_value
);

}  // namespace ed::bfg
