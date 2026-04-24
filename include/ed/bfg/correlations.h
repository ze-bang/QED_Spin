// =============================================================================
// include/ed/bfg/correlations.h
//
// Two-body spin correlations and bond expectations on a BFG cluster (P2.1
// correlations slice). Pulled out of `compute_bfg_order_parameters.cpp` so:
//   * the GPU driver can share an identical CPU reference path for
//     equivalence tests,
//   * pybind11 bindings can expose the same arrays Python notebooks already
//     consume from HDF5,
//   * the upstream computations are unit-testable in isolation against
//     analytic single-state cases (Catch2).
//
// All functions are thread-parallel via OpenMP (when available) and operate
// on the *full* 2^N spin-1/2 Hilbert space encoded as a `std::vector<Complex>`
// indexed by basis-state integer. This matches the convention used
// everywhere else in the codebase: bit=0 -> spin UP, bit=1 -> spin DOWN.
//
// Audit ref: P2.1.
// =============================================================================

#pragma once

#include <complex>
#include <map>
#include <utility>
#include <vector>

#include "ed/bfg/cluster.h"

namespace ed::bfg {

using Complex = std::complex<double>;

/**
 * Site-to-site S^- S^+ correlation matrix:
 *   corr[i][j] = <psi| S^-_i S^+_j |psi>
 * Diagonal entries i==j reduce to <psi| (1/2 - S^z_i) |psi> for spin-1/2
 * (i.e. the down-spin density at site i), matching the `(Sm, Sp)` pair
 * convention used by `ed::dssf::build_observable_pairs` for the
 * spin-combination "0,0" in ladder basis.
 */
std::vector<std::vector<Complex>> compute_smsp_correlations(
    const std::vector<Complex>& psi,
    int n_sites);

/**
 * Site-to-site S^z S^z correlation matrix:
 *   corr[i][j] = <psi| S^z_i S^z_j |psi>
 * Diagonal: corr[i][i] = <(S^z_i)^2> = 1/4 for spin-1/2.
 */
std::vector<std::vector<double>> compute_szsz_correlations(
    const std::vector<Complex>& psi,
    int n_sites);

/**
 * XY bond expectation per nearest-neighbour edge:
 *   bonds[(i,j)] = <S^+_i S^-_j + S^-_i S^+_j>
 * Keyed by the same (i,j) pairs that appear in `cluster.edges_nn`.
 */
std::map<std::pair<int, int>, Complex> compute_xy_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster);

/**
 * Asymmetric S^+ S^- bond expectation per nearest-neighbour edge:
 *   bonds[(i,j)] = <S^+_i S^-_j>
 * (Not symmetrized; used for visualization of bond chirality.)
 */
std::map<std::pair<int, int>, Complex> compute_spsm_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster);

/**
 * Diagonal S^z S^z bond expectation per nearest-neighbour edge:
 *   bonds[(i,j)] = <S^z_i S^z_j>
 */
std::map<std::pair<int, int>, double> compute_szsz_bond_expectations(
    const std::vector<Complex>& psi,
    const Cluster& cluster);

/**
 * Full Heisenberg bond expectation per edge:
 *   bonds[(i,j)] = <S_i . S_j>
 *               = <S^z_i S^z_j> + (1/2) Re<S^+_i S^-_j + S^-_i S^+_j>
 * Computed by combining the SzSz and XY bond maps; the imaginary part of
 * the XY map is dropped (it is zero up to noise for the symmetrized
 * combination on a Hermitian state).
 */
std::map<std::pair<int, int>, double> compute_heisenberg_bond_expectations(
    const std::map<std::pair<int, int>, double>& szsz_bonds,
    const std::map<std::pair<int, int>, Complex>& xy_bonds);

}  // namespace ed::bfg
