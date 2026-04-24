// =============================================================================
// include/ed/bfg/topology.h
//
// Combinatorial helpers for the BFG order-parameter pipeline that operate
// only on the cluster connectivity (no wavefunctions, no HDF5, no OpenMP).
// They live in `ed_bfg` so the CPU driver
// (`compute_bfg_order_parameters.cpp`), the GPU driver
// (`compute_bfg_order_parameters_gpu.cu`), and the future Python bindings can
// all call the same authoritative implementations instead of copy-pasting
// the search routines.
//
// Audit ref: P2.1 (BFG library extraction, topology slice).
// =============================================================================

#pragma once

#include <array>
#include <vector>

#include "ed/bfg/cluster.h"

namespace ed::bfg {

/**
 * One bowtie = two NN-triangles sharing exactly one vertex (`s0`) on the
 * kagome lattice.
 *
 *   s1   s3
 *    \\ //
 *    s0
 *    // \\
 *   s2   s4
 *
 * `center` is the mean Cartesian position of the five sites and is what
 * downstream Fourier-transform code uses as the bowtie phase center.
 * `orientation` is the sublattice index of the shared vertex (0, 1, or 2)
 * which encodes the bowtie's spatial orientation on the kagome lattice.
 */
struct Bowtie {
    int s0;
    int s1;
    int s2;
    int s3;
    int s4;
    std::array<double, 2> center;
    int orientation;
};

/**
 * Enumerate every (i, j, k) triple such that all three of (i,j), (j,k),
 * (i,k) are nearest-neighbour edges in `cluster.edges_nn`. The returned
 * triples are canonicalized so that i < j < k, and each triangle appears
 * exactly once.
 */
std::vector<std::array<int, 3>> find_triangles(const Cluster& cluster);

/**
 * Enumerate every bowtie (pair of NN-triangles sharing exactly one
 * vertex) in `cluster`. Built on top of `find_triangles`. Duplicates
 * coming from the symmetric (t1, t2) <-> (t2, t1) swap are removed.
 */
std::vector<Bowtie> find_bowties(const Cluster& cluster);

}  // namespace ed::bfg
