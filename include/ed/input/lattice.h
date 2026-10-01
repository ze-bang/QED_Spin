// =============================================================================
// include/ed/input/lattice.h
//
// Standalone lattice geometry generators, all reachable from one factory
// namespace:
//
//   * chain     -- 1D chain (PBC / OBC)
//   * square    -- 2D square lattice
//   * triangular-- 2D triangular lattice
//   * honeycomb -- 2D honeycomb (2-site basis)
//   * kagome    -- 2D kagome (3-site basis)
//   * pyrochlore-- 3D pyrochlore (4-site basis, FCC)
//   * from_neighbor_lists -- build from an arbitrary user-supplied
//                            adjacency list.
//
// The output `Lattice` struct is purely geometric: sites, 3D Cartesian
// positions, sublattice indices, NN/NNN/NNNN bond lists, and the lattice
// vectors (when applicable). It carries **no** Hamiltonian information --
// the `HamiltonianBuilder` consumes the bond lists separately.
// =============================================================================

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <ed/input/types.h>

namespace ed::input {

struct Lattice {
    // Number of sites.
    std::size_t num_sites = 0;

    // Cartesian site positions (3D; 2D lattices set z = 0).
    std::vector<Position> positions;

    // Sublattice index per site (e.g. 0..3 for pyrochlore, 0..2 for kagome,
    // 0..1 for honeycomb, 0 for chain/square/triangular).
    std::vector<int> sublattice;

    // Nearest-neighbour bonds, each pair once, oriented as generated: from site r
    // to the site r + delta it links to (chain i -> i+1, the wrap bond included);
    // every kagome triangle counter-clockwise; every honeycomb bond from A to B;
    // every pyrochlore bond from the lower sublattice to the higher. A uniform
    // DM vector over these bonds is the translation-invariant model -- except
    // along a periodic length of 2, where a pair's two bonds are stored as one.
    std::vector<Bond> nn_bonds;

    // The second and third distance shells (i < j): the pairs whose distance,
    // the minimum over periodic images on a periodic lattice, is the second /
    // third smallest of the infinite lattice. On the kagome the third shell holds
    // both the hexagon diagonals and the straight two-step pairs.
    std::vector<Bond> nnn_bonds;
    std::vector<Bond> nnnn_bonds;
    // Set by the generators. A lattice built from an adjacency list knows only
    // the bonds it was given, and asking it for a shell is an error.
    bool shells_known = false;

    // Lattice basis vectors (`{a1, a2, a3}`); unused entries are zeroed out.
    std::array<Position, 3> lattice_vectors{};

    // True iff the lattice was generated with periodic boundary conditions.
    bool pbc = false;

    // Free-form lattice label, e.g. "pyrochlore[2x2x2 PBC]". Useful when
    // inspecting / logging / tagging output directories.
    std::string label;

    // Helpers ---------------------------------------------------------------

    // (i, j) of every bond, in order and orientation.
    std::vector<std::pair<std::size_t, std::size_t>> nn_pairs() const;
    // The second / third distance shell. InvalidRequest for a lattice whose
    // shells are unknown (built from an adjacency list) unless they were filled in.
    std::vector<std::pair<std::size_t, std::size_t>> nnn_pairs() const;
    std::vector<std::pair<std::size_t, std::size_t>> nnnn_pairs() const;

    // Convenience: sites 0..num_sites-1 as a vector.
    std::vector<std::size_t> all_sites() const;
};

// =============================================================================
// Factory functions
// =============================================================================

namespace lattice {

// Every generator numbers the sites cell by cell, the first direction fastest
// (pyrochlore: the last), and the basis sites within a cell. On a periodic
// length of 2 the two neighbours along a direction are one site, and that pair
// carries one bond. Lengths must be positive; the lattices with a basis
// (honeycomb, kagome, pyrochlore) need at least 2 cells along a periodic
// direction, because 1 would join bonds of different kinds to one pair of sites.

// 1D chain of `length` sites along x-hat.
Lattice chain(std::size_t length, bool pbc);

// 2D square lattice (`Lx` x `Ly`, axes x-hat / y-hat).
Lattice square(std::size_t Lx, std::size_t Ly, bool pbc);

// 2D triangular lattice (a1 = (1,0,0), a2 = (1/2, sqrt(3)/2, 0)).
Lattice triangular(std::size_t Lx, std::size_t Ly, bool pbc);

// 2D honeycomb lattice; 2-site basis (A, B). Bonds carry `bond_type`
// in {0, 1, 2} corresponding to the three Kitaev colours x / y / z.
Lattice honeycomb(std::size_t Lx, std::size_t Ly, bool pbc);

// 2D kagome lattice; 3-site basis A = (0,0), B = (1/2,0), C = (1/4, sqrt(3)/4)
// on the triangular Bravais lattice.
Lattice kagome(std::size_t Lx, std::size_t Ly, bool pbc);

// 3D pyrochlore lattice (FCC of corner-sharing tetrahedra); 4-site basis.
// Each unit cell hosts 4 sites; total = 4 * Lx * Ly * Lz.
Lattice pyrochlore(std::size_t Lx, std::size_t Ly, std::size_t Lz, bool pbc);

// Build a Lattice from a user-supplied adjacency description (the generic
// escape hatch for clusters no factory above covers).
//
//   * `positions` -- one entry per site (length = `num_sites`)
//   * `nn_pairs`  -- nearest-neighbour edges (i, j), i != j, kept in their
//                    orientation; a pair listed twice (either way) is one bond
//   * `sublattice`-- optional; zero-filled if empty.
// The lattice knows no shells beyond these bonds (nnn_pairs() raises).
Lattice from_neighbor_lists(
    const std::vector<Position>& positions,
    const std::vector<std::pair<std::size_t, std::size_t>>& nn_pairs,
    const std::vector<int>& sublattice = {});

// Read a `cluster.txt`-style file into a Lattice. A "positions" block holds one
// site per line, as "x y", "x y z" or "id x y z" (ids counting from 0 in order);
// an "edges" or "bonds" block holds one "i j" per line. Headers are case-blind
// and may end in ':'. A block may state its length, on its header line or alone
// on its first line, and must then hold exactly that many lines. '#' starts a
// comment line. Anything else is an InvalidRequest naming the line.
Lattice from_cluster_file(const std::string& path);

}  // namespace lattice

}  // namespace ed::input
