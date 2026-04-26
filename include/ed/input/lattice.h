// =============================================================================
// include/ed/input/lattice.h
//
// Standalone lattice geometry generators that **replace** the per-helper
// Python files in `python/edlib/helper_*.py`. Every lattice the legacy
// `edlib` package shipped is reachable from one factory:
//
//   * chain     -- 1D chain (PBC / OBC)
//   * square    -- 2D square lattice
//   * triangular-- 2D triangular lattice
//   * honeycomb -- 2D honeycomb (2-site basis)
//   * kagome    -- 2D kagome (3-site basis)
//   * pyrochlore-- 3D pyrochlore (4-site basis, FCC)
//   * from_neighbor_lists -- build from an arbitrary user-supplied
//                            adjacency list (covers `helper_cluster.py`).
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

    // Nearest-neighbour bonds, canonicalised (i < j).
    std::vector<Bond> nn_bonds;

    // Optional next-nearest and third-nearest bond lists. Some helpers
    // (Kitaev honeycomb, kagome BFG) need them.
    std::vector<Bond> nnn_bonds;
    std::vector<Bond> nnnn_bonds;

    // Lattice basis vectors (`{a1, a2, a3}`); unused entries are zeroed out.
    std::array<Position, 3> lattice_vectors{};

    // True iff the lattice was generated with periodic boundary conditions.
    bool pbc = false;

    // Free-form lattice label, e.g. "pyrochlore[2x2x2 PBC]". Useful when
    // inspecting / logging / tagging output directories.
    std::string label;

    // Helpers ---------------------------------------------------------------

    // Returns a sorted, deduplicated copy of `nn_bonds` with i<j canonical
    // orientation. Useful when feeding the bond list into Hamiltonian
    // shortcuts that need each edge counted once.
    std::vector<std::pair<std::size_t, std::size_t>> nn_pairs() const;
    std::vector<std::pair<std::size_t, std::size_t>> nnn_pairs() const;
    std::vector<std::pair<std::size_t, std::size_t>> nnnn_pairs() const;

    // Convenience: sites 0..num_sites-1 as a vector.
    std::vector<std::size_t> all_sites() const;
};

// =============================================================================
// Factory functions
// =============================================================================

namespace lattice {

// 1D chain of `length` sites along x-hat.
Lattice chain(std::size_t length, bool pbc);

// 2D square lattice (`Lx` x `Ly`, axes x-hat / y-hat).
Lattice square(std::size_t Lx, std::size_t Ly, bool pbc);

// 2D triangular lattice (a1 = (1,0,0), a2 = (1/2, sqrt(3)/2, 0)).
Lattice triangular(std::size_t Lx, std::size_t Ly, bool pbc);

// 2D honeycomb lattice; 2-site basis (A, B). Bonds carry `bond_type`
// in {0, 1, 2} corresponding to the three Kitaev colours x / y / z.
Lattice honeycomb(std::size_t Lx, std::size_t Ly, bool pbc);

// 2D kagome lattice; 3-site basis. NN within the triangle, NNN across
// hexagons. Used by helper_kagome_bfg.
Lattice kagome(std::size_t Lx, std::size_t Ly, bool pbc);

// 3D pyrochlore lattice (FCC of corner-sharing tetrahedra); 4-site basis.
// Each unit cell hosts 4 sites; total = 4 * Lx * Ly * Lz.
Lattice pyrochlore(std::size_t Lx, std::size_t Ly, std::size_t Lz, bool pbc);

// Build a Lattice from a user-supplied adjacency description. This is the
// generic escape hatch that `helper_cluster.py` historically filled.
//
//   * `positions` -- one entry per site (length = `num_sites`)
//   * `nn_pairs`  -- nearest-neighbour edges (each `(i, j)` with i != j)
//   * `sublattice`-- optional; zero-filled if empty.
Lattice from_neighbor_lists(
    const std::vector<Position>& positions,
    const std::vector<std::pair<std::size_t, std::size_t>>& nn_pairs,
    const std::vector<int>& sublattice = {});

// Read a legacy `cluster.txt`-style file (positions + edges block) into
// a Lattice. Recognises the format used by `helper_cluster.py` and
// `helper_cluster_triangular.py`.
Lattice from_cluster_file(const std::string& path);

}  // namespace lattice

}  // namespace ed::input
