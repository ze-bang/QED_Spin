#pragma once
// =============================================================================
// include/ed/api/symmetry_helpers.h
//
// `ed::find_symmetries(...)` -- thin C++ wrapper around
// `ed::sym::group_from_generators` (plus the named presets in
// `ed/symmetry/group.h`) that mirrors the Python `qed.find_symmetries`
// surface for the cases where the user supplies the symmetry generators
// by hand.
//
// What this header does NOT do (deliberate scope cut from the
// "mirror examples" plan):
//
//   * No graph-automorphism discovery. Python's `qed.find_symmetries`
//     can detect the full symmetry group of a lattice + Hamiltonian by
//     reading `connectivity_graph.json` and running nauty. The C++
//     side keeps generator-set construction; the automorphism step
//     stays Python-only. C++ callers who need automorphism discovery
//     should build the `automorphism_results/` directory once via
//     Python and then pass a `DirectoryPath` source to
//     `ed::make_operator(spec)` with `streaming_symmetry=true`.
//
// The wrapper exists so that small C++ examples (the "twin" of a Python
// example that uses spatial symmetry) can stay self-contained instead
// of either redoing the BFS by hand or shelling out to Python.
// =============================================================================

#include <string_view>
#include <vector>

#include <ed/core/construct_ham.h>   // SymmetryGroupInfo
#include <ed/symmetry/group.h>       // ed::sym::group_from_generators

namespace ed {

/// Build the canonical `SymmetryGroupInfo` for one of the named 1D
/// presets, mirroring Python's `qed.find_symmetries(..., generators="auto")`
/// path for plain ring lattices.
///
/// Accepted preset names:
///   * "translation"            => translation_group_1d(n_sites)
///   * "translation+reflection" => translation_group_with_reflection_1d(n_sites)
///   * "reflection"             => Z2 reflection_1d(n_sites)
///   * "identity"               => trivial group (no symmetry sectors)
///
/// Throws `std::invalid_argument` on unknown preset.
[[nodiscard]] SymmetryGroupInfo
find_symmetries(int n_sites, std::string_view preset = "translation");

/// Free-form overload: hand the wrapper a list of permutation generators
/// and (optionally) an explicit list of sector quantum-numbers.
///
/// Internally calls `ed::sym::group_from_generators` with full
/// validation. Use this for arbitrary 1D / 2D point groups that the
/// caller can describe as site permutations.
[[nodiscard]] SymmetryGroupInfo
find_symmetries(int n_sites,
                std::vector<ed::sym::Permutation> generators,
                std::vector<std::vector<int>> sector_quantum_numbers = {});

}  // namespace ed
