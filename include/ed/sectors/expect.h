#pragma once
// =============================================================================
// include/ed/sectors/expect.h
//
// Expectation values and matrix elements over the eigenpairs eigs() returns.
//
// expect(): <O> of each level, averaged over the level's symmetry multiplet -- the value
// that does not depend on which partner the solver happened to return. The average is
// <psi| Obar |psi>, where Obar is O averaged over the symmetry group (the site
// permutations, and the Sz flip where it folds the level); complex conjugation, which
// pairs a level with its time-reversed partner, is averaged in the result. Obar commutes
// with every symmetry, so it acts inside the level's own block basis with the same kernels
// as H -- nothing is expanded to the Sz sector or the full space. Terms of O that change
// Sz (or its parity, in a parity sector) have no diagonal element and are dropped. With a
// total-spin restriction, O must be SU(2) invariant (the value is then the same for every
// member of the spin multiplet).
//
// matrix_element(): <v_i| O |v_j> between two returned vectors -- the partners the solver
// chose. O is arbitrary: it may change Sz and break every symmetry.
// =============================================================================

#include <ed/sectors/sectors.h>

#include <vector>

namespace ed::sectors {

/// values[level][op] for every level of `r`; each level must carry a vector (eigs with
/// vectors = true). `s` is the Spec `r` was computed with.
[[nodiscard]] std::vector<std::vector<Complex>>
expect(const EigsResult& r, const Spec& s, const std::vector<const ::Operator*>& ops);

/// <v_i| O |v_j> for levels i and j of `r` (one- and two-body terms).
[[nodiscard]] Complex matrix_element(const EigsResult& r, const ::Operator& O,
                                     std::size_t i, std::size_t j);

}  // namespace ed::sectors
