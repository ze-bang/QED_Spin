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
// Sz (or its parity, in a parity sector) have no diagonal element and are dropped. Under a
// total-spin restriction with an SU(2)-symmetric H a level is a whole spin multiplet, and an O
// that is not SU(2) invariant enters through its SU(2)-scalar part (ed::ops::su2_scalar_part),
// whose expectation is the multiplet average; such an O with a term on more than 5 sites raises
// ed::Unsupported. In a uniform field every member is a level of its own and O enters as it is.
//
// matrix_element(): <v_i| O |v_j> between two returned vectors -- the partners the solver
// chose. O is arbitrary: it may change Sz and break every symmetry.
//
// evaluate(): <X> for operators and <A_a^dag B_b> for every pair of operator lists, averaged as
// expect() averages them, from one sweep. The products are formed exactly in the spin-1/2
// algebra, and the averaged operators are deduplicated before the sweep: pairs related by a
// symmetry average to the same operator (up to a factor), so a family of n operators costs about
// n^2 / |G| operators, not n^2. expect() itself goes through the same evaluator.
// =============================================================================

#include <ed/sectors/sectors.h>

#include <vector>

namespace ed::sectors {

/// values[level][op] for every level of `r`; each level must carry a vector (eigs with
/// vectors = true). `s` is the Spec `r` was computed with.
[[nodiscard]] std::vector<std::vector<Complex>> expect(const EigsResult& r, const Spec& s,
                                                       const std::vector<const ::Operator*>& ops);

/// <v_i| O |v_j> for levels i and j of `r`; O may hold terms on any number of sites. A level
/// index out of range throws std::out_of_range; O on another number of sites or with a
/// non-finite coefficient, or a level without a vector, throws ed::InvalidRequest.
[[nodiscard]] Complex matrix_element(const EigsResult& r, const ::Operator& O, std::size_t i, std::size_t j);

/// Two operator lists whose pair values <A_a^dag B_b> evaluate() returns.
struct PairRequest {
    std::vector<const ::Operator*> A;
    std::vector<const ::Operator*> B;
};

/// Many quantities from one sweep: out[level] lists <X> for every X in `singles`, then, for each
/// pair request in order, <A_a^dag B_b> (B_b acts first) at a * B.size() + b -- each multiplet-
/// averaged as expect() averages it. Each level must carry a vector.
[[nodiscard]] std::vector<std::vector<Complex>> evaluate(const EigsResult& r, const Spec& s,
                                                         const std::vector<const ::Operator*>& singles,
                                                         const std::vector<PairRequest>& pair_requests);

/// The evaluator behind expect() and evaluate(): out[level][x] = the multiplet average of the
/// canonical operator xs[x] in each level.
[[nodiscard]] std::vector<std::vector<Complex>> averaged_values(const EigsResult& r, const Spec& s,
                                                                const std::vector<ed::ops::MaskedOperator>& xs);

}  // namespace ed::sectors
