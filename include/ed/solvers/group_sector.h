#pragma once
// =============================================================================
// Group-sector lane: a symmetric basis under the FULL little group of one block.
//
// The little-group lane (little_group_blocks.h) builds its rep basis under the ABELIAN part only (translations x spin
// flip) and projects point-group irreps afterwards through an isotypic basis W: every block then carries the whole
// k-sector (3.8e8 states at N = 36) and each apply is W v -> H -> W^dag u. When the wanted irrep is ONE-dimensional
// (A1/A2/B1/B2 of C6v and C2v, A1/A2 of C3v, the m-irreps of C6/C3, any irrep of an abelian group), the classic
// symmetrised basis does the same job directly:
//     G = { (t, p, f) }  (translations x little co-group x {1, flip}),  chi a 1-dim representation of G,
//     |r> = N_r sum_g chi(g)^* g |s_r>,  one rep per G-orbit with nonzero norm,
// so the block dimension is C(N, n_up) / |G| (3.15e7 for a Gamma A-irrep of the 36-site C6v cluster, 12x smaller) and
// the reduced CSR, the apply and the stored vectors shrink with it. Two-dimensional irreps (E of C6v / C3v) are reached
// one partner at a time through an abelian subgroup (C6 m = +-1, +-2; C3 m = 1), whose 1-dim irreps split them.
//
// Nothing here is new machinery: the orbit table is build_orbit_table_fixed_sz_streaming (any permutation group,
// flip-extended), the norms are projected_norm_sq_stab, the operator is the rep-sector matvec + reduced CSR, and the
// eigensolver is the little-group block solver. RepSectorData / sector dicts are the same objects the rest of the
// engine (rep_sector_matrix_elements, ...) already consumes.
// =============================================================================

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include <ed/symmetry/rep_sector_data.h>

class Operator;  // include/ed/core/operator.h

namespace ed::solvers {

/// Rep basis of the n_up sector under the group `perms` (site permutations, closed under composition; element order
/// = character order) optionally extended by the global spin flip (elements [g..., g*F...], n_up must be N/2), in the
/// 1-dim representation `characters` (length |G| or 2|G| with flip). Reps with vanishing projected norm are dropped.
[[nodiscard]] ed::symmetry::RepSectorData
build_group_sector(const std::vector<std::vector<int>>&       perms,
                   int                                        n_sites,
                   int                                        n_up,
                   bool                                       flip,
                   const std::vector<std::complex<double>>&   characters);

struct GroupSectorSolveOptions {
    int  levels        = 3;     ///< eigenpairs wanted (>= 2 keeps the Krylov-Schur lane)
    int  block_size    = 1;     ///< Krylov-Schur block size
    int  dense_max_dim = 256;   ///< dense diagonalisation below this dimension
    bool return_vectors = true;
};

struct GroupSectorSolveResult {
    std::vector<double>                            energies;
    std::vector<double>                            residuals;   ///< ||H v - E v|| per returned pair
    std::vector<std::vector<std::complex<double>>> vectors;     ///< in the group-sector basis
    bool          converged = false;
    std::uint64_t dim       = 0;
    double        t_setup   = 0.0;   ///< operator construction (s); the reduced CSR is built lazily in the solve
    double        t_solve   = 0.0;   ///< eigensolve incl. the CSR build (s)
};

/// Lowest `opt.levels` eigenpairs of `op` restricted to the group sector `rd`.
[[nodiscard]] GroupSectorSolveResult
solve_group_sector(const ::Operator&                                     op,
                   std::shared_ptr<const ed::symmetry::RepSectorData>    rd,
                   const GroupSectorSolveOptions&                        opt);

/// Re-express a state given in the group sector `src` (group G) in the sector `dst` of a SUBGROUP H of G (same sites,
/// n_up, flip convention; dst characters = the restriction of the src ones). Used to pair states of different G
/// (e.g. a C6v A-irrep with a C6 E-partner) in rep_sector_matrix_elements. `conjugate` selects the phase convention
/// (the engine's projection accumulates conj(chi)); conjugate = true is correct (group_convert_test.py: with complex
/// characters only it reproduces <H>). The result keeps the norm (rescaled by sqrt(|H|/|G|)).
[[nodiscard]] std::vector<std::complex<double>>
convert_group_vector(const std::vector<std::complex<double>>& v,
                     const ed::symmetry::RepSectorData&        src,
                     const ed::symmetry::RepSectorData&        dst,
                     bool                                      conjugate);

}  // namespace ed::solvers
