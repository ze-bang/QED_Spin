#pragma once
// =============================================================================
// include/ed/solvers/symmetry_adapted_solve.h
//
// Iterative-capable consumer of the symmetry reduction. Lives in ed_solvers_cpu
// (not ed_symmetry) because it calls the Krylov solver, and ed_solvers_cpu
// depends on ed_symmetry (never the reverse).
//
// This is the seam that makes SYMMETRY (Axis A) orthogonal to METHOD (Axis B):
// the reduction produces a `SymAdaptedBlockOp` per irrep (the reduced matvec
// H_Γ·v); this consumer solves each block by EITHER a dense eigensolve (small
// blocks) OR Lanczos on that matvec (large blocks), selected by block size.
// The non-abelian reduction is therefore no longer welded to dense.
// =============================================================================

#include <vector>

#include <ed/symmetry/symmetry_adapted.h>  // SymAdaptedSpectrum, ConnectFn, GroupIrreps

namespace ed::solvers {

/// Lowest-`k` eigenvalues of H under the (Sz / abelian / non-abelian spatial /
/// combined) symmetry reduction, recombined with each block's d_Γ multiplicity
/// and returned sorted. For every irrep block H_Γ of reduced dimension n_Γ:
///   * n_Γ <= dense_max_dim  -> dense eigensolve (optimal for small blocks);
///   * n_Γ >  dense_max_dim  -> Lanczos on SymAdaptedBlockOp::apply (the reduced
///                              matvec), so large non-abelian blocks are tractable.
/// Both paths consume the SAME reduced matvec. `n_up >= 0` restricts to the
/// fixed-Sz sector. `k` is the number of lowest eigenvalues per block.
[[nodiscard]] ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_lowest_eigenvalues(
    const ed::symmetry::ConnectFn&        connect,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   k,
    int                                   n_up         = -1,
    int                                   dense_max_dim = 512);

}  // namespace ed::solvers
