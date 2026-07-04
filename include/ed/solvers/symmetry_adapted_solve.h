#pragma once
// =============================================================================
// include/ed/solvers/symmetry_adapted_solve.h
//
// Iterative-capable consumer of the symmetry reduction. Lives in ed_solvers_cpu
// (not ed_symmetry) because it calls the Krylov solver, and ed_solvers_cpu
// depends on ed_symmetry (never the reverse).
//
// This is the seam that makes SYMMETRY (Axis A) orthogonal to METHOD (Axis B):
// each irrep block H_Γ is a MatVecOperator on the PRODUCTION engine
// (CpuMatVecBackend<NonAbelianSymmetryBasisPolicy> over the operator's terms),
// the same engine abelian/Sz use. This consumer solves each block by a dense
// eigensolve (small) or Lanczos/Krylov-Schur (large), selected by block size —
// the reduction is no longer welded to dense, and there is no parallel matvec.
// =============================================================================

#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ed/matvec/matvec.h>              // MatVecOperator, MemorySpace
#include <ed/core/operator.h>              // ed::Operator (complete: subspace_operator.h needs it)
#include <ed/symmetry/symmetry_adapted.h>  // SymAdaptedSpectrum, SymAdaptedBlockOp, ...

namespace ed::solvers {

// Build the non-abelian symmetry-reduced block H_Γ of one irrep as a
// MatVecOperator backed by the PRODUCTION engine: CpuMatVecBackend over a
// NonAbelianSymmetryBasisPolicy (the SAB packed into a ::SymmetrySector, norm=1)
// applied to the operator's own term SoA. This is the SAME engine the abelian/Sz
// reductions use — non-abelian is just another BasisPolicy — replacing the
// bespoke SymAdaptedBlockOp matvec. `n_up >= 0` restricts to the fixed-Sz sector.
[[nodiscard]] std::unique_ptr<ed::matvec::MatVecOperator>
build_nonabelian_sector_matvec(const ::Operator&                    op,
                               const ed::symmetry::GroupIrreps&     gi,
                               const std::vector<std::vector<int>>& max_clique,
                               int                                  irrep_index,
                               int                                  n_sites,
                               int                                  n_up = -1,
                               int                                  sz_parity = -1);

/// Per-block eigensolver method for the symmetry reduction. `Auto` = dense for
/// n_Γ <= dense_max_dim, else Lanczos. The others FORCE that method per block
/// (with a dense fallback for n_Γ <= 2, where iterative methods are degenerate),
/// demonstrating that the reduction composes with any method.
enum class BlockMethod { Auto, Dense, Lanczos, KrylovSchur };

/// Lowest-`k` eigenvalues of H under the (Sz / abelian / non-abelian spatial /
/// combined) symmetry reduction, recombined with each block's d_Γ multiplicity
/// and returned sorted. Each irrep block H_Γ is a NonAbelianSectorMatVec on the
/// production CpuMatVecBackend, solved through the STANDARD MatVecOperator& method
/// overloads selected by `method`. `n_up >= 0` restricts to the fixed-Sz sector.
[[nodiscard]] ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_lowest_eigenvalues(
    const ::Operator&                     op,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   k,
    int                                   n_up          = -1,
    int                                   dense_max_dim = 512,
    BlockMethod                           method        = BlockMethod::Auto,
    int                                   sz_parity     = -1);

// ---------------------------------------------------------------------------
// Full symmetry-reduced consumers, ALL backed by the production engine (each
// per-irrep block H_Γ is materialised by applying NonAbelianSectorMatVec to the
// unit columns — no SymAdaptedBlockOp, no parallel matvec). These replace the
// retired ed::symmetry::symmetry_adapted_{spectrum_terms,thermodynamics,
// ground_state_dssf} / build_symmetry_blocks_packed. `n_up >= 0` -> fixed-Sz.
// ---------------------------------------------------------------------------

/// Full reduced spectrum (every eigenvalue, each ×d_Γ), sorted.
[[nodiscard]] ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_full_spectrum(
    const ::Operator&                     op,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1,
    int                                   sz_parity = -1);

/// Exact canonical thermodynamics from the full reduced spectrum.
[[nodiscard]] ThermodynamicData
symmetry_adapted_thermodynamics(
    const ::Operator&                     op,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    const std::vector<double>&            temperatures,
    int                                   n_up = -1,
    int                                   sz_parity = -1);

/// Ground-state DSSF S(ω) = Σ_n |<n|O|0>|² Lorentzian, summing all d_Γ partners
/// (completeness). H from `op_h`, the observable O from `op_o`.
[[nodiscard]] ed::symmetry::SymDSSFResult
symmetry_adapted_ground_state_dssf(
    const ::Operator&                     op_h,
    const ::Operator&                     op_o,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    double                                omega_min,
    double                                omega_max,
    int                                   n_omega,
    double                                broadening);

/// Per-irrep blocks H_Γ packed column-major (engine-materialised) for the GPU
/// batched eigensolver.
[[nodiscard]] ed::symmetry::SymBlocksPacked
symmetry_adapted_blocks_packed(
    const ::Operator&                     op,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   n_up = -1);

}  // namespace ed::solvers
