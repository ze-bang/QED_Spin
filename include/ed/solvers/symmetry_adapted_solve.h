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
                               int                                  n_up = -1);

// ---------------------------------------------------------------------------
// The non-abelian symmetry-reduced block H_Γ as a first-class MatVecOperator.
// This is the unification seam (campaign Stage D): once the reduced block is a
// MatVecOperator, it rides EVERY method that takes one (lanczos, block_lanczos,
// full_diagonalization, krylov_schur, FTLM, cf-DSSF) — so the symmetry reduction
// (Axis A) is orthogonal to the eigensolver (Axis B) at the operator boundary,
// without forcing the SAB through the abelian orbit-representative gather kernel.
// Host-resident; `apply` is the reduced matvec from ed_symmetry's SymAdaptedBlockOp.
// ---------------------------------------------------------------------------
class SymAdaptedBlockMatVec final : public ed::matvec::MatVecOperator {
public:
    explicit SymAdaptedBlockMatVec(ed::symmetry::SymAdaptedBlockOp op)
        : op_(std::move(op)) {}

    void apply(const std::complex<double>* in,
               std::complex<double>*       out,
               std::size_t /*size*/) const override { op_.apply(in, out); }

    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(op_.dim());
    }
    [[nodiscard]] ed::matvec::MemorySpace memory_space() const override {
        return ed::matvec::MemorySpace::Host;
    }
    [[nodiscard]] bool is_hermitian() const override { return true; }
    [[nodiscard]] std::string description() const override {
        return "SymAdaptedBlock(H_Gamma)";
    }

private:
    ed::symmetry::SymAdaptedBlockOp op_;
};

/// Per-block eigensolver method for the symmetry reduction. `Auto` = dense for
/// n_Γ <= dense_max_dim, else Lanczos. The others FORCE that method per block
/// (with a dense fallback for n_Γ <= 2, where iterative methods are degenerate),
/// demonstrating that the reduction composes with any method.
enum class BlockMethod { Auto, Dense, Lanczos, KrylovSchur };

/// Lowest-`k` eigenvalues of H under the (Sz / abelian / non-abelian spatial /
/// combined) symmetry reduction, recombined with each block's d_Γ multiplicity
/// and returned sorted. Each irrep block H_Γ is wrapped as a SymAdaptedBlockMatVec
/// and solved through the STANDARD MatVecOperator& method overloads selected by
/// `method`. `n_up >= 0` restricts to the fixed-Sz sector.
[[nodiscard]] ed::symmetry::SymAdaptedSpectrum
symmetry_adapted_lowest_eigenvalues(
    const ::Operator&                     op,
    const ed::symmetry::GroupIrreps&      gi,
    const std::vector<std::vector<int>>&  max_clique,
    int                                   n_sites,
    int                                   k,
    int                                   n_up          = -1,
    int                                   dense_max_dim = 512,
    BlockMethod                           method        = BlockMethod::Auto);

}  // namespace ed::solvers
