#pragma once
// =============================================================================
// include/ed/solvers/little_group_blocks.h
//
// The little-group engine's (momentum sector x isotypic irrep) blocks as
// OWNED handles, consumed by the ed::sectors verbs (include/ed/sectors/).
//
// A LittleGroupBlock wraps ONE diagonal block of H:
//
//   * a PROJECTED block  W_sigma^dag H_{k0} W_sigma   (dim = m_sigma), or
//   * a GROUP-SECTOR block (1-dim irrep solved in the rep basis of the full
//     little group), or
//   * the PLAIN k0 block H_{k0}                        (dim = #reps(k0))
//     when the star declined projection (trivial little co-group, projective
//     factor system, failed monomial probe, incomplete covering -- the
//     engine's graceful-fallback floor).
//
// Each block is an `ed::LinearOperator` inside the engine, so anything the
// orchestrator can drive (Lanczos, dense eigensolve, FTLM / TPQ sampling
// kernels via ed::workflows::thermal) runs inside the reduced dimension with
// no kernel changes; the public handle lifts block vectors back to the
// momentum-sector rep basis. Ownership is by shared_ptr: all irrep blocks of
// one star co-own their star's matrix-free H_{k0} (RepSectorMatVec).
//
// The concrete types behind the handle (RepSectorMatVec, SparseColumns,
// ProjectedBlockOp, Monomial) stay PRIVATE to the engine (src/solvers/little_group/lg_internal.h) -- this
// header is deliberately pimpl so the basis layer exposes none of them.
// =============================================================================

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include <ed/solvers/little_group_solve.h>  // LittleGroupOptions, LittleGroupStarInfo

namespace ed {
class LinearOperator;  // include/ed/core/linear_operator.h
}
namespace ed::symmetry {
struct RepSectorData;  // include/ed/symmetry/rep_sector_data.h
}

namespace ed::solvers {

// -----------------------------------------------------------------------------
// One block's quantum-number tag. `k0` / `k_raw` are ENGINE-INTERNAL irrep
// indices -- k_raw is NOT the physical momentum; decode momenta through the
// abelian irrep characters chi_k (EngineContext::giA in the engine).
// -----------------------------------------------------------------------------
struct LittleGroupBlockTag {
    int n_up      = -1;   ///< fixed-Sz subspace (-1 = none)
    int sz_parity = -1;   ///< Sz-parity half (-1 = none)
    int k0        = -1;   ///< extended irrep index (k_raw + flip_parity * n_irr_raw)
    int k_raw     = -1;   ///< raw abelian irrep index (NOT the momentum)
    int flip_parity = -1; ///< 0 = (k,+), 1 = (k,-), -1 = flip axis off
    int irrep     = -1;   ///< little-co-group irrep index; -1 = plain floor block
    int irrep_dim = 1;    ///< d_sigma
    int star_size = 1;    ///< |star| (residue orbit of momenta)
    bool tr_folded = false; ///< sigma* partner folded in (multiplicity doubled)

    std::uint64_t dim          = 0; ///< block operator dimension (m_sigma or dim_k0)
    /// How many times this block's spectrum appears in the subspace:
    /// star_size * irrep_dim * (tr_folded ? 2 : 1). NEVER includes the Sz
    /// flip-transport mirror -- that axis lives in the subspace sweep.
    std::uint64_t multiplicity = 1;
};

// -----------------------------------------------------------------------------
// Owned handle to one diagonal block. Copyable (shared ownership); the
// underlying operator is lazily materialised state (reduced CSR / GPU mirror)
// shared by every copy.
// One in-flight apply per DISTINCT block at a time (the internal scratch is
// per-block); concurrent applies on different blocks of the same star are
// safe -- the shared H_{k0} apply path is re-entrant.
// -----------------------------------------------------------------------------
class LittleGroupBlock {
public:
    struct Impl;  // defined in src/solvers/little_group/lg_internal.h
    explicit LittleGroupBlock(std::shared_ptr<Impl> impl);
    ~LittleGroupBlock();
    LittleGroupBlock(const LittleGroupBlock&);
    LittleGroupBlock& operator=(const LittleGroupBlock&);
    LittleGroupBlock(LittleGroupBlock&&) noexcept;
    LittleGroupBlock& operator=(LittleGroupBlock&&) noexcept;

    /// Lift a block-coordinate vector to the momentum sector's rep
    /// basis, u = W_sigma v (identity copy for plain blocks). `v` must
    /// have the block dimension; the result has one entry per momentum-sector rep.
    /// W's columns are orthonormal (SVD), so norms are preserved.
    [[nodiscard]] std::vector<std::complex<double>>
    lift_to_rep(const std::complex<double>* v) const;

private:
    std::shared_ptr<Impl> impl_;
};

}  // namespace ed::solvers
