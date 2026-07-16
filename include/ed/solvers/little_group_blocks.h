#pragma once
// =============================================================================
// include/ed/solvers/little_group_blocks.h
//
// Unified-stack U1a: the little-group engine's (momentum sector x isotypic
// irrep) blocks as FIRST-CLASS, OWNED handles -- the surface that lets every
// verb (eigenvalues today; thermal sampling, U1b; eigenvectors, U2a) consume
// the same maximal block decomposition instead of re-deriving it per verb.
//
// A LittleGroupBlock wraps ONE diagonal block of H:
//
//   * a PROJECTED block  W_sigma^dag H_{k0} W_sigma   (dim = m_sigma), or
//   * the PLAIN k0 block H_{k0}                        (dim = #reps(k0))
//     when the star declined projection (trivial little co-group, projective
//     factor system, failed monomial probe, incomplete covering -- the
//     engine's graceful-fallback floor).
//
// The handle exposes the block as an `ed::LinearOperator`, so anything the
// orchestrator can drive (Lanczos, dense eigensolve, FTLM/LTLM/TPQ sampling
// kernels via ed::workflows::thermal) runs inside the reduced dimension with
// no kernel changes. Ownership is by shared_ptr: all irrep blocks of one star
// co-own their star's matrix-free H_{k0} (RepSectorMatVec), so a caller may
// keep any subset of blocks alive independently of the set.
//
// Memory note: a LittleGroupBlockSet holds every star's RepSectorData
// simultaneously (O(#reps) per star). At frontier N prefer consuming stars
// one at a time (the engine-internal per-star builder does exactly that);
// this set surface is sized for the verb loops and tests at small-to-mid N.
//
// The concrete types behind the handle (RepSectorMatVec, SparseColumns,
// ProjectedBlockOp, Monomial) stay PRIVATE to little_group_solve.cpp -- this
// header is deliberately pimpl so the basis layer keeps zero new surface.
// =============================================================================

#include <cstdint>
#include <memory>
#include <vector>

#include <ed/orchestrator.h>                // ed::workflows::ThermalOptions (U1b)
#include <ed/solvers/little_group_solve.h>  // LittleGroupOptions, LittleGroupSpectrum

namespace ed {
class LinearOperator;  // include/ed/core/linear_operator.h
}
namespace ed::symmetry {
struct RepSectorData;  // include/ed/symmetry/rep_sector_data.h
}

namespace ed::solvers {

// -----------------------------------------------------------------------------
// One block's quantum-number tag. `k0` / `k_raw` are ENGINE-INTERNAL irrep
// indices -- k_raw is NOT the physical momentum; decode momenta through
// `LittleGroupSpectrum::irrep_characters` (chi_k of the abelian generators),
// exactly like decode_star_for_sector does on the Python side.
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
// shared by every copy, so `gpu_engaged()` is truthful only after applies ran.
// One in-flight apply per DISTINCT block at a time (the internal scratch is
// per-block); concurrent applies on different blocks of the same star are
// safe -- the shared H_{k0} apply path is re-entrant.
// -----------------------------------------------------------------------------
class LittleGroupBlock {
public:
    struct Impl;  // defined in little_group_solve.cpp
    explicit LittleGroupBlock(std::shared_ptr<Impl> impl);
    ~LittleGroupBlock();
    LittleGroupBlock(const LittleGroupBlock&);
    LittleGroupBlock& operator=(const LittleGroupBlock&);
    LittleGroupBlock(LittleGroupBlock&&) noexcept;
    LittleGroupBlock& operator=(LittleGroupBlock&&) noexcept;

    [[nodiscard]] const LittleGroupBlockTag& tag() const noexcept;
    /// The block as a LinearOperator (projected sandwich or plain H_{k0}).
    [[nodiscard]] ed::LinearOperator& op() const noexcept;
    /// The star's momentum-sector rep data (shared across the star's blocks).
    [[nodiscard]] const ed::symmetry::RepSectorData& rep_data() const noexcept;
    /// True when this is an isotypic W^dag H W block (irrep >= 0).
    [[nodiscard]] bool projected() const noexcept;
    /// Truthful post-apply report from the star's H_{k0} matvec.
    [[nodiscard]] bool gpu_engaged() const noexcept;
    /// U2a: lift a block-coordinate vector to the momentum sector's rep
    /// basis, u = W_sigma v (identity copy for plain blocks). `v` must
    /// have tag().dim entries; the result has rep_data().reps.size().
    /// W's columns are orthonormal (SVD), so norms are preserved.
    [[nodiscard]] std::vector<std::complex<double>>
    lift_to_rep(const std::complex<double>* v) const;

private:
    std::shared_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------------
// Every block of every star of one (subspace, group) decomposition, in the
// engine's canonical row order (stars ascending by k0; within a star, irreps
// ascending; TR-folded later partners absent -- their weight is in the earlier
// partner's multiplicity; a declined star contributes its single plain block).
// `meta` carries the star table / irrep characters / flip+tr flags exactly as
// the spectra do (its eigenvalue fields stay empty; gpu_engaged reflects
// construction only -- blocks report their own truthfully after applies).
// -----------------------------------------------------------------------------
struct LittleGroupBlockSet {
    std::vector<LittleGroupBlock> blocks;
    LittleGroupSpectrum           meta;
};

/// Build the block decomposition WITHOUT solving anything. Honours
/// LittleGroupOptions the same way the spectrum entry points do
/// (n_up / sz_parity / spin_flip / time_reversal / only_k0 / only_irrep;
/// the ED_SYM_LG_ONLY_K0 env override wins, "plan" is ignored here --
/// this factory never solves).
[[nodiscard]] LittleGroupBlockSet
build_little_group_blocks(const ::Operator&                    op,
                          const std::vector<std::vector<int>>& abelian_group,
                          const std::vector<std::vector<int>>& residue_perms,
                          int                                  n_sites,
                          const LittleGroupOptions&            opt);

// =============================================================================
// U1b: SAMPLED thermodynamics inside the projected blocks.
//
// Where little_group_thermodynamics (little_group_solve.h) EXACT-diagonalizes
// every block (the point_group='full' contract), this runs the caller's
// sampling method (FTLM / LTLM / mTPQ / OFTLM) per block through
// ed::workflows::thermal(block.op(), ...) -- same kernels, same mem_guard,
// same small-dim exact fallback -- and Z-recombines with each block's
// spectral multiplicity folded in as a free-energy shift
// F_b[t] -= T[t] * ln(m_b)  (exactly Z_b -> m_b * Z_b).
//
// KPM_DOS is REFUSED (std::invalid_argument): its deliverable is one
// full-spectrum DOS; per-block sub-DOS on different Chebyshev grids cannot
// recombine into it. Route KPM to the abelian lane.
// =============================================================================
struct LittleGroupThermalResult {
    ThermodynamicData                 thermo;      ///< combined across blocks
    std::vector<LittleGroupBlockTag>  block_tags;  ///< one per solved block
    /// Per-block thermodynamics BEFORE the multiplicity F-shift
    /// (diagnostics / tests), parallel to block_tags.
    std::vector<ThermodynamicData>    per_block;
    /// The weight actually folded into each block's Z: tag.multiplicity
    /// times the Sz flip-transport mirror factor (2 for a mirrored
    /// n_up != N/2 subspace in the unnamed-Sz sweep, else 1).
    std::vector<std::uint64_t>        weights;
    double ground_state_energy = 0.0; ///< min per-block GS estimate
    bool   projected_any       = false;
    bool   gpu_engaged         = false;
};

[[nodiscard]] LittleGroupThermalResult
little_group_thermal(const ::Operator&                    op,
                     const std::vector<std::vector<int>>& abelian_group,
                     const std::vector<std::vector<int>>& residue_perms,
                     int                                  n_sites,
                     ed::workflows::ThermalOptions        topts,
                     const LittleGroupOptions&            opt);

}  // namespace ed::solvers
