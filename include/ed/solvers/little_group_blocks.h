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
    /// d_sigma partners (Jul 2026): given a REP-BASIS eigenvector of this
    /// block (row 0 of irrep sigma -- the W_sigma construction's row),
    /// return the d-1 degenerate partners via the shift projector
    /// P_{j0} = (d/|P|) sum_p conj(D_{j0}(p)) M_p. Empty for d == 1 and
    /// plain blocks. Partners are normalized; certify with your own
    /// residual pin (they are eigenvectors of H_k0 at the same E).
    [[nodiscard]] std::vector<std::vector<std::complex<double>>>
    degenerate_partners(
        const std::vector<std::complex<double>>& u_rep) const;

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

// =============================================================================
// U2b: FLIP-AWARE expansion of a rep-basis vector to computational-basis
// amplitudes. The regeneration arithmetic is the rep matvec's own
// (RepSymmetryBasisPolicy::apply_perm applies the Stage-5b XOR flip masks;
// the characters are extended), so this works identically for raw and
// flip-extended (k, +/-) sectors -- the expansion the orbit-CSR lane
// refuses ("no orbit-CSR form exists for flip-projected sectors") exists
// arithmetically here.
//
// Returns a DENSE 2^n_sites amplitude vector, normalized. Sized for
// moderate N (tests, DSSF sources, observable evaluation); at frontier N
// consume the rep-basis vector directly instead.
// =============================================================================
[[nodiscard]] std::vector<std::complex<double>>
expand_rep_vector_to_computational(
    const ed::symmetry::RepSectorData&        rd,
    const std::vector<std::complex<double>>&  u);

// =============================================================================
// U2b-r2: the lowest-k EIGENPAIRS of the block decomposition, vectors
// included -- the capability whose absence forced eigenvector consumers
// off the projection lane. Per block: dense below the crossover, else
// FullCGS2 Lanczos with kept basis; every returned pair is
// residual-CERTIFIED in the momentum sector's rep basis
// (||H_k0 u - E u|| / ||u|| <= 1e-8 after the isotypic lift) -- an
// uncertified pair is dropped, never returned.
//
// Vectors are returned in the REP BASIS of their momentum sector
// (rows[i].vec over sectors[rows[i].sector_slot]); expand to
// computational amplitudes with expand_rep_vector_to_computational
// (flip-aware). A row with multiplicity > 1 carries ONE representative
// vector -- the star/TR/d_sigma partners are isospectral copies whose
// vectors need the U3 fold transport (not yet built); consumers that
// need every partner's vector must solve those sectors directly.
// =============================================================================
struct LittleGroupVectorRow {
    double                            eigenvalue = 0.0;
    LittleGroupBlockTag               tag;          ///< incl. multiplicity
    std::vector<std::complex<double>> vec;          ///< rep basis, certified
    std::size_t                       sector_slot = 0;  ///< into sectors[]
};

struct LittleGroupVectors {
    std::vector<LittleGroupVectorRow>        rows;     ///< ascending eigenvalue
    std::vector<ed::symmetry::RepSectorData> sectors;  ///< one per touched star
    bool flip_engaged = false;
    bool tr_engaged   = false;
};

[[nodiscard]] LittleGroupVectors
little_group_lowest_vectors(const ::Operator&                    op,
                            const std::vector<std::vector<int>>& abelian_group,
                            const std::vector<std::vector<int>>& residue_perms,
                            int                                  n_sites,
                            int                                  k,
                            const LittleGroupOptions&            opt);

// =============================================================================
// U3: FOLD TRANSPORT for vectors -- the partners a fold skipped. A star
// fold proves spec(k') == spec(k0) by U_p; this applies that U_p (or the
// antiunitary K for a TR pair) to an actual eigenvector:
//
//   U_p |b_i^{k0}> = conj(chi_{k'}(a*)) (N_j / N_i) |b_j^{k'}>
//
// -- the cross-sector generalization of the engine's monomial rows (same
// canonicalization, destination characters/norms). Returns the partner
// sector's RepSectorData plus the transported vector, residual-certified
// by the CALLER's pin (transport itself is exact bookkeeping; a failed
// canonicalization or a non-unit phase throws).
//
// Scope: star members (residue mapping k0 -> k_dst), TR conjugates
// (complex conjugation), and the FLIP MIRROR (n_up_dst = N - n_up,
// k0_dst == k0_src, XOR pre-map; requires [H, prod sigma^x] == 0).
// The d_sigma-internal degenerate copies (sigma_0j isotypic columns)
// remain a documented follow-up.
// =============================================================================
[[nodiscard]] std::pair<ed::symmetry::RepSectorData,
                        std::vector<std::complex<double>>>
little_group_transport(const ::Operator&                    op,
                       const std::vector<std::vector<int>>& abelian_group,
                       const std::vector<std::vector<int>>& residue_perms,
                       int                                  n_sites,
                       int                                  k0_src,
                       int                                  k0_dst,
                       const std::vector<std::complex<double>>& vec,
                       const LittleGroupOptions&            opt,
                       int                                  n_up_dst = -1);

}  // namespace ed::solvers
