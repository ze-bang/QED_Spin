#pragma once
// =============================================================================
// include/ed/solvers/little_group_solve.h
//
// Stage 7 (SymmetryEngine v2): FACTORIZED non-abelian reduction via little
// co-groups -- the scalable alternative to the monolithic SAB engine
// (`symmetry_adapted_solve.h`, which stores O(d·dim) SAB amplitudes and is
// capped at moderate N).
//
// Structure exploited: G = A ⋊ P with A the abelian clique (translations)
// and P the retained residue (point-group coset representatives,
// `GeneratorSet.star_perms`). The abelian irreps k of A are the momentum
// sectors; residues permute them (χ_k → χ_k^p). Per STAR (residue orbit of
// momenta):
//
//   1. Solve only the star representative k0; every member contributes the
//      same spectrum (multiplicity |star| -- the proven Stage-7a folding).
//   2. The little co-group P_k0 = {p : p·k0 = k0} acts WITHIN the k0
//      sector. On the matrix-free rep basis {|ψ^{k0}_i⟩} its action is a
//      MONOMIAL matrix M_p (index permutation + unit phase): with
//      U_p|r_i⟩ = U_b|r_j⟩ for b ∈ A,
//
//        U_p |ψ^{k0}_i⟩ = χ_{k0}(b) · |ψ^{k0}_j⟩.
//
//   3. The abstract little co-group (elements close modulo A; the factor
//      system ω(p,q) = χ_{k0}(a_{pq}) must be trivial -- checked) is
//      decomposed with `decompose_irreps_tables`; per irrep σ a sparse
//      isotypic basis W_σ over the k0 REP INDICES (SVD per index-orbit,
//      the same construction as `build_sab_partition0` one level up).
//   4. Block = W_σ† H_{k0} W_σ with H_{k0} the MATRIX-FREE rep-kernel
//      matvec -- memory O(#reps(k0)), never O(2^N). Eigenvalues carry
//      multiplicity |star| × d_σ.
//
// Robustness contract: every reduction step degrades GRACEFULLY. A residue
// that does not normalise A, a nontrivial (projective) factor system, a
// monomial action that fails the numerical [M_p, H] = 0 check -- each just
// falls back to solving the plain k0 block (correct, merely less reduced).
// Correctness never depends on the little-group bookkeeping.
// =============================================================================

#include <ed/core/operator.h>
#include <ed/core/results.h>   // ThermodynamicData
#include <ed/matvec/matvec.h>              // 9d: MatVecOperator factory
#include <ed/symmetry/rep_sector_data.h>   // 9d: RepSectorData

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace ed::solvers {

struct LittleGroupOptions {
    int  n_up          = -1;   ///< fixed-Sz subspace (-1 = none)
    int  sz_parity     = -1;   ///< Sz-parity half (-1 = none; excludes n_up)
    int  dense_max_dim = 64;   ///< per-block dense/Lanczos crossover (lowest-k path)
    bool use_gpu       = false;///< full-spectrum path: ONE batched cuSOLVER
                               ///< eigensolve over all packed blocks (WITH_CUDA)
    bool verbose       = false;
    /// Stage 9a: spin-flip Z2 through the ABELIAN factor (A' = A x Z2 -- the
    /// flip commutes with every site permutation, so it never belongs to the
    /// little co-group). SymToggle convention: -1 auto (engage when
    /// [H, prod sigma^x] = 0 AND the subspace is flip-invariant: n_up = N/2,
    /// parity with N even, or the full space; ED_SYM_LG_FLIP=0 vetoes),
    /// 0 off, 1 require (throws when the symmetry or admissibility is absent).
    int  spin_flip     = -1;
    /// Stage 9b: time-reversal folding (star-level k <-> conj(k) merge +
    /// conjugate-irrep pairing). Same SymToggle convention; dormant until 9b.
    int  time_reversal = -1;
    /// Solve ONLY these star representatives (extended irrep indices -- the
    /// same ``k0`` that ``LittleGroupStarInfo::k0`` reports). Empty = every
    /// star, the default.
    ///
    /// This is the programmatic form of the ED_SYM_LG_ONLY_K0 job-splitting
    /// env var, added so a caller can NAME a momentum block and have the
    /// engine do only that block's work -- previously naming a sector forced
    /// the whole call off the projection lane and onto the abelian one, i.e.
    /// asking for a smaller block silently bought a BIGGER one (the
    /// (n_up,k) sector instead of (n_up,k,irrep)).
    ///
    /// Note a star is the solve unit: members of one star are isospectral by
    /// construction (the point-group residue maps them onto each other), so
    /// restricting to a star representative answers for every member of that
    /// star. The env var still wins when set, so existing job scripts keep
    /// working.
    std::vector<int> only_k0;
    /// Build every star's sector and report it (``stars`` + the full
    /// ``irrep_characters`` table), solving NOTHING. The programmatic form of
    /// ED_SYM_LG_ONLY_K0="plan".
    ///
    /// This is what makes ``only_k0`` usable from a caller who knows physics
    /// rather than engine indices: ``only_k0`` filters on star
    /// REPRESENTATIVES, and a caller's momentum may be a non-representative
    /// MEMBER of a star, so it must first read the star membership (and
    /// decode its momentum from ``irrep_characters``, since k_raw is an
    /// internal irrep index and NOT the momentum). Plan, decode, then solve
    /// the one star.
    bool plan_only = false;
};

/// One star's diagnostics.
struct LittleGroupStarInfo {
    int  k0            = 0;    ///< star representative (extended irrep index:
                               ///< k + s*n_irr_raw when flip is engaged)
    int  star_size     = 1;    ///< |star| (spectrum multiplicity factor)
    int  little_order  = 1;    ///< |P_k0| actually used (1 = plain fallback)
    bool projected     = false;///< true when the little-group blocks were used
    std::uint64_t dim_k0 = 0;  ///< k0 sector dimension (#surviving reps)
    int  flip_parity   = -1;   ///< 9a: 0 = (k,+), 1 = (k,-); -1 = flip not engaged
    int  tr_pairs      = 0;    ///< 9b: # sigma <-> sigma* pairs solved once
    /// Every extended irrep index folded into this star (always includes
    /// ``k0``). The engine has always known this -- the star loop iterates
    /// ``(k0, members)`` -- but only published ``star_size``, which is not
    /// enough to answer "which star holds MY momentum?". Naming a block needs
    /// exactly that: ``only_k0`` filters on REPRESENTATIVES, so a caller whose
    /// momentum is a non-representative member has to find its star first.
    /// Members are isospectral by construction (the residue maps them onto
    /// each other), so the representative's spectrum answers for all of them.
    std::vector<int> members;

    // -----------------------------------------------------------------------
    // The little co-group P_k0 of THIS star, published so a caller can NAME an
    // irrep by its character instead of by an index.
    //
    // `LittleGroupLabel::irrep` is an index into decompose_irreps' own
    // ordering -- engine-internal, exactly like `k_raw` (which is NOT the
    // momentum). Momentum is nameable because `irrep_characters` publishes
    // chi_k over the ABELIAN group; the co-group's characters had no such
    // table, so a `irrep=<index>` API would have handed callers an internal
    // convention and called it physics. These three fields are that missing
    // table.
    //
    //   little_elems[e]        -- which residue is co-group element e: an index
    //                             into the caller's own `residue_perms`, or -1
    //                             for the identity (always element 0). This is
    //                             what makes the character columns identifiable
    //                             as permutations the caller already holds.
    //   little_characters[s][e]-- chi_sigma(element e) for irrep s. Rows are
    //                             parallel to `LittleGroupLabel::irrep`.
    //   little_irrep_dims[s]   -- d_sigma. Sum of d_sigma^2 == little_order.
    //
    // Empty when the star was not projected (trivial co-group, or a graceful
    // per-star fallback): there is no table to report.
    // -----------------------------------------------------------------------
    std::vector<int> little_elems;
    std::vector<std::vector<std::complex<double>>> little_characters;
    std::vector<int> little_irrep_dims;
};

/// Stage 9f: per-eigenvalue quantum-number label, parallel to
/// ``LittleGroupSpectrum::eigenvalues``. ``k_raw`` is the star
/// REPRESENTATIVE's abelian irrep index -- fold partners (star members,
/// TR conjugates) share the representative's label; the full membership
/// is in ``stars[]``. ``irrep`` indexes the little co-group's irrep
/// decomposition at that star (-1 = plain / unprojected block; for a
/// TR-folded sigma/sigma* pair it names the SOLVED member).
struct LittleGroupLabel {
    int k_raw       = -1;   ///< abelian irrep (momentum) of the star rep
    int flip_parity = -1;   ///< 0 = (k,+), 1 = (k,-); -1 = flip not engaged
    int irrep       = -1;   ///< little-co-group irrep index; -1 = plain block
    int irrep_dim   = 1;    ///< d_sigma (1 for plain blocks)
    /// Jul 2026: true iff the block delivered the full requested window of
    /// DISTINCT, residual-bound-converged Ritz values (dense blocks are
    /// exact and always true). A budget-capped Lanczos that found fewer is
    /// flagged false -- merged campaigns must be able to tell.
    bool converged  = true;
};

struct LittleGroupSpectrum {
    /// Distinct block eigenvalues, ascending, with their TOTAL multiplicity
    /// (|star| × d_σ). Σ multiplicities == subspace dimension.
    std::vector<double> eigenvalues;
    std::vector<int>    multiplicities;
    std::vector<LittleGroupLabel>    labels;   ///< 9f: parallel to eigenvalues
    /// 9f: character table of the RAW abelian irreps in the engine's own
    /// ordering -- ``irrep_characters[k_raw][a]`` = chi_k(A[a]) with ``a``
    /// indexing the caller's ``abelian_group`` element order. This is what
    /// makes ``k_raw`` physically unambiguous: read the momentum off the
    /// translation generator's phase instead of trusting an index
    /// convention (decompose_irreps ordering != directory sector order).
    std::vector<std::vector<std::complex<double>>> irrep_characters;
    std::vector<LittleGroupStarInfo> stars;
    std::uint64_t       total_dim = 0;
    bool                flip_engaged = false;  ///< 9a: A' = A x Z2 was used
    bool                tr_engaged   = false;  ///< 9b: TR folding was active

    /// Flat sorted spectrum with multiplicities expanded (dense-diag shape).
    [[nodiscard]] std::vector<double> expanded() const;
};

/// Full spectrum of H over the (n_up | parity | full) subspace, factorized
/// star-by-star. `abelian_group` must be the CLOSED abelian group (all |A|
/// elements); `residue_perms` are point-group coset representatives (any
/// subset -- fewer residues just means less reduction).
[[nodiscard]] LittleGroupSpectrum
little_group_full_spectrum(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    const std::vector<std::vector<int>>&  residue_perms,
    int                                   n_sites,
    const LittleGroupOptions&             opt = {});

/// Lowest `k` eigenvalues (dense on small blocks, Lanczos on the projected
/// matrix-free matvec otherwise), multiplicities expanded and sorted.
[[nodiscard]] std::vector<double>
little_group_lowest_eigenvalues(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    const std::vector<std::vector<int>>&  residue_perms,
    int                                   n_sites,
    int                                   k,
    const LittleGroupOptions&             opt = {});

/// Stage 9f: lowest-`k`-per-block spectrum WITH labels (the structured form
/// `little_group_lowest_eigenvalues` flattens). Each block contributes its
/// lowest `k`; consumers expand by multiplicity and truncate.
[[nodiscard]] LittleGroupSpectrum
little_group_lowest_spectrum(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    const std::vector<std::vector<int>>&  residue_perms,
    int                                   n_sites,
    int                                   k,
    const LittleGroupOptions&             opt = {});

/// Stage 9d: the ground state localized in its momentum sector by the
/// star walk (flip/TR folds reduce the search), then solved PLAIN
/// (unprojected) in that sector with an in-memory eigenvector -- dense
/// for small blocks, FullCGS2 Lanczos Ritz vector otherwise, residual-
/// guarded (throws rather than returning a stale pair). The vector is
/// expressed in ``rd``'s rep basis, ready for
/// ``CrossSectorOrbitObservable::OperatorRef::from_rep``.
struct LittleGroupGroundState {
    double                             energy = 0.0;
    int                                k0     = -1;  ///< extended irrep index
    ed::symmetry::RepSectorData        rd;           ///< the GS sector
    std::vector<std::complex<double>>  vec;          ///< GS in rd's basis
};

[[nodiscard]] LittleGroupGroundState
little_group_ground_state(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    const std::vector<std::vector<int>>&  residue_perms,
    int                                   n_sites,
    const LittleGroupOptions&             opt = {});

/// Stage 9d: every non-empty RAW momentum sector of one diagonal
/// subspace (destination sweep of the factorized DSSF -- folding never
/// applies to matrix elements, so destinations enumerate raw irreps).
[[nodiscard]] std::vector<ed::symmetry::RepSectorData>
little_group_k_sectors(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    int                                   n_sites,
    int                                   n_up      = -1,
    int                                   sz_parity = -1);

/// Stage 9d: matrix-free H restricted to one rep-basis sector (public
/// factory over the engine's internal RepSectorMatVec).
[[nodiscard]] std::unique_ptr<ed::matvec::MatVecOperator>
make_rep_sector_matvec(
    const ::Operator&            op,
    ed::symmetry::RepSectorData  rd);

/// Exact canonical thermodynamics from the factorized full spectrum.
[[nodiscard]] ThermodynamicData
little_group_thermodynamics(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    const std::vector<std::vector<int>>&  residue_perms,
    int                                   n_sites,
    const std::vector<double>&            temperatures,
    const LittleGroupOptions&             opt = {});

}  // namespace ed::solvers
