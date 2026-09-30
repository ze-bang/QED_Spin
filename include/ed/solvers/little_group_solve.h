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
#include <ed/symmetry/rep_sector_data.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <functional>
#include <vector>

namespace ed::solvers {

struct LittleGroupOptions {
    int  n_up          = -1;   ///< fixed-Sz subspace (-1 = none)
    int  sz_parity     = -1;   ///< Sz-parity half (-1 = none; excludes n_up)
    int  dense_max_dim = 64;   ///< per-block dense/Lanczos crossover (lowest-k path)
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
    /// Note a star is the solve unit: members of one star are isospectral by
    /// construction (the point-group residue maps them onto each other), so
    /// restricting to a star representative answers for every member of that
    /// star.
    std::vector<int> only_k0;
    /// Solve ONLY these little-co-group irreps (indices into the star's own
    /// isotypic decomposition -- the same index ``LittleGroupBlockTag::irrep``
    /// reports, and the row index of ``LittleGroupStarInfo::little_characters``).
    /// Empty = every irrep, the default.
    ///
    /// Only meaningful together with ``only_k0``: the irrep index is PER STAR
    /// (decompose_irreps orders each star's decomposition independently), so
    /// "irrep 1" means different things in different stars. Callers name an
    /// irrep by its CHARACTER and resolve it to an index against the plan's
    /// published table -- never by assuming an index convention.
    ///
    /// TR INTERACTION: when this is non-empty the sigma <-> sigma* pairing is
    /// disabled. That fold solves ONE member of a conjugate pair and reports
    /// the pair's doubled multiplicity under the earlier member's label, so a
    /// caller who named the LATER member would get nothing back. Naming one
    /// irrep forfeits a 2x fold that is irrelevant next to the |P_k| the
    /// projection already bought.
    std::vector<int> only_irrep;
    /// Lowest-k solves above the dense crossover: 1 (default) = single-vector
    /// Krylov-Schur with locking when k > 1 (the basis-free scan when k = 1);
    /// p >= 2 = block Krylov-Schur with block width p, which also resolves an
    /// accidental degeneracy of up to p levels INSIDE one (k, irrep, flip) block --
    /// something no single-vector method can see.
    int block_size = 1;
};

/// One star's diagnostics.
struct LittleGroupStarInfo {
    int  k0            = 0;    ///< star representative (extended irrep index:
                               ///< k + s*n_irr_raw when flip is engaged)
    int  star_size     = 1;    ///< |star| (spectrum multiplicity factor)
    int  little_order  = 1;    ///< |P_k0| actually used (1 = plain fallback)
    int  flip_parity   = -1;   ///< 9a: 0 = (k,+), 1 = (k,-); -1 = flip not engaged
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
    // `LittleGroupBlockTag::irrep` is an index into decompose_irreps' own
    // ordering -- engine-internal, exactly like `k_raw` (which is NOT the
    // momentum). Momentum is nameable because the abelian irrep characters give
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
    //                             parallel to `LittleGroupBlockTag::irrep`.
    //   little_irrep_dims[s]   -- d_sigma. Sum of d_sigma^2 == little_order.
    //
    // Empty when the star was not projected (trivial co-group, or a graceful
    // per-star fallback): there is no table to report.
    // -----------------------------------------------------------------------
    std::vector<int> little_elems;
    std::vector<std::vector<std::complex<double>>> little_characters;
    std::vector<int> little_irrep_dims;
};

/// Streaming sweep over every non-empty RAW momentum sector of one diagonal
/// subspace (destination sweep of the factorized DSSF -- folding never applies
/// to matrix elements, so destinations enumerate raw irreps): invoke ``fn`` on
/// each sector one at a time, freeing it before the next is built. Bounds the
/// resident set to a single destination sector (each is ~15-20 GB at N=36
/// half-filling; holding all 12 OOMs a 128 GB node).
void little_group_k_sectors_stream(
    const ::Operator&                     op,
    const std::vector<std::vector<int>>&  abelian_group,
    int                                   n_sites,
    int                                   n_up,
    int                                   sz_parity,
    const std::function<void(ed::symmetry::RepSectorData&)>& fn);

/// One shared, read-only copy of a sector basis with its permutation LUT built, so
/// several operators on the SAME (k, n_up) sector (H and a batch of probes) do not
/// each copy the reps / norms / perm tables (1-2 GB at N = 36).
[[nodiscard]] std::shared_ptr<const ed::symmetry::RepSectorData>
share_rep_sector(ed::symmetry::RepSectorData rd);

}  // namespace ed::solvers
