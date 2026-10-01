#pragma once
// =============================================================================
// include/ed/sectors/little_group.h
//
// FACTORIZED non-abelian reduction via little co-groups: never stores
// symmetry-adapted amplitudes over the full space, so it scales to large N.
//
// Structure exploited: G = A ⋊ P with A the normal abelian part (translations)
// and P the residues (point-group coset representatives, Spec::residues).
// The abelian irreps k of A are the momentum sectors; residues permute them
// (χ_k → χ_k^p). Per STAR (residue orbit of momenta):
//
//   1. Solve only the star representative k0; every member contributes the
//      same spectrum (multiplicity |star|).
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
//      isotypic basis W_σ over the k0 REP INDICES (SVD per index-orbit).
//   4. Block = W_σ† H_{k0} W_σ with H_{k0} the MATRIX-FREE rep-kernel
//      matvec -- memory O(#reps(k0)), never O(2^N). Eigenvalues carry
//      multiplicity |star| × d_σ.
//
// Robustness contract: a residue that does not normalise A is refused
// (ed::InvalidRequest); every other reduction step degrades GRACEFULLY. A
// nontrivial (projective) factor system or a monomial action that fails the
// numerical [M_p, H] = 0 check falls back to solving the plain k0 block
// (correct, merely less reduced; LittleGroupStarInfo::declined says why).
// =============================================================================

#include <ed/ops/operator.h>
#include <ed/basis/rep_sector.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <functional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ed::solvers {

/// Irreps named by character: (residue index, chi) pairs that must all hold. Residue -1 is the
/// identity, whose character is the irrep dimension.
using CharConstraint = std::vector<std::pair<int, std::complex<double>>>;

struct LittleGroupOptions {
    int  n_up          = -1;   ///< fixed-Sz subspace (-1 = none)
    int  sz_parity     = -1;   ///< Sz-parity half (-1 = none; excludes n_up)
    bool verbose       = false;
    /// Spin-flip Z2 through the ABELIAN factor (A' = A x Z2 -- the
    /// flip commutes with every site permutation, so it never belongs to the
    /// little co-group). SymToggle convention: -1 auto (engage when
    /// [H, prod sigma^x] = 0 AND the subspace is flip-invariant: n_up = N/2,
    /// parity with N even, or the full space),
    /// 0 off, 1 require (throws when the symmetry or admissibility is absent).
    int  spin_flip     = -1;
    /// Time-reversal folding (star-level k <-> conj(k) merge +
    /// conjugate-irrep pairing). Same SymToggle convention.
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
    /// Solve ONLY the irreps meeting one of these character constraints (empty = every irrep).
    /// Unlike ``only_irrep`` they mean the same in every star, and they reach the path
    /// decision: a star whose wanted irreps are all one-dimensional takes the group-sector
    /// path even when the co-group also has larger irreps.
    std::vector<CharConstraint> only_irrep_chars;
};

/// One star's diagnostics.
struct LittleGroupStarInfo {
    int  k0            = 0;    ///< star representative (extended irrep index:
                               ///< k + s*n_irr_raw when flip is engaged)
    int  star_size     = 1;    ///< |star| (spectrum multiplicity factor)
    int  little_order  = 1;    ///< |P_k0| actually used (1 = plain fallback)
    int  flip_parity   = -1;   ///< 0 = (k,+), 1 = (k,-); -1 = flip not engaged
    /// Every extended irrep index folded into this star (always includes
    /// ``k0``). Answers "which star holds MY momentum?": ``only_k0`` filters
    /// on REPRESENTATIVES, so a caller whose momentum is a non-representative
    /// member has to find its star first.
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
    // chi_k over the ABELIAN group; these three fields are the corresponding
    // character table for the co-group, so callers never depend on the
    // internal index convention.
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
    /// Residues missing from little_elems because they act on this k-sector as a fixed multiple
    /// of a listed element (M_p = c M_e; element 0, the identity: a scalar). Each is (residue
    /// index, element e, c), and chi_sigma(residue) = c chi_sigma(element e).
    std::vector<std::tuple<int, int, std::complex<double>>> little_aliases;
    /// Why a NON-TRIVIAL little co-group could not be projected; empty when it was, or when the
    /// co-group is trivial. A declined star has one plain block that mixes irreps.
    std::string declined;
    /// chi_k0(a) over the RAW abelian group (the caller's order) for the representative.
    std::vector<std::complex<double>> momentum;
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
