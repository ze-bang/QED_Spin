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

#include <cstdint>
#include <vector>

namespace ed::solvers {

struct LittleGroupOptions {
    int  n_up          = -1;   ///< fixed-Sz subspace (-1 = none)
    int  sz_parity     = -1;   ///< Sz-parity half (-1 = none; excludes n_up)
    int  dense_max_dim = 64;   ///< per-block dense/Lanczos crossover (lowest-k path)
    bool verbose       = false;
};

/// One star's diagnostics.
struct LittleGroupStarInfo {
    int  k0            = 0;    ///< star representative (abelian irrep index)
    int  star_size     = 1;    ///< |star| (spectrum multiplicity factor)
    int  little_order  = 1;    ///< |P_k0| actually used (1 = plain fallback)
    bool projected     = false;///< true when the little-group blocks were used
    std::uint64_t dim_k0 = 0;  ///< k0 sector dimension (#surviving reps)
};

struct LittleGroupSpectrum {
    /// Distinct block eigenvalues, ascending, with their TOTAL multiplicity
    /// (|star| × d_σ). Σ multiplicities == subspace dimension.
    std::vector<double> eigenvalues;
    std::vector<int>    multiplicities;
    std::vector<LittleGroupStarInfo> stars;
    std::uint64_t       total_dim = 0;

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
