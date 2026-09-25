// =============================================================================
// include/ed/observables/masked_program.h -- operators built from MaskedTerm, and their
// compilation into a flat program evaluated inside symmetry-reduced sectors.
//
// MaskedOperator stores a sum of canonical MaskedTerms (masked_term.h), keyed by
// (flip_mask, cond_val, sign_mask), so the expansion of an operator is unique and
// addition, products, adjoints and group images are exact.
//
// compile_program() turns a list of observables into the program that
// rep_matrix_elements() sweeps. It projects every observable onto the component that
// can connect the two sectors (the "lambda projection"):
//
//     O_lambda = (1/|G|) sum_g conj(lambda(g)) U_g O U_g^dagger,
//     lambda(g) = chi_bra(g) * conj(chi_ket(g)),
//
// so <bra|O|ket> == <bra|O_lambda|ket> for ANY O: a single bond, plaquette or string may
// be passed, no invariance is required. U_g acts as the engine's group element
// (site permutation in the apply_perm convention, then XOR flip_masks[g]).
// =============================================================================
#pragma once

#include <ed/observables/masked_term.h>

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace ed::symmetry { struct RepSectorData; }

namespace ed::observables {

class MaskedOperator {
public:
    using Complex = std::complex<double>;

    explicit MaskedOperator(int n_sites);

    /// coeff * prod_k O_k(site_k), applied right to left (the LAST factor acts first).
    /// ops[k] in {'+', '-', 'z', 'x', 'y', 'u', 'd', 'I'}: S+, S-, S^z, S^x, S^y,
    /// |up><up|, |dn><dn|, identity. Sites may repeat (spin-1/2 algebra is exact).
    static MaskedOperator product(int n_sites, const std::string& ops,
                                  const std::vector<int>& sites, Complex coeff = 1.0);

    [[nodiscard]] int n_sites() const noexcept { return n_; }
    [[nodiscard]] std::size_t size() const noexcept { return t_.size(); }
    [[nodiscard]] bool empty() const noexcept { return t_.empty(); }

    MaskedOperator& add(const MaskedOperator& o, Complex scale = 1.0);
    [[nodiscard]] MaskedOperator operator+(const MaskedOperator& o) const;
    [[nodiscard]] MaskedOperator operator*(const MaskedOperator& o) const;  ///< this * o (o first)
    [[nodiscard]] MaskedOperator scaled(Complex s) const;
    [[nodiscard]] MaskedOperator dagger() const;
    /// U_g O U_g^dagger for the element (perm in the apply_perm convention, flip mask m).
    [[nodiscard]] MaskedOperator image(const int* perm, std::uint64_t flip_xor) const;
    [[nodiscard]] bool is_hermitian(double tol = 1e-12) const;
    /// Net change of the number of set bits (down spins) if uniform over all terms,
    /// else throws: an operator that changes S^z by different amounts is two operators.
    [[nodiscard]] int delta_set_bits() const;
    /// Canonical terms (merged, coefficients above `drop`).
    [[nodiscard]] std::vector<MaskedTerm> terms(double drop = 0.0) const;
    /// Dense matrix M[t * 2^n + s] = <t|O|s> (tests only; n <= 12).
    [[nodiscard]] std::vector<Complex> to_dense() const;

    /// Add one term in any form; it is brought to canonical form (projector conditions
    /// expanded into I and Z, unconstrained flips into S+ + S-, Z on conditioned sites
    /// folded into the coefficient).
    void add_term(const MaskedTerm& t);

private:
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>;  // (flip, cond_val, sign)
    int n_;
    std::map<Key, Complex> t_;
    void add_canonical(std::uint64_t flip, std::uint64_t val, std::uint64_t sign, Complex c);
};

/// Flat program over which the sector sweep runs. Terms are grouped by flip mask (one
/// target state and ONE rep lookup per group per ket rep, shared by every observable that
/// flips those bits), then by the required values on the flipped bits (for a given ket
/// state at most one value subgroup matches: v == s & flip).
struct MaskedProgram {
    int n_obs = 0;
    int delta_set_bits = 0;                        ///< n_set(bra) - n_set(ket)
    std::vector<std::uint64_t> group_flip;         ///< flip mask per group
    std::vector<int>           group_setbits;      ///< popcount(v) every subgroup of the group has
    std::vector<std::uint32_t> group_vbegin;       ///< n_groups + 1 offsets into vsub_*
    std::vector<std::uint64_t> vsub_val;           ///< required values, sorted within a group
    std::vector<std::uint32_t> vsub_tbegin;        ///< n_vsub + 1 offsets into term_*
    std::vector<std::uint64_t> term_sign;          ///< sign_mask per term
    std::vector<std::complex<double>> term_coeff;  ///< coefficient per term
    std::vector<std::uint32_t> term_obs;           ///< observable index per term
    std::vector<std::size_t>   terms_per_obs;      ///< after projection (0: selection-rule zero)
    /// Characters the program was compiled for (checked by rep_matrix_elements).
    std::vector<std::complex<double>> src_characters, tgt_characters;
    int src_n_up = -1, tgt_n_up = -1;

    [[nodiscard]] std::size_t n_groups() const noexcept { return group_flip.size(); }
    [[nodiscard]] std::size_t n_terms() const noexcept { return term_sign.size(); }
};

struct CompileOptions {
    /// Apply the lambda projection (default). Only turn off for operators the caller
    /// guarantees are already covariant (U_g O U_g^dagger = lambda(g) O).
    bool project = true;
    /// Projected coefficients below drop * max|coeff| of the UNPROJECTED observable are
    /// discarded after merging (selection-rule zeros compile to no terms).
    double drop = 1e-14;
};

/// Compile `ops` for matrix elements <bra|O|ket> with ket in sector `src` and bra in
/// sector `tgt`. Both sectors must come from the same group (identical group_size,
/// n_sites, perms_flat and flip_masks). Terms whose change of the set-bit count does not
/// match tgt.n_up - src.n_up cannot connect the sectors and are dropped.
[[nodiscard]] MaskedProgram compile_program(const std::vector<MaskedOperator>& ops,
                                            const ed::symmetry::RepSectorData& src,
                                            const ed::symmetry::RepSectorData& tgt,
                                            const CompileOptions& opt = {});

}  // namespace ed::observables
