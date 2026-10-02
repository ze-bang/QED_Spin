// =============================================================================
// include/ed/ops/algebra.h -- operators as sums of canonical MaskedTerms (term.h), keyed by
// (flip_mask, cond_val, sign_mask), so the expansion of an operator is unique and addition,
// products, adjoints and group images are exact.
// =============================================================================
#pragma once

#include <ed/ops/term.h>

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace ed::ops {

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

}  // namespace ed::ops
