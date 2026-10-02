// =============================================================================
// src/ops/invariance.cpp -- the records-to-terms adapter and the symmetry verdicts.
// =============================================================================
#include <ed/ops/invariance.h>
#include <ed/ops/operator.h>

#include <stdexcept>
#include <string>

namespace ed::ops {

namespace {

char op_char(std::uint8_t t) {
    switch (t) {
        case 0: return '+';
        case 1: return '-';
        case 2: return 'z';
        default:
            throw std::invalid_argument("Operator: op_type " + std::to_string(t)
                                        + " is not 0 (S+), 1 (S-) or 2 (Sz)");
    }
}

int site_of(std::uint64_t s, int n) {
    if (s >= static_cast<std::uint64_t>(n))
        throw std::invalid_argument("Operator: site " + std::to_string(s) + " is outside [0, "
                                    + std::to_string(n) + ")");
    return static_cast<int>(s);
}

MaskedOperator total(int n, char a) {   // S^a_tot
    MaskedOperator S(n);
    for (int i = 0; i < n; ++i) S.add(MaskedOperator::product(n, std::string(1, a), {i}));
    return S;
}

}  // namespace

MaskedOperator masked(const ::Operator& op) {
    const int n = static_cast<int>(op.getNumBits());
    MaskedOperator m(n);
    for (const auto& t : op.transform_data_) {
        if (t.is_two_body)
            m.add(MaskedOperator::product(n, {op_char(t.op_type), op_char(t.op_type_2)},
                                          {site_of(t.site_index, n), site_of(t.site_index_2, n)},
                                          t.coefficient));
        else
            m.add(MaskedOperator::product(n, std::string(1, op_char(t.op_type)),
                                          {site_of(t.site_index, n)}, t.coefficient));
    }
    for (const auto& t : op.three_body_data_)   // O1 acts first: the last factor of a product
        m.add(MaskedOperator::product(n, {op_char(t.op_type_3), op_char(t.op_type_2), op_char(t.op_type_1)},
                                      {site_of(t.site_index_3, n), site_of(t.site_index_2, n),
                                       site_of(t.site_index_1, n)},
                                      t.coefficient));
    return m;
}

bool commutes_with_permutation(const MaskedOperator& H, const std::vector<int>& perm, double rtol) {
    const int n = H.n_sites();
    if (static_cast<int>(perm.size()) != n)
        throw std::invalid_argument("commutes_with_permutation: " + std::to_string(perm.size())
                                    + " entries for " + std::to_string(n) + " sites");
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    for (int p : perm) {
        if (p < 0 || p >= n || seen[static_cast<std::size_t>(p)])
            throw std::invalid_argument("commutes_with_permutation: not a permutation of 0.."
                                        + std::to_string(n - 1));
        seen[static_cast<std::size_t>(p)] = 1;
    }
    return invariant(H, H.image(perm.data(), 0), rtol);
}

SzContent sz_content(const MaskedOperator& H, double rtol) {
    bool u1 = true, parity = true;
    for (const auto& t : H.terms(rtol * H.max_abs())) {
        const int d = masked_delta_set_bits(t);   // the change of either spin count, up to sign
        if (d != 0) u1 = false;
        if (d % 2 != 0) parity = false;
    }
    return u1 ? SzContent::U1 : (parity ? SzContent::Parity : SzContent::None);
}

bool su2_invariant(const MaskedOperator& H, double rtol) {
    const double cut = rtol * H.max_abs();
    for (char a : {'+', '-', 'z'})
        if (commutator(H, total(H.n_sites(), a)).max_abs() > cut) return false;
    return true;
}

}  // namespace ed::ops
