// =============================================================================
// tests/unit/test_masked_terms.cpp
//
// MaskedOperator (include/ed/observables/masked_program.h) against dense matrices
// built independently here from 2x2 single-site matrices in the engine convention
// (bit 0 = up, bit 1 = down; S+ |dn> = |up>):
//   1. every single-site op character, embedded on a small chain;
//   2. random products of up to 6 factors, repeated sites included (spin-1/2 algebra);
//   3. dagger() == conjugate transpose;
//   4. image(perm, xor) == U O U^dagger with U|s> = |P(s) ^ m>, P in the apply_perm convention;
//   5. is_hermitian() and delta_set_bits().
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/ops/algebra.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::MaskedOperator;
using ed::ops::MaskedTerm;

namespace {

using M2 = std::array<Cx, 4>;   // m[t * 2 + s] = <t|m|s>, t,s in {0 = up, 1 = down}

M2 single(char op) {
    const Cx i(0.0, 1.0);
    switch (op) {
        case '+': return {0, 1, 0, 0};                       // <up|S+|dn> = 1
        case '-': return {0, 0, 1, 0};                       // <dn|S-|up> = 1
        case 'z': return {0.5, 0, 0, -0.5};
        case 'x': return {0, 0.5, 0.5, 0};
        case 'y': return {0, -0.5 * i, 0.5 * i, 0};          // (S+ - S-) / 2i
        case 'u': return {1, 0, 0, 0};
        case 'd': return {0, 0, 0, 1};
        default:  return {1, 0, 0, 1};
    }
}

std::vector<Cx> embed(int n, int site, const M2& m) {
    const std::uint64_t dim = 1ULL << n;
    std::vector<Cx> M(dim * dim, 0.0);
    for (std::uint64_t s = 0; s < dim; ++s)
        for (int tb = 0; tb < 2; ++tb) {
            const int sb = static_cast<int>((s >> site) & 1ULL);
            const Cx v = m[static_cast<std::size_t>(tb * 2 + sb)];
            if (v == Cx(0.0)) continue;
            const std::uint64_t t = (s & ~(1ULL << site)) | (static_cast<std::uint64_t>(tb) << site);
            M[t * dim + s] += v;
        }
    return M;
}

std::vector<Cx> matmul(const std::vector<Cx>& A, const std::vector<Cx>& B, std::uint64_t dim) {
    std::vector<Cx> C(dim * dim, 0.0);
    for (std::uint64_t i = 0; i < dim; ++i)
        for (std::uint64_t k = 0; k < dim; ++k) {
            const Cx a = A[i * dim + k];
            if (a == Cx(0.0)) continue;
            for (std::uint64_t j = 0; j < dim; ++j) C[i * dim + j] += a * B[k * dim + j];
        }
    return C;
}

double maxdiff(const std::vector<Cx>& A, const std::vector<Cx>& B) {
    double d = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) d = std::max(d, std::abs(A[i] - B[i]));
    return d;
}

std::vector<Cx> dense_product(int n, const std::string& ops, const std::vector<int>& sites, Cx c) {
    const std::uint64_t dim = 1ULL << n;
    std::vector<Cx> R(dim * dim, 0.0);
    for (std::uint64_t s = 0; s < dim; ++s) R[s * dim + s] = c;
    for (std::size_t k = 0; k < ops.size(); ++k)
        R = matmul(R, embed(n, sites[k], single(ops[k])), dim);
    return R;
}

}  // namespace

TEST_CASE("single-site operators match the engine convention", "[masked]") {
    const int n = 3;
    for (char op : std::string("+-zxyudI"))
        for (int site = 0; site < n; ++site) {
            const auto O = MaskedOperator::product(n, std::string(1, op), {site}, 1.0);
            REQUIRE(maxdiff(O.to_dense(), embed(n, site, single(op))) < 1e-14);
        }
}

TEST_CASE("random products, adjoints, images, hermiticity", "[masked]") {
    const int n = 5;
    const std::uint64_t dim = 1ULL << n;
    std::mt19937 rng(20260924);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1);
    std::uniform_int_distribution<int> pick_site(0, n - 1);
    std::uniform_int_distribution<int> pick_len(1, 6);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int trial = 0; trial < 200; ++trial) {
        const int K = pick_len(rng);
        std::string ops;
        std::vector<int> sites;
        for (int k = 0; k < K; ++k) {
            ops.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
            sites.push_back(pick_site(rng));
        }
        const Cx c(gauss(rng), gauss(rng));
        const auto O = MaskedOperator::product(n, ops, sites, c);
        const auto D = dense_product(n, ops, sites, c);
        INFO("ops " << ops);
        REQUIRE(maxdiff(O.to_dense(), D) < 1e-12);

        // adjoint
        std::vector<Cx> Dt(dim * dim);
        for (std::uint64_t i = 0; i < dim; ++i)
            for (std::uint64_t j = 0; j < dim; ++j) Dt[j * dim + i] = std::conj(D[i * dim + j]);
        REQUIRE(maxdiff(O.dagger().to_dense(), Dt) < 1e-12);

        // image under a random permutation + xor: U|s> = |P(s) ^ m>
        std::vector<int> perm(static_cast<std::size_t>(n));
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        const std::uint64_t m = (trial % 2) ? (dim - 1) : static_cast<std::uint64_t>(trial % static_cast<int>(dim));
        std::vector<std::uint64_t> U(dim);
        for (std::uint64_t s = 0; s < dim; ++s)
            U[s] = ed::ops::permute_mask(s, perm.data(), n) ^ m;
        std::vector<Cx> UDU(dim * dim, 0.0);
        for (std::uint64_t t = 0; t < dim; ++t)
            for (std::uint64_t s = 0; s < dim; ++s) UDU[U[t] * dim + U[s]] = D[t * dim + s];
        REQUIRE(maxdiff(O.image(perm.data(), m).to_dense(), UDU) < 1e-12);

        // hermiticity of O + O^dagger, and delta_set_bits of a pure ladder string
        REQUIRE((O + O.dagger()).is_hermitian());
    }
    // a non-Hermitian single term is detected; S^z changes are counted in set bits (down spins)
    const auto sp = MaskedOperator::product(n, "+", {0}, 1.0);
    CHECK_FALSE(sp.is_hermitian());
    CHECK(sp.delta_set_bits() == -1);                                  // S+ removes a down spin
    CHECK(MaskedOperator::product(n, "-", {2}, 1.0).delta_set_bits() == 1);
    CHECK(MaskedOperator::product(n, "+-+-", {0, 1, 2, 3}, 1.0).delta_set_bits() == 0);
    // spin-1/2 identities: S+ S+ = 0, S+ S- = |up><up|, (2 S^z)^2 = 1
    CHECK(MaskedOperator::product(n, "++", {1, 1}, 1.0).terms(1e-15).empty());
    CHECK(maxdiff(MaskedOperator::product(n, "+-", {1, 1}, 1.0).to_dense(),
                  MaskedOperator::product(n, "u", {1}, 1.0).to_dense()) < 1e-14);
    CHECK(maxdiff(MaskedOperator::product(n, "zz", {4, 4}, 4.0).to_dense(),
                  MaskedOperator::product(n, "I", {0}, 1.0).to_dense()) < 1e-14);
}
