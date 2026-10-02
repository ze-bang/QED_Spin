// =============================================================================
// tests/unit/test_masked_terms.cpp
//
// MaskedOperator (include/ed/ops/algebra.h) against dense matrices built independently
// here from 2x2 single-site matrices in the (up, down) basis; which bit value is up follows
// the engine convention (kSetBitIsDown, term.h; S+ |dn> = |up>):
//   1. every single-site op character, embedded on a small chain;
//   2. random products of up to 6 factors, repeated sites included (spin-1/2 algebra);
//   3. dagger() == conjugate transpose;
//   4. image(perm, xor) == U O U^dagger with U|s> = |P(s) ^ m>, P in the apply_perm convention;
//   5. is_hermitian(), delta_set_bits() and delta_up();
//   6. -, commutator, equals, and the global maps K, F, Dz, Theta against U O U^dagger
//      (U O* U^dagger for the antiunitary ones) with U a product of 2x2 matrices.
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
#include <tuple>
#include <utility>
#include <vector>

using Cx = std::complex<double>;
using ed::ops::MaskedOperator;
using ed::ops::MaskedTerm;

namespace {

using M2 = std::array<Cx, 4>;   // m[t * 2 + s] = <t|m|s>, t,s in {0 = up, 1 = down}

// (up, down) index of a bit value and back (an involution).
int ud(int bit) { return ed::ops::kSetBitIsDown ? bit : 1 - bit; }

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
            const int sb = ud(static_cast<int>((s >> site) & 1ULL));
            const Cx v = m[static_cast<std::size_t>(tb * 2 + sb)];
            if (v == Cx(0.0)) continue;
            const std::uint64_t t = (s & ~(1ULL << site)) | (static_cast<std::uint64_t>(ud(tb)) << site);
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
    // a non-Hermitian single term is detected; S^z changes in up spins and in set bits
    const auto sp = MaskedOperator::product(n, "+", {0}, 1.0);
    CHECK_FALSE(sp.is_hermitian());
    CHECK(sp.delta_up() == 1);
    CHECK(sp.delta_set_bits() == (ed::ops::kSetBitIsDown ? -1 : 1));
    CHECK(MaskedOperator::product(n, "-", {2}, 1.0).delta_up() == -1);
    CHECK(MaskedOperator::product(n, "+-+-", {0, 1, 2, 3}, 1.0).delta_up() == 0);
    CHECK_THROWS(MaskedOperator::product(n, "x", {0}, 1.0).delta_up());
    // spin-1/2 identities: S+ S+ = 0, S+ S- = |up><up|, (2 S^z)^2 = 1
    CHECK(MaskedOperator::product(n, "++", {1, 1}, 1.0).terms(1e-15).empty());
    CHECK(maxdiff(MaskedOperator::product(n, "+-", {1, 1}, 1.0).to_dense(),
                  MaskedOperator::product(n, "u", {1}, 1.0).to_dense()) < 1e-14);
    CHECK(maxdiff(MaskedOperator::product(n, "zz", {4, 4}, 4.0).to_dense(),
                  MaskedOperator::product(n, "I", {0}, 1.0).to_dense()) < 1e-14);
}

TEST_CASE("commutators, equality and the global maps K, F, Dz, Theta", "[masked]") {
    using Map = MaskedOperator::Map;
    const int n = 4;
    const std::uint64_t dim = 1ULL << n;
    auto global = [&](const M2& m) {             // prod_i m_i
        std::vector<Cx> U(dim * dim, 0.0);
        for (std::uint64_t s = 0; s < dim; ++s) U[s * dim + s] = 1.0;
        for (int i = 0; i < n; ++i) U = matmul(U, embed(n, i, m), dim);
        return U;
    };
    const auto UF = global({0, 1, 1, 0});        // sigma^x
    const auto UD = global({1, 0, 0, -1});       // sigma^z
    const auto UT = global({0, 1, -1, 0});       // i sigma^y: <up|.|dn> = 1, <dn|.|up> = -1
    const std::vector<Cx> UI = global({1, 0, 0, 1});
    auto conjugate = [&](const std::vector<Cx>& U, std::vector<Cx> D, bool antiunitary) {
        if (antiunitary) for (auto& x : D) x = std::conj(x);   // (U K) O (U K)^-1 = U O* U^dagger
        std::vector<Cx> Ud(dim * dim);
        for (std::uint64_t i = 0; i < dim; ++i)
            for (std::uint64_t j = 0; j < dim; ++j) Ud[j * dim + i] = std::conj(U[i * dim + j]);
        return matmul(matmul(U, D, dim), Ud, dim);
    };
    auto lin = [&](const std::vector<Cx>& A, Cx a, const std::vector<Cx>& B, Cx b) {
        std::vector<Cx> C(A.size());
        for (std::size_t i = 0; i < A.size(); ++i) C[i] = a * A[i] + b * B[i];
        return C;
    };
    std::vector<int> identity(static_cast<std::size_t>(n));
    std::iota(identity.begin(), identity.end(), 0);

    std::mt19937 rng(20261001);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1);
    std::uniform_int_distribution<int> pick_site(0, n - 1);
    std::uniform_int_distribution<int> pick_len(1, 4);
    std::normal_distribution<double> gauss(0.0, 1.0);
    auto random_op = [&] {
        MaskedOperator O(n);
        for (int t = 0; t < 3; ++t) {
            const int K = pick_len(rng);
            std::string ops;
            std::vector<int> sites;
            for (int k = 0; k < K; ++k) {
                ops.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
                sites.push_back(pick_site(rng));
            }
            O.add(MaskedOperator::product(n, ops, sites, Cx(gauss(rng), gauss(rng))));
        }
        return O;
    };
    for (int trial = 0; trial < 100; ++trial) {
        const auto A = random_op(), B = random_op();
        const auto DA = A.to_dense(), DB = B.to_dense();
        REQUIRE(maxdiff((A - B).to_dense(), lin(DA, 1.0, DB, -1.0)) < 1e-12);
        REQUIRE(maxdiff((-A).to_dense(), lin(DA, -1.0, DA, 0.0)) < 1e-12);
        REQUIRE(maxdiff(ed::ops::commutator(A, B).to_dense(),
                        lin(matmul(DA, DB, dim), 1.0, matmul(DB, DA, dim), -1.0)) < 1e-12);

        REQUIRE(maxdiff(A.image(Map::K).to_dense(), conjugate(UI, DA, true)) < 1e-12);
        REQUIRE(maxdiff(A.image(Map::F).to_dense(), conjugate(UF, DA, false)) < 1e-12);
        REQUIRE(maxdiff(A.image(Map::Dz).to_dense(), conjugate(UD, DA, false)) < 1e-12);
        REQUIRE(maxdiff(A.image(Map::Theta).to_dense(), conjugate(UT, DA, true)) < 1e-12);
        REQUIRE(A.image(Map::F).equals(A.image(identity.data(), dim - 1)));
        REQUIRE(A.image(Map::Theta).equals(A.image(Map::K).image(Map::F).image(Map::Dz)));
        REQUIRE(A.image(Map::Theta).image(Map::Theta).equals(A));   // Theta^2 = (-1)^N commutes out

        REQUIRE(A.equals(A + A.scaled(1e-14)));
        if (!A.terms(1e-9).empty()) REQUIRE_FALSE(A.equals(A.scaled(1.5)));
    }

    const Cx i(0.0, 1.0);
    auto P = [&](const char* ops, std::vector<int> sites, Cx c = 1.0) {
        return MaskedOperator::product(n, ops, sites, c);
    };
    CHECK(ed::ops::commutator(P("x", {0}), P("y", {0})).equals(P("z", {0}, i)));   // [Sx, Sy] = i Sz
    CHECK(ed::ops::commutator(P("z", {1}), P("+", {1})).equals(P("+", {1})));       // [Sz, S+] = S+
    CHECK(ed::ops::commutator(P("x", {0}), P("y", {1})).empty());
    for (const char* a : {"x", "y", "z"}) {
        CHECK(P(a, {2}).image(Map::Theta).equals(-P(a, {2})));                     // Theta S^a = -S^a
        CHECK(P(a, {2}).image(Map::F).equals(std::string(a) == "x" ? P(a, {2}) : -P(a, {2})));
    }
    CHECK(P("+", {3}).image(Map::Dz).equals(-P("+", {3})));
    CHECK(P("z", {3}).image(Map::Dz).equals(P("z", {3})));
    CHECK(P("y", {0}).image(Map::K).equals(-P("y", {0})));
    const auto heis = P("xx", {0, 1}) + P("yy", {0, 1}) + P("zz", {0, 1});
    for (Map g : {Map::K, Map::F, Map::Dz, Map::Theta}) CHECK(heis.image(g).equals(heis));
    CHECK_FALSE(P("z", {0}).equals(MaskedOperator::product(n + 1, "z", {0}, 1.0)));
}

namespace {

// <t|O|s> of a sum of products by acting with the single-site matrices on each |s>, the
// rightmost factor first: the Kronecker reference without dim^3 matrix products.
using Product = std::tuple<std::string, std::vector<int>, Cx>;
std::vector<Cx> dense_by_action(int n, const std::vector<Product>& prods) {
    const std::uint64_t dim = 1ULL << n;
    std::vector<Cx> M(dim * dim, 0.0);
    for (std::uint64_t s = 0; s < dim; ++s)
        for (const auto& [ops, sites, c] : prods) {
            std::vector<std::pair<std::uint64_t, Cx>> v{{s, c}};
            for (std::size_t k = ops.size(); k-- > 0;) {
                const M2 m = single(ops[k]);
                const int site = sites[k];
                std::vector<std::pair<std::uint64_t, Cx>> w;
                for (const auto& [st, a] : v) {
                    const int sb = ud(static_cast<int>((st >> site) & 1ULL));
                    for (int tb = 0; tb < 2; ++tb) {
                        const Cx x = m[static_cast<std::size_t>(tb * 2 + sb)];
                        if (x == Cx(0.0)) continue;
                        w.emplace_back((st & ~(1ULL << site)) | (static_cast<std::uint64_t>(ud(tb)) << site), a * x);
                    }
                }
                v = std::move(w);
            }
            for (const auto& [t, a] : v) M[t * dim + s] += a;
        }
    return M;
}

}  // namespace

TEST_CASE("2000 random operators match the Kronecker reference", "[masked]") {
    // N = 2..10, sums of 1-3 products of 1-4 factors over + - z x y u d I with repeated sites.
    std::mt19937 rng(20261002);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1);
    std::uniform_int_distribution<int> pick_len(1, 4), pick_terms(1, 3);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (int trial = 0; trial < 2000; ++trial) {
        const int n = 2 + trial % 9;
        std::uniform_int_distribution<int> pick_site(0, n - 1);
        std::vector<Product> prods;
        MaskedOperator O(n);
        for (int t = pick_terms(rng); t > 0; --t) {
            std::string ops;
            std::vector<int> sites;
            for (int k = pick_len(rng); k > 0; --k) {
                ops.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
                sites.push_back(pick_site(rng));
            }
            const Cx c(gauss(rng), gauss(rng));
            prods.emplace_back(ops, sites, c);
            O.add(MaskedOperator::product(n, ops, sites, c));
        }
        INFO("trial " << trial << " n " << n << " first ops " << std::get<0>(prods[0]));
        REQUIRE(maxdiff(O.to_dense(), dense_by_action(n, prods)) < 1e-13);
    }
}

TEST_CASE("the algebra laws hold exactly on the canonical keys", "[masked]") {
    // Dyadic coefficients (k / 4) keep every sum and product exact, so the laws must hold bit
    // for bit: equals(..., 0.0).
    const int n = 5;
    std::mt19937 rng(20261008);
    const std::string alphabet = "+-zxyudI";
    std::uniform_int_distribution<int> pick_op(0, static_cast<int>(alphabet.size()) - 1);
    std::uniform_int_distribution<int> pick_site(0, n - 1), pick_len(1, 3), pick_k(-8, 8);
    auto dyadic = [&] { return Cx(pick_k(rng) / 4.0, pick_k(rng) / 4.0); };
    auto random_op = [&] {
        MaskedOperator O(n);
        for (int t = 0; t < 3; ++t) {
            std::string ops;
            std::vector<int> sites;
            for (int k = pick_len(rng); k > 0; --k) {
                ops.push_back(alphabet[static_cast<std::size_t>(pick_op(rng))]);
                sites.push_back(pick_site(rng));
            }
            O.add(MaskedOperator::product(n, ops, sites, dyadic()));
        }
        return O;
    };
    for (int trial = 0; trial < 200; ++trial) {
        const auto A = random_op(), B = random_op(), C = random_op();
        const Cx a = dyadic();
        INFO("trial " << trial);
        REQUIRE((A + B).equals(B + A, 0.0));
        REQUIRE(((A + B) + C).equals(A + (B + C), 0.0));
        REQUIRE(((A * B) * C).equals(A * (B * C), 0.0));
        REQUIRE((A * (B + C)).equals(A * B + A * C, 0.0));
        REQUIRE(((A + B) * C).equals(A * C + B * C, 0.0));
        REQUIRE((A * B).dagger().equals(B.dagger() * A.dagger(), 0.0));
        REQUIRE(A.dagger().dagger().equals(A, 0.0));
        REQUIRE((A - A).empty());
        REQUIRE((A.scaled(a) * B).equals((A * B).scaled(a), 0.0));
    }
}
