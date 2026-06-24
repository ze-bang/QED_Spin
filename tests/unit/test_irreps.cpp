// =============================================================================
// tests/unit/test_irreps.cpp
//
// Validate the numerical irrep engine (ed::symmetry::decompose_irreps) against
// groups with known representation theory:
//   * Z_6 (cyclic): 6 one-dimensional irreps.
//   * D_4 / C4v (dihedral, order 8): four 1-D + one 2-D irrep (Σ d² = 8).
// Checks: Σ d_Γ² == |G|, #irreps == #conjugacy-classes, dims, character
// orthogonality (first/second orthogonality relations), and that the extracted
// D^Γ(g) are unitary and a homomorphism (D(a)D(b) == D(a·b)).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/symmetry/group.h>
#include <ed/symmetry/irreps.h>

#include <algorithm>
#include <complex>
#include <vector>

using ed::sym::Permutation;
using ed::sym::generate_group;
using ed::symmetry::decompose_irreps;
using Complex = std::complex<double>;

namespace {

std::vector<int> translation(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

// (1/|G|) Σ_g χ_Γ(g) conj(χ_Λ(g)) — should be δ_{ΓΛ}.
Complex char_overlap(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    Complex s(0, 0);
    for (std::size_t g = 0; g < a.size(); ++g) s += a[g] * std::conj(b[g]);
    return s / static_cast<double>(a.size());
}

}  // namespace

TEST_CASE("irreps: Z_6 has six 1-D irreps", "[irreps][abelian]") {
    auto G = generate_group({translation(6, 1)});
    REQUIRE(G.size() == 6);
    auto gi = decompose_irreps(G, 6);

    REQUIRE(gi.order == 6);
    REQUIRE(gi.num_classes == 6);              // abelian: every element its own class
    REQUIRE(gi.irreps.size() == 6);
    long long sumd2 = 0;
    for (const auto& ir : gi.irreps) { REQUIRE(ir.dim == 1); sumd2 += ir.dim * ir.dim; }
    REQUIRE(sumd2 == 6);
    REQUIRE(gi.is_abelian());

    // Character orthonormality.
    for (std::size_t a = 0; a < gi.irreps.size(); ++a)
        for (std::size_t b = 0; b < gi.irreps.size(); ++b) {
            const Complex ov = char_overlap(gi.irreps[a].character, gi.irreps[b].character);
            REQUIRE(std::abs(ov - (a == b ? Complex(1, 0) : Complex(0, 0))) < 1e-8);
        }
}

TEST_CASE("irreps: D_4 (order 8) has four 1-D and one 2-D irrep", "[irreps][nonabelian]") {
    // 4-fold rotation r and a reflection s of the square 0-1-2-3.
    const Permutation r{1, 2, 3, 0};   // i <- i+1
    const Permutation s{0, 3, 2, 1};   // fix 0,2 ; swap 1,3
    auto G = generate_group({r, s});
    REQUIRE(G.size() == 8);            // genuinely non-abelian D_4

    auto gi = decompose_irreps(G, 4);
    REQUIRE(gi.order == 8);
    REQUIRE(gi.num_classes == 5);
    REQUIRE(gi.irreps.size() == 5);

    std::vector<int> dims;
    long long sumd2 = 0;
    for (const auto& ir : gi.irreps) { dims.push_back(ir.dim); sumd2 += ir.dim * ir.dim; }
    std::sort(dims.begin(), dims.end());
    REQUIRE(dims == std::vector<int>({1, 1, 1, 1, 2}));
    REQUIRE(sumd2 == 8);
    REQUIRE_FALSE(gi.is_abelian());

    // Character orthonormality (first orthogonality relation).
    for (std::size_t a = 0; a < gi.irreps.size(); ++a)
        for (std::size_t b = 0; b < gi.irreps.size(); ++b) {
            const Complex ov = char_overlap(gi.irreps[a].character, gi.irreps[b].character);
            REQUIRE(std::abs(ov - (a == b ? Complex(1, 0) : Complex(0, 0))) < 1e-6);
        }

    // The 2-D irrep: D(g) unitary, and a homomorphism D(a)D(b) == D(a·b).
    const ed::symmetry::IrrepData* two = nullptr;
    for (const auto& ir : gi.irreps) if (ir.dim == 2) two = &ir;
    REQUIRE(two != nullptr);
    const int n = gi.order;
    auto matmul2 = [](const std::vector<Complex>& A, const std::vector<Complex>& B) {
        std::vector<Complex> C(4, Complex(0, 0));
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) C[i * 2 + j] += A[i * 2 + k] * B[k * 2 + j];
        return C;
    };
    for (int g = 0; g < n; ++g) {
        const auto& D = two->matrices[g];
        // unitary: D† D == I
        auto Dh = std::vector<Complex>{std::conj(D[0]), std::conj(D[2]),
                                       std::conj(D[1]), std::conj(D[3])};
        auto P = matmul2(Dh, D);
        REQUIRE(std::abs(P[0] - Complex(1, 0)) < 1e-6);
        REQUIRE(std::abs(P[3] - Complex(1, 0)) < 1e-6);
        REQUIRE(std::abs(P[1]) < 1e-6);
        REQUIRE(std::abs(P[2]) < 1e-6);
    }
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
            const auto prod = matmul2(two->matrices[a], two->matrices[b]);
            const auto& Dab = two->matrices[gi.mult[a][b]];
            for (int e = 0; e < 4; ++e) REQUIRE(std::abs(prod[e] - Dab[e]) < 1e-6);
        }
}
