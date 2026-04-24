// =============================================================================
// test_bfg_ring_observables (Catch2 v3, P2.1 ring-observables slice)
//
// Locks down the bowtie ring-flip + triangle ring-exchange kernels that
// were promoted out of `compute_bfg_order_parameters.cpp` into
// `ed_bfg::ring_observables`.
//
// All cases run on hand-checkable spin-1/2 product states. Convention
// matches the rest of the BFG suite: bit=0 -> spin UP, bit=1 -> spin DOWN.
//
// Triangle (3 sites, 8 basis states):
//   * |up,down,up> = bits (0,1,0) -> integer 2  ("DOWN_at_site_1" pattern)
//   * |down,up,down> = bits (1,0,1) -> integer 5  ("UP_at_site_1" pattern)
//   For psi = a|up,down,up> + b|down,up,down>:
//     S+_0 S-_1 S+_2 needs (DOWN, UP, DOWN) at (0,1,2) -> only the second
//        component contributes; it flips to (UP, DOWN, UP) i.e. state 2.
//     S-_0 S+_1 S-_2 needs (UP, DOWN, UP) -> only the first component
//        contributes; it flips to state 5.
//   So <triangle_chiral> = conj(psi[2]) * b + conj(psi[5]) * a
//                        = conj(a)*b + conj(b)*a = 2 Re(conj(a) b).
//
// Bowtie (4 outer corners + 1 center; the kernel ignores the center):
//   For one bowtie at q=0 we test the action on
//      |1010>_outer (s1=DOWN, s2=UP, s3=DOWN, s4=UP)
//   which is bit pattern (1,0,1,0) at outer sites (s1,s2,s3,s4) = (0,1,2,3),
//   integer state = 0b00000101 = 5 (note: site 0 -> bit 0 = 1, site 2 ->
//   bit 2 = 1, all other bits 0).
//   S+_1 S-_2 S+_3 S-_4 needs (DOWN, UP, DOWN, UP) at (s1..s4) so it fires;
//   the outer-site flips toggle bits 0,1,2,3 -> integer 0b1010 = 10.
//   The h.c. branch needs (UP, DOWN, UP, DOWN), which doesn't match.
//   So apply_bowtie_fourier just maps |state=5> -> |state=10>.
//
// Memory-efficient mode is exercised by toggling the global flag inside
// each test (it's a process-wide state, but we always reset to false at
// the end).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/bfg/ring_observables.h>
#include <ed/bfg/structure_factor.h>
#include <ed/bfg/topology.h>

#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <vector>

using Catch::Approx;
using ed::bfg::Complex;

namespace {

constexpr Complex I{0.0, 1.0};

ed::bfg::Bowtie make_bowtie(int s1, int s2, int s3, int s4,
                            std::array<double, 2> center) {
    ed::bfg::Bowtie bt;
    bt.s0 = -1;            // unused by the kernel
    bt.s1 = s1;
    bt.s2 = s2;
    bt.s3 = s3;
    bt.s4 = s4;
    bt.center = center;
    bt.orientation = 0;    // unused by the kernel
    return bt;
}

}  // namespace

// -----------------------------------------------------------------------------
// compute_triangle_chiral
// -----------------------------------------------------------------------------

TEST_CASE("ed::bfg::compute_triangle_chiral vanishes on Sz-polarised state",
          "[bfg][ring_observables][p2-1]") {
    // |up,up,up> only -- no S+S-S+ or h.c. matches.
    std::vector<Complex> psi(8, 0.0);
    psi[0b000] = 1.0;

    auto chi = ed::bfg::compute_triangle_chiral(psi, 0, 1, 2);
    REQUIRE(chi.real() == Approx(0.0).margin(1e-12));
    REQUIRE(chi.imag() == Approx(0.0).margin(1e-12));
}

TEST_CASE("ed::bfg::compute_triangle_chiral on (a|UDU> + b|DUD>) gives 2 Re(conj(a) b)",
          "[bfg][ring_observables][p2-1]") {
    // |up,down,up> -> bits (0,1,0) -> integer 0b010 = 2
    // |down,up,down> -> bits (1,0,1) -> integer 0b101 = 5
    std::vector<Complex> psi(8, 0.0);
    const Complex a{0.6, 0.0};
    const Complex b{0.0, 0.8};
    psi[2] = a;
    psi[5] = b;

    auto chi = ed::bfg::compute_triangle_chiral(psi, 0, 1, 2);

    const Complex expected =
        std::conj(psi[2]) * b + std::conj(psi[5]) * a;  // = 2 Re(conj(a) b)
    REQUIRE(chi.real() == Approx(expected.real()).margin(1e-12));
    REQUIRE(chi.imag() == Approx(expected.imag()).margin(1e-12));
    // Numerical sanity: a is real, b is imaginary -> Re(conj(a) b) = 0.
    REQUIRE(std::abs(chi) < 1e-12);
}

TEST_CASE("ed::bfg::compute_triangle_chiral picks up real part on real superposition",
          "[bfg][ring_observables][p2-1]") {
    std::vector<Complex> psi(8, 0.0);
    const double a = 0.6;
    const double b = 0.8;
    psi[2] = Complex(a, 0.0);
    psi[5] = Complex(b, 0.0);

    auto chi = ed::bfg::compute_triangle_chiral(psi, 0, 1, 2);
    REQUIRE(chi.real() == Approx(2.0 * a * b).margin(1e-12));
    REQUIRE(chi.imag() == Approx(0.0).margin(1e-12));
}

// -----------------------------------------------------------------------------
// compute_bowtie_resonance
// -----------------------------------------------------------------------------

TEST_CASE("ed::bfg::compute_bowtie_resonance vanishes on polarised state",
          "[bfg][ring_observables][p2-1]") {
    // 4 sites, |up,up,up,up> -> state 0; no S+S-S+S- match.
    std::vector<Complex> psi(16, 0.0);
    psi[0b0000] = 1.0;

    auto p = ed::bfg::compute_bowtie_resonance(psi, 0, 1, 2, 3);
    REQUIRE(p.real() == Approx(0.0).margin(1e-12));
    REQUIRE(p.imag() == Approx(0.0).margin(1e-12));
}

TEST_CASE("ed::bfg::compute_bowtie_resonance on (a|DUDU> + b|UDUD>) gives 2 Re(conj(a) b)",
          "[bfg][ring_observables][p2-1]") {
    // (s1, s2, s3, s4) = (0, 1, 2, 3).
    // |D,U,D,U> -> bits (1,0,1,0) -> int 0b0101 = 5.
    // |U,D,U,D> -> bits (0,1,0,1) -> int 0b1010 = 10.
    std::vector<Complex> psi(16, 0.0);
    const Complex a{0.6, 0.1};
    const Complex b{0.2, -0.7};
    psi[5]  = a;
    psi[10] = b;

    auto p = ed::bfg::compute_bowtie_resonance(psi, 0, 1, 2, 3);

    // S+_1 S-_2 S+_3 S-_4 fires on |DUDU>=5 and lands on |UDUD>=10:
    //    contrib = conj(psi[10]) * a = conj(b) * a
    // h.c. fires on |UDUD>=10 and lands on |DUDU>=5:
    //    contrib = conj(psi[5]) * b = conj(a) * b
    const Complex expected = std::conj(b) * a + std::conj(a) * b;
    REQUIRE(p.real() == Approx(expected.real()).margin(1e-12));
    REQUIRE(p.imag() == Approx(expected.imag()).margin(1e-12));
    // expected = 2 Re(conj(a) b) is purely real.
    REQUIRE(p.imag() == Approx(0.0).margin(1e-12));
}

// -----------------------------------------------------------------------------
// apply_bowtie_fourier
// -----------------------------------------------------------------------------

TEST_CASE("ed::bfg::apply_bowtie_fourier with no bowties returns zero ket",
          "[bfg][ring_observables][p2-1]") {
    std::vector<Complex> psi(16, Complex(0.5, 0.0));
    auto out = ed::bfg::apply_bowtie_fourier(/*bowties=*/{}, psi,
                                             /*q=*/{0.0, 0.0});
    REQUIRE(out.size() == psi.size());
    for (const auto& c : out) {
        REQUIRE(c.real() == Approx(0.0).margin(1e-15));
        REQUIRE(c.imag() == Approx(0.0).margin(1e-15));
    }
}

TEST_CASE("ed::bfg::apply_bowtie_fourier flips |DUDU> -> |UDUD> at q=0",
          "[bfg][ring_observables][p2-1]") {
    // 4 outer sites -> Hilbert dimension 16. center is unused by the kernel.
    std::vector<ed::bfg::Bowtie> bts = {
        make_bowtie(0, 1, 2, 3, {1.5, 2.5}),
    };

    // Default (fast) path.
    ed::bfg::set_memory_efficient_mode(0);
    REQUIRE(ed::bfg::memory_efficient_mode_enabled() == false);

    std::vector<Complex> psi(16, 0.0);
    psi[0b0101] = 1.0;  // |DUDU>

    auto out = ed::bfg::apply_bowtie_fourier(bts, psi, {0.0, 0.0});
    // S+_1 S-_2 S+_3 S-_4 |DUDU> = |UDUD>, multiplied by phase 1 at q=0.
    REQUIRE(out[0b1010].real() == Approx(1.0).margin(1e-12));
    REQUIRE(out[0b1010].imag() == Approx(0.0).margin(1e-12));
    // Everything else stays zero.
    for (int s = 0; s < 16; ++s) {
        if (s == 0b1010) continue;
        REQUIRE(std::abs(out[s]) < 1e-12);
    }
}

TEST_CASE("ed::bfg::apply_bowtie_fourier picks up exp(i q . r) phase",
          "[bfg][ring_observables][p2-1]") {
    const std::array<double, 2> center{1.5, -0.5};
    const std::array<double, 2> q{0.7, 1.3};
    std::vector<ed::bfg::Bowtie> bts = {make_bowtie(0, 1, 2, 3, center)};

    std::vector<Complex> psi(16, 0.0);
    psi[0b0101] = 1.0;  // |DUDU>

    auto out = ed::bfg::apply_bowtie_fourier(bts, psi, q);
    const double phi = q[0] * center[0] + q[1] * center[1];
    const Complex expected = std::exp(I * phi);
    REQUIRE(out[0b1010].real() == Approx(expected.real()).margin(1e-12));
    REQUIRE(out[0b1010].imag() == Approx(expected.imag()).margin(1e-12));
}

TEST_CASE("ed::bfg::apply_bowtie_fourier hermitian branch flips |UDUD> -> |DUDU>",
          "[bfg][ring_observables][p2-1]") {
    std::vector<ed::bfg::Bowtie> bts = {make_bowtie(0, 1, 2, 3, {0.0, 0.0})};
    std::vector<Complex> psi(16, 0.0);
    psi[0b1010] = 1.0;  // |UDUD>

    auto out = ed::bfg::apply_bowtie_fourier(bts, psi, {0.0, 0.0});
    // h.c. branch: S-_1 S+_2 S-_3 S+_4 |UDUD> = |DUDU> (state 0b0101 = 5).
    REQUIRE(out[0b0101].real() == Approx(1.0).margin(1e-12));
    REQUIRE(out[0b0101].imag() == Approx(0.0).margin(1e-12));
}

TEST_CASE("ed::bfg::apply_bowtie_fourier matches the slow path under memory-efficient mode",
          "[bfg][ring_observables][p2-1]") {
    std::vector<ed::bfg::Bowtie> bts = {
        make_bowtie(0, 1, 2, 3, {0.25, 0.75}),
        make_bowtie(0, 1, 2, 3, {-1.0, 0.5}),  // duplicated geometry, distinct phase
    };
    const std::array<double, 2> q{0.4, -0.3};

    std::vector<Complex> psi(16, 0.0);
    psi[0b0101] = Complex(0.7, 0.1);
    psi[0b1010] = Complex(-0.2, 0.5);

    ed::bfg::set_memory_efficient_mode(0);
    auto fast = ed::bfg::apply_bowtie_fourier(bts, psi, q);

    // Force the atomic / memory-efficient branch.
    ed::bfg::set_memory_efficient_mode(uint64_t{1} << 40);
    REQUIRE(ed::bfg::memory_efficient_mode_enabled() == true);
    auto slow = ed::bfg::apply_bowtie_fourier(bts, psi, q);

    // Reset for downstream tests.
    ed::bfg::set_memory_efficient_mode(0);
    REQUIRE(ed::bfg::memory_efficient_mode_enabled() == false);

    REQUIRE(fast.size() == slow.size());
    for (std::size_t i = 0; i < fast.size(); ++i) {
        REQUIRE(fast[i].real() == Approx(slow[i].real()).margin(1e-12));
        REQUIRE(fast[i].imag() == Approx(slow[i].imag()).margin(1e-12));
    }
}
