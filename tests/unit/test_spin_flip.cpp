// =============================================================================
// tests/unit/test_spin_flip.cpp
//
// Stage-5 guards of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): the term-level spin-flip
// commutation check ``hamiltonian_is_spin_flip_symmetric``.
//
// Under X = prod_i sigma^x_i:  Sz -> -Sz,  S+ <-> S-.
// Cases pinned:
//   * Heisenberg / XXZ (SzSz + (S+S- + S-S+)/2)  -> symmetric
//   * transverse field hx*Sx = hx*(S+ + S-)/...  -> symmetric
//   * longitudinal (Zeeman) field h*Sz           -> NOT symmetric
//   * unpaired single S+                          -> NOT symmetric
//   * SzS+ with the -coeff SzS- partner           -> symmetric
//   * SzS+ with the +coeff SzS- partner           -> NOT symmetric
//   * S+S+ paired with S-S- (same coeff)          -> symmetric
//   * any three-body content                      -> NOT (conservative)
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/matvec/term_storage.h>
#include <ed/symmetry/spin_flip.h>

#include <complex>

using ed::matvec::TermStorage;
using ed::symmetry::hamiltonian_is_spin_flip_symmetric;
using Cx = std::complex<double>;

TEST_CASE("spin-flip check: Heisenberg / XXZ bond is symmetric",
          "[spin_flip]") {
    TermStorage t;
    t.diag_two_body.push_back({0, 1, Cx(0.7, 0.0)});                // Jz SzSz
    t.offdiag_two_body.push_back({0, 1, 0, 1, Cx(0.5, 0.0)});       // S+S-
    t.offdiag_two_body.push_back({0, 1, 1, 0, Cx(0.5, 0.0)});       // S-S+
    REQUIRE(hamiltonian_is_spin_flip_symmetric(t));
}

TEST_CASE("spin-flip check: transverse field is symmetric, Zeeman is not",
          "[spin_flip]") {
    TermStorage t;
    t.offdiag_one_body.push_back({3, 0, Cx(0.25, 0.0)});  // hx S+
    t.offdiag_one_body.push_back({3, 1, Cx(0.25, 0.0)});  // hx S-
    REQUIRE(hamiltonian_is_spin_flip_symmetric(t));

    t.diag_one_body.push_back({2, Cx(0.1, 0.0)});         // hz Sz
    REQUIRE_FALSE(hamiltonian_is_spin_flip_symmetric(t));
}

TEST_CASE("spin-flip check: unpaired raising operator breaks it",
          "[spin_flip]") {
    TermStorage t;
    t.offdiag_one_body.push_back({0, 0, Cx(0.25, 0.0)});  // lone S+
    REQUIRE_FALSE(hamiltonian_is_spin_flip_symmetric(t));
}

TEST_CASE("spin-flip check: mixed SzS+/- needs the NEGATED partner",
          "[spin_flip]") {
    TermStorage good;
    good.mixed_two_body.push_back({0, 1, 0, Cx(0.3, 0.0)});    // c Sz S+
    good.mixed_two_body.push_back({0, 1, 1, Cx(-0.3, 0.0)});   // -c Sz S-
    REQUIRE(hamiltonian_is_spin_flip_symmetric(good));

    TermStorage bad;
    bad.mixed_two_body.push_back({0, 1, 0, Cx(0.3, 0.0)});
    bad.mixed_two_body.push_back({0, 1, 1, Cx(0.3, 0.0)});     // wrong sign
    REQUIRE_FALSE(hamiltonian_is_spin_flip_symmetric(bad));
}

TEST_CASE("spin-flip check: S+S+ pairs with S-S- (either site order)",
          "[spin_flip]") {
    TermStorage t;
    t.offdiag_two_body.push_back({0, 1, 0, 0, Cx(0.2, 0.05)});  // c S+S+
    t.offdiag_two_body.push_back({1, 0, 1, 1, Cx(0.2, 0.05)});  // c S-S- swapped
    REQUIRE(hamiltonian_is_spin_flip_symmetric(t));

    TermStorage bad;
    bad.offdiag_two_body.push_back({0, 1, 0, 0, Cx(0.2, 0.0)});
    REQUIRE_FALSE(hamiltonian_is_spin_flip_symmetric(bad));
}

TEST_CASE("spin-flip check: three-body content is conservatively rejected",
          "[spin_flip]") {
    TermStorage t;
    t.three_body.push_back({2, 0, 2, 1, 2, 2, Cx(0.1, 0.0)});  // SzSzSz (even!)
    REQUIRE_FALSE(hamiltonian_is_spin_flip_symmetric(t));  // conservative
}
