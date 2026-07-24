// =============================================================================
// tests/unit/test_su2_detect.cpp
//
// Stage 12b of the SU(2) rollout: the term-level [H, S_tot] = 0 detector
// ``hamiltonian_is_su2_symmetric`` (include/ed/symmetry/su2.h).
//
// Cases pinned:
//   * Heisenberg bond J*(SzSz + 0.5 S+S- + 0.5 S-S+)      -> SU(2)
//   * two bonds with DIFFERENT J, each isotropic          -> SU(2)
//   * reversed storage order (S-_j S+_i style records)    -> SU(2)
//   * identity-shift diag(i,i) terms                      -> ignored
//   * XXZ (Delta != 1), XY-only, Ising-only bonds         -> NOT
//   * Zeeman field, transverse field                      -> NOT
//   * mixed Sz S+/- (DM/Gamma-type)                       -> NOT
//   * S+S+ pair terms                                     -> NOT
//   * any three-body content                              -> NOT (conservative)
//   * algebra containment: su2 => U1 & spin-flip & real
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/matvec/term_storage.h>
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/su2.h>

#include <complex>

using ed::matvec::TermStorage;
using ed::symmetry::hamiltonian_is_su2_symmetric;
using Cx = std::complex<double>;

namespace {

// Emit an isotropic Heisenberg bond the way hamiltonian_builder does:
// J * (SzSz + 0.5 S+S- + 0.5 S-S+).
void add_heisenberg_bond(TermStorage& t, std::uint64_t i, std::uint64_t j,
                         double J) {
    t.add_two_body(2, i, 2, j, Cx(J, 0.0));
    t.add_two_body(0, i, 1, j, Cx(0.5 * J, 0.0));
    t.add_two_body(1, i, 0, j, Cx(0.5 * J, 0.0));
}

}  // namespace

TEST_CASE("su2 detect: Heisenberg bonds pass, per-bond isotropy suffices",
          "[su2]") {
    TermStorage t;
    add_heisenberg_bond(t, 0, 1, 0.7);
    REQUIRE(hamiltonian_is_su2_symmetric(t));

    // A second bond with a different J is still SU(2)-invariant.
    add_heisenberg_bond(t, 1, 2, -1.3);
    REQUIRE(hamiltonian_is_su2_symmetric(t));

    // Identity shifts (site_1 == site_2 diagonal) are constants: ignored.
    t.add_diag_two_body(0, 0, Cx(3.0, 0.0));
    REQUIRE(hamiltonian_is_su2_symmetric(t));
}

TEST_CASE("su2 detect: reversed storage order aggregates correctly", "[su2]") {
    // Same bond, but stored with the larger site first:
    // S+_2 S-_0 and S-_2 S+_0 partner the (0,2) zz coupling.
    TermStorage t;
    t.add_two_body(2, 2, 2, 0, Cx(1.0, 0.0));
    t.add_two_body(0, 2, 1, 0, Cx(0.5, 0.0));
    t.add_two_body(1, 2, 0, 0, Cx(0.5, 0.0));
    REQUIRE(hamiltonian_is_su2_symmetric(t));
}

TEST_CASE("su2 detect: split transverse coefficients still aggregate", "[su2]") {
    // The transverse weight arrives in two half-strength records per
    // orientation; only the per-pair SUM must satisfy c = J/2.
    TermStorage t;
    t.add_two_body(2, 0, 2, 1, Cx(1.0, 0.0));
    t.add_two_body(0, 0, 1, 1, Cx(0.25, 0.0));
    t.add_two_body(1, 1, 0, 0, Cx(0.25, 0.0));  // same operator, swapped record
    t.add_two_body(1, 0, 0, 1, Cx(0.5, 0.0));
    REQUIRE(hamiltonian_is_su2_symmetric(t));
}

TEST_CASE("su2 detect: anisotropy breaks it", "[su2]") {
    // XXZ with Delta != 1.
    {
        TermStorage t;
        t.add_two_body(2, 0, 2, 1, Cx(1.5, 0.0));   // Delta = 1.5
        t.add_two_body(0, 0, 1, 1, Cx(0.5, 0.0));
        t.add_two_body(1, 0, 0, 1, Cx(0.5, 0.0));
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    // XY-only bond (no zz partner).
    {
        TermStorage t;
        t.add_two_body(0, 0, 1, 1, Cx(0.5, 0.0));
        t.add_two_body(1, 0, 0, 1, Cx(0.5, 0.0));
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    // Ising-only bond (no transverse partner).
    {
        TermStorage t;
        t.add_two_body(2, 0, 2, 1, Cx(1.0, 0.0));
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    // Unequal +- / -+ weights (chiral in-plane exchange).
    {
        TermStorage t;
        t.add_two_body(2, 0, 2, 1, Cx(1.0, 0.0));
        t.add_two_body(0, 0, 1, 1, Cx(0.7, 0.0));
        t.add_two_body(1, 0, 0, 1, Cx(0.3, 0.0));
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
}

TEST_CASE("su2 detect: fields and DM-type terms break it", "[su2]") {
    TermStorage base;
    add_heisenberg_bond(base, 0, 1, 1.0);

    {
        TermStorage t = base;
        t.add_one_body(2, 0, Cx(0.1, 0.0));         // Zeeman
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    {
        TermStorage t = base;
        t.add_one_body(0, 0, Cx(0.1, 0.0));         // transverse S+
        t.add_one_body(1, 0, Cx(0.1, 0.0));         // transverse S-
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    {
        TermStorage t = base;
        t.add_two_body(2, 0, 0, 1, Cx(0.1, 0.0));   // Sz S+ (DM/Gamma)
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    {
        TermStorage t = base;
        t.add_two_body(0, 0, 0, 1, Cx(0.1, 0.0));   // S+S+ anisotropy
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    {
        TermStorage t = base;
        t.add_three_body(2, 0, 2, 1, 2, 2, Cx(0.1, 0.0));  // conservative
        REQUIRE_FALSE(hamiltonian_is_su2_symmetric(t));
    }
    // Zero-coefficient noise terms must NOT flip the verdict.
    {
        TermStorage t = base;
        t.add_one_body(2, 0, Cx(0.0, 0.0));
        t.add_three_body(2, 0, 2, 1, 2, 2, Cx(0.0, 0.0));
        REQUIRE(hamiltonian_is_su2_symmetric(t));
    }
}

TEST_CASE("su2 detect: algebra containment su2 => U1 & flip & real", "[su2]") {
    TermStorage t;
    add_heisenberg_bond(t, 0, 1, 1.0);
    add_heisenberg_bond(t, 1, 2, 0.5);
    REQUIRE(hamiltonian_is_su2_symmetric(t));
    REQUIRE(ed::symmetry::sz_axis_of(t) == ed::symmetry::SzAxis::U1);
    REQUIRE(ed::symmetry::hamiltonian_is_spin_flip_symmetric(t));
    REQUIRE(t.is_real());
}
