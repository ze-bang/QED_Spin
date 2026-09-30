// =============================================================================
// tests/unit/test_time_reversal.cpp
//
// Stage-6 guard (SymmetryEngine v2): the real-Hamiltonian gate behind the
// time-reversal sector pairing.
//
//   * hamiltonian_is_real: real Heisenberg passes; any imaginary
//     coefficient (DM-like) fails.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/matvec/term_storage.h>
#include <ed/symmetry/time_reversal.h>

#include <complex>

using Cx = std::complex<double>;
using namespace ed::symmetry;

TEST_CASE("hamiltonian_is_real: real couplings pass, imaginary fail",
          "[time_reversal]") {
    ed::matvec::TermStorage t;
    t.diag_two_body.push_back({0, 1, Cx(1.0, 0.0)});
    t.offdiag_two_body.push_back({0, 1, 0, 1, Cx(0.5, 0.0)});
    t.offdiag_two_body.push_back({0, 1, 1, 0, Cx(0.5, 0.0)});
    REQUIRE(hamiltonian_is_real(t));

    // DM-like: imaginary coefficient on S+S- breaks conjugation symmetry.
    t.offdiag_two_body.push_back({1, 2, 0, 1, Cx(0.0, 0.3)});
    REQUIRE_FALSE(hamiltonian_is_real(t));
}
