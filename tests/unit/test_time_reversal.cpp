// =============================================================================
// tests/unit/test_time_reversal.cpp
//
// Stage-6 guards (SymmetryEngine v2): the conjugate-sector pairing map
// and the real-Hamiltonian gate behind the time-reversal sector pairing.
//
//   * Z_N translations: partner(k) == (N - k) mod N; k = 0 and k = N/2
//     are self-conjugate.
//   * hamiltonian_is_real: real Heisenberg passes; any imaginary
//     coefficient (DM-like) fails.
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/matvec/term_storage.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/time_reversal.h>

#include <complex>

using Cx = std::complex<double>;
using namespace ed::symmetry;

TEST_CASE("conjugate_sector_pairing: Z_N momenta pair k <-> N-k",
          "[time_reversal]") {
    for (int N : {6, 8, 12}) {
        const SymmetryGroupInfo info = ed::sym::translation_group_1d(N);
        REQUIRE(static_cast<int>(info.sectors.size()) == N);
        const auto partner = conjugate_sector_pairing(info);
        REQUIRE(partner.size() == static_cast<std::size_t>(N));

        // Identify each sector's momentum from its quantum number.
        for (std::size_t s = 0; s < partner.size(); ++s) {
            REQUIRE(partner[s] >= 0);
            const int k  = info.sectors[s].quantum_numbers[0];
            const int pk = info.sectors[static_cast<std::size_t>(partner[s])]
                               .quantum_numbers[0];
            REQUIRE(pk == (N - k) % N);
            // Involution.
            REQUIRE(partner[static_cast<std::size_t>(partner[s])]
                    == static_cast<std::int32_t>(s));
        }
    }
}

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
