#pragma once
// =============================================================================
// include/ed/operators/casimir.h
//
// The SU(2) Casimir S^2_tot = (S_tot)^2 as an `::Operator` carrier, the
// operator the total-spin (Lowdin) projections and block restrictions use.
//
// Operator identity (spin-1/2, N sites):
//
//   S^2 = sum_i S_i^2 + 2 sum_{i<j} S_i . S_j
//       = (3N/4) Id + sum_{i<j} [ 2 Sz_i Sz_j + S+_i S-_j + S-_i S+_j ]
//
// Every piece is expressible in the standard term ABI, so the same CPU/GPU
// term kernels that apply H in any symmetry-adapted basis (rep/orbit per
// momentum x irrep, flip-projected, little-group isotypic via lift) apply
// S^2 there too, because [S^2, g] = 0 for every site permutation g, every
// flip mask, and Sz (S^2 conserves popcount term-by-term).
//
// The (3N/4) Id shift needs NO new kernel: a diag_two_body term with
// site_1 == site_2 evaluates through diag_two_body_factor (term_gate_math.h)
// to spin_sq * sign^2 = 1/4 for EVERY basis state, on the CPU and GPU gate
// math alike. N such terms with coefficient 3.0 are exactly (3N/4) Id.
//
// Cost model: S^2 carries ~1.5*N^2 terms vs ~3*z*N for a short-range H, so
// one S^2 matvec costs about (N/2z) H-matvecs.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <ed/core/operator.h>
#include <ed/core/linear_operator.h>
#include <ed/matvec/term_storage.h>

namespace ed::ops {

// ---------------------------------------------------------------------------
// Term emission
// ---------------------------------------------------------------------------

/// S^2_tot as a full `::Operator` carrier (AoS terms, spin-1/2), consumable
/// by every per-sector factory that restricts H itself: the rep-basis
/// RepSectorMatVec (lg_walk.h block_operator builds it over a block basis) and
/// the dense assembly paths.
[[nodiscard]] inline std::shared_ptr<::Operator>
make_S2_carrier(std::uint64_t n_sites) {
    using Cx = std::complex<double>;
    auto op = std::make_shared<::Operator>(n_sites, 0.5f);
    for (std::uint64_t i = 0; i < n_sites; ++i) {
        op->addTwoBodyTerm(2, i, 2, i, Cx(3.0, 0.0));
    }
    for (std::uint64_t i = 0; i < n_sites; ++i) {
        for (std::uint64_t j = i + 1; j < n_sites; ++j) {
            op->addTwoBodyTerm(2, i, 2, j, Cx(2.0, 0.0));
            op->addTwoBodyTerm(0, i, 1, j, Cx(1.0, 0.0));
            op->addTwoBodyTerm(1, i, 0, j, Cx(1.0, 0.0));
        }
    }
    return op;
}

}  // namespace ed::ops
