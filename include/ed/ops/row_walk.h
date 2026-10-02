// =============================================================================
// include/ed/ops/row_walk.h -- the one walk over an operator's flip-grouped terms.
//
// for_each_connection(P, s, emit) calls emit(t, h) once for every group of the program P
// that acts on the basis state s, with t = s ^ F the group's target and h = <t|O|s> the sum
// of the group's terms whose condition s meets (one subgroup at most: its value is s & F).
// Groups come in ascending flip mask, so the diagonal (t == s) is emitted first; a group's
// terms are summed in their key order, so a state's elements equal MaskedOperator::to_dense
// bit for bit. h may be 0 (terms cancelling on s): callers that store elements decide.
//
// The lanes build rows, so they walk the program of O^dagger and conjugate:
// <s|O|t> = conj(<t|O^dagger|s>), exact for any O.
//
// Host and device: C is the complex type the program's coefficients are read as
// (std::complex<double> on the host, thrust::complex<double> in a kernel); it needs C(0),
// +=, and unary -.
// =============================================================================
#pragma once

#include <ed/ops/program.h>
#include <ed/ops/term.h>

#include <cstdint>

namespace ed::ops {

template <class C, class Emit>
ED_OPS_HD void for_each_connection(const ProgramView<C>& P, std::uint64_t s, Emit&& emit) {
    for (std::uint32_t g = 0; g < P.n_groups; ++g) {
        const std::uint64_t F = P.group_flip[g];
        const std::uint64_t v = s & F;
        if (P.group_setbits[g] >= 0 && masked_popcount(v) != P.group_setbits[g]) continue;
        // the subgroup whose value is v (values ascending within the group)
        std::uint32_t lo = P.group_vbegin[g], hi = P.group_vbegin[g + 1];
        while (lo < hi) {
            const std::uint32_t mid = lo + (hi - lo) / 2;
            if (P.vsub_val[mid] < v) lo = mid + 1; else hi = mid;
        }
        if (lo == P.group_vbegin[g + 1] || P.vsub_val[lo] != v) continue;
        C h(0);
        for (std::uint32_t k = P.vsub_tbegin[lo]; k < P.vsub_tbegin[lo + 1]; ++k) {
            if (masked_popcount(s & P.term_sign[k]) & 1) h += -P.term_coeff[k];
            else                                          h += P.term_coeff[k];
        }
        emit(s ^ F, h);
    }
}

}  // namespace ed::ops
