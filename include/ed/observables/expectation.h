#pragma once
// =============================================================================
// include/ed/observables/expectation.h
//
// expectation_value(O, psi): the canonical <psi|O|psi> primitive.
// Takes any operator that satisfies the MatVecOperator interface plus
// a `Backend` so that the apply + dot pair stays in the correct memory
// space.
//
// Phase-6 primitive 1 of 5. Replaces the inline patterns scattered
// across `ftlm.cpp:2037`, `TPQ.cpp:1201`, and similar call sites.
// =============================================================================

#include <complex>
#include <cstddef>

#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>

namespace ed::observables {

using Complex = std::complex<double>;

/// `out_O_psi` is a workspace buffer of dimension `local_n` (caller-
/// supplied so that batched calls can reuse it). The backend supplies
/// the inner-product reduction across MPI ranks where applicable.
template <typename Backend>
[[nodiscard]] Complex expectation_value(
    const Backend&                              backend,
    const ed::matvec::MatVecOperator&           O,
    const Complex*                              psi,
    Complex*                                    out_O_psi,
    std::size_t                                 local_n)
{
    O.apply(psi, out_O_psi, local_n);
    return backend.dot(psi, out_O_psi, local_n);
}

}  // namespace ed::observables
