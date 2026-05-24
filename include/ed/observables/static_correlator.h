#pragma once
// =============================================================================
// include/ed/observables/static_correlator.h
//
// static_correlator(A, B, psi):  S_{AB} = <psi| A^† B |psi>.
// Accepts square (`MatVecOperator`) A and B; the dimension contract is
// checked at runtime by the backing kernel.
//
// Phase-6 primitive 2 of 5. Replaces the `compute_static_response`,
// `compute_sssf_jp`, and `compute_sssf_ltlm` patterns in the retired
// dynamical kernel files. The rectangular-operator variant described
// in the original draft was retired in May 2026 along with the unused
// `ed::core::RectangularOperator<MS>` wrapper; cross-sector
// correlators continue to call into `ed::dssf::CrossSectorObservable`
// directly.
// =============================================================================

#include <complex>
#include <cstddef>
#include <vector>

#include <ed/matvec/backend.h>
#include <ed/matvec/matvec.h>

namespace ed::observables {

using Complex = std::complex<double>;

/// `psi` has dimension `local_n`. The two workspace buffers
/// `out_B_psi` and `out_AhB_psi` are caller-supplied for batching.
/// Returns S_{AB} = <psi| A^† B |psi> reduced across MPI ranks (via
/// `backend.dot()`).
template <typename Backend>
[[nodiscard]] Complex static_correlator(
    const Backend&                              backend,
    const ed::matvec::MatVecOperator&           A,
    const ed::matvec::MatVecOperator&           B,
    const Complex*                              psi,
    Complex*                                    out_B_psi,
    Complex*                                    out_AhB_psi,
    std::size_t                                 local_n)
{
    // S_{AB} = <psi|A^† B|psi>.
    // For Hermitian A: <A psi| B psi>; for non-Hermitian A we apply
    // A from the right (A psi) and dot with B psi using the conjugate
    // of the right-hand side -- backend.dot conjugates its first arg.
    B.apply(psi, out_B_psi, local_n);
    A.apply(psi, out_AhB_psi, local_n);
    return backend.dot(out_AhB_psi, out_B_psi, local_n);
}

}  // namespace ed::observables
