#pragma once
// =============================================================================
// include/ed/core/fixed_sz_operator.h
//
// FixedSzOperator: Operator restricted to a fixed total-Sz sector.
// Reduces Hilbert space dimension from 2^N to C(N, N_up).
//
// Operator-collapse Phase 4 (Jun 2026): FixedSzOperator is now a using-alias
// over the collapsed ``ed::SubspaceOperator<FixedSzBasisPolicy>`` template,
// which owns a ``FixedSzSubspace`` producer (the sorted basis + Lin (1990) O(1)
// index table) and routes its matvec through ``CpuMatVecBackend<
// FixedSzBasisPolicy>``. Construction sites (``FixedSzOperator(n_bits, spin_l,
// n_up)``) and the full public surface (getNUp / getBasisStates /
// projectToFixedSz / embedToFull / getFixedSzMatrix / subspace / ...) are
// unchanged.
//
// Depends on: subspace_operator.h (-> operator.h, subspace.h).
// =============================================================================

#include <memory>

#include <ed/core/subspace_operator.h>
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>

// Back-compat alias: FixedSzOperator == SubspaceOperator<FixedSzBasisPolicy>.
using FixedSzOperator =
    ed::SubspaceOperator<ed::matvec::basis::FixedSzBasisPolicy>;

namespace ed {

// ---------------------------------------------------------------------------
// make_backend_ specialization (FixedSz lane). A CpuMatVecBackend over the
// FixedSzBasisPolicy view onto the producer's (basis_states, lin_index) tables.
// ---------------------------------------------------------------------------
template <>
inline std::unique_ptr<ed::matvec::MatVecBackendBase>
SubspaceOperator<ed::matvec::basis::FixedSzBasisPolicy,
                 ed::matvec::MemorySpace::Host>::make_backend_() const {
    return ed::matvec::make_cpu_fixed_sz_backend<
        DiagonalOneBody, OffDiagonalOneBody,
        DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
        ThreeBodyTransformData>(producer_.basis_states(), producer_.lin_index());
}

#ifdef WITH_CUDA
// Device matvec build for the FixedSz lane. Strong definition in
// src/core/operator_gpu.cu (ed_solvers_gpu); weak CPU fallback in
// src/core/operator_gpu.cpp (ed_core) so CPU-only binaries still link.
template <>
ed::LinearOperator::MatvecFn
SubspaceOperator<ed::matvec::basis::FixedSzBasisPolicy,
                 ed::matvec::MemorySpace::Host>::bind_cuda_impl_() const;
#endif

}  // namespace ed
