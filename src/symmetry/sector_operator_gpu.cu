// =============================================================================
// src/symmetry/sector_operator_gpu.cu
//
// WITH_CUDA definition of ``ed::symmetry::SectorOperator::bind_cuda()``.
// Compiled into ``ed_solvers_gpu``. The matching non-CUDA throwing stub
// lives in ``sector_operator_gpu.cpp`` (ed_core).
//
// This is the GPU lane of the operator-collapse work: it makes a
// standalone ``SectorOperator`` backend-complete (CPU + GPU), matching
// the legacy ``StreamingSymmetryOperator::SectorView`` so the production
// sector loop can be cut over with zero caller changes. The heavy lifting
// (rep mirror build + unified kernel launch) lives in
// ``ed::symmetry::make_sector_matvec_gpu_rep``.
//
// Phase A of the operator-collapse GPU-parity work (Jun 2026).
// =============================================================================

#ifdef WITH_CUDA

#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_gpu_mirror.h>

#include <cstdlib>


// Operator-collapse Phase 4: SectorOperator is now the alias
// SubspaceOperator<SymmetryBasisPolicy, Host>; this strong definition of its
// bind_cuda_impl_ member specialization replaces the legacy out-of-line
// ed::symmetry::SectorOperator::bind_cuda. The orbit data + lazy-rep state come
// from the owned SectorBasis producer.
template <>
ed::LinearOperator::MatvecFn
ed::SubspaceOperator<ed::matvec::basis::SymmetryBasisPolicy,
                     ed::matvec::MemorySpace::Host>::bind_cuda_impl_() const {
    // Bake any pending in-place transforms into ``terms_`` before the
    // device mirror snapshots the term SoA (mirrors the legacy
    // ``bind_cuda_for_sector`` ordering).
    commitPendingTransforms();

    // ONE representation on device too (Stage 11c-2b): the on-the-fly
    // representative kernel. It was already preferred in BOTH regimes (it
    // skips the Pass-2 CSR build + the multi-GiB device upload; measured
    // N=24: rep 4.8 s vs CSR-mirror 18.2 s end-to-end); the legacy
    // orbit-CSR mirror and its ED_GPU_SYMMETRY_REP=0 escape are retired --
    // every production SectorBasis carries a rep provider (11c-1/2a), so
    // the escape was the only route in. Full-Hilbert (sym-only) sectors
    // ride the same kernel (identity rank + dense reverse table / hash).
    const ed::symmetry::RepSectorData& rd = producer_.ensureRepData();
    if (!rd.usable()) {
        throw std::runtime_error(
            "SectorOperator::bind_cuda_impl_: RepSectorData unusable -- every "
            "production SectorBasis carries a rep provider since Stage "
            "11c-2a, so this indicates a construction bug.");
    }
    return ed::symmetry::make_sector_matvec_gpu_rep(
        rd,
        static_cast<double>(spin_l_),
        terms_);
}

#endif  // WITH_CUDA
