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
// (device mirror build + unified kernel launch) is shared with the legacy
// path via ``ed::symmetry::make_sector_matvec_gpu``.
//
// Phase A of the operator-collapse GPU-parity work (Jun 2026).
// =============================================================================

#ifdef WITH_CUDA

#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_gpu_mirror.h>

ed::LinearOperator::MatvecFn
ed::symmetry::SectorOperator::bind_cuda() const {
    // Bake any pending in-place transforms into ``terms_`` before the
    // device mirror snapshots the term SoA (mirrors the legacy
    // ``bind_cuda_for_sector`` ordering).
    commitPendingTransforms();

    // One sector == one mirror, built once and captured by the returned
    // callable. n_up = -1 selects the legacy open-addressing hash lookup,
    // which is correct for both full-Hilbert and fixed-Sz sectors (the
    // dense rank-table perf opt is deferred).
    return ed::symmetry::make_sector_matvec_gpu(
        sector_basis_.sector(),
        static_cast<double>(sector_basis_.group_size()),
        static_cast<double>(spin_l_),
        terms_,
        static_cast<int>(n_bits_),
        /*n_up=*/-1);
}

#endif  // WITH_CUDA
