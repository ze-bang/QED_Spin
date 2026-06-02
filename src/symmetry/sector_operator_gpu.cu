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
    // callable. ``sector_n_up_()`` returns the shared magnetization of a
    // fixed-Sz sector (every orbit representative has the same popcount),
    // which selects the O(1) dense combinadic rank-table device lookup;
    // it returns -1 for a full-Hilbert (sym-only) sector, falling back to
    // the open-addressing hash that handles mixed-popcount states.
    return ed::symmetry::make_sector_matvec_gpu(
        sector_basis_.sector(),
        static_cast<double>(sector_basis_.group_size()),
        static_cast<double>(spin_l_),
        terms_,
        static_cast<int>(n_bits_),
        /*n_up=*/sector_n_up_());
}

#endif  // WITH_CUDA
