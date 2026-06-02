// =============================================================================
// src/symmetry/sector_operator_gpu.cpp
//
// Non-CUDA throwing stub for ``ed::symmetry::SectorOperator::bind_cuda()``.
// Compiled into ed_core. On a WITH_CUDA build this TU is empty (the real
// definition is in ``sector_operator_gpu.cu`` -> ed_solvers_gpu).
//
// Callers reach ``bind_cuda()`` only through ``ed::select_backend``, which
// never picks the GPU lane unless ``Geometry::supports_device_matvec`` is
// set -- and ``SectorOperator::geometry()`` only sets that flag under
// WITH_CUDA. So on a non-CUDA build this throw is effectively unreachable;
// it exists to satisfy the out-of-line override declaration.
//
// Phase A of the operator-collapse GPU-parity work (Jun 2026).
// =============================================================================

#ifndef WITH_CUDA

#include <ed/symmetry/sector_operator.h>

#include <stdexcept>

ed::LinearOperator::MatvecFn
ed::symmetry::SectorOperator::bind_cuda() const {
    throw std::logic_error(
        "ed::symmetry::SectorOperator::bind_cuda: built without WITH_CUDA. "
        "Rebuild with -DWITH_CUDA=ON to enable the GPU symmetry mirror, "
        "or route the workload through CpuBackend (device='cpu').");
}

#endif  // !WITH_CUDA
