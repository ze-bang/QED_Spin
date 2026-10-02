// =============================================================================
// src/gpu/rep_matvec_stub.cpp
//
// CPU-only stubs for the GPU sector-matvec factories declared in
// ed/gpu/rep_matvec.h.
//
// When WITH_CUDA is OFF this TU is the sole provider of the
// ``make_sector_matvec_gpu_rep*`` factories. The real implementation lives
// in ``rep_matvec.cu`` (compiled into ``ed_solvers_gpu``
// only when WITH_CUDA is ON). When WITH_CUDA is ON this file is an empty
// TU -- the strong definitions come from the .cu sibling.
//
// The stubs throw ``std::logic_error`` with a clear message so callers
// that misroute to ``bind_cuda()`` on a non-CUDA build get a loud,
// localised failure rather than a silent fallback. Nothing calls them on a
// non-CUDA build: has_device_kernel() is false there, so place() never
// returns a device lane.
// =============================================================================

#ifndef WITH_CUDA

#include <ed/gpu/rep_matvec.h>

#include <stdexcept>
#include <string>

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep(const ed::symmetry::RepSectorData& /*rep*/, const ed::ops::MaskedProgram& /*rows*/) {
    throw std::logic_error(
        "ed::symmetry::make_sector_matvec_gpu_rep: built without WITH_CUDA. "
        "Rebuild with -DWITH_CUDA=ON to enable the on-the-fly representative "
        "GPU matvec, or route the workload through CpuBackend (device='cpu').");
}

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
    const ed::symmetry::RepSectorData& /*rep*/, const ed::ops::MaskedProgram& /*rows*/) {
    throw std::logic_error(
        "ed::symmetry::make_sector_matvec_gpu_rep_hostptr: built without "
        "WITH_CUDA. Rebuild with -DWITH_CUDA=ON, or route the workload "
        "through CpuBackend (device='cpu').");
}

ed::LinearOperator::MultiMatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_multi(const ed::symmetry::RepSectorData& /*rep*/, const ed::ops::MaskedProgram& /*rows*/) {
    throw std::logic_error(
        "ed::symmetry::make_sector_matvec_gpu_rep_multi: built without WITH_CUDA.");
}

#endif  // !WITH_CUDA
