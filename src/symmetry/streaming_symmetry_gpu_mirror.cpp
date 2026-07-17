// =============================================================================
// src/symmetry/streaming_symmetry_gpu_mirror.cpp
//
// CPU-only stub for the lazy GPU sector mirror entry points.
//
// When WITH_CUDA is OFF this TU is the sole provider of
// ``StreamingSymmetryOperator::bind_cuda_for_sector`` and
// ``FixedSzStreamingSymmetryOperator::bind_cuda_for_sector``. The real
// implementation lives in ``streaming_symmetry_gpu_mirror.cu`` (compiled
// into ``ed_solvers_gpu`` only when WITH_CUDA is ON). When WITH_CUDA is
// ON this file is an empty TU -- the strong definitions come from the
// .cu sibling.
//
// The stub throws ``std::logic_error`` with a clear message so callers
// that misroute to ``bind_cuda()`` on a non-CUDA build get a loud,
// localised failure rather than a silent fallback. The
// ``select_backend`` gate (Phase 1c plumbing) avoids calling this on
// a non-CUDA build because ``Geometry::supports_device_matvec`` is
// only set when WITH_CUDA is defined AND the build runtime has at
// least one GPU.
// =============================================================================

#ifndef WITH_CUDA

#include <ed/symmetry/sector_gpu_mirror.h>

#include <stdexcept>
#include <string>

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep(const ed::symmetry::RepSectorData& /*rep*/,
                                         double                         /*spin_l*/,
                                         const ed::matvec::TermStorage& /*terms*/) {
    throw std::logic_error(
        "ed::symmetry::make_sector_matvec_gpu_rep: built without WITH_CUDA. "
        "Rebuild with -DWITH_CUDA=ON to enable the on-the-fly representative "
        "GPU matvec, or route the workload through CpuBackend (device='cpu').");
}

ed::LinearOperator::MatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_hostptr(
    const ed::symmetry::RepSectorData& /*rep*/,
    double                             /*spin_l*/,
    const ed::matvec::TermStorage&     /*terms*/) {
    throw std::logic_error(
        "ed::symmetry::make_sector_matvec_gpu_rep_hostptr: built without "
        "WITH_CUDA. Rebuild with -DWITH_CUDA=ON, or route the workload "
        "through CpuBackend (device='cpu').");
}

#endif  // !WITH_CUDA
