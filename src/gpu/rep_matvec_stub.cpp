// =============================================================================
// src/gpu/rep_matvec_stub.cpp
//
// CPU-only stubs for the GPU sector-matvec factories declared in
// ed/gpu/rep_matvec.h.
//
// When WITH_CUDA is OFF this TU is the sole provider of the
// ``make_sector_matvec_gpu_rep*`` factories and the device CSR functions. The real implementation lives
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

ed::LinearOperator::MatvecFn
ed::symmetry::make_cross_matvec_gpu_rep(const ed::symmetry::RepSectorData& /*src*/,
                                        const ed::symmetry::RepSectorData& /*tgt*/,
                                        const ed::ops::MaskedProgram& /*rows*/) {
    throw std::logic_error("ed::symmetry::make_cross_matvec_gpu_rep: built without WITH_CUDA.");
}

ed::LinearOperator::MultiMatvecFn
ed::symmetry::make_sector_matvec_gpu_rep_multi(const ed::symmetry::RepSectorData& /*rep*/, const ed::ops::MaskedProgram& /*rows*/) {
    throw std::logic_error(
        "ed::symmetry::make_sector_matvec_gpu_rep_multi: built without WITH_CUDA.");
}

std::shared_ptr<const ed::symmetry::DeviceCsr>
ed::symmetry::build_sector_csr_gpu(const ed::symmetry::RepSectorData& /*rep*/, const ed::ops::MaskedProgram& /*rows*/,
                                   std::uint64_t /*max_bytes*/) {
    throw std::logic_error("ed::symmetry::build_sector_csr_gpu: built without WITH_CUDA.");
}

std::shared_ptr<const ed::symmetry::DeviceCsr>
ed::symmetry::upload_csr_gpu(const ed::matvec::ReducedSymmetryCsr<std::complex<double>>& /*csr*/,
                             std::uint64_t /*max_bytes*/) {
    throw std::logic_error("ed::symmetry::upload_csr_gpu: built without WITH_CUDA.");
}

ed::LinearOperator::MatvecFn ed::symmetry::csr_matvec_gpu(std::shared_ptr<const ed::symmetry::DeviceCsr> /*csr*/) {
    throw std::logic_error("ed::symmetry::csr_matvec_gpu: built without WITH_CUDA.");
}

ed::LinearOperator::MultiMatvecFn
ed::symmetry::csr_matvec_gpu_multi(std::shared_ptr<const ed::symmetry::DeviceCsr> /*csr*/) {
    throw std::logic_error("ed::symmetry::csr_matvec_gpu_multi: built without WITH_CUDA.");
}

ed::symmetry::DeviceCsrInfo ed::symmetry::device_csr_info(const ed::symmetry::DeviceCsr& /*csr*/) {
    throw std::logic_error("ed::symmetry::device_csr_info: built without WITH_CUDA.");
}

ed::matvec::ReducedSymmetryCsr<std::complex<double>> ed::symmetry::download_csr(const ed::symmetry::DeviceCsr& /*csr*/) {
    throw std::logic_error("ed::symmetry::download_csr: built without WITH_CUDA.");
}

#endif  // !WITH_CUDA
