#pragma once
// =============================================================================
// include/ed/gpu/device_csr.h
//
// A general complex CSR matrix resident on a CUDA device, applied as y = A x with
// device pointers (one thread per row). Used for rectangular operators between two
// sectors (the cross-sector probe of the dynamics kernels) whose host CSR is built
// once and then applied many times per sample.
// =============================================================================

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace ed::matvec {

using DeviceMatvecFn = std::function<void(const std::complex<double>*, std::complex<double>*,
                                          std::size_t)>;

/// Upload a host CSR (rows x cols, row_ptr has rows + 1 entries) and return y = A x over
/// device pointers; the device copy lives as long as the returned function. CUDA builds only.
[[nodiscard]] DeviceMatvecFn
make_device_csr_matvec(const std::int64_t* row_ptr, const std::uint32_t* col,
                       const std::complex<double>* val, std::size_t rows, std::size_t nnz);

}  // namespace ed::matvec
