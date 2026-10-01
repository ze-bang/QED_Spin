// =============================================================================
// src/gpu/device_csr.cu -- device-resident complex CSR, y = A x (device_csr.h).
// =============================================================================

#include <ed/gpu/device_csr.h>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace ed::matvec {

namespace {

__global__ void csr_spmv(const std::int64_t* __restrict__ rp, const std::uint32_t* __restrict__ col,
                         const cuDoubleComplex* __restrict__ val, const cuDoubleComplex* __restrict__ x,
                         cuDoubleComplex* __restrict__ y, std::size_t rows) {
    const std::size_t r = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (r >= rows) return;
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (std::int64_t q = rp[r]; q < rp[r + 1]; ++q) acc = cuCfma(val[q], x[col[q]], acc);
    y[r] = acc;
}

struct DeviceCsr {
    thrust::device_vector<std::int64_t>    rp;
    thrust::device_vector<std::uint32_t>   col;
    thrust::device_vector<cuDoubleComplex> val;
    std::size_t rows = 0;
};

}  // namespace

DeviceMatvecFn make_device_csr_matvec(const std::int64_t* row_ptr, const std::uint32_t* col,
                                      const std::complex<double>* val, std::size_t rows,
                                      std::size_t nnz) {
    auto m = std::make_shared<DeviceCsr>();
    m->rows = rows;
    m->rp.assign(row_ptr, row_ptr + rows + 1);
    m->col.assign(col, col + nnz);
    const auto* v = reinterpret_cast<const cuDoubleComplex*>(val);
    m->val.assign(v, v + nnz);
    return [m](const std::complex<double>* x, std::complex<double>* y, std::size_t n) {
        if (n != m->rows)
            throw std::invalid_argument("device CSR: output length " + std::to_string(n)
                                        + " != rows " + std::to_string(m->rows));
        if (m->rows == 0) return;
        const unsigned threads = 256;
        const auto blocks = static_cast<unsigned>((m->rows + threads - 1) / threads);
        csr_spmv<<<blocks, threads>>>(thrust::raw_pointer_cast(m->rp.data()),
                                      thrust::raw_pointer_cast(m->col.data()),
                                      thrust::raw_pointer_cast(m->val.data()),
                                      reinterpret_cast<const cuDoubleComplex*>(x),
                                      reinterpret_cast<cuDoubleComplex*>(y), m->rows);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("device CSR spmv: ") + cudaGetErrorString(err));
    };
}

}  // namespace ed::matvec
