// =============================================================================
// examples/04_cpp_gpu_lanczos.cpp
//
// Single-process GPU ground-state Lanczos via the unified ED interface
// (Full Unified-Interface Collapse, May 2026):
//
//     GPUOperator (constructed device-side)
//     ed::workflows::solve(*op, opts)         [auto-picks CudaBackend]
//
// Because `GPUOperator` is a device-resident `LinearOperator`
// subclass, `select_backend` picks the CudaBackend automatically
// (geometry().memory_space == Device). No `make_operator` indirection
// is needed; the GPU operator IS the LinearOperator the orchestrator
// consumes directly.
//
// Build requires CUDA (-DWITH_CUDA=ON in the parent build).
//
// Run:
//     ./build/examples/ex04_cpp_gpu_lanczos                # default N=16 PBC
//     ./build/examples/ex04_cpp_gpu_lanczos 20 0           # N=20 OBC
// =============================================================================

#include <ed/gpu/gpu_operator.cuh>
#include <ed/orchestrator.h>

#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
    const std::uint64_t N        = (argc > 1) ? std::stoull(argv[1]) : 16;
    const bool          periodic = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::cerr << "No CUDA device available.\n";
        return 1;
    }

    using Complex = std::complex<double>;
    const Complex Jz(1.0, 0.0);
    const Complex Jpm(0.5, 0.0);

    GPUOperator op(static_cast<int>(N), 0.5f);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, static_cast<std::uint32_t>(i),
                          2, static_cast<std::uint32_t>(j), Jz);
        op.addTwoBodyTerm(0, static_cast<std::uint32_t>(i),
                          1, static_cast<std::uint32_t>(j), Jpm);
        op.addTwoBodyTerm(1, static_cast<std::uint32_t>(i),
                          0, static_cast<std::uint32_t>(j), Jpm);
    }
    op.copyTransformDataToDevice();

    // Because `op.geometry().memory_space == Device`,
    // `select_backend` picks the CudaBackend automatically.
    ed::SolveOptions opts;
    opts.num_eigs        = 3;
    opts.method          = ed::SolveMethod::Lanczos;
    opts.tolerance       = 1e-10;
    opts.compute_vectors = false;

    auto result = ed::workflows::solve(op, opts);

    const std::uint64_t dim = 1ULL << N;
    std::cout << "GPU Lanczos  N=" << N
              << "  PBC=" << (periodic ? 1 : 0)
              << "  dim=" << dim << "\n";
    std::cout << "  backend lane      = " << result.backend.lane << "\n";
    std::cout << "  total_time        = " << result.backend.wall_seconds
              << " s\n";
    std::cout << "  iters_done        = " << result.krylov.iters_done << "\n";
    std::cout << "  residual_norm     = " << result.krylov.residual_norm
              << "\n";

    std::cout << "Lowest 3 eigenvalues:\n";
    for (std::size_t k = 0; k < result.eigenvalues.size() && k < 3; ++k) {
        std::cout << "  E[" << k << "] = " << result.eigenvalues[k] << "\n";
    }
    return 0;
}
