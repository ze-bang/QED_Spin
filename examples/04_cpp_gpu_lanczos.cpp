// =============================================================================
// examples/04_cpp_gpu_lanczos.cpp
//
// GPU ground-state Lanczos via `GPUOperator` + `GPULanczos`. Same
// 1D Heisenberg chain as the CPU example, run end-to-end on the
// GPU (cuSPARSE for SpMV, cuBLAS for the inner products).
//
// Build requires CUDA (-DWITH_CUDA=ON in the parent build).
//
// Run:
//     ./build/ex04_cpp_gpu_lanczos                # default N=16 PBC
//     ./build/ex04_cpp_gpu_lanczos 20 0           # N=20 OBC
// =============================================================================

#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/gpu_operator.cuh>

#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char** argv) {
    const std::uint64_t N        = (argc > 1) ? std::stoull(argv[1]) : 16;
    const bool          periodic = (argc > 2) ? (std::atoi(argv[2]) != 0) : true;

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::cerr << "No CUDA device available.\n";
        return 1;
    }

    auto op = std::make_unique<GPUOperator>(static_cast<int>(N), /*spin=*/0.5f);
    using Complex = std::complex<double>;
    const Complex Jz(1.0, 0.0);
    const Complex Jpm(0.5, 0.0);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op->addTwoBodyTerm(/*op1=*/2, static_cast<std::uint32_t>(i),
                           /*op2=*/2, static_cast<std::uint32_t>(j), Jz);
        op->addTwoBodyTerm(/*op1=*/0, static_cast<std::uint32_t>(i),
                           /*op2=*/1, static_cast<std::uint32_t>(j), Jpm);
        op->addTwoBodyTerm(/*op1=*/1, static_cast<std::uint32_t>(i),
                           /*op2=*/0, static_cast<std::uint32_t>(j), Jpm);
    }
    op->copyTransformDataToDevice();

    GPULanczos lan(op.get(), /*max_iter=*/200, /*tolerance=*/1e-10);
    std::vector<double> eigenvalues;
    std::vector<std::vector<std::complex<double>>> eigenvectors;
    lan.run(/*num_eigenvalues=*/3, eigenvalues, eigenvectors,
            /*compute_vectors=*/false);

    const std::uint64_t dim = 1ULL << N;
    std::cout << "GPU Lanczos  N=" << N
              << "  PBC=" << (periodic ? 1 : 0)
              << "  dim=" << dim << "\n";

    auto stats = lan.getStats();
    std::cout << "  iterations         = " << stats.iterations           << "\n";
    std::cout << "  total_time         = " << stats.total_time   << " s\n";
    std::cout << "  matvec_time        = " << stats.matvec_time  << " s\n";
    std::cout << "  ortho_time         = " << stats.ortho_time   << " s\n";
    std::cout << "  convergence_error  = " << stats.convergence_error    << "\n";

    std::cout << "Lowest 3 eigenvalues:\n";
    for (std::size_t k = 0; k < eigenvalues.size() && k < 3; ++k) {
        std::cout << "  E[" << k << "] = " << eigenvalues[k] << "\n";
    }
    return 0;
}
