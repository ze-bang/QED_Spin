#ifdef WITH_CUDA

// Prevent inclusion of CPU Operator class
#define CONSTRUCT_HAM_H

#include <ed/gpu/gpu_operator.cuh>
#include <cusolverDn.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <complex>

#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        std::cerr << "cuSOLVER error at " << __FILE__ << ":" << __LINE__ \
                  << " - status = " << status << std::endl; \
        throw std::runtime_error("cuSOLVER call failed"); \
    } \
} while(0)
#endif

using namespace GPUConfig;

// ============================================================================
// GPU Full Diagonalization
// ============================================================================
//
// Strategy:
// 1. Build the dense Hamiltonian matrix on GPU column-by-column:
//    For each column j, compute H * e_j  via the GPUOperator matvec kernel.
//    This produces column j of H_dense directly in device memory.
//
// 2. Call cuSOLVER's zheevd (divide-and-conquer Hermitian eigensolver)
//    to obtain ALL eigenvalues (and optionally eigenvectors).
//
// Memory: O(N^2) for the dense matrix on GPU.
// The matrix is stored column-major for cuSOLVER compatibility.
// ============================================================================

namespace {

/**
 * @brief CUDA kernel to set a unit vector: d_vec[idx] = 1.0, all others = 0.0
 */
__global__ void setUnitVectorKernel(cuDoubleComplex* d_vec, int N, int idx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_vec[i] = (i == idx) ? make_cuDoubleComplex(1.0, 0.0)
                              : make_cuDoubleComplex(0.0, 0.0);
    }
}

} // anonymous namespace

/**
 * @brief Build the dense Hamiltonian matrix on GPU and diagonalize using cuSOLVER
 *
 * @param gpu_op       Pointer to GPUOperator (or subclass) with Hamiltonian terms loaded
 * @param N            Hilbert space dimension (or sector dimension)
 * @param num_eigs     Number of lowest eigenvalues to return (0 = all)
 * @param eigenvalues  Output vector of eigenvalues (sorted ascending)
 * @param eigenvectors Output vector-of-vectors of eigenvectors (empty if not requested)
 * @param compute_eigenvectors Whether to compute and return eigenvectors
 */
void gpuFullDiagonalization(
    GPUOperator* gpu_op,
    int N,
    int num_eigs,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<std::complex<double>>>& eigenvectors,
    bool compute_eigenvectors
) {
    auto total_start = std::chrono::high_resolution_clock::now();

    if (num_eigs <= 0 || num_eigs > N) num_eigs = N;

    std::cout << "GPU Full Diagonalization: N=" << N << std::endl;
    std::cout << "  Dense matrix: " << std::fixed << std::setprecision(2)
              << (static_cast<double>(N) * N * sizeof(cuDoubleComplex)) / (1024.0 * 1024.0 * 1024.0)
              << " GB" << std::defaultfloat << std::endl;

    // ===== Allocate GPU memory =====
    cuDoubleComplex* d_dense = nullptr;     // N x N dense Hamiltonian (column-major)
    cuDoubleComplex* d_unit  = nullptr;     // unit vector e_j
    cuDoubleComplex* d_col   = nullptr;     // H * e_j result
    double*          d_evals = nullptr;     // eigenvalues (N doubles)
    cuDoubleComplex* d_work  = nullptr;     // cuSOLVER workspace
    double*          d_rwork = nullptr;     // cuSOLVER real workspace
    int*             d_info  = nullptr;     // cuSOLVER info

    size_t dense_bytes = static_cast<size_t>(N) * N * sizeof(cuDoubleComplex);
    size_t vec_bytes   = static_cast<size_t>(N) * sizeof(cuDoubleComplex);

    CUDA_CHECK(cudaMalloc(&d_dense, dense_bytes));
    CUDA_CHECK(cudaMalloc(&d_unit,  vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_col,   vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_evals, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info,  sizeof(int)));

    // ===== Build dense matrix column-by-column =====
    auto build_start = std::chrono::high_resolution_clock::now();
    std::cout << "  Building dense matrix..." << std::flush;

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Determine if the operator has its own input/output vectors
    // We use the operator's matVec interface which expects device pointers.
    // We'll set the unit vector in d_unit, copy to operator's input,
    // call matVec, and copy the result into the dense matrix column.

    const int progress_interval = std::max(1, N / 20);  // Report every 5%

    for (int j = 0; j < N; ++j) {
        // Set d_unit = e_j on device
        setUnitVectorKernel<<<blocks, threads>>>(d_unit, N, j);

        // Apply H: d_col = H * d_unit
        // Use the operator's matVec which handles its own input/output buffers
        gpu_op->matVecGPU(d_unit, d_col, N);

        // Copy result column into d_dense[:, j] (column-major => offset j*N)
        CUDA_CHECK(cudaMemcpy(d_dense + static_cast<size_t>(j) * N, d_col,
                              vec_bytes, cudaMemcpyDeviceToDevice));

        if (j % progress_interval == 0 || j == N - 1) {
            double pct = 100.0 * (j + 1) / N;
            std::cout << "\r  Building dense matrix... " << std::fixed
                      << std::setprecision(1) << pct << "%" << std::flush;
        }
    }
    std::cout << std::endl;

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "  Matrix build time: " << std::fixed << std::setprecision(3)
              << build_time << " s" << std::defaultfloat << std::endl;

    // ===== cuSOLVER setup =====
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // Query workspace size
    cusolverEigMode_t jobz = compute_eigenvectors ? CUSOLVER_EIG_MODE_VECTOR
                                                  : CUSOLVER_EIG_MODE_NOVECTOR;
    int work_size = 0;
    CUSOLVER_CHECK(cusolverDnZheevd_bufferSize(
        cusolver_handle, jobz, CUBLAS_FILL_MODE_UPPER,
        N, d_dense, N, d_evals, &work_size));

    CUDA_CHECK(cudaMalloc(&d_work, work_size * sizeof(cuDoubleComplex)));

    std::cout << "  cuSOLVER workspace: " << std::fixed << std::setprecision(2)
              << (work_size * sizeof(cuDoubleComplex)) / (1024.0 * 1024.0)
              << " MB" << std::defaultfloat << std::endl;

    // ===== Diagonalize =====
    auto diag_start = std::chrono::high_resolution_clock::now();
    std::cout << "  Running cuSOLVER zheevd..." << std::flush;

    CUSOLVER_CHECK(cusolverDnZheevd(
        cusolver_handle, jobz, CUBLAS_FILL_MODE_UPPER,
        N, d_dense, N, d_evals, d_work, work_size, d_info));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for errors
    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        std::cerr << "\n  cusolverDnZheevd failed with info=" << h_info << std::endl;
        // Cleanup
        cudaFree(d_dense); cudaFree(d_unit); cudaFree(d_col);
        cudaFree(d_evals); cudaFree(d_work); cudaFree(d_info);
        cusolverDnDestroy(cusolver_handle);
        throw std::runtime_error("cuSOLVER zheevd failed (info=" + std::to_string(h_info) + ")");
    }

    auto diag_end = std::chrono::high_resolution_clock::now();
    double diag_time = std::chrono::duration<double>(diag_end - diag_start).count();
    std::cout << " done (" << std::fixed << std::setprecision(3)
              << diag_time << " s)" << std::defaultfloat << std::endl;

    // ===== Copy eigenvalues to host =====
    std::vector<double> all_evals(N);
    CUDA_CHECK(cudaMemcpy(all_evals.data(), d_evals, N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    eigenvalues.assign(all_evals.begin(), all_evals.begin() + num_eigs);

    // ===== Copy eigenvectors to host (if requested) =====
    if (compute_eigenvectors) {
        // After zheevd, d_dense contains eigenvectors in column-major order.
        // Eigenvector i is in column i: d_dense[i*N .. (i+1)*N-1]
        eigenvectors.resize(num_eigs);
        std::vector<cuDoubleComplex> h_evec(N);

        for (int i = 0; i < num_eigs; ++i) {
            CUDA_CHECK(cudaMemcpy(h_evec.data(),
                                  d_dense + static_cast<size_t>(i) * N,
                                  vec_bytes, cudaMemcpyDeviceToHost));
            eigenvectors[i].resize(N);
            for (int k = 0; k < N; ++k) {
                eigenvectors[i][k] = std::complex<double>(h_evec[k].x, h_evec[k].y);
            }
        }
    }

    // ===== Cleanup =====
    cudaFree(d_dense);
    cudaFree(d_unit);
    cudaFree(d_col);
    cudaFree(d_evals);
    cudaFree(d_work);
    cudaFree(d_info);
    if (d_rwork) cudaFree(d_rwork);
    cusolverDnDestroy(cusolver_handle);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\nGPU Full Diagonalization Statistics:" << std::endl;
    std::cout << "  Matrix build: " << std::fixed << std::setprecision(3) << build_time << " s" << std::endl;
    std::cout << "  cuSOLVER:     " << diag_time << " s" << std::endl;
    std::cout << "  Total:        " << total_time << " s" << std::defaultfloat << std::endl;
    std::cout << "  Eigenvalues returned: " << eigenvalues.size() << " / " << N << std::endl;
    if (!eigenvalues.empty()) {
        std::cout << "  Ground state: " << eigenvalues[0] << std::endl;
    }
}

#endif // WITH_CUDA
