#include <ed/gpu/gpu_cg.cuh>
#include <ed/gpu/gpu_operator.cuh>
#include <ed/core/hdf5_io.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Kernel to initialize random vectors
__global__ void initRandomVectorsKernel(cuDoubleComplex* vectors, int N, int num_vecs, 
                                        unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = blockIdx.y;
    
    if (idx < N && vec_idx < num_vecs) {
        curandState rand_state;
        curand_init(seed + vec_idx * N + idx, idx, 0, &rand_state);
        
        double real = curand_normal_double(&rand_state);
        double imag = curand_normal_double(&rand_state);
        vectors[vec_idx * N + idx] = make_cuDoubleComplex(real, imag);
    }
}

// Constructor
GPUIterativeSolver::GPUIterativeSolver(GPUOperator* gpu_op, int N)
    : gpu_op_(gpu_op), N_(N), d_V_(nullptr), d_AV_(nullptr), d_work_(nullptr),
      d_subspace_H_(nullptr), d_subspace_eigs_(nullptr), d_residual_norms_(nullptr),
      d_info_(nullptr), lwork_(0) {
    
    // Create cuBLAS handle
    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed!" << std::endl;
        throw std::runtime_error("cuBLAS init failed");
    }
    
    // Create cuSOLVER handle
    cusolverStatus_t solver_stat = cusolverDnCreate(&cusolver_handle_);
    if (solver_stat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER initialization failed!" << std::endl;
        throw std::runtime_error("cuSOLVER init failed");
    }
    
    // Initialize stats
    stats_.total_time = 0.0;
    stats_.matvec_time = 0.0;
    stats_.ortho_time = 0.0;
    stats_.subspace_time = 0.0;
    stats_.iterations = 0;
    stats_.throughput = 0.0;
}

// Destructor
GPUIterativeSolver::~GPUIterativeSolver() {
    freeMemory();
    cublasDestroy(cublas_handle_);
    cusolverDnDestroy(cusolver_handle_);
}

void GPUIterativeSolver::allocateMemory(int max_subspace) {
    // Allocate memory for subspace vectors
    cudaMalloc(&d_V_, N_ * max_subspace * sizeof(cuDoubleComplex));
    cudaMalloc(&d_AV_, N_ * max_subspace * sizeof(cuDoubleComplex));
    cudaMalloc(&d_subspace_H_, max_subspace * max_subspace * sizeof(cuDoubleComplex));
    cudaMalloc(&d_subspace_eigs_, max_subspace * sizeof(double));
    cudaMalloc(&d_residual_norms_, max_subspace * sizeof(double));
    cudaMalloc(&d_info_, sizeof(int));
    
    // Query workspace size for cuSOLVER
    // Note: We need to query with dummy pointers, actual matrix will be provided later
    cuDoubleComplex* dummy_A;
    double* dummy_W;
    cudaMalloc(&dummy_A, max_subspace * max_subspace * sizeof(cuDoubleComplex));
    cudaMalloc(&dummy_W, max_subspace * sizeof(double));
    
    cusolverDnZheevd_bufferSize(cusolver_handle_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
                                 max_subspace, dummy_A, max_subspace, 
                                 dummy_W, &lwork_);
    cudaMalloc(&d_work_, lwork_ * sizeof(cuDoubleComplex));
    
    cudaFree(dummy_A);
    cudaFree(dummy_W);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "GPU memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("GPU malloc failed");
    }
}

void GPUIterativeSolver::freeMemory() {
    if (d_V_) cudaFree(d_V_);
    if (d_AV_) cudaFree(d_AV_);
    if (d_work_) cudaFree(d_work_);
    if (d_subspace_H_) cudaFree(d_subspace_H_);
    if (d_subspace_eigs_) cudaFree(d_subspace_eigs_);
    if (d_residual_norms_) cudaFree(d_residual_norms_);
    if (d_info_) cudaFree(d_info_);
}

void GPUIterativeSolver::initializeRandomVectors(cuDoubleComplex* vectors, int num_vecs) {
    dim3 blockSize(256, 1);
    dim3 numBlocks((N_ + blockSize.x - 1) / blockSize.x, num_vecs);
    
    initRandomVectorsKernel<<<numBlocks, blockSize>>>(vectors, N_, num_vecs, 123456789ULL);
    cudaDeviceSynchronize();
    
    // Orthogonalize
    orthogonalize(vectors, num_vecs);
}

void GPUIterativeSolver::gramSchmidt(cuDoubleComplex* vec, cuDoubleComplex* basis, int num_basis) {
    // Use iterative classical Gram-Schmidt (CGS with reorthogonalization)
    // This is more numerically stable for large systems
    const int max_reorth = 2;  // Typically 2 passes is sufficient
    
    for (int reorth = 0; reorth < max_reorth; ++reorth) {
        for (int i = 0; i < num_basis; ++i) {
            // Compute <basis[i] | vec>
            cuDoubleComplex proj;
            cublasZdotc(cublas_handle_, N_,
                        basis + i * N_, 1,
                        vec, 1,
                        &proj);
            
            // vec -= proj * basis[i]
            cuDoubleComplex neg_proj = make_cuDoubleComplex(-cuCreal(proj), -cuCimag(proj));
            cublasZaxpy(cublas_handle_, N_, &neg_proj, basis + i * N_, 1, vec, 1);
        }
    }
    
    // Normalize vec
    double norm;
    cublasDznrm2(cublas_handle_, N_, vec, 1, &norm);
    if (norm > 1e-14) {
        cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
        cublasZscal(cublas_handle_, N_, &scale, vec, 1);
    }
}

void GPUIterativeSolver::orthogonalize(cuDoubleComplex* vectors, int num_vecs, int vec_offset) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = vec_offset; i < num_vecs; ++i) {
        // Orthogonalize against all previous vectors
        if (i > 0) {
            gramSchmidt(vectors + i * N_, vectors, i);
        } else {
            // Just normalize the first vector
            double norm;
            cublasDznrm2(cublas_handle_, N_, vectors, 1, &norm);
            if (norm > 1e-14) {
                cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
                cublasZscal(cublas_handle_, N_, &scale, vectors, 1);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.ortho_time += std::chrono::duration<double>(end - start).count();
}

void GPUIterativeSolver::projectSubspaceHamiltonian(cuDoubleComplex* V, cuDoubleComplex* AV,
                                                     int subspace_dim, cuDoubleComplex* H_sub) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // H_sub = V^H * AV (subspace_dim x subspace_dim)
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    cublasZgemm(cublas_handle_,
                CUBLAS_OP_C, CUBLAS_OP_N,
                subspace_dim, subspace_dim, N_,
                &alpha,
                V, N_,
                AV, N_,
                &beta,
                H_sub, subspace_dim);
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.subspace_time += std::chrono::duration<double>(end - start).count();
}

void GPUIterativeSolver::solveSubspaceProblem(cuDoubleComplex* H_sub, int subspace_dim,
                                               double* eigs, cuDoubleComplex* evecs) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy H_sub to evecs (will be overwritten with eigenvectors)
    cudaMemcpy(evecs, H_sub, subspace_dim * subspace_dim * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToDevice);
    
    // Solve eigenvalue problem
    cusolverDnZheevd(cusolver_handle_,
                     CUSOLVER_EIG_MODE_VECTOR,
                     CUBLAS_FILL_MODE_UPPER,
                     subspace_dim,
                     evecs,
                     subspace_dim,
                     eigs,
                     d_work_,
                     lwork_,
                     d_info_);
    
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    stats_.subspace_time += std::chrono::duration<double>(end - start).count();
}

void GPUIterativeSolver::computeRitzVectors(cuDoubleComplex* V, cuDoubleComplex* evecs,
                                             int subspace_dim, int num_eigs,
                                             cuDoubleComplex* ritz_vecs) {
    // ritz_vecs = V * evecs[:, :num_eigs]
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    
    cublasZgemm(cublas_handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N_, num_eigs, subspace_dim,
                &alpha,
                V, N_,
                evecs, subspace_dim,
                &beta,
                ritz_vecs, N_);
}

double GPUIterativeSolver::computeResidualNorm(cuDoubleComplex* vec, cuDoubleComplex* Avec,
                                                double eigenvalue, cuDoubleComplex* d_temp) {
    // residual = Avec - eigenvalue * vec
    // Use provided temporary buffer to avoid repeated allocations
    cudaMemcpy(d_temp, Avec, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    cuDoubleComplex neg_lambda = make_cuDoubleComplex(-eigenvalue, 0.0);
    cublasZaxpy(cublas_handle_, N_, &neg_lambda, vec, 1, d_temp, 1);
    
    double norm;
    cublasDznrm2(cublas_handle_, N_, d_temp, 1, &norm);
    
    return norm;
}

void GPUIterativeSolver::saveEigenvector(const std::string& filename, cuDoubleComplex* vec) {
    std::vector<std::complex<double>> h_vec(N_);
    cudaMemcpy(h_vec.data(), vec, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    
    for (int i = 0; i < N_; ++i) {
        file << std::setprecision(16) << h_vec[i].real() << " " << h_vec[i].imag() << "\n";
    }
    file.close();
}

void GPUIterativeSolver::runDavidson(
    int num_eigenvalues,
    int max_iter,
    int max_subspace,
    double tol,
    std::vector<double>& eigenvalues,
    std::vector<std::vector<std::complex<double>>>& eigenvectors,
    const std::string& dir,
    bool compute_eigenvectors
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== GPU Davidson Method ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N_ << std::endl;
    std::cout << "Number of eigenvalues: " << num_eigenvalues << std::endl;
    std::cout << "Max subspace size: " << max_subspace << std::endl;
    
    // Allocate memory
    allocateMemory(max_subspace);
    
    // Initialize subspace with random vectors
    initializeRandomVectors(d_V_, num_eigenvalues);
    
    int subspace_dim = num_eigenvalues;
    eigenvalues.clear();
    eigenvectors.clear();
    
    // Allocate persistent working buffers (reused across iterations)
    cuDoubleComplex* d_subspace_evecs;
    cuDoubleComplex* d_ritz_vecs;
    cuDoubleComplex* d_ritz_Avecs;
    cuDoubleComplex* d_temp;  // For residual computation
    
    cudaMalloc(&d_subspace_evecs, max_subspace * max_subspace * sizeof(cuDoubleComplex));
    cudaMalloc(&d_ritz_vecs, N_ * num_eigenvalues * sizeof(cuDoubleComplex));
    cudaMalloc(&d_ritz_Avecs, N_ * num_eigenvalues * sizeof(cuDoubleComplex));
    cudaMalloc(&d_temp, N_ * sizeof(cuDoubleComplex));
    
    bool converged = false;
    
    for (int iter = 0; iter < max_iter && !converged; ++iter) {
        // Apply H to all subspace vectors
        auto matvec_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < subspace_dim; ++i) {
            gpu_op_->matVecGPU(d_V_ + i * N_, d_AV_ + i * N_, N_);
        }
        auto matvec_end = std::chrono::high_resolution_clock::now();
        stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
        
        // Project Hamiltonian onto subspace
        projectSubspaceHamiltonian(d_V_, d_AV_, subspace_dim, d_subspace_H_);
        
        // Solve subspace eigenvalue problem
        solveSubspaceProblem(d_subspace_H_, subspace_dim, d_subspace_eigs_, d_subspace_evecs);
        
        // Copy eigenvalues to host
        std::vector<double> current_eigs(subspace_dim);
        cudaMemcpy(current_eigs.data(), d_subspace_eigs_, subspace_dim * sizeof(double),
                   cudaMemcpyDeviceToHost);
        
        // Compute Ritz vectors and residuals for lowest eigenvalues
        computeRitzVectors(d_V_, d_subspace_evecs, subspace_dim, num_eigenvalues, d_ritz_vecs);
        computeRitzVectors(d_AV_, d_subspace_evecs, subspace_dim, num_eigenvalues, d_ritz_Avecs);
        
        // Check convergence
        converged = true;
        double max_residual = 0.0;
        std::vector<double> residual_norms(num_eigenvalues);
        for (int i = 0; i < num_eigenvalues; ++i) {
            double res_norm = computeResidualNorm(d_ritz_vecs + i * N_, 
                                                 d_ritz_Avecs + i * N_,
                                                 current_eigs[i], d_temp);
            residual_norms[i] = res_norm;
            max_residual = std::max(max_residual, res_norm);
            if (res_norm > tol) {
                converged = false;
            }
        }
        
        std::cout << "Iteration " << iter + 1 << ": ";
        std::cout << "E[0] = " << std::setprecision(10) << current_eigs[0];
        std::cout << ", max_residual = " << std::scientific << max_residual;
        std::cout << " (residuals: ";
        for (int i = 0; i < num_eigenvalues; ++i) {
            std::cout << residual_norms[i];
            if (i < num_eigenvalues - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        
        if (converged || subspace_dim >= max_subspace) {
            // Save results
            eigenvalues.assign(current_eigs.begin(), current_eigs.begin() + num_eigenvalues);
            
            // Save results to HDF5
            if (!dir.empty()) {
                if (compute_eigenvectors) {
                    // Copy eigenvectors from GPU to CPU for HDF5 save
                    std::vector<std::vector<Complex>> cpu_eigenvectors(num_eigenvalues);
                    for (int i = 0; i < num_eigenvalues; ++i) {
                        std::vector<std::complex<double>> h_vec(N_);
                        cudaMemcpy(h_vec.data(), d_ritz_vecs + i * N_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
                        cpu_eigenvectors[i] = std::move(h_vec);
                    }
                    
                    // Use unified HDF5 save function
                    HDF5IO::saveDiagonalizationResults(dir, eigenvalues, cpu_eigenvectors, "GPU_DAVIDSON");
                    std::cout << "GPU Davidson: Saved " << num_eigenvalues << " eigenvalues and " 
                              << num_eigenvalues << " eigenvectors to " << dir << "/ed_results.h5" << std::endl;
                } else {
                    // Save eigenvalues only
                    try {
                        std::string hdf5_file = HDF5IO::createOrOpenFile(dir);
                        HDF5IO::saveEigenvalues(hdf5_file, eigenvalues);
                        std::cout << "GPU Davidson: Saved " << num_eigenvalues << " eigenvalues to " << hdf5_file << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to save eigenvalues to HDF5: " << e.what() << std::endl;
                    }
                }
            }
            
            break;
        }
        
        // Expand subspace with residual vectors for unconverged eigenpairs
        // Only add residuals for vectors that haven't converged yet
        int new_vecs = 0;
        int max_new = std::min(num_eigenvalues, max_subspace - subspace_dim);
        
        for (int i = 0; i < num_eigenvalues && new_vecs < max_new; ++i) {
            double res_norm = computeResidualNorm(d_ritz_vecs + i * N_, 
                                                 d_ritz_Avecs + i * N_,
                                                 current_eigs[i], d_temp);
            
            // Only add residual if not converged
            if (res_norm > tol) {
                // Compute residual: r = A*v - lambda*v
                cuDoubleComplex* new_vec = d_V_ + (subspace_dim + new_vecs) * N_;
                cudaMemcpy(new_vec, d_ritz_Avecs + i * N_, N_ * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToDevice);
                
                cuDoubleComplex neg_lambda = make_cuDoubleComplex(-current_eigs[i], 0.0);
                cublasZaxpy(cublas_handle_, N_, &neg_lambda, d_ritz_vecs + i * N_, 1, new_vec, 1);
                
                // Note: Without access to diagonal of H, we skip preconditioning
                // and use the raw residual. This makes the method more like 
                // subspace iteration but is still valid. For faster convergence,
                // consider implementing Jacobi-Davidson with diagonal preconditioning.
                
                new_vecs++;
            }
        }
        
        // Orthogonalize new vectors against existing subspace
        if (new_vecs > 0) {
            orthogonalize(d_V_, subspace_dim + new_vecs, subspace_dim);
            
            // Check if any new vectors became zero after orthogonalization
            // (can happen if residuals are linearly dependent on existing subspace)
            int valid_vecs = 0;
            for (int i = 0; i < new_vecs; ++i) {
                double norm;
                cublasDznrm2(cublas_handle_, N_, d_V_ + (subspace_dim + i) * N_, 1, &norm);
                
                if (norm > 1e-10) {
                    // Keep this vector - copy it to the correct position if needed
                    if (valid_vecs != i) {
                        cudaMemcpy(d_V_ + (subspace_dim + valid_vecs) * N_,
                                  d_V_ + (subspace_dim + i) * N_,
                                  N_ * sizeof(cuDoubleComplex),
                                  cudaMemcpyDeviceToDevice);
                    }
                    valid_vecs++;
                }
                // Silently discard near-zero vectors (linearly dependent residuals)
            }
            
            if (valid_vecs > 0) {
                subspace_dim += valid_vecs;
            } else {
                // All new vectors were zero after orthogonalization
                // Check if this is true convergence or numerical breakdown
                if (max_residual < tol) {
                    std::cout << "Davidson converged: all residuals below tolerance." << std::endl;
                } else {
                    std::cout << "Warning: All new vectors became zero after orthogonalization!" << std::endl;
                    std::cout << "  This may indicate numerical issues. Current max residual: " 
                              << max_residual << " (tolerance: " << tol << ")" << std::endl;
                    std::cout << "  Consider: 1) increasing max_subspace, 2) using a different solver, "
                              << "or 3) checking for near-degeneracy in your system." << std::endl;
                }
                converged = true;
            }
        } else {
            // All eigenpairs converged but we didn't catch it above
            // This shouldn't happen but handle it gracefully
            std::cout << "Warning: No new vectors to add, all converged!" << std::endl;
            converged = true;
        }
        
        stats_.iterations++;
    }
    
    // Free persistent working buffers
    cudaFree(d_subspace_evecs);
    cudaFree(d_ritz_vecs);
    cudaFree(d_ritz_Avecs);
    cudaFree(d_temp);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Correct throughput calculation:
    // Each iteration does subspace_dim matvecs, each matvec has N rows with ~32 non-zeros
    // Each complex multiply-add = 8 FLOPs
    constexpr int NNZ_PER_ROW = 32;
    constexpr int FLOPS_PER_NNZ = 8;
    double flops_per_matvec = static_cast<double>(N_) * NNZ_PER_ROW * FLOPS_PER_NNZ;
    stats_.throughput = (stats_.iterations * subspace_dim * flops_per_matvec) / stats_.matvec_time / 1e9;
    
    std::cout << "\n=== GPU Davidson Statistics ===" << std::endl;
    std::cout << "Total time: " << stats_.total_time << " s" << std::endl;
    std::cout << "MatVec time: " << stats_.matvec_time << " s" << std::endl;
    std::cout << "Ortho time: " << stats_.ortho_time << " s" << std::endl;
    std::cout << "Subspace time: " << stats_.subspace_time << " s" << std::endl;
    std::cout << "Throughput: " << stats_.throughput << " GFLOPS" << std::endl;
}

void GPUIterativeSolver::runLOBPCG(
    int num_eigenvalues,
    int max_iter,
    double tol,
    std::vector<double>& eigenvalues,
    const std::string& dir,
    bool compute_eigenvectors
){

    
}