#include "gpu_cg.cuh"
#include "gpu_operator.cuh"
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
    cuDoubleComplex alpha = make_cuDoubleComplex(-1.0, 0.0);
    cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    
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
            
            if (compute_eigenvectors) {
                for (int i = 0; i < num_eigenvalues; ++i) {
                    if (!dir.empty()) {
                        std::string filename = dir + "/eigenvector_" + std::to_string(i) + ".dat";
                        saveEigenvector(filename, d_ritz_vecs + i * N_);
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
                
                // Simple diagonal preconditioning: r / (lambda - diag(H))
                // For now, skip preconditioning or use simple scaling
                double scale = 1.0 / std::max(std::abs(current_eigs[i]), 0.1);
                cuDoubleComplex scale_c = make_cuDoubleComplex(scale, 0.0);
                cublasZscal(cublas_handle_, N_, &scale_c, new_vec, 1);
                
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
                // All new vectors were zero - likely converged
                std::cout << "Warning: All new vectors became zero after orthogonalization!" << std::endl;
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
    stats_.throughput = (stats_.iterations * subspace_dim * 2.0 * N_) / stats_.matvec_time / 1e9;
    
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
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n=== GPU LOBPCG Method ===" << std::endl;
    std::cout << "Hilbert space dimension: " << N_ << std::endl;
    std::cout << "Number of eigenvalues: " << num_eigenvalues << std::endl;
    
    int block_size = num_eigenvalues;
    int nsub = 3 * block_size;  // Subspace dimension: W + X + P
    
    // Allocate memory for max subspace
    allocateMemory(nsub);
    
    // Allocate GPU memory for W, X, P blocks and their H applications
    cuDoubleComplex* d_W;   // Residuals
    cuDoubleComplex* d_X;   // Current eigenvectors
    cuDoubleComplex* d_P;   // Search directions
    cuDoubleComplex* d_HW;  // H*W
    cuDoubleComplex* d_HX;  // H*X
    cuDoubleComplex* d_HP;  // H*P
    
    cudaMalloc(&d_W, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_X, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_P, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_HW, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_HX, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_HP, N_ * block_size * sizeof(cuDoubleComplex));
    
    // Allocate subspace matrices
    cuDoubleComplex* d_hsub;    // Subspace Hamiltonian
    cuDoubleComplex* d_ovlp;    // Subspace overlap matrix
    double* d_eigsub;            // Subspace eigenvalues
    cuDoubleComplex* d_eigvec_sub;  // Subspace eigenvectors
    
    cudaMalloc(&d_hsub, nsub * nsub * sizeof(cuDoubleComplex));
    cudaMalloc(&d_ovlp, nsub * nsub * sizeof(cuDoubleComplex));
    cudaMalloc(&d_eigsub, nsub * sizeof(double));
    cudaMalloc(&d_eigvec_sub, nsub * nsub * sizeof(cuDoubleComplex));
    
    // Initialize X with random orthonormal vectors
    initializeRandomVectors(d_X, block_size);
    
    // Initialize P to zero
    cudaMemset(d_P, 0, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMemset(d_HP, 0, N_ * block_size * sizeof(cuDoubleComplex));
    
    // Compute H*X
    auto matvec_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < block_size; ++i) {
        gpu_op_->matVecGPU(d_X + i * N_, d_HX + i * N_, N_);
    }
    auto matvec_end = std::chrono::high_resolution_clock::now();
    stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
    
    // Calculate initial Rayleigh quotients
    eigenvalues.resize(block_size);
    for (int i = 0; i < block_size; ++i) {
        cuDoubleComplex rayleigh;
        cublasZdotc(cublas_handle_, N_, d_X + i * N_, 1, d_HX + i * N_, 1, &rayleigh);
        eigenvalues[i] = cuCreal(rayleigh);
    }
    
    std::cout << "Initial eigenvalues: ";
    for (int i = 0; i < block_size; ++i) {
        std::cout << eigenvalues[i] << " ";
    }
    std::cout << std::endl;
    
    bool converged = false;
    
    // Temporary buffers
    cuDoubleComplex* d_new_X;
    cuDoubleComplex* d_new_P;
    cudaMalloc(&d_new_X, N_ * block_size * sizeof(cuDoubleComplex));
    cudaMalloc(&d_new_P, N_ * block_size * sizeof(cuDoubleComplex));
    
    for (int stp = 1; stp <= max_iter && !converged; ++stp) {
        // Compute residuals: W = H*X - λ*X
        double max_res_norm = 0.0;
        std::vector<double> residual_norms(block_size);
        
        for (int i = 0; i < block_size; ++i) {
            // W[i] = HX[i] - eigenvalues[i] * X[i]
            cudaMemcpy(d_W + i * N_, d_HX + i * N_, N_ * sizeof(cuDoubleComplex),
                      cudaMemcpyDeviceToDevice);
            
            cuDoubleComplex neg_lambda = make_cuDoubleComplex(-eigenvalues[i], 0.0);
            cublasZaxpy(cublas_handle_, N_, &neg_lambda, d_X + i * N_, 1, d_W + i * N_, 1);
            
            // Compute residual norm
            double res_norm;
            cublasDznrm2(cublas_handle_, N_, d_W + i * N_, 1, &res_norm);
            residual_norms[i] = res_norm;
            max_res_norm = std::max(max_res_norm, res_norm);
        }
        
        // Output progress
        if (stp % 10 == 0 || max_res_norm < tol) {
            std::cout << "Iteration " << stp << ": ";
            std::cout << "E[0] = " << std::setprecision(10) << eigenvalues[0];
            std::cout << ", max_residual = " << std::scientific << max_res_norm;
            std::cout << " (residuals: ";
            for (int i = 0; i < block_size; ++i) {
                std::cout << residual_norms[i];
                if (i < block_size - 1) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
        }
        
        // Check convergence
        if (max_res_norm < tol) {
            std::cout << "LOBPCG converged after " << stp << " iterations." << std::endl;
            converged = true;
            break;
        }
        
        // No preconditioning for now - can add proper preconditioner later
        
        // Apply H to W
        matvec_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < block_size; ++i) {
            gpu_op_->matVecGPU(d_W + i * N_, d_HW + i * N_, N_);
        }
        matvec_end = std::chrono::high_resolution_clock::now();
        stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
        
        // Build subspace matrices: H_sub and O_sub
        // Subspace organized as [W0, W1, ..., X0, X1, ..., P0, P1, ...]
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        
        // Zero out matrices
        cudaMemset(d_hsub, 0, nsub * nsub * sizeof(cuDoubleComplex));
        cudaMemset(d_ovlp, 0, nsub * nsub * sizeof(cuDoubleComplex));
        
        // Compute blocks of H_sub and O_sub
        // Block structure: [W|X|P] x [W|X|P]
        cuDoubleComplex* basis_ptrs[3] = {d_W, d_X, d_P};
        cuDoubleComplex* H_basis_ptrs[3] = {d_HW, d_HX, d_HP};
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // H_sub block (i,j): basis[i]^H * H*basis[j]
                cublasZgemm(cublas_handle_,
                           CUBLAS_OP_C, CUBLAS_OP_N,
                           block_size, block_size, N_,
                           &alpha,
                           basis_ptrs[i], N_,
                           H_basis_ptrs[j], N_,
                           &beta,
                           d_hsub + (i * block_size * nsub + j * block_size),
                           nsub);
                
                // O_sub block (i,j): basis[i]^H * basis[j]
                cublasZgemm(cublas_handle_,
                           CUBLAS_OP_C, CUBLAS_OP_N,
                           block_size, block_size, N_,
                           &alpha,
                           basis_ptrs[i], N_,
                           basis_ptrs[j], N_,
                           &beta,
                           d_ovlp + (i * block_size * nsub + j * block_size),
                           nsub);
            }
        }
        
        // Copy to host for generalized eigenvalue problem
        std::vector<std::complex<double>> h_hsub(nsub * nsub);
        std::vector<std::complex<double>> h_ovlp(nsub * nsub);
        cudaMemcpy(h_hsub.data(), d_hsub, nsub * nsub * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ovlp.data(), d_ovlp, nsub * nsub * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToHost);
        
        // Solve generalized eigenvalue problem using Eigen (Lowdin orthogonalization)
        auto subspace_start = std::chrono::high_resolution_clock::now();
        
        Eigen::MatrixXcd eigen_hsub(nsub, nsub);
        Eigen::MatrixXcd eigen_ovlp(nsub, nsub);
        
        for (int i = 0; i < nsub; ++i) {
            for (int j = 0; j < nsub; ++j) {
                eigen_hsub(i, j) = h_hsub[j + i * nsub];
                eigen_ovlp(i, j) = h_ovlp[j + i * nsub];
            }
        }
        
        // Diagonalize overlap matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> o_solver(eigen_ovlp);
        Eigen::VectorXd o_eigenvals = o_solver.eigenvalues();
        Eigen::MatrixXcd o_eigenvecs = o_solver.eigenvectors();
        
        // Compute O^(-1/2) using only non-negligible eigenvalues
        // Ensure we keep at least block_size vectors for stability
        double threshold = 1.0e-10;
        int nsub_cut = 0;
        for (int i = 0; i < nsub; ++i) {
            if (o_eigenvals(i) > threshold) {
                nsub_cut++;
            }
        }
        
        // If we filtered too aggressively, keep more vectors
        if (nsub_cut < block_size) {
            nsub_cut = std::min(block_size, nsub);
            // Find the block_size-th largest eigenvalue and use that as threshold
            std::vector<double> sorted_eigs(o_eigenvals.data(), o_eigenvals.data() + nsub);
            std::sort(sorted_eigs.begin(), sorted_eigs.end(), std::greater<double>());
            if (nsub_cut < (int)sorted_eigs.size()) {
                threshold = sorted_eigs[nsub_cut - 1] * 0.1;  // Keep slightly more
            } else {
                threshold = 0.0;
            }
            nsub_cut = 0;
            for (int i = 0; i < nsub; ++i) {
                if (o_eigenvals(i) > threshold) {
                    nsub_cut++;
                }
            }
        }
        
        Eigen::MatrixXcd o_sqrt_inv = Eigen::MatrixXcd::Zero(nsub, nsub_cut);
        int idx = 0;
        for (int i = 0; i < nsub; ++i) {
            if (o_eigenvals(i) > threshold) {
                o_sqrt_inv.col(idx) = o_eigenvecs.col(i) / std::sqrt(o_eigenvals(i));
                idx++;
            }
        }
        
        // Transform Hamiltonian: H' = O^(-1/2)† * H * O^(-1/2)
        Eigen::MatrixXcd h_prime = o_sqrt_inv.adjoint() * eigen_hsub * o_sqrt_inv;
        
        // Solve standard eigenvalue problem
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> h_solver(h_prime);
        Eigen::VectorXd h_eigenvals = h_solver.eigenvalues();
        Eigen::MatrixXcd h_eigenvecs = h_solver.eigenvectors();
        
        // Transform eigenvectors back: v = O^(-1/2) * v'
        Eigen::MatrixXcd transformed_evecs = o_sqrt_inv * h_eigenvecs;
        
        auto subspace_end = std::chrono::high_resolution_clock::now();
        stats_.subspace_time += std::chrono::duration<double>(subspace_end - subspace_start).count();
        
        // Update eigenvalues
        for (int i = 0; i < block_size; ++i) {
            eigenvalues[i] = h_eigenvals(i);
        }
        
        // Copy eigenvector coefficients back to GPU
        std::vector<std::complex<double>> h_evecs(nsub * block_size);
        for (int i = 0; i < nsub; ++i) {
            for (int j = 0; j < block_size; ++j) {
                h_evecs[j * nsub + i] = transformed_evecs(i, j);
            }
        }
        cudaMemcpy(d_eigvec_sub, h_evecs.data(), nsub * block_size * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
        
        // Update X: X_new = W*c_W + X*c_X + P*c_P
        cudaMemset(d_new_X, 0, N_ * block_size * sizeof(cuDoubleComplex));
        
        for (int block_idx = 0; block_idx < 3; ++block_idx) {
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta = (block_idx == 0) ? make_cuDoubleComplex(0.0, 0.0) : make_cuDoubleComplex(1.0, 0.0);
            
            cublasZgemm(cublas_handle_,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N_, block_size, block_size,
                       &alpha,
                       basis_ptrs[block_idx], N_,
                       d_eigvec_sub + block_idx * block_size, nsub,
                       &beta,
                       d_new_X, N_);
        }
        
        // Update P: P_new = W*c_W + P*c_P (no X component for conjugate direction)
        cudaMemset(d_new_P, 0, N_ * block_size * sizeof(cuDoubleComplex));
        
        int p_block = 0;
        for (int block_idx = 0; block_idx < 3; block_idx += 2) {  // Only W (0) and P (2)
            cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
            cuDoubleComplex beta = (p_block == 0) ? make_cuDoubleComplex(0.0, 0.0) : make_cuDoubleComplex(1.0, 0.0);
            
            cublasZgemm(cublas_handle_,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N_, block_size, block_size,
                       &alpha,
                       basis_ptrs[block_idx], N_,
                       d_eigvec_sub + block_idx * block_size, nsub,
                       &beta,
                       d_new_P, N_);
            p_block++;
        }
        
        // Copy new X and P
        cudaMemcpy(d_X, d_new_X, N_ * block_size * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P, d_new_P, N_ * block_size * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToDevice);
        
        // Apply H to the updated X and P vectors FIRST
        matvec_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < block_size; ++i) {
            gpu_op_->matVecGPU(d_X + i * N_, d_HX + i * N_, N_);
            gpu_op_->matVecGPU(d_P + i * N_, d_HP + i * N_, N_);
        }
        matvec_end = std::chrono::high_resolution_clock::now();
        stats_.matvec_time += std::chrono::duration<double>(matvec_end - matvec_start).count();
        
        // Normalize X and P (and their H-applications) - matching CPU approach
        auto ortho_start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < block_size; ++i) {
            // Normalize X and H*X together
            double norm;
            cublasDznrm2(cublas_handle_, N_, d_X + i * N_, 1, &norm);
            if (norm > 1e-10) {
                cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
                cublasZscal(cublas_handle_, N_, &scale, d_X + i * N_, 1);
                cublasZscal(cublas_handle_, N_, &scale, d_HX + i * N_, 1);
            }
            
            // Normalize P and H*P together
            cublasDznrm2(cublas_handle_, N_, d_P + i * N_, 1, &norm);
            if (norm > 1e-10) {
                cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
                cublasZscal(cublas_handle_, N_, &scale, d_P + i * N_, 1);
                cublasZscal(cublas_handle_, N_, &scale, d_HP + i * N_, 1);
            }
        }
        
        auto ortho_end = std::chrono::high_resolution_clock::now();
        stats_.ortho_time += std::chrono::duration<double>(ortho_end - ortho_start).count();
        
        stats_.iterations++;
    }
    
    // Save eigenvectors if requested
    if (compute_eigenvectors && !dir.empty()) {
        for (int i = 0; i < block_size; ++i) {
            std::string filename = dir + "/eigenvector_" + std::to_string(i) + ".dat";
            saveEigenvector(filename, d_X + i * N_);
        }
    }
    
    // Cleanup
    cudaFree(d_W);
    cudaFree(d_X);
    cudaFree(d_P);
    cudaFree(d_HW);
    cudaFree(d_HX);
    cudaFree(d_HP);
    cudaFree(d_hsub);
    cudaFree(d_ovlp);
    cudaFree(d_eigsub);
    cudaFree(d_eigvec_sub);
    cudaFree(d_new_X);
    cudaFree(d_new_P);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(total_end - total_start).count();
    stats_.throughput = (stats_.iterations * block_size * 3.0 * N_) / stats_.matvec_time / 1e9;
    
    std::cout << "\n=== GPU LOBPCG Statistics ===" << std::endl;
    std::cout << "Total time: " << stats_.total_time << " s" << std::endl;
    std::cout << "MatVec time: " << stats_.matvec_time << " s" << std::endl;
    std::cout << "Ortho time: " << stats_.ortho_time << " s" << std::endl;
    std::cout << "Subspace time: " << stats_.subspace_time << " s" << std::endl;
    std::cout << "Throughput: " << stats_.throughput << " GFLOPS" << std::endl;
}
