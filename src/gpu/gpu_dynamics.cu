#include "gpu_dynamics.cuh"
#include "gpu_operator.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>

// LAPACK routine for tridiagonal eigenvalue decomposition
extern "C" {
    void dstev_(const char* jobz, const int* n, double* d, double* e, 
                double* z, const int* ldz, double* work, int* info, size_t jobz_len);
}

using Complex = std::complex<double>;

GPUDynamicsSolver::GPUDynamicsSolver(GPUOperator* gpu_op, int N)
    : gpu_op_(gpu_op), N_(N), d_state_(nullptr), d_evolved_state_(nullptr),
      d_O1_state_(nullptr), d_evolved_O1_state_(nullptr), d_O2_temp_(nullptr),
      d_temp_(nullptr), d_krylov_basis_(nullptr), d_krylov_temp_(nullptr),
      d_krylov_H_(nullptr) {
    
    // Create cuBLAS handle
    cublasCreate(&cublas_handle_);
    
    // Allocate basic GPU memory (Krylov allocated on demand)
    cudaMalloc(&d_state_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_evolved_state_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_O1_state_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_evolved_O1_state_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_O2_temp_, N_ * sizeof(cuDoubleComplex));
    cudaMalloc(&d_temp_, N_ * sizeof(cuDoubleComplex));
    
    stats_ = {0.0, 0.0, 0.0, 0};
}

GPUDynamicsSolver::~GPUDynamicsSolver() {
    freeMemory();
    cublasDestroy(cublas_handle_);
}

void GPUDynamicsSolver::allocateMemory(int max_krylov_dim) {
    if (d_krylov_basis_ == nullptr) {
        cudaMalloc(&d_krylov_basis_, max_krylov_dim * N_ * sizeof(cuDoubleComplex));
        cudaMalloc(&d_krylov_temp_, N_ * sizeof(cuDoubleComplex));
        cudaMalloc(&d_krylov_H_, max_krylov_dim * max_krylov_dim * sizeof(double));
    }
}

void GPUDynamicsSolver::freeMemory() {
    if (d_state_) cudaFree(d_state_);
    if (d_evolved_state_) cudaFree(d_evolved_state_);
    if (d_O1_state_) cudaFree(d_O1_state_);
    if (d_evolved_O1_state_) cudaFree(d_evolved_O1_state_);
    if (d_O2_temp_) cudaFree(d_O2_temp_);
    if (d_temp_) cudaFree(d_temp_);
    if (d_krylov_basis_) cudaFree(d_krylov_basis_);
    if (d_krylov_temp_) cudaFree(d_krylov_temp_);
    if (d_krylov_H_) cudaFree(d_krylov_H_);
}

void GPUDynamicsSolver::copyStateToGPU(const std::vector<Complex>& state) {
    std::vector<cuDoubleComplex> cu_state(N_);
    for (int i = 0; i < N_; i++) {
        cu_state[i] = make_cuDoubleComplex(state[i].real(), state[i].imag());
    }
    cudaMemcpy(d_state_, cu_state.data(), N_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
}

void GPUDynamicsSolver::copyStateFromGPU(const cuDoubleComplex* d_state, std::vector<Complex>& h_state) {
    std::vector<cuDoubleComplex> cu_state(N_);
    cudaMemcpy(cu_state.data(), d_state, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    h_state.resize(N_);
    for (int i = 0; i < N_; i++) {
        h_state[i] = Complex(cuCreal(cu_state[i]), cuCimag(cu_state[i]));
    }
}

void GPUDynamicsSolver::copyStateToGPU(const std::vector<Complex>& h_state, cuDoubleComplex* d_state) {
    std::vector<cuDoubleComplex> cu_state(N_);
    for (int i = 0; i < N_; i++) {
        cu_state[i] = make_cuDoubleComplex(h_state[i].real(), h_state[i].imag());
    }
    cudaMemcpy(d_state, cu_state.data(), N_ * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
}

Complex GPUDynamicsSolver::dotProductGPU(const cuDoubleComplex* v1, const cuDoubleComplex* v2) {
    cuDoubleComplex result;
    cublasZdotc(cublas_handle_, N_, v1, 1, v2, 1, &result);
    return Complex(cuCreal(result), cuCimag(result));
}

void GPUDynamicsSolver::krylovTimeStep(cuDoubleComplex* state, double dt, int krylov_dim) {
    // Optimized Krylov exponential propagator
    // Uses the property that |ψ(t)> = exp(-iHt)|ψ(0)> ≈ β||ψ|| V_m exp(-it H_m) e_1
    // where V_m is Krylov basis, H_m is tridiagonal projection, e_1 = [1,0,...,0]
    
    allocateMemory(krylov_dim);
    
    // Normalize initial state and store norm
    double norm;
    cublasDznrm2(cublas_handle_, N_, state, 1, &norm);
    cuDoubleComplex scale = make_cuDoubleComplex(1.0 / norm, 0.0);
    cublasZscal(cublas_handle_, N_, &scale, state, 1);
    
    // Copy to first Krylov vector
    cudaMemcpy(d_krylov_basis_, state, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
    
    // Build tridiagonal matrix using Lanczos
    std::vector<double> alpha(krylov_dim, 0.0);
    std::vector<double> beta(krylov_dim - 1, 0.0);
    
    for (int j = 0; j < krylov_dim - 1; j++) {
        cuDoubleComplex* v_j = d_krylov_basis_ + j * N_;
        cuDoubleComplex* v_jp1 = d_krylov_basis_ + (j + 1) * N_;
        
        // Apply Hamiltonian: w = H|v_j>
        gpu_op_->matVecGPU(v_j, d_krylov_temp_, N_);
        
        // Compute alpha_j = <v_j|H|v_j>
        Complex alpha_complex = dotProductGPU(v_j, d_krylov_temp_);
        alpha[j] = alpha_complex.real();
        
        // w = w - alpha_j * v_j
        cuDoubleComplex minus_alpha = make_cuDoubleComplex(-alpha[j], 0.0);
        cublasZaxpy(cublas_handle_, N_, &minus_alpha, v_j, 1, d_krylov_temp_, 1);
        
        // w = w - beta_{j-1} * v_{j-1} (if j > 0)
        if (j > 0) {
            cuDoubleComplex* v_jm1 = d_krylov_basis_ + (j - 1) * N_;
            cuDoubleComplex minus_beta = make_cuDoubleComplex(-beta[j - 1], 0.0);
            cublasZaxpy(cublas_handle_, N_, &minus_beta, v_jm1, 1, d_krylov_temp_, 1);
        }
        
        // beta_j = ||w||
        cublasDznrm2(cublas_handle_, N_, d_krylov_temp_, 1, &beta[j]);
        
        // v_{j+1} = w / beta_j
        if (beta[j] > 1e-12) {
            scale = make_cuDoubleComplex(1.0 / beta[j], 0.0);
            cudaMemcpy(v_jp1, d_krylov_temp_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
            cublasZscal(cublas_handle_, N_, &scale, v_jp1, 1);
        } else {
            // Krylov space has converged early
            krylov_dim = j + 1;
            break;
        }
    }
    
    // Compute last diagonal element
    if (krylov_dim > 0) {
        cuDoubleComplex* v_last = d_krylov_basis_ + (krylov_dim - 1) * N_;
        gpu_op_->matVecGPU(v_last, d_krylov_temp_, N_);
        Complex alpha_complex = dotProductGPU(v_last, d_krylov_temp_);
        alpha[krylov_dim - 1] = alpha_complex.real();
    }
    
    // Diagonalize tridiagonal matrix T = U * Lambda * U^T
    std::vector<double> eigenvalues = alpha;
    std::vector<double> eigenvectors(krylov_dim * krylov_dim, 0.0);
    std::vector<double> offdiag = beta;
    
    char jobz = 'V';
    int n = krylov_dim;
    int ldz = krylov_dim;
    std::vector<double> work(std::max(1, 2 * krylov_dim - 2));
    int info = 0;
    
    // Call LAPACK to diagonalize tridiagonal matrix (AOCL version requires string length)
    dstev_(&jobz, &n, eigenvalues.data(), offdiag.data(), 
           eigenvectors.data(), &ldz, work.data(), &info, 1);
    
    if (info != 0) {
        std::cerr << "Warning: LAPACK dstev failed with info=" << info << std::endl;
        return;
    }
    
    // Compute exp(-i*dt*T) * e_1 in eigenbasis
    // Since initial state was normalized, we propagate e_1 = [1, 0, ..., 0]^T
    // Result: c_j = sum_i U[0,i] * exp(-i*dt*lambda_i) * U[j,i]
    std::vector<Complex> evolved_coeffs(krylov_dim, 0.0);
    
    for (int j = 0; j < krylov_dim; j++) {
        Complex sum = 0.0;
        for (int i = 0; i < krylov_dim; i++) {
            // U is stored column-major from LAPACK: U[row,col] = eigenvectors[col*ldz + row]
            double u_0i = eigenvectors[i * ldz + 0];  // U[0,i]
            double u_ji = eigenvectors[i * ldz + j];  // U[j,i]
            Complex phase = std::exp(Complex(0.0, -dt * eigenvalues[i]));
            sum += u_0i * phase * u_ji;
        }
        evolved_coeffs[j] = sum * norm;  // Restore original norm
    }
    
    // Reconstruct state: |ψ(t)> = sum_j c_j * |v_j>
    cudaMemset(state, 0, N_ * sizeof(cuDoubleComplex));
    
    for (int j = 0; j < krylov_dim; j++) {
        cuDoubleComplex* v_j = d_krylov_basis_ + j * N_;
        cuDoubleComplex cu_coeff = make_cuDoubleComplex(
            evolved_coeffs[j].real(),
            evolved_coeffs[j].imag()
        );
        cublasZaxpy(cublas_handle_, N_, &cu_coeff, v_j, 1, state, 1);
    }
}

void GPUDynamicsSolver::computeKrylovCorrelations(
    const std::vector<Complex>& initial_state,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const std::vector<std::string>& operator_names,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end,
    double dt,
    int krylov_dim
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "GPU Krylov: Computing " << operators_1.size() << " correlations, "
              << "Krylov dim=" << krylov_dim << ", dt=" << dt << ", t_end=" << t_end << std::endl;
    
    int num_steps = static_cast<int>(t_end / dt) + 1;
    stats_.num_steps = num_steps;
    
    // Copy initial state to GPU
    copyStateToGPU(initial_state);
    
    // Process each operator pair
    for (size_t op_idx = 0; op_idx < operators_1.size(); op_idx++) {
        std::string op_name = operator_names[op_idx];
        std::cout << "  GPU computing " << op_name << " correlation..." << std::endl;
        
        // Apply O_1 to initial state on CPU
        auto op_start = std::chrono::high_resolution_clock::now();
        std::vector<Complex> h_state(N_);
        std::vector<Complex> h_O1_state(N_);
        copyStateFromGPU(d_state_, h_state);
        operators_1[op_idx](h_state.data(), h_O1_state.data(), N_);
        copyStateToGPU(h_O1_state, d_O1_state_);
        auto op_end = std::chrono::high_resolution_clock::now();
        stats_.operator_time += std::chrono::duration<double>(op_end - op_start).count();
        
        // Compute initial correlation C(0) = <ψ|O_2†O_1|ψ>
        std::vector<Complex> h_O2_state(N_);
        operators_2[op_idx](h_state.data(), h_O2_state.data(), N_);
        copyStateToGPU(h_O2_state, d_O2_temp_);
        Complex initial_corr = dotProductGPU(d_O2_temp_, d_O1_state_);
        
        std::cout << "    Initial correlation C(0) = " << initial_corr << std::endl;
        
        // Initialize evolved states
        cudaMemcpy(d_evolved_state_, d_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_evolved_O1_state_, d_O1_state_, N_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice);
        
        // Open output file
        std::string output_file = dir + "/time_corr_rand" + std::to_string(sample) + "_" 
                                + op_name + "_beta=" + std::to_string(inv_temp) + ".dat";
        
        std::ofstream out(output_file);
        if (!out.is_open()) {
            std::cerr << "GPU Error: Could not open " << output_file << std::endl;
            continue;
        }
        
        out << "# t time_correlation_real time_correlation_imag" << std::endl;
        out << std::setprecision(16);
        out << 0.0 << " " << initial_corr.real() << " " << initial_corr.imag() << std::endl;
        
        // Time evolution loop
        auto evol_start = std::chrono::high_resolution_clock::now();
        for (int step = 1; step < num_steps; step++) {
            // Evolve both states by dt on GPU
            krylovTimeStep(d_evolved_state_, dt, krylov_dim);
            krylovTimeStep(d_evolved_O1_state_, dt, krylov_dim);
            
            // Apply O_2 to evolved state on CPU
            std::vector<Complex> h_evolved_state(N_);
            copyStateFromGPU(d_evolved_state_, h_evolved_state);
            operators_2[op_idx](h_evolved_state.data(), h_O2_state.data(), N_);
            copyStateToGPU(h_O2_state, d_O2_temp_);
            
            // Compute correlation on GPU
            Complex corr_t = dotProductGPU(d_O2_temp_, d_evolved_O1_state_);
            
            double t = step * dt;
            out << t << " " << corr_t.real() << " " << corr_t.imag() << std::endl;
            
            if (step % 100 == 0) {
                out.flush();
            }
        }
        auto evol_end = std::chrono::high_resolution_clock::now();
        stats_.evolution_time += std::chrono::duration<double>(evol_end - evol_start).count();
        
        out.close();
        std::cout << "    Saved to " << output_file << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    std::cout << "GPU Krylov complete: " << stats_.total_time << " seconds" << std::endl;
}

void GPUDynamicsSolver::computeTaylorCorrelations(
    std::function<void(const Complex*, Complex*, int)> U_t_cpu,
    const std::vector<Complex>& initial_state,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const std::vector<std::string>& operator_names,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end,
    double dt
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "GPU Taylor: Computing " << operators_1.size() << " correlations, "
              << "dt=" << dt << ", t_end=" << t_end << std::endl;
    
    int num_steps = static_cast<int>(t_end / dt) + 1;
    stats_.num_steps = num_steps;
    
    // Copy initial state to GPU
    copyStateToGPU(initial_state);
    
    // Create CPU buffers for time evolution (U_t is CPU function)
    std::vector<Complex> h_evolved_state(N_);
    std::vector<Complex> h_evolved_O1_state(N_);
    std::vector<Complex> h_O2_state(N_);
    
    // Process each operator pair
    for (size_t op_idx = 0; op_idx < operators_1.size(); op_idx++) {
        std::string op_name = operator_names[op_idx];
        std::cout << "  GPU computing " << op_name << " correlation..." << std::endl;
        
        // Apply O_1 to initial state on CPU
        auto op_start = std::chrono::high_resolution_clock::now();
        std::vector<Complex> h_state(N_);
        std::vector<Complex> h_O1_state(N_);
        copyStateFromGPU(d_state_, h_state);
        operators_1[op_idx](h_state.data(), h_O1_state.data(), N_);
        copyStateToGPU(h_O1_state, d_O1_state_);
        auto op_end = std::chrono::high_resolution_clock::now();
        stats_.operator_time += std::chrono::duration<double>(op_end - op_start).count();
        
        // Compute initial correlation
        operators_2[op_idx](h_state.data(), h_O2_state.data(), N_);
        copyStateToGPU(h_O2_state, d_O2_temp_);
        Complex initial_corr = dotProductGPU(d_O2_temp_, d_O1_state_);
        
        std::cout << "    Initial correlation C(0) = " << initial_corr << std::endl;
        
        // Copy to host for time evolution
        h_evolved_state = h_state;
        h_evolved_O1_state = h_O1_state;
        
        // Open output file
        std::string output_file = dir + "/time_corr_rand" + std::to_string(sample) + "_" 
                                + op_name + "_beta=" + std::to_string(inv_temp) + ".dat";
        
        std::ofstream out(output_file);
        if (!out.is_open()) {
            std::cerr << "GPU Error: Could not open " << output_file << std::endl;
            continue;
        }
        
        out << "# t time_correlation_real time_correlation_imag" << std::endl;
        out << std::setprecision(16);
        out << 0.0 << " " << initial_corr.real() << " " << initial_corr.imag() << std::endl;
        
        // Time evolution loop (on CPU using provided U_t)
        auto evol_start = std::chrono::high_resolution_clock::now();
        for (int step = 1; step < num_steps; step++) {
            // Evolve states using CPU operator
            std::vector<Complex> temp_state(N_);
            U_t_cpu(h_evolved_state.data(), temp_state.data(), N_);
            h_evolved_state = temp_state;
            
            U_t_cpu(h_evolved_O1_state.data(), temp_state.data(), N_);
            h_evolved_O1_state = temp_state;
            
            // Apply O_2 to evolved state
            operators_2[op_idx](h_evolved_state.data(), h_O2_state.data(), N_);
            
            // Copy to GPU and compute correlation
            copyStateToGPU(h_O2_state, d_O2_temp_);
            copyStateToGPU(h_evolved_O1_state, d_evolved_O1_state_);
            Complex corr_t = dotProductGPU(d_O2_temp_, d_evolved_O1_state_);
            
            double t = step * dt;
            out << t << " " << corr_t.real() << " " << corr_t.imag() << std::endl;
            
            if (step % 100 == 0) {
                out.flush();
            }
        }
        auto evol_end = std::chrono::high_resolution_clock::now();
        stats_.evolution_time += std::chrono::duration<double>(evol_end - evol_start).count();
        
        out.close();
        std::cout << "    Saved to " << output_file << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    std::cout << "GPU Taylor complete: " << stats_.total_time << " seconds" << std::endl;
}
