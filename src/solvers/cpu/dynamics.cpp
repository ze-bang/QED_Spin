// dynamics.cpp - Implementation of general quantum dynamics computation module

#include <ed/solvers/dynamics.h>
#include <ed/core/hdf5_io.h>  // For HDF5 output
#include <ed/core/blas_lapack_wrapper.h>  // For LAPACKE interface
#include <ed/parallel/thread_budget.h>  // Phase 6.1: dim-aware OMP+BLAS cap

// ============================================================================
// TIME EVOLUTION METHODS
// ============================================================================

void time_evolve_krylov(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    uint64_t krylov_dim,
    bool normalize
) {
    if (N <= 0) {
        return;
    }

    // Ensure Krylov dimension doesn't exceed system size
    krylov_dim = std::max(static_cast<uint64_t>(1), std::min(krylov_dim, N));
    
    // Allocate buffers locally (more thread-safe and less memory waste than thread_local)
    std::vector<ComplexVector> krylov_vectors(krylov_dim);
    for (auto& vec : krylov_vectors) {
        vec.resize(N);
    }
    std::vector<double> alpha(krylov_dim);
    std::vector<double> beta(krylov_dim - 1);
    ComplexVector w(N);
    
    // Initialize first Krylov vector as normalized input state
    double norm = cblas_dznrm2(N, state.data(), 1);
    if (norm < 1e-14) {
        return;
    }
    
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zcopy(N, state.data(), 1, krylov_vectors[0].data(), 1);
    cblas_zscal(N, &scale_factor, krylov_vectors[0].data(), 1);
    
    // Lanczos iteration - three-term recurrence (no full reorthogonalization needed in theory)
    // For better stability, we do selective reorthogonalization only when needed
    uint64_t effective_dim = krylov_dim;
    constexpr double breakdown_threshold = 1e-14;
    constexpr double reortho_threshold = 0.7;  // Reorthogonalize if ||w|| drops below this fraction
    
    for (int j = 0; j < krylov_dim - 1; j++) {
        // Apply Hamiltonian: w = H * v_j
        H(krylov_vectors[j].data(), w.data(), N);
        
        // Compute alpha_j = <v_j | w>
        Complex alpha_complex;
        cblas_zdotc_sub(N, krylov_vectors[j].data(), 1, w.data(), 1, &alpha_complex);
        alpha[j] = alpha_complex.real();
        
        // w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, krylov_vectors[j].data(), 1, w.data(), 1);
        
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j-1], 0.0);
            cblas_zaxpy(N, &neg_beta, krylov_vectors[j-1].data(), 1, w.data(), 1);
        }
        
        // Compute beta_j = ||w||
        double beta_j = cblas_dznrm2(N, w.data(), 1);
        
        // Selective reorthogonalization: only if norm drops significantly
        if (j > 0 && beta_j < reortho_threshold * beta[j-1]) {
            // Reorthogonalize against all previous vectors
            for (int i = 0; i <= j; i++) {
                Complex overlap;
                cblas_zdotc_sub(N, krylov_vectors[i].data(), 1, w.data(), 1, &overlap);
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, krylov_vectors[i].data(), 1, w.data(), 1);
            }
            beta_j = cblas_dznrm2(N, w.data(), 1);
        }
        
        beta[j] = beta_j;
        
        // Check for breakdown
        if (beta[j] < breakdown_threshold) {
            effective_dim = j + 1;
            break;
        }
        
        // v_{j+1} = w / beta_j (combined copy and scale)
        Complex inv_beta = Complex(1.0/beta[j], 0.0);
        cblas_zcopy(N, w.data(), 1, krylov_vectors[j+1].data(), 1);
        cblas_zscal(N, &inv_beta, krylov_vectors[j+1].data(), 1);
    }
    
    // Compute last alpha
    if (effective_dim == krylov_dim) {
        H(krylov_vectors[effective_dim-1].data(), w.data(), N);
        Complex alpha_complex;
        cblas_zdotc_sub(N, krylov_vectors[effective_dim-1].data(), 1, w.data(), 1, &alpha_complex);
        alpha[effective_dim-1] = alpha_complex.real();
    }
    
    // Diagonalize tridiagonal matrix using LAPACKE C interface (portable across MKL/AOCL)
    std::vector<double> eigenvalues = alpha;
    std::vector<double> eigenvectors_data(effective_dim * effective_dim);
    std::vector<double> offdiag(beta.begin(), beta.begin() + effective_dim - 1);
    
    lapack_int n = static_cast<lapack_int>(effective_dim);
    lapack_int ldz = static_cast<lapack_int>(effective_dim);
    
    // Use LAPACKE C interface for portability across different BLAS backends
    lapack_int info = LAPACKE_dstev(LAPACK_COL_MAJOR, 'V', n, eigenvalues.data(), 
                                     offdiag.data(), eigenvectors_data.data(), ldz);
    
    if (info != 0) {
        std::cerr << "Warning: Krylov eigendecomposition failed with info=" << info << std::endl;
        return;
    }
    
    // Compute exp(-i*H*t) * e_1 in eigenbasis efficiently
    // Result: c_j = sum_i U[0,i] * exp(-i*λ_i*t) * U[j,i] 
    // This avoids the intermediate transforms that were in the old code
    std::fill(state.begin(), state.end(), Complex(0, 0));
    
    for (int j = 0; j < effective_dim; j++) {
        Complex coeff(0, 0);
        for (int i = 0; i < effective_dim; i++) {
            // eigenvectors stored column-major: U[row,col] = data[col*ldz + row]
            double u_0i = eigenvectors_data[i * ldz];        // U[0,i]
            double u_ji = eigenvectors_data[i * ldz + j];    // U[j,i]
            Complex phase = std::exp(Complex(0, -eigenvalues[i] * delta_t));
            coeff += u_0i * phase * u_ji;
        }
        coeff *= norm;  // Restore original norm
        cblas_zaxpy(N, &coeff, krylov_vectors[j].data(), 1, state.data(), 1);
    }
    
    // Normalize if requested
    if (normalize) {
        double final_norm = cblas_dznrm2(N, state.data(), 1);
        if (final_norm > 1e-14) {
            Complex final_scale = Complex(1.0/final_norm, 0.0);
            cblas_zscal(N, &final_scale, state.data(), 1);
        }
    }
}

// ============================================================================
// Retired in the minimalist-architecture rev (May 2026):
//
//   * time_evolve_taylor / time_evolve_rk4 / time_evolve_chebyshev /
//     time_evolve_adaptive / imaginary_time_evolve_taylor
//     -- alternate integrators that nobody outside this TU called.
//   * create_time_evolution_operator
//     -- lambda-factory wrapper for the above.
//   * compute_time_correlation / compute_multiple_time_correlations /
//     compute_time_correlations_with_U_t /
//     compute_time_correlations_incremental
//     -- dead callers (called only by the now-retired TPQ wrappers in
//        the legacy calculate_spectral_function_from_tpq family).
//   * compute_spectral_function (dynamics.cpp overload returning
//     SpectralFunctionData)
//     -- only ever called by the deleted compute_time_correlation chain;
//        the FTLM static overload in ftlm.cpp is the live spectral path.
//   * compute_operator_dynamics
//     -- top-level driver for the legacy TPQ time-evolution path; no
//        external callers.
//
// The live quantum-dynamics surface is now just time_evolve_krylov +
// ed::observables::time_evolution_correlator. See ftlm.cpp for the
// CF/KPM spectral functions.
// ============================================================================

