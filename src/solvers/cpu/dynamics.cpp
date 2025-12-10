// dynamics.cpp - Implementation of general quantum dynamics computation module

#include <ed/solvers/dynamics.h>
#include <ed/core/hdf5_io.h>  // For HDF5 output

// ============================================================================
// TIME EVOLUTION METHODS
// ============================================================================

void time_evolve_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    uint64_t n_max,
    bool normalize
) {
    // Temporary vectors for calculation
    ComplexVector result(N);
    ComplexVector term(N);
    ComplexVector Hterm(N);
    
    // Copy initial state to term
    std::copy(state.begin(), state.end(), term.begin());
    
    // Copy initial state to result for the first term in Taylor series
    std::copy(state.begin(), state.end(), result.begin());
    
    // Precompute coefficients for each term in the Taylor series
    std::vector<Complex> coefficients(n_max + 1);
    coefficients[0] = Complex(1.0, 0.0);  // 0th order term
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        // For exp(-iH*t), each term has (-i)^order
        Complex coef = std::pow(Complex(0.0, -1.0), order);  
        coefficients[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    const double base_norm = std::max(1e-16, cblas_dznrm2(N, state.data(), 1));
    const double contribution_tol = 1e-12;

    // Apply Taylor expansion terms
    for (int order = 1; order <= n_max; order++) {
        // Apply H to the previous term
        H(term.data(), Hterm.data(), N);
        std::swap(term, Hterm);

        // Add this term to the result
        for (int i = 0; i < N; i++) {
            result[i] += coefficients[order] * term[i];
        }

        double term_norm = cblas_dznrm2(N, term.data(), 1);
        double scaled_norm = std::abs(coefficients[order]) * term_norm;
        if (scaled_norm < contribution_tol * base_norm) {
            break;
        }
    }

    // Replace state with the evolved state
    std::swap(state, result);

    // Normalize if requested
    if (normalize) {
        double norm = cblas_dznrm2(N, state.data(), 1);
        if (norm > 0.0) {
            Complex scale_factor = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale_factor, state.data(), 1);
        }
    }
}

void imaginary_time_evolve_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_beta,
    uint64_t n_max,
    bool normalize
) {
    // result = sum_{n=0}^{n_max} (-Δβ H)^n / n! |ψ⟩
    ComplexVector term(N), Hterm(N), result(N);
    std::copy(state.begin(), state.end(), term.begin());
    std::copy(state.begin(), state.end(), result.begin());

    // Iteratively build coefficients to avoid factorial overflow
    // c0 = 1; c_{k} = c_{k-1} * (-Δβ) / k
    double coef_real = 1.0;
    double base_norm = std::max(1e-16, cblas_dznrm2(N, state.data(), 1));
    const double contribution_tol = 1e-12;

    for (int order = 1; order <= n_max; ++order) {
        // term <- H * term
        H(term.data(), Hterm.data(), N);
        std::swap(term, Hterm);

        coef_real *= (-delta_beta) / double(order);
        Complex coef(coef_real, 0.0);

        for (int i = 0; i < N; ++i) {
            result[i] += coef * term[i];
        }

        double term_norm = cblas_dznrm2(N, term.data(), 1);
        double scaled_norm = std::abs(coef_real) * term_norm;
        if (scaled_norm < contribution_tol * base_norm) {
            break;
        }
    }

    std::swap(state, result);

    if (normalize) {
        double norm = cblas_dznrm2(N, state.data(), 1);
        if (norm > 0.0) {
            Complex scale(1.0 / norm, 0.0);
            cblas_zscal(N, &scale, state.data(), 1);
        }
    }
}

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
    
    // Diagonalize tridiagonal matrix using LAPACK (faster than Eigen for small matrices)
    std::vector<double> eigenvalues = alpha;
    std::vector<double> eigenvectors_data(effective_dim * effective_dim);
    std::vector<double> offdiag(beta.begin(), beta.begin() + effective_dim - 1);
    std::vector<double> work(std::max(static_cast<uint64_t>(1), 2 * effective_dim - 2));
    
    char jobz = 'V';
    int32_t n = static_cast<int32_t>(effective_dim);
    int32_t ldz = static_cast<int32_t>(effective_dim);
    int32_t info = 0;
    
    // Call LAPACK dstev (AOCL version requires string length as last parameter)
    dstev_(&jobz, &n, eigenvalues.data(), offdiag.data(), 
           eigenvectors_data.data(), &ldz, work.data(), &info, 1);
    
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

void time_evolve_chebyshev(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    double E_min,
    double E_max,
    uint64_t num_terms,
    bool normalize
) {
    // Scale Hamiltonian to [-1, 1] range for Chebyshev expansion
    double E_center = (E_max + E_min) / 2.0;
    double E_scale = (E_max - E_min) / 2.0;

    constexpr double scale_epsilon = 1e-12;
    if (std::abs(E_scale) < scale_epsilon) {
        // Spectrum is effectively flat: evolution is a global phase
        Complex phase = std::exp(Complex(0, -E_center * delta_t));
        cblas_zscal(N, &phase, state.data(), 1);
        if (normalize) {
            double norm = cblas_dznrm2(N, state.data(), 1);
            if (norm > 1e-14) {
                Complex scale_factor = Complex(1.0/norm, 0.0);
                cblas_zscal(N, &scale_factor, state.data(), 1);
            }
        }
        return;
    }
    
    // Scaled time parameter
    double tau = delta_t * E_scale;
    
    // Create scaled Hamiltonian operator: H_scaled = (H - E_center * I) / E_scale
    auto H_scaled = [&](const Complex* input, Complex* output, uint64_t size) {
        H(input, output, size);
        // output = (H - E_center * I) * input / E_scale
        Complex center_factor = Complex(-E_center / E_scale, 0.0);
        Complex scale_factor = Complex(1.0 / E_scale, 0.0);
        
        cblas_zscal(size, &scale_factor, output, 1);
        cblas_zaxpy(size, &center_factor, input, 1, output, 1);
    };
    
    // Initialize result and Chebyshev polynomials
    ComplexVector result(N, Complex(0, 0));
    ComplexVector T_prev(N), T_curr(N), T_next(N);
    
    // T_0(H_scaled) |ψ⟩ = |ψ⟩
    std::copy(state.begin(), state.end(), T_prev.begin());
    
    // First term: J_0(-iτ) * T_0
    double J_0 = std::cyl_bessel_j(0, tau);
    Complex coeff_0 = Complex(J_0, 0.0);
    cblas_zaxpy(N, &coeff_0, T_prev.data(), 1, result.data(), 1);
    
    if (num_terms > 1) {
        // T_1(H_scaled) |ψ⟩ = H_scaled |ψ⟩
        H_scaled(state.data(), T_curr.data(), N);
        
        // Second term: 2 * J_1(-iτ) * (-i) * T_1
        double J_1 = std::cyl_bessel_j(1, tau);
        Complex coeff_1 = Complex(0, -2.0 * J_1);
        cblas_zaxpy(N, &coeff_1, T_curr.data(), 1, result.data(), 1);
        
        // Higher order terms using Chebyshev recurrence
        for (int n = 2; n < num_terms; n++) {
            // T_n = 2 * H_scaled * T_{n-1} - T_{n-2}
            H_scaled(T_curr.data(), T_next.data(), N);
            Complex two = Complex(2.0, 0.0);
            Complex neg_one = Complex(-1.0, 0.0);
            
            cblas_zscal(N, &two, T_next.data(), 1);
            cblas_zaxpy(N, &neg_one, T_prev.data(), 1, T_next.data(), 1);
            
            // Add contribution: 2 * J_n(-iτ) * (-i)^n * T_n
            double J_n = std::cyl_bessel_j(n, tau);
            Complex i_power = std::pow(Complex(0, -1), n);
            Complex coeff_n = 2.0 * J_n * i_power;
            
            cblas_zaxpy(N, &coeff_n, T_next.data(), 1, result.data(), 1);
            
            // Cycle for next iteration
            std::swap(T_prev, T_curr);
            std::swap(T_curr, T_next);
        }
    }
    
    // Apply phase factor from center shift: exp(-i * E_center * delta_t)
    Complex phase = std::exp(Complex(0, -E_center * delta_t));
    cblas_zscal(N, &phase, result.data(), 1);
    
    // Replace state with evolved result
    std::swap(state, result);
    
    // Normalize if requested
    if (normalize) {
        double norm = cblas_dznrm2(N, state.data(), 1);
        if (norm > 0.0) {
            Complex scale_factor = Complex(1.0/norm, 0.0);
            cblas_zscal(N, &scale_factor, state.data(), 1);
        }
    }
}

void time_evolve_rk4(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    bool normalize
) {
    ComplexVector k1(N), k2(N), k3(N), k4(N);
    ComplexVector temp_state(N);
    Complex neg_i = Complex(0, -1);
    Complex half = Complex(0.5, 0);
    
    // k1 = -i * H * ψ(t)
    H(state.data(), k1.data(), N);
    cblas_zscal(N, &neg_i, k1.data(), 1);
    
    // temp_state = ψ(t) + (Δt/2) * k1
    std::copy(state.begin(), state.end(), temp_state.begin());
    Complex dt_half = Complex(delta_t / 2.0, 0);
    cblas_zaxpy(N, &dt_half, k1.data(), 1, temp_state.data(), 1);
    
    // k2 = -i * H * (ψ(t) + (Δt/2) * k1)
    H(temp_state.data(), k2.data(), N);
    cblas_zscal(N, &neg_i, k2.data(), 1);
    
    // temp_state = ψ(t) + (Δt/2) * k2
    std::copy(state.begin(), state.end(), temp_state.begin());
    cblas_zaxpy(N, &dt_half, k2.data(), 1, temp_state.data(), 1);
    
    // k3 = -i * H * (ψ(t) + (Δt/2) * k2)
    H(temp_state.data(), k3.data(), N);
    cblas_zscal(N, &neg_i, k3.data(), 1);
    
    // temp_state = ψ(t) + Δt * k3
    std::copy(state.begin(), state.end(), temp_state.begin());
    Complex dt = Complex(delta_t, 0);
    cblas_zaxpy(N, &dt, k3.data(), 1, temp_state.data(), 1);
    
    // k4 = -i * H * (ψ(t) + Δt * k3)
    H(temp_state.data(), k4.data(), N);
    cblas_zscal(N, &neg_i, k4.data(), 1);
    
    // ψ(t + Δt) = ψ(t) + (Δt/6) * (k1 + 2*k2 + 2*k3 + k4)
    Complex dt_sixth = Complex(delta_t / 6.0, 0);
    Complex two_dt_sixth = Complex(delta_t / 3.0, 0);
    
    cblas_zaxpy(N, &dt_sixth, k1.data(), 1, state.data(), 1);
    cblas_zaxpy(N, &two_dt_sixth, k2.data(), 1, state.data(), 1);
    cblas_zaxpy(N, &two_dt_sixth, k3.data(), 1, state.data(), 1);
    cblas_zaxpy(N, &dt_sixth, k4.data(), 1, state.data(), 1);
    
    // Normalize if requested
    if (normalize) {
        double norm = cblas_dznrm2(N, state.data(), 1);
        Complex scale_factor = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale_factor, state.data(), 1);
    }
}

void time_evolve_adaptive(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_t,
    uint64_t accuracy_level,
    bool normalize
) {
    if (accuracy_level == 1) {
        // Fast: Use improved Taylor with higher order
        time_evolve_taylor(H, state, N, delta_t, 50, normalize);
    } else if (accuracy_level == 2) {
        // Balanced: Use Krylov method
        uint64_t krylov_dim = std::max(static_cast<uint64_t>(1), std::min(static_cast<uint64_t>(30), std::max(N / 2, static_cast<uint64_t>(1))));
        time_evolve_krylov(H, state, N, delta_t, krylov_dim, normalize);
    } else {
        // High accuracy: Use RK4 for small systems, Krylov for large
        if (N < 1000) {
            time_evolve_rk4(H, state, N, delta_t, normalize);
        } else {
            uint64_t krylov_dim = std::max(static_cast<uint64_t>(1), std::min(static_cast<uint64_t>(50), std::max(N / 2, static_cast<uint64_t>(1))));
            time_evolve_krylov(H, state, N, delta_t, krylov_dim, normalize);
        }
    }
}

std::function<void(const Complex*, Complex*, int)> create_time_evolution_operator(
    std::function<void(const Complex*, Complex*, int)> H,
    double delta_t,
    uint64_t n_max,
    bool normalize
) {
    // Precompute coefficients for each term in the Taylor series
    auto coefficients = std::make_shared<std::vector<Complex>>(n_max + 1);
    (*coefficients)[0] = Complex(1.0, 0.0);  // 0th order term
    double factorial = 1.0;
    
    for (int order = 1; order <= n_max; order++) {
        factorial *= order;
        // For exp(-iH*t), each term has (-i)^order
        Complex coef = std::pow(Complex(0.0, -1.0), order);  
        (*coefficients)[order] = coef * std::pow(delta_t, order) / factorial;
    }
    
    // Return a function that applies the time evolution
    return [H, coefficients, n_max, normalize](const Complex* input, Complex* output, uint64_t size) -> void {
        // Temporary vectors for calculation
        std::vector<Complex> term(size);
        std::vector<Complex> Hterm(size);
        std::vector<Complex> result(size);
        
        // Copy input to term and result
        std::copy(input, input + size, term.begin());
        std::copy(input, input + size, result.begin());
        
        // Apply Taylor expansion terms
        for (int order = 1; order <= n_max; order++) {
            // Apply H to the previous term
            H(term.data(), Hterm.data(), size);
            std::swap(term, Hterm);
            
            // Add this term to the result
            for (int i = 0; i < size; i++) {
                result[i] += (*coefficients)[order] * term[i];
            }
        }
        
        // Normalize if requested
        if (normalize) {
            double norm = cblas_dznrm2(size, result.data(), 1);
            if (norm > 0.0) {
                Complex scale_factor = Complex(1.0/norm, 0.0);
                cblas_zscal(size, &scale_factor, result.data(), 1);
            }
        }
        
        // Copy result to output
        std::copy(result.begin(), result.end(), output);
    };
}

// ============================================================================
// DYNAMICAL CORRELATION FUNCTION COMPUTATION
// ============================================================================

std::vector<Complex> compute_time_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    double t_max,
    double dt,
    uint64_t time_evolution_method,
    uint64_t taylor_order,
    uint64_t krylov_dim
) {
    uint64_t num_steps = static_cast<int>(t_max / dt) + 1;
    std::vector<Complex> time_correlation(num_steps);
    
    // Apply O1 to initial state: |φ⟩ = O1|ψ⟩
    ComplexVector O1_psi(N);
    O1(state.data(), O1_psi.data(), N);
    
    // Calculate initial correlation: C(0) = ⟨ψ|O1†O2|ψ⟩
    ComplexVector O2_psi(N);
    O2(state.data(), O2_psi.data(), N);
    
    Complex initial_corr;
    cblas_zdotc_sub(N, O1_psi.data(), 1, O2_psi.data(), 1, &initial_corr);
    time_correlation[0] = initial_corr;
    
    // Initialize evolution states
    ComplexVector evolved_psi(state);
    ComplexVector evolved_O1_psi(O1_psi);
    ComplexVector O2_evolved_psi(N);
    
    // Time evolution loop
    for (int step = 1; step < num_steps; step++) {
        // Evolve states
        if (time_evolution_method == 0) {
            time_evolve_taylor(H, evolved_psi, N, dt, taylor_order, false);
            time_evolve_taylor(H, evolved_O1_psi, N, dt, taylor_order, false);
        } else if (time_evolution_method == 1) {
            time_evolve_krylov(H, evolved_psi, N, dt, krylov_dim, false);
            time_evolve_krylov(H, evolved_O1_psi, N, dt, krylov_dim, false);
        } else if (time_evolution_method == 2) {
            time_evolve_rk4(H, evolved_psi, N, dt, false);
            time_evolve_rk4(H, evolved_O1_psi, N, dt, false);
        }

        // Apply O2 to evolved state
        O2(evolved_psi.data(), O2_evolved_psi.data(), N);

        // Calculate correlation
        Complex corr_t;
        cblas_zdotc_sub(N, evolved_O1_psi.data(), 1, O2_evolved_psi.data(), 1, &corr_t);
        time_correlation[step] = corr_t;
    }
    
    return time_correlation;
}

std::vector<std::vector<Complex>> compute_multiple_time_correlations(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const ComplexVector& state,
    uint64_t N,
    double t_max,
    double dt,
    uint64_t time_evolution_method,
    uint64_t taylor_order,
    uint64_t krylov_dim
) {
    uint64_t num_steps = static_cast<int>(t_max / dt) + 1;
    uint64_t num_operators = operators_1.size();
    
    std::vector<std::vector<Complex>> time_correlations(num_operators, std::vector<Complex>(num_steps));
    
    // Pre-allocate buffers
    std::vector<ComplexVector> O1_psi_vec(num_operators, ComplexVector(N));
    std::vector<ComplexVector> O2_psi_vec(num_operators, ComplexVector(N));
    ComplexVector evolved_psi(state);
    std::vector<ComplexVector> evolved_O1_psi_vec(num_operators, ComplexVector(N));
    std::vector<ComplexVector> O2_evolved_psi_vec(num_operators, ComplexVector(N));
    
    // Initial setup - parallel over operators
    #pragma omp parallel for schedule(static)
    for (int op = 0; op < num_operators; op++) {
        operators_1[op](state.data(), O1_psi_vec[op].data(), N);
        evolved_O1_psi_vec[op] = O1_psi_vec[op];
        
        operators_2[op](state.data(), O2_psi_vec[op].data(), N);
        
        Complex init_corr;
        cblas_zdotc_sub(N, O1_psi_vec[op].data(), 1, O2_psi_vec[op].data(), 1, &init_corr);
        time_correlations[op][0] = init_corr;
    }
    
    // Time evolution loop
    for (int step = 1; step < num_steps; step++) {
        // Evolve state
        if (time_evolution_method == 0) {
            time_evolve_taylor(H, evolved_psi, N, dt, taylor_order, false);
        } else if (time_evolution_method == 1) {
            time_evolve_krylov(H, evolved_psi, N, dt, krylov_dim, false);
        } else if (time_evolution_method == 2) {
            time_evolve_rk4(H, evolved_psi, N, dt, false);
        }
        
        // Parallel over operators
        #pragma omp parallel for schedule(static)
        for (int op = 0; op < num_operators; op++) {
            // Evolve O1_psi
            if (time_evolution_method == 0) {
                time_evolve_taylor(H, evolved_O1_psi_vec[op], N, dt, taylor_order, false);
            } else if (time_evolution_method == 1) {
                time_evolve_krylov(H, evolved_O1_psi_vec[op], N, dt, krylov_dim, false);
            } else if (time_evolution_method == 2) {
                time_evolve_rk4(H, evolved_O1_psi_vec[op], N, dt, false);
            }
            
            // Apply O2 to evolved state
            operators_2[op](evolved_psi.data(), O2_evolved_psi_vec[op].data(), N);
            
            // Calculate correlation
            Complex corr_t;
            cblas_zdotc_sub(N, evolved_O1_psi_vec[op].data(), 1, O2_evolved_psi_vec[op].data(), 1, &corr_t);
            time_correlations[op][step] = corr_t;
        }
    }
    
    return time_correlations;
}

std::vector<std::vector<Complex>> compute_time_correlations_with_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const ComplexVector& state,
    uint64_t N,
    uint64_t num_steps
) {
    uint64_t num_operators = operators_1.size();
    
    // Pre-allocate all buffers needed for calculation
    std::vector<ComplexVector> O_psi_vec(num_operators, ComplexVector(N));
    std::vector<ComplexVector> O_psi_next_vec(num_operators, ComplexVector(N));
    ComplexVector evolved_state(N);
    ComplexVector state_next(N);
    std::vector<ComplexVector> O_dag_state_vec(num_operators, ComplexVector(N));
    
    // Initialize state
    std::copy(state.begin(), state.end(), evolved_state.begin());
    
    // Calculate O|ψ> once for each operator (parallel over operators)
    #pragma omp parallel for schedule(static)
    for (int op = 0; op < num_operators; op++) {
        operators_1[op](evolved_state.data(), O_psi_vec[op].data(), N);
    }
    
    // Prepare time evolution
    std::vector<std::vector<Complex>> time_correlations(num_operators, std::vector<Complex>(num_steps));
    
    // Calculate initial O†|ψ> for each operator (parallel over operators)
    #pragma omp parallel for schedule(static)
    for (int op = 0; op < num_operators; op++) {
        operators_2[op](evolved_state.data(), O_dag_state_vec[op].data(), N);
        
        // Calculate initial correlation C(0) = <ψ|O†O|ψ>
        Complex init_corr = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            init_corr += std::conj(O_dag_state_vec[op][i]) * O_psi_vec[op][i];
        }
        time_correlations[op][0] = init_corr;
    }
    
    // Time evolution loop
    for (int step = 1; step < num_steps; step++) {
        // Evolve state: |ψ(t)> = U_t|ψ(t-dt)>
        U_t(evolved_state.data(), state_next.data(), N);

        // Parallel over operators for this time slice
        #pragma omp parallel for schedule(static)
        for (int op = 0; op < num_operators; op++) {
            // Evolve O_psi: O|ψ(t)> = U_t(O|ψ(t-dt)>)
            U_t(O_psi_vec[op].data(), O_psi_next_vec[op].data(), N);

            // Calculate O†|ψ(t)>
            operators_2[op](state_next.data(), O_dag_state_vec[op].data(), N);

            // Calculate correlation C(t) = <ψ(t)|O†O|ψ(t)>
            Complex corr = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                corr += std::conj(O_dag_state_vec[op][i]) * O_psi_next_vec[op][i];
            }
            time_correlations[op][step] = corr;
        }

        // Update buffers and state for next step
        for (int op = 0; op < num_operators; op++) {
            std::swap(O_psi_vec[op], O_psi_next_vec[op]);
        }
        std::swap(evolved_state, state_next);
    }
    
    return time_correlations;
}

void compute_time_correlations_incremental(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,
    const ComplexVector& state,
    uint64_t N,
    uint64_t num_steps,
    double dt,
    std::vector<std::ofstream>& output_files
) {
    uint64_t num_operators = operators_1.size();
    
    // Pre-allocate all buffers
    std::vector<ComplexVector> O_psi_vec(num_operators, ComplexVector(N));
    std::vector<ComplexVector> O_psi_next_vec(num_operators, ComplexVector(N));
    ComplexVector evolved_state(N);
    ComplexVector state_next(N);
    std::vector<ComplexVector> O_dag_state_vec(num_operators, ComplexVector(N));
    
    // Initialize state
    std::copy(state.begin(), state.end(), evolved_state.begin());
    
    // Calculate O|ψ> once for each operator (parallel over operators)
    #pragma omp parallel for schedule(static)
    for (int op = 0; op < num_operators; op++) {
        operators_1[op](evolved_state.data(), O_psi_vec[op].data(), N);
    }
    
    // Calculate initial O†|ψ> for each operator and write initial time point
    #pragma omp parallel for schedule(static)
    for (int op = 0; op < num_operators; op++) {
        operators_2[op](evolved_state.data(), O_dag_state_vec[op].data(), N);
        
        // Calculate initial correlation C(0) = <ψ|O†O|ψ>
        Complex init_corr = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            init_corr += std::conj(O_dag_state_vec[op][i]) * O_psi_vec[op][i];
        }
        
        // Write initial time point (t=0) to file
        #pragma omp critical
        {
            if (output_files[op].is_open()) {
                output_files[op] << 0.0 << " " 
                          << init_corr.real() << " " 
                          << init_corr.imag() << std::endl;
                output_files[op].flush();
            }
        }
    }
    
    // Time evolution loop
    for (int step = 1; step < num_steps; step++) {
        double current_time = step * dt;
        
        // Evolve state: |ψ(t)> = U_t|ψ(t-dt)>
        U_t(evolved_state.data(), state_next.data(), N);

        // Parallel over operators for this time slice
        #pragma omp parallel for schedule(static)
        for (int op = 0; op < num_operators; op++) {
            // Evolve O_psi: O|ψ(t)> = U_t(O|ψ(t-dt)>)
            U_t(O_psi_vec[op].data(), O_psi_next_vec[op].data(), N);

            // Calculate O†|ψ(t)>
            operators_2[op](state_next.data(), O_dag_state_vec[op].data(), N);

            // Calculate correlation C(t) = <ψ(t)|O†O|ψ(t)>
            Complex corr = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                corr += std::conj(O_dag_state_vec[op][i]) * O_psi_next_vec[op][i];
            }
            
            // Write current time point to file immediately
            #pragma omp critical
            {
                if (output_files[op].is_open()) {
                    output_files[op] << current_time << " " 
                              << corr.real() << " " 
                              << corr.imag() << std::endl;
                    
                    // Flush every 10 steps to balance performance and data safety
                    if (step % 10 == 0) {
                        output_files[op].flush();
                    }
                }
            }
        }

        // Update buffers and state for next step
        for (int op = 0; op < num_operators; op++) {
            std::swap(O_psi_vec[op], O_psi_next_vec[op]);
        }
        std::swap(evolved_state, state_next);
    }
    
    // Final flush for all files
    for (auto& file : output_files) {
        if (file.is_open()) {
            file.flush();
        }
    }
}

SpectralFunctionData compute_spectral_function(
    const std::vector<Complex>& time_correlation,
    double dt,
    double omega_min,
    double omega_max,
    uint64_t num_omega,
    double eta,
    bool use_lorentzian
) {
    SpectralFunctionData result;
    
    // Generate frequency points
    result.frequencies.resize(num_omega);
    double omega_step = (omega_max - omega_min) / (num_omega - 1);
    for (int i = 0; i < num_omega; i++) {
        result.frequencies[i] = omega_min + i * omega_step;
    }
    result.spectral_function.resize(num_omega, 0.0);
    
    uint64_t num_steps = time_correlation.size();
    
    // Perform Fourier transform to get spectral function
    for (int i = 0; i < num_omega; i++) {
        double omega = result.frequencies[i];
        Complex spectral_value = Complex(0.0, 0.0);
        
        for (int step = 0; step < num_steps; step++) {
            double t = step * dt;
            Complex phase = std::exp(Complex(0.0, -omega * t));
            
            // Add damping factor
            double damping;
            if (use_lorentzian) {
                damping = std::exp(-eta * t);
            } else {
                damping = std::exp(-eta * t * t / 2.0);
            }
            
            spectral_value += time_correlation[step] * phase * damping * dt;
        }
        
        // The spectral function is the real part of the Fourier transform
        result.spectral_function[i] = spectral_value.real();
    }
    
    return result;
}

void compute_operator_dynamics(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& state,
    const std::vector<Operator>& operators_1,
    const std::vector<Operator>& operators_2,
    const std::vector<std::string>& operator_names,
    uint64_t N,
    const std::string& output_dir,
    const std::string& label,
    double t_max,
    double dt,
    uint64_t krylov_dim
) {
    std::cout << "Computing dynamical correlations using Krylov method, label: " << label << std::endl;
    
    // Ensure Krylov dimension doesn't exceed system size
    krylov_dim = std::min(krylov_dim, N/2);
    
    uint64_t num_steps = static_cast<int>(t_max / dt) + 1;
    
    // Pre-allocate reusable buffers
    ComplexVector evolved_psi(N);
    ComplexVector evolved_O1_psi(N);
    ComplexVector O2_psi(N);
    ComplexVector O1_psi(N);
    ComplexVector O2_evolved_psi(N);
    
    // Buffers for negative time evolution
    ComplexVector evolved_psi_neg(N);
    ComplexVector evolved_O1_psi_neg(N);
    ComplexVector O2_evolved_psi_neg(N);
    
    // Process each operator pair
    for (size_t op_idx = 0; op_idx < operators_1.size(); op_idx++) {
        std::string op_name = operator_names[op_idx];
        
        std::cout << "  Computing " << op_name << " correlations..." << std::endl;

        // Apply O_1 to initial state: |φ⟩ = O_1|ψ⟩
        operators_1[op_idx].apply(state.data(), O1_psi.data(), N);
        
        // Calculate initial correlation: C(0) = ⟨ψ|O_1†O_2|ψ⟩
        operators_2[op_idx].apply(state.data(), O2_psi.data(), N);
        
        // Use BLAS for dot product
        Complex initial_corr;
        cblas_zdotc_sub(N, O1_psi.data(), 1, O2_psi.data(), 1, &initial_corr);

        std::cout << "    Initial correlation C(0) = " 
                  << initial_corr.real() << " + i*" 
                  << initial_corr.imag() << std::endl;
        
        // Storage for time correlation data
        std::vector<std::tuple<double, double, double>> time_data; // (time, real, imag)
        time_data.reserve(2 * num_steps - 1); // Reserve space for both positive and negative times
        
        // Add initial time point
        time_data.push_back(std::make_tuple(0.0, initial_corr.real(), initial_corr.imag()));
        
        // ===== POSITIVE TIME EVOLUTION =====
        std::cout << "    Computing positive time evolution (0 to " << t_max << ")..." << std::endl;
        std::copy(state.begin(), state.end(), evolved_psi.begin());
        std::copy(O1_psi.begin(), O1_psi.end(), evolved_O1_psi.begin());
        
        for (int step = 1; step < num_steps; step++) {
            // Evolve states using Krylov method
            time_evolve_krylov(H, evolved_psi, N, dt, krylov_dim, false);
            time_evolve_krylov(H, evolved_O1_psi, N, dt, krylov_dim, false);
            
            // Apply O_2 to evolved state
            operators_2[op_idx].apply(evolved_psi.data(), O2_evolved_psi.data(), N);
            
            // Calculate correlation using BLAS
            Complex corr_t;
            cblas_zdotc_sub(N, evolved_O1_psi.data(), 1, O2_evolved_psi.data(), 1, &corr_t);
            
            double t = step * dt;
            time_data.push_back(std::make_tuple(t, corr_t.real(), corr_t.imag()));
            
            if (step % 100 == 0) {
                std::cout << "      Positive time step " << step << " / " << num_steps << std::endl;
            }
        }
        
        // ===== NEGATIVE TIME EVOLUTION =====
        std::cout << "    Computing negative time evolution (0 to " << -t_max << ")..." << std::endl;
        std::copy(state.begin(), state.end(), evolved_psi_neg.begin());
        std::copy(O1_psi.begin(), O1_psi.end(), evolved_O1_psi_neg.begin());
        
        for (int step = 1; step < num_steps; step++) {
            // Evolve states backward in time (use -dt)
            time_evolve_krylov(H, evolved_psi_neg, N, -dt, krylov_dim, false);
            time_evolve_krylov(H, evolved_O1_psi_neg, N, -dt, krylov_dim, false);
            
            // Apply O_2 to evolved state
            operators_2[op_idx].apply(evolved_psi_neg.data(), O2_evolved_psi_neg.data(), N);
            
            // Calculate correlation using BLAS
            Complex corr_t_neg;
            cblas_zdotc_sub(N, evolved_O1_psi_neg.data(), 1, O2_evolved_psi_neg.data(), 1, &corr_t_neg);
            
            double t_neg = -step * dt;
            time_data.push_back(std::make_tuple(t_neg, corr_t_neg.real(), corr_t_neg.imag()));
            
            if (step % 100 == 0) {
                std::cout << "      Negative time step " << step << " / " << num_steps << std::endl;
            }
        }
        
        // Sort by time (ascending order)
        std::sort(time_data.begin(), time_data.end(), 
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        
        // Save time correlation data to HDF5
        std::string h5_file = output_dir + "/ed_results.h5";
        if (!HDF5IO::fileExists(h5_file)) {
            HDF5IO::createOrOpenFile(output_dir);
        }
        
        HDF5IO::TimeCorrelationData h5_data;
        h5_data.times.reserve(time_data.size());
        h5_data.correlation_real.reserve(time_data.size());
        h5_data.correlation_imag.reserve(time_data.size());
        
        for (const auto& data_point : time_data) {
            h5_data.times.push_back(std::get<0>(data_point));
            h5_data.correlation_real.push_back(std::get<1>(data_point));
            h5_data.correlation_imag.push_back(std::get<2>(data_point));
        }
        
        // Use sample=0 for ground state dynamics, beta=0 indicates ground state
        HDF5IO::saveTimeCorrelation(h5_file, op_name, 0, 0.0, h5_data, label);
        
        std::cout << "    Time correlation saved to HDF5" << std::endl;
        std::cout << "    Time range: [" << std::get<0>(time_data.front()) << ", " 
                  << std::get<0>(time_data.back()) << "]" << std::endl;
    }
    
    std::cout << "Dynamical correlation calculation complete for label: " << label << std::endl;
}
