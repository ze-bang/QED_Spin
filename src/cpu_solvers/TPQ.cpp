#include "TPQ.h"

// Note: Time evolution and dynamics computation methods have been moved to dynamics.cpp
// This file now focuses on TPQ-specific functionality and wraps the general dynamics module

// ============================================================================
// TPQ-SPECIFIC WRAPPER FUNCTIONS
// These provide backward compatibility while delegating to the dynamics module
// ============================================================================

/**
 * TPQ-specific wrapper for time evolution using Taylor method
 * Delegates to the general dynamics module
 */
void time_evolve_tpq_state(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int n_max,
    bool normalize
) {
    time_evolve_taylor(H, tpq_state, N, delta_t, n_max, normalize);
}

/**
 * TPQ-specific wrapper for Krylov time evolution
 * Delegates to the general dynamics module
 */
void time_evolve_tpq_krylov(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int krylov_dim,
    bool normalize
) {
    time_evolve_krylov(H, tpq_state, N, delta_t, krylov_dim, normalize);
}

/**
 * TPQ-specific wrapper for Chebyshev time evolution
 * Delegates to the general dynamics module
 */
void time_evolve_tpq_chebyshev(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    double E_min,
    double E_max,
    int num_terms,
    bool normalize
) {
    time_evolve_chebyshev(H, tpq_state, N, delta_t, E_min, E_max, num_terms, normalize);
}

/**
 * TPQ-specific wrapper for RK4 time evolution
 * Delegates to the general dynamics module
 */
void time_evolve_tpq_rk4(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    bool normalize
) {
    time_evolve_rk4(H, tpq_state, N, delta_t, normalize);
}

/**
 * TPQ-specific wrapper for adaptive time evolution
 * Delegates to the general dynamics module
 */
void time_evolve_tpq_adaptive(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& tpq_state,
    int N,
    double delta_t,
    int accuracy_level,
    bool normalize
) {
    time_evolve_adaptive(H, tpq_state, N, delta_t, accuracy_level, normalize);
}

/**
 * Compute dynamical correlations for TPQ using Krylov method
 * Uses the same output format as Taylor method for consistency
 */
void computeDynamicCorrelationsKrylov(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& operators_1,
    const std::vector<Operator>& operators_2,
    const std::vector<std::string>& operator_names,
    int N,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end,
    double dt,
    int krylov_dim
) {
    std::cout << "Computing dynamical susceptibility for sample " << sample 
              << ", beta = " << inv_temp << ", for " << operators_1.size() << " observables" << std::endl;
    
    // Ensure Krylov dimension doesn't exceed system size
    krylov_dim = std::min(krylov_dim, N/2);
    
    int num_steps = static_cast<int>(t_end / dt) + 1;
    
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
        operators_1[op_idx].apply(tpq_state.data(), O1_psi.data(), N);
        
        // Calculate initial correlation: C(0) = ⟨ψ|O_2†O_1|ψ⟩
        operators_2[op_idx].apply(tpq_state.data(), O2_psi.data(), N);
        
        // Use BLAS for dot product
        Complex initial_corr;
        cblas_zdotc_sub(N, O2_psi.data(), 1, O1_psi.data(), 1, &initial_corr);

        std::cout << "    Initial correlation C(0) = " 
                  << initial_corr.real() << " + i*" 
                  << initial_corr.imag() << std::endl;
        
        // Storage for time correlation data
        std::vector<std::tuple<double, double, double>> time_data; // (time, real, imag)
        time_data.reserve(2 * num_steps - 1); // Reserve space for both positive and negative times
        
        // Add initial time point
        time_data.push_back(std::make_tuple(0.0, initial_corr.real(), initial_corr.imag()));
        
        // ===== POSITIVE TIME EVOLUTION =====
        std::cout << "    Computing positive time evolution (0 to " << t_end << ")..." << std::endl;
        std::copy(tpq_state.begin(), tpq_state.end(), evolved_psi.begin());
        std::copy(O1_psi.begin(), O1_psi.end(), evolved_O1_psi.begin());
        
        for (int step = 1; step < num_steps; step++) {
            // Evolve states using Krylov method (forward in time)
            time_evolve_krylov(H, evolved_psi, N, dt, krylov_dim, true);
            time_evolve_krylov(H, evolved_O1_psi, N, dt, krylov_dim, false);
            
            // Apply O_2 to evolved state
            operators_2[op_idx].apply(evolved_psi.data(), O2_evolved_psi.data(), N);
            
            // Calculate correlation using BLAS
            Complex corr_t;
            cblas_zdotc_sub(N, O2_evolved_psi.data(), 1, evolved_O1_psi.data(), 1, &corr_t);
            
            double t = step * dt;
            time_data.push_back(std::make_tuple(t, corr_t.real(), corr_t.imag()));
            
            if (step % 100 == 0) {
                std::cout << "      Positive time step " << step << " / " << num_steps << std::endl;
            }
        }
        
        // ===== NEGATIVE TIME EVOLUTION =====
        std::cout << "    Computing negative time evolution (0 to " << -t_end << ")..." << std::endl;
        std::copy(tpq_state.begin(), tpq_state.end(), evolved_psi_neg.begin());
        std::copy(O1_psi.begin(), O1_psi.end(), evolved_O1_psi_neg.begin());
        
        for (int step = 1; step < num_steps; step++) {
            // Evolve states backward in time (use -dt)
            time_evolve_krylov(H, evolved_psi_neg, N, -dt, krylov_dim, true);
            time_evolve_krylov(H, evolved_O1_psi_neg, N, -dt, krylov_dim, false);
            
            // Apply O_2 to evolved state
            operators_2[op_idx].apply(evolved_psi_neg.data(), O2_evolved_psi_neg.data(), N);
            
            // Calculate correlation using BLAS
            Complex corr_t_neg;
            cblas_zdotc_sub(N, O2_evolved_psi_neg.data(), 1, evolved_O1_psi_neg.data(), 1, &corr_t_neg);
            
            double t_neg = -step * dt;
            time_data.push_back(std::make_tuple(t_neg, corr_t_neg.real(), corr_t_neg.imag()));
            
            if (step % 100 == 0) {
                std::cout << "      Negative time step " << step << " / " << num_steps << std::endl;
            }
        }
        
        // Sort by time (ascending order)
        std::sort(time_data.begin(), time_data.end(), 
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        
        // Write sorted data to file
        std::string time_corr_file = dir + "/time_corr_rand" + std::to_string(sample) + "_" 
                             + op_name + "_beta=" + std::to_string(inv_temp) + ".dat";
        
        std::ofstream time_corr_out(time_corr_file);
        if (!time_corr_out.is_open()) {
            std::cerr << "Error: Could not open file " << time_corr_file << " for writing" << std::endl;
            continue;
        }
        
        time_corr_out << "# t time_correlation_real time_correlation_imag" << std::endl;
        time_corr_out << std::setprecision(16);
        
        for (const auto& data_point : time_data) {
            time_corr_out << std::get<0>(data_point) << " " 
                         << std::get<1>(data_point) << " " 
                         << std::get<2>(data_point) << std::endl;
        }
        
        time_corr_out.close();
        std::cout << "    Time correlation saved to " << time_corr_file << std::endl;
        std::cout << "    Time range: [" << std::get<0>(time_data.front()) << ", " 
                  << std::get<0>(time_data.back()) << "]" << std::endl;
    }
}

/**
 * Legacy wrapper for backward compatibility with old TPQ code
 */
SpectralFunctionData calculate_spectral_function_from_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& tpq_state,
    int N,
    double omega_min,
    double omega_max,
    int num_points,
    double tmax,
    double dt,
    double eta,
    bool use_lorentzian,
    int n_max
) {
    // Compute time correlation using the general dynamics module
    std::vector<Complex> time_corr = compute_time_correlation(
        H, O, O, tpq_state, N, tmax, dt, 0, n_max, 30);
    
    // Compute spectral function from time correlation
    return compute_spectral_function(time_corr, dt, omega_min, omega_max, 
                                    num_points, eta, use_lorentzian);
}

/**
 * Legacy wrapper for computing time correlations with pre-constructed U_t
 */
std::vector<std::vector<Complex>> calculate_spectral_function_from_tpq_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,   
    const ComplexVector& tpq_state,
    int N,
    const int num_steps
) {
    return compute_time_correlations_with_U_t(U_t, operators_1, operators_2, 
                                             tpq_state, N, num_steps);
}

/**
 * Legacy wrapper for incremental time correlation computation
 */
void calculate_spectral_function_from_tpq_U_t_incremental(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_1,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& operators_2,   
    const ComplexVector& tpq_state,
    int N,
    const int num_steps,
    double dt,
    std::vector<std::ofstream>& output_files
) {
    compute_time_correlations_incremental(U_t, operators_1, operators_2, 
                                         tpq_state, N, num_steps, dt, output_files);
}

/**
 * Compute observable dynamics for TPQ with legacy interface
 */
void computeObservableDynamics_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& observables_1,
    const std::vector<Operator>& observables_2,
    const std::vector<std::string>& observable_names,
    int N,
    const std::string& dir,
    int sample,
    double inv_temp,
    double t_end,
    double dt
) {
    // Save the current TPQ state for later analysis
    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
    save_tpq_state(tpq_state, state_file);

    std::cout << "Computing dynamical susceptibility for sample " << sample 
              << ", beta = " << inv_temp << ", for " << observables_1.size() << " observables" << std::endl;
    
    // Prebuild sparse matrices for all observables to ensure thread-safe applies
    for (const auto& op : observables_1) {
        op.buildSparseMatrix();
    }
    for (const auto& op : observables_2) {
        op.buildSparseMatrix();
    }

    int num_steps = static_cast<int>(t_end / dt) + 1;
    
    // Pre-allocate all buffers needed for calculation
    std::vector<ComplexVector> O_psi_vec(observables_1.size(), ComplexVector(N));
    std::vector<ComplexVector> O_psi_next_vec(observables_1.size(), ComplexVector(N));
    ComplexVector evolved_state(N);
    ComplexVector state_next(N);
    std::vector<ComplexVector> O_dag_state_vec(observables_1.size(), ComplexVector(N));
    
    // Buffers for negative time evolution
    std::vector<ComplexVector> O_psi_vec_neg(observables_1.size(), ComplexVector(N));
    std::vector<ComplexVector> O_psi_next_vec_neg(observables_1.size(), ComplexVector(N));
    ComplexVector evolved_state_neg(N);
    ComplexVector state_next_neg(N);
    std::vector<ComplexVector> O_dag_state_vec_neg(observables_1.size(), ComplexVector(N));
    
    // Storage for time correlation data for all observables
    std::vector<std::vector<std::tuple<double, double, double>>> all_time_data(observables_1.size());
    for (auto& time_data : all_time_data) {
        time_data.reserve(2 * num_steps - 1); // Reserve space for both positive and negative times
    }
    
    // ===== INITIALIZE =====
    std::copy(tpq_state.begin(), tpq_state.end(), evolved_state.begin());
    
    // Calculate O|ψ> once for each operator (parallel over operators)
    #pragma omp parallel for schedule(static)
    for (size_t op = 0; op < observables_1.size(); op++) {
        observables_1[op].apply(evolved_state.data(), O_psi_vec[op].data(), N);
    }
    
    // Calculate initial O†|ψ> for each operator and store initial time point
    #pragma omp parallel for schedule(static)
    for (size_t op = 0; op < observables_1.size(); op++) {
        observables_2[op].apply(evolved_state.data(), O_dag_state_vec[op].data(), N);
        
        // Calculate initial correlation C(0) = <ψ|O†O|ψ>
        Complex init_corr = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            init_corr += std::conj(O_dag_state_vec[op][i]) * O_psi_vec[op][i];
        }
        
        all_time_data[op].push_back(std::make_tuple(0.0, init_corr.real(), init_corr.imag()));
    }
    
    // ===== POSITIVE TIME EVOLUTION =====
    std::cout << "  Computing positive time evolution (0 to " << t_end << ")..." << std::endl;
    
    for (int step = 1; step < num_steps; step++) {
        double current_time = step * dt;
        
        // Evolve state: |ψ(t)> = U_t|ψ(t-dt)>
        U_t(evolved_state.data(), state_next.data(), N);

        // Parallel over operators for this time slice
        #pragma omp parallel for schedule(static)
        for (size_t op = 0; op < observables_1.size(); op++) {
            // Evolve O_psi: O|ψ(t)> = U_t(O|ψ(t-dt)>)
            U_t(O_psi_vec[op].data(), O_psi_next_vec[op].data(), N);

            // Calculate O†|ψ(t)>
            observables_2[op].apply(state_next.data(), O_dag_state_vec[op].data(), N);

            // Calculate correlation C(t) = <ψ(t)|O†O|ψ(t)>
            Complex corr = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                corr += std::conj(O_dag_state_vec[op][i]) * O_psi_next_vec[op][i];
            }
            
            all_time_data[op].push_back(std::make_tuple(current_time, corr.real(), corr.imag()));
        }

        // Update buffers and state for next step
        for (size_t op = 0; op < observables_1.size(); op++) {
            std::swap(O_psi_vec[op], O_psi_next_vec[op]);
        }
        std::swap(evolved_state, state_next);
        
        if (step % 100 == 0) {
            std::cout << "    Positive time step " << step << " / " << num_steps << std::endl;
        }
    }
    
    // ===== NEGATIVE TIME EVOLUTION =====
    std::cout << "  Computing negative time evolution (0 to " << -t_end << ")..." << std::endl;
    
    // Re-initialize for negative time evolution
    std::copy(tpq_state.begin(), tpq_state.end(), evolved_state_neg.begin());
    
    // Re-calculate O|ψ> for each operator
    #pragma omp parallel for schedule(static)
    for (size_t op = 0; op < observables_1.size(); op++) {
        observables_1[op].apply(evolved_state_neg.data(), O_psi_vec_neg[op].data(), N);
    }
    
    // Create inverse time evolution operator (U_t^†)
    // Note: For backward evolution, we need to apply U_t^† which for Hermitian H means U(-dt)
    // Since we already have U(dt), we'll create a wrapper that applies the adjoint
    auto U_t_dagger = [&U_t, N](const Complex* in, Complex* out, int size) {
        // For a unitary operator U = exp(-iHt), U^† = exp(iHt)
        // This is equivalent to time evolution with -t
        // However, since we only have U(dt), we need to apply it in reverse
        // For now, we'll compute U^†|ψ> = (U|ψ*>)*
        ComplexVector in_conj(size);
        ComplexVector out_temp(size);
        
        // Conjugate input
        for (int i = 0; i < size; i++) {
            in_conj[i] = std::conj(in[i]);
        }
        
        // Apply U_t
        U_t(in_conj.data(), out_temp.data(), size);
        
        // Conjugate output
        for (int i = 0; i < size; i++) {
            out[i] = std::conj(out_temp[i]);
        }
    };
    
    for (int step = 1; step < num_steps; step++) {
        double current_time = -step * dt;
        
        // Evolve state backward: |ψ(-t)> = U_t^†|ψ(-t+dt)>
        U_t_dagger(evolved_state_neg.data(), state_next_neg.data(), N);

        // Parallel over operators for this time slice
        #pragma omp parallel for schedule(static)
        for (size_t op = 0; op < observables_1.size(); op++) {
            // Evolve O_psi backward
            U_t_dagger(O_psi_vec_neg[op].data(), O_psi_next_vec_neg[op].data(), N);

            // Calculate O†|ψ(-t)>
            observables_2[op].apply(state_next_neg.data(), O_dag_state_vec_neg[op].data(), N);

            // Calculate correlation C(-t) = <ψ(-t)|O†O|ψ(-t)>
            Complex corr = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                corr += std::conj(O_dag_state_vec_neg[op][i]) * O_psi_next_vec_neg[op][i];
            }
            
            all_time_data[op].push_back(std::make_tuple(current_time, corr.real(), corr.imag()));
        }

        // Update buffers and state for next step
        for (size_t op = 0; op < observables_1.size(); op++) {
            std::swap(O_psi_vec_neg[op], O_psi_next_vec_neg[op]);
        }
        std::swap(evolved_state_neg, state_next_neg);
        
        if (step % 100 == 0) {
            std::cout << "    Negative time step " << step << " / " << num_steps << std::endl;
        }
    }
    
    // ===== WRITE SORTED OUTPUT =====
    std::cout << "  Writing sorted time correlation data..." << std::endl;
    
    for (size_t i = 0; i < observables_1.size(); i++) {
        // Sort by time (ascending order)
        std::sort(all_time_data[i].begin(), all_time_data[i].end(), 
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        
        // Write to file
        std::string time_corr_file = dir + "/time_corr_rand" + std::to_string(sample) + "_" 
                             + observable_names[i] + "_beta=" + std::to_string(inv_temp) + ".dat";
        
        std::ofstream time_corr_out(time_corr_file);
        if (!time_corr_out.is_open()) {
            std::cerr << "Error: Could not open file " << time_corr_file << " for writing" << std::endl;
            continue;
        }
        
        time_corr_out << "# t time_correlation_real time_correlation_imag" << std::endl;
        time_corr_out << std::setprecision(16);
        
        for (const auto& data_point : all_time_data[i]) {
            time_corr_out << std::get<0>(data_point) << " " 
                         << std::get<1>(data_point) << " " 
                         << std::get<2>(data_point) << std::endl;
        }
        
        time_corr_out.close();
        std::cout << "    Time correlation saved to " << time_corr_file << std::endl;
        std::cout << "    Time range: [" << std::get<0>(all_time_data[i].front()) << ", " 
                  << std::get<0>(all_time_data[i].back()) << "]" << std::endl;
    }
}

// ============================================================================
// TPQ-SPECIFIC UTILITY FUNCTIONS
// ============================================================================

/**
 * Generate a random normalized vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector(int N, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v(N);
    
    for (int i = 0; i < N; i++) {
        double real = dist(gen);
        double imag = dist(gen);
        v[i] = Complex(real, imag);
    }
    
    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return std::move(v); // Explicitly signal move semantics for the return value
}

/**
 * Create directory if it doesn't exist
 */
bool ensureDirectoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        std::string cmd = "mkdir -p " + path;
        return system(cmd.c_str()) == 0;
    } else if (info.st_mode & S_IFDIR) {
        // Path exists and is a directory
        return true;
    } else {
        // Path exists but is not a directory
        return false;
    }
}

/**
 * Calculate energy and variance for a TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector
 * @param N Dimension of the Hilbert space
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateEnergyAndVariance(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v,
    int N
) {
    // Calculate H|v⟩
    ComplexVector Hv(N);
    H(v.data(), Hv.data(), N);
    
    // Calculate energy = ⟨v|H|v⟩
    Complex energy_complex = Complex(0, 0);
    for (int i = 0; i < N; i++) {
        energy_complex += std::conj(v[i]) * Hv[i];
    }
    double energy = energy_complex.real();
    
    // Calculate H²|v⟩
    ComplexVector H2v(N);
    H(Hv.data(), H2v.data(), N);
    
    // Calculate variance = ⟨v|H²|v⟩ - ⟨v|H|v⟩²
    Complex h2_complex = Complex(0, 0);
    for (int i = 0; i < N; i++) {
        h2_complex += std::conj(v[i]) * H2v[i];
    }
    double variance = h2_complex.real() - energy * energy;
    
    return {energy, variance};
}

std::vector<SingleSiteOperator> createSzOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sz_ops;
    for (int site = 0; site < num_sites; site++) {
        Sz_ops.emplace_back(num_sites, spin_length, 2, site);
    }
    return Sz_ops;
}

std::vector<SingleSiteOperator> createSxOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sx_ops;
    for (int site = 0; site < num_sites; site++) {
        Sx_ops.emplace_back(num_sites, spin_length, 3, site);
    }
    return Sx_ops;
}

std::vector<SingleSiteOperator> createSyOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sy_ops;
    for (int site = 0; site < num_sites; site++) {
        Sy_ops.emplace_back(num_sites, spin_length, 4, site);
    }
    return Sy_ops;
}

std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit to avoid overflow)
    
    ComplexVector Sz_exps(sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Sz2_exps(sublattice_size+1, Complex(0.0, 0.0));
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        int i = site % sublattice_size;

        // Apply operators - use direct vector construction to avoid copy
        std::vector<Complex> Sz_psi = Sz_ops[i].apply({tpq_state.begin(), tpq_state.end()});
        
        // Calculate expectation values
        Complex Sz_exp = Complex(0.0, 0.0);
        
        for (size_t j = 0; j < N; j++) {
            Sz_exp += std::conj(tpq_state[j]) * Sz_psi[j];
        }
        
        // Store expectation values
        Sz_exps[i] += Sz_exp;

        // Apply operator directly to avoid temporary vector copy
        std::vector<Complex> Sz2_psi = Sz_ops[i].apply(std::move(Sz_psi));

        Complex Sz2_exp = Complex(0.0, 0.0);
    for (size_t j = 0; j < N; j++) {
            Sz2_exp += std::conj(tpq_state[j]) * Sz2_psi[j];
        }
        Sz2_exps[i] += Sz2_exp;
    }

    for (int i = 0; i < sublattice_size; i++) {
        Sz_exps[i] /= double(num_sites);
        Sz2_exps[i] /= double(num_sites);
        Sz_exps[sublattice_size] += Sz_exps[i];
        Sz2_exps[sublattice_size] += Sz2_exps[i];
    }


    return {Sz_exps, Sz2_exps};
}


Complex calculateSpm_onsite(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Spm_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)

    Complex Spm_exp(0.0, 0.0);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        int i = site % sublattice_size;

        // Apply operators - use direct vector construction to avoid copy
        std::vector<Complex> Spm_psi = Spm_ops[i].apply({tpq_state.begin(), tpq_state.end()});

    for (size_t j = 0; j < N; j++) {
            Spm_exp += std::conj(Spm_psi[j]) * Spm_psi[j];
        }
    }

    return Spm_exp/ double(num_sites);
}


std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> createDoubleSiteOperators(int num_sites, float spin_length) {
    std::vector<DoubleSiteOperator> Szz_ops;
    std::vector<DoubleSiteOperator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            Szz_ops.emplace_back(num_sites, spin_length, 2, site, 2, site2);
            Spm_ops.emplace_back(num_sites, spin_length, 0, site, 1, site2);
        }
    }
    return {Szz_ops, Spm_ops};
}


std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> createSingleOperators_pair(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Szz_ops;
    std::vector<SingleSiteOperator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        Szz_ops.emplace_back(num_sites, spin_length, 2, site);
        Spm_ops.emplace_back(num_sites, spin_length, 0, site);
    }
    return {Szz_ops, Spm_ops};
}



std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> double_site_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));

    // Create S operators for each site
    std::vector<DoubleSiteOperator> Szz_ops;
    std::vector<DoubleSiteOperator> Spm_ops;

    Szz_ops = double_site_ops.first;
    Spm_ops = double_site_ops.second;            // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            int n1 = site % sublattice_size;
            int n2 = site2 % sublattice_size;

            // Apply operators
            std::vector<Complex> Szz_psi = Szz_ops[site*num_sites+site2].apply({tpq_state.begin(), tpq_state.end()});
            std::vector<Complex> Spm_psi = Spm_ops[site*num_sites+site2].apply({tpq_state.begin(), tpq_state.end()});

            // Calculate expectation values
            Complex Szz_exp = Complex(0.0, 0.0);
            Complex Spm_exp = Complex(0.0, 0.0);


            for (size_t i = 0; i < N; i++) {
                Szz_exp += std::conj(tpq_state[i]) * Szz_psi[i];
            }
            for (size_t i = 0; i < N; i++) {
                Spm_exp += std::conj(tpq_state[i]) * Spm_psi[i];
            }
            Spm_exps[n1*sublattice_size+n2] += Spm_exp;
            Szz_exps[n1*sublattice_size+n2] += Szz_exp;
        }
    }

    for (int i = 0; i < sublattice_size*sublattice_size; i++) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size*sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size*sublattice_size] += Szz_exps[i];
    }
    
    return {Szz_exps, Spm_exps};

}

std::tuple<std::vector<Complex>, std::vector<Complex>, std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops,
    int sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spp_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    // Create S operators for each site
    std::vector<SingleSiteOperator> Szz_ops;
    std::vector<SingleSiteOperator> Spm_ops;

    Szz_ops = double_site_ops.first;
    Spm_ops = double_site_ops.second;            // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            int n1 = site % sublattice_size;
            int n2 = site2 % sublattice_size;

            // Apply operators
            // SzSz
            std::vector<Complex> Szz_psi = Szz_ops[site].apply({tpq_state.begin(), tpq_state.end()});
            std::vector<Complex> Szz_psi2 = Szz_ops[site2].apply({tpq_state.begin(), tpq_state.end()});

            // S+S-
            std::vector<Complex> Spm_psi = Spm_ops[site].apply({tpq_state.begin(), tpq_state.end()});
            std::vector<Complex> Spm_psi2 = Spm_ops[site2].apply({tpq_state.begin(), tpq_state.end()});

            // S+S+
            std::vector<Complex> Spp_psi = Spm_ops[site2].apply({Spm_psi.begin(), Spm_psi.end()});

            // Calculate expectation values
            Complex Szz_exp = Complex(0.0, 0.0);
            Complex Spm_exp = Complex(0.0, 0.0);
            Complex Spp_exp = Complex(0.0, 0.0);
            Complex Spz_exp = Complex(0.0, 0.0);

            for (size_t i = 0; i < N; i++) {
                Szz_exp += std::conj(Szz_psi[i]) * Szz_psi2[i];
            }
            for (size_t i = 0; i < N; i++) {
                Spm_exp += std::conj(Spm_psi[i]) * Spm_psi2[i];
            }
            for (size_t i = 0; i < N; i++) {
                Spp_exp += std::conj(tpq_state[i]) * Spp_psi[i];
            }
            for (size_t i = 0; i < N; i++) {
                Spz_exp += std::conj(Spm_psi[i]) * Szz_psi2[i];
            }
            Spm_exps[n1*sublattice_size+n2] += Spm_exp;
            Szz_exps[n1*sublattice_size+n2] += Szz_exp;
            Spp_exps[n1*sublattice_size+n2] += Spp_exp;
            Spz_exps[n1*sublattice_size+n2] += Spz_exp;

        }
    }

    for (int i = 0; i < sublattice_size*sublattice_size; i++) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spp_exps[i] /= double(num_sites);
        Spz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size*sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size*sublattice_size] += Szz_exps[i];
        Spp_exps[sublattice_size*sublattice_size] += Spp_exps[i];
        Spz_exps[sublattice_size*sublattice_size] += Spz_exps[i];
    }
    
    return {Szz_exps, Spm_exps, Spp_exps, Spz_exps};

}


/**
 * Write TPQ data to file
 */
void writeTPQData(const std::string& filename, double inv_temp, double energy, 
                 double variance, double norm, int step) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << std::setprecision(16) << inv_temp << " " << energy << " " 
             << variance << " " << 0.0 << " " << 0.0 << " " << step << std::endl;
        file.close();
    }
}

/**
 * Read TPQ data from file
 */
bool readTPQData(const std::string& filename, int step, double& energy, 
                double& temp, double& specificHeat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    double inv_temp, e, var, n, doublon;
    int s;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (!(iss >> inv_temp >> e >> var >> n >> doublon >> s)) {
            continue;
        }
        
        if (s == step) {
            energy = e;
            temp = 1.0/inv_temp;
            specificHeat = (var-e*e)*(inv_temp*inv_temp);
            return true;
        }
    }
    
    return false;
}



/**
 * Save the current TPQ state to a file
 * 
 * @param tpq_state TPQ state vector to save
 * @param filename Name of the file to save to
 * @return True if successful
 */
bool save_tpq_state(const ComplexVector& tpq_state, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    size_t size = tpq_state.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(tpq_state.data()), size * sizeof(Complex));
    
    out.close();
    return true;
}

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    tpq_state.resize(size);
    in.read(reinterpret_cast<char*>(tpq_state.data()), size * sizeof(Complex));
    
    in.close();
    return true;
}


/**
 * Load eigenvector data from a raw binary file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @param N Expected size of the vector
 * @return True if successful
 */
bool load_raw_data(ComplexVector& tpq_state, const std::string& filename, int N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    tpq_state.resize(N);
    in.read(reinterpret_cast<char*>(tpq_state.data()), N * sizeof(Complex));
    
    if (!in.good()) {
        std::cerr << "Error: Failed to read data from " << filename << std::endl;
        in.close();
        return false;
    }
    
    in.close();
    return true;
}

/**
 * Compute spin expectations (S^+, S^-, S^z) at each site using a TPQ state
 * 
 * @param tpq_state The TPQ state vector
 * @param num_sites Number of lattice sites
 * @param spin_l Spin value (e.g., 0.5 for spin-1/2)
 * @param output_file Output file path
 * @param print_output Whether to print results to console
 * @return Vector of spin expectation values organized as [site][S+,S-,Sz]
 */
std::vector<std::vector<Complex>> compute_spin_expectations_from_tpq(
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_l,
    const std::string& output_file,
    bool print_output
) {
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    // Initialize expectations matrix: 3 rows (S^+, S^-, S^z) x num_sites columns
    std::vector<std::vector<Complex>> expectations(3, std::vector<Complex>(num_sites, Complex(0.0, 0.0)));
    
    // Create S operators for each site
    std::vector<SingleSiteOperator> Sp_ops;
    std::vector<SingleSiteOperator> Sm_ops;
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.emplace_back(num_sites, spin_l, 0, site);
        Sm_ops.emplace_back(num_sites, spin_l, 1, site);
        Sz_ops.emplace_back(num_sites, spin_l, 2, site);
    }
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        // Apply operators
        std::vector<Complex> Sp_psi = Sp_ops[site].apply({tpq_state.begin(), tpq_state.end()});
        std::vector<Complex> Sm_psi = Sm_ops[site].apply({tpq_state.begin(), tpq_state.end()});
        std::vector<Complex> Sz_psi = Sz_ops[site].apply({tpq_state.begin(), tpq_state.end()});
        
        // Calculate expectation values
        Complex Sp_exp = Complex(0.0, 0.0);
        Complex Sm_exp = Complex(0.0, 0.0);
        Complex Sz_exp = Complex(0.0, 0.0);
        
    for (size_t i = 0; i < N; i++) {
            Sp_exp += std::conj(tpq_state[i]) * Sp_psi[i];
            Sm_exp += std::conj(tpq_state[i]) * Sm_psi[i];
            Sz_exp += std::conj(tpq_state[i]) * Sz_psi[i];
        }
        
        // Store expectation values
        expectations[0][site] = Sp_exp;
        expectations[1][site] = Sm_exp;
        expectations[2][site] = Sz_exp;
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin Expectation Values from TPQ state:" << std::endl;
        std::cout << std::setw(5) << "Site" 
                << std::setw(20) << "S^+ (real)" 
                << std::setw(20) << "S^+ (imag)" 
                << std::setw(20) << "S^- (real)"
                << std::setw(20) << "S^- (imag)"
                << std::setw(20) << "S^z (real)"
                << std::setw(20) << "S^z (imag)" << std::endl;
        
        for (int site = 0; site < num_sites; site++) {
            std::cout << std::setw(5) << site 
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
    }
    
    // Save to file if output_file is specified
    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << "# Site S+_real S+_imag S-_real S-_imag Sz_real Sz_imag" << std::endl;
            for (int site = 0; site < num_sites; site++) {
                out << site << " "
                    << std::setprecision(10) << expectations[0][site].real() << " "
                    << std::setprecision(10) << expectations[0][site].imag() << " "
                    << std::setprecision(10) << expectations[1][site].real() << " "
                    << std::setprecision(10) << expectations[1][site].imag() << " "
                    << std::setprecision(10) << expectations[2][site].real() << " "
                    << std::setprecision(10) << expectations[2][site].imag() << std::endl;
            }
            out.close();
            std::cout << "Spin expectations saved to " << output_file << std::endl;
        }
    }
    
    return expectations;
}





void writeFluctuationData(
    const std::string& flct_file,
    const std::vector<std::string>& spin_corr,
    double inv_temp,
    const ComplexVector& tpq_state,
    int num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sx_ops,
    const std::vector<SingleSiteOperator>& Sy_ops,
    const std::vector<SingleSiteOperator>& Sz_ops,
    const std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>>& double_site_ops,
    int sublattice_size,
    int step
) {
    auto [Sz, Sz2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sz_ops, sublattice_size);
    auto [Sy, Sy2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sy_ops, sublattice_size);
    auto [Sx, Sx2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sx_ops, sublattice_size);

    auto Spm2exp = calculateSpm_onsite(tpq_state, num_sites, spin_length, double_site_ops.second, sublattice_size);

    std::ofstream flct_out(flct_file, std::ios::app);
    flct_out << std::setprecision(16) << inv_temp 
             << " " << Sz[sublattice_size].real() << " " << Sz[sublattice_size].imag() 
             << " " << Sz2[sublattice_size].real() << " " << Sz2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out << " " << Sz[i].real() << " " << Sz[i].imag() 
                 << " " << Sz2[i].real() << " " << Sz2[i].imag();
    }

    flct_out << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();

    flct_out << " " << step << std::endl;   

    std::string flct_file_x_string = flct_file.substr(0,flct_file.size()-4) + "_Sx.dat";
    std::ofstream flct_out_x(flct_file_x_string, std::ios::app);
    flct_out_x << std::setprecision(16) << inv_temp 
               << " " << Sx[sublattice_size].real() << " " << Sx[sublattice_size].imag() 
               << " " << Sx2[sublattice_size].real() << " " << Sx2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out_x << " " << Sx[i].real() << " " << Sx[i].imag() 
                   << " " << Sx2[i].real() << " " << Sx2[i].imag();
    }

    flct_out_x << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();

    flct_out_x << " " << step << std::endl;

    std::string flct_file_y_string = flct_file.substr(0,flct_file.size()-4) + "_Sy.dat";
    std::ofstream flct_out_y(flct_file_y_string, std::ios::app);
    flct_out_y << std::setprecision(16) << inv_temp 
               << " " << Sy[sublattice_size].real() << " " << Sy[sublattice_size].imag() 
               << " " << Sy2[sublattice_size].real() << " " << Sy2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out_y << " " << Sy[i].real() << " " << Sy[i].imag() 
                   << " " << Sy2[i].real() << " " << Sy2[i].imag();
    }

    flct_out_y << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();

    flct_out_y << " " << step << std::endl;


    auto [Szz, Spm, Spp, Spz] = calculateSzzSpm(tpq_state, num_sites, spin_length, double_site_ops, sublattice_size);
    for (size_t idx = 0; idx < spin_corr.size(); idx++) {
        std::ofstream corr_out(spin_corr[idx], std::ios::app);
        
        corr_out << std::setprecision(16) << inv_temp;
        
        // Write total (last element)
        std::vector<Complex>* data_ptr = nullptr;
        if (idx == 0) data_ptr = &Szz;
        else if (idx == 1) data_ptr = &Spm;
        else if (idx == 2) data_ptr = &Spp;
        else if (idx == 3) data_ptr = &Spz;
        
        corr_out << " " << (*data_ptr)[sublattice_size*sublattice_size].real() 
                 << " " << (*data_ptr)[sublattice_size*sublattice_size].imag();
        
        // Write individual correlations
        for (int i = 0; i < sublattice_size*sublattice_size; i++) {
            corr_out << " " << (*data_ptr)[i].real() 
                     << " " << (*data_ptr)[i].imag();
        }
        
        corr_out << " " << step << std::endl;
        corr_out.close();
    }
}

/**
 * Get a TPQ state at a specific inverse temperature by loading the closest available state
 * 
 * @param tpq_dir Directory containing TPQ data
 * @param sample TPQ sample index
 * @param target_beta Target inverse temperature
 * @param N Dimension of Hilbert space
 * @return TPQ state vector at the specified temperature
 */
ComplexVector get_tpq_state_at_temperature(
    const std::string& tpq_dir,
    int sample,
    double target_beta,
    int N
) {
    std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::ifstream file(ss_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open TPQ data file " << ss_file << std::endl;
        return ComplexVector(N);
    }
    
    // Skip header
    std::string line;
    std::getline(file, line);
    
    double best_beta = 0.0;
    int best_step = 0;
    double min_diff = std::numeric_limits<double>::max();
    
    // Find the step with the closest inverse temperature
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double beta, energy, variance, norm, doublon;
        int step;
        
        if (!(iss >> beta >> energy >> variance >> norm >> doublon >> step)) {
            continue;
        }
        
        double diff = std::abs(beta - target_beta);
        if (diff < min_diff) {
            min_diff = diff;
            best_beta = beta;
            best_step = step;
        }
    }
    file.close();
    
    if (best_step == 0) {
        std::cerr << "Error: Could not find appropriate TPQ state" << std::endl;
        return ComplexVector(N);
    }
    
    std::cout << "Loading TPQ state at step " << best_step 
              << ", beta = " << best_beta 
              << " (target beta = " << target_beta << ")" << std::endl;
    
    // Load the state from file
    std::string state_file = tpq_dir + "/tpq_state_" + std::to_string(sample) 
                             + "_step" + std::to_string(best_step) + ".dat";
    
    ComplexVector tpq_state(N);
    if (!load_tpq_state(tpq_state, state_file)) {
        std::cerr << "Error: Could not load TPQ state from " << state_file << std::endl;
        return ComplexVector(N);
    }
    
    return tpq_state;
}



/**
 * Initialize TPQ output files with appropriate headers
 * 
 * @param dir Directory for output files
 * @param sample Current sample index
 * @param sublattice_size Size of sublattice for measurements
 * @return Tuple of filenames (ss_file, norm_file, flct_file, spin_corr)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>> initializeTPQFiles(
    const std::string& dir,
    int sample,
    int sublattice_size
) {
    std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
    std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
    
    // Create vector of spin correlation files
    std::vector<std::string> spin_corr_files;
    std::vector<std::string> suffixes = {"SzSz", "SpSm", "SmSm", "SpSz"};
    
    for (const auto& suffix : suffixes) {
        std::string filename = dir + "/spin_corr_" + suffix + "_rand" + std::to_string(sample) + ".dat";
        spin_corr_files.push_back(filename);
    }
    
    // Initialize output files
    {
        std::ofstream ss_out(ss_file);
        ss_out << "# inv_temp energy variance num doublon step" << std::endl;
        
        std::ofstream norm_out(norm_file);
        norm_out << "# inv_temp norm first_norm step" << std::endl;
        
        std::ofstream flct_out(flct_file);
        flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag)";

        for (int i = 0; i < sublattice_size; i++) {
            flct_out << " sz" << i << "(real) sz" << i << "(imag)"  << " sz2" << i << "(real) sz2" << i << "(imag)";
        }

        flct_out << " Spm2(real) Spm2(imag)";

        flct_out << " step" << std::endl;

        // Initialize each spin correlation file
        for (const auto& file : spin_corr_files) {
            std::ofstream spin_out(file);
            spin_out << "# inv_temp total(real) total(imag)";
            
            for (int i = 0; i < sublattice_size*sublattice_size; i++) {
                spin_out << " site" << i << "(real) site" << i << "(imag)";
            }
            spin_out << " step" << std::endl;
        }
    }
    
    return {ss_file, norm_file, flct_file, spin_corr_files};
}


/**
 * Calculate spectrum function from TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param tpq_sample Sample index to use from TPQ calculation
 * @param tpq_step TPQ step to use
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param omega_step Step size in frequency domain
 * @param eta Broadening factor
 * @param tpq_dir Directory containing TPQ data
 * @param out_file Output file for spectrum
 */
void calculate_spectrum_from_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int tpq_sample,
    int tpq_step,
    double omega_min,
    double omega_max,
    double omega_step,
    double eta,
    const std::string& tpq_dir,
    const std::string& out_file
) {
    std::cout << "Calculating spectrum from TPQ state..." << std::endl;
    
    // Read TPQ data
    std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(tpq_sample) + ".dat";
    double energy, temp, specificHeat;
    
    if (!readTPQData(ss_file, tpq_step, energy, temp, specificHeat)) {
        std::cerr << "Error: Could not read TPQ data from " << ss_file << std::endl;
        return;
    }
    
    std::cout << "Using TPQ state at step " << tpq_step 
              << ", temperature: " << temp 
              << ", energy: " << energy << std::endl;
    
    // Open output file
    std::ofstream spectrum_file(out_file);
    if (!spectrum_file.is_open()) {
        std::cerr << "Error: Could not open output file " << out_file << std::endl;
        return;
    }
    spectrum_file << "# omega re(spectrum) im(spectrum)" << std::endl;
    
    // Calculate number of frequency points
    int n_omega = static_cast<int>((omega_max - omega_min) / omega_step) + 1;
    
    // Pre-factor for Gaussian broadening
    double pre_factor = 2.0 * temp * temp * specificHeat;
    double factor = 1.0 / sqrt(M_PI * pre_factor);
    
    // Calculate spectrum for each frequency
    for (int i = 0; i < n_omega; i++) {
        double omega = omega_min + i * omega_step;
        Complex z(omega, eta); // Complex frequency with broadening
        
        // This is a simplified version - the full algorithm would perform
        // continued fraction expansion using Lanczos tridiagonalization
        
        // Calculate the spectrum using Gaussian broadening approximation
        double spectrum_val = factor * exp(-pow((omega - energy), 2) / pre_factor);
        
        spectrum_file << std::setprecision(16) 
                     << omega << " " 
                     << spectrum_val << " " 
                     << 0.0 << std::endl;
    }
    
    spectrum_file.close();
    std::cout << "Spectrum calculation complete. Written to " << out_file << std::endl;
}


/**
 * Standard TPQ (microcanonical) implementation
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param compute_spectrum Whether to compute spectrum
 */
void microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N, 
    int max_iter,
    int num_samples,
    int temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_spectrum,
    double LargeValue,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    int num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    int sublattice_size,
    int num_sites
) {
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    double D_S = std::log2(N);
    eigenvalues.clear();

    // Create Sz operators
    std::vector<SingleSiteOperator> Sz_ops = createSzOperators(num_sites, spin_length);
    std::vector<SingleSiteOperator> Sx_ops = createSxOperators(num_sites, spin_length);
    std::vector<SingleSiteOperator> Sy_ops = createSyOperators(num_sites, spin_length);

    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops = createSingleOperators_pair(num_sites, spin_length);

    std::function<void(const Complex*, Complex*, int)> U_t;
    std::function<void(const Complex*, Complex*, int)> U_nt;
    std::vector<std::vector<double>> momentum_positions;
    if (compute_observables) {
        momentum_positions = {{0,0,0},
                             {0,0,4*M_PI},
                             {0,0,2*M_PI}};
    }
    std::cout << "Begin TPQ calculation with dimension " << N << std::endl;
    std::string position_file;
    if (!dir.empty()) {
        size_t last_slash_pos = dir.find_last_of('/');
        if (last_slash_pos != std::string::npos) {
            // Check if the last character is a slash, if so, find the previous one
            if (last_slash_pos == dir.length() - 1) {
                last_slash_pos = dir.find_last_of('/', last_slash_pos - 1);
            }
            if (last_slash_pos != std::string::npos) {
                position_file = dir.substr(0, last_slash_pos) + "/positions.dat";
            } else {
                position_file = "positions.dat"; // In case dir is just a name without slashes
            }
        } else {
            position_file = "positions.dat"; // Relative path
        }
    }



    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    // For each random sample
    for (int sample = 0; sample < num_samples; sample++) {
        std::vector<bool> temp_measured(num_temp_points, false);
        std::cout << "TPQ sample " << sample+1 << " of " << num_samples << std::endl;
        
        // Setup filenames
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);
        
        // Generate initial random state
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector v1 = generateTPQVector(N, seed);
        
        // Apply hamiltonian to get v0 = H|v1⟩
        ComplexVector v0(N);
        H(v1.data(), v0.data(), N);
        // For each element, compute v0 = (L-H)|v1⟩ = Lv1 - v0
        for (int i = 0; i < N; i++) {
            v0[i] = (LargeValue * num_sites * v1[i]) - v0[i];
        }
        H(v1.data(), v0.data(), N);
        
        // Write initial state (infinite temperature)
        double inv_temp = 0.0;
        int step = 1;
        
        // Calculate energy and variance for step 1
        auto [energy1, variance1] = calculateEnergyAndVariance(H, v0, N);
        double nsite = N; // This should be the actual number of sites, approximating as N for now
        inv_temp = (2.0) / (LargeValue* D_S - energy1);

        double first_norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale_factor = Complex(1.0/first_norm, 0.0);

        cblas_zscal(N, &scale_factor, v0.data(), 1);

        double current_norm = first_norm;
        
        writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, step);
        
        {
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " 
                     << current_norm << " " << first_norm << " " << step << std::endl;
        }
        
        // Main TPQ loop
        for (step = 2; step <= max_iter; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == max_iter) {
                std::cout << "  Step " << step << " of " << max_iter << std::endl;
            }
            
            // Compute v1 = H|v0⟩
            H(v0.data(), v1.data(), N);
            
            // For each element, compute v0 = (L-H)|v0⟩ = L*v0 - v1
            for (int i = 0; i < N; i++) {
                v0[i] = (LargeValue * D_S * v0[i]) - v1[i];
            }

            current_norm = cblas_dznrm2(N, v0.data(), 1);
            scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v0.data(), 1);
            
            // Calculate energy and variance
            auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v0, N);
            // Update inverse temperature
            inv_temp = (2.0*step) / (LargeValue * num_sites - energy_step);
            
            // Write data
            writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            energy1 = energy_step;

            // Write fluctuation data at specified intervals
            if (step % temp_interval == 0 || step == max_iter) {
                if (measure_sz){
                    writeFluctuationData(flct_file, spin_corr, inv_temp, v0, num_sites, spin_length, Sx_ops, Sy_ops, Sz_ops, double_site_ops, sublattice_size, step);
                }
            }
            // If inv_temp is at one of the specified inverse temperature points, compute observables
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i] && std::abs(inv_temp - measure_inv_temp[i]) < 4e-3) {
                    std::cout << "Computing observables at inv_temp = " << inv_temp << std::endl;
                    if (compute_observables) {
                        // computeSpinStructureFactorKrylov(H, v0, momentum_positions, position_file, N, num_sites, spin_length, dir, sample, inv_temp);
                        // Just save the state for now
                        std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
                        save_tpq_state(v0, state_file);
                    }
                    temp_measured[i] = true; // Mark this temperature as measured
                }
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy1);
    }
}

// Canonical TPQ using imaginary-time propagation e^{-βH} |r>
inline void imaginary_time_evolve_tpq_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    int N,
    double delta_beta,
    int n_max,
    bool normalize
){
    // result = sum_{n=0}^{n_max} (-Δβ H)^n / n! |ψ⟩
    ComplexVector term(N), Hterm(N), result(N);
    std::copy(state.begin(), state.end(), term.begin());
    std::copy(state.begin(), state.end(), result.begin());

    // Iteratively build coefficients to avoid factorial overflow
    // c0 = 1; c_{k} = c_{k-1} * (-Δβ) / k
    double coef_real = 1.0;
    for (int order = 1; order <= n_max; ++order) {
        // term <- H * term
        H(term.data(), Hterm.data(), N);
        std::swap(term, Hterm);

        coef_real *= (-delta_beta) / double(order);
        Complex coef(coef_real, 0.0);

        for (int i = 0; i < N; ++i) {
            result[i] += coef * term[i];
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

void canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double beta_max,
    int num_samples,
    int temp_interval,
    std::vector<double>& energies,
    std::string dir,
    double delta_beta,
    int taylor_order,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    int num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    int sublattice_size,
    int num_sites
){
    if (!dir.empty()) { ensureDirectoryExists(dir); }
    energies.clear();

    // Operators for fluctuations/correlations
    std::vector<SingleSiteOperator> Sz_ops = createSzOperators(num_sites, spin_length);
    std::vector<SingleSiteOperator> Sx_ops = createSxOperators(num_sites, spin_length);
    std::vector<SingleSiteOperator> Sy_ops = createSyOperators(num_sites, spin_length);
    auto double_site_ops = createSingleOperators_pair(num_sites, spin_length);

    // Temperature checkpoints (log-spaced β for saving states)
    const int num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1.0);
    double log_max = std::log10(1000.0);
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    int max_steps = std::max(1, int(std::ceil(beta_max / delta_beta)));

    for (int sample = 0; sample < num_samples; ++sample) {
        std::vector<bool> temp_measured(num_temp_points, false);
        std::cout << "Canonical TPQ sample " << sample + 1 << " of " << num_samples << std::endl;

        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);

        // Initial random normalized state (β=0)
        unsigned int seed = static_cast<unsigned int>(time(NULL)) + sample;
        ComplexVector psi = generateTPQVector(N, seed);

        // Step 1: record β=0
        {
            auto [e0, var0] = calculateEnergyAndVariance(H, psi, N);
            double inv_temp = 0.0;
            writeTPQData(ss_file, inv_temp, e0, var0, /*norm*/1.0, /*step*/1);
            std::ofstream norm_out(norm_file, std::ios::app);
            norm_out << std::setprecision(16) << inv_temp << " " << 1.0 << " " << 1.0 << " " << 1 << std::endl;
        }

        // Main imaginary-time loop
        int step = 2;
        double beta = 0.0;
        for (int k = 1; k <= max_steps; ++k, ++step) {
            beta += delta_beta;
            if (beta > beta_max + 1e-15) { beta = beta_max; }

            // Evolve by Δβ
            imaginary_time_evolve_tpq_taylor(H, psi, N, delta_beta, taylor_order, /*normalize=*/true);

            // Measurements
            auto [e, var] = calculateEnergyAndVariance(H, psi, N);
            double inv_temp = beta;

            writeTPQData(ss_file, inv_temp, e, var, /*norm*/1.0, step);
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " << 1.0 << " " << 1.0 << " " << step << std::endl;
            }

            if ((measure_sz && (k % std::max(1, temp_interval) == 0 || k == max_steps))) {
                writeFluctuationData(flct_file, spin_corr, inv_temp, psi,
                                     num_sites, spin_length, Sx_ops, Sy_ops, Sz_ops, double_site_ops, sublattice_size, step);
            }

            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i] && std::abs(inv_temp - measure_inv_temp[i]) < 4e-3) {
                    if (compute_observables) {
                        std::string state_file = dir + "/tpq_state_" + std::to_string(sample)
                                               + "_beta=" + std::to_string(inv_temp) + ".dat";
                        save_tpq_state(psi, state_file);
                    }
                    temp_measured[i] = true;
                }
            }

            if (k % std::max(1, max_steps / 10) == 0 || k == max_steps) {
                std::cout << "  β = " << beta << " (" << k << "/" << max_steps << "), E = " << e << std::endl;
            }
        }

        // Final energy at β_max
        auto [ef, _varf] = calculateEnergyAndVariance(H, psi, N);
        energies.push_back(ef);
    }
}
