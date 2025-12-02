#include <ed/solvers/TPQ.h>
#include <ed/core/construct_ham.h>
#include <filesystem>
#include <regex>

#ifdef WITH_MPI
#include <mpi.h>
#endif

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
    uint64_t N,
    double delta_t,
    uint64_t n_max,
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
    uint64_t N,
    double delta_t,
    uint64_t krylov_dim,
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
    uint64_t N,
    double delta_t,
    double E_min,
    double E_max,
    uint64_t num_terms,
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
    uint64_t N,
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
    uint64_t N,
    double delta_t,
    uint64_t accuracy_level,
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
    uint64_t N,
    const std::string& dir,
    uint64_t sample,
    double inv_temp,
    double t_end,
    double dt,
    uint64_t krylov_dim
) {
    std::cout << "Computing dynamical susceptibility for sample " << sample 
              << ", beta = " << inv_temp << ", for " << operators_1.size() << " observables" << std::endl;
    
    // Ensure Krylov dimension doesn't exceed system size
    krylov_dim = std::min(krylov_dim, N/2);
    
    uint64_t num_steps = static_cast<int>(t_end / dt) + 1;
    
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
    uint64_t N,
    double omega_min,
    double omega_max,
    uint64_t num_points,
    double tmax,
    double dt,
    double eta,
    bool use_lorentzian,
    uint64_t n_max
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
    uint64_t N,
    const uint64_t num_steps
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
    uint64_t N,
    const uint64_t num_steps,
    double dt,
    std::vector<std::ofstream>& output_files
) {
    compute_time_correlations_incremental(U_t, operators_1, operators_2, 
                                         tpq_state, N, num_steps, dt, output_files);
}

/**
 * Compute observable dynamics for TPQ with legacy interface
 * OPTIMIZED: Process observables on-demand and stream results to disk
 */
void computeObservableDynamics_U_t(
    std::function<void(const Complex*, Complex*, int)> U_t,
    const ComplexVector& tpq_state,
    const std::vector<Operator>& observables_1,
    const std::vector<Operator>& observables_2,
    const std::vector<std::string>& observable_names,
    uint64_t N,
    const std::string& dir,
    uint64_t sample,
    double inv_temp,
    double t_end,
    double dt
) {
    // Save the current TPQ state for later analysis
    std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + "_beta=" + std::to_string(inv_temp) + ".dat";
    save_tpq_state(tpq_state, state_file);

    std::cout << "Computing dynamical susceptibility for sample " << sample 
              << ", beta = " << inv_temp << ", for " << observables_1.size() << " observables" << std::endl;
    std::cout << "  Using memory-optimized matrix-free observable computation" << std::endl;
    
    uint64_t num_steps = static_cast<int>(t_end / dt) + 1;
    
    // Create inverse time evolution operator (U_t^†) for negative time
    auto U_t_dagger = [&U_t, N](const Complex* in, Complex* out, uint64_t size) {
        // For a unitary operator U = exp(-iHt), U^† = exp(iHt)
        // We compute U^†|ψ> = (U|ψ*>)*
        ComplexVector in_conj(size);
        ComplexVector out_temp(size);
        
        for (int i = 0; i < size; i++) {
            in_conj[i] = std::conj(in[i]);
        }
        U_t(in_conj.data(), out_temp.data(), size);
        for (int i = 0; i < size; i++) {
            out[i] = std::conj(out_temp[i]);
        }
    };
    
    // ===== PROCESS EACH OBSERVABLE ON-DEMAND =====
    // This saves memory by not keeping all observables in memory simultaneously
    for (size_t op_idx = 0; op_idx < observables_1.size(); op_idx++) {
        std::cout << "  Processing observable " << (op_idx+1) << "/" << observables_1.size() 
                  << " (" << observable_names[op_idx] << ")..." << std::endl;
        
        // Open output file for streaming results
        std::string time_corr_file = dir + "/time_corr_rand" + std::to_string(sample) + "_" 
                                   + observable_names[op_idx] + "_beta=" + std::to_string(inv_temp) + ".dat";
        std::ofstream time_corr_out(time_corr_file);
        if (!time_corr_out.is_open()) {
            std::cerr << "Error: Could not open file " << time_corr_file << " for writing" << std::endl;
            continue;
        }
        time_corr_out << "# t time_correlation_real time_correlation_imag" << std::endl;
        time_corr_out << std::setprecision(16);
        
        // Temporary storage for this observable (will be sorted before writing)
        std::vector<std::tuple<double, double, double>> time_data;
        time_data.reserve(2 * num_steps - 1);
        
        // Buffers for this observable only (reused for positive and negative time)
        ComplexVector O_psi(N);
        ComplexVector O_psi_next(N);
        ComplexVector evolved_state(N);
        ComplexVector state_next(N);
        ComplexVector O_dag_state(N);
        
        // ===== INITIALIZE =====
        std::copy(tpq_state.begin(), tpq_state.end(), evolved_state.begin());
        observables_1[op_idx].apply(evolved_state.data(), O_psi.data(), N);
        observables_2[op_idx].apply(evolved_state.data(), O_dag_state.data(), N);
        
        // Calculate initial correlation C(0)
        Complex init_corr = Complex(0.0, 0.0);
        for (int i = 0; i < N; i++) {
            init_corr += std::conj(O_dag_state[i]) * O_psi[i];
        }
        time_data.push_back(std::make_tuple(0.0, init_corr.real(), init_corr.imag()));
        
        // ===== POSITIVE TIME EVOLUTION =====
        std::cout << "    Computing positive time evolution (0 to " << t_end << ")..." << std::endl;
        
        for (int step = 1; step < num_steps; step++) {
            double current_time = step * dt;
            
            // Evolve state and O|ψ>
            U_t(evolved_state.data(), state_next.data(), N);
            U_t(O_psi.data(), O_psi_next.data(), N);
            
            // Calculate O†|ψ(t)>
            observables_2[op_idx].apply(state_next.data(), O_dag_state.data(), N);
            
            // Calculate correlation
            Complex corr = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                corr += std::conj(O_dag_state[i]) * O_psi_next[i];
            }
            time_data.push_back(std::make_tuple(current_time, corr.real(), corr.imag()));
            
            // Update for next step
            std::swap(O_psi, O_psi_next);
            std::swap(evolved_state, state_next);
            
            if (step % 100 == 0) {
                std::cout << "      Positive time step " << step << " / " << num_steps << std::endl;
            }
        }
        
        // ===== NEGATIVE TIME EVOLUTION =====
        std::cout << "    Computing negative time evolution (0 to " << -t_end << ")..." << std::endl;
        
        // Re-initialize
        std::copy(tpq_state.begin(), tpq_state.end(), evolved_state.begin());
        observables_1[op_idx].apply(evolved_state.data(), O_psi.data(), N);
    
        
        for (int step = 1; step < num_steps; step++) {
            double current_time = -step * dt;
            
            // Evolve backward
            U_t_dagger(evolved_state.data(), state_next.data(), N);
            U_t_dagger(O_psi.data(), O_psi_next.data(), N);
            
            // Calculate O†|ψ(-t)>
            observables_2[op_idx].apply(state_next.data(), O_dag_state.data(), N);
            
            // Calculate correlation
            Complex corr = Complex(0.0, 0.0);
            for (int i = 0; i < N; i++) {
                corr += std::conj(O_dag_state[i]) * O_psi_next[i];
            }
            time_data.push_back(std::make_tuple(current_time, corr.real(), corr.imag()));
            
            // Update for next step
            std::swap(O_psi, O_psi_next);
            std::swap(evolved_state, state_next);
            
            if (step % 100 == 0) {
                std::cout << "      Negative time step " << step << " / " << num_steps << std::endl;
            }
        }
        
        // ===== STREAM SORTED OUTPUT =====
        // Sort by time and write to file
        std::sort(time_data.begin(), time_data.end(), 
                  [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
        
        for (const auto& data_point : time_data) {
            time_corr_out << std::get<0>(data_point) << " " 
                         << std::get<1>(data_point) << " " 
                         << std::get<2>(data_point) << std::endl;
        }
        
        time_corr_out.close();
        std::cout << "    Time correlation saved to " << time_corr_file << std::endl;
        std::cout << "    Time range: [" << std::get<0>(time_data.front()) << ", " 
                  << std::get<0>(time_data.back()) << "]" << std::endl;
        
        // time_data is freed here before next observable
    }
    
    std::cout << "  All observables processed successfully!" << std::endl;
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
ComplexVector generateTPQVector(int N,  uint64_t seed) {
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
    uint64_t N
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
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit to avoid overflow)
    
    ComplexVector Sz_exps(sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Sz2_exps(sublattice_size+1, Complex(0.0, 0.0));
    
    // OPTIMIZED: Pre-allocate reusable buffers outside loop
    std::vector<Complex> Sz_psi(N);
    std::vector<Complex> Sz2_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        uint64_t i = site % sublattice_size;

        // Apply operator into pre-allocated buffer
        Sz_ops[i].apply(tpq_state.data(), Sz_psi.data(), N);
        
        // Calculate expectation value using BLAS
        Complex Sz_exp;
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sz_psi.data(), 1, &Sz_exp);
        Sz_exps[i] += Sz_exp;

        // Apply operator again for Sz^2
        Sz_ops[i].apply(Sz_psi.data(), Sz2_psi.data(), N);
        
        // Calculate Sz^2 expectation using BLAS
        Complex Sz2_exp;
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sz2_psi.data(), 1, &Sz2_exp);
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
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Spm_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)

    Complex Spm_exp(0.0, 0.0);
    
    // OPTIMIZED: Pre-allocate reusable buffer
    std::vector<Complex> Spm_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        uint64_t i = site % sublattice_size;

        // Apply operator into pre-allocated buffer
        Spm_ops[i].apply(tpq_state.data(), Spm_psi.data(), N);

        // Calculate <Spm_psi|Spm_psi> using BLAS
        Complex site_exp;
        cblas_zdotc_sub(N, Spm_psi.data(), 1, Spm_psi.data(), 1, &site_exp);
        Spm_exp += site_exp;
    }

    return Spm_exp / double(num_sites);
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
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> double_site_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));

    // Reference operators (avoid copy)
    const std::vector<DoubleSiteOperator>& Szz_ops = double_site_ops.first;
    const std::vector<DoubleSiteOperator>& Spm_ops = double_site_ops.second;
    
    // OPTIMIZED: Pre-allocate reusable buffers outside nested loop
    std::vector<Complex> Szz_psi(N);
    std::vector<Complex> Spm_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            uint64_t n1 = site % sublattice_size;
            uint64_t n2 = site2 % sublattice_size;

            // Apply operators into pre-allocated buffers
            Szz_ops[site*num_sites+site2].apply(tpq_state.data(), Szz_psi.data(), N);
            Spm_ops[site*num_sites+site2].apply(tpq_state.data(), Spm_psi.data(), N);

            // Calculate expectation values using BLAS
            Complex Szz_exp, Spm_exp;
            cblas_zdotc_sub(N, tpq_state.data(), 1, Szz_psi.data(), 1, &Szz_exp);
            cblas_zdotc_sub(N, tpq_state.data(), 1, Spm_psi.data(), 1, &Spm_exp);
            
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
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spp_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    
    // Reference operators (avoid copy)
    const std::vector<SingleSiteOperator>& Szz_ops = double_site_ops.first;
    const std::vector<SingleSiteOperator>& Spm_ops = double_site_ops.second;
    
    // OPTIMIZED: Pre-allocate reusable buffers outside nested loop
    std::vector<Complex> Szz_psi(N);
    std::vector<Complex> Szz_psi2(N);
    std::vector<Complex> Spm_psi(N);
    std::vector<Complex> Spm_psi2(N);
    std::vector<Complex> Spp_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            uint64_t n1 = site % sublattice_size;
            uint64_t n2 = site2 % sublattice_size;

            // Apply operators into pre-allocated buffers
            Szz_ops[site].apply(tpq_state.data(), Szz_psi.data(), N);
            Szz_ops[site2].apply(tpq_state.data(), Szz_psi2.data(), N);
            Spm_ops[site].apply(tpq_state.data(), Spm_psi.data(), N);
            Spm_ops[site2].apply(tpq_state.data(), Spm_psi2.data(), N);
            Spm_ops[site2].apply(Spm_psi.data(), Spp_psi.data(), N);

            // Calculate expectation values using BLAS
            Complex Szz_exp, Spm_exp, Spp_exp, Spz_exp;
            cblas_zdotc_sub(N, Szz_psi.data(), 1, Szz_psi2.data(), 1, &Szz_exp);
            cblas_zdotc_sub(N, Spm_psi.data(), 1, Spm_psi2.data(), 1, &Spm_exp);
            cblas_zdotc_sub(N, tpq_state.data(), 1, Spp_psi.data(), 1, &Spp_exp);
            cblas_zdotc_sub(N, Spm_psi.data(), 1, Szz_psi2.data(), 1, &Spz_exp);
            
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
                 double variance, double norm, uint64_t step) {
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
bool readTPQData(const std::string& filename, uint64_t step, double& energy, 
                double& temp, double& specificHeat) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    double inv_temp, e, var, n, doublon;
    uint64_t s;
    
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
 * @param tpq_state TPQ state vector to save (in fixed-Sz or full basis)
 * @param filename Name of the file to save to
 * @param fixed_sz_op Optional FixedSzOperator - if provided, transforms to full basis before saving
 * @return True if successful
 */
bool save_tpq_state(const ComplexVector& tpq_state, const std::string& filename, 
                    FixedSzOperator* fixed_sz_op) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return false;
    }
    
    // Transform to full basis if using fixed-Sz
    if (fixed_sz_op != nullptr) {
        std::vector<Complex> full_state = fixed_sz_op->embedToFull(tpq_state);
        size_t full_size = full_state.size();
        out.write(reinterpret_cast<const char*>(&full_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(full_state.data()), full_size * sizeof(Complex));
        std::cout << "  [Fixed-Sz] Transformed state from dim " << tpq_state.size() 
                  << " to full space dim " << full_size << " before saving" << std::endl;
    } else {
        size_t size = tpq_state.size();
        out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(tpq_state.data()), size * sizeof(Complex));
    }
    
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
bool load_raw_data(ComplexVector& tpq_state, const std::string& filename, uint64_t N) {
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
    uint64_t num_sites,
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
        std::vector<Complex> Sp_psi(N);
        std::vector<Complex> Sm_psi(N);
        std::vector<Complex> Sz_psi(N);
        Sp_ops[site].apply(tpq_state.data(), Sp_psi.data(), N);
        Sm_ops[site].apply(tpq_state.data(), Sm_psi.data(), N);
        Sz_ops[site].apply(tpq_state.data(), Sz_psi.data(), N);
        
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
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sx_ops,
    const std::vector<SingleSiteOperator>& Sy_ops,
    const std::vector<SingleSiteOperator>& Sz_ops,
    const std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>>& double_site_ops,
    uint64_t sublattice_size,
    uint64_t step
) {
    // Compute and write Sz on-demand (memory is freed after computation)
    auto [Sz, Sz2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sz_ops, sublattice_size);
    
    std::ofstream flct_out(flct_file, std::ios::app);
    flct_out << std::setprecision(16) << inv_temp 
             << " " << Sz[sublattice_size].real() << " " << Sz[sublattice_size].imag() 
             << " " << Sz2[sublattice_size].real() << " " << Sz2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out << " " << Sz[i].real() << " " << Sz[i].imag() 
                 << " " << Sz2[i].real() << " " << Sz2[i].imag();
    }

    auto Spm2exp = calculateSpm_onsite(tpq_state, num_sites, spin_length, double_site_ops.second, sublattice_size);
    flct_out << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();
    flct_out << " " << step << std::endl;
    flct_out.close();
    
    // Compute and write Sx on-demand (Sz memory is freed before this)
    auto [Sx, Sx2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sx_ops, sublattice_size);
    
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
    flct_out_x.close();

    // Compute and write Sy on-demand (Sx memory is freed before this)
    auto [Sy, Sy2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sy_ops, sublattice_size);
    
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
    flct_out_y.close();

    // Compute and stream correlation data one type at a time
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
    uint64_t sample,
    double target_beta,
    uint64_t N
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
    uint64_t best_step = 0;
    double min_diff = std::numeric_limits<double>::max();
    
    // Find the step with the closest inverse temperature
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double beta, energy, variance, norm, doublon;
        uint64_t step;
        
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
 * Find the lowest energy state from saved TPQ state files
 * Searches through tpq_state_*_beta=*_step=*.dat (or legacy tpq_state_*_beta=*.dat) files 
 * to find the highest beta (lowest energy)
 */
bool find_lowest_energy_tpq_state(
    const std::string& tpq_dir,
    uint64_t N,
    uint64_t& out_sample,
    double& out_beta,
    uint64_t& out_step
) {
    namespace fs = std::filesystem;
    
    double max_beta = -1.0;
    bool found = false;
    
    std::cout << "Searching for lowest energy state (highest beta) in " << tpq_dir << std::endl;
    
    // Search through directory for tpq_state files
    if (!fs::exists(tpq_dir) || !fs::is_directory(tpq_dir)) {
        std::cerr << "Error: Directory " << tpq_dir << " does not exist" << std::endl;
        return false;
    }
    
    // Regex to match tpq_state_i_beta=*_step=*.dat files (new format with step)
    std::regex state_pattern_new("tpq_state_([0-9]+)_beta=([0-9.]+)_step=([0-9]+)\\.dat");
    // Also support legacy pattern: tpq_state_i_beta=*.dat
    std::regex state_pattern_legacy("tpq_state_([0-9]+)_beta=([0-9.]+)\\.dat");
    
    for (const auto& entry : fs::directory_iterator(tpq_dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        std::smatch matches;
        
        uint64_t sample = 0;
        double beta = 0.0;
        uint64_t step = 0;
        bool matched = false;
        
        // Try new format first
        if (std::regex_match(filename, matches, state_pattern_new)) {
            if (matches.size() == 4) {
                try {
                    sample = std::stoull(matches[1].str());
                    beta = std::stod(matches[2].str());
                    step = std::stoull(matches[3].str());
                    matched = true;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse filename: " << filename << std::endl;
                    continue;
                }
            }
        }
        // Fall back to legacy format
        else if (std::regex_match(filename, matches, state_pattern_legacy)) {
            if (matches.size() == 3) {
                try {
                    sample = std::stoull(matches[1].str());
                    beta = std::stod(matches[2].str());
                    step = 0; // Will need to look up from SS_rand
                    matched = true;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse filename: " << filename << std::endl;
                    continue;
                }
            }
        }
        
        if (matched) {
            // Higher beta = lower energy, so we want the maximum beta
            if (beta > max_beta) {
                max_beta = beta;
                out_sample = sample;
                out_beta = beta;
                out_step = step;
                found = true;
            }
        }
    }
    
    if (found) {
        // If step was not in filename (legacy format), look it up from SS_rand file
        if (out_step == 0) {
            std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(out_sample) + ".dat";
            std::ifstream file(ss_file);
            
            if (file.is_open()) {
                std::string line;
                std::getline(file, line); // Skip header
                
                // Find the step corresponding to this beta
                double min_diff = std::numeric_limits<double>::max();
                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    double beta, energy, variance, norm, doublon;
                    uint64_t step;
                    
                    if (!(iss >> beta >> energy >> variance >> norm >> doublon >> step)) {
                        continue;
                    }
                    
                    double diff = std::abs(beta - out_beta);
                    if (diff < min_diff) {
                        min_diff = diff;
                        out_step = step;
                    }
                }
                file.close();
            } else {
                std::cout << "Warning: Could not find SS_rand file to determine step number" << std::endl;
            }
        }
        
        std::cout << "Found lowest energy state (highest beta):" << std::endl;
        std::cout << "  Sample: " << out_sample << std::endl;
        std::cout << "  Beta: " << out_beta << std::endl;
        std::cout << "  Step: " << out_step << std::endl;
    } else {
        std::cerr << "Error: No valid TPQ state files found in " << tpq_dir << std::endl;
        std::cerr << "  Looking for files matching patterns: tpq_state_*_beta=*_step=*.dat or tpq_state_*_beta=*.dat" << std::endl;
    }
    
    return found;
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
    uint64_t sample,
    uint64_t sublattice_size
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
    uint64_t N,
    uint64_t tpq_sample,
    uint64_t tpq_step,
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
    uint64_t n_omega = static_cast<int>((omega_max - omega_min) / omega_step) + 1;
    
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
 * @param N Dimension of the Hilbert space (fixed-Sz dimension if using fixed-Sz)
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param compute_spectrum Whether to compute spectrum
 * @param fixed_sz_op Optional FixedSzOperator - if provided, transforms states to full basis before saving
 */
void microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N, 
    uint64_t max_iter,
    uint64_t num_samples,
    uint64_t temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_spectrum,
    double LargeValue,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    uint64_t num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    uint64_t sublattice_size,
    uint64_t num_sites,
    FixedSzOperator* fixed_sz_op,
    bool continue_quenching,
    uint64_t continue_sample,
    double continue_beta
) {
    #ifdef WITH_MPI
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Distribute samples across ranks using balanced assignment
    uint64_t samples_per_rank = num_samples / size;
    uint64_t remainder = num_samples % size;
    
    // Calculate start and end sample for this rank
    // Ranks with index < remainder get one extra sample
    uint64_t start_sample = rank * samples_per_rank + std::min((uint64_t)rank, remainder);
    uint64_t end_sample = start_sample + samples_per_rank + (rank < remainder ? 1 : 0);
    uint64_t local_num_samples = end_sample - start_sample;
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "MPI-Parallel TPQ Calculation\n";
        std::cout << "==========================================\n";
        std::cout << "Total MPI ranks: " << size << "\n";
        std::cout << "Total samples: " << num_samples << "\n";
        std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
        std::cout << "==========================================\n\n";
    }
    
    std::cout << "Rank " << rank << " processing samples [" 
              << start_sample << ", " << end_sample << ")\n";
    
    // Synchronize before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    // Serial execution: process all samples on single rank
    uint64_t start_sample = 0;
    uint64_t end_sample = num_samples;
    uint64_t local_num_samples = num_samples;
    #endif
    
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    double D_S = std::log2(N);
    eigenvalues.clear();
    eigenvalues.reserve(local_num_samples);

    // NOTE: Operators are now created INSIDE the sample loop on-demand
    // to avoid keeping all operators in memory simultaneously
    std::cout << "Begin TPQ calculation with dimension " << N << std::endl;
    if (measure_sz) {
        std::cout << "Observable measurements enabled (operators created on-demand per sample)" << std::endl;
    }
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



    const uint64_t num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1);   // Start from β = 1
    double log_max = std::log10(1000); // End at β = 1000
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    
    // Handle continue-quenching mode
    bool is_continuing = false;
    uint64_t resume_sample = 0;
    uint64_t original_sample = 0;  // Track original sample for loading
    double resume_beta = 0.0;
    uint64_t resume_step = 0;
    
    if (continue_quenching) {
        std::cout << "\n==========================================\n";
        std::cout << "CONTINUE QUENCHING MODE ENABLED\n";
        std::cout << "==========================================\n";
        
        if (continue_sample == 0) {
            // Auto-detect lowest energy state (highest beta) from any sample
            std::cout << "Auto-detecting lowest energy state (highest beta)..." << std::endl;
            if (find_lowest_energy_tpq_state(dir, N, original_sample, resume_beta, resume_step)) {
                is_continuing = true;
                resume_sample = 0;  // Will write to sample 0
            } else {
                std::cout << "Warning: Could not find saved state to continue from. Falling back to normal TPQ (starting fresh)." << std::endl;
                is_continuing = false;
            }
        } else {
            // Use specified sample
            original_sample = continue_sample;
            resume_sample = 0;  // Will write to sample 0
            std::cout << "Continuing from sample " << continue_sample << std::endl;
            
            // Try to find the state file directly
            if (find_lowest_energy_tpq_state(dir, N, original_sample, resume_beta, resume_step)) {
                // Verify it's the requested sample
                if (original_sample == continue_sample) {
                    is_continuing = true;
                } else {
                    std::cout << "Warning: Found sample " << original_sample 
                              << " but requested " << continue_sample << std::endl;
                    is_continuing = false;
                }
            } else {
                std::cout << "Warning: Could not find state file for sample " << continue_sample 
                          << ". Falling back to normal TPQ (starting fresh)." << std::endl;
                is_continuing = false;
            }
        }
        
        if (is_continuing) {
            std::cout << "Resuming from:" << std::endl;
            std::cout << "  Original sample: " << original_sample << std::endl;
            std::cout << "  Continuing as sample: 0 (output to SS_rand0.dat)" << std::endl;
            std::cout << "  Beta: " << resume_beta << std::endl;
            std::cout << "  Step: " << resume_step << std::endl;
            std::cout << "==========================================\n" << std::endl;
        }
    }
    
    // Modified loop: only process samples assigned to this rank
    for (uint64_t sample = start_sample; sample < end_sample; sample++) {
        #ifdef WITH_MPI
        std::cout << "[Rank " << rank << "] TPQ sample " << sample 
                  << " of " << num_samples 
                  << " (local: " << (sample-start_sample+1) 
                  << " of " << local_num_samples << ")" << std::endl;
        #else
        std::cout << "TPQ sample " << (sample+1) << " of " << num_samples << std::endl;
        #endif
        
        std::vector<bool> temp_measured(num_temp_points, false);
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);
        
        // Variables that will be initialized differently for continue mode
        ComplexVector v0;
        uint64_t step;
        double inv_temp;
        double energy1, variance1;
        double first_norm, current_norm;
        
        // Temp buffer for Hamiltonian applications (reused throughout)
        ComplexVector temp(N);
        Complex minus_one(-1.0, 0.0);
        
        // Check if we should continue from saved state (only for first sample = 0)
        if (is_continuing && sample == 0) {
            std::cout << "Loading saved state to continue quenching..." << std::endl;
            
            // Construct state file path - try new format first, then fall back to legacy
            std::string state_file_new = dir + "/tpq_state_" + std::to_string(original_sample) 
                                       + "_beta=" + std::to_string(resume_beta) 
                                       + "_step=" + std::to_string(resume_step) + ".dat";
            std::string state_file_legacy = dir + "/tpq_state_" + std::to_string(original_sample) 
                                          + "_beta=" + std::to_string(resume_beta) + ".dat";
            
            v0.resize(N);
            bool loaded = false;
            
            // Try new format first
            if (load_tpq_state(v0, state_file_new)) {
                std::cout << "Loaded state from: " << state_file_new << std::endl;
                loaded = true;
            } 
            // Fall back to legacy format
            else if (load_tpq_state(v0, state_file_legacy)) {
                std::cout << "Loaded state from: " << state_file_legacy << std::endl;
                loaded = true;
            }
            
            if (!loaded) {
                std::cerr << "Error: Could not load TPQ state from either format" << std::endl;
                std::cerr << "  Tried: " << state_file_new << std::endl;
                std::cerr << "  Tried: " << state_file_legacy << std::endl;
                std::cerr << "Starting fresh for this sample." << std::endl;
                goto fresh_start;
            }
            
            // Set starting point from loaded state
            step = resume_step;
            inv_temp = resume_beta;
            
            // Calculate energy and variance of loaded state
            auto [loaded_energy, loaded_variance] = calculateEnergyAndVariance(H, v0, N);
            energy1 = loaded_energy;
            variance1 = loaded_variance;
            
            first_norm = cblas_dznrm2(N, v0.data(), 1);
            current_norm = first_norm;
            
            std::cout << "Loaded state properties:" << std::endl;
            std::cout << "  Energy: " << energy1 << std::endl;
            std::cout << "  Variance: " << variance1 << std::endl;
            std::cout << "  Beta: " << inv_temp << std::endl;
            std::cout << "  Will run " << max_iter << " additional iterations" << std::endl;
            std::cout << "  Target final step: " << (step + max_iter) << std::endl;
            std::cout << "Continuing from step " << step << "..." << std::endl;
        } else {
            fresh_start:
            // Generate initial random state (already normalized)
            uint64_t seed = static_cast< int>(time(NULL)) + sample;
            v0 = generateTPQVector(N, seed);
            
            // Apply initial transformation: v0 = (L*D_S - H)|v0⟩
            H(v0.data(), temp.data(), N);
            Complex scale_factor_large(LargeValue * D_S, 0.0);
            cblas_zscal(N, &scale_factor_large, v0.data(), 1);  // v0 *= L*D_S
            cblas_zaxpy(N, &minus_one, temp.data(), 1, v0.data(), 1);  // v0 = v0 - temp
            
            // Write initial state (infinite temperature)
            step = 1;
            
            // Calculate energy and variance for step 1
            auto [e1, v1] = calculateEnergyAndVariance(H, v0, N);
            energy1 = e1;
            variance1 = v1;
            inv_temp = (2.0) / (LargeValue* D_S - energy1);

            first_norm = cblas_dznrm2(N, v0.data(), 1);
            Complex scale_factor = Complex(1.0/first_norm, 0.0);

            cblas_zscal(N, &scale_factor, v0.data(), 1);

            current_norm = first_norm;
            
            writeTPQData(ss_file, inv_temp, energy1, variance1, current_norm, step);
            
            {
                std::ofstream norm_out(norm_file, std::ios::app);
                norm_out << std::setprecision(16) << inv_temp << " " 
                         << current_norm << " " << first_norm << " " << step << std::endl;
            }
            
            step = 2; // Start main loop from step 2
        }
        
        // Determine final step: if continuing, run for additional max_iter iterations
        uint64_t final_step = is_continuing && sample == 0 ? (step - 1 + max_iter) : max_iter;
        
        // Main TPQ loop - using in-place operations with single temp buffer
        for (; step <= final_step; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == final_step) {
                std::cout << "  Step " << step << " of " << final_step << std::endl;
            }
            
            // In-place evolution: v0 = (L*D_S - H)|v0⟩
            // First compute temp = H|v0⟩
            H(v0.data(), temp.data(), N);
            
            // Then v0 = L*D_S*v0 - temp (in-place)
            Complex scale_ld(LargeValue * D_S, 0.0);
            cblas_zscal(N, &scale_ld, v0.data(), 1);  // v0 *= L*D_S
            cblas_zaxpy(N, &minus_one, temp.data(), 1, v0.data(), 1);  // v0 = v0 - temp

            current_norm = cblas_dznrm2(N, v0.data(), 1);
            Complex scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v0.data(), 1);
            
            // Check if we should measure observables at target temperatures
            // We need to check this at EVERY step to avoid missing temperature points
            bool should_measure_observables = false;
            int target_temp_idx = -1;
            
            // First, do a quick check using estimated temperature
            // Estimate current inverse temperature (using last known energy)
            double estimated_inv_temp = (2.0 * step) / (LargeValue * D_S - energy1);
            
            // Check if we're potentially near any target temperature
            // Use a wider search window (5% instead of 1%) for the initial check
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i]) {
                    double search_tolerance = 0.05 * measure_inv_temp[i];  // 5% search window
                    if (std::abs(estimated_inv_temp - measure_inv_temp[i]) < search_tolerance) {
                        // We're potentially close - need to compute actual energy to be sure
                        should_measure_observables = true;
                        target_temp_idx = i;
                        break;
                    }
                }
            }
            
            // Determine if we should do measurements this step
            bool do_regular_measurement = (step % temp_interval == 0 || step == final_step);
            bool do_measurement = do_regular_measurement || should_measure_observables;
            
            // OPTIMIZED: Calculate energy and variance only when needed
            // This significantly reduces computational cost for large systems
            if (do_measurement) {
                // Calculate energy and variance
                auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v0, N);
                // Update inverse temperature with accurate energy
                inv_temp = (2.0*step) / (LargeValue * D_S - energy_step);
                
                // Update energy for next iteration's estimate
                energy1 = energy_step;
                
                // Now check with accurate temperature if we're really at the target
                bool actually_at_target = false;
                if (should_measure_observables && target_temp_idx >= 0) {
                    double precise_tolerance = 0.01 * measure_inv_temp[target_temp_idx];  // 1% precise tolerance
                    if (std::abs(inv_temp - measure_inv_temp[target_temp_idx]) < precise_tolerance) {
                        actually_at_target = true;
                    }
                }
                
                // Write data (always write when we compute energy)
                writeTPQData(ss_file, inv_temp, energy_step, variance_step, current_norm, step);
                
                {
                    std::ofstream norm_out(norm_file, std::ios::app);
                    norm_out << std::setprecision(16) << inv_temp << " " 
                             << current_norm << " " << first_norm << " " << step << std::endl;
                }
                
                // Report detailed progress
                if (step % (temp_interval * 10) == 0 || step == final_step) {
                    std::cout << "  Step " << step << ": E = " << energy_step 
                              << ", var = " << variance_step 
                              << ", β = " << inv_temp << std::endl;
                }
                
                // Write fluctuation data only at regular intervals
                if (measure_sz && do_regular_measurement){
                    // Create operators on-demand only when needed (they are freed after use)
                    std::cout << "  Creating operators on-demand for fluctuation measurement..." << std::endl;
                    auto Sx_ops = createSxOperators(num_sites, spin_length);
                    auto Sy_ops = createSyOperators(num_sites, spin_length);
                    auto Sz_ops = createSzOperators(num_sites, spin_length);
                    auto double_site_ops = createSingleOperators_pair(num_sites, spin_length);
                    
                    writeFluctuationData(flct_file, spin_corr, inv_temp, v0, num_sites, spin_length, Sx_ops, Sy_ops, Sz_ops, double_site_ops, sublattice_size, step);
                    // Operators are automatically freed here when they go out of scope
                }
                
                // Save observables at target temperatures (with accurate inv_temp)
                if (actually_at_target) {
                    std::cout << "  *** Saving TPQ state at β = " << inv_temp 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    if (compute_observables) {
                        std::string state_file = dir + "/tpq_state_" + std::to_string(sample) + 
                                               "_beta=" + std::to_string(inv_temp) + 
                                               "_step=" + std::to_string(step) + ".dat";
                        save_tpq_state(v0, state_file, fixed_sz_op);
                    }
                    temp_measured[target_temp_idx] = true;
                }
            }
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy1);
    }
    
    #ifdef WITH_MPI
    // Gather all eigenvalues from all ranks to rank 0
    std::vector<double> all_eigenvalues;
    if (rank == 0) {
        all_eigenvalues.resize(num_samples);
    }
    
    // Calculate receive counts and displacements for MPI_Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; r++) {
        uint64_t r_samples_per_rank = num_samples / size;
        uint64_t r_remainder = num_samples % size;
        uint64_t r_start = r * r_samples_per_rank + std::min((uint64_t)r, r_remainder);
        uint64_t r_count = r_samples_per_rank + (r < r_remainder ? 1 : 0);
        
        recvcounts[r] = static_cast<int>(r_count);
        displs[r] = static_cast<int>(r_start);
    }
    
    // Gather eigenvalues from all ranks
    MPI_Gatherv(eigenvalues.data(), static_cast<int>(eigenvalues.size()), MPI_DOUBLE,
                all_eigenvalues.data(), recvcounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Update eigenvalues on rank 0 with complete set
    if (rank == 0) {
        eigenvalues = std::move(all_eigenvalues);
        std::cout << "\n==========================================\n";
        std::cout << "MPI TPQ Computation Complete\n";
        std::cout << "Collected " << eigenvalues.size() << " sample energies\n";
        std::cout << "==========================================\n";
    } else {
        // Clear eigenvalues on non-root ranks to save memory
        eigenvalues.clear();
    }
    
    // Final barrier before returning
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}

// Canonical TPQ using imaginary-time propagation e^{-βH} |r>
inline void imaginary_time_evolve_tpq_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_beta,
    uint64_t n_max,
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
    uint64_t N,
    double beta_max,
    uint64_t num_samples,
    uint64_t temp_interval,
    std::vector<double>& energies,
    std::string dir,
    double delta_beta,
    uint64_t taylor_order,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    uint64_t num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    uint64_t sublattice_size,
    uint64_t num_sites,
    FixedSzOperator* fixed_sz_op
){
    #ifdef WITH_MPI
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Distribute samples across ranks using balanced assignment
    uint64_t samples_per_rank = num_samples / size;
    uint64_t remainder = num_samples % size;
    
    // Calculate start and end sample for this rank
    uint64_t start_sample = rank * samples_per_rank + std::min((uint64_t)rank, remainder);
    uint64_t end_sample = start_sample + samples_per_rank + (rank < remainder ? 1 : 0);
    uint64_t local_num_samples = end_sample - start_sample;
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "MPI-Parallel Canonical TPQ Calculation\n";
        std::cout << "==========================================\n";
        std::cout << "Total MPI ranks: " << size << "\n";
        std::cout << "Total samples: " << num_samples << "\n";
        std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
        std::cout << "==========================================\n\n";
    }
    
    std::cout << "Rank " << rank << " processing samples [" 
              << start_sample << ", " << end_sample << ")\n";
    
    // Synchronize before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    #else
    // Serial execution: process all samples
    uint64_t start_sample = 0;
    uint64_t end_sample = num_samples;
    uint64_t local_num_samples = num_samples;
    #endif
    
    if (!dir.empty()) { ensureDirectoryExists(dir); }
    energies.clear();
    energies.reserve(local_num_samples);

    // NOTE: Operators are now created INSIDE the sample loop on-demand
    // to avoid keeping all operators in memory simultaneously
    std::cout << "Begin Canonical TPQ calculation with dimension " << N << std::endl;
    if (measure_sz) {
        std::cout << "Observable measurements enabled (operators created on-demand per sample)" << std::endl;
    }

    // Temperature checkpoints (log-spaced β for saving states)
    const uint64_t num_temp_points = 20;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(1.0);
    double log_max = std::log10(1000.0);
    for (int i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    uint64_t max_steps = std::max(1, int(std::ceil(beta_max / delta_beta)));

    // Modified loop: only process samples assigned to this rank
    for (uint64_t sample = start_sample; sample < end_sample; ++sample) {
        #ifdef WITH_MPI
        std::cout << "[Rank " << rank << "] Canonical TPQ sample " << sample 
                  << " of " << num_samples 
                  << " (local: " << (sample-start_sample+1) 
                  << " of " << local_num_samples << ")" << std::endl;
        #else
        std::cout << "Canonical TPQ sample " << (sample + 1) << " of " << num_samples << std::endl;
        #endif
        
        std::vector<bool> temp_measured(num_temp_points, false);
        
        // Setup filenames for this sample
        auto [ss_file, norm_file, flct_file, spin_corr] = initializeTPQFiles(dir, sample, sublattice_size);

        // Initial random normalized state (β=0)
         uint64_t seed = static_cast< int>(time(NULL)) + sample;
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
        uint64_t step = 2;
        double beta = 0.0;
        for (int k = 1; k <= max_steps; ++k, ++step) {
            beta += delta_beta;
            if (beta > beta_max + 1e-15) { beta = beta_max; }

            // Evolve by Δβ
            imaginary_time_evolve_tpq_taylor(H, psi, N, delta_beta, taylor_order, /*normalize=*/true);

            // Check if we should measure observables at target temperatures
            // In canonical TPQ, beta is known exactly, so we can check directly
            bool should_measure_observables = false;
            int target_temp_idx = -1;
            
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i]) {
                    // Use relative tolerance
                    double tolerance = 0.01 * measure_inv_temp[i];  // 1% tolerance
                    if (std::abs(beta - measure_inv_temp[i]) < tolerance) {
                        should_measure_observables = true;
                        target_temp_idx = i;
                        break;
                    }
                }
            }

            // Determine if we should do measurements this step
            bool do_regular_measurement = (k % temp_interval == 0 || k == max_steps);
            bool do_measurement = do_regular_measurement || should_measure_observables;

            // OPTIMIZED: Measurements only when needed
            // This significantly reduces computational cost for large systems
            if (do_measurement) {
                auto [e, var] = calculateEnergyAndVariance(H, psi, N);
                double inv_temp = beta;

                writeTPQData(ss_file, inv_temp, e, var, /*norm*/1.0, step);
                {
                    std::ofstream norm_out(norm_file, std::ios::app);
                    norm_out << std::setprecision(16) << inv_temp << " " << 1.0 << " " << 1.0 << " " << step << std::endl;
                }

                // Write fluctuation data only at regular intervals
                if (measure_sz && do_regular_measurement){
                    // Create operators on-demand only when needed (they are freed after use)
                    std::cout << "  Creating operators on-demand for fluctuation measurement..." << std::endl;
                    auto Sx_ops = createSxOperators(num_sites, spin_length);
                    auto Sy_ops = createSyOperators(num_sites, spin_length);
                    auto Sz_ops = createSzOperators(num_sites, spin_length);
                    auto double_site_ops = createSingleOperators_pair(num_sites, spin_length);
                    
                    writeFluctuationData(flct_file, spin_corr, inv_temp, psi,
                                         num_sites, spin_length, Sx_ops, Sy_ops, Sz_ops, double_site_ops, sublattice_size, step);
                    // Operators are automatically freed here when they go out of scope
                }
                
                // Save state at target temperature checkpoints
                if (should_measure_observables && target_temp_idx >= 0) {
                    std::cout << "  *** Saving TPQ state at β = " << inv_temp 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    if (compute_observables) {
                        std::string state_file = dir + "/tpq_state_" + std::to_string(sample)
                                               + "_beta=" + std::to_string(inv_temp) 
                                               + "_step=" + std::to_string(step) + ".dat";
                        save_tpq_state(psi, state_file, fixed_sz_op);
                    }
                    temp_measured[target_temp_idx] = true;
                }
                
                if (k % std::max(static_cast<uint64_t>(1), max_steps / 10) == 0 || k == max_steps) {
                    std::cout << "  β = " << beta << " (" << k << "/" << max_steps << "), E = " << e << std::endl;
                }
            }
        }

        // Final energy at β_max
        auto [ef, _varf] = calculateEnergyAndVariance(H, psi, N);
        energies.push_back(ef);
    }
    
    #ifdef WITH_MPI
    // Gather all energies from all ranks to rank 0
    std::vector<double> all_energies;
    if (rank == 0) {
        all_energies.resize(num_samples);
    }
    
    // Calculate receive counts and displacements for MPI_Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; r++) {
        uint64_t r_samples_per_rank = num_samples / size;
        uint64_t r_remainder = num_samples % size;
        uint64_t r_start = r * r_samples_per_rank + std::min((uint64_t)r, r_remainder);
        uint64_t r_count = r_samples_per_rank + (r < r_remainder ? 1 : 0);
        
        recvcounts[r] = static_cast<int>(r_count);
        displs[r] = static_cast<int>(r_start);
    }
    
    // Gather energies from all ranks
    MPI_Gatherv(energies.data(), static_cast<int>(energies.size()), MPI_DOUBLE,
                all_energies.data(), recvcounts.data(), displs.data(), 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Update energies on rank 0 with complete set
    if (rank == 0) {
        energies = std::move(all_energies);
        std::cout << "\n==========================================\n";
        std::cout << "MPI Canonical TPQ Computation Complete\n";
        std::cout << "Collected " << energies.size() << " sample energies\n";
        std::cout << "==========================================\n";
    } else {
        // Clear energies on non-root ranks to save memory
        energies.clear();
    }
    
    // Final barrier before returning
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}
