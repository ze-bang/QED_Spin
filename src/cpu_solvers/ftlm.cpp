// ftlm.cpp - Finite Temperature Lanczos Method implementation

#include "ftlm.h"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cstring>

/**
 * @brief Build Krylov subspace and extract tridiagonal matrix coefficients
 */
int build_lanczos_tridiagonal(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v0,
    int N,
    int max_iter,
    double tol,
    bool full_reorth,
    int reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta
) {
    alpha.clear();
    beta.clear();
    beta.push_back(0.0); // β_0 is not used
    
    // Working vectors
    ComplexVector v_current = v0;
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Normalize initial vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    // Store basis vectors for reorthogonalization (if needed)
    std::vector<ComplexVector> basis_vectors;
    if (full_reorth || reorth_freq > 0) {
        basis_vectors.push_back(v_current);
    }
    
    max_iter = std::min(N, max_iter);
    
    // Lanczos iteration
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Reorthogonalization
        if (full_reorth) {
            // Full reorthogonalization against all previous vectors
            for (int k = 0; k <= j; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
            }
        } else if (reorth_freq > 0 && (j + 1) % reorth_freq == 0) {
            // Periodic reorthogonalization
            for (int k = 0; k <= j; k++) {
                Complex overlap;
                cblas_zdotc_sub(N, basis_vectors[k].data(), 1, w.data(), 1, &overlap);
                if (std::abs(overlap) > tol) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, basis_vectors[k].data(), 1, w.data(), 1);
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown or convergence
        if (norm < tol) {
            return j + 1;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store for reorthogonalization
        if (full_reorth || reorth_freq > 0) {
            basis_vectors.push_back(v_next);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
    }
    
    return max_iter;
}

/**
 * @brief Compute thermodynamic observables from a single FTLM sample
 */
ThermodynamicData compute_ftlm_thermodynamics(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures
) {
    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    
    int n_temps = temperatures.size();
    int n_states = ritz_values.size();
    
    thermo.energy.resize(n_temps);
    thermo.specific_heat.resize(n_temps);
    thermo.entropy.resize(n_temps);
    thermo.free_energy.resize(n_temps);
    
    // Find minimum energy for numerical stability
    double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
    
    for (int t = 0; t < n_temps; t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function and observables using shifted energies
        // Z = Σ_i w_i * exp(-β * (E_i - E_min))
        double Z = 0.0;
        double E_avg = 0.0;
        double E2_avg = 0.0;
        
        std::vector<double> boltzmann_factors(n_states);
        
        // Compute Boltzmann factors with shift
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            boltzmann_factors[i] = weights[i] * std::exp(-beta * shifted_energy);
            Z += boltzmann_factors[i];
        }
        
        // Normalize and compute expectations
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double prob = boltzmann_factors[i] / Z;
                E_avg += prob * ritz_values[i];
                E2_avg += prob * ritz_values[i] * ritz_values[i];
            }
            
            // Thermodynamic quantities
            thermo.energy[t] = E_avg;
            thermo.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
            thermo.entropy[t] = beta * (E_avg - e_min) + std::log(Z);
            thermo.free_energy[t] = e_min - T * std::log(Z);
        } else {
            // Very low temperature - use ground state
            thermo.energy[t] = e_min;
            thermo.specific_heat[t] = 0.0;
            thermo.entropy[t] = 0.0;
            thermo.free_energy[t] = e_min;
        }
    }
    
    return thermo;
}

/**
 * @brief Average thermodynamic data across multiple samples with error estimation
 */
void average_ftlm_samples(
    const std::vector<ThermodynamicData>& sample_data,
    FTLMResults& results
) {
    int n_samples = sample_data.size();
    if (n_samples == 0) return;
    
    int n_temps = sample_data[0].temperatures.size();
    
    results.thermo_data.temperatures = sample_data[0].temperatures;
    results.thermo_data.energy.resize(n_temps, 0.0);
    results.thermo_data.specific_heat.resize(n_temps, 0.0);
    results.thermo_data.entropy.resize(n_temps, 0.0);
    results.thermo_data.free_energy.resize(n_temps, 0.0);
    
    results.energy_error.resize(n_temps, 0.0);
    results.specific_heat_error.resize(n_temps, 0.0);
    results.entropy_error.resize(n_temps, 0.0);
    results.free_energy_error.resize(n_temps, 0.0);
    
    // First pass: compute means
    for (int s = 0; s < n_samples; s++) {
        for (int t = 0; t < n_temps; t++) {
            results.thermo_data.energy[t] += sample_data[s].energy[t];
            results.thermo_data.specific_heat[t] += sample_data[s].specific_heat[t];
            results.thermo_data.entropy[t] += sample_data[s].entropy[t];
            results.thermo_data.free_energy[t] += sample_data[s].free_energy[t];
        }
    }
    
    for (int t = 0; t < n_temps; t++) {
        results.thermo_data.energy[t] /= n_samples;
        results.thermo_data.specific_heat[t] /= n_samples;
        results.thermo_data.entropy[t] /= n_samples;
        results.thermo_data.free_energy[t] /= n_samples;
    }
    
    // Second pass: compute standard errors
    if (n_samples > 1) {
        for (int s = 0; s < n_samples; s++) {
            for (int t = 0; t < n_temps; t++) {
                double diff_e = sample_data[s].energy[t] - results.thermo_data.energy[t];
                double diff_c = sample_data[s].specific_heat[t] - results.thermo_data.specific_heat[t];
                double diff_s = sample_data[s].entropy[t] - results.thermo_data.entropy[t];
                double diff_f = sample_data[s].free_energy[t] - results.thermo_data.free_energy[t];
                
                results.energy_error[t] += diff_e * diff_e;
                results.specific_heat_error[t] += diff_c * diff_c;
                results.entropy_error[t] += diff_s * diff_s;
                results.free_energy_error[t] += diff_f * diff_f;
            }
        }
        
        // Standard error = sqrt(variance / n_samples)
        double norm = std::sqrt(static_cast<double>(n_samples * (n_samples - 1)));
        for (int t = 0; t < n_temps; t++) {
            results.energy_error[t] = std::sqrt(results.energy_error[t]) / norm;
            results.specific_heat_error[t] = std::sqrt(results.specific_heat_error[t]) / norm;
            results.entropy_error[t] = std::sqrt(results.entropy_error[t]) / norm;
            results.free_energy_error[t] = std::sqrt(results.free_energy_error[t]) / norm;
        }
    }
}

/**
 * @brief Main FTLM driver function
 */
FTLMResults finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const FTLMParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir
) {
    std::cout << "\n==========================================\n";
    std::cout << "Finite Temperature Lanczos Method (FTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Number of samples: " << params.num_samples << std::endl;
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    
    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(1, num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for results
    FTLMResults results;
    results.total_samples = params.num_samples;
    std::vector<ThermodynamicData> sample_data;
    std::vector<double> ground_state_estimates;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/ftlm_samples";
        system(cmd.c_str());
    }
    
    // Loop over samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- FTLM Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state
        ComplexVector v0(N);
        for (int i = 0; i < N; i++) {
            v0[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v0.data(), 1);
        
        // Build Lanczos tridiagonal
        std::vector<double> alpha, beta;
        int iterations = build_lanczos_tridiagonal(
            H, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta
        );
        
        std::cout << "  Lanczos iterations: " << iterations << std::endl;
        
        int m = alpha.size();
        
        // Diagonalize tridiagonal matrix
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        std::vector<double> evecs(m * m);
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed with code " << info << std::endl;
            continue;
        }
        
        // Extract Ritz values and weights
        std::vector<double> ritz_values(m);
        std::vector<double> weights(m);
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] = diag[i];
            // Weight is squared first component of eigenvector
            weights[i] = evecs[i * m] * evecs[i * m];
        }
        
        ground_state_estimates.push_back(ritz_values[0]);
        std::cout << "  Ground state estimate: " << ritz_values[0] << std::endl;
        
        // Compute thermodynamics for this sample
        ThermodynamicData sample_thermo = compute_ftlm_thermodynamics(
            ritz_values, weights, temperatures
        );
        sample_data.push_back(sample_thermo);
        
        // Save intermediate data if requested
        if (params.store_intermediate && !output_dir.empty()) {
            std::string sample_file = output_dir + "/ftlm_samples/sample_" + std::to_string(sample) + ".txt";
            std::ofstream f(sample_file);
            if (f.is_open()) {
                f << "# Temperature  Energy  Specific_Heat  Entropy  Free_Energy\n";
                for (size_t t = 0; t < temperatures.size(); t++) {
                    f << std::scientific << std::setprecision(12)
                      << temperatures[t] << " "
                      << sample_thermo.energy[t] << " "
                      << sample_thermo.specific_heat[t] << " "
                      << sample_thermo.entropy[t] << " "
                      << sample_thermo.free_energy[t] << "\n";
                }
                f.close();
            }
        }
    }
    
    // Average over all samples
    std::cout << "\n--- Averaging over " << sample_data.size() << " samples ---\n";
    
    if (params.compute_error_bars) {
        results.per_sample_data = sample_data;
    }
    
    average_ftlm_samples(sample_data, results);
    
    // Estimate ground state as minimum across all samples
    if (!ground_state_estimates.empty()) {
        results.ground_state_estimate = *std::min_element(
            ground_state_estimates.begin(), ground_state_estimates.end()
        );
        std::cout << "Best ground state estimate: " << results.ground_state_estimate << std::endl;
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "FTLM Calculation Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}

/**
 * @brief Save FTLM results to file
 */
void save_ftlm_results(
    const FTLMResults& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "# FTLM Results (averaged over " << results.total_samples << " samples)\n";
    file << "# Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error\n";
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < results.thermo_data.temperatures.size(); i++) {
        file << results.thermo_data.temperatures[i] << " "
             << results.thermo_data.energy[i] << " "
             << results.energy_error[i] << " "
             << results.thermo_data.specific_heat[i] << " "
             << results.specific_heat_error[i] << " "
             << results.thermo_data.entropy[i] << " "
             << results.entropy_error[i] << " "
             << results.thermo_data.free_energy[i] << " "
             << results.free_energy_error[i] << "\n";
    }
    
    file.close();
    std::cout << "FTLM results saved to: " << filename << std::endl;
}

/**
 * @brief Helper function to compute spectral function from Ritz values and weights
 * 
 * @param ritz_values Eigenvalues (energies)
 * @param weights Statistical weights (without thermal factors)
 * @param frequencies Frequency grid
 * @param broadening Lorentzian broadening parameter
 * @param temperature Temperature (if <= 0, no thermal weighting applied)
 * @param spectral_function Output spectral function
 */
static void compute_spectral_function(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& frequencies,
    double broadening,
    double temperature,
    std::vector<double>& spectral_function
){
    int n_omega = frequencies.size();
    int n_states = ritz_values.size();
    
    spectral_function.resize(n_omega, 0.0);
    
    // Compute thermal weights if temperature > 0
    std::vector<double> thermal_weights = weights;
    
    if (temperature > 1e-14) {
        double beta = 1.0 / temperature;
        
        // Find minimum energy for numerical stability
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        // Compute partition function with shifted energies
        double Z = 0.0;
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            thermal_weights[i] = weights[i] * std::exp(-beta * shifted_energy);
            Z += thermal_weights[i];
        }
        
        // Normalize by partition function
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                thermal_weights[i] /= Z;
            }
        } else {
            // Very low temperature - only ground state contributes
            thermal_weights.assign(n_states, 0.0);
            int gs_idx = std::distance(ritz_values.begin(),
                                      std::min_element(ritz_values.begin(), ritz_values.end()));
            thermal_weights[gs_idx] = weights[gs_idx];
            // Normalize
            double sum = 0.0;
            for (double w : thermal_weights) sum += w;
            if (sum > 0) {
                for (double& w : thermal_weights) w /= sum;
            }
        }
    }
    
    // For each frequency, sum contributions from all states
    // S(ω,T) = Σ_i w_i * exp(-βE_i)/Z * δ(ω - E_i)
    // Using Lorentzian broadening: δ(ω - E) → (η/π) / ((ω - E)² + η²)
    double norm_factor = broadening / M_PI;
    
    for (int i_omega = 0; i_omega < n_omega; i_omega++) {
        double omega = frequencies[i_omega];
        
        for (int i = 0; i < n_states; i++) {
            double delta = omega - ritz_values[i];
            double lorentzian = norm_factor / (delta * delta + broadening * broadening);
            spectral_function[i_omega] += thermal_weights[i] * lorentzian;
        }
    }
}
/**
 * @brief Compute dynamical response S(ω) for operator O using Lanczos method
 */
DynamicalResponseResults compute_dynamical_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& psi,
    int N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double temperature,
    const std::string& output_dir
){
    std::cout << "\n==========================================\n";
    std::cout << "Dynamical Response: S(ω) = <O†δ(ω-H)O>\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
    std::cout << "Broadening: " << params.broadening << std::endl;
    if (temperature > 1e-14) {
        std::cout << "Temperature: " << temperature << std::endl;
    } else {
        std::cout << "Temperature: 0 (no thermal weighting)" << std::endl;
    }
    
    DynamicalResponseResults results;
    results.total_samples = 1;
    
    // Generate frequency grid
    results.frequencies.resize(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(1, num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
    }
    
    // Apply operator O to initial state: |φ⟩ = O|ψ⟩
    ComplexVector phi(N);
    O(psi.data(), phi.data(), N);
    
    // Normalize |φ⟩
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    if (phi_norm < 1e-14) {
        std::cerr << "Warning: O|ψ⟩ has zero norm, operator has no matrix elements\n";
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        return results;
    }
    
    std::cout << "Norm of O|ψ⟩: " << phi_norm << std::endl;
    Complex scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &scale, phi.data(), 1);
    
    // Build Lanczos tridiagonal for H starting from |φ⟩
    std::vector<double> alpha, beta;
    int iterations = build_lanczos_tridiagonal(
        H, phi, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta
    );
    
    std::cout << "Lanczos iterations: " << iterations << std::endl;
    int m = alpha.size();
    
    // Diagonalize tridiagonal matrix
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    std::vector<double> evecs(m * m);
    int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
    
    if (info != 0) {
        std::cerr << "Error: Tridiagonal diagonalization failed with code " << info << std::endl;
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        return results;
    }
    
    // Extract Ritz values and weights
    // Weight for state i is |⟨v_0|i⟩|² where v_0 is the first Lanczos vector
    // This equals the squared first component of the eigenvector
    std::vector<double> ritz_values(m);
    std::vector<double> weights(m);
    
    for (int i = 0; i < m; i++) {
        ritz_values[i] = diag[i];
        weights[i] = evecs[i * m] * evecs[i * m];
    }
    
    // Scale weights by the norm factor
    double norm_factor = phi_norm * phi_norm;
    for (int i = 0; i < m; i++) {
        weights[i] *= norm_factor;
    }
    
    std::cout << "Ground state estimate: " << ritz_values[0] << std::endl;
    
    // Compute spectral function with thermal weighting
    compute_spectral_function(ritz_values, weights, results.frequencies, 
                             params.broadening, temperature, results.spectral_function);
    
    // No error bars for single state
    results.spectral_error.resize(num_omega_bins, 0.0);
    
    std::cout << "\n==========================================\n";
    std::cout << "Dynamical Response Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}
/**
 * @brief Compute dynamical response with random initial states (finite temperature)
 */
DynamicalResponseResults compute_dynamical_response_thermal(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    int N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double temperature,
    const std::string& output_dir
){
    std::cout << "\n==========================================\n";
    std::cout << "Thermal Dynamical Response (FTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Number of samples: " << params.num_samples << std::endl;
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
    std::cout << "Broadening: " << params.broadening << std::endl;
    if (temperature > 1e-14) {
        std::cout << "Temperature: " << temperature << std::endl;
    } else {
        std::cout << "Temperature: 0 (no thermal weighting)" << std::endl;
    }
    
    DynamicalResponseResults results;
    results.total_samples = params.num_samples;
    
    // Generate frequency grid
    results.frequencies.resize(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(1, num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for per-sample spectral functions
    std::vector<std::vector<double>> sample_spectra;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/dynamical_samples";
        system(cmd.c_str());
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state |ψ⟩
        ComplexVector psi(N);
        for (int i = 0; i < N; i++) {
            psi[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize |ψ⟩
        double norm = cblas_dznrm2(N, psi.data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, psi.data(), 1);
        
        // Apply operator O: |φ⟩ = O|ψ⟩
        ComplexVector phi(N);
        O(psi.data(), phi.data(), N);
        
        // Get norm of |φ⟩
        double phi_norm = cblas_dznrm2(N, phi.data(), 1);
        if (phi_norm < 1e-14) {
            std::cout << "  Warning: O|ψ⟩ has zero norm, skipping sample\n";
            continue;
        }
        
        std::cout << "  Norm of O|ψ⟩: " << phi_norm << std::endl;
        
        // Normalize |φ⟩
        Complex phi_scale(1.0/phi_norm, 0.0);
        cblas_zscal(N, &phi_scale, phi.data(), 1);
        
        // Build Lanczos tridiagonal
        std::vector<double> alpha, beta;
        int iterations = build_lanczos_tridiagonal(
            H, phi, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta
        );
        
        std::cout << "  Lanczos iterations: " << iterations << std::endl;
        int m = alpha.size();
        
        // Diagonalize tridiagonal
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        std::vector<double> evecs(m * m);
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            continue;
        }
        
        // Extract Ritz values and weights
        std::vector<double> ritz_values(m);
        std::vector<double> weights(m);
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] = diag[i];
            weights[i] = evecs[i * m] * evecs[i * m] * phi_norm * phi_norm;
        }
        
        // Compute spectral function for this sample
        std::vector<double> sample_spectrum;
        compute_spectral_function(ritz_values, weights, results.frequencies,
                                 params.broadening, temperature, sample_spectrum);
        
        sample_spectra.push_back(sample_spectrum);
        
        // Save intermediate data if requested
        if (params.store_intermediate && !output_dir.empty()) {
            std::string sample_file = output_dir + "/dynamical_samples/sample_" + std::to_string(sample) + ".txt";
            std::ofstream f(sample_file);
            if (f.is_open()) {
                f << "# Frequency  Spectral_Function\n";
                for (int i = 0; i < num_omega_bins; i++) {
                    f << std::scientific << std::setprecision(12)
                      << results.frequencies[i] << " "
                      << sample_spectrum[i] << "\n";
                }
                f.close();
            }
        }
    }
    
    // Average over all samples
    int n_valid_samples = sample_spectra.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_error.resize(num_omega_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained\n";
        return results;
    }
    
    // Compute mean
    for (int s = 0; s < n_valid_samples; s++) {
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_function[i] += sample_spectra[s][i];
        }
    }
    
    for (int i = 0; i < num_omega_bins; i++) {
        results.spectral_function[i] /= n_valid_samples;
    }
    
    // Compute standard error
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int i = 0; i < num_omega_bins; i++) {
                double diff = sample_spectra[s][i] - results.spectral_function[i];
                results.spectral_error[i] += diff * diff;
            }
        }
        
        double norm_factor = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_error[i] = std::sqrt(results.spectral_error[i]) / norm_factor;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Thermal Dynamical Response Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}
/**
 * @brief Save dynamical response results to file
 */
void save_dynamical_response_results(
    const DynamicalResponseResults& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "# Dynamical Response Results (averaged over " << results.total_samples << " samples)\n";
    file << "# Frequency  Spectral_Function  Error\n";
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < results.frequencies.size(); i++) {
        file << results.frequencies[i] << " "
             << results.spectral_function[i] << " "
             << results.spectral_error[i] << "\n";
    }
    
    file.close();
    std::cout << "Dynamical response results saved to: " << filename << std::endl;
}

/**
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = <ψ|O1†δ(ω - H)O2|ψ>
 */
DynamicalResponseResults compute_dynamical_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    int N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    int num_omega_bins,
    double temperature,
    const std::string& output_dir
){
    std::cout << "\n==========================================\n";
    std::cout << "Dynamical Correlation: S(ω) = <O₁†δ(ω-H)O₂>\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Number of samples: " << params.num_samples << std::endl;
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
    std::cout << "Broadening: " << params.broadening << std::endl;
    if (temperature > 1e-14) {
        std::cout << "Temperature: " << temperature << std::endl;
    } else {
        std::cout << "Temperature: 0 (no thermal weighting)" << std::endl;
    }
    
    DynamicalResponseResults results;
    results.total_samples = params.num_samples;
    
    // Generate frequency grid
    results.frequencies.resize(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(1, num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for per-sample spectral functions
    std::vector<std::vector<double>> sample_spectra;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/dynamical_correlation_samples";
        system(cmd.c_str());
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state |ψ⟩
        ComplexVector psi(N);
        for (int i = 0; i < N; i++) {
            psi[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize |ψ⟩
        double norm = cblas_dznrm2(N, psi.data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, psi.data(), 1);
        
        // Apply operator O2: |φ⟩ = O₂|ψ⟩
        ComplexVector phi(N);
        O2(psi.data(), phi.data(), N);
        
        // Get norm of |φ⟩
        double phi_norm = cblas_dznrm2(N, phi.data(), 1);
        if (phi_norm < 1e-14) {
            std::cout << "  Warning: O₂|ψ⟩ has zero norm, skipping sample\n";
            continue;
        }
        
        std::cout << "  Norm of O₂|ψ⟩: " << phi_norm << std::endl;
        
        // Normalize |φ⟩
        Complex phi_scale(1.0/phi_norm, 0.0);
        cblas_zscal(N, &phi_scale, phi.data(), 1);
        
        // Build Lanczos tridiagonal for H starting from |φ⟩
        // Store basis vectors for computing matrix elements
        std::vector<double> alpha, beta;
        std::vector<ComplexVector> lanczos_vectors;
        lanczos_vectors.push_back(phi);
        
        ComplexVector v_current = phi;
        ComplexVector v_prev(N, Complex(0.0, 0.0));
        ComplexVector v_next(N);
        ComplexVector w(N);
        
        beta.push_back(0.0);
        int max_iter = std::min(N, params.krylov_dim);
        
        // Lanczos iteration
        for (int j = 0; j < max_iter; j++) {
            H(v_current.data(), w.data(), N);
            
            if (j > 0) {
                Complex neg_beta(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
            }
            
            Complex dot_product;
            cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
            alpha.push_back(std::real(dot_product));
            
            Complex neg_alpha(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
            
            // Reorthogonalization
            if (params.full_reorthogonalization) {
                for (const auto& v : lanczos_vectors) {
                    Complex overlap;
                    cblas_zdotc_sub(N, v.data(), 1, w.data(), 1, &overlap);
                    Complex neg_overlap(-overlap.real(), -overlap.imag());
                    cblas_zaxpy(N, &neg_overlap, v.data(), 1, w.data(), 1);
                }
            } else if (params.reorth_frequency > 0 && (j + 1) % params.reorth_frequency == 0) {
                for (const auto& v : lanczos_vectors) {
                    Complex overlap;
                    cblas_zdotc_sub(N, v.data(), 1, w.data(), 1, &overlap);
                    if (std::abs(overlap) > params.tolerance) {
                        Complex neg_overlap(-overlap.real(), -overlap.imag());
                        cblas_zaxpy(N, &neg_overlap, v.data(), 1, w.data(), 1);
                    }
                }
            }
            
            norm = cblas_dznrm2(N, w.data(), 1);
            beta.push_back(norm);
            
            if (norm < params.tolerance) {
                std::cout << "  Converged at iteration " << j + 1 << std::endl;
                break;
            }
            
            for (int i = 0; i < N; i++) {
                v_next[i] = w[i] / norm;
            }
            
            lanczos_vectors.push_back(v_next);
            v_prev = v_current;
            v_current = v_next;
        }
        
        int m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << std::endl;
        
        // Diagonalize tridiagonal
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        std::vector<double> evecs(m * m);
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            continue;
        }
        
        // Extract Ritz values
        std::vector<double> ritz_values(m);
        for (int i = 0; i < m; i++) {
            ritz_values[i] = diag[i];
        }
        
        // Compute weights |⟨n|O₁|ψ⟩|² for cross-correlation
        // |n⟩ = Σ_j evecs[n,j] |v_j⟩ where |v_0⟩ = O₂|ψ⟩/||O₂|ψ⟩||
        // We need ⟨ψ|O₁†|n⟩ = ⟨O₁ψ|n⟩
        
        // Apply O1 to original state
        ComplexVector O1_psi(N);
        O1(psi.data(), O1_psi.data(), N);
        
        std::vector<double> weights(m);
        
        for (int n = 0; n < m; n++) {
            // Reconstruct |n⟩ in full space
            ComplexVector psi_n(N, Complex(0.0, 0.0));
            for (int j = 0; j < m; j++) {
                Complex coeff(evecs[n * m + j], 0.0);
                cblas_zaxpy(N, &coeff, lanczos_vectors[j].data(), 1, psi_n.data(), 1);
            }
            
            // Compute ⟨O₁ψ|n⟩
            Complex overlap;
            cblas_zdotc_sub(N, O1_psi.data(), 1, psi_n.data(), 1, &overlap);
            
            // Weight is |overlap|² * phi_norm²
            weights[n] = std::norm(overlap) * phi_norm * phi_norm;
        }
        
        // Compute spectral function for this sample
        std::vector<double> sample_spectrum;
        compute_spectral_function(ritz_values, weights, results.frequencies,
                                 params.broadening, temperature, sample_spectrum);
        
        sample_spectra.push_back(sample_spectrum);
        
        // Save intermediate data if requested
        if (params.store_intermediate && !output_dir.empty()) {
            std::string sample_file = output_dir + "/dynamical_correlation_samples/sample_" + std::to_string(sample) + ".txt";
            std::ofstream f(sample_file);
            if (f.is_open()) {
                f << "# Frequency  Spectral_Function\n";
                for (int i = 0; i < num_omega_bins; i++) {
                    f << std::scientific << std::setprecision(12)
                      << results.frequencies[i] << " "
                      << sample_spectrum[i] << "\n";
                }
                f.close();
            }
        }
    }
    
    // Average over all samples
    int n_valid_samples = sample_spectra.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_error.resize(num_omega_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained\n";
        return results;
    }
    
    // Compute mean
    for (int s = 0; s < n_valid_samples; s++) {
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_function[i] += sample_spectra[s][i];
        }
    }
    
    for (int i = 0; i < num_omega_bins; i++) {
        results.spectral_function[i] /= n_valid_samples;
    }
    
    // Compute standard error
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int i = 0; i < num_omega_bins; i++) {
                double diff = sample_spectra[s][i] - results.spectral_function[i];
                results.spectral_error[i] += diff * diff;
            }
        }
        
        double norm_factor = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_error[i] = std::sqrt(results.spectral_error[i]) / norm_factor;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Dynamical Correlation Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}
/**
 * @brief Helper function to compute expectation values in Krylov basis
 */
static void compute_krylov_expectation_values(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& v0,
    int N,
    int krylov_dim,
    double tolerance,
    bool full_reorth,
    int reorth_freq,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>& expectation_values
) {
    // Build Lanczos tridiagonal for Hamiltonian
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    lanczos_vectors.push_back(v0);
    
    ComplexVector v_current = v0;
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Normalize initial vector
    double norm = cblas_dznrm2(N, v_current.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v_current.data(), 1);
    
    beta.push_back(0.0);
    int max_iter = std::min(N, krylov_dim);
    
    // Lanczos iteration (store basis vectors for later)
    for (int j = 0; j < max_iter; j++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (j > 0) {
            Complex neg_beta = Complex(-beta[j], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[j], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Reorthogonalization
        if (full_reorth) {
            for (size_t k = 0; k < lanczos_vectors.size(); k++) {
                Complex overlap;
                cblas_zdotc_sub(N, lanczos_vectors[k].data(), 1, w.data(), 1, &overlap);
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, lanczos_vectors[k].data(), 1, w.data(), 1);
            }
        } else if (reorth_freq > 0 && (j + 1) % reorth_freq == 0) {
            for (size_t k = 0; k < lanczos_vectors.size(); k++) {
                Complex overlap;
                cblas_zdotc_sub(N, lanczos_vectors[k].data(), 1, w.data(), 1, &overlap);
                if (std::abs(overlap) > tolerance) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, lanczos_vectors[k].data(), 1, w.data(), 1);
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown
        if (norm < tolerance) {
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        lanczos_vectors.push_back(v_next);
        v_prev = v_current;
        v_current = v_next;
    }
    
    int m = alpha.size();
    
    // Diagonalize tridiagonal matrix
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    std::vector<double> evecs(m * m);
    int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
    
    if (info != 0) {
        std::cerr << "Warning: Tridiagonal diagonalization failed" << std::endl;
        return;
    }
    
    ritz_values.resize(m);
    weights.resize(m);
    expectation_values.resize(m);
    
    for (int i = 0; i < m; i++) {
        ritz_values[i] = diag[i];
        weights[i] = evecs[i * m] * evecs[i * m];
    }
    
    // Now compute <n|O|n> for each Ritz state in the Krylov basis
    // |n> = Σ_j evecs[n,j] |v_j>
    for (int n = 0; n < m; n++) {
        // Reconstruct |n> in full Hilbert space
        ComplexVector psi_n(N, Complex(0.0, 0.0));
        for (int j = 0; j < m; j++) {
            double coeff = evecs[n * m + j];
            Complex alpha(coeff, 0.0);
            cblas_zaxpy(N, &alpha, lanczos_vectors[j].data(), 1, psi_n.data(), 1);
        }
        
        // Apply O to |n>
        ComplexVector O_psi_n(N);
        O(psi_n.data(), O_psi_n.data(), N);
        
        // Compute <n|O|n>
        Complex expectation_complex;
        cblas_zdotc_sub(N, psi_n.data(), 1, O_psi_n.data(), 1, &expectation_complex);
        expectation_values[n] = std::real(expectation_complex);
    }
}

/**
 * @brief Compute thermal expectation value (single operator)
 */
StaticResponseResults compute_thermal_expectation_value(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    int N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir
) {
    std::cout << "\n==========================================\n";
    std::cout << "Thermal Expectation Value (FTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Number of samples: " << params.num_samples << std::endl;
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    
    StaticResponseResults results;
    results.total_samples = params.num_samples;
    
    // Generate temperature grid (logarithmic spacing)
    results.temperatures.resize(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(1, num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        results.temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for per-sample thermal averages
    std::vector<std::vector<double>> sample_expectations(params.num_samples);
    std::vector<std::vector<double>> sample_variances(params.num_samples);
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_samples";
        system(cmd.c_str());
    }
    
    // Loop over samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state
        ComplexVector v0(N);
        for (int i = 0; i < N; i++) {
            v0[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v0.data(), 1);
        
        // Build Krylov subspace and compute expectation values
        std::vector<double> ritz_values, weights, expectation_values;
        compute_krylov_expectation_values(
            H, O, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            ritz_values, weights, expectation_values
        );
        
        int m = ritz_values.size();
        std::cout << "  Krylov subspace size: " << m << std::endl;
        
        if (m == 0) {
            std::cerr << "  Warning: Failed to build Krylov subspace, skipping sample\n";
            continue;
        }
        
        // Store sample data if requested
        if (params.store_intermediate) {
            StaticResponseSample sample_data;
            sample_data.ritz_values = ritz_values;
            sample_data.weights = weights;
            sample_data.expectation_values = expectation_values;
            sample_data.lanczos_iterations = m;
            results.per_sample_data.push_back(sample_data);
        }
        
        // Compute thermal averages for this sample
        sample_expectations[sample].resize(num_temp_bins);
        sample_variances[sample].resize(num_temp_bins);
        
        // Find minimum energy for numerical stability
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int t = 0; t < num_temp_bins; t++) {
            double T = results.temperatures[t];
            double beta = 1.0 / T;
            
            // Compute partition function and thermal averages
            double Z = 0.0;
            double O_avg = 0.0;
            double O2_avg = 0.0;
            
            // Compute Boltzmann factors with energy shift
            std::vector<double> boltzmann_factors(m);
            for (int i = 0; i < m; i++) {
                double shifted_energy = ritz_values[i] - e_min;
                boltzmann_factors[i] = weights[i] * std::exp(-beta * shifted_energy);
                Z += boltzmann_factors[i];
            }
            
            // Compute expectations
            if (Z > 1e-300) {
                for (int i = 0; i < m; i++) {
                    double prob = boltzmann_factors[i] / Z;
                    O_avg += prob * expectation_values[i];
                    O2_avg += prob * expectation_values[i] * expectation_values[i];
                }
                
                sample_expectations[sample][t] = O_avg;
                sample_variances[sample][t] = O2_avg - O_avg * O_avg;
            } else {
                // Very low temperature - use ground state
                int gs_idx = std::distance(ritz_values.begin(), 
                                          std::min_element(ritz_values.begin(), ritz_values.end()));
                sample_expectations[sample][t] = expectation_values[gs_idx];
                sample_variances[sample][t] = 0.0;
            }
        }
        
        // Save intermediate data if requested
        if (params.store_intermediate && !output_dir.empty()) {
            std::string sample_file = output_dir + "/static_samples/sample_" + std::to_string(sample) + ".txt";
            std::ofstream f(sample_file);
            if (f.is_open()) {
                f << "# Temperature  Expectation  Variance\n";
                for (int t = 0; t < num_temp_bins; t++) {
                    f << std::scientific << std::setprecision(12)
                      << results.temperatures[t] << " "
                      << sample_expectations[sample][t] << " "
                      << sample_variances[sample][t] << "\n";
                }
                f.close();
            }
        }
    }
    
    // Average over all samples
    int n_valid_samples = 0;
    for (int s = 0; s < params.num_samples; s++) {
        if (!sample_expectations[s].empty()) n_valid_samples++;
    }
    
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.expectation.resize(num_temp_bins, 0.0);
    results.variance.resize(num_temp_bins, 0.0);
    results.susceptibility.resize(num_temp_bins, 0.0);
    results.expectation_error.resize(num_temp_bins, 0.0);
    results.variance_error.resize(num_temp_bins, 0.0);
    results.susceptibility_error.resize(num_temp_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained" << std::endl;
        return results;
    }
    
    // Compute means
    for (int s = 0; s < params.num_samples; s++) {
        if (sample_expectations[s].empty()) continue;
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation[t] += sample_expectations[s][t];
            results.variance[t] += sample_variances[s][t];
        }
    }
    
    for (int t = 0; t < num_temp_bins; t++) {
        results.expectation[t] /= n_valid_samples;
        results.variance[t] /= n_valid_samples;
        // Susceptibility χ = β * variance
        double beta = 1.0 / results.temperatures[t];
        results.susceptibility[t] = beta * results.variance[t];
    }
    
    // Compute standard errors
    if (params.compute_error_bars && n_valid_samples > 1) {
        for (int s = 0; s < params.num_samples; s++) {
            if (sample_expectations[s].empty()) continue;
            for (int t = 0; t < num_temp_bins; t++) {
                double diff_exp = sample_expectations[s][t] - results.expectation[t];
                double diff_var = sample_variances[s][t] - results.variance[t];
                
                results.expectation_error[t] += diff_exp * diff_exp;
                results.variance_error[t] += diff_var * diff_var;
            }
        }
        
        double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation_error[t] = std::sqrt(results.expectation_error[t]) / norm;
            results.variance_error[t] = std::sqrt(results.variance_error[t]) / norm;
            
            // Error propagation for susceptibility: δχ ≈ β * δ(variance)
            double beta = 1.0 / results.temperatures[t];
            results.susceptibility_error[t] = beta * results.variance_error[t];
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Static Response Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}

/**
 * @brief Compute static response function (two-operator correlation)
 */
StaticResponseResults compute_static_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    int N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir
) {
    std::cout << "\n==========================================\n";
    std::cout << "Static Response Function (FTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Computing correlation ⟨O₁†O₂⟩\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Number of samples: " << params.num_samples << std::endl;
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    
    StaticResponseResults results;
    results.total_samples = params.num_samples;
    
    // Generate temperature grid
    results.temperatures.resize(num_temp_bins);
    double temp_step = (temp_max - temp_min) / std::max(1, num_temp_bins - 1);
    for (int i = 0; i < num_temp_bins; i++) {
        results.temperatures[i] = temp_min + i * temp_step;
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for per-sample results
    std::vector<std::vector<double>> sample_expectations;
    std::vector<std::vector<double>> sample_variances;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_correlation_samples";
        system(cmd.c_str());
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state
        ComplexVector v0(N);
        for (int i = 0; i < N; i++) {
            v0[i] = Complex(dist(gen), dist(gen));
        }
        
        // Normalize
        double norm = cblas_dznrm2(N, v0.data(), 1);
        Complex scale = Complex(1.0/norm, 0.0);
        cblas_zscal(N, &scale, v0.data(), 1);
        
        // Build Lanczos tridiagonal for Hamiltonian (store basis vectors)
        std::vector<double> alpha, beta;
        std::vector<ComplexVector> lanczos_vectors;
        lanczos_vectors.push_back(v0);
        
        ComplexVector v_current = v0;
        ComplexVector v_prev(N, Complex(0.0, 0.0));
        ComplexVector v_next(N);
        ComplexVector w(N);
        
        beta.push_back(0.0);
        int max_iter = std::min(N, params.krylov_dim);
        
        // Lanczos iteration
        for (int j = 0; j < max_iter; j++) {
            H(v_current.data(), w.data(), N);
            
            if (j > 0) {
                Complex neg_beta(-beta[j], 0.0);
                cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
            }
            
            Complex dot_product;
            cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
            alpha.push_back(std::real(dot_product));
            
            Complex neg_alpha(-alpha[j], 0.0);
            cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
            
            // Reorthogonalization
            if (params.full_reorthogonalization) {
                for (const auto& v : lanczos_vectors) {
                    Complex overlap;
                    cblas_zdotc_sub(N, v.data(), 1, w.data(), 1, &overlap);
                    Complex neg_overlap(-overlap.real(), -overlap.imag());
                    cblas_zaxpy(N, &neg_overlap, v.data(), 1, w.data(), 1);
                }
            } else if (params.reorth_frequency > 0 && (j + 1) % params.reorth_frequency == 0) {
                for (const auto& v : lanczos_vectors) {
                    Complex overlap;
                    cblas_zdotc_sub(N, v.data(), 1, w.data(), 1, &overlap);
                    if (std::abs(overlap) > params.tolerance) {
                        Complex neg_overlap(-overlap.real(), -overlap.imag());
                        cblas_zaxpy(N, &neg_overlap, v.data(), 1, w.data(), 1);
                    }
                }
            }
            
            norm = cblas_dznrm2(N, w.data(), 1);
            beta.push_back(norm);
            
            if (norm < params.tolerance) {
                std::cout << "  Converged at iteration " << j + 1 << std::endl;
                break;
            }
            
            for (int i = 0; i < N; i++) {
                v_next[i] = w[i] / norm;
            }
            
            lanczos_vectors.push_back(v_next);
            v_prev = v_current;
            v_current = v_next;
        }
        
        int m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << std::endl;
        
        // Diagonalize tridiagonal
        std::vector<double> diag = alpha;
        std::vector<double> offdiag(m - 1);
        for (int i = 0; i < m - 1; i++) {
            offdiag[i] = beta[i + 1];
        }
        
        std::vector<double> evecs(m * m);
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
        
        if (info != 0) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed" << std::endl;
            continue;
        }
        
        std::vector<double> ritz_values(m);
        std::vector<double> weights(m);
        std::vector<double> correlation_values(m);
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] = diag[i];
            weights[i] = evecs[i * m] * evecs[i * m];
        }
        
        // Compute ⟨n|O₁†O₂|n⟩ for each eigenstate |n⟩
        // This equals ⟨O₁n|O₂n⟩ = (O₁|n⟩)† · (O₂|n⟩)
        for (int n = 0; n < m; n++) {
            // Reconstruct |n⟩ in full space: |n⟩ = Σ_j evecs[n,j] |v_j⟩
            ComplexVector psi_n(N, Complex(0.0, 0.0));
            for (int j = 0; j < m; j++) {
                Complex coeff(evecs[n * m + j], 0.0);
                cblas_zaxpy(N, &coeff, lanczos_vectors[j].data(), 1, psi_n.data(), 1);
            }
            
            // Apply O₁ and O₂
            ComplexVector O1_psi_n(N);
            ComplexVector O2_psi_n(N);
            O1(psi_n.data(), O1_psi_n.data(), N);
            O2(psi_n.data(), O2_psi_n.data(), N);
            
            // Compute ⟨O₁n|O₂n⟩ = ⟨n|O₁†O₂|n⟩
            Complex correlation_complex;
            cblas_zdotc_sub(N, O1_psi_n.data(), 1, O2_psi_n.data(), 1, &correlation_complex);
            correlation_values[n] = std::real(correlation_complex);
        }
        
        // Compute thermal averages for this sample
        std::vector<double> sample_exp(num_temp_bins);
        std::vector<double> sample_var(num_temp_bins);
        
        for (int t = 0; t < num_temp_bins; t++) {
            double T = results.temperatures[t];
            double beta = 1.0 / T;
            
            // Compute partition function
            double Z = 0.0;
            for (int i = 0; i < m; i++) {
                Z += weights[i] * std::exp(-beta * ritz_values[i]);
            }
            
            // Compute ⟨O₁†O₂⟩
            double expectation = 0.0;
            for (int i = 0; i < m; i++) {
                double boltzmann = std::exp(-beta * ritz_values[i]);
                expectation += weights[i] * correlation_values[i] * boltzmann / Z;
            }
            
            // Compute ⟨(O₁†O₂)²⟩ for variance
            double expectation_squared = 0.0;
            for (int i = 0; i < m; i++) {
                double boltzmann = std::exp(-beta * ritz_values[i]);
                expectation_squared += weights[i] * correlation_values[i] * correlation_values[i] * boltzmann / Z;
            }
            
            sample_exp[t] = expectation;
            sample_var[t] = expectation_squared - expectation * expectation;
        }
        
        sample_expectations.push_back(sample_exp);
        sample_variances.push_back(sample_var);
        
        // Store per-sample data if requested
        if (params.store_intermediate) {
            StaticResponseSample sample_data;
            sample_data.ritz_values = ritz_values;
            sample_data.weights = weights;
            sample_data.expectation_values = correlation_values;
            sample_data.lanczos_iterations = m;
            results.per_sample_data.push_back(sample_data);
        }
    }
    
    // Average over samples
    int n_valid_samples = sample_expectations.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.expectation.resize(num_temp_bins, 0.0);
    results.variance.resize(num_temp_bins, 0.0);
    results.expectation_error.resize(num_temp_bins, 0.0);
    results.variance_error.resize(num_temp_bins, 0.0);
    results.susceptibility.resize(num_temp_bins, 0.0);
    results.susceptibility_error.resize(num_temp_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples" << std::endl;
        return results;
    }
    
    // Compute means
    for (int s = 0; s < n_valid_samples; s++) {
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation[t] += sample_expectations[s][t];
            results.variance[t] += sample_variances[s][t];
        }
    }
    
    for (int t = 0; t < num_temp_bins; t++) {
        results.expectation[t] /= n_valid_samples;
        results.variance[t] /= n_valid_samples;
        double beta = 1.0 / results.temperatures[t];
        results.susceptibility[t] = beta * results.variance[t];
    }
    
    // Compute standard errors
    if (params.compute_error_bars && n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int t = 0; t < num_temp_bins; t++) {
                double diff_exp = sample_expectations[s][t] - results.expectation[t];
                double diff_var = sample_variances[s][t] - results.variance[t];
                
                results.expectation_error[t] += diff_exp * diff_exp;
                results.variance_error[t] += diff_var * diff_var;
            }
        }
        
        double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation_error[t] = std::sqrt(results.expectation_error[t]) / norm;
            results.variance_error[t] = std::sqrt(results.variance_error[t]) / norm;
            
            double beta = 1.0 / results.temperatures[t];
            results.susceptibility_error[t] = beta * results.variance_error[t];
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Static Response Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}

/**
 * @brief Save static response results to file
 */
void save_static_response_results(
    const StaticResponseResults& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "# Static Response Results (averaged over " << results.total_samples << " samples)\n";
    file << "# Temperature  Expectation  Exp_Error  Variance  Var_Error  Susceptibility  Chi_Error\n";
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < results.temperatures.size(); i++) {
        file << results.temperatures[i] << " "
             << results.expectation[i] << " "
             << results.expectation_error[i] << " ";
        
        if (!results.variance.empty()) {
            file << results.variance[i] << " "
                 << results.variance_error[i] << " "
                 << results.susceptibility[i] << " "
                 << results.susceptibility_error[i];
        }
        
        file << "\n";
    }
    
    file.close();
    std::cout << "Static response results saved to: " << filename << std::endl;
}
