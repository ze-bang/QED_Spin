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
