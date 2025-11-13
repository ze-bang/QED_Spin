// ftlm.cpp - Finite Temperature Lanczos Method implementation
#include "../core/system_utils.h"

#include "ftlm.h"
#include "lanczos.h"
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
    uint64_t N,
    uint64_t max_iter,
    double tol,
    bool full_reorth,
    uint64_t reorth_freq,
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
    
    uint64_t n_temps = temperatures.size();
    uint64_t n_states = ritz_values.size();
    
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
    uint64_t n_samples = sample_data.size();
    if (n_samples == 0) return;
    
    uint64_t n_temps = sample_data[0].temperatures.size();
    
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
    uint64_t N,
    const FTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
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
    double log_step = (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);
    
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
        safe_system_call(cmd);
    }
    
    // Loop over samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- FTLM Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state
        ComplexVector v0 = generateRandomVector(N, gen, dist);
        
        // Build Lanczos tridiagonal
        std::vector<double> alpha, beta;
        uint64_t iterations = build_lanczos_tridiagonal(
            H, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta
        );
        
        std::cout << "  Lanczos iterations: " << iterations << std::endl;
        
        // Diagonalize tridiagonal and extract Ritz values/weights
        std::vector<double> ritz_values, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed" << std::endl;
            continue;
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
 * @brief Combine FTLM results from multiple symmetry sectors
 */
ThermodynamicData combine_ftlm_sector_results(
    const std::vector<FTLMResults>& sector_results,
    const std::vector<uint64_t>& sector_dims
) {
    if (sector_results.empty()) {
        throw std::runtime_error("combine_ftlm_sector_results: No sector results to combine");
    }
    
    if (sector_results.size() != sector_dims.size()) {
        throw std::runtime_error("combine_ftlm_sector_results: Mismatch between number of sectors and dimensions");
    }
    
    size_t n_sectors = sector_results.size();
    std::cout << "\n=== Combining FTLM Results from " << n_sectors << " Symmetry Sectors ===" << std::endl;
    
    // All sectors should have the same temperature grid
    const auto& temps = sector_results[0].thermo_data.temperatures;
    size_t n_temps = temps.size();
    
    // Verify all sectors have same temperature grid
    for (size_t s = 1; s < n_sectors; ++s) {
        if (sector_results[s].thermo_data.temperatures.size() != n_temps) {
            throw std::runtime_error("combine_ftlm_sector_results: Sectors have different temperature grids");
        }
    }
    
    // Initialize combined results
    ThermodynamicData combined;
    combined.temperatures = temps;
    combined.energy.resize(n_temps, 0.0);
    combined.specific_heat.resize(n_temps, 0.0);
    combined.entropy.resize(n_temps, 0.0);
    combined.free_energy.resize(n_temps, 0.0);
    
    // Report sector dimensions
    uint64_t total_dim = 0;
    for (size_t s = 0; s < n_sectors; ++s) {
        std::cout << "  Sector " << s << ": dimension = " << sector_dims[s] << std::endl;
        total_dim += sector_dims[s];
    }
    std::cout << "  Total dimension: " << total_dim << std::endl;
    
    // For each temperature, combine sector contributions
    for (size_t t = 0; t < n_temps; ++t) {
        double T = temps[t];
        double beta = 1.0 / T;
        
        // Step 1: Compute partition function for each sector
        // Z_s(β) = exp(-β F_s)
        // Since sectors may have different ground state energies, we need to use a reference
        // to avoid numerical overflow/underflow
        
        // Find minimum free energy across all sectors for numerical stability
        double F_ref = sector_results[0].thermo_data.free_energy[t];
        for (size_t s = 1; s < n_sectors; ++s) {
            double F_s = sector_results[s].thermo_data.free_energy[t];
            if (F_s < F_ref) {
                F_ref = F_s;
            }
        }
        
        // Compute shifted partition functions: Z_s = exp(-β(F_s - F_ref))
        std::vector<double> Z_sectors;
        double Z_total = 0.0;
        
        for (size_t s = 0; s < n_sectors; ++s) {
            double F_s = sector_results[s].thermo_data.free_energy[t];
            double delta_F = F_s - F_ref;
            double Z_s = std::exp(-beta * delta_F);
            
            // Handle numerical overflow/underflow
            if (!std::isfinite(Z_s) || Z_s < 0.0) {
                std::cerr << "Warning: Numerical issue in sector " << s << " at T=" << T 
                          << ", F_s=" << F_s << ", delta_F=" << delta_F << std::endl;
                Z_s = 0.0;  // Will be handled below
            }
            
            Z_sectors.push_back(Z_s);
            Z_total += Z_s;
        }
        
        // Check for numerical issues
        if (Z_total <= 1e-300 || !std::isfinite(Z_total)) {
            std::cerr << "Error: Total partition function is zero or invalid at T=" << T << std::endl;
            std::cerr << "  This suggests all sectors have very high free energies." << std::endl;
            // Use the minimum free energy sector as fallback
            combined.free_energy[t] = F_ref;
            combined.energy[t] = sector_results[0].thermo_data.energy[t];  // Will be overwritten if Z_total > 0
            combined.specific_heat[t] = 0.0;
            combined.entropy[t] = 0.0;
            continue;  // Skip to next temperature
        }
        
        // Total free energy with reference shift: F_total = F_ref - T ln(Z_total)
        combined.free_energy[t] = F_ref - T * std::log(Z_total);
        
        // Step 2: Compute sector weights (normalized partition function contributions)
        std::vector<double> weights(n_sectors);
        for (size_t s = 0; s < n_sectors; ++s) {
            weights[s] = Z_sectors[s] / Z_total;
        }
        
        // Debug output for first and last temperature
        if (t == 0 || t == n_temps - 1) {
            std::cout << "\n  T=" << T << " (beta=" << beta << "):" << std::endl;
            std::cout << "    F_ref=" << F_ref << std::endl;
            for (size_t s = 0; s < n_sectors; ++s) {
                std::cout << "    Sector " << s << ": F=" << sector_results[s].thermo_data.free_energy[t]
                          << ", Z_s/Z_total=" << weights[s] << ", <E>=" << sector_results[s].thermo_data.energy[t]
                          << std::endl;
            }
        }
        
        // Step 3: Combine observables with proper weights
        // For energy: <E>_total = Σ_s (Z_s/Z_total) <E>_s
        // For variance: Var[E]_total requires combining sector variances
        double E_total = 0.0;
        double E2_total = 0.0;
        
        for (size_t s = 0; s < n_sectors; ++s) {
            double w_s = weights[s];
            double E_s = sector_results[s].thermo_data.energy[t];
            double C_s = sector_results[s].thermo_data.specific_heat[t];
            
            // Weighted energy: <E> = Σ_s w_s <E>_s
            E_total += w_s * E_s;
            
            // For specific heat combination, we need <E²>:
            // C_s = β²(<E²>_s - <E>_s²) → <E²>_s = C_s/β² + <E>_s²
            // Then: <E²>_total = Σ_s w_s <E²>_s
            double E2_s = C_s / (beta * beta) + E_s * E_s;
            E2_total += w_s * E2_s;
        }
        
        // Step 4: Final thermodynamic quantities
        combined.energy[t] = E_total;
        
        // Combined specific heat: C = β²(<E²> - <E>²)
        combined.specific_heat[t] = beta * beta * (E2_total - E_total * E_total);
        
        // Entropy from thermodynamic relation: S = β(E - F)
        combined.entropy[t] = beta * (E_total - combined.free_energy[t]);
        
        // Additional diagnostic output for first/last temperature
        if (t == 0 || t == n_temps - 1) {
            std::cout << "    Combined: F=" << combined.free_energy[t] 
                      << ", <E>=" << combined.energy[t]
                      << ", C=" << combined.specific_heat[t]
                      << ", S=" << combined.entropy[t] << std::endl;
        }
    }
    
    // Final verification: check that combined results make physical sense
    std::cout << "\n=== Verification of Combined Results ===" << std::endl;
    
    // Check a mid-range temperature for sanity
    size_t mid_t = n_temps / 2;
    double mid_T = temps[mid_t];
    double mid_E = combined.energy[mid_t];
    
    // Find min/max energies across sectors at this temperature
    double E_min = sector_results[0].thermo_data.energy[mid_t];
    double E_max = E_min;
    for (size_t s = 1; s < n_sectors; ++s) {
        double E_s = sector_results[s].thermo_data.energy[mid_t];
        if (E_s < E_min) E_min = E_s;
        if (E_s > E_max) E_max = E_s;
    }
    
    std::cout << "  At T=" << mid_T << ":" << std::endl;
    std::cout << "    Sector energy range: [" << E_min << ", " << E_max << "]" << std::endl;
    std::cout << "    Combined energy: " << mid_E << std::endl;
    
    if (mid_E < E_min || mid_E > E_max) {
        std::cout << "    ⚠ WARNING: Combined energy is outside sector range!" << std::endl;
        std::cout << "    This may indicate an issue with sector combination." << std::endl;
    } else {
        std::cout << "    ✓ Combined energy is within expected range" << std::endl;
    }
    
    // Check that specific heat is non-negative
    bool all_positive_C = true;
    for (size_t t = 0; t < n_temps; ++t) {
        if (combined.specific_heat[t] < -1e-10) {  // Allow small numerical errors
            all_positive_C = false;
            std::cout << "  ⚠ WARNING: Negative specific heat at T=" << temps[t] 
                      << ", C=" << combined.specific_heat[t] << std::endl;
        }
    }
    
    if (all_positive_C) {
        std::cout << "  ✓ All specific heat values are non-negative" << std::endl;
    }
    
    std::cout << "\nSuccessfully combined thermodynamic data from all sectors" << std::endl;
    std::cout << "=== Sector Combination Complete ===" << std::endl;
    
    return combined;
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
    uint64_t n_omega = frequencies.size();
    uint64_t n_states = ritz_values.size();
    
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
            uint64_t gs_idx = std::distance(ritz_values.begin(),
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
 * @brief Compute complex spectral function from complex weights
 * 
 * For cross-correlation S_{O1,O2}(ω) = ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩, the weights can be complex.
 * This function computes both real and imaginary parts of the spectral function.
 */
static void compute_spectral_function_complex(
    const std::vector<double>& ritz_values,
    const std::vector<Complex>& complex_weights,
    const std::vector<double>& frequencies,
    double broadening,
    double temperature,
    std::vector<double>& spectral_function_real,
    std::vector<double>& spectral_function_imag
){
    uint64_t n_omega = frequencies.size();
    uint64_t n_states = ritz_values.size();
    
    spectral_function_real.resize(n_omega, 0.0);
    spectral_function_imag.resize(n_omega, 0.0);
    
    // Compute thermal weights if temperature > 0
    std::vector<Complex> thermal_weights = complex_weights;
    
    if (temperature > 1e-14) {
        double beta = 1.0 / temperature;
        
        // Find minimum energy for numerical stability
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        // Compute partition function with shifted energies
        // For complex weights: Z = Σ_i Re[w_i] * exp(-βE_i)
        double Z = 0.0;
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            double boltzmann_factor = std::exp(-beta * shifted_energy);
            thermal_weights[i] = complex_weights[i] * boltzmann_factor;
            Z += thermal_weights[i].real() * boltzmann_factor;
        }
        
        // Normalize by partition function
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                thermal_weights[i] /= Z;
            }
        } else {
            // Very low temperature - only ground state contributes
            thermal_weights.assign(n_states, Complex(0.0, 0.0));
            uint64_t gs_idx = std::distance(ritz_values.begin(),
                                      std::min_element(ritz_values.begin(), ritz_values.end()));
            thermal_weights[gs_idx] = complex_weights[gs_idx];
            // Normalize by real part sum
            Complex sum = Complex(0.0, 0.0);
            for (const auto& w : thermal_weights) sum += w;
            if (std::abs(sum) > 0) {
                for (auto& w : thermal_weights) w /= sum;
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
            Complex contribution = thermal_weights[i] * lorentzian;
            spectral_function_real[i_omega] += contribution.real();
            spectral_function_imag[i_omega] += contribution.imag();
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
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
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
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
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
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        return results;
    }
    
    std::cout << "Norm of O|ψ⟩: " << phi_norm << std::endl;
    Complex scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &scale, phi.data(), 1);
    
    // Build Lanczos tridiagonal for H starting from |φ⟩
    std::vector<double> alpha, beta;
    uint64_t iterations = build_lanczos_tridiagonal(
        H, phi, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta
    );
    
    std::cout << "Lanczos iterations: " << iterations << std::endl;
    
    // Diagonalize tridiagonal and extract Ritz values/weights
    std::vector<double> ritz_values, weights;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);
    
    if (ritz_values.empty()) {
        std::cerr << "Error: Tridiagonal diagonalization failed" << std::endl;
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        return results;
    }
    
    // Scale weights by the norm factor
    double norm_factor = phi_norm * phi_norm;
    for (int i = 0; i < weights.size(); i++) {
        weights[i] *= norm_factor;
    }
    
    std::cout << "Ground state estimate: " << ritz_values[0] << std::endl;
    
    // Compute spectral function with thermal weighting
    compute_spectral_function(ritz_values, weights, results.frequencies, 
                             params.broadening, temperature, results.spectral_function);
    
    // No error bars for single state
    // For self-correlation (O†O), imaginary part is zero
    results.spectral_function_imag.resize(num_omega_bins, 0.0);
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
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
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
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
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
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
        safe_system_call(cmd);
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state |ψ⟩
        ComplexVector psi = generateRandomVector(N, gen, dist);
        
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
        uint64_t iterations = build_lanczos_tridiagonal(
            H, phi, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta
        );
        
        std::cout << "  Lanczos iterations: " << iterations << std::endl;
        
        // Diagonalize tridiagonal and extract Ritz values/weights
        std::vector<double> ritz_values, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            continue;
        }
        
        // Scale weights by phi_norm squared
        for (int i = 0; i < weights.size(); i++) {
            weights[i] *= phi_norm * phi_norm;
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
    
    // Average over all samples (FTLM thermal)
    uint64_t n_valid_samples = sample_spectra.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_function_imag.resize(num_omega_bins, 0.0);  // Self-correlation: imaginary part is zero
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
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
    
    // Check if we have imaginary parts (non-zero for cross-correlation)
    bool has_imaginary = false;
    if (!results.spectral_function_imag.empty()) {
        for (double val : results.spectral_function_imag) {
            if (std::abs(val) > 1e-14) {
                has_imaginary = true;
                break;
            }
        }
    }
    
    if (has_imaginary) {
        file << "# Frequency  Re[S(ω)]  Im[S(ω)]  Re[Error]  Im[Error]\n";
    } else {
        file << "# Frequency  Spectral_Function  Error\n";
    }
    
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < results.frequencies.size(); i++) {
        file << results.frequencies[i] << " "
             << results.spectral_function[i];
        
        if (has_imaginary) {
            file << " " << results.spectral_function_imag[i]
                 << " " << results.spectral_error[i]
                 << " " << results.spectral_error_imag[i];
        } else {
            file << " " << results.spectral_error[i];
        }
        
        file << "\n";
    }
    
    file.close();
    std::cout << "Dynamical response results saved to: " << filename << std::endl;
    if (has_imaginary) {
        std::cout << "  (Complex spectral function with both real and imaginary parts)" << std::endl;
    }
}

/**
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = <ψ|O1†δ(ω - H)O2|ψ>
 */
DynamicalResponseResults compute_dynamical_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
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
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
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
        safe_system_call(cmd);
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state |ψ⟩
        ComplexVector psi = generateRandomVector(N, gen, dist);
        
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
        
        uint64_t iterations = build_lanczos_tridiagonal_with_basis(
            H, phi, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta, &lanczos_vectors
        );
        
        uint64_t m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << std::endl;
        
        // Diagonalize tridiagonal (need eigenvectors for weight computation)
        std::vector<double> ritz_values, dummy_weights;
        std::vector<double> evecs;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, dummy_weights, &evecs);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            continue;
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
    
    // Average over all samples (Dynamical Correlation FTLM)
    uint64_t n_valid_samples = sample_spectra.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_function_imag.resize(num_omega_bins, 0.0);  // FTLM averages: imaginary parts cancel
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
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
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = ⟨O₁†(ω)O₂⟩ for a given state
 * 
 * Computes the spectral function S(ω) = Σₙ ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - Eₙ)
 * where |n⟩ are eigenstates of H with energy Eₙ, for a specific state |ψ⟩.
 * 
 * This function uses the Lehmann representation computed via Lanczos:
 * - Applies O₂ to the given state: |φ⟩ = O₂|ψ⟩
 * - Builds Krylov subspace starting from |φ⟩
 * - Diagonalizes H in the Krylov basis to get approximate eigenstates
 * - Computes weights: ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
 * - Constructs spectral function with Lorentzian broadening
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O1 First operator (O₁) matrix-vector product function
 * @param O2 Second operator (O₂) matrix-vector product function
 * @param state Input quantum state |ψ⟩ (must be normalized)
 * @param N Hilbert space dimension
 * @param params Parameters for dynamical response calculation
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega_bins Number of frequency points
 * @param temperature Temperature for Boltzmann weighting of eigenstates (0 = no weighting)
 * @return DynamicalResponseResults containing S_{O1,O2}(ω) vs frequency
 */
DynamicalResponseResults compute_dynamical_correlation_state(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature,
    double energy_shift
){
    std::cout << "\n==========================================\n";
    std::cout << "Dynamical Correlation (Given State): S(ω) = ⟨O₁†(ω)O₂⟩\n";
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
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
    }
    
    // Verify state is normalized
    double state_norm = cblas_dznrm2(N, state.data(), 1);
    if (std::abs(state_norm - 1.0) > 1e-10) {
        std::cout << "  Warning: Input state norm = " << state_norm << " (expected 1.0)\n";
        std::cout << "  Normalizing state...\n";
    }
    
    ComplexVector psi = state;
    Complex scale(1.0/state_norm, 0.0);
    cblas_zscal(N, &scale, psi.data(), 1);
    
    // Apply operator O2: |φ⟩ = O₂|ψ⟩
    ComplexVector phi(N);
    O2(psi.data(), phi.data(), N);
    
    // Get norm of |φ⟩
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    if (phi_norm < 1e-14) {
        std::cerr << "  Error: O₂|ψ⟩ has zero norm\n";
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        return results;
    }
    
    std::cout << "  Norm of O₂|ψ⟩: " << phi_norm << std::endl;
    
    // Normalize |φ⟩
    Complex phi_scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &phi_scale, phi.data(), 1);
    
    // Build Lanczos tridiagonal for H starting from |φ⟩
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    
    uint64_t iterations = build_lanczos_tridiagonal_with_basis(
        H, phi, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta, &lanczos_vectors
    );
    
    uint64_t m = alpha.size();
    std::cout << "  Lanczos iterations: " << m << std::endl;
    
    // Diagonalize tridiagonal (need eigenvectors for weight computation)
    std::vector<double> ritz_values, dummy_weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, dummy_weights, &evecs);
    
    if (ritz_values.empty()) {
        std::cerr << "  Error: Tridiagonal diagonalization failed\n";
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        return results;
    }
    
    // For dynamical structure factors, shift energies so ground state is at E=0
    // This ensures spectral function has weight only at positive frequencies (excitation energies)
    double E_shift;
    if (std::abs(energy_shift) > 1e-14) {
        // Use provided ground state energy shift
        E_shift = energy_shift;
        std::cout << "  Using provided ground state energy shift: " << E_shift << std::endl;
    } else {
        // Auto-detect from Krylov subspace
        E_shift = *std::min_element(ritz_values.begin(), ritz_values.end());
        std::cout << "  Ground state energy (auto-detected from Krylov): " << E_shift << std::endl;
    }
    
    for (int i = 0; i < m; i++) {
        ritz_values[i] -= E_shift;
    }
    std::cout << "  Shifted to excitation energies (E_gs = 0)" << std::endl;
    
    // Compute weights for S(ω) = Σₙ ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - Eₙ)
    // where |n⟩ are eigenstates in the Krylov basis
    
    // Apply O1 to original state: |O₁ψ⟩
    ComplexVector O1_psi(N);
    O1(psi.data(), O1_psi.data(), N);
    
    std::vector<Complex> complex_weights(m);
    
    for (int n = 0; n < m; n++) {
        // Compute ⟨ψ|O₁†|n⟩ = ⟨O₁ψ|n⟩ = Σⱼ evecs[n,j] ⟨O₁ψ|vⱼ⟩
        Complex overlap_O1 = Complex(0.0, 0.0);
        for (int j = 0; j < m; j++) {
            Complex bracket;
            cblas_zdotc_sub(N, O1_psi.data(), 1, lanczos_vectors[j].data(), 1, &bracket);
            overlap_O1 += Complex(evecs[n * m + j], 0.0) * bracket;
        }
        
        // Compute ⟨n|O₂|ψ⟩ = evecs[n,0] × ||O₂|ψ|| (since |v₀⟩ = O₂|ψ⟩/||O₂|ψ||)
        Complex overlap_O2 = Complex(evecs[n * m + 0] * phi_norm, 0.0);
        
        // Weight is ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ (complex product for general cross-correlation)
        Complex weight_complex = std::conj(overlap_O1) * overlap_O2;
        complex_weights[n] = weight_complex;
    }
    
    // Compute spectral function (both real and imaginary parts)
    compute_spectral_function_complex(ritz_values, complex_weights, results.frequencies,
                                      params.broadening, temperature, 
                                      results.spectral_function, results.spectral_function_imag);
    
    // No error bars for single state
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
    std::cout << "\n==========================================\n";
    std::cout << "Dynamical Correlation (Given State) Complete\n";
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
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>& expectation_values
) {
    // Build Lanczos tridiagonal for Hamiltonian with basis storage
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    
    uint64_t iterations = build_lanczos_tridiagonal_with_basis(
        H, v0, N, krylov_dim, tolerance,
        full_reorth, reorth_freq,
        alpha, beta, &lanczos_vectors
    );
    
    uint64_t m = alpha.size();
    
    // Diagonalize tridiagonal (need eigenvectors for expectation value computation)
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
    
    if (ritz_values.empty()) {
        std::cerr << "Warning: Tridiagonal diagonalization failed" << std::endl;
        return;
    }
    
    expectation_values.resize(m);
    
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
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
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
    double log_step = (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);
    
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
        safe_system_call(cmd);
    }
    
    // Loop over samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state
        ComplexVector v0 = generateRandomVector(N, gen, dist);
        
        // Build Krylov subspace and compute expectation values
        std::vector<double> ritz_values, weights, expectation_values;
        compute_krylov_expectation_values(
            H, O, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            ritz_values, weights, expectation_values
        );
        
        uint64_t m = ritz_values.size();
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
                uint64_t gs_idx = std::distance(ritz_values.begin(), 
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
    uint64_t n_valid_samples = 0;
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
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
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
    double temp_step = (temp_max - temp_min) / std::max(uint64_t(1), num_temp_bins - 1);
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
        safe_system_call(cmd);
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        
        // Generate random initial state
        ComplexVector v0 = generateRandomVector(N, gen, dist);
        
        // Build Lanczos tridiagonal for Hamiltonian (store basis vectors)
        std::vector<double> alpha, beta;
        std::vector<ComplexVector> lanczos_vectors;
        
        uint64_t iterations = build_lanczos_tridiagonal_with_basis(
            H, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta, &lanczos_vectors
        );
        
        uint64_t m = alpha.size();
        std::cout << "  Lanczos iterations: " << m << std::endl;
        
        // Diagonalize tridiagonal (need eigenvectors for correlation computation)
        std::vector<double> ritz_values, weights;
        std::vector<double> evecs;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed" << std::endl;
            continue;
        }
        
        std::vector<double> correlation_values(m);
        
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
    uint64_t n_valid_samples = sample_expectations.size();
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
