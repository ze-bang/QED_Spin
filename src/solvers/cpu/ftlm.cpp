// ftlm.cpp - Finite Temperature Lanczos Method implementation
#include <ed/core/system_utils.h>
#include <ed/core/hdf5_io.h>       // For HDF5 output

#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <chrono>
#ifdef WITH_MPI
#include <mpi.h>
#endif

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
 * 
 * In FTLM, each sample approximates Tr[O exp(-βH)] / D where D is the Hilbert space dimension.
 * The weights w_i from the Lanczos decomposition satisfy Σ w_i = 1, not D.
 * 
 * This function stores both the derived thermodynamic quantities and the raw partition
 * function data (Z_sample, E_weighted, E2_weighted) needed for proper sample averaging.
 * 
 * For proper averaging: We must average Z_sample (not ln(Z_sample)) across samples
 * because <ln Z> ≠ ln<Z> (Jensen's inequality).
 */
ThermodynamicData compute_ftlm_thermodynamics(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures,
    uint64_t hilbert_dim
) {
    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    
    uint64_t n_temps = temperatures.size();
    uint64_t n_states = ritz_values.size();
    
    thermo.energy.resize(n_temps);
    thermo.specific_heat.resize(n_temps);
    thermo.entropy.resize(n_temps);
    thermo.free_energy.resize(n_temps);
    
    // Store raw data for proper averaging
    thermo.Z_sample.resize(n_temps);
    thermo.E_weighted.resize(n_temps);
    thermo.E2_weighted.resize(n_temps);
    
    // Find minimum energy for numerical stability
    double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
    thermo.e_min = e_min;
    
    // ln(D) contribution to entropy - this is crucial for proper normalization
    // If hilbert_dim = 0, skip this correction (backward compatibility)
    double ln_D = (hilbert_dim > 0) ? std::log(static_cast<double>(hilbert_dim)) : 0.0;
    
    for (int t = 0; t < n_temps; t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function and observables using shifted energies
        // Z_sample = Σ_i w_i * exp(-β * (E_i - E_min))
        // This Z_sample approximates Tr[exp(-β(H-E_min))]/D
        double Z_sample = 0.0;
        double E_weighted_sum = 0.0;
        double E2_weighted_sum = 0.0;
        
        // Compute Boltzmann-weighted sums
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            double boltz = weights[i] * std::exp(-beta * shifted_energy);
            Z_sample += boltz;
            E_weighted_sum += ritz_values[i] * boltz;
            E2_weighted_sum += ritz_values[i] * ritz_values[i] * boltz;
        }
        
        // Store raw values for averaging
        thermo.Z_sample[t] = Z_sample;
        thermo.E_weighted[t] = E_weighted_sum;
        thermo.E2_weighted[t] = E2_weighted_sum;
        
        // Compute derived quantities for this sample
        if (Z_sample > 1e-300) {
            double E_avg = E_weighted_sum / Z_sample;
            double E2_avg = E2_weighted_sum / Z_sample;
            
            // Thermodynamic quantities
            thermo.energy[t] = E_avg;
            thermo.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
            
            // Entropy: S = ln(Z_true) + β*E = ln(D) + ln(Z_sample) + β*(E - E_min)
            thermo.entropy[t] = ln_D + std::log(Z_sample) + beta * (E_avg - e_min);
            
            // Free energy: F = E - T*S = -T*ln(Z_true) = E_min - T*ln(D) - T*ln(Z_sample)
            thermo.free_energy[t] = e_min - T * ln_D - T * std::log(Z_sample);
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
 * 
 * NOTE: Energy and specific heat can be directly averaged since they are expectation values.
 * 
 * For entropy and free energy, we use the individual sample values which correctly
 * include the log(Z) contribution. Since each sample approximates the trace over the
 * full Hilbert space, we can directly average them. The FTLM entropy formula
 * S = β(E - e_min) + ln(Z) includes the proper normalization.
 * 
 * At high temperatures, this correctly approaches ln(D) where D is the Hilbert space 
 * dimension. At low temperatures, the entropy reflects the ground state degeneracy 
 * through the ln(Z) term.
 */
/**
 * @brief Average FTLM samples using proper partition function averaging
 * 
 * CRITICAL: Due to Jensen's inequality, <ln(Z)> ≤ ln(<Z>).
 * This causes a systematic bias when averaging entropy directly.
 * 
 * The correct approach is:
 *   1. Average the partition functions: <Z_sample>
 *   2. Average the weighted energy observables: <E_weighted>, <E2_weighted>
 *   3. Compute thermodynamics from the averaged quantities
 * 
 * Since each sample uses the same e_min (Lanczos converges to the same ground state),
 * we can average Z_sample directly.
 * 
 * The Hilbert space dimension D is extracted from the stored ln(D) in the entropy formula.
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
    
    // Check if we have raw partition function data
    bool have_Z_data = !sample_data[0].Z_sample.empty();
    
    if (have_Z_data) {
        // Proper averaging: average Z_sample first, then compute S, F
        
        // Find global minimum energy across all samples
        double e_min_global = sample_data[0].e_min;
        for (int s = 1; s < n_samples; s++) {
            e_min_global = std::min(e_min_global, sample_data[s].e_min);
        }
        
        // Extract ln(D) from the first sample's entropy formula at highest T
        // At high T: Z_sample → 1, E → <E>_uniform, β*(E-e_min) is small
        // S = ln(D) + ln(Z_sample) + β*(E - e_min)
        // So ln(D) = S - ln(Z_sample) - β*(E - e_min)
        // We average ln(D) over all samples for robustness
        double ln_D = 0.0;
        int t_high = n_temps - 1;  // Highest temperature for smallest β*(E-e_min)
        for (int s = 0; s < n_samples; s++) {
            double T = sample_data[s].temperatures[t_high];
            double beta = 1.0 / T;
            double S = sample_data[s].entropy[t_high];
            double Z_s = sample_data[s].Z_sample[t_high];
            double E_s = sample_data[s].energy[t_high];
            double e_min_s = sample_data[s].e_min;
            ln_D += S - std::log(Z_s) - beta * (E_s - e_min_s);
        }
        ln_D /= n_samples;
        
        // Average Z_sample, E_weighted, E2_weighted at each temperature
        std::vector<double> Z_avg(n_temps, 0.0);
        std::vector<double> E_weighted_avg(n_temps, 0.0);
        std::vector<double> E2_weighted_avg(n_temps, 0.0);
        
        for (int t = 0; t < n_temps; t++) {
            double T = sample_data[0].temperatures[t];
            double beta = 1.0 / T;
            
            for (int s = 0; s < n_samples; s++) {
                // Rescale Z_sample to common reference energy
                double delta_e = sample_data[s].e_min - e_min_global;
                double rescale = std::exp(-beta * delta_e);
                
                Z_avg[t] += sample_data[s].Z_sample[t] * rescale;
                E_weighted_avg[t] += sample_data[s].E_weighted[t] * rescale;
                E2_weighted_avg[t] += sample_data[s].E2_weighted[t] * rescale;
            }
            
            Z_avg[t] /= n_samples;
            E_weighted_avg[t] /= n_samples;
            E2_weighted_avg[t] /= n_samples;
        }
        
        // Compute thermodynamics from averaged quantities
        for (int t = 0; t < n_temps; t++) {
            double T = sample_data[0].temperatures[t];
            double beta = 1.0 / T;
            
            if (Z_avg[t] > 1e-300) {
                double E_avg = E_weighted_avg[t] / Z_avg[t];
                double E2_avg = E2_weighted_avg[t] / Z_avg[t];
                
                results.thermo_data.energy[t] = E_avg;
                results.thermo_data.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
                
                // S = ln(D) + ln(<Z_sample>) + β*(<E> - e_min_global)
                results.thermo_data.entropy[t] = ln_D + std::log(Z_avg[t]) + beta * (E_avg - e_min_global);
                
                // F = e_min_global - T*ln(D) - T*ln(<Z_sample>)
                results.thermo_data.free_energy[t] = e_min_global - T * ln_D - T * std::log(Z_avg[t]);
            } else {
                results.thermo_data.energy[t] = e_min_global;
                results.thermo_data.specific_heat[t] = 0.0;
                results.thermo_data.entropy[t] = 0.0;
                results.thermo_data.free_energy[t] = e_min_global;
            }
        }
        
        // Compute errors from variance in the raw quantities
        // Use jackknife-like variance estimation
        if (n_samples > 1) {
            for (int t = 0; t < n_temps; t++) {
                double sum_sq_e = 0.0, sum_sq_c = 0.0, sum_sq_s = 0.0, sum_sq_f = 0.0;
                
                for (int s = 0; s < n_samples; s++) {
                    double diff_e = sample_data[s].energy[t] - results.thermo_data.energy[t];
                    double diff_c = sample_data[s].specific_heat[t] - results.thermo_data.specific_heat[t];
                    double diff_s = sample_data[s].entropy[t] - results.thermo_data.entropy[t];
                    double diff_f = sample_data[s].free_energy[t] - results.thermo_data.free_energy[t];
                    
                    sum_sq_e += diff_e * diff_e;
                    sum_sq_c += diff_c * diff_c;
                    sum_sq_s += diff_s * diff_s;
                    sum_sq_f += diff_f * diff_f;
                }
                
                double norm = std::sqrt(static_cast<double>(n_samples * (n_samples - 1)));
                results.energy_error[t] = std::sqrt(sum_sq_e) / norm;
                results.specific_heat_error[t] = std::sqrt(sum_sq_c) / norm;
                results.entropy_error[t] = std::sqrt(sum_sq_s) / norm;
                results.free_energy_error[t] = std::sqrt(sum_sq_f) / norm;
            }
        }
    } else {
        // Fallback: direct averaging (backward compatibility, but biased for S and F)
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
            
            double norm = std::sqrt(static_cast<double>(n_samples * (n_samples - 1)));
            for (int t = 0; t < n_temps; t++) {
                results.energy_error[t] = std::sqrt(results.energy_error[t]) / norm;
                results.specific_heat_error[t] = std::sqrt(results.specific_heat_error[t]) / norm;
                results.entropy_error[t] = std::sqrt(results.entropy_error[t]) / norm;
                results.free_energy_error[t] = std::sqrt(results.free_energy_error[t]) / norm;
            }
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
        // Pass N (Hilbert space dimension) for proper entropy normalization
        ThermodynamicData sample_thermo = compute_ftlm_thermodynamics(
            ritz_values, weights, temperatures, N
        );
        sample_data.push_back(sample_thermo);
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMThermodynamicSample h5_sample;
            h5_sample.temperatures = temperatures;
            h5_sample.energy = sample_thermo.energy;
            h5_sample.specific_heat = sample_thermo.specific_heat;
            h5_sample.entropy = sample_thermo.entropy;
            h5_sample.free_energy = sample_thermo.free_energy;
            
            HDF5IO::saveFTLMThermodynamicSample(h5_file, sample, h5_sample);
            std::cout << "Saved FTLM sample " << sample << " to HDF5" << std::endl;
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
 * @brief Save FTLM results to HDF5 file and unified text format
 */
void save_ftlm_results(
    const FTLMResults& results,
    const std::string& filename
) {
    // Extract directory from filename to create HDF5 file
    std::string directory = filename.substr(0, filename.find_last_of('/'));
    if (directory.empty()) directory = ".";
    
    try {
        std::string h5_path = HDF5IO::createOrOpenFile(directory);
        
        HDF5IO::saveFTLMThermodynamics(
            h5_path,
            results.thermo_data.temperatures,
            results.thermo_data.energy,
            results.energy_error,
            results.thermo_data.specific_heat,
            results.specific_heat_error,
            results.thermo_data.entropy,
            results.entropy_error,
            results.thermo_data.free_energy,
            results.free_energy_error,
            results.total_samples,
            "FTLM"
        );
        
        std::cout << "FTLM results saved to: " << h5_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving FTLM results to HDF5: " << e.what() << std::endl;
    }
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
        // Z = Σ_i exp(-βE_i) where the sum is over Krylov states
        // The weights already contain the ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ matrix elements
        double Z = 0.0;
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            double boltzmann_factor = std::exp(-beta * shifted_energy);
            Z += boltzmann_factor;
        }
        
        // Apply thermal weights: w_n → w_n * exp(-βE_n) / Z
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double shifted_energy = ritz_values[i] - e_min;
                double boltzmann_factor = std::exp(-beta * shifted_energy);
                thermal_weights[i] = complex_weights[i] * (boltzmann_factor / Z);
            }
        } else {
            // Very low temperature - only ground state contributes
            thermal_weights.assign(n_states, Complex(0.0, 0.0));
            uint64_t gs_idx = std::distance(ritz_values.begin(),
                                      std::min_element(ritz_values.begin(), ritz_values.end()));
            thermal_weights[gs_idx] = complex_weights[gs_idx];
            // Normalize by complex magnitude
            Complex sum = Complex(0.0, 0.0);
            for (const auto& w : thermal_weights) sum += w;
            if (std::abs(sum) > 1e-300) {
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
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMDynamicalSample h5_sample;
            h5_sample.frequencies = results.frequencies;
            h5_sample.spectral_real = sample_spectrum;
            h5_sample.spectral_imag = std::vector<double>(sample_spectrum.size(), 0.0);  // Real for self-correlation
            
            HDF5IO::saveFTLMDynamicalSample(h5_file, sample, h5_sample, false);
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
 * @brief Save spectral function to text file in unified format
 * 
 * Unified format: 5 columns
 *   # Frequency  Re[S(ω)]  Im[S(ω)]  Re[Error]  Im[Error]
 * 
 * This provides consistent output across all spectral function methods:
/**
 * @brief Save dynamical response results to HDF5 file
 * 
 * Saves to HDF5 file: directory/ed_results.h5 under /dynamical/<operator_name>/
 */
void save_dynamical_response_results(
    const DynamicalResponseResults& results,
    const std::string& filename
) {
    // Extract directory from filename to create HDF5 file
    std::string directory = filename.substr(0, filename.find_last_of('/'));
    if (directory.empty()) directory = ".";
    
    // Extract operator name from filename (remove path and extension)
    std::string basename = filename.substr(filename.find_last_of('/') + 1);
    std::string operator_name = basename.substr(0, basename.find_last_of('.'));
    if (operator_name.empty()) operator_name = "dynamical_response";
    
    // Save to HDF5
    try {
        std::string h5_path = HDF5IO::createOrOpenFile(directory);
        
        // Prepare imaginary parts (use zeros if not provided)
        std::vector<double> spectral_imag = results.spectral_function_imag;
        if (spectral_imag.empty()) {
            spectral_imag.resize(results.frequencies.size(), 0.0);
        }
        
        std::vector<double> error_imag = results.spectral_error_imag;
        if (error_imag.empty()) {
            error_imag.resize(results.frequencies.size(), 0.0);
        }
        
        HDF5IO::saveDynamicalResponseFull(
            h5_path,
            operator_name,
            results.frequencies,
            results.spectral_function,
            spectral_imag,
            results.spectral_error,
            error_imag,
            results.total_samples,
            0.0  // temperature not stored in results struct
        );
        
        std::cout << "Dynamical response results saved to: " << h5_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving dynamical response to HDF5: " << e.what() << std::endl;
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
    const std::string& output_dir,
    double energy_shift
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
    
    // Storage for per-sample spectral functions (real and imaginary parts)
    std::vector<std::vector<double>> sample_spectra_real;
    std::vector<std::vector<double>> sample_spectra_imag;
    
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
        
        // For dynamical structure factors, shift energies so ground state is at E=0
        // This ensures spectral function has weight only at positive frequencies (excitation energies)
        if (sample == 0) {
            // Only print this message once for the first sample
            double E_shift;
            if (std::abs(energy_shift) > 1e-14) {
                // Use provided ground state energy shift
                E_shift = energy_shift;
                std::cout << "  Using provided ground state energy shift: " << E_shift << std::endl;
            } else {
                // Auto-detect from Krylov subspace (first sample only)
                E_shift = *std::min_element(ritz_values.begin(), ritz_values.end());
                std::cout << "  Ground state energy (auto-detected from Krylov): " << E_shift << std::endl;
            }
            std::cout << "  Shifting to excitation energies (E_gs = 0)" << std::endl;
        }
        
        // Apply energy shift for this sample
        double E_shift = (std::abs(energy_shift) > 1e-14) ? 
                         energy_shift : 
                         *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] -= E_shift;
        }
        
        // Compute weights ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ for cross-correlation
        // |n⟩ = Σ_j evecs[n,j] |v_j⟩ where |v_0⟩ = O₂|ψ⟩/||O₂|ψ⟩||
        // Need complex weights to preserve phase information
        
        // Apply O1 to original state
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
        
        // Compute spectral function for this sample (both real and imaginary parts)
        std::vector<double> sample_spectrum_real, sample_spectrum_imag;
        compute_spectral_function_complex(ritz_values, complex_weights, results.frequencies,
                                         params.broadening, temperature, 
                                         sample_spectrum_real, sample_spectrum_imag);
        
        sample_spectra_real.push_back(sample_spectrum_real);
        sample_spectra_imag.push_back(sample_spectrum_imag);
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMDynamicalSample h5_sample;
            h5_sample.frequencies = results.frequencies;
            h5_sample.spectral_real = sample_spectrum_real;
            h5_sample.spectral_imag = sample_spectrum_imag;
            
            HDF5IO::saveFTLMDynamicalSample(h5_file, sample, h5_sample, true);  // is_correlation=true
        }
    }
    
    // Average over all samples (Dynamical Correlation FTLM)
    uint64_t n_valid_samples = sample_spectra_real.size();
    std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    
    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_function_imag.resize(num_omega_bins, 0.0);
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained\n";
        return results;
    }
    
    // Compute mean (real and imaginary parts)
    for (int s = 0; s < n_valid_samples; s++) {
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_function[i] += sample_spectra_real[s][i];
            results.spectral_function_imag[i] += sample_spectra_imag[s][i];
        }
    }
    
    for (int i = 0; i < num_omega_bins; i++) {
        results.spectral_function[i] /= n_valid_samples;
        results.spectral_function_imag[i] /= n_valid_samples;
    }
    
    // Compute standard error (real and imaginary parts)
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int i = 0; i < num_omega_bins; i++) {
                double diff_real = sample_spectra_real[s][i] - results.spectral_function[i];
                double diff_imag = sample_spectra_imag[s][i] - results.spectral_function_imag[i];
                results.spectral_error[i] += diff_real * diff_real;
                results.spectral_error_imag[i] += diff_imag * diff_imag;
            }
        }
        
        double norm_factor = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_error[i] = std::sqrt(results.spectral_error[i]) / norm_factor;
            results.spectral_error_imag[i] = std::sqrt(results.spectral_error_imag[i]) / norm_factor;
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
 * @brief MEMORY-EFFICIENT spectral function via continued fraction (O1=O2 case)
 * 
 * This version DOES NOT store Lanczos basis vectors, making it suitable for
 * very large Hilbert spaces (>16M states).
 * 
 * Memory: O(N) instead of O(krylov_dim × N)
 */
DynamicalResponseResults compute_dynamical_correlation_state_cf(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double energy_shift
) {
    std::cout << "\n==========================================\n";
    std::cout << "Spectral Function via Continued Fraction (Memory-Efficient)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Memory mode: NO BASIS STORAGE (O(N) memory)" << std::endl;
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
    std::cout << "Broadening: " << params.broadening << std::endl;
    
    DynamicalResponseResults results;
    results.total_samples = 1;
    
    // Generate frequency grid
    results.frequencies.resize(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (size_t i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
    }
    
    // Verify state is normalized
    double state_norm = cblas_dznrm2(N, state.data(), 1);
    if (std::abs(state_norm - 1.0) > 1e-10) {
        std::cout << "  Warning: Input state norm = " << state_norm << " (expected 1.0)\n";
    }
    
    ComplexVector psi = state;
    Complex scale(1.0/state_norm, 0.0);
    cblas_zscal(N, &scale, psi.data(), 1);
    
    // Apply operator O: |φ⟩ = O|ψ⟩
    ComplexVector phi(N);
    O(psi.data(), phi.data(), N);
    
    // Get norm of |φ⟩ = ||O|ψ⟩||
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    double phi_norm_sq = phi_norm * phi_norm;
    
    if (phi_norm < 1e-14) {
        std::cerr << "  Error: O|ψ⟩ has zero norm\n";
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        return results;
    }
    
    std::cout << "  Norm of O|ψ⟩: " << phi_norm << std::endl;
    std::cout << "  ||O|ψ⟩||² = " << phi_norm_sq << std::endl;
    
    // Normalize |φ⟩ for Lanczos
    Complex phi_scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &phi_scale, phi.data(), 1);
    
    // Build Lanczos tridiagonal WITHOUT storing basis vectors (memory-efficient!)
    std::vector<double> alpha, beta;
    uint64_t iterations = build_lanczos_tridiagonal(
        H, phi, N, params.krylov_dim, params.tolerance,
        false, 0,  // No reorthogonalization, no basis storage
        alpha, beta
    );
    
    uint64_t m = alpha.size();
    std::cout << "  Lanczos iterations: " << m << std::endl;
    
    // Shift energies by ground state energy
    double E_shift = energy_shift;
    if (std::abs(E_shift) < 1e-14) {
        // Auto-detect from tridiagonal minimum eigenvalue
        std::vector<double> diag_copy = alpha;
        std::vector<double> offdiag_copy(m > 1 ? m - 1 : 1);
        for (size_t i = 0; i < m - 1; i++) {
            offdiag_copy[i] = beta[i + 1];
        }
        int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m,
                                 diag_copy.data(), offdiag_copy.data(),
                                 nullptr, 1);
        if (info == 0 && !diag_copy.empty()) {
            E_shift = diag_copy[0];  // Smallest eigenvalue
        }
        std::cout << "  Ground state energy (auto-detected): " << E_shift << std::endl;
    } else {
        std::cout << "  Using provided ground state energy: " << E_shift << std::endl;
    }
    
    // Shift alpha values
    for (size_t i = 0; i < m; i++) {
        alpha[i] -= E_shift;
    }
    
    // Compute spectral function via continued fraction
    results.spectral_function = continued_fraction_spectral_function(
        alpha, beta, results.frequencies, params.broadening, phi_norm_sq
    );
    
    // Imaginary part is zero for self-correlation
    results.spectral_function_imag.resize(num_omega_bins, 0.0);
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
    std::cout << "\n==========================================\n";
    std::cout << "Continued Fraction Spectral Complete\n";
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
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMStaticSample h5_sample;
            h5_sample.temperatures = results.temperatures;
            h5_sample.expectation = sample_expectations[sample];
            h5_sample.variance = sample_variances[sample];
            
            HDF5IO::saveFTLMStaticSample(h5_file, sample, h5_sample);
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
 * @brief Save static response to text file in unified format
 * 
 * Unified format: 8 columns
 *   # T  <O>  <O>_err  Var  Var_err  chi  chi_err  N_samples
 * 
/**
 * @brief Save static response results to HDF5 file
 * 
 * Saves to HDF5 file: directory/ed_results.h5 under /correlations/<operator_name>/
 */
void save_static_response_results(
    const StaticResponseResults& results,
    const std::string& filename
) {
    // Extract directory from filename to create HDF5 file
    std::string directory = filename.substr(0, filename.find_last_of('/'));
    if (directory.empty()) directory = ".";
    
    // Extract operator name from filename (remove path and extension)
    std::string basename = filename.substr(filename.find_last_of('/') + 1);
    std::string operator_name = basename.substr(0, basename.find_last_of('.'));
    if (operator_name.empty()) operator_name = "static_response";
    
    // Save to HDF5
    try {
        std::string h5_path = HDF5IO::createOrOpenFile(directory);
        
        HDF5IO::saveStaticResponse(
            h5_path,
            operator_name,
            results.temperatures,
            results.expectation,
            results.expectation_error,
            results.variance,
            results.variance_error,
            results.susceptibility,
            results.susceptibility_error,
            results.total_samples
        );
        
        std::cout << "Static response results saved to: " << h5_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error saving static response to HDF5: " << e.what() << std::endl;
    }
}

// ============================================================================
// TEMPERATURE-INDEPENDENT SPECTRAL DECOMPOSITION (OPTIMIZATION)
// ============================================================================

/**
 * @brief Compute temperature-independent spectral decomposition via Lanczos
 * 
 * This function runs the Lanczos iteration once to compute the spectral
 * decomposition (eigenvalues and weights) which is temperature-independent.
 * The results can then be reused to efficiently compute S(ω,T) at multiple
 * temperatures without re-running the expensive Lanczos iteration.
 */
LanczosSpectralData compute_lanczos_spectral_data(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double energy_shift
) {
    std::cout << "\n==========================================\n";
    std::cout << "Computing Temperature-Independent Spectral Data\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Broadening: " << params.broadening << std::endl;
    
    LanczosSpectralData spectral_data;
    
    // Verify state is normalized
    double state_norm = cblas_dznrm2(N, state.data(), 1);
    ComplexVector psi = state;
    if (std::abs(state_norm - 1.0) > 1e-10) {
        std::cout << "  Normalizing input state (norm = " << state_norm << ")\n";
        Complex scale(1.0/state_norm, 0.0);
        cblas_zscal(N, &scale, psi.data(), 1);
    }
    
    // Apply operator O2: |φ⟩ = O₂|ψ⟩
    ComplexVector phi(N);
    O2(psi.data(), phi.data(), N);
    
    // Get norm of |φ⟩
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    if (phi_norm < 1e-14) {
        std::cerr << "  Error: O₂|ψ⟩ has zero norm\n";
        return spectral_data;
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
    
    spectral_data.krylov_dim = m;
    spectral_data.lanczos_iterations = iterations;
    
    // Diagonalize tridiagonal (need eigenvectors for weight computation)
    std::vector<double> ritz_values, dummy_weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, dummy_weights, &evecs);
    
    if (ritz_values.empty()) {
        std::cerr << "  Error: Tridiagonal diagonalization failed\n";
        return spectral_data;
    }
    
    // Determine and apply energy shift
    double E_shift;
    if (std::abs(energy_shift) > 1e-14) {
        E_shift = energy_shift;
        std::cout << "  Using provided ground state energy shift: " << E_shift << std::endl;
    } else {
        E_shift = *std::min_element(ritz_values.begin(), ritz_values.end());
        std::cout << "  Ground state energy (auto-detected from Krylov): " << E_shift << std::endl;
    }
    
    spectral_data.ground_state_energy = E_shift;
    
    // Shift to excitation energies
    for (int i = 0; i < m; i++) {
        ritz_values[i] -= E_shift;
    }
    spectral_data.ritz_values = ritz_values;
    
    std::cout << "  Shifted to excitation energies (E_gs = 0)" << std::endl;
    std::cout << "  Energy range: [" << *std::min_element(ritz_values.begin(), ritz_values.end())
              << ", " << *std::max_element(ritz_values.begin(), ritz_values.end()) << "]" << std::endl;
    
    // Compute temperature-independent spectral weights
    // w_n = ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
    
    // Apply O1 to original state: |O₁ψ⟩
    ComplexVector O1_psi(N);
    O1(psi.data(), O1_psi.data(), N);
    
    spectral_data.spectral_weights.resize(m);
    
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
        
        // Weight is ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
        Complex weight_complex = std::conj(overlap_O1) * overlap_O2;
        spectral_data.spectral_weights[n] = weight_complex;
    }
    
    std::cout << "==========================================\n";
    std::cout << "Spectral Data Computation Complete\n";
    std::cout << "==========================================\n";
    
    return spectral_data;
}

/**
 * @brief Compute spectral function from Lanczos data for multiple temperatures
 * 
 * This function takes pre-computed spectral data and efficiently computes
 * S(ω,T) for multiple temperatures. This is much faster than re-running
 * Lanczos for each temperature.
 */
std::map<double, DynamicalResponseResults> compute_spectral_function_from_lanczos_data(
    const LanczosSpectralData& spectral_data,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double broadening,
    uint64_t num_samples,
    const std::vector<std::vector<Complex>>* per_sample_weights
) {
    std::cout << "\n==========================================\n";
    std::cout << "Computing Spectral Functions for Multiple Temperatures\n";
    std::cout << "==========================================\n";
    std::cout << "Number of temperatures: " << temperatures.size() << std::endl;
    std::cout << "Temperature range: [" << *std::min_element(temperatures.begin(), temperatures.end())
              << ", " << *std::max_element(temperatures.begin(), temperatures.end()) << "]" << std::endl;
    std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
    std::cout << "Broadening: " << broadening << std::endl;
    
    std::map<double, DynamicalResponseResults> results_map;
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    const auto& ritz_values = spectral_data.ritz_values;
    const auto& weights = spectral_data.spectral_weights;
    uint64_t m = ritz_values.size();
    
    // Compute spectral function for each temperature
    for (double T : temperatures) {
        std::cout << "  Computing for T = " << T << " ..." << std::endl;
        
        DynamicalResponseResults results;
        results.frequencies = frequencies;
        results.total_samples = num_samples;
        results.omega_min = omega_min;
        results.omega_max = omega_max;
        
        // Initialize spectral function arrays
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        
        // Compute partition function and thermal weights
        double beta = 1.0 / T;
        std::vector<double> thermal_weights(m);
        double Z = 0.0;
        
        // Find minimum energy for numerical stability
        double E_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int n = 0; n < m; n++) {
            double shifted_energy = ritz_values[n] - E_min;
            double boltzmann = std::exp(-beta * shifted_energy);
            thermal_weights[n] = boltzmann;
            Z += boltzmann;
        }
        
        // Normalize thermal weights
        if (Z > 0.0) {
            for (int n = 0; n < m; n++) {
                thermal_weights[n] /= Z;
            }
        }
        
        // Compute spectral function at each frequency
        double eta = broadening;
        double norm_factor = eta / M_PI;
        
        for (int i = 0; i < num_omega_bins; i++) {
            double omega = frequencies[i];
            Complex S_omega(0.0, 0.0);
            
            for (int n = 0; n < m; n++) {
                double E_n = ritz_values[n];
                double lorentzian = norm_factor / ((omega - E_n) * (omega - E_n) + eta * eta);
                
                // Include thermal weight and spectral weight
                S_omega += weights[n] * lorentzian * thermal_weights[n];
            }
            
            results.spectral_function[i] = std::real(S_omega);
            results.spectral_function_imag[i] = std::imag(S_omega);
        }
        
        // Compute error bars if per-sample data is available
        if (per_sample_weights && num_samples > 1) {
            std::vector<std::vector<double>> per_sample_spectral_real(num_samples, std::vector<double>(num_omega_bins, 0.0));
            std::vector<std::vector<double>> per_sample_spectral_imag(num_samples, std::vector<double>(num_omega_bins, 0.0));
            
            // Compute spectral function for each sample
            for (uint64_t s = 0; s < num_samples && s < per_sample_weights->size(); s++) {
                const auto& sample_weights = (*per_sample_weights)[s];
                
                for (int i = 0; i < num_omega_bins; i++) {
                    double omega = frequencies[i];
                    Complex S_omega_sample(0.0, 0.0);
                    
                    for (int n = 0; n < m && n < sample_weights.size(); n++) {
                        double E_n = ritz_values[n];
                        double lorentzian = norm_factor / ((omega - E_n) * (omega - E_n) + eta * eta);
                        S_omega_sample += sample_weights[n] * lorentzian * thermal_weights[n];
                    }
                    
                    per_sample_spectral_real[s][i] = std::real(S_omega_sample);
                    per_sample_spectral_imag[s][i] = std::imag(S_omega_sample);
                }
            }
            
            // Compute standard error: SE = sqrt(variance / num_samples)
            for (int i = 0; i < num_omega_bins; i++) {
                double mean_real = results.spectral_function[i];
                double mean_imag = results.spectral_function_imag[i];
                double var_real = 0.0, var_imag = 0.0;
                
                for (uint64_t s = 0; s < num_samples && s < per_sample_weights->size(); s++) {
                    double diff_real = per_sample_spectral_real[s][i] - mean_real;
                    double diff_imag = per_sample_spectral_imag[s][i] - mean_imag;
                    var_real += diff_real * diff_real;
                    var_imag += diff_imag * diff_imag;
                }
                
                // Standard error of the mean
                results.spectral_error[i] = std::sqrt(var_real / (num_samples * (num_samples - 1)));
                results.spectral_error_imag[i] = std::sqrt(var_imag / (num_samples * (num_samples - 1)));
            }
        }
        
        results_map[T] = results;
    }
    
    std::cout << "==========================================\n";
    std::cout << "Multi-Temperature Spectral Function Complete\n";
    std::cout << "==========================================\n";
    
    return results_map;
}

/**
 * @brief Optimized version for computing dynamical correlation at multiple temperatures
 * 
 * This combines compute_lanczos_spectral_data and compute_spectral_function_from_lanczos_data
 * into a single convenient function for temperature scans.
 */
std::map<double, DynamicalResponseResults> compute_dynamical_correlation_state_multi_temperature(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift
) {
    std::cout << "\n=========================================="  << std::endl;
    std::cout << "OPTIMIZED MULTI-TEMPERATURE DYNAMICAL CORRELATION" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Running Lanczos ONCE for " << temperatures.size() << " temperature points" << std::endl;
    std::cout << "This is much more efficient than running Lanczos " << temperatures.size() << " times!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Step 1: Compute temperature-independent spectral data (Lanczos run)
    LanczosSpectralData spectral_data = compute_lanczos_spectral_data(
        H, O1, O2, state, N, params, energy_shift
    );
    
    if (spectral_data.ritz_values.empty()) {
        std::cerr << "Error: Failed to compute spectral data\n";
        return {};
    }
    
    // Step 2: Compute spectral functions for all temperatures (fast!)
    return compute_spectral_function_from_lanczos_data(
        spectral_data, omega_min, omega_max, num_omega_bins,
        temperatures, params.broadening, 1
    );
}

// ============================================================================
// CORRECTED FTLM MULTI-SAMPLE MULTI-TEMPERATURE SPECTRAL FUNCTION
// ============================================================================

/**
 * @brief Multi-sample multi-temperature dynamical correlation (CORRECTED FTLM!)
 * 
 * CORRECTED VERSION: This implementation properly handles thermal averaging
 * for FTLM spectral functions. The key insight is that for random state 
 * sampling, we don't apply Boltzmann weights to the spectral peaks directly.
 * Instead, the random sampling itself provides the thermal averaging through
 * the trace identity: Tr[A] = N × E_r[⟨r|A|r⟩] where |r⟩ are random states.
 * 
 * For T→0 (low temperature limit), the spectral function should match the
 * ground state result. This is achieved because random states have some
 * overlap with the ground state.
 * 
 * For finite T, the formula becomes:
 *   S(ω,T) ∝ Tr[e^{-βH} O†δ(ω-H+E₀)O] / Z
 *          = (N/Z) × E_r[⟨r|e^{-βH} O†δ(ω-H+E₀)O|r⟩]
 * 
 * The key correction is to not double-apply thermal weights. The spectral
 * weights already capture the transition amplitudes - we only need the
 * thermal prefactor from the partition function.
 */
std::map<double, DynamicalResponseResults> compute_dynamical_correlation_multi_sample_multi_temperature(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir
) {
    std::cout << "\n=========================================="  << std::endl;
    std::cout << "FTLM SPECTRAL FUNCTION (CORRECT FORMULATION)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Samples: " << params.num_samples << std::endl;
    std::cout << "Temperatures: " << temperatures.size() << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Broadening: " << params.broadening << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "\nUsing correct FTLM formulation:" << std::endl;
    std::cout << "  S(ω,T) = (N/Z) × Σ_r Σ_i e^{-βε_i} |c_i|² S_i(ω)" << std::endl;
    std::cout << "  where S_i(ω) is computed via continued fraction" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Ground state energy for shifting
    double E_gs = energy_shift;
    if (std::abs(E_gs) < 1e-14) {
        std::cout << "\nDetermining ground state energy from Lanczos...\n";
        ComplexVector test_state(N);
        for (uint64_t i = 0; i < N; i++) {
            test_state[i] = Complex(dist(gen), dist(gen));
        }
        double norm = cblas_dznrm2(N, test_state.data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, test_state.data(), 1);
        
        std::vector<double> alpha, beta;
        build_lanczos_tridiagonal(H, test_state, N, std::min(params.krylov_dim, (uint64_t)100),
                                  params.tolerance, false, 10, alpha, beta);
        
        std::vector<double> ritz_vals, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_vals, weights);
        
        if (!ritz_vals.empty()) {
            E_gs = *std::min_element(ritz_vals.begin(), ritz_vals.end());
            std::cout << "Ground state energy (estimated): " << E_gs << std::endl;
        }
    } else {
        std::cout << "Using provided ground state energy: " << E_gs << std::endl;
    }
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (uint64_t i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    // For each temperature, accumulate numerator (Σ_r Σ_i e^{-βε_i} |c_i|² S_i(ω))
    // and partition function (Σ_r Σ_i e^{-βε_i} |c_i|²)
    std::map<double, std::vector<double>> accumulated_spectral;
    std::map<double, double> accumulated_Z;
    std::map<double, std::vector<std::vector<double>>> per_sample_spectral;  // For error estimation
    
    for (double T : temperatures) {
        accumulated_spectral[T] = std::vector<double>(num_omega_bins, 0.0);
        accumulated_Z[T] = 0.0;
        per_sample_spectral[T] = std::vector<std::vector<double>>();
    }
    
    // MPI parallelization: distribute samples across ranks
    int mpi_rank = 0, mpi_size = 1;
#ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif
    
    // Calculate sample distribution for this rank
    uint64_t samples_per_rank = params.num_samples / mpi_size;
    uint64_t remainder = params.num_samples % mpi_size;
    uint64_t start_sample = mpi_rank * samples_per_rank + std::min((uint64_t)mpi_rank, remainder);
    uint64_t end_sample = start_sample + samples_per_rank + (mpi_rank < (int)remainder ? 1 : 0);
    uint64_t local_num_samples = end_sample - start_sample;
    
    if (mpi_rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "FTLM Spectral Function\n";
        std::cout << "==========================================\n";
#ifdef WITH_MPI
        std::cout << "Total MPI ranks: " << mpi_size << "\n";
#endif
        std::cout << "Total samples: " << params.num_samples << "\n";
#ifdef WITH_MPI
        std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
#endif
        std::cout << "==========================================\n";
    }
    
#ifdef WITH_MPI
    std::cout << "Rank " << mpi_rank << " processing samples [" 
              << start_sample << ", " << end_sample << ") - " << local_num_samples << " samples\n";
    
    // Synchronize before starting
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        std::cout << "\nStarting parallel sample processing across " << mpi_size << " ranks...\n";
        std::cout << "(Only rank 0 output shown for clarity)\n" << std::endl;
    }
#endif
    
    // How many Ritz states to use per sample for spectral function
    // Using all states is expensive; use states with significant thermal weight
    uint64_t max_ritz_states = std::min(params.krylov_dim, (uint64_t)50);  // Limit for efficiency
    
    // Pre-allocate working vectors to avoid repeated allocations in inner loop
    ComplexVector psi_work(N);  // For eigenstate construction
    ComplexVector phi_work(N);  // For O|psi>
    
#ifdef WITH_MPI
    double start_time = MPI_Wtime();
#else
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    
    // Loop over random samples assigned to this rank
    for (uint64_t sample_idx = start_sample; sample_idx < end_sample; sample_idx++) {
        if (mpi_rank == 0) {
            uint64_t local_idx = sample_idx - start_sample + 1;
            std::cout << "\n--- Rank 0: Sample " << local_idx << "/" << local_num_samples 
                      << " (Global: " << (sample_idx + 1) << "/" << params.num_samples << ") ---\n";
        }
        
        // Seed RNG deterministically based on sample index (not rank) for reproducibility
        std::mt19937 sample_gen(params.random_seed + sample_idx * 12345);
        
        // Generate random state |r⟩
        ComplexVector r_state(N);
        for (uint64_t i = 0; i < N; i++) {
            r_state[i] = Complex(dist(sample_gen), dist(sample_gen));
        }
        double r_norm = cblas_dznrm2(N, r_state.data(), 1);
        Complex r_scale(1.0/r_norm, 0.0);
        cblas_zscal(N, &r_scale, r_state.data(), 1);
        
        // Step 1: Build Lanczos from |r⟩ to get approximate eigenstates
        std::vector<double> alpha_H, beta_H;
        std::vector<ComplexVector> lanczos_vectors;
        
        uint64_t H_iterations = build_lanczos_tridiagonal_with_basis(
            H, r_state, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha_H, beta_H, &lanczos_vectors
        );
        
        uint64_t m_H = alpha_H.size();
        if (mpi_rank == 0) {
            std::cout << "  Hamiltonian Lanczos: " << m_H << " iterations\n";
        }
        
        if (m_H == 0) {
            std::cerr << "  Warning: Lanczos failed, skipping sample\n";
            continue;
        }
        
        // Diagonalize tridiagonal to get Ritz values and vectors
        std::vector<double> ritz_values;
        std::vector<double> dummy_weights;
        std::vector<double> evecs;  // Row-major: evecs[i*m_H + j] = V[i,j]
        diagonalize_tridiagonal_ritz(alpha_H, beta_H, ritz_values, dummy_weights, &evecs);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Diagonalization failed, skipping sample\n";
            continue;
        }
        
        // Compute |c_i|² = |⟨ψ_i|r⟩|² = V[i,0]² (first Lanczos vector is |r⟩)
        std::vector<double> c_sq(m_H);
        for (uint64_t i = 0; i < m_H; i++) {
            c_sq[i] = evecs[i * m_H + 0] * evecs[i * m_H + 0];
        }
        
        // Find minimum energy for numerical stability
        double E_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        if (mpi_rank == 0 && m_H > 0 && !ritz_values.empty()) {
            std::cout << "  Ritz values range: [" << *std::min_element(ritz_values.begin(), ritz_values.end()) 
                      << ", " << *std::max_element(ritz_values.begin(), ritz_values.end()) << "]\n";
        }
        
        // ============================================================
        // OPTIMIZATION: Precompute spectral functions for Ritz states
        // The Lanczos expansion and continued fraction are temperature-
        // independent, so we compute S_i(ω) once and reuse across all T
        // ============================================================
        
        // Step 2a: Determine which Ritz states are significant for ANY temperature
        // Use the highest temperature (smallest beta) for most inclusive threshold
        double T_max_local = *std::max_element(temperatures.begin(), temperatures.end());
        double beta_min = 1.0 / T_max_local;
        
        // Compute thermal weights at highest T to find potentially significant states
        std::vector<double> max_weights(m_H);
        double Z_max = 0.0;
        for (uint64_t i = 0; i < m_H; i++) {
            double boltzmann = std::exp(-beta_min * (ritz_values[i] - E_min));
            max_weights[i] = c_sq[i] * boltzmann;
            Z_max += max_weights[i];
        }
        
        // Identify significant Ritz states (union across all temperatures)
        double weight_threshold = 1e-10 * Z_max;  // Use looser threshold to catch all
        std::vector<uint64_t> significant_states;
        significant_states.reserve(max_ritz_states);
        
        for (uint64_t i = 0; i < std::min(m_H, max_ritz_states); i++) {
            if (max_weights[i] >= weight_threshold || c_sq[i] > 1e-12) {
                significant_states.push_back(i);
            }
        }
        
        if (mpi_rank == 0) {
            std::cout << "  Identified " << significant_states.size() << " potentially significant Ritz states\n";
        }
        
        // Step 2b: Precompute S_i(ω) for each significant Ritz state
        // This is the expensive part - Lanczos + continued fraction per state
        std::vector<std::vector<double>> precomputed_S_i(significant_states.size());
        std::vector<double> precomputed_energies(significant_states.size());
        std::vector<double> precomputed_c_sq(significant_states.size());
        std::vector<bool> state_valid(significant_states.size(), false);
        
        // OpenMP parallelization over Ritz states (most expensive loop)
        #pragma omp parallel for schedule(dynamic) 
        for (size_t idx = 0; idx < significant_states.size(); idx++) {
            uint64_t i = significant_states[idx];
            
            // Thread-local working vectors
            ComplexVector psi_local(N, Complex(0.0, 0.0));
            ComplexVector phi_local(N);
            
            // Construct approximate eigenstate |ψ_i⟩ = Σ_j V[i,j] |v_j⟩
            for (uint64_t j = 0; j < m_H; j++) {
                double coeff = evecs[i * m_H + j];
                Complex coeff_c(coeff, 0.0);
                cblas_zaxpy(N, &coeff_c, lanczos_vectors[j].data(), 1, psi_local.data(), 1);
            }
            
            // Normalize
            double psi_norm = cblas_dznrm2(N, psi_local.data(), 1);
            if (psi_norm < 1e-14) continue;
            Complex psi_scale(1.0/psi_norm, 0.0);
            cblas_zscal(N, &psi_scale, psi_local.data(), 1);
            
            // Apply operator O2: |φ_i⟩ = O₂|ψ_i⟩
            O2(psi_local.data(), phi_local.data(), N);
            
            double phi_norm = cblas_dznrm2(N, phi_local.data(), 1);
            double phi_norm_sq = phi_norm * phi_norm;
            
            if (phi_norm < 1e-14) continue;
            
            // Normalize for Lanczos
            Complex phi_scale(1.0/phi_norm, 0.0);
            cblas_zscal(N, &phi_scale, phi_local.data(), 1);
            
            // Build Lanczos from |φ_i⟩
            std::vector<double> alpha_S, beta_S;
            build_lanczos_tridiagonal(
                H, phi_local, N, params.krylov_dim, params.tolerance,
                params.full_reorthogonalization, params.reorth_frequency,
                alpha_S, beta_S
            );
            
            if (alpha_S.empty()) continue;
            
            // Shift energies by E_gs
            for (size_t k = 0; k < alpha_S.size(); k++) {
                alpha_S[k] -= E_gs;
            }
            
            // Compute spectral function via continued fraction (ONCE per state)
            std::vector<double> S_i = continued_fraction_spectral_function(
                alpha_S, beta_S, frequencies, params.broadening, phi_norm_sq
            );
            
            // Store precomputed results (thread-safe since each idx is unique)
            precomputed_S_i[idx] = std::move(S_i);
            precomputed_energies[idx] = ritz_values[i];
            precomputed_c_sq[idx] = c_sq[i];
            state_valid[idx] = true;
        }
        
        // Count valid states
        uint64_t n_valid = 0;
        for (size_t idx = 0; idx < significant_states.size(); idx++) {
            if (state_valid[idx]) n_valid++;
        }
        if (mpi_rank == 0) {
            std::cout << "  Precomputed spectral functions for " << n_valid << " Ritz states\n";
        }
        
        // Step 3: For each temperature, apply thermal weights to precomputed spectra
        // This is now O(num_temps × num_states × num_omega) - no Lanczos!
        for (double T : temperatures) {
            double beta = 1.0 / T;
            
            // Compute partition function contribution for this sample
            double Z_sample = 0.0;
            for (size_t idx = 0; idx < significant_states.size(); idx++) {
                if (!state_valid[idx]) continue;
                double boltzmann = std::exp(-beta * (precomputed_energies[idx] - E_min));
                Z_sample += precomputed_c_sq[idx] * boltzmann;
            }
            
            accumulated_Z[T] += Z_sample;
            
            // Accumulate weighted spectral contributions
            std::vector<double> sample_spectral(num_omega_bins, 0.0);
            
            for (size_t idx = 0; idx < significant_states.size(); idx++) {
                if (!state_valid[idx]) continue;
                
                double boltzmann = std::exp(-beta * (precomputed_energies[idx] - E_min));
                double thermal_weight = precomputed_c_sq[idx] * boltzmann;
                
                // Skip if negligible for this temperature
                if (thermal_weight < 1e-14 * Z_sample) continue;
                
                // Add contribution (vectorized)
                const std::vector<double>& S_i = precomputed_S_i[idx];
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    double contrib = thermal_weight * S_i[iw];
                    sample_spectral[iw] += contrib;
                    accumulated_spectral[T][iw] += contrib;
                }
            }
            
            // Store sample contribution for error estimation
            if (Z_sample > 1e-300) {
                std::vector<double> normalized_sample(num_omega_bins);
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    normalized_sample[iw] = sample_spectral[iw] / Z_sample;
                }
                per_sample_spectral[T].push_back(normalized_sample);
            }
        }
        
        if (mpi_rank == 0) {
            std::cout << "  Applied thermal weights for " << temperatures.size() << " temperatures\n";
        }
    }
    
    // Report timing for this rank
#ifdef WITH_MPI
    double elapsed_time = MPI_Wtime() - start_time;
#else
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
#endif
    
    if (mpi_rank == 0) {
        std::cout << "\nCompleted " << local_num_samples << " samples in " 
                  << elapsed_time << " seconds (" << (elapsed_time / local_num_samples) 
                  << " s/sample)\n";
    }
    
#ifdef WITH_MPI
    // MPI Reduce: gather accumulated results from all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        std::cout << "\n--- Gathering results from all MPI ranks ---\n";
        std::cout << "All ranks have completed their sample processing.\n";
    }
    
    // Reduce accumulated_spectral and accumulated_Z across all ranks
    for (double T : temperatures) {
        std::vector<double> global_spectral(num_omega_bins, 0.0);
        double global_Z = 0.0;
        
        MPI_Reduce(accumulated_spectral[T].data(), global_spectral.data(), 
                   num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&accumulated_Z[T], &global_Z, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // Only rank 0 needs the final values
        if (mpi_rank == 0) {
            accumulated_spectral[T] = global_spectral;
            accumulated_Z[T] = global_Z;
        }
    }
    
    // Gather total sample count for error estimation
    uint64_t global_total_samples = 0;
    MPI_Reduce(&local_num_samples, &global_total_samples, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    uint64_t global_total_samples = local_num_samples;
#endif
    
    // Compute final results: S(ω) = N × (Σ accumulated_spectral) / (Σ accumulated_Z)
    if (mpi_rank == 0) {
        std::cout << "\n--- Computing final results ---\n";
    }
    
    std::map<double, DynamicalResponseResults> results_map;
    
    for (double T : temperatures) {
        DynamicalResponseResults results;
        results.frequencies = frequencies;
        results.omega_min = omega_min;
        results.omega_max = omega_max;
        results.total_samples = (mpi_rank == 0) ? global_total_samples : local_num_samples;
        
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        
        double Z_total = accumulated_Z[T];
        if (mpi_rank == 0 && Z_total < 1e-300) {
            std::cerr << "  Warning: Z ≈ 0 for T = " << T << std::endl;
            results_map[T] = results;
            continue;
        }
        
        // Only rank 0 computes final results
        if (mpi_rank != 0) {
            results_map[T] = results;
            continue;
        }
        
        // Compute spectral function: S(ω) = accumulated_spectral / Z_total
        // Note: The trace sampling identity Tr[A] = N × E_r[⟨r|A|r⟩] means we should
        // multiply by N, but in FTLM the ratio of accumulated sums automatically gives
        // the correct thermal average without an additional factor.
        // 
        // accumulated_spectral = Σ_r Σ_i e^{-βε_i} |c_i|² S_i(ω)
        // accumulated_Z = Σ_r Σ_i e^{-βε_i} |c_i|²
        // 
        // The ratio gives: S(ω,T) = ⟨O† δ(ω-H+E₀) O⟩_β 
        // which is the thermal expectation value as desired.
        for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
            results.spectral_function[iw] = accumulated_spectral[T][iw] / Z_total;
        }
        
        // Note: Error estimation with MPI requires gathering per-sample data from all ranks
        // For now, skip detailed error estimation in MPI mode (use simpler estimate)
        uint64_t n_samples = per_sample_spectral[T].size();
        if (n_samples > 1 && mpi_size == 1) {
            // Compute mean of per-sample normalized spectra (serial only)
            std::vector<double> mean(num_omega_bins, 0.0);
            for (uint64_t s = 0; s < n_samples; s++) {
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    mean[iw] += per_sample_spectral[T][s][iw];
                }
            }
            for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                mean[iw] /= n_samples;
            }
            
            // Compute variance and standard error
            for (uint64_t s = 0; s < n_samples; s++) {
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    double diff = per_sample_spectral[T][s][iw] - mean[iw];
                    results.spectral_error[iw] += diff * diff;
                }
            }
            
            double norm = std::sqrt(static_cast<double>(n_samples * (n_samples - 1)));
            for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                results.spectral_error[iw] = std::sqrt(results.spectral_error[iw]) / norm;
            }
        }
        
        std::cout << "  T = " << T << ": " << global_total_samples << " samples, Z = " << Z_total << std::endl;
        results_map[T] = results;
    }
    
    if (mpi_rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "FTLM Spectral Function Complete\n";
        std::cout << "==========================================" << std::endl;
    }
    
    return results_map;
}

// ============================================================================
// GROUND STATE DYNAMICAL STRUCTURE FACTOR (CONTINUED FRACTION METHOD)
// ============================================================================

/**
 * @brief Evaluate spectral function using continued fraction representation
 * 
 * Computes S(ω) = -Im[G(ω + iη)] / π where G is the continued fraction:
 * G(z) = norm_sq / (z - α₀ - β₁²/(z - α₁ - β₂²/(z - α₂ - ...)))
 * 
 * Uses numerically stable bottom-up evaluation to avoid overflow.
 */
std::vector<double> continued_fraction_spectral_function(
    const std::vector<double>& alpha,
    const std::vector<double>& beta,
    const std::vector<double>& omega_grid,
    double broadening,
    double norm_sq
) {
    if (alpha.empty()) {
        return std::vector<double>(omega_grid.size(), 0.0);
    }
    
    size_t M = alpha.size();
    size_t num_omega = omega_grid.size();
    std::vector<double> spectral(num_omega, 0.0);
    
    // Parallel evaluation over frequency points
    #pragma omp parallel for schedule(static)
    for (size_t iw = 0; iw < num_omega; iw++) {
        double omega = omega_grid[iw];
        Complex z(omega, broadening);  // ω + iη
        
        // Evaluate continued fraction from bottom up (numerically stable)
        // G_M = 0 (termination)
        // G_{n-1} = β_n² / (z - α_n - G_n)
        // ...
        // G(z) = norm_sq / (z - α₀ - G_1)
        
        Complex G(0.0, 0.0);
        
        // Bottom-up: start from n = M-1 down to n = 1
        for (int n = M - 1; n >= 1; n--) {
            // G = β_n² / (z - α_n - G)
            // Note: beta[n] corresponds to β_n (off-diagonal element)
            double beta_n_sq = (n < beta.size()) ? beta[n] * beta[n] : 0.0;
            Complex denom = z - Complex(alpha[n], 0.0) - G;
            
            // Avoid division by zero
            if (std::abs(denom) > 1e-300) {
                G = Complex(beta_n_sq, 0.0) / denom;
            } else {
                G = Complex(0.0, 0.0);
            }
        }
        
        // Final step: G(z) = norm_sq / (z - α₀ - G)
        Complex denom = z - Complex(alpha[0], 0.0) - G;
        Complex G_final;
        if (std::abs(denom) > 1e-300) {
            G_final = Complex(norm_sq, 0.0) / denom;
        } else {
            G_final = Complex(0.0, 0.0);
        }
        
        // Spectral function: S(ω) = -Im[G(ω + iη)] / π
        spectral[iw] = -G_final.imag() / M_PI;
    }
    
    return spectral;
}

/**
 * @brief Compute ground state dynamical structure factor S(ω)
 */
DynamicalResponseResults compute_ground_state_dssf(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t N,
    const GroundStateDSSFParameters& params
) {
    std::cout << "\n==========================================\n";
    std::cout << "Ground State Dynamical Structure Factor\n";
    std::cout << "(Continued Fraction / Lanczos Method)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Ground state energy: " << std::setprecision(10) << ground_state_energy << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    std::cout << "Broadening η: " << params.broadening << std::endl;
    std::cout << "Frequency range: [" << params.omega_min << ", " << params.omega_max << "]" << std::endl;
    std::cout << "Frequency points: " << params.num_omega_points << std::endl;
    std::cout << "Method: " << (params.use_continued_fraction ? "Continued Fraction" : "Eigendecomposition") << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DynamicalResponseResults results;
    results.total_samples = 1;  // Exact calculation, no random sampling
    results.omega_min = params.omega_min;
    results.omega_max = params.omega_max;
    
    // Generate frequency grid
    results.frequencies.resize(params.num_omega_points);
    double omega_step = (params.omega_max - params.omega_min) / 
                        std::max(uint64_t(1), params.num_omega_points - 1);
    for (size_t i = 0; i < params.num_omega_points; i++) {
        results.frequencies[i] = params.omega_min + i * omega_step;
    }
    
    // Apply operator to ground state: |φ⟩ = O|0⟩
    std::cout << "\nApplying operator to ground state..." << std::endl;
    ComplexVector phi(N);
    O(ground_state.data(), phi.data(), N);
    
    // Compute norm ||O|0⟩||²
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    double phi_norm_sq = phi_norm * phi_norm;
    
    std::cout << "  ||O|0⟩|| = " << phi_norm << std::endl;
    std::cout << "  ||O|0⟩||² = " << phi_norm_sq << std::endl;
    
    if (phi_norm < 1e-14) {
        std::cerr << "Warning: O|0⟩ has zero norm. Operator has no matrix elements from ground state.\n";
        results.spectral_function.resize(params.num_omega_points, 0.0);
        results.spectral_function_imag.resize(params.num_omega_points, 0.0);
        results.spectral_error.resize(params.num_omega_points, 0.0);
        results.spectral_error_imag.resize(params.num_omega_points, 0.0);
        return results;
    }
    
    // Normalize |φ⟩ for Lanczos
    Complex scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &scale, phi.data(), 1);
    
    // Build Lanczos tridiagonal starting from |φ⟩
    std::cout << "\nBuilding Lanczos tridiagonal matrix..." << std::endl;
    auto lanczos_start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> alpha, beta;
    uint64_t iterations = build_lanczos_tridiagonal(
        H, phi, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta
    );
    
    auto lanczos_end = std::chrono::high_resolution_clock::now();
    double lanczos_time = std::chrono::duration<double>(lanczos_end - lanczos_start).count();
    
    std::cout << "  Lanczos iterations: " << iterations << std::endl;
    std::cout << "  Lanczos time: " << lanczos_time << " seconds" << std::endl;
    
    if (alpha.empty()) {
        std::cerr << "Error: Lanczos failed to build tridiagonal matrix\n";
        results.spectral_function.resize(params.num_omega_points, 0.0);
        results.spectral_function_imag.resize(params.num_omega_points, 0.0);
        results.spectral_error.resize(params.num_omega_points, 0.0);
        results.spectral_error_imag.resize(params.num_omega_points, 0.0);
        return results;
    }
    
    // Shift eigenvalues: ω - E₀ + E_n → we need to shift α values
    // The resolvent is (ω + E₀ - H + iη)⁻¹, so effectively we shift by E₀
    std::cout << "\nShifting energies by ground state energy E₀ = " << ground_state_energy << std::endl;
    for (size_t i = 0; i < alpha.size(); i++) {
        alpha[i] -= ground_state_energy;
    }
    
    // Compute spectral function
    std::cout << "\nComputing spectral function..." << std::endl;
    auto spectral_start = std::chrono::high_resolution_clock::now();
    
    if (params.use_continued_fraction) {
        // Use continued fraction method (faster, O(M) per ω)
        results.spectral_function = continued_fraction_spectral_function(
            alpha, beta, results.frequencies, params.broadening, phi_norm_sq
        );
    } else {
        // Use eigendecomposition method (for comparison/validation)
        std::vector<double> ritz_values, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);
        
        // Scale weights by norm²
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] *= phi_norm_sq;
        }
        
        // Compute spectral function using Lorentzian broadening
        results.spectral_function.resize(params.num_omega_points, 0.0);
        double eta = params.broadening;
        
        #pragma omp parallel for schedule(static)
        for (size_t iw = 0; iw < params.num_omega_points; iw++) {
            double omega = results.frequencies[iw];
            double sum = 0.0;
            
            for (size_t n = 0; n < ritz_values.size(); n++) {
                double delta = omega - ritz_values[n];
                double lorentzian = (eta / M_PI) / (delta * delta + eta * eta);
                sum += weights[n] * lorentzian;
            }
            
            results.spectral_function[iw] = sum;
        }
    }
    
    auto spectral_end = std::chrono::high_resolution_clock::now();
    double spectral_time = std::chrono::duration<double>(spectral_end - spectral_start).count();
    
    std::cout << "  Spectral function time: " << spectral_time << " seconds" << std::endl;
    
    // For ground state, imaginary part is zero (self-correlation)
    results.spectral_function_imag.resize(params.num_omega_points, 0.0);
    
    // No error bars for exact ground state calculation
    results.spectral_error.resize(params.num_omega_points, 0.0);
    results.spectral_error_imag.resize(params.num_omega_points, 0.0);
    
    // Compute sum rule: ∫ S(ω) dω should equal ||O|0⟩||²
    double integral = 0.0;
    for (size_t i = 1; i < params.num_omega_points; i++) {
        double dw = results.frequencies[i] - results.frequencies[i-1];
        integral += 0.5 * (results.spectral_function[i] + results.spectral_function[i-1]) * dw;
    }
    
    std::cout << "\n--- Sum Rule Check ---" << std::endl;
    std::cout << "  ∫ S(ω) dω = " << integral << std::endl;
    std::cout << "  ||O|0⟩||² = " << phi_norm_sq << std::endl;
    std::cout << "  Ratio: " << integral / phi_norm_sq << " (should be ≈ 1.0)" << std::endl;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n==========================================\n";
    std::cout << "Ground State DSSF Complete\n";
    std::cout << "Total time: " << total_time << " seconds\n";
    std::cout << "==========================================\n";
    
    return results;
}

/**
 * @brief Compute ground state two-operator cross-correlation
 */
DynamicalResponseResults compute_ground_state_cross_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t N,
    const GroundStateDSSFParameters& params
) {
    std::cout << "\n==========================================\n";
    std::cout << "Ground State Cross-Correlation S_{O1,O2}(ω)\n";
    std::cout << "(Lanczos Method)\n";
    std::cout << "==========================================\n";
    
    DynamicalResponseResults results;
    results.total_samples = 1;
    results.omega_min = params.omega_min;
    results.omega_max = params.omega_max;
    
    // Generate frequency grid
    results.frequencies.resize(params.num_omega_points);
    double omega_step = (params.omega_max - params.omega_min) / 
                        std::max(uint64_t(1), params.num_omega_points - 1);
    for (size_t i = 0; i < params.num_omega_points; i++) {
        results.frequencies[i] = params.omega_min + i * omega_step;
    }
    
    // Apply O2 to ground state: |φ₂⟩ = O₂|0⟩
    ComplexVector phi2(N);
    O2(ground_state.data(), phi2.data(), N);
    
    double phi2_norm = cblas_dznrm2(N, phi2.data(), 1);
    
    if (phi2_norm < 1e-14) {
        std::cerr << "Warning: O₂|0⟩ has zero norm.\n";
        results.spectral_function.resize(params.num_omega_points, 0.0);
        results.spectral_function_imag.resize(params.num_omega_points, 0.0);
        results.spectral_error.resize(params.num_omega_points, 0.0);
        results.spectral_error_imag.resize(params.num_omega_points, 0.0);
        return results;
    }
    
    // Normalize for Lanczos
    ComplexVector phi2_normalized = phi2;
    Complex scale(1.0/phi2_norm, 0.0);
    cblas_zscal(N, &scale, phi2_normalized.data(), 1);
    
    // Build Lanczos tridiagonal starting from |φ₂⟩
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> basis_vectors;
    
    // We need basis vectors for cross-correlation
    int iterations = build_lanczos_tridiagonal_with_basis(
        H, phi2_normalized, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta, &basis_vectors
    );
    
    std::cout << "Lanczos iterations: " << iterations << std::endl;
    
    // Diagonalize tridiagonal to get eigenvectors in Krylov basis
    std::vector<double> ritz_values, weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
    
    size_t M = ritz_values.size();
    
    // Apply O1† to ground state: |φ₁⟩ = O₁†|0⟩
    // Note: For cross-correlation ⟨0|O₁†|n⟩⟨n|O₂|0⟩, we need O₁†|0⟩
    // If O1 is Hermitian, O₁† = O₁
    ComplexVector phi1(N);
    O1(ground_state.data(), phi1.data(), N);  // Assuming O1 is Hermitian
    
    // Compute overlap ⟨0|O₁†|n⟩ = ⟨φ₁|n⟩ for each Ritz state |n⟩
    // |n⟩ = Σⱼ V[j,n] |vⱼ⟩ where |vⱼ⟩ are Lanczos basis vectors
    std::vector<Complex> spectral_weights(M);
    
    for (size_t n = 0; n < M; n++) {
        // Reconstruct |n⟩ in full Hilbert space
        ComplexVector ritz_state(N, Complex(0.0, 0.0));
        
        for (size_t j = 0; j < std::min(M, basis_vectors.size()); j++) {
            // evecs is column-major: V[j,n] = evecs[n*M + j]
            double v_jn = evecs[n * M + j];
            Complex coeff(v_jn * phi2_norm, 0.0);  // Scale by original norm
            cblas_zaxpy(N, &coeff, basis_vectors[j].data(), 1, ritz_state.data(), 1);
        }
        
        // Compute ⟨φ₁|n⟩
        Complex overlap;
        cblas_zdotc_sub(N, phi1.data(), 1, ritz_state.data(), 1, &overlap);
        
        // Weight = ⟨0|O₁†|n⟩⟨n|O₂|0⟩ = ⟨φ₁|n⟩ * ⟨n|φ₂⟩
        // ⟨n|φ₂⟩ = sqrt(weights[n]) * phi2_norm (from diagonalization)
        Complex weight_n = overlap;  // ⟨φ₁|n⟩ already includes proper normalization
        spectral_weights[n] = weight_n;
    }
    
    // Shift eigenvalues by ground state energy
    for (size_t i = 0; i < ritz_values.size(); i++) {
        ritz_values[i] -= ground_state_energy;
    }
    
    // Compute spectral function
    results.spectral_function.resize(params.num_omega_points, 0.0);
    results.spectral_function_imag.resize(params.num_omega_points, 0.0);
    
    double eta = params.broadening;
    
    #pragma omp parallel for schedule(static)
    for (size_t iw = 0; iw < params.num_omega_points; iw++) {
        double omega = results.frequencies[iw];
        Complex sum(0.0, 0.0);
        
        for (size_t n = 0; n < M; n++) {
            double delta = omega - ritz_values[n];
            // Lorentzian: (η/π) / ((ω - E)² + η²)
            double lorentzian = (eta / M_PI) / (delta * delta + eta * eta);
            sum += spectral_weights[n] * lorentzian;
        }
        
        results.spectral_function[iw] = sum.real();
        results.spectral_function_imag[iw] = sum.imag();
    }
    
    results.spectral_error.resize(params.num_omega_points, 0.0);
    results.spectral_error_imag.resize(params.num_omega_points, 0.0);
    
    return results;
}

/**
 * @brief Load ground state from eigenvector files (HDF5 or legacy formats)
 */
bool load_ground_state_from_file(
    const std::string& eigenvector_dir,
    ComplexVector& ground_state,
    double& ground_state_energy,
    uint64_t expected_dim
) {
    std::cout << "\n--- Loading ground state from " << eigenvector_dir << " ---\n";
    
    // Try HDF5 first (preferred format)
    try {
        std::string hdf5_file = eigenvector_dir + "/ed_results.h5";
        std::ifstream test_file(hdf5_file);
        if (test_file.good()) {
            test_file.close();
            
            // Load eigenvalues
            std::vector<double> eigenvalues = HDF5IO::loadEigenvalues(hdf5_file);
            if (!eigenvalues.empty()) {
                ground_state_energy = eigenvalues[0];
                std::cout << "Loaded ground state energy from HDF5: " << ground_state_energy << std::endl;
            }
            
            // Load eigenvector
            std::vector<Complex> gs_vec = HDF5IO::loadEigenvector(hdf5_file, 0);
            if (!gs_vec.empty()) {
                if (expected_dim > 0 && gs_vec.size() != expected_dim) {
                    std::cerr << "Warning: Dimension mismatch: HDF5 has " << gs_vec.size() 
                              << ", expected " << expected_dim << std::endl;
                } else {
                    ground_state = std::move(gs_vec);
                    
                    // Normalize (should already be normalized, but just in case)
                    double norm = cblas_dznrm2(ground_state.size(), ground_state.data(), 1);
                    if (std::abs(norm - 1.0) > 1e-6) {
                        std::cout << "Normalizing eigenvector (norm was " << norm << ")" << std::endl;
                        Complex scale(1.0/norm, 0.0);
                        cblas_zscal(ground_state.size(), &scale, ground_state.data(), 1);
                    }
                    std::cout << "Loaded ground state eigenvector from HDF5 (dim=" << ground_state.size() << ")" << std::endl;
                    return true;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "HDF5 load failed, trying legacy formats: " << e.what() << std::endl;
    }
    
    // Legacy file naming conventions (fallback)
    std::vector<std::string> eigenvector_files = {
        eigenvector_dir + "/eigenvector_0.dat",
        eigenvector_dir + "/eigenvector_block0_0.dat"
    };
    
    std::vector<std::string> eigenvalue_files = {
        eigenvector_dir + "/eigenvalues.dat",
        eigenvector_dir + "/eigenvalues.txt"
    };
    
    // Try to load eigenvector
    bool loaded_eigenvector = false;
    
    for (const auto& filename : eigenvector_files) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) continue;
        
        std::cout << "Found eigenvector file: " << filename << std::endl;
        
        // Read dimension
        uint64_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(uint64_t));
        
        if (expected_dim > 0 && dim != expected_dim) {
            std::cerr << "Warning: Dimension mismatch: file has " << dim 
                      << ", expected " << expected_dim << std::endl;
            file.close();
            continue;
        }
        
        // Read complex vector
        ground_state.resize(dim);
        file.read(reinterpret_cast<char*>(ground_state.data()), dim * sizeof(Complex));
        
        if (file.good()) {
            loaded_eigenvector = true;
            std::cout << "Loaded eigenvector with dimension " << dim << std::endl;
            
            // Normalize (should already be normalized, but just in case)
            double norm = cblas_dznrm2(dim, ground_state.data(), 1);
            if (std::abs(norm - 1.0) > 1e-6) {
                std::cout << "Normalizing eigenvector (norm was " << norm << ")" << std::endl;
                Complex scale(1.0/norm, 0.0);
                cblas_zscal(dim, &scale, ground_state.data(), 1);
            }
            break;
        }
        file.close();
    }
    
    if (!loaded_eigenvector) {
        std::cerr << "Error: Could not load eigenvector from any expected location\n";
        return false;
    }
    
    // Try to load eigenvalue (ground state energy)
    bool loaded_energy = false;
    
    for (const auto& filename : eigenvalue_files) {
        std::ifstream file(filename);
        if (!file.is_open()) continue;
        
        // Check if binary or text
        if (filename.find(".dat") != std::string::npos) {
            file.close();
            std::ifstream binfile(filename, std::ios::binary);
            if (!binfile.is_open()) continue;
            
            // Binary format: num_eigenvalues followed by eigenvalues
            size_t num_eig;
            binfile.read(reinterpret_cast<char*>(&num_eig), sizeof(size_t));
            
            if (num_eig > 0) {
                binfile.read(reinterpret_cast<char*>(&ground_state_energy), sizeof(double));
                loaded_energy = true;
                std::cout << "Loaded ground state energy: " << ground_state_energy << std::endl;
            }
            binfile.close();
        } else {
            // Text format: one eigenvalue per line
            if (file >> ground_state_energy) {
                loaded_energy = true;
                std::cout << "Loaded ground state energy: " << ground_state_energy << std::endl;
            }
            file.close();
        }
        
        if (loaded_energy) break;
    }
    
    if (!loaded_energy) {
        std::cerr << "Warning: Could not load ground state energy, using 0.0\n";
        ground_state_energy = 0.0;
    }
    
    return loaded_eigenvector;
}
