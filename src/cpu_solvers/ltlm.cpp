// ltlm.cpp - Low Temperature Lanczos Method implementation

#include "ltlm.h"
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cstring>
#include <map>

/**
 * @brief Compute low-energy eigenstates using Lanczos method
 */
LTLMEigenResults compute_low_energy_spectrum(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    int num_states,
    const LTLMParameters& params
) {
    LTLMEigenResults results;
    
    std::cout << "\n==========================================\n";
    std::cout << "Low Temperature Lanczos Method (LTLM)\n";
    std::cout << "Computing Low-Energy Eigenspectrum\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Requested eigenstates: " << num_states << std::endl;
    std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
    
    if (params.krylov_dim <= num_states) {
        std::cerr << "Warning: Krylov dimension should be larger than num_eigenstates\n";
        std::cerr << "         Adjusting krylov_dim to " << num_states * 2 << std::endl;
    }
    
    int krylov_dim = std::max(params.krylov_dim, num_states * 2);
    krylov_dim = std::min(krylov_dim, N);
    
    // Initialize random initial vector
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    ComplexVector v0(N);
    for (int i = 0; i < N; i++) {
        v0[i] = Complex(dist(gen), dist(gen));
    }
    
    // Normalize initial vector
    double norm = cblas_dznrm2(N, v0.data(), 1);
    Complex scale_factor = Complex(1.0/norm, 0.0);
    cblas_zscal(N, &scale_factor, v0.data(), 1);
    
    // Working vectors
    ComplexVector v_current = v0;
    ComplexVector v_prev(N, Complex(0.0, 0.0));
    ComplexVector v_next(N);
    ComplexVector w(N);
    
    // Store basis vectors for reconstruction
    std::vector<ComplexVector> lanczos_vectors;
    if (params.store_eigenvectors || params.verify_eigenvalues) {
        lanczos_vectors.push_back(v_current);
    }
    
    // Tridiagonal matrix elements
    std::vector<double> alpha;
    std::vector<double> beta;
    beta.push_back(0.0); // β_0 is not used
    
    std::cout << "Running Lanczos iterations..." << std::endl;
    
    // Lanczos iteration
    int iterations;
    for (iterations = 0; iterations < krylov_dim; iterations++) {
        // w = H*v_j
        H(v_current.data(), w.data(), N);
        
        // w = w - beta_j * v_{j-1}
        if (iterations > 0) {
            Complex neg_beta = Complex(-beta[iterations], 0.0);
            cblas_zaxpy(N, &neg_beta, v_prev.data(), 1, w.data(), 1);
        }
        
        // alpha_j = <v_j, w>
        Complex dot_product;
        cblas_zdotc_sub(N, v_current.data(), 1, w.data(), 1, &dot_product);
        alpha.push_back(std::real(dot_product));
        
        // w = w - alpha_j * v_j
        Complex neg_alpha = Complex(-alpha[iterations], 0.0);
        cblas_zaxpy(N, &neg_alpha, v_current.data(), 1, w.data(), 1);
        
        // Reorthogonalization
        if (params.full_reorthogonalization) {
            // Full reorthogonalization against all previous vectors
            for (size_t k = 0; k < lanczos_vectors.size(); k++) {
                Complex overlap;
                cblas_zdotc_sub(N, lanczos_vectors[k].data(), 1, w.data(), 1, &overlap);
                Complex neg_overlap = -overlap;
                cblas_zaxpy(N, &neg_overlap, lanczos_vectors[k].data(), 1, w.data(), 1);
            }
        } else if (params.reorth_frequency > 0 && (iterations + 1) % params.reorth_frequency == 0) {
            // Periodic reorthogonalization
            for (size_t k = 0; k < lanczos_vectors.size(); k++) {
                Complex overlap;
                cblas_zdotc_sub(N, lanczos_vectors[k].data(), 1, w.data(), 1, &overlap);
                if (std::abs(overlap) > params.tolerance) {
                    Complex neg_overlap = -overlap;
                    cblas_zaxpy(N, &neg_overlap, lanczos_vectors[k].data(), 1, w.data(), 1);
                }
            }
        }
        
        // beta_{j+1} = ||w||
        norm = cblas_dznrm2(N, w.data(), 1);
        beta.push_back(norm);
        
        // Check for breakdown or convergence
        if (norm < params.tolerance) {
            std::cout << "Lanczos breakdown at iteration " << iterations + 1 << std::endl;
            iterations++;
            break;
        }
        
        // v_{j+1} = w / beta_{j+1}
        for (int i = 0; i < N; i++) {
            v_next[i] = w[i] / norm;
        }
        
        // Store for reorthogonalization or reconstruction
        if (params.store_eigenvectors || params.verify_eigenvalues) {
            lanczos_vectors.push_back(v_next);
        }
        
        // Update for next iteration
        v_prev = v_current;
        v_current = v_next;
        
        // Print progress periodically
        if ((iterations + 1) % 50 == 0) {
            std::cout << "  Iteration " << iterations + 1 << "/" << krylov_dim << std::endl;
        }
    }
    
    results.lanczos_iterations = iterations;
    int m = alpha.size();
    
    std::cout << "Lanczos iterations completed: " << iterations << std::endl;
    std::cout << "Tridiagonal matrix size: " << m << " x " << m << std::endl;
    
    // Diagonalize tridiagonal matrix
    std::cout << "Diagonalizing tridiagonal matrix..." << std::endl;
    
    std::vector<double> diag = alpha;
    std::vector<double> offdiag(m - 1);
    for (int i = 0; i < m - 1; i++) {
        offdiag[i] = beta[i + 1];
    }
    
    std::vector<double> evecs;
    if (params.store_eigenvectors || params.verify_eigenvalues) {
        evecs.resize(m * m);
    }
    
    int info;
    if (params.store_eigenvectors || params.verify_eigenvalues) {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', m, diag.data(), offdiag.data(), evecs.data(), m);
    } else {
        info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'N', m, diag.data(), offdiag.data(), nullptr, m);
    }
    
    if (info != 0) {
        std::cerr << "Error: Tridiagonal diagonalization failed with code " << info << std::endl;
        results.all_converged = false;
        return results;
    }
    
    std::cout << "Diagonalization successful" << std::endl;
    
    // Extract the lowest num_states eigenvalues
    int num_extracted = std::min(num_states, m);
    results.eigenvalues.resize(num_extracted);
    
    for (int i = 0; i < num_extracted; i++) {
        results.eigenvalues[i] = diag[i];
    }
    
    std::cout << "Extracted " << num_extracted << " lowest eigenvalues" << std::endl;
    std::cout << "Ground state energy: " << results.eigenvalues[0] << std::endl;
    if (num_extracted > 1) {
        std::cout << "First excitation gap: " << results.eigenvalues[1] - results.eigenvalues[0] << std::endl;
    }
    
    // Reconstruct eigenvectors in full space if requested
    if (params.store_eigenvectors && !lanczos_vectors.empty()) {
        std::cout << "Reconstructing eigenvectors in full space..." << std::endl;
        results.eigenvectors.resize(num_extracted);
        
        for (int i = 0; i < num_extracted; i++) {
            results.eigenvectors[i].resize(N, Complex(0.0, 0.0));
            
            // |ψ_i⟩ = Σ_j evecs[i,j] |v_j⟩
            for (int j = 0; j < m; j++) {
                double coeff = evecs[i * m + j];
                Complex alpha_c(coeff, 0.0);
                cblas_zaxpy(N, &alpha_c, lanczos_vectors[j].data(), 1, 
                           results.eigenvectors[i].data(), 1);
            }
            
            // Normalize (should already be normalized, but double-check)
            double vec_norm = cblas_dznrm2(N, results.eigenvectors[i].data(), 1);
            if (std::abs(vec_norm - 1.0) > 1e-10) {
                Complex norm_factor(1.0/vec_norm, 0.0);
                cblas_zscal(N, &norm_factor, results.eigenvectors[i].data(), 1);
            }
        }
        
        std::cout << "Eigenvector reconstruction complete" << std::endl;
    }
    
    // Verify eigenvalues using residual test if requested
    if (params.verify_eigenvalues && !lanczos_vectors.empty()) {
        std::cout << "Verifying eigenvalues using residual norms..." << std::endl;
        results.residual_norms.resize(num_extracted);
        results.converged_states = 0;
        
        // Reconstruct eigenvectors if not already done
        std::vector<ComplexVector> eigenvecs_full;
        if (!params.store_eigenvectors) {
            eigenvecs_full.resize(num_extracted);
            for (int i = 0; i < num_extracted; i++) {
                eigenvecs_full[i].resize(N, Complex(0.0, 0.0));
                for (int j = 0; j < m; j++) {
                    double coeff = evecs[i * m + j];
                    Complex alpha_c(coeff, 0.0);
                    cblas_zaxpy(N, &alpha_c, lanczos_vectors[j].data(), 1, 
                               eigenvecs_full[i].data(), 1);
                }
                // Normalize
                double vec_norm = cblas_dznrm2(N, eigenvecs_full[i].data(), 1);
                Complex norm_factor(1.0/vec_norm, 0.0);
                cblas_zscal(N, &norm_factor, eigenvecs_full[i].data(), 1);
            }
        }
        
        for (int i = 0; i < num_extracted; i++) {
            // Compute residual: r = H|ψ⟩ - λ|ψ⟩
            ComplexVector residual(N);
            const ComplexVector& eigenvec = params.store_eigenvectors ? 
                                           results.eigenvectors[i] : eigenvecs_full[i];
            
            H(eigenvec.data(), residual.data(), N);
            
            Complex lambda(-results.eigenvalues[i], 0.0);
            cblas_zaxpy(N, &lambda, eigenvec.data(), 1, residual.data(), 1);
            
            double residual_norm = cblas_dznrm2(N, residual.data(), 1);
            results.residual_norms[i] = residual_norm;
            
            if (residual_norm < params.residual_tolerance) {
                results.converged_states++;
            }
            
            if (i < 10 || residual_norm > params.residual_tolerance) {
                std::cout << "  State " << i << ": E = " << std::setw(15) << std::setprecision(10)
                         << results.eigenvalues[i] << ", ||r|| = " << std::scientific 
                         << residual_norm << std::fixed << std::endl;
            }
        }
        
        std::cout << "Converged states: " << results.converged_states << " / " 
                 << num_extracted << std::endl;
        results.all_converged = (results.converged_states == num_extracted);
        
        if (!results.all_converged) {
            std::cout << "Warning: Not all eigenvalues converged to requested tolerance" << std::endl;
        }
    } else {
        results.converged_states = num_extracted;
        results.all_converged = true;
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Eigenspectrum Computation Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}

/**
 * @brief Compute thermodynamic properties from exact low-energy eigenstates
 */
ThermodynamicData compute_ltlm_thermodynamics(
    const std::vector<double>& eigenvalues,
    const std::vector<double>& degeneracies,
    const std::vector<double>& temperatures
) {
    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    
    int n_temps = temperatures.size();
    int n_states = eigenvalues.size();
    
    if (n_states == 0) {
        std::cerr << "Error: No eigenvalues provided" << std::endl;
        return thermo;
    }
    
    thermo.energy.resize(n_temps);
    thermo.specific_heat.resize(n_temps);
    thermo.entropy.resize(n_temps);
    thermo.free_energy.resize(n_temps);
    
    // Use degeneracies if provided, otherwise assume all states are non-degenerate
    std::vector<double> degeneracy_weights = degeneracies;
    if (degeneracy_weights.empty() || degeneracy_weights.size() != eigenvalues.size()) {
        degeneracy_weights.assign(n_states, 1.0);
    }
    
    // Find minimum energy for numerical stability
    double e_min = eigenvalues[0];  // Eigenvalues should be sorted
    
    std::cout << "Computing thermodynamic properties..." << std::endl;
    std::cout << "Number of states: " << n_states << std::endl;
    std::cout << "Ground state energy: " << e_min << std::endl;
    
    for (int t = 0; t < n_temps; t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function and observables using shifted energies
        // Z = Σ_i g_i * exp(-β * (E_i - E_min))
        double Z = 0.0;
        double E_avg = 0.0;
        double E2_avg = 0.0;
        
        std::vector<double> boltzmann_factors(n_states);
        
        // Compute Boltzmann factors with shift
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = eigenvalues[i] - e_min;
            boltzmann_factors[i] = degeneracy_weights[i] * std::exp(-beta * shifted_energy);
            Z += boltzmann_factors[i];
        }
        
        // Normalize and compute expectations
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double prob = boltzmann_factors[i] / Z;
                E_avg += prob * eigenvalues[i];
                E2_avg += prob * eigenvalues[i] * eigenvalues[i];
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
            thermo.entropy[t] = std::log(degeneracy_weights[0]);  // Ground state degeneracy
            thermo.free_energy[t] = e_min;
        }
    }
    
    std::cout << "Thermodynamic calculation complete" << std::endl;
    
    return thermo;
}

/**
 * @brief Compute thermodynamics with automatic degeneracy detection
 */
ThermodynamicData compute_ltlm_thermodynamics_auto_degeneracy(
    const std::vector<double>& eigenvalues,
    const std::vector<double>& temperatures,
    double degeneracy_threshold
) {
    if (eigenvalues.empty()) {
        std::cerr << "Error: No eigenvalues provided" << std::endl;
        return ThermodynamicData();
    }
    
    std::cout << "Detecting degeneracies (threshold = " << degeneracy_threshold << ")..." << std::endl;
    
    // Group nearly-degenerate eigenvalues
    std::vector<double> unique_energies;
    std::vector<double> degeneracies;
    
    unique_energies.push_back(eigenvalues[0]);
    degeneracies.push_back(1.0);
    
    for (size_t i = 1; i < eigenvalues.size(); i++) {
        double energy_diff = std::abs(eigenvalues[i] - unique_energies.back());
        
        if (energy_diff < degeneracy_threshold) {
            // Degenerate with previous level
            degeneracies.back() += 1.0;
        } else {
            // New energy level
            unique_energies.push_back(eigenvalues[i]);
            degeneracies.push_back(1.0);
        }
    }
    
    std::cout << "Found " << unique_energies.size() << " unique energy levels from " 
             << eigenvalues.size() << " states" << std::endl;
    
    // Print degeneracies for first few levels
    int num_to_print = std::min(10, (int)unique_energies.size());
    for (int i = 0; i < num_to_print; i++) {
        std::cout << "  Level " << i << ": E = " << std::setprecision(12) 
                 << unique_energies[i] << ", g = " << (int)degeneracies[i] << std::endl;
    }
    
    return compute_ltlm_thermodynamics(unique_energies, degeneracies, temperatures);
}

/**
 * @brief Main LTLM driver function
 */
LTLMResults low_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const LTLMParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir
) {
    std::cout << "\n==========================================\n";
    std::cout << "Low Temperature Lanczos Method (LTLM)\n";
    std::cout << "Full Thermodynamic Calculation\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Number of eigenstates: " << params.num_eigenstates << std::endl;
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    
    LTLMResults results;
    
    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(1, num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Step 1: Compute low-energy spectrum
    results.eigen_results = compute_low_energy_spectrum(H, N, params.num_eigenstates, params);
    
    if (results.eigen_results.eigenvalues.empty()) {
        std::cerr << "Error: Failed to compute eigenspectrum" << std::endl;
        return results;
    }
    
    results.ground_state_energy = results.eigen_results.eigenvalues[0];
    results.num_states_used = results.eigen_results.eigenvalues.size();
    
    // Step 2: Compute thermodynamic properties with automatic degeneracy detection
    results.thermo_data = compute_ltlm_thermodynamics_auto_degeneracy(
        results.eigen_results.eigenvalues,
        temperatures,
        1e-10  // Degeneracy threshold
    );
    
    // Save eigenspectrum if output directory is provided
    if (!output_dir.empty()) {
        std::string spectrum_file = output_dir + "/ltlm_spectrum.txt";
        save_eigenspectrum(results.eigen_results, spectrum_file);
    }
    
    // Estimate validity range of LTLM
    if (results.num_states_used > 1) {
        double max_energy = results.eigen_results.eigenvalues.back();
        double energy_window = max_energy - results.ground_state_energy;
        double max_valid_temp = energy_window / 10.0;  // Rule of thumb: kT < ΔE/10
        
        std::cout << "\n==========================================\n";
        std::cout << "LTLM Validity Estimate\n";
        std::cout << "==========================================\n";
        std::cout << "Energy window covered: " << energy_window << std::endl;
        std::cout << "LTLM recommended for T < " << max_valid_temp << std::endl;
        
        if (temp_max > max_valid_temp) {
            std::cout << "\nWarning: Maximum temperature exceeds recommended range!" << std::endl;
            std::cout << "         Consider using FTLM for T > " << max_valid_temp << std::endl;
            std::cout << "         or increase num_eigenstates in LTLM parameters." << std::endl;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "LTLM Calculation Complete\n";
    std::cout << "==========================================\n";
    
    return results;
}

/**
 * @brief Save LTLM results to file
 */
void save_ltlm_results(
    const LTLMResults& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "# LTLM Results (exact low-energy spectrum with " << results.num_states_used << " states)\n";
    file << "# Ground state energy: " << results.ground_state_energy << "\n";
    file << "# Temperature  Energy  Specific_Heat  Entropy  Free_Energy\n";
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < results.thermo_data.temperatures.size(); i++) {
        file << results.thermo_data.temperatures[i] << " "
             << results.thermo_data.energy[i] << " "
             << results.thermo_data.specific_heat[i] << " "
             << results.thermo_data.entropy[i] << " "
             << results.thermo_data.free_energy[i] << "\n";
    }
    
    file.close();
    std::cout << "LTLM results saved to: " << filename << std::endl;
}

/**
 * @brief Save eigenspectrum to file
 */
void save_eigenspectrum(
    const LTLMEigenResults& eigen_results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    file << "# Low-Energy Eigenspectrum from LTLM\n";
    file << "# Lanczos iterations: " << eigen_results.lanczos_iterations << "\n";
    file << "# Converged states: " << eigen_results.converged_states << " / " 
         << eigen_results.eigenvalues.size() << "\n";
    file << "# Index  Eigenvalue  Residual_Norm\n";
    file << std::scientific << std::setprecision(15);
    
    for (size_t i = 0; i < eigen_results.eigenvalues.size(); i++) {
        file << i << " " << eigen_results.eigenvalues[i];
        
        if (i < eigen_results.residual_norms.size()) {
            file << " " << eigen_results.residual_norms[i];
        }
        
        file << "\n";
    }
    
    file.close();
    std::cout << "Eigenspectrum saved to: " << filename << std::endl;
}
