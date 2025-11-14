// ltlm.cpp - Low Temperature Lanczos Method implementation
#include "../core/system_utils.h"

#include "ltlm.h"
#include "ftlm.h"     // For build_lanczos_tridiagonal function
#include "lanczos.h"  // For helper functions
#include <fstream>
#include <iomanip>
#include <numeric>
#include <cstring>

/**
 * @brief Find ground state using Lanczos iteration
 */
double find_ground_state_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    ComplexVector& ground_state
) {
    std::cout << "  Finding ground state via Lanczos...\n";
    
    // Generate random initial vector using helper function
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    ComplexVector v0 = generateRandomVector(N, gen, dist);
    
    // Build Lanczos tridiagonal with basis storage
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    uint64_t iterations = build_lanczos_tridiagonal_with_basis(
        H, v0, N, krylov_dim, tolerance,
        full_reorth, reorth_freq,
        alpha, beta, &lanczos_vectors
    );
    
    std::cout << "  Lanczos iterations for ground state: " << iterations << std::endl;
    
    uint64_t m = alpha.size();
    
    // Diagonalize tridiagonal matrix using helper function
    std::vector<double> ritz_values, weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
    
    if (ritz_values.empty()) {
        std::cerr << "  Error: Ground state tridiagonal diagonalization failed" << std::endl;
        ground_state = v0;  // Return initial state as fallback
        return 0.0;
    }
    
    double ground_energy = ritz_values[0];
    std::cout << "  Ground state energy: " << ground_energy << std::endl;
    
    // Reconstruct ground state in full Hilbert space
    // |ψ_0⟩ = Σ_j c_j |v_j⟩ where c_j = evecs[j] (first eigenvector)
    ground_state.resize(N, Complex(0.0, 0.0));
    
    for (int j = 0; j < m; j++) {
        double coeff = evecs[j];  // First eigenvector (ground state)
        Complex alpha_c(coeff, 0.0);
        cblas_zaxpy(N, &alpha_c, lanczos_vectors[j].data(), 1, ground_state.data(), 1);
    }
    
    // Normalize
    double norm = cblas_dznrm2(N, ground_state.data(), 1);
    Complex scale(1.0/norm, 0.0);
    cblas_zscal(N, &scale, ground_state.data(), 1);
    
    return ground_energy;
}

/**
 * @brief Build Krylov subspace from ground state for low-lying excitations
 */
int build_excitation_spectrum(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& ground_state,
    double ground_energy,
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& excitation_energies,
    std::vector<double>& weights
) {
    std::cout << "  Building excitation spectrum from ground state...\n";
    
    // Build Lanczos tridiagonal starting from ground state
    std::vector<double> alpha, beta;
    uint64_t iterations = build_lanczos_tridiagonal(
        H, ground_state, N, krylov_dim, tolerance,
        full_reorth, reorth_freq,
        alpha, beta
    );
    
    std::cout << "  Lanczos iterations for excitations: " << iterations << std::endl;
    
    uint64_t m = alpha.size();
    
    // Diagonalize tridiagonal matrix using helper function
    diagonalize_tridiagonal_ritz(alpha, beta, excitation_energies, weights);
    
    if (excitation_energies.empty()) {
        std::cerr << "  Warning: Excitation spectrum diagonalization failed" << std::endl;
        return 0;
    }
    
    std::cout << "  Found " << m << " excitation states\n";
    std::cout << "  Energy range: [" << excitation_energies[0] << ", " 
              << excitation_energies[m-1] << "]\n";
    
    return m;
}

/**
 * @brief Compute thermodynamics from ground state and low-lying excitations
 */
ThermodynamicData compute_ltlm_thermodynamics(
    double ground_energy,
    const std::vector<double>& excitation_energies,
    const std::vector<double>& weights,
    const std::vector<double>& temperatures
) {
    ThermodynamicData thermo;
    thermo.temperatures = temperatures;
    
    uint64_t n_temps = temperatures.size();
    uint64_t n_states = excitation_energies.size();
    
    thermo.energy.resize(n_temps);
    thermo.specific_heat.resize(n_temps);
    thermo.entropy.resize(n_temps);
    thermo.free_energy.resize(n_temps);
    
    for (int t = 0; t < n_temps; t++) {
        double T = temperatures[t];
        double beta = 1.0 / T;
        
        // Compute partition function using shifted energies
        // All energies are already relative to some reference
        // Z = Σ_i w_i * exp(-β * E_i)
        double Z = 0.0;
        double E_avg = 0.0;
        double E2_avg = 0.0;
        
        std::vector<double> boltzmann_factors(n_states);
        
        // Compute Boltzmann factors (excitation_energies already include ground state energy)
        for (int i = 0; i < n_states; i++) {
            // For numerical stability, shift by ground state energy
            double shifted_energy = excitation_energies[i] - ground_energy;
            boltzmann_factors[i] = weights[i] * std::exp(-beta * shifted_energy);
            Z += boltzmann_factors[i];
        }
        
        // Normalize and compute expectations
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double prob = boltzmann_factors[i] / Z;
                E_avg += prob * excitation_energies[i];
                E2_avg += prob * excitation_energies[i] * excitation_energies[i];
            }
            
            // Thermodynamic quantities
            thermo.energy[t] = E_avg;
            thermo.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
            thermo.entropy[t] = beta * (E_avg - ground_energy) + std::log(Z);
            thermo.free_energy[t] = ground_energy - T * std::log(Z);
        } else {
            // Very low temperature - use ground state only
            thermo.energy[t] = ground_energy;
            thermo.specific_heat[t] = 0.0;
            thermo.entropy[t] = 0.0;
            thermo.free_energy[t] = ground_energy;
        }
    }
    
    return thermo;
}

/**
 * @brief Main LTLM driver function
 */
LTLMResults low_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const LTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const ComplexVector* ground_state_input,
    const std::string& output_dir
) {
    std::cout << "\n==========================================\n";
    std::cout << "Low Temperature Lanczos Method (LTLM)\n";
    std::cout << "==========================================\n";
    std::cout << "Hilbert space dimension: " << N << std::endl;
    std::cout << "Ground state Krylov dim: " << params.ground_state_krylov << std::endl;
    std::cout << "Excitation Krylov dim: " << params.krylov_dim << std::endl;
    std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    
    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(static_cast<uint64_t>(1), num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    LTLMResults results;
    results.total_samples = 1;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/ltlm_data";
        safe_system_call(cmd);
    }
    
    // Step 1: Find or use ground state
    ComplexVector ground_state;
    double ground_energy;
    
    if (ground_state_input != nullptr && params.use_exact_ground_state) {
        std::cout << "\n--- Using provided ground state ---\n";
        ground_state = *ground_state_input;
        
        // Compute ground state energy
        ComplexVector H_gs(N);
        H(ground_state.data(), H_gs.data(), N);
        Complex energy_complex;
        cblas_zdotc_sub(N, ground_state.data(), 1, H_gs.data(), 1, &energy_complex);
        ground_energy = std::real(energy_complex);
        
        std::cout << "Ground state energy: " << ground_energy << std::endl;
    } else {
        std::cout << "\n--- Step 1: Finding Ground State ---\n";
        ground_energy = find_ground_state_lanczos(
            H, N, params.ground_state_krylov, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            ground_state
        );
    }
    
    results.ground_state_energy = ground_energy;
    
    // Step 2: Build excitation spectrum from ground state
    std::cout << "\n--- Step 2: Building Excitation Spectrum ---\n";
    std::vector<double> excitation_energies, weights;
    uint64_t n_excitations = build_excitation_spectrum(
        H, ground_state, ground_energy, N, params.krylov_dim,
        params.tolerance, params.full_reorthogonalization, params.reorth_frequency,
        excitation_energies, weights
    );
    
    results.krylov_dimension = n_excitations;
    results.low_lying_spectrum = excitation_energies;
    
    if (n_excitations == 0) {
        std::cerr << "Error: Failed to build excitation spectrum\n";
        return results;
    }
    
    // Step 3: Compute thermodynamics
    std::cout << "\n--- Step 3: Computing Thermodynamics ---\n";
    results.thermo_data = compute_ltlm_thermodynamics(
        ground_energy, excitation_energies, weights, temperatures
    );
    
    // Initialize error bars to zero (LTLM is deterministic with single sample)
    results.energy_error.resize(num_temp_bins, 0.0);
    results.specific_heat_error.resize(num_temp_bins, 0.0);
    results.entropy_error.resize(num_temp_bins, 0.0);
    results.free_energy_error.resize(num_temp_bins, 0.0);
    
    // Save intermediate data if requested
    if (params.store_intermediate && !output_dir.empty()) {
        // Save excitation spectrum
        std::string spectrum_file = output_dir + "/ltlm_data/excitation_spectrum.txt";
        std::ofstream f(spectrum_file);
        if (f.is_open()) {
            f << "# Index  Energy  Weight\n";
            for (int i = 0; i < n_excitations; i++) {
                f << std::scientific << std::setprecision(12)
                  << i << " "
                  << excitation_energies[i] << " "
                  << weights[i] << "\n";
            }
            f.close();
            std::cout << "Saved excitation spectrum to: " << spectrum_file << std::endl;
        }
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "LTLM Calculation Complete\n";
    std::cout << "==========================================\n";
    std::cout << "Ground state energy: " << ground_energy << std::endl;
    std::cout << "Number of excitations: " << n_excitations << std::endl;
    
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
    
    file << "# LTLM Results\n";
    file << "# Ground state energy: " << results.ground_state_energy << "\n";
    file << "# Krylov dimension: " << results.krylov_dimension << "\n";
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
    std::cout << "LTLM results saved to: " << filename << std::endl;
}
