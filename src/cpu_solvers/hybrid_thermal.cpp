// hybrid_thermal.cpp - Implementation of Hybrid Thermal Method

#include "hybrid_thermal.h"
#include "ltlm.h"
#include "ftlm.h"
#include <fstream>
#include <iomanip>
#include <algorithm>

/**
 * @brief Main Hybrid Thermal Method implementation
 */
HybridThermalResults hybrid_thermal_method(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const HybridThermalParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir
) {
    std::cout << "\n================================================\n";
    std::cout << "       Hybrid Thermal Method (LTLM+FTLM)       \n";
    std::cout << "================================================\n";
    std::cout << "Temperature range: " << temp_min << " - " << temp_max << std::endl;
    std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    std::cout << "Crossover temperature: " << params.crossover_temperature << std::endl;
    std::cout << "  • LTLM for T < " << params.crossover_temperature << " (accurate ground state)\n";
    std::cout << "  • FTLM for T ≥ " << params.crossover_temperature << " (efficient sampling)\n";
    std::cout << "================================================\n\n";
    
    // Generate full temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(1, num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Find crossover index in temperature grid
    int crossover_index = 0;
    double actual_crossover = params.crossover_temperature;
    
    for (int i = 0; i < num_temp_bins; i++) {
        if (temperatures[i] >= params.crossover_temperature) {
            crossover_index = i;
            actual_crossover = temperatures[i];
            break;
        }
    }
    
    // If crossover is beyond temp_max, use only LTLM
    if (crossover_index == 0 && params.crossover_temperature > temp_max) {
        crossover_index = num_temp_bins;
        actual_crossover = temp_max;
    }
    
    // Initialize results structure
    HybridThermalResults results;
    results.thermo_data.temperatures = temperatures;
    results.thermo_data.energy.resize(num_temp_bins);
    results.thermo_data.specific_heat.resize(num_temp_bins);
    results.thermo_data.entropy.resize(num_temp_bins);
    results.thermo_data.free_energy.resize(num_temp_bins);
    results.energy_error.resize(num_temp_bins, 0.0);
    results.specific_heat_error.resize(num_temp_bins, 0.0);
    results.entropy_error.resize(num_temp_bins, 0.0);
    results.free_energy_error.resize(num_temp_bins, 0.0);
    results.actual_crossover_temp = actual_crossover;
    results.crossover_index = crossover_index;
    results.ltlm_points = 0;
    results.ftlm_points = 0;
    results.ftlm_samples_used = 0;
    
    // ========================================================================
    // LTLM Phase: Low Temperature (T < T_crossover)
    // ========================================================================
    
    if (crossover_index > 0) {
        std::cout << "┌────────────────────────────────────────────┐\n";
        std::cout << "│  Phase 1: LTLM (Low Temperature)          │\n";
        std::cout << "└────────────────────────────────────────────┘\n";
        std::cout << "Temperature points: " << crossover_index 
                  << " (T = " << temperatures[0] << " to " << temperatures[crossover_index-1] << ")\n\n";
        
        // Setup LTLM parameters
        LTLMParameters ltlm_params;
        ltlm_params.krylov_dim = params.ltlm_krylov_dim;
        ltlm_params.ground_state_krylov = params.ltlm_ground_krylov;
        ltlm_params.full_reorthogonalization = params.ltlm_full_reorth;
        ltlm_params.reorth_frequency = params.ltlm_reorth_freq;
        ltlm_params.random_seed = params.ltlm_seed;
        ltlm_params.store_intermediate = params.ltlm_store_data;
        ltlm_params.max_iterations = params.max_iterations;
        ltlm_params.tolerance = params.tolerance;
        ltlm_params.num_samples = 1;  // LTLM uses single deterministic calculation
        ltlm_params.compute_error_bars = false;
        
        // Extract low temperature range
        std::vector<double> ltlm_temps(temperatures.begin(), temperatures.begin() + crossover_index);
        
        // Run LTLM
        LTLMResults ltlm_results = low_temperature_lanczos(
            H, N, ltlm_params,
            ltlm_temps.front(), ltlm_temps.back(), ltlm_temps.size(),
            nullptr, output_dir
        );
        
        // Store LTLM results
        results.ground_state_energy = ltlm_results.ground_state_energy;
        results.ltlm_points = ltlm_temps.size();
        results.low_lying_spectrum = ltlm_results.low_lying_spectrum;
        
        // Copy LTLM thermodynamic data
        for (size_t i = 0; i < ltlm_temps.size(); i++) {
            results.thermo_data.energy[i] = ltlm_results.thermo_data.energy[i];
            results.thermo_data.specific_heat[i] = ltlm_results.thermo_data.specific_heat[i];
            results.thermo_data.entropy[i] = ltlm_results.thermo_data.entropy[i];
            results.thermo_data.free_energy[i] = ltlm_results.thermo_data.free_energy[i];
            results.energy_error[i] = ltlm_results.energy_error[i];
            results.specific_heat_error[i] = ltlm_results.specific_heat_error[i];
            results.entropy_error[i] = ltlm_results.entropy_error[i];
            results.free_energy_error[i] = ltlm_results.free_energy_error[i];
        }
        
        std::cout << "\n✓ LTLM phase completed successfully\n";
        std::cout << "  Ground state energy: " << results.ground_state_energy << "\n";
        std::cout << "  Low-lying excitations found: " << results.low_lying_spectrum.size() << "\n\n";
    }
    
    // ========================================================================
    // FTLM Phase: High Temperature (T >= T_crossover)
    // ========================================================================
    
    if (crossover_index < num_temp_bins) {
        std::cout << "┌────────────────────────────────────────────┐\n";
        std::cout << "│  Phase 2: FTLM (High Temperature)         │\n";
        std::cout << "└────────────────────────────────────────────┘\n";
        std::cout << "Temperature points: " << (num_temp_bins - crossover_index)
                  << " (T = " << temperatures[crossover_index] << " to " << temperatures.back() << ")\n";
        std::cout << "Random samples: " << params.ftlm_num_samples << "\n\n";
        
        // Setup FTLM parameters
        FTLMParameters ftlm_params;
        ftlm_params.num_samples = params.ftlm_num_samples;
        ftlm_params.krylov_dim = params.ftlm_krylov_dim;
        ftlm_params.full_reorthogonalization = params.ftlm_full_reorth;
        ftlm_params.reorth_frequency = params.ftlm_reorth_freq;
        ftlm_params.random_seed = params.ftlm_seed;
        ftlm_params.store_intermediate = params.ftlm_store_samples;
        ftlm_params.compute_error_bars = params.ftlm_error_bars;
        ftlm_params.max_iterations = params.max_iterations;
        ftlm_params.tolerance = params.tolerance;
        
        // Extract high temperature range
        std::vector<double> ftlm_temps(temperatures.begin() + crossover_index, temperatures.end());
        
        // Run FTLM
        FTLMResults ftlm_results = finite_temperature_lanczos(
            H, N, ftlm_params,
            ftlm_temps.front(), ftlm_temps.back(), ftlm_temps.size(),
            output_dir
        );
        
        // Store FTLM results
        results.ftlm_points = ftlm_temps.size();
        results.ftlm_samples_used = params.ftlm_num_samples;
        
        // If LTLM didn't run, use FTLM's ground state estimate
        if (crossover_index == 0) {
            results.ground_state_energy = ftlm_results.ground_state_estimate;
        }
        
        // Copy FTLM thermodynamic data
        for (size_t i = 0; i < ftlm_temps.size(); i++) {
            int idx = crossover_index + i;
            results.thermo_data.energy[idx] = ftlm_results.thermo_data.energy[i];
            results.thermo_data.specific_heat[idx] = ftlm_results.thermo_data.specific_heat[i];
            results.thermo_data.entropy[idx] = ftlm_results.thermo_data.entropy[i];
            results.thermo_data.free_energy[idx] = ftlm_results.thermo_data.free_energy[i];
            results.energy_error[idx] = ftlm_results.energy_error[i];
            results.specific_heat_error[idx] = ftlm_results.specific_heat_error[i];
            results.entropy_error[idx] = ftlm_results.entropy_error[i];
            results.free_energy_error[idx] = ftlm_results.free_energy_error[i];
        }
        
        std::cout << "\n✓ FTLM phase completed successfully\n";
        std::cout << "  Samples per temperature: " << params.ftlm_num_samples << "\n";
        std::cout << "  Error bars computed: " << (params.ftlm_error_bars ? "Yes" : "No") << "\n\n";
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    std::cout << "================================================\n";
    std::cout << "       Hybrid Thermal Calculation Complete     \n";
    std::cout << "================================================\n";
    std::cout << "Total temperature points: " << num_temp_bins << "\n";
    std::cout << "  • LTLM contribution: " << results.ltlm_points << " points\n";
    std::cout << "  • FTLM contribution: " << results.ftlm_points << " points\n";
    std::cout << "Crossover at index " << crossover_index << " (T = " << actual_crossover << ")\n";
    std::cout << "Ground state energy: " << results.ground_state_energy << "\n";
    std::cout << "================================================\n\n";
    
    return results;
}

/**
 * @brief Save hybrid thermal results to file
 */
void save_hybrid_thermal_results(
    const HybridThermalResults& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Write header with metadata
    file << "# Hybrid Thermal Method Results (LTLM + FTLM)\n";
    file << "# ============================================\n";
    file << "# Ground state energy: " << results.ground_state_energy << "\n";
    file << "# Crossover temperature: " << results.actual_crossover_temp << "\n";
    file << "# Crossover index: " << results.crossover_index << "\n";
    file << "# LTLM temperature points: " << results.ltlm_points << "\n";
    file << "# FTLM temperature points: " << results.ftlm_points << "\n";
    file << "# FTLM samples used: " << results.ftlm_samples_used << "\n";
    file << "#\n";
    file << "# Method assignment:\n";
    file << "#   Indices [0, " << results.crossover_index << "): LTLM (deterministic, accurate ground state)\n";
    file << "#   Indices [" << results.crossover_index << ", " << results.thermo_data.temperatures.size() 
         << "): FTLM (random sampling, error bars)\n";
    file << "#\n";
    file << "# Columns:\n";
    file << "# 1: Temperature\n";
    file << "# 2: Energy\n";
    file << "# 3: Energy_Error\n";
    file << "# 4: Specific_Heat\n";
    file << "# 5: SpecificHeat_Error\n";
    file << "# 6: Entropy\n";
    file << "# 7: Entropy_Error\n";
    file << "# 8: Free_Energy\n";
    file << "# 9: FreeEnergy_Error\n";
    file << "# 10: Method (L=LTLM, F=FTLM)\n";
    file << "#\n";
    
    file << std::scientific << std::setprecision(12);
    
    for (size_t i = 0; i < results.thermo_data.temperatures.size(); i++) {
        char method_flag = (i < static_cast<size_t>(results.crossover_index)) ? 'L' : 'F';
        
        file << results.thermo_data.temperatures[i] << "  "
             << results.thermo_data.energy[i] << "  "
             << results.energy_error[i] << "  "
             << results.thermo_data.specific_heat[i] << "  "
             << results.specific_heat_error[i] << "  "
             << results.thermo_data.entropy[i] << "  "
             << results.entropy_error[i] << "  "
             << results.thermo_data.free_energy[i] << "  "
             << results.free_energy_error[i] << "  "
             << method_flag << "\n";
    }
    
    file.close();
    std::cout << "Hybrid thermal results saved to: " << filename << std::endl;
}

/**
 * @brief Estimate optimal crossover temperature (future enhancement)
 * 
 * Current implementation uses a simple heuristic:
 * T_crossover ≈ (E1 - E0) / 2
 * 
 * where E0 is ground state energy and E1 is first excited state energy.
 * This ensures LTLM is used when ground state dominates the partition function.
 */
double estimate_optimal_crossover(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double ground_energy,
    double first_excitation
) {
    // Energy gap to first excited state
    double gap = first_excitation - ground_energy;
    
    // Heuristic: Use LTLM when k_B T < gap/2
    // This ensures ground state contribution is > 60% of partition function
    double T_opt = gap / 2.0;
    
    // Sanity bounds: crossover should be in reasonable range
    T_opt = std::max(0.01, std::min(T_opt, 10.0));
    
    std::cout << "Estimated optimal crossover temperature: " << T_opt << std::endl;
    std::cout << "  (Based on energy gap: " << gap << ")\n";
    
    return T_opt;
}
