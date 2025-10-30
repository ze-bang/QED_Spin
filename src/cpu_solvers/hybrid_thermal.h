// hybrid_thermal.h - Hybrid Thermal Method (LTLM + FTLM with automatic crossover)
// Standalone method that automatically combines LTLM and FTLM for optimal accuracy

#ifndef HYBRID_THERMAL_H
#define HYBRID_THERMAL_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <cmath>
#include "../core/blas_lapack_wrapper.h"
#include "../core/construct_ham.h"

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

/**
 * @brief Parameters for Hybrid Thermal calculation
 * 
 * The Hybrid Thermal method automatically combines:
 * - LTLM (Low Temperature Lanczos Method) for T < T_crossover
 * - FTLM (Finite Temperature Lanczos Method) for T >= T_crossover
 * 
 * This provides optimal accuracy across the full temperature range:
 * - LTLM captures the ground state correctly at low T
 * - FTLM efficiently samples the full spectrum at high T
 */
struct HybridThermalParameters {
    // Temperature settings
    double crossover_temperature = 1.0;  // Temperature to switch from LTLM to FTLM
    bool auto_crossover = false;         // Automatically determine crossover (future feature)
    
    // LTLM parameters (for T < T_crossover)
    int ltlm_krylov_dim = 200;           // Krylov dimension for excitations
    int ltlm_ground_krylov = 100;        // Krylov dimension for ground state finding
    bool ltlm_full_reorth = false;       // Full reorthogonalization for LTLM
    int ltlm_reorth_freq = 10;           // Reorthogonalization frequency
    unsigned int ltlm_seed = 0;          // Random seed (0 = auto)
    bool ltlm_store_data = false;        // Store intermediate excitation data
    
    // FTLM parameters (for T >= T_crossover)
    int ftlm_num_samples = 20;           // Number of random samples
    int ftlm_krylov_dim = 100;           // Krylov dimension per sample
    bool ftlm_full_reorth = false;       // Full reorthogonalization for FTLM
    int ftlm_reorth_freq = 10;           // Reorthogonalization frequency
    unsigned int ftlm_seed = 0;          // Random seed (0 = auto)
    bool ftlm_store_samples = false;     // Store per-sample data
    bool ftlm_error_bars = true;         // Compute error bars from samples
    
    // General parameters
    int max_iterations = 1000;           // Maximum Lanczos iterations
    double tolerance = 1e-12;            // Convergence tolerance
};

/**
 * @brief Results from Hybrid Thermal calculation
 */
struct HybridThermalResults {
    ThermodynamicData thermo_data;           // Combined thermodynamic properties
    std::vector<double> energy_error;        // Standard error in energy
    std::vector<double> specific_heat_error; // Standard error in specific heat
    std::vector<double> entropy_error;       // Standard error in entropy
    std::vector<double> free_energy_error;   // Standard error in free energy
    
    // Metadata
    double ground_state_energy;              // Ground state energy (from LTLM or FTLM)
    double actual_crossover_temp;            // Actual crossover temperature used
    int ltlm_points;                         // Number of temperature points from LTLM
    int ftlm_points;                         // Number of temperature points from FTLM
    int crossover_index;                     // Index where method switches
    
    // Method-specific data (optional)
    std::vector<double> low_lying_spectrum;  // Low-lying excitations from LTLM
    int ftlm_samples_used;                   // Number of FTLM samples
};

/**
 * @brief Main Hybrid Thermal Method
 * 
 * Automatically combines LTLM and FTLM for optimal thermodynamic calculations:
 * 
 * Algorithm:
 * 1. Generate full temperature grid (temp_min to temp_max)
 * 2. Split grid at crossover temperature T_c
 * 3. For T < T_c: Use LTLM (accurate ground state + low excitations)
 * 4. For T >= T_c: Use FTLM (efficient random sampling)
 * 5. Seamlessly merge results
 * 
 * Advantages:
 * - Low T: LTLM gives exact ground state, no statistical fluctuations
 * - High T: FTLM efficiently samples full spectrum with error bars
 * - Automatic switching at optimal temperature
 * - No discontinuities in thermodynamic functions
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param N Hilbert space dimension
 * @param params Hybrid thermal parameters
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param num_temp_bins Number of temperature points
 * @param output_dir Directory for output files (optional)
 * @return HybridThermalResults containing merged thermodynamic data
 */
HybridThermalResults hybrid_thermal_method(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    const HybridThermalParameters& params,
    double temp_min,
    double temp_max,
    int num_temp_bins,
    const std::string& output_dir = ""
);

/**
 * @brief Save hybrid thermal results to file
 * 
 * Saves merged thermodynamic data with metadata indicating which method
 * was used for each temperature point.
 * 
 * @param results Hybrid thermal results
 * @param filename Output filename
 */
void save_hybrid_thermal_results(
    const HybridThermalResults& results,
    const std::string& filename
);

/**
 * @brief Estimate optimal crossover temperature (future feature)
 * 
 * Analyzes the system to automatically determine the best crossover
 * temperature between LTLM and FTLM based on:
 * - Energy scale (characteristic J)
 * - Gap to first excited state
 * - System size
 * 
 * @param H Hamiltonian matrix-vector product
 * @param N Hilbert space dimension
 * @param ground_energy Ground state energy
 * @param first_excitation First excitation energy
 * @return Estimated optimal crossover temperature
 */
double estimate_optimal_crossover(
    std::function<void(const Complex*, Complex*, int)> H,
    int N,
    double ground_energy,
    double first_excitation
);

#endif // HYBRID_THERMAL_H
