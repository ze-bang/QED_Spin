#ifndef THERMAL_TYPES_H
#define THERMAL_TYPES_H

#include <vector>

/**
 * @brief Thermodynamic data structure
 * 
 * Common data structure for thermodynamic properties used by both CPU and GPU solvers.
 * This header can be included in both regular C++ and CUDA code.
 */
struct ThermodynamicData {
    std::vector<double> temperatures;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;
    
    // FTLM-specific: raw partition function data for proper averaging
    // Z_sample[t] = Σ_i w_i * exp(-β_t * (E_i - e_min))
    // These are needed to properly average across samples
    std::vector<double> Z_sample;     // Partition function samples (for FTLM averaging)
    std::vector<double> E_weighted;   // Σ_i w_i * E_i * exp(-β*(E_i - e_min)) for energy averaging
    std::vector<double> E2_weighted;  // Σ_i w_i * E_i^2 * exp(-β*(E_i - e_min)) for Cv
    double e_min = 0.0;               // Reference energy (ground state) for this sample
};

/**
 * @brief FTLM results structure
 * 
 * Contains averaged thermodynamic data and error estimates from FTLM calculations.
 * Compatible with both CPU and GPU implementations.
 */
struct FTLMResults {
    ThermodynamicData thermo_data;           // Averaged thermodynamic properties
    std::vector<ThermodynamicData> per_sample_data;  // Per-sample data (if stored)
    std::vector<double> energy_error;        // Standard error in energy
    std::vector<double> specific_heat_error; // Standard error in specific heat
    std::vector<double> entropy_error;       // Standard error in entropy
    std::vector<double> free_energy_error;   // Standard error in free energy
    double ground_state_estimate;            // Best estimate of ground state energy
    uint64_t total_samples;                  // Number of samples used
};

#endif // THERMAL_TYPES_H
