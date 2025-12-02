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
