/**
 * @file hybrid_thermal.h
 * @brief Hybrid thermal calculation combining LTLM and FTLM methods
 * 
 * This module provides automatic method selection and seamless stitching
 * of Low-Temperature Lanczos Method (LTLM) and Finite-Temperature Lanczos 
 * Method (FTLM) for optimal thermodynamic property calculations across 
 * wide temperature ranges.
 * 
 * Key Features:
 * - Automatic crossover temperature determination
 * - Smooth interpolation between LTLM and FTLM results
 * - Intelligent method selection based on temperature regime
 * - Optimized computational efficiency
 * 
 * @author Copilot
 * @date 2024
 */

#ifndef HYBRID_THERMAL_H
#define HYBRID_THERMAL_H

#include "../core/construct_ham.h"
#include "ltlm.h"
#include "ftlm.h"
#include <vector>

namespace hybrid_thermal {

/**
 * @brief Configuration for hybrid thermal calculation
 */
struct HybridThermalConfig {
    // Temperature range
    double temp_min;
    double temp_max;
    int num_temp_bins;
    
    // LTLM configuration
    LTLMParameters ltlm_params;
    
    // FTLM configuration
    FTLMParameters ftlm_params;
    
    // Hybrid-specific parameters
    double crossover_temp = -1.0;      // Auto-determine if < 0
    int overlap_bins = 10;              // Number of overlapping temperature points
    double ltlm_temp_max_factor = 0.5;  // Maximum temperature for LTLM as fraction of temp_max
    bool auto_method_selection = true;  // Automatically select method based on temperature
    
    // Output control
    bool verbose = false;
    bool store_intermediate = false;    // Store both LTLM and FTLM results separately
};

/**
 * @brief Results from hybrid thermal calculation
 */
struct HybridThermalResults {
    ThermodynamicData combined_data;     // Stitched thermodynamic data
    ThermodynamicData ltlm_data;         // LTLM-only results (if stored)
    ThermodynamicData ftlm_data;         // FTLM-only results (if stored)
    
    double crossover_temp;               // Actual crossover temperature used
    int num_ltlm_bins;                   // Number of temperature bins from LTLM
    int num_ftlm_bins;                   // Number of temperature bins from FTLM
    int num_overlap_bins;                // Number of overlapping bins
    
    bool success = false;
    std::string error_message;
};

/**
 * @brief Determine optimal crossover temperature between LTLM and FTLM
 * 
 * Automatically selects the crossover temperature based on:
 * - Temperature range
 * - System size
 * - Available computational resources
 * - Convergence characteristics
 * 
 * @param temp_min Minimum temperature
 * @param temp_max Maximum temperature
 * @param hilbert_space_dim Hilbert space dimension
 * @param num_eigenstates Number of eigenstates computed by LTLM
 * @return Optimal crossover temperature
 */
double determine_crossover_temperature(
    double temp_min,
    double temp_max,
    size_t hilbert_space_dim,
    int num_eigenstates
);

/**
 * @brief Stitch LTLM and FTLM thermodynamic data
 * 
 * Combines low-temperature LTLM results with high-temperature FTLM results
 * using smooth interpolation in the overlap region.
 * 
 * @param ltlm_data Thermodynamic data from LTLM
 * @param ftlm_data Thermodynamic data from FTLM
 * @param crossover_temp Temperature at which to transition between methods
 * @param overlap_bins Number of temperature bins for smooth interpolation
 * @return Combined thermodynamic data
 */
ThermodynamicData stitch_thermodynamic_data(
    const ThermodynamicData& ltlm_data,
    const ThermodynamicData& ftlm_data,
    double crossover_temp,
    int overlap_bins
);

/**
 * @brief Helper: Smooth interpolation weight function
 * 
 * Computes interpolation weight for smooth transition between methods.
 * Returns 0 at T_min, 1 at T_max, with smooth transition in between.
 * 
 * @param temp Current temperature
 * @param temp_min Start of transition region
 * @param temp_max End of transition region
 * @return Interpolation weight [0,1]
 */
double smooth_interpolation_weight(
    double temp,
    double temp_min,
    double temp_max
);

} // namespace hybrid_thermal

#endif // HYBRID_THERMAL_H
