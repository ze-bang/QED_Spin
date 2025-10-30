/**
 * @file hybrid_thermal.cpp
 * @brief Implementation of hybrid LTLM+FTLM thermal calculations
 */

#include "hybrid_thermal.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace hybrid_thermal {

double determine_crossover_temperature(
    double temp_min,
    double temp_max,
    size_t hilbert_space_dim,
    int num_eigenstates
) {
    // Rule of thumb: LTLM is efficient when k_B*T << (E_max - E_0)
    // where E_max is the highest computed eigenstate
    
    // Estimate: use temperature where ~50% of the Boltzmann weight
    // is outside the computed eigenstates
    
    // Simple heuristic: crossover at T where k_B*T ~ spacing between eigenstates
    // For quantum spin systems, this is typically around 0.1-0.5 * J (exchange coupling)
    
    // Conservative approach: use geometric mean with bounds
    double geometric_mean = std::sqrt(temp_min * temp_max);
    
    // Adjust based on coverage: fraction of Hilbert space covered by eigenstates
    double coverage_fraction = static_cast<double>(num_eigenstates) / hilbert_space_dim;
    
    // If we have good coverage (>1%), we can push crossover higher
    double coverage_factor = 1.0;
    if (coverage_fraction > 0.01) {
        coverage_factor = 1.5;
    } else if (coverage_fraction > 0.001) {
        coverage_factor = 1.2;
    }
    
    double crossover = geometric_mean * coverage_factor;
    
    // Clamp to reasonable range: between 10% and 60% of temp_max
    crossover = std::max(crossover, 0.1 * temp_max);
    crossover = std::min(crossover, 0.6 * temp_max);
    
    return crossover;
}

std::vector<double> create_ltlm_temperature_grid(
    double temp_min,
    double temp_max,
    int num_bins,
    bool log_spacing
) {
    std::vector<double> temps(num_bins);
    
    if (log_spacing && temp_min > 0) {
        // Logarithmic spacing for low temperatures
        double log_min = std::log(temp_min);
        double log_max = std::log(temp_max);
        double dlog = (log_max - log_min) / (num_bins - 1);
        
        for (int i = 0; i < num_bins; ++i) {
            temps[i] = std::exp(log_min + i * dlog);
        }
    } else {
        // Linear spacing
        double dt = (temp_max - temp_min) / (num_bins - 1);
        for (int i = 0; i < num_bins; ++i) {
            temps[i] = temp_min + i * dt;
        }
    }
    
    return temps;
}

std::vector<double> create_ftlm_temperature_grid(
    double temp_min,
    double temp_max,
    int num_bins
) {
    std::vector<double> temps(num_bins);
    double dt = (temp_max - temp_min) / (num_bins - 1);
    
    for (int i = 0; i < num_bins; ++i) {
        temps[i] = temp_min + i * dt;
    }
    
    return temps;
}

double smooth_interpolation_weight(
    double temp,
    double temp_min,
    double temp_max
) {
    if (temp <= temp_min) return 0.0;
    if (temp >= temp_max) return 1.0;
    
    // Use smooth Hermite interpolation (3rd order polynomial)
    // Ensures C^1 continuity (smooth first derivative)
    double x = (temp - temp_min) / (temp_max - temp_min);
    return x * x * (3.0 - 2.0 * x);
}

ThermodynamicData stitch_thermodynamic_data(
    const ThermodynamicData& ltlm_data,
    const ThermodynamicData& ftlm_data,
    double crossover_temp,
    int overlap_bins
) {
    ThermodynamicData result;
    
    // Find indices for overlap region
    // Overlap region: [crossover_temp - delta, crossover_temp + delta]
    
    const auto& ltlm_temps = ltlm_data.temperatures;
    const auto& ftlm_temps = ftlm_data.temperatures;
    
    if (ltlm_temps.empty() || ftlm_temps.empty()) {
        std::cerr << "Error: Empty temperature arrays in stitch_thermodynamic_data" << std::endl;
        return result;
    }
    
    // Determine overlap region width
    double temp_span = std::min(
        ltlm_temps.back() - ltlm_temps.front(),
        ftlm_temps.back() - ftlm_temps.front()
    );
    double overlap_width = temp_span * 0.2; // 20% overlap
    
    double overlap_min = crossover_temp - overlap_width / 2;
    double overlap_max = crossover_temp + overlap_width / 2;
    
    // Build combined temperature array
    std::vector<double> combined_temps;
    std::vector<double> combined_energy;
    std::vector<double> combined_entropy;
    std::vector<double> combined_cv;
    std::vector<double> combined_free_energy;
    
    // Add LTLM data (below overlap region)
    for (size_t i = 0; i < ltlm_temps.size(); ++i) {
        double T = ltlm_temps[i];
        
        if (T < overlap_min) {
            // Pure LTLM region
            combined_temps.push_back(T);
            combined_energy.push_back(ltlm_data.energy[i]);
            combined_entropy.push_back(ltlm_data.entropy[i]);
            combined_cv.push_back(ltlm_data.specific_heat[i]);
            combined_free_energy.push_back(ltlm_data.free_energy[i]);
        } else if (T <= overlap_max) {
            // Overlap region: interpolate
            // Find corresponding FTLM point
            auto ftlm_it = std::lower_bound(ftlm_temps.begin(), ftlm_temps.end(), T);
            if (ftlm_it != ftlm_temps.end()) {
                size_t ftlm_idx = ftlm_it - ftlm_temps.begin();
                
                // Interpolation weight (0 = LTLM, 1 = FTLM)
                double w = smooth_interpolation_weight(T, overlap_min, overlap_max);
                
                combined_temps.push_back(T);
                combined_energy.push_back(
                    (1 - w) * ltlm_data.energy[i] + w * ftlm_data.energy[ftlm_idx]
                );
                combined_entropy.push_back(
                    (1 - w) * ltlm_data.entropy[i] + w * ftlm_data.entropy[ftlm_idx]
                );
                combined_cv.push_back(
                    (1 - w) * ltlm_data.specific_heat[i] + w * ftlm_data.specific_heat[ftlm_idx]
                );
                combined_free_energy.push_back(
                    (1 - w) * ltlm_data.free_energy[i] + w * ftlm_data.free_energy[ftlm_idx]
                );
            }
        }
    }
    
    // Add FTLM data (above overlap region)
    for (size_t i = 0; i < ftlm_temps.size(); ++i) {
        double T = ftlm_temps[i];
        
        if (T > overlap_max) {
            // Pure FTLM region
            combined_temps.push_back(T);
            combined_energy.push_back(ftlm_data.energy[i]);
            combined_entropy.push_back(ftlm_data.entropy[i]);
            combined_cv.push_back(ftlm_data.specific_heat[i]);
            combined_free_energy.push_back(ftlm_data.free_energy[i]);
        }
    }
    
    // Store in result
    result.temperatures = combined_temps;
    result.energy = combined_energy;
    result.entropy = combined_entropy;
    result.specific_heat = combined_cv;
    result.free_energy = combined_free_energy;
    
    return result;
}

// NOTE: compute_hybrid_thermodynamics is temporarily disabled pending proper integration
// with the Hamiltonian construction infrastructure. To use hybrid mode, implement a
// wrapper that constructs the Hamiltonian matrix-vector product function from your
// specific Hamiltonian representation.
//
// Example usage pattern:
// auto H = [&your_hamiltonian](const Complex* in, Complex* out, int N) {
//     // Your matrix-vector product implementation
// };
// Then call LTLM and FTLM separately and stitch results using stitch_thermodynamic_data()

} // namespace hybrid_thermal
