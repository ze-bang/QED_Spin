#include <ed/solvers/observables.h>


// Calculate thermodynamic quantities directly from eigenvalues
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min,        // Minimum temperature
    double T_max,        // Maximum temperature
    uint64_t num_points        // Number of temperature points
) {
    ThermodynamicData results;
    
    if (eigenvalues.empty()) {
        return results;
    }
    
    // Generate logarithmically spaced temperature points
    results.temperatures.resize(num_points);
    const double log_T_min = std::log(T_min);
    const double log_T_max = std::log(T_max);
    const double log_T_step = (num_points > 1) ? (log_T_max - log_T_min) / (num_points - 1) : 0.0;
    
    for (uint64_t i = 0; i < num_points; i++) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }
    
    // Resize other arrays
    results.energy.resize(num_points);
    results.specific_heat.resize(num_points);
    results.entropy.resize(num_points);
    results.free_energy.resize(num_points);
    
    // Find ground state energy (useful for numerical stability)
    double E0 = *std::min_element(eigenvalues.begin(), eigenvalues.end());
    
    // For each temperature
    for (uint64_t i = 0; i < num_points; i++) {
        double T = results.temperatures[i];
        double beta = 1.0 / T;
        
        // Use log-sum-exp trick for numerical stability in calculating Z
        // Find the maximum value for normalization
        double max_exp = -beta * E0;  // Start with ground state
        
        // Calculate partition function Z and energy using log-sum-exp trick
        double sum_exp = 0.0;
        double sum_E_exp = 0.0;
        double sum_E2_exp = 0.0;
        
        for (double E : eigenvalues) {
            double delta_E = E - E0;
            double exp_term = std::exp(-beta * delta_E);
            
            sum_exp += exp_term;
            sum_E_exp += E * exp_term;
            sum_E2_exp += E * E * exp_term;
        }
        
        // Calculate log(Z) = log(sum_exp) + (-beta*E0)
        double log_Z = std::log(sum_exp) - beta * E0;
        
        // Free energy F = -T * log(Z)
        results.free_energy[i] = -T * log_Z;
        
        // Energy E = (1/Z) * sum_i E_i * exp(-beta*E_i)
        results.energy[i] = sum_E_exp / sum_exp;
        
        // Specific heat C_v = beta^2 * (⟨E^2⟩ - ⟨E⟩^2)
        double avg_E2 = sum_E2_exp / sum_exp;
        double avg_E_squared = results.energy[i] * results.energy[i];
        results.specific_heat[i] = beta * beta * (avg_E2 - avg_E_squared);
        
        // Entropy S = (E - F) / T
        results.entropy[i] = (results.energy[i] - results.free_energy[i]) / T;
    }
    
    // Handle special case for T → 0 (avoid numerical issues)
    if (T_min < 1e-6) {
        // In the limit T → 0, only the ground state contributes
        // Energy → E0
        results.energy[0] = E0;
        
        // Specific heat → 0
        results.specific_heat[0] = 0.0;
        
        // Entropy → 0 (third law of thermodynamics) or ln(g) if g-fold degenerate
        uint64_t degeneracy = 0;
        for (double E : eigenvalues) {
            if (std::abs(E - E0) < 1e-10) degeneracy++;
        }
        results.entropy[0] = (degeneracy > 1) ? std::log(degeneracy) : 0.0;
        
        // Free energy → E0 - TS
        results.free_energy[0] = E0 - results.temperatures[0] * results.entropy[0];
    }
    
    return results;
}

StaticResponseResults compute_connected_qh_from_spectrum(
    const std::vector<double>& eigenvalues,
    const std::vector<double>& q_expectations,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins
) {
    StaticResponseResults results;
    if (eigenvalues.empty() || eigenvalues.size() != q_expectations.size()) {
        return results;
    }
    if (temp_min <= 0.0 || temp_max <= 0.0 || num_temp_bins == 0) {
        throw std::invalid_argument(
            "compute_connected_qh_from_spectrum: temperatures must be positive");
    }

    results.total_samples = 1;
    results.temperatures.resize(num_temp_bins);
    const double log_T_min = std::log(temp_min);
    const double log_T_max = std::log(temp_max);
    const double log_T_step =
        (num_temp_bins > 1) ? (log_T_max - log_T_min) / (num_temp_bins - 1) : 0.0;
    for (uint64_t i = 0; i < num_temp_bins; ++i) {
        results.temperatures[i] = std::exp(log_T_min + i * log_T_step);
    }

    results.expectation.assign(num_temp_bins, 0.0);
    results.variance.assign(num_temp_bins, 0.0);
    results.susceptibility.assign(num_temp_bins, 0.0);
    results.expectation_error.assign(num_temp_bins, 0.0);
    results.variance_error.assign(num_temp_bins, 0.0);
    results.susceptibility_error.assign(num_temp_bins, 0.0);

    const double e_min = *std::min_element(eigenvalues.begin(), eigenvalues.end());

    for (uint64_t t = 0; t < num_temp_bins; ++t) {
        const double T = results.temperatures[t];
        const double beta = 1.0 / T;

        double Z = 0.0;
        double sum_Q = 0.0;
        double sum_H = 0.0;
        double sum_QH = 0.0;

        for (size_t n = 0; n < eigenvalues.size(); ++n) {
            const double boltz = std::exp(-beta * (eigenvalues[n] - e_min));
            const double qn = q_expectations[n];
            const double en = eigenvalues[n];
            Z += boltz;
            sum_Q += boltz * qn;
            sum_H += boltz * en;
            sum_QH += boltz * qn * en;
        }

        if (Z <= 0.0) {
            continue;
        }

        const double inv_Z = 1.0 / Z;
        const double q_avg = sum_Q * inv_Z;
        const double h_avg = sum_H * inv_Z;
        const double qh_avg = sum_QH * inv_Z;
        const double connected = qh_avg - q_avg * h_avg;

        results.variance[t] = connected;
        results.susceptibility[t] = connected / T;
        results.expectation[t] = connected / (T * T);
    }

    return results;
}

// Calculate thermal expectation value of operator A using eigenvalues and eigenvectors
