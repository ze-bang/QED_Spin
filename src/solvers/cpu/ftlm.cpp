// ftlm.cpp - Finite Temperature Lanczos Method implementation
#include <ed/config/env_registry.h>

#include <ed/observables/cf_spectral_kernel.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>
#include <ed/matvec/backends/cpu_backend.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <limits>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <omp.h>

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
            E_weighted_sum += shifted_energy * boltz;                    // moments about e_min:
            E2_weighted_sum += shifted_energy * shifted_energy * boltz;  // no E^2 cancellation
        }
        
        // Store raw values for averaging
        thermo.Z_sample[t] = Z_sample;
        thermo.E_weighted[t] = E_weighted_sum;
        thermo.E2_weighted[t] = E2_weighted_sum;
        
        // Compute derived quantities for this sample
        if (Z_sample > 1e-300) {
            double E_avg = e_min + E_weighted_sum / Z_sample;
            double dE = E_weighted_sum / Z_sample;
            double dE2 = E2_weighted_sum / Z_sample;
            
            // Thermodynamic quantities
            thermo.energy[t] = E_avg;
            thermo.specific_heat[t] = beta * beta * std::max(dE2 - dE * dE, 0.0);
            
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
                // Rescale to the common reference e_min_global: x - e_min_global
                // = (x - e_min_s) + delta for the moments about each sample's e_min.
                double delta_e = sample_data[s].e_min - e_min_global;
                double rescale = std::exp(-beta * delta_e);
                const double z1 = sample_data[s].Z_sample[t];
                const double e1 = sample_data[s].E_weighted[t];
                const double e2 = sample_data[s].E2_weighted[t];
                
                Z_avg[t] += z1 * rescale;
                E_weighted_avg[t] += (e1 + delta_e * z1) * rescale;
                E2_weighted_avg[t] += (e2 + 2.0 * delta_e * e1 + delta_e * delta_e * z1) * rescale;
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
                double dE = E_weighted_avg[t] / Z_avg[t];
                double dE2 = E2_weighted_avg[t] / Z_avg[t];
                double E_avg = e_min_global + dE;
                
                results.thermo_data.energy[t] = E_avg;
                results.thermo_data.specific_heat[t] = beta * beta * std::max(dE2 - dE * dE, 0.0);
                
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

// S(w) = -Im G(w + i*eta) / pi for the continued fraction G(z) = norm_sq / (z - a0 - b1^2 / (z - a1 - ...)) of a
// Lanczos tridiagonal (alpha, beta), evaluated bottom-up.
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
