// ftlm.cpp - the continued-fraction spectral function of a Lanczos tridiagonal (the FTLM
// thermodynamics live in include/ed/thermal/ftlm_kernel.h)
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
