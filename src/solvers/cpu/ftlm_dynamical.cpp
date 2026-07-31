// ftlm.cpp - Finite Temperature Lanczos Method implementation
#include <ed/core/system_utils.h>
#include <ed/core/hdf5_io.h>       // For HDF5 output
#include <ed/parallel/thread_budget.h>  // Phase 6.1: dim-aware OMP+BLAS cap

#include <ed/observables/cf_spectral_kernel.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ftlm_dist.h>
#include <ed/solvers/lanczos.h>
#include <ed/krylov/lanczos_tridiag.h>
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
#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace {
// Gate per-sample / per-iteration progress prints in DSSF/SSSF/FTLM
// kernels behind ED_DSSF_VERBOSE=1. Errors and final summaries always
// print. Mirrors the ED_LANCZOS_VERBOSE pattern in lanczos.cpp.
inline bool ed_dssf_verbose() {
    static const bool v = []() {
        const char* env = std::getenv("ED_DSSF_VERBOSE");
        return env && env[0] == '1';
    }();
    return v;
}
} // namespace

// ftlm_dynamical.cpp - split out of ftlm.cpp (architecture hardening D2).
// Dynamical / spectral / static-response FTLM kernels. The core-thermo
// helpers it calls (build_lanczos_tridiagonal, ...) are header-declared
// (external linkage) and defined in ftlm.cpp; the file-static spectral
// helpers used only here moved with these functions.

// ---------------------------------------------------------------------------
// Family-3 retirement (audit 2026-07-31): the Gen-1 dynamical-DSSF block
// that lived here -- compute_spectral_function[_complex] (file-static),
// compute_dynamical_response_thermal, compute_dynamical_correlation, and
// compute_dynamical_correlation_state_cf -- is deleted. Every consumer
// routes through the backend-generic
// ed::observables::ftlm_dynamical_kernel_via_backend[_multitemp]
// (include/ed/observables/cf_dynamical.h), which was gated equivalent to
// this block at matching T (~5 decimals, Family-3 step 3) and serves CPU
// and CUDA from one body. The static / connected-response functions below
// are LIVE (dM/dT campaign) and are not part of the retirement.
// ---------------------------------------------------------------------------


/**
 * @brief Helper function to compute expectation values in Krylov basis
 */
static void compute_krylov_expectation_values(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& v0,
    uint64_t N,
    uint64_t krylov_dim,
    double tolerance,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& ritz_values,
    std::vector<double>& weights,
    std::vector<double>& expectation_values
) {
    // Build Lanczos tridiagonal for Hamiltonian with basis storage
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;
    
    uint64_t iterations = build_lanczos_tridiagonal_with_basis(
        H, v0, N, krylov_dim, tolerance,
        full_reorth, reorth_freq,
        alpha, beta, &lanczos_vectors
    );
    
    uint64_t m = alpha.size();
    
    // Diagonalize tridiagonal (need eigenvectors for expectation value computation)
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
    
    if (ritz_values.empty()) {
        std::cerr << "Warning: Tridiagonal diagonalization failed" << std::endl;
        return;
    }
    
    expectation_values.resize(m);
    
    // Now compute <n|O|n> for each Ritz state in the Krylov basis
    // |n> = Σ_j evecs[n,j] |v_j>
    for (int n = 0; n < m; n++) {
        // Reconstruct |n> in full Hilbert space
        ComplexVector psi_n(N, Complex(0.0, 0.0));
        for (int j = 0; j < m; j++) {
            double coeff = evecs[n * m + j];
            Complex alpha(coeff, 0.0);
            cblas_zaxpy(N, &alpha, lanczos_vectors[j].data(), 1, psi_n.data(), 1);
        }
        
        // Apply O to |n>
        ComplexVector O_psi_n(N);
        O(psi_n.data(), O_psi_n.data(), N);
        
        // Compute <n|O|n>
        Complex expectation_complex;
        cblas_zdotc_sub(N, psi_n.data(), 1, O_psi_n.data(), 1, &expectation_complex);
        expectation_values[n] = std::real(expectation_complex);
    }
}

/**
 * @brief Compute thermal expectation value (single operator)
 */
StaticResponseResults compute_thermal_expectation_value(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir
) {
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Thermal Expectation Value (FTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
        std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    }

    // Logarithmic temperature grid — T must be strictly positive at both
    // ends, otherwise log() returns -inf/NaN and 1/T below diverges.
    if (!(temp_min > 0.0) || !(temp_max > 0.0)) {
        throw std::invalid_argument(
            "compute_thermal_expectation_value: temp_min and temp_max must "
            "both be > 0 (got temp_min=" + std::to_string(temp_min) +
            ", temp_max=" + std::to_string(temp_max) + ").");
    }

    StaticResponseResults results;
    results.total_samples = params.num_samples;

    // Generate temperature grid (logarithmic spacing)
    results.temperatures.resize(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);

    for (int i = 0; i < num_temp_bins; i++) {
        results.temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for per-sample thermal averages
    std::vector<std::vector<double>> sample_expectations(params.num_samples);
    std::vector<std::vector<double>> sample_variances(params.num_samples);
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_samples";
        safe_system_call(cmd);
    }
    
    // Loop over samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        if (verbose) {
            std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        }

        // Generate random initial state (Gaussian for unbiased trace estimate)
        ComplexVector v0 = generateGaussianRandomVector(N, gen);

        // Build Krylov subspace and compute expectation values
        std::vector<double> ritz_values, weights, expectation_values;
        compute_krylov_expectation_values(
            H, O, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            ritz_values, weights, expectation_values
        );

        uint64_t m = ritz_values.size();
        if (verbose) {
            std::cout << "  Krylov subspace size: " << m << std::endl;
        }
        
        if (m == 0) {
            std::cerr << "  Warning: Failed to build Krylov subspace, skipping sample\n";
            continue;
        }
        
        // Store sample data if requested
        if (params.store_intermediate) {
            StaticResponseSample sample_data;
            sample_data.ritz_values = ritz_values;
            sample_data.weights = weights;
            sample_data.expectation_values = expectation_values;
            sample_data.lanczos_iterations = m;
            results.per_sample_data.push_back(sample_data);
        }
        
        // Compute thermal averages for this sample.
        //
        // Fused single-pass loop per (sample, T): one std::exp evaluation
        // per Ritz state, accumulating Z, ⟨O⟩, ⟨O²⟩ together. Old code
        // had two passes per T plus a per-T m-sized boltzmann_factors heap
        // allocation; new code is single-pass and allocation-free.
        sample_expectations[sample].resize(num_temp_bins);
        sample_variances[sample].resize(num_temp_bins);

        // Energy shift (subtract minimum) prevents Boltzmann overflow at low T.
        const double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        const uint64_t gs_idx = static_cast<uint64_t>(std::distance(
            ritz_values.begin(),
            std::min_element(ritz_values.begin(), ritz_values.end())));

        for (int t = 0; t < num_temp_bins; t++) {
            const double T = results.temperatures[t];
            const double beta = 1.0 / T;  // T > 0 enforced above.

            double Z = 0.0;
            double sum_O  = 0.0;
            double sum_O2 = 0.0;
            for (uint64_t i = 0; i < m; ++i) {
                const double bw = weights[i] * std::exp(-beta * (ritz_values[i] - e_min));
                const double oi = expectation_values[i];
                Z      += bw;
                sum_O  += bw * oi;
                sum_O2 += bw * oi * oi;
            }

            if (Z > 1e-300) {
                const double inv_Z = 1.0 / Z;
                const double O_avg  = sum_O  * inv_Z;
                const double O2_avg = sum_O2 * inv_Z;
                sample_expectations[sample][t] = O_avg;
                sample_variances[sample][t]    = O2_avg - O_avg * O_avg;
            } else {
                // Z ≈ 0: temperature is so low that even shifted Boltzmann
                // factors underflowed; fall back to the ground-state value.
                sample_expectations[sample][t] = expectation_values[gs_idx];
                sample_variances[sample][t] = 0.0;
            }
        }
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMStaticSample h5_sample;
            h5_sample.temperatures = results.temperatures;
            h5_sample.expectation = sample_expectations[sample];
            h5_sample.variance = sample_variances[sample];
            
            HDF5IO::saveFTLMStaticSample(h5_file, sample, h5_sample);
        }
    }
    
    // Average over all samples
    uint64_t n_valid_samples = 0;
    for (int s = 0; s < params.num_samples; s++) {
        if (!sample_expectations[s].empty()) n_valid_samples++;
    }
    
    if (verbose) {
        std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    }

    results.expectation.resize(num_temp_bins, 0.0);
    results.variance.resize(num_temp_bins, 0.0);
    results.susceptibility.resize(num_temp_bins, 0.0);
    results.expectation_error.resize(num_temp_bins, 0.0);
    results.variance_error.resize(num_temp_bins, 0.0);
    results.susceptibility_error.resize(num_temp_bins, 0.0);

    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained" << std::endl;
        return results;
    }

    // Sample-mean accumulation
    for (int s = 0; s < params.num_samples; s++) {
        if (sample_expectations[s].empty()) continue;
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation[t] += sample_expectations[s][t];
            results.variance[t] += sample_variances[s][t];
        }
    }

    for (int t = 0; t < num_temp_bins; t++) {
        results.expectation[t] /= n_valid_samples;
        results.variance[t] /= n_valid_samples;
        // Susceptibility χ = β * variance
        double beta = 1.0 / results.temperatures[t];
        results.susceptibility[t] = beta * results.variance[t];
    }

    // Standard errors of the sample means
    if (params.compute_error_bars && n_valid_samples > 1) {
        for (int s = 0; s < params.num_samples; s++) {
            if (sample_expectations[s].empty()) continue;
            for (int t = 0; t < num_temp_bins; t++) {
                double diff_exp = sample_expectations[s][t] - results.expectation[t];
                double diff_var = sample_variances[s][t] - results.variance[t];

                results.expectation_error[t] += diff_exp * diff_exp;
                results.variance_error[t] += diff_var * diff_var;
            }
        }

        double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation_error[t] = std::sqrt(results.expectation_error[t]) / norm;
            results.variance_error[t] = std::sqrt(results.variance_error[t]) / norm;

            // Error propagation for susceptibility: δχ ≈ β * δ(variance)
            double beta = 1.0 / results.temperatures[t];
            results.susceptibility_error[t] = beta * results.variance_error[t];
        }
    }

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Static Response Complete\n";
        std::cout << "==========================================\n";
    }

    return results;
}

/**
 * @brief Compute static response function (two-operator correlation)
 */
StaticResponseResults compute_static_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir
) {
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Static Response Function (FTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Computing correlation ⟨O₁†O₂⟩\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    }

    StaticResponseResults results;
    results.total_samples = params.num_samples;

    // Generate temperature grid
    results.temperatures.resize(num_temp_bins);
    double temp_step = (temp_max - temp_min) / std::max(uint64_t(1), num_temp_bins - 1);
    for (int i = 0; i < num_temp_bins; i++) {
        results.temperatures[i] = temp_min + i * temp_step;
    }

    // χ = β·variance and Boltzmann e^{-βE} both diverge at T=0; refuse
    // here rather than silently producing inf/NaN downstream. Callers
    // wanting the T→0 limit should request the lowest physically
    // sensible T (or use the ground-state path).
    for (int i = 0; i < num_temp_bins; ++i) {
        if (!(results.temperatures[i] > 0.0)) {
            throw std::invalid_argument(
                "compute_static_response: temperature grid contains a "
                "non-positive value at index " + std::to_string(i) +
                " (T = " + std::to_string(results.temperatures[i]) +
                "). Use a strictly positive temperature range.");
        }
    }
    
    // Initialize random number generator
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // Storage for per-sample results
    std::vector<std::vector<double>> sample_expectations;
    std::vector<std::vector<double>> sample_variances;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_correlation_samples";
        safe_system_call(cmd);
    }
    
    // Pre-allocate per-Ritz scratch buffers reused across samples and Ritz
    // states. Old code allocated psi_n / O1_psi_n / O2_psi_n inside the
    // inner n-loop ⇒ 3 × N-sized vectors per Ritz state per sample.
    ComplexVector psi_n(N);
    ComplexVector O1_psi_n(N);
    ComplexVector O2_psi_n(N);

    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        if (verbose) {
            std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        }

        // Generate random initial state (Gaussian for unbiased trace estimate)
        ComplexVector v0 = generateGaussianRandomVector(N, gen);

        // Build Lanczos tridiagonal for Hamiltonian (store basis vectors)
        std::vector<double> alpha, beta;
        std::vector<ComplexVector> lanczos_vectors;

        uint64_t iterations = build_lanczos_tridiagonal_with_basis(
            H, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta, &lanczos_vectors
        );

        uint64_t m = alpha.size();
        if (verbose) {
            std::cout << "  Lanczos iterations: " << m << std::endl;
        }

        // Diagonalize tridiagonal (need eigenvectors for correlation computation)
        std::vector<double> ritz_values, weights;
        std::vector<double> evecs;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);

        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed" << std::endl;
            continue;
        }

        std::vector<double> correlation_values(m);

        // Compute ⟨n|O₁†O₂|n⟩ for each eigenstate |n⟩
        // This equals ⟨O₁n|O₂n⟩ = (O₁|n⟩)† · (O₂|n⟩)
        for (uint64_t n = 0; n < m; n++) {
            // Reconstruct |n⟩ in full space: |n⟩ = Σ_j evecs[n,j] |v_j⟩
            std::fill(psi_n.begin(), psi_n.end(), Complex(0.0, 0.0));
            for (uint64_t j = 0; j < m; j++) {
                const Complex coeff(evecs[n * m + j], 0.0);
                cblas_zaxpy(N, &coeff, lanczos_vectors[j].data(), 1,
                            psi_n.data(), 1);
            }

            // Apply O₁ and O₂
            O1(psi_n.data(), O1_psi_n.data(), N);
            O2(psi_n.data(), O2_psi_n.data(), N);

            // Compute ⟨O₁n|O₂n⟩ = ⟨n|O₁†O₂|n⟩
            Complex correlation_complex;
            cblas_zdotc_sub(N, O1_psi_n.data(), 1,
                            O2_psi_n.data(), 1, &correlation_complex);
            correlation_values[n] = std::real(correlation_complex);
        }
        
        // Compute thermal averages for this sample.
        //
        // Old version: three independent for-i loops per temperature, each
        // calling std::exp(-β·ΔE) ⇒ 3 × m exp evaluations per (sample, T).
        // New version: a single fused pass that evaluates the Boltzmann
        // factor once per (sample, T, i) and accumulates Z, ⟨O⟩, ⟨O²⟩ in
        // one sweep. ⟨O²⟩ - ⟨O⟩² is computed in the standard normalized
        // form with a single division by Z.
        std::vector<double> sample_exp(num_temp_bins);
        std::vector<double> sample_var(num_temp_bins);

        // Shift energies by e_min to prevent Boltzmann factor overflow at low T
        const double e_min = *std::min_element(ritz_values.begin(),
                                               ritz_values.begin() + m);

        for (int t = 0; t < num_temp_bins; t++) {
            const double T = results.temperatures[t];
            const double beta = 1.0 / T;  // T > 0 guaranteed by validation above.

            double Z = 0.0;
            double sum_O  = 0.0;
            double sum_O2 = 0.0;
            for (uint64_t i = 0; i < m; i++) {
                const double bw = weights[i] * std::exp(-beta * (ritz_values[i] - e_min));
                const double ci = correlation_values[i];
                Z      += bw;
                sum_O  += bw * ci;
                sum_O2 += bw * ci * ci;
            }

            const double inv_Z = (Z > 1e-300) ? (1.0 / Z) : 0.0;
            const double expectation         = sum_O  * inv_Z;
            const double expectation_squared = sum_O2 * inv_Z;
            sample_exp[t] = expectation;
            sample_var[t] = expectation_squared - expectation * expectation;
        }
        
        sample_expectations.push_back(sample_exp);
        sample_variances.push_back(sample_var);
        
        // Store per-sample data if requested
        if (params.store_intermediate) {
            StaticResponseSample sample_data;
            sample_data.ritz_values = ritz_values;
            sample_data.weights = weights;
            sample_data.expectation_values = correlation_values;
            sample_data.lanczos_iterations = m;
            results.per_sample_data.push_back(sample_data);
        }
    }
    
    // Average over samples
    uint64_t n_valid_samples = sample_expectations.size();
    if (verbose) {
        std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    }

    results.expectation.resize(num_temp_bins, 0.0);
    results.variance.resize(num_temp_bins, 0.0);
    results.expectation_error.resize(num_temp_bins, 0.0);
    results.variance_error.resize(num_temp_bins, 0.0);
    results.susceptibility.resize(num_temp_bins, 0.0);
    results.susceptibility_error.resize(num_temp_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples" << std::endl;
        return results;
    }
    
    // Compute means
    for (int s = 0; s < n_valid_samples; s++) {
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation[t] += sample_expectations[s][t];
            results.variance[t] += sample_variances[s][t];
        }
    }
    
    for (int t = 0; t < num_temp_bins; t++) {
        results.expectation[t] /= n_valid_samples;
        results.variance[t] /= n_valid_samples;
        double beta = 1.0 / results.temperatures[t];
        results.susceptibility[t] = beta * results.variance[t];
    }
    
    // Compute standard errors
    if (params.compute_error_bars && n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int t = 0; t < num_temp_bins; t++) {
                double diff_exp = sample_expectations[s][t] - results.expectation[t];
                double diff_var = sample_variances[s][t] - results.variance[t];
                
                results.expectation_error[t] += diff_exp * diff_exp;
                results.variance_error[t] += diff_var * diff_var;
            }
        }
        
        double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation_error[t] = std::sqrt(results.expectation_error[t]) / norm;
            results.variance_error[t] = std::sqrt(results.variance_error[t]) / norm;
            
            double beta = 1.0 / results.temperatures[t];
            results.susceptibility_error[t] = beta * results.variance_error[t];
        }
    }

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Static Response Complete\n";
        std::cout << "==========================================\n";
    }

    return results;
}

/**
 * @brief Compute ∂T⟨O⟩ = (⟨OH⟩ - ⟨O⟩⟨H⟩) / T² from one FTLM Krylov basis.
 */
StaticResponseResults compute_connected_qh_response(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const StaticResponseParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir
) {
    const bool verbose = ed_dssf_verbose();

    if (!(temp_min > 0.0) || !(temp_max > 0.0)) {
        throw std::invalid_argument(
            "compute_connected_qh_response: temp_min and temp_max must both be > 0.");
    }

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Connected Q-H Response (FTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Computing (⟨OH⟩ - ⟨O⟩⟨H⟩) / T²\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
    }

    StaticResponseResults results;
    results.total_samples = params.num_samples;

    results.temperatures.resize(num_temp_bins);
    const double log_tmin = std::log(temp_min);
    const double log_tmax = std::log(temp_max);
    const double log_step = (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);
    for (int i = 0; i < num_temp_bins; i++) {
        results.temperatures[i] = std::exp(log_tmin + i * log_step);
    }

    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }

    std::vector<std::vector<double>> sample_alpha;
    std::vector<std::vector<double>> sample_connected;

    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/static_connected_qh_samples";
        safe_system_call(cmd);
    }

    ComplexVector psi_n(N);
    ComplexVector O_psi_n(N);

    for (int sample = 0; sample < params.num_samples; sample++) {
        if (verbose) {
            std::cout << "\n--- Sample " << sample + 1 << " / "
                      << params.num_samples << " ---\n";
        }

        ComplexVector v0 = generateGaussianRandomVector(N, gen);

        std::vector<double> alpha, beta;
        std::vector<ComplexVector> lanczos_vectors;
        build_lanczos_tridiagonal_with_basis(
            H, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta, &lanczos_vectors
        );

        const uint64_t m = alpha.size();
        if (m == 0) {
            std::cerr << "  Warning: Failed to build Krylov subspace, skipping sample\n";
            continue;
        }

        std::vector<double> ritz_values, weights;
        std::vector<double> evecs;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed, skipping sample\n";
            continue;
        }

        std::vector<double> q_values(m, 0.0);
        for (uint64_t n = 0; n < m; n++) {
            std::fill(psi_n.begin(), psi_n.end(), Complex(0.0, 0.0));
            for (uint64_t j = 0; j < m; j++) {
                const Complex coeff(evecs[n * m + j], 0.0);
                cblas_zaxpy(N, &coeff, lanczos_vectors[j].data(), 1,
                            psi_n.data(), 1);
            }

            O(psi_n.data(), O_psi_n.data(), N);

            Complex q_complex;
            cblas_zdotc_sub(N, psi_n.data(), 1, O_psi_n.data(), 1, &q_complex);
            q_values[n] = std::real(q_complex);
        }

        std::vector<double> alpha_T(num_temp_bins, 0.0);
        std::vector<double> connected_T(num_temp_bins, 0.0);
        const double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());

        for (int t = 0; t < num_temp_bins; t++) {
            const double T = results.temperatures[t];
            const double beta_T = 1.0 / T;

            double Z = 0.0;
            double sum_Q = 0.0;
            double sum_H = 0.0;
            double sum_QH = 0.0;
            for (uint64_t i = 0; i < m; i++) {
                const double bw = weights[i] * std::exp(-beta_T * (ritz_values[i] - e_min));
                const double qi = q_values[i];
                const double ei = ritz_values[i];
                Z      += bw;
                sum_Q  += bw * qi;
                sum_H  += bw * ei;
                sum_QH += bw * qi * ei;
            }

            if (Z > 1e-300) {
                const double inv_Z = 1.0 / Z;
                const double q_avg = sum_Q * inv_Z;
                const double h_avg = sum_H * inv_Z;
                const double qh_avg = sum_QH * inv_Z;
                const double connected = qh_avg - q_avg * h_avg;
                connected_T[t] = connected;
                alpha_T[t] = connected / (T * T);
            }
        }

        sample_alpha.push_back(alpha_T);
        sample_connected.push_back(connected_T);
    }

    const uint64_t n_valid_samples = sample_alpha.size();
    results.expectation.resize(num_temp_bins, 0.0);
    results.variance.resize(num_temp_bins, 0.0);
    results.susceptibility.resize(num_temp_bins, 0.0);
    results.expectation_error.resize(num_temp_bins, 0.0);
    results.variance_error.resize(num_temp_bins, 0.0);
    results.susceptibility_error.resize(num_temp_bins, 0.0);

    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained" << std::endl;
        return results;
    }

    for (uint64_t s = 0; s < n_valid_samples; s++) {
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation[t] += sample_alpha[s][t];
            results.variance[t] += sample_connected[s][t];
        }
    }

    for (int t = 0; t < num_temp_bins; t++) {
        results.expectation[t] /= n_valid_samples;
        results.variance[t] /= n_valid_samples;
        results.susceptibility[t] = results.variance[t] / results.temperatures[t];
    }

    if (params.compute_error_bars && n_valid_samples > 1) {
        for (uint64_t s = 0; s < n_valid_samples; s++) {
            for (int t = 0; t < num_temp_bins; t++) {
                const double diff_alpha = sample_alpha[s][t] - results.expectation[t];
                const double diff_conn = sample_connected[s][t] - results.variance[t];
                results.expectation_error[t] += diff_alpha * diff_alpha;
                results.variance_error[t] += diff_conn * diff_conn;
            }
        }

        const double norm = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int t = 0; t < num_temp_bins; t++) {
            results.expectation_error[t] = std::sqrt(results.expectation_error[t]) / norm;
            results.variance_error[t] = std::sqrt(results.variance_error[t]) / norm;
            results.susceptibility_error[t] = results.variance_error[t] / results.temperatures[t];
        }
    }

    return results;
}

/**
 * @brief Save static response to text file in unified format
 * 
 * Unified format: 8 columns
 *   # T  <O>  <O>_err  Var  Var_err  chi  chi_err  N_samples
 * 
// save_static_response_results was retired in the minimalist-architecture
// rev (May 2026): no external callers. Workflows that need to persist a
// StaticResponseResults go through HDF5IO::saveStaticResponse directly.

// ============================================================================
// TEMPERATURE-INDEPENDENT SPECTRAL DECOMPOSITION -- RETIRED (Jul 2026)
// ============================================================================
// The `compute_lanczos_spectral_data` / `compute_spectral_function_from_
// lanczos_data` pair and `compute_dynamical_correlation_state_multi_
// temperature` had no callers left after the Family 3 unification onto
// `ftlm_dynamical_kernel_via_backend`; deleted in the follow-up sweep.

// ============================================================================
// CORRECTED FTLM MULTI-SAMPLE MULTI-TEMPERATURE SPECTRAL FUNCTION
// ============================================================================

/**
 * @brief Multi-sample multi-temperature dynamical correlation (CORRECTED FTLM!)
 * 
 * CORRECTED VERSION: This implementation properly handles thermal averaging
 * for FTLM spectral functions. The key insight is that for random state 
 * sampling, we don't apply Boltzmann weights to the spectral peaks directly.
 * Instead, the random sampling itself provides the thermal averaging through
 * the trace identity: Tr[A] = N × E_r[⟨r|A|r⟩] where |r⟩ are random states.
 * 
 * For T→0 (low temperature limit), the spectral function should match the
 * ground state result. This is achieved because random states have some
 * overlap with the ground state.
 * 
 * For finite T, the formula becomes:
 *   S(ω,T) ∝ Tr[e^{-βH} O†δ(ω-H+E₀)O] / Z
 *          = (N/Z) × E_r[⟨r|e^{-βH} O†δ(ω-H+E₀)O|r⟩]
 * 
 * The key correction is to not double-apply thermal weights. The spectral
 * weights already capture the transition amplitudes - we only need the
 * thermal prefactor from the partition function.
 */
// ============================================================================
// MULTI-OPERATOR FTLM CORE: shares the per-sample H-Lanczos chain (and the
// cached Ritz eigenstates |psi_i>) across all (O1[p], O2[p]) pairs. This is
// the single source of truth for the FTLM Lehmann machinery; the historical
// single-pair implementation (a byte-for-byte P == 1 specialisation of this
// body) was deleted in the debt-cleanup sweep (Jul 2026) and its entry point
// now delegates here (see the wrapper right below the core).
// ============================================================================
static std::vector<std::map<double, DynamicalResponseResults>>
compute_dynamical_correlation_multi_operator_multi_temperature_impl(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O1_list,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O2_list,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir
#ifdef WITH_MPI
    , MPI_Comm comm
#endif
) {
    if (O1_list.size() != O2_list.size()) {
        throw std::invalid_argument(
            "compute_dynamical_correlation_multi_operator_multi_temperature: "
            "O1_list and O2_list must have the same size");
    }
    const size_t P = O1_list.size();
    if (P == 0) {
        return {};
    }

    const bool verbose = ed_dssf_verbose();
    int mpi_rank_early = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(comm, &mpi_rank_early);
#endif

    for (double T : temperatures) {
        if (!(T > 0.0)) {
            throw std::invalid_argument(
                "compute_dynamical_correlation_multi_operator_multi_temperature: "
                "temperatures must be > 0 (got T = " + std::to_string(T) + ")");
        }
    }

    if (verbose && mpi_rank_early == 0) {
        std::cout << "\n=========================================="  << std::endl;
        std::cout << "FTLM SPECTRAL (MULTI-OPERATOR, SHARED H-LANCZOS)" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Operator pairs: " << P << std::endl;
        std::cout << "Samples:        " << params.num_samples << std::endl;
        std::cout << "Temperatures:   " << temperatures.size() << std::endl;
        std::cout << "Krylov dim:     " << params.krylov_dim << std::endl;
        std::cout << "Broadening:     " << params.broadening << std::endl;
        std::cout << "==========================================" << std::endl;
    }

    // RNG / E_gs setup mirrors the per-pair entry point.
    std::mt19937 gen;
    if (params.random_seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        gen.seed(params.random_seed);
    }
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double E_gs = energy_shift;
    if (std::abs(E_gs) < 1e-14) {
        ComplexVector test_state(N);
        const char* env_complex_seed = std::getenv("ED_LANCZOS_COMPLEX_SEED");
        const bool complex_seed = (env_complex_seed && env_complex_seed[0] == '1');
        for (uint64_t i = 0; i < N; i++) {
            test_state[i] = complex_seed ? Complex(dist(gen), dist(gen))
                                         : Complex(dist(gen), 0.0);
        }
        double norm = cblas_dznrm2(N, test_state.data(), 1);
        Complex scale(1.0/norm, 0.0);
        cblas_zscal(N, &scale, test_state.data(), 1);
        std::vector<double> a, b;
        build_lanczos_tridiagonal(H, test_state, N,
                                  std::min(params.krylov_dim, (uint64_t)100),
                                  params.tolerance, false, 10, a, b);
        std::vector<double> rv, w;
        diagonalize_tridiagonal_ritz(a, b, rv, w);
        if (!rv.empty()) E_gs = *std::min_element(rv.begin(), rv.end());
    }

    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) /
                        std::max(uint64_t(1), num_omega_bins - 1);
    for (uint64_t i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }

    // Per-pair accumulators.
    using Vec = std::vector<double>;
    std::vector<std::map<double, Vec>>    accumulated_spectral(P);
    std::vector<std::map<double, Vec>>    accumulated_spectral_imag(P);
    std::vector<std::map<double, double>> accumulated_Z(P);
    std::vector<std::map<double, std::vector<Vec>>> per_sample_spectral(P);
    std::vector<std::map<double, std::vector<Vec>>> per_sample_spectral_imag(P);
    for (size_t p = 0; p < P; ++p) {
        for (double T : temperatures) {
            accumulated_spectral[p][T]      = Vec(num_omega_bins, 0.0);
            accumulated_spectral_imag[p][T] = Vec(num_omega_bins, 0.0);
            accumulated_Z[p][T]             = 0.0;
            per_sample_spectral[p][T]       = {};
            per_sample_spectral_imag[p][T]  = {};
        }
    }

    int mpi_rank = 0, mpi_size = 1;
#ifdef WITH_MPI
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
#endif
    uint64_t samples_per_rank = params.num_samples / mpi_size;
    uint64_t remainder        = params.num_samples % mpi_size;
    uint64_t start_sample = mpi_rank * samples_per_rank +
                            std::min((uint64_t)mpi_rank, remainder);
    uint64_t end_sample   = start_sample + samples_per_rank +
                            (mpi_rank < (int)remainder ? 1 : 0);
    uint64_t local_num_samples = end_sample - start_sample;

#ifdef WITH_MPI
    MPI_Barrier(comm);
#endif

    uint64_t max_ritz_states = std::min(params.krylov_dim, (uint64_t)50);
    const double eta = params.broadening;
    const double eta_sq = eta * eta;
    const double inv_pi_eta = eta / M_PI;

    for (uint64_t sample_idx = start_sample; sample_idx < end_sample; sample_idx++) {
        if (mpi_rank == 0 && verbose) {
            std::cout << "\n--- Sample " << (sample_idx - start_sample + 1)
                      << "/" << local_num_samples
                      << " (Global " << (sample_idx + 1) << "/" << params.num_samples
                      << ") ---" << std::endl;
        }

        std::mt19937 sample_gen(params.random_seed + sample_idx * 12345);
        ComplexVector r_state =
            generateGaussianRandomVector(static_cast<int>(N), sample_gen);

        // -------- shared per-sample work: outer Lanczos on H from |r> --------
        std::vector<double> alpha_H, beta_H;
        std::vector<ComplexVector> lanczos_vectors;
        build_lanczos_tridiagonal_with_basis(
            H, r_state, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha_H, beta_H, &lanczos_vectors);
        const uint64_t m_H = alpha_H.size();
        if (m_H == 0) continue;

        std::vector<double> ritz_values, dummy_w, evecs;
        diagonalize_tridiagonal_ritz(alpha_H, beta_H, ritz_values, dummy_w, &evecs);
        if (ritz_values.empty()) continue;

        std::vector<double> c_sq(m_H);
        for (uint64_t i = 0; i < m_H; i++) {
            c_sq[i] = evecs[i * m_H + 0] * evecs[i * m_H + 0];
        }
        const double E_min =
            *std::min_element(ritz_values.begin(), ritz_values.end());

        // Significance threshold (use highest T to be most permissive).
        const double T_max_local =
            *std::max_element(temperatures.begin(), temperatures.end());
        const double beta_min = 1.0 / T_max_local;
        std::vector<double> max_weights(m_H);
        double Z_max = 0.0;
        for (uint64_t i = 0; i < m_H; i++) {
            max_weights[i] = c_sq[i] *
                std::exp(-beta_min * (ritz_values[i] - E_min));
            Z_max += max_weights[i];
        }
        const double weight_threshold = 1e-10 * Z_max;
        std::vector<uint64_t> significant;
        significant.reserve(max_ritz_states);
        for (uint64_t i = 0; i < std::min(m_H, max_ritz_states); i++) {
            if (max_weights[i] >= weight_threshold || c_sq[i] > 1e-12) {
                significant.push_back(i);
            }
        }

        // -------- shared per-sample work: reconstruct Ritz eigenstates -------
        // psi_cache[s] corresponds to significant[s], normalized.
        std::vector<ComplexVector> psi_cache;
        std::vector<bool> psi_valid;
        psi_cache.reserve(significant.size());
        psi_valid.reserve(significant.size());
        for (uint64_t i_sig : significant) {
            ComplexVector psi(N, Complex(0.0, 0.0));
            for (uint64_t j = 0; j < m_H; j++) {
                Complex coeff(evecs[i_sig * m_H + j], 0.0);
                cblas_zaxpy(N, &coeff, lanczos_vectors[j].data(), 1,
                            psi.data(), 1);
            }
            const double pn = cblas_dznrm2(N, psi.data(), 1);
            if (pn < 1e-14) {
                psi_cache.emplace_back();   // empty, marked invalid
                psi_valid.push_back(false);
                continue;
            }
            Complex sc(1.0/pn, 0.0);
            cblas_zscal(N, &sc, psi.data(), 1);
            psi_cache.push_back(std::move(psi));
            psi_valid.push_back(true);
        }

        // Lanczos basis no longer needed -- free before the inner heavy loops.
        lanczos_vectors.clear();
        lanczos_vectors.shrink_to_fit();

        // -------- per-pair work: inner Lanczos + Lehmann weights -------------
        for (size_t p = 0; p < P; ++p) {
            const auto& O1 = O1_list[p];
            const auto& O2 = O2_list[p];

            std::vector<Vec> precomputed_S_i(significant.size());
            std::vector<Vec> precomputed_S_i_imag(significant.size());
            std::vector<double> precomputed_E(significant.size(), 0.0);
            std::vector<double> precomputed_csq(significant.size(), 0.0);
            std::vector<bool>   state_valid(significant.size(), false);

            for (size_t idx = 0; idx < significant.size(); ++idx) {
                if (!psi_valid[idx]) continue;
                const ComplexVector& psi_local = psi_cache[idx];

                ComplexVector phi2(N);
                O2(psi_local.data(), phi2.data(), N);
                double phi2_norm = cblas_dznrm2(N, phi2.data(), 1);
                if (phi2_norm < 1e-14) continue;
                Complex sc2(1.0/phi2_norm, 0.0);
                cblas_zscal(N, &sc2, phi2.data(), 1);

                std::vector<double> alpha_S, beta_S;
                std::vector<ComplexVector> basis_S;
                build_lanczos_tridiagonal_with_basis(
                    H, phi2, N, params.krylov_dim, params.tolerance,
                    params.full_reorthogonalization, params.reorth_frequency,
                    alpha_S, beta_S, &basis_S);
                if (alpha_S.empty()) continue;
                const uint64_t m_S = alpha_S.size();
                for (uint64_t k = 0; k < m_S; k++) alpha_S[k] -= E_gs;

                ComplexVector phi1(N);
                O1(psi_local.data(), phi1.data(), N);
                std::vector<Complex> phi1_overlaps(m_S);
                for (uint64_t j = 0; j < m_S; j++) {
                    Complex ov;
                    cblas_zdotc_sub(N, phi1.data(), 1,
                                    basis_S[j].data(), 1, &ov);
                    phi1_overlaps[j] = ov;
                }
                basis_S.clear();
                basis_S.shrink_to_fit();

                std::vector<double> ritz_S, dum_S, evecs_S;
                diagonalize_tridiagonal_ritz(alpha_S, beta_S,
                                             ritz_S, dum_S, &evecs_S);
                if (ritz_S.empty()) continue;

                const uint64_t n_ritz = ritz_S.size();
                Vec S_i(num_omega_bins, 0.0);
                Vec S_i_im(num_omega_bins, 0.0);
                std::vector<double> w_re(n_ritz), w_im(n_ritz), E_arr(n_ritz);
                for (uint64_t k = 0; k < n_ritz; k++) {
                    Complex ov_O1(0.0, 0.0);
                    for (uint64_t j = 0; j < m_S; j++) {
                        ov_O1 += Complex(evecs_S[k * m_S + j], 0.0)
                                 * phi1_overlaps[j];
                    }
                    const double V_0k = evecs_S[k * m_S + 0];
                    const Complex w_k = ov_O1 *
                        Complex(V_0k * phi2_norm, 0.0);
                    w_re[k]  = w_k.real();
                    w_im[k]  = w_k.imag();
                    E_arr[k] = ritz_S[k];
                }
                #pragma omp parallel for schedule(static)
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    const double omega = frequencies[iw];
                    double s_re = 0.0, s_im = 0.0;
                    for (uint64_t k = 0; k < n_ritz; k++) {
                        const double delta = omega - E_arr[k];
                        const double L = inv_pi_eta /
                                         (delta * delta + eta_sq);
                        s_re += w_re[k] * L;
                        s_im += w_im[k] * L;
                    }
                    S_i[iw]    = s_re;
                    S_i_im[iw] = s_im;
                }

                precomputed_S_i[idx]      = std::move(S_i);
                precomputed_S_i_imag[idx] = std::move(S_i_im);
                precomputed_E[idx]        = ritz_values[significant[idx]];
                precomputed_csq[idx]      = c_sq[significant[idx]];
                state_valid[idx]          = true;
            }

            // Apply thermal weights for this pair p.
            for (double T : temperatures) {
                const double beta = 1.0 / T;
                double Z_sample = 0.0;
                for (size_t idx = 0; idx < significant.size(); ++idx) {
                    if (!state_valid[idx]) continue;
                    Z_sample += precomputed_csq[idx] *
                        std::exp(-beta * (precomputed_E[idx] - E_min));
                }
                accumulated_Z[p][T] += Z_sample;

                Vec sample_S(num_omega_bins, 0.0);
                Vec sample_S_im(num_omega_bins, 0.0);
                for (size_t idx = 0; idx < significant.size(); ++idx) {
                    if (!state_valid[idx]) continue;
                    const double th =
                        precomputed_csq[idx] *
                        std::exp(-beta * (precomputed_E[idx] - E_min));
                    if (th < 1e-14 * Z_sample) continue;
                    const auto& Si  = precomputed_S_i[idx];
                    const auto& Sii = precomputed_S_i_imag[idx];
                    for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                        const double a  = th * Si[iw];
                        const double ai = th * Sii[iw];
                        sample_S[iw]    += a;
                        sample_S_im[iw] += ai;
                        accumulated_spectral[p][T][iw]      += a;
                        accumulated_spectral_imag[p][T][iw] += ai;
                    }
                }
                if (Z_sample > 1e-300) {
                    Vec ns(num_omega_bins), ni(num_omega_bins);
                    for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                        ns[iw] = sample_S[iw]    / Z_sample;
                        ni[iw] = sample_S_im[iw] / Z_sample;
                    }
                    per_sample_spectral[p][T].push_back(std::move(ns));
                    per_sample_spectral_imag[p][T].push_back(std::move(ni));
                }
            }
        } // end pairs
    } // end samples

#ifdef WITH_MPI
    MPI_Barrier(comm);
    for (size_t p = 0; p < P; ++p) {
        for (double T : temperatures) {
            Vec gS(num_omega_bins, 0.0), gSi(num_omega_bins, 0.0);
            double gZ = 0.0;
            MPI_Reduce(accumulated_spectral[p][T].data(), gS.data(),
                       num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, comm);
            MPI_Reduce(accumulated_spectral_imag[p][T].data(), gSi.data(),
                       num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, comm);
            MPI_Reduce(&accumulated_Z[p][T], &gZ, 1,
                       MPI_DOUBLE, MPI_SUM, 0, comm);
            if (mpi_rank == 0) {
                accumulated_spectral[p][T]      = std::move(gS);
                accumulated_spectral_imag[p][T] = std::move(gSi);
                accumulated_Z[p][T]             = gZ;
            }
        }
    }
    uint64_t global_total_samples = 0;
    MPI_Reduce(&local_num_samples, &global_total_samples, 1,
               MPI_UINT64_T, MPI_SUM, 0, comm);
#else
    uint64_t global_total_samples = local_num_samples;
#endif

    std::vector<std::map<double, DynamicalResponseResults>> out(P);
    for (size_t p = 0; p < P; ++p) {
        for (double T : temperatures) {
            DynamicalResponseResults r;
            r.frequencies = frequencies;
            r.omega_min = omega_min;
            r.omega_max = omega_max;
            r.total_samples = (mpi_rank == 0) ? global_total_samples
                                              : local_num_samples;
            r.spectral_function.assign(num_omega_bins, 0.0);
            r.spectral_function_imag.assign(num_omega_bins, 0.0);
            r.spectral_error.assign(num_omega_bins, 0.0);
            r.spectral_error_imag.assign(num_omega_bins, 0.0);
            const double Zt = accumulated_Z[p][T];
            if (mpi_rank == 0 && Zt > 1e-300) {
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    r.spectral_function[iw] =
                        accumulated_spectral[p][T][iw] / Zt;
                    r.spectral_function_imag[iw] =
                        accumulated_spectral_imag[p][T][iw] / Zt;
                }
                const uint64_t ns = per_sample_spectral[p][T].size();
                if (ns > 1 && mpi_size == 1) {
                    Vec mean(num_omega_bins, 0.0), mean_i(num_omega_bins, 0.0);
                    for (uint64_t s = 0; s < ns; s++) {
                        for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                            mean[iw]   += per_sample_spectral[p][T][s][iw];
                            mean_i[iw] += per_sample_spectral_imag[p][T][s][iw];
                        }
                    }
                    for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                        mean[iw]   /= ns;
                        mean_i[iw] /= ns;
                    }
                    for (uint64_t s = 0; s < ns; s++) {
                        for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                            const double d  = per_sample_spectral[p][T][s][iw]
                                              - mean[iw];
                            const double di = per_sample_spectral_imag[p][T][s][iw]
                                              - mean_i[iw];
                            r.spectral_error[iw]      += d * d;
                            r.spectral_error_imag[iw] += di * di;
                        }
                    }
                    const double nrm =
                        std::sqrt(static_cast<double>(ns * (ns - 1)));
                    for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                        r.spectral_error[iw]      = std::sqrt(r.spectral_error[iw]) / nrm;
                        r.spectral_error_imag[iw] = std::sqrt(r.spectral_error_imag[iw]) / nrm;
                    }
                }
            }
            out[p][T] = std::move(r);
        }
    }
    return out;
}

#ifdef WITH_MPI
// P == 1 specialisation: delegate to the multi-operator core above so the
// FTLM Lehmann machinery has exactly one implementation.
static std::map<double, DynamicalResponseResults>
compute_dynamical_correlation_multi_sample_multi_temperature_impl(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir
#ifdef WITH_MPI
    , MPI_Comm comm
#endif
) {
    auto per_pair = compute_dynamical_correlation_multi_operator_multi_temperature_impl(
        H, {std::move(O1)}, {std::move(O2)}, N, params,
        omega_min, omega_max, num_omega_bins,
        temperatures, energy_shift, output_dir
#ifdef WITH_MPI
        , comm
#endif
    );
    if (per_pair.empty()) return {};
    return std::move(per_pair.front());
}
#endif  // WITH_MPI

// ============================================================================
// Public dispatch wrapper for the multi-operator FTLM dynamical kernel.
// The plain single-pair (multi_sample) wrapper was retired in the Family 3
// follow-up sweep -- single-pair callers use the `_comm` variant (MPI) or
// `ed::observables::ftlm_dynamical_kernel_via_backend` (in-memory API).
// Audit #4: the heavy bodies above are MPI-Comm-aware (`comm` parameter);
// these thin wrappers preserve the legacy MPI_COMM_WORLD-only API and
// expose a parallel `_comm` API in `ed::dssf` for distributed-DSSF
// orchestration via MPI_Comm_split.
// ============================================================================
std::vector<std::map<double, DynamicalResponseResults>>
compute_dynamical_correlation_multi_operator_multi_temperature(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O1_list,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O2_list,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir
) {
    return compute_dynamical_correlation_multi_operator_multi_temperature_impl(
        H, O1_list, O2_list, N, params, omega_min, omega_max, num_omega_bins,
        temperatures, energy_shift, output_dir
#ifdef WITH_MPI
        , MPI_COMM_WORLD
#endif
    );
}

#ifdef WITH_MPI
namespace ed {
namespace dssf {

std::map<double, DynamicalResponseResults>
compute_dynamical_correlation_multi_sample_multi_temperature_comm(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir,
    MPI_Comm comm
) {
    return ::compute_dynamical_correlation_multi_sample_multi_temperature_impl(
        H, O1, O2, N, params, omega_min, omega_max, num_omega_bins,
        temperatures, energy_shift, output_dir, comm);
}

std::vector<std::map<double, DynamicalResponseResults>>
compute_dynamical_correlation_multi_operator_multi_temperature_comm(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O1_list,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O2_list,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir,
    MPI_Comm comm
) {
    return ::compute_dynamical_correlation_multi_operator_multi_temperature_impl(
        H, O1_list, O2_list, N, params, omega_min, omega_max, num_omega_bins,
        temperatures, energy_shift, output_dir, comm);
}

}  // namespace dssf
}  // namespace ed
#endif  // WITH_MPI

// ============================================================================
// GROUND STATE DYNAMICAL STRUCTURE FACTOR (CONTINUED FRACTION METHOD)
// ============================================================================

/**
 * @brief Evaluate spectral function using continued fraction representation
 * 
 * Computes S(ω) = -Im[G(ω + iη)] / π where G is the continued fraction:
 * G(z) = norm_sq / (z - α₀ - β₁²/(z - α₁ - β₂²/(z - α₂ - ...)))
 * 
 * Uses numerically stable bottom-up evaluation to avoid overflow.
 */
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

/**
 * @brief Compute ground state dynamical structure factor S(ω)
 */
DynamicalResponseResults compute_ground_state_dssf(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t N,
    const GroundStateDSSFParameters& params
) {
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Ground State Dynamical Structure Factor\n";
        std::cout << "(Continued Fraction / Lanczos Method)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Ground state energy: " << std::setprecision(10) << ground_state_energy << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Broadening η: " << params.broadening << std::endl;
        std::cout << "Frequency range: [" << params.omega_min << ", " << params.omega_max << "]" << std::endl;
        std::cout << "Frequency points: " << params.num_omega_points << std::endl;
        std::cout << "Method: " << (params.use_continued_fraction ? "Continued Fraction" : "Eigendecomposition") << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    
    DynamicalResponseResults results;
    results.total_samples = 1;  // Exact calculation, no random sampling
    results.omega_min = params.omega_min;
    results.omega_max = params.omega_max;
    
    // Generate frequency grid
    results.frequencies.resize(params.num_omega_points);
    double omega_step = (params.omega_max - params.omega_min) / 
                        std::max(uint64_t(1), params.num_omega_points - 1);
    for (size_t i = 0; i < params.num_omega_points; i++) {
        results.frequencies[i] = params.omega_min + i * omega_step;
    }
    
    // Apply operator to ground state: |φ⟩ = O|0⟩
    if (verbose) std::cout << "\nApplying operator to ground state..." << std::endl;
    ComplexVector phi(N);
    O(ground_state.data(), phi.data(), N);

    // Compute norm ||O|0⟩||²
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    double phi_norm_sq = phi_norm * phi_norm;

    if (verbose) {
        std::cout << "  ||O|0⟩|| = " << phi_norm << std::endl;
        std::cout << "  ||O|0⟩||² = " << phi_norm_sq << std::endl;
    }

    if (phi_norm < 1e-14) {
        std::cerr << "Warning: O|0⟩ has zero norm. Operator has no matrix elements from ground state.\n";
        results.spectral_function.resize(params.num_omega_points, 0.0);
        results.spectral_function_imag.resize(params.num_omega_points, 0.0);
        results.spectral_error.resize(params.num_omega_points, 0.0);
        results.spectral_error_imag.resize(params.num_omega_points, 0.0);
        return results;
    }

    // Normalize |φ⟩ for Lanczos
    Complex scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &scale, phi.data(), 1);

    // Build Lanczos tridiagonal starting from |φ⟩
    if (verbose) std::cout << "\nBuilding Lanczos tridiagonal matrix..." << std::endl;
    auto lanczos_start = std::chrono::high_resolution_clock::now();

    std::vector<double> alpha, beta;
    uint64_t iterations = build_lanczos_tridiagonal(
        H, phi, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta
    );

    auto lanczos_end = std::chrono::high_resolution_clock::now();
    double lanczos_time = std::chrono::duration<double>(lanczos_end - lanczos_start).count();

    if (verbose) {
        std::cout << "  Lanczos iterations: " << iterations << std::endl;
        std::cout << "  Lanczos time: " << lanczos_time << " seconds" << std::endl;
    }

    if (alpha.empty()) {
        std::cerr << "Error: Lanczos failed to build tridiagonal matrix\n";
        results.spectral_function.resize(params.num_omega_points, 0.0);
        results.spectral_function_imag.resize(params.num_omega_points, 0.0);
        results.spectral_error.resize(params.num_omega_points, 0.0);
        results.spectral_error_imag.resize(params.num_omega_points, 0.0);
        return results;
    }

    // Shift eigenvalues: ω - E₀ + E_n → we need to shift α values
    // The resolvent is (ω + E₀ - H + iη)⁻¹, so effectively we shift by E₀
    if (verbose) std::cout << "\nShifting energies by ground state energy E₀ = " << ground_state_energy << std::endl;
    for (size_t i = 0; i < alpha.size(); i++) {
        alpha[i] -= ground_state_energy;
    }

    // Compute spectral function
    if (verbose) std::cout << "\nComputing spectral function..." << std::endl;
    auto spectral_start = std::chrono::high_resolution_clock::now();
    
    if (params.use_continued_fraction) {
        // Use continued fraction method (faster, O(M) per ω)
        results.spectral_function = continued_fraction_spectral_function(
            alpha, beta, results.frequencies, params.broadening, phi_norm_sq
        );
    } else {
        // Use eigendecomposition method (for comparison/validation)
        std::vector<double> ritz_values, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);
        
        // Scale weights by norm²
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] *= phi_norm_sq;
        }
        
        // Compute spectral function using Lorentzian broadening
        results.spectral_function.resize(params.num_omega_points, 0.0);
        double eta = params.broadening;
        
        #pragma omp parallel for schedule(static)
        for (size_t iw = 0; iw < params.num_omega_points; iw++) {
            double omega = results.frequencies[iw];
            double sum = 0.0;
            
            for (size_t n = 0; n < ritz_values.size(); n++) {
                double delta = omega - ritz_values[n];
                double lorentzian = (eta / M_PI) / (delta * delta + eta * eta);
                sum += weights[n] * lorentzian;
            }
            
            results.spectral_function[iw] = sum;
        }
    }
    
    auto spectral_end = std::chrono::high_resolution_clock::now();
    double spectral_time = std::chrono::duration<double>(spectral_end - spectral_start).count();

    if (verbose) {
        std::cout << "  Spectral function time: " << spectral_time << " seconds" << std::endl;
    }

    // For ground state, imaginary part is zero (self-correlation)
    results.spectral_function_imag.resize(params.num_omega_points, 0.0);

    // No error bars for exact ground state calculation
    results.spectral_error.resize(params.num_omega_points, 0.0);
    results.spectral_error_imag.resize(params.num_omega_points, 0.0);

    // Compute sum rule: ∫ S(ω) dω should equal ||O|0⟩||². Trapezoid rule.
    double integral = 0.0;
    for (size_t i = 1; i < params.num_omega_points; i++) {
        double dw = results.frequencies[i] - results.frequencies[i-1];
        integral += 0.5 * (results.spectral_function[i] + results.spectral_function[i-1]) * dw;
    }

    if (verbose) {
        std::cout << "\n--- Sum Rule Check ---" << std::endl;
        std::cout << "  ∫ S(ω) dω = " << integral << std::endl;
        std::cout << "  ||O|0⟩||² = " << phi_norm_sq << std::endl;
        std::cout << "  Ratio: " << integral / phi_norm_sq << " (should be ≈ 1.0)" << std::endl;
    }

    // Sum-rule diagnostic: warn if spectral weight is significantly missing.
    // Do NOT renormalize — uniform rescaling enforces the integral but
    // distorts relative peak heights (finite Krylov clips low-weight peaks
    // differently than high-weight ones).  Increase krylov_dim instead.
    if (integral > 1e-14) {
        const double ratio = integral / phi_norm_sq;
        if (ratio < 0.95 || ratio > 1.05) {
            std::cerr << "[GS-DSSF] Warning: sum-rule ratio = " << ratio
                      << " (|1 - ratio| > 5%). Consider increasing krylov_dim ("
                      << params.krylov_dim << ") or widening [omega_min, omega_max].\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Ground State DSSF Complete\n";
        std::cout << "Total time: " << total_time << " seconds\n";
        std::cout << "==========================================\n";
    }

    return results;
}

/**
 * @brief Compute ground state two-operator cross-correlation
 */
DynamicalResponseResults compute_ground_state_cross_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t N,
    const GroundStateDSSFParameters& params
) {
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Ground State Cross-Correlation S_{O1,O2}(ω)\n";
        std::cout << "(Lanczos Method)\n";
        std::cout << "==========================================\n";
    }

    DynamicalResponseResults results;
    results.total_samples = 1;
    results.omega_min = params.omega_min;
    results.omega_max = params.omega_max;

    // Generate frequency grid
    results.frequencies.resize(params.num_omega_points);
    double omega_step = (params.omega_max - params.omega_min) /
                        std::max(uint64_t(1), params.num_omega_points - 1);
    for (size_t i = 0; i < params.num_omega_points; i++) {
        results.frequencies[i] = params.omega_min + i * omega_step;
    }

    // Apply O2 to ground state: |φ₂⟩ = O₂|0⟩
    ComplexVector phi2(N);
    O2(ground_state.data(), phi2.data(), N);

    double phi2_norm = cblas_dznrm2(N, phi2.data(), 1);

    if (phi2_norm < 1e-14) {
        std::cerr << "Warning: O₂|0⟩ has zero norm.\n";
        results.spectral_function.resize(params.num_omega_points, 0.0);
        results.spectral_function_imag.resize(params.num_omega_points, 0.0);
        results.spectral_error.resize(params.num_omega_points, 0.0);
        results.spectral_error_imag.resize(params.num_omega_points, 0.0);
        return results;
    }

    // Normalize for Lanczos (in-place on phi2; we no longer need the
    // un-normalized vector after this point).
    {
        Complex scale(1.0 / phi2_norm, 0.0);
        cblas_zscal(N, &scale, phi2.data(), 1);
    }

    // Build Lanczos tridiagonal starting from |φ₂⟩
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> basis_vectors;

    // We need basis vectors for cross-correlation
    int iterations = build_lanczos_tridiagonal_with_basis(
        H, phi2, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta, &basis_vectors
    );

    if (verbose) {
        std::cout << "Lanczos iterations: " << iterations << std::endl;
    }

    // Diagonalize tridiagonal to get eigenvectors in Krylov basis
    std::vector<double> ritz_values, weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);

    const size_t M = ritz_values.size();
    const size_t M_basis = std::min(M, basis_vectors.size());

    // Apply O1 to ground state: |φ₁⟩ = O₁|0⟩
    //
    // Note on conjugation: in the Lehmann representation we need
    //   ⟨0|O₁†|n⟩ = (O₁|0⟩)† |n⟩ = ⟨φ₁|n⟩,
    // computed below via cblas_zdotc_sub(phi1, ritz). zdotc returns
    // φ₁†·n, which is exactly ⟨0|O₁†|n⟩ for any (Hermitian or not) O₁.
    // No extra adjoint application is required.
    ComplexVector phi1(N);
    O1(ground_state.data(), phi1.data(), N);

    // Compute  p[j] = ⟨φ₁|v_j⟩  once over the Krylov basis. Then
    // ⟨φ₁|n⟩ = Σ_j V[j,n] · p[j] is a tiny Σ over the Krylov dimension
    // and we never need to materialise the full-space Ritz state |n⟩
    // (saves M extra zaxpy(N) reconstructions vs the previous version).
    // The reduction over j is done with a single BLAS-2 dgemv on the real
    // and imaginary parts of p (V is real, M×M) -- O(M²) Flops instead of
    // an explicit nested C++ loop.
    std::vector<Complex> p(M_basis);
    for (size_t j = 0; j < M_basis; ++j) {
        cblas_zdotc_sub(N, phi1.data(), 1,
                        basis_vectors[j].data(), 1, &p[j]);
    }

    // Pad p to M (in case M_basis < M from early Lanczos termination).
    std::vector<double> p_re(M, 0.0), p_im(M, 0.0);
    for (size_t j = 0; j < M_basis; ++j) {
        p_re[j] = p[j].real();
        p_im[j] = p[j].imag();
    }
    std::vector<double> overlap_re(M), overlap_im(M);
    cblas_dgemv(CblasColMajor, CblasTrans,
                static_cast<int>(M), static_cast<int>(M),
                1.0, evecs.data(), static_cast<int>(M),
                p_re.data(), 1, 0.0, overlap_re.data(), 1);
    cblas_dgemv(CblasColMajor, CblasTrans,
                static_cast<int>(M), static_cast<int>(M),
                1.0, evecs.data(), static_cast<int>(M),
                p_im.data(), 1, 0.0, overlap_im.data(), 1);

    std::vector<Complex> spectral_weights(M);
    for (size_t n = 0; n < M; ++n) {
        const Complex overlap_O1(overlap_re[n], overlap_im[n]);
        // ⟨n|φ₂⟩ = phi2_norm · V[0,n]
        const Complex me_O2(phi2_norm * evecs[n * M + 0], 0.0);
        spectral_weights[n] = overlap_O1 * me_O2;
    }

    // Free the Krylov basis -- the spectral kernel below only needs the
    // tridiagonal eigenvalues / weights and the precomputed p[j].
    basis_vectors.clear();
    basis_vectors.shrink_to_fit();
    
    // Shift eigenvalues by ground state energy
    for (size_t i = 0; i < ritz_values.size(); i++) {
        ritz_values[i] -= ground_state_energy;
    }
    
    // Compute spectral function
    results.spectral_function.resize(params.num_omega_points, 0.0);
    results.spectral_function_imag.resize(params.num_omega_points, 0.0);
    
    double eta = params.broadening;
    
    #pragma omp parallel for schedule(static)
    for (size_t iw = 0; iw < params.num_omega_points; iw++) {
        double omega = results.frequencies[iw];
        Complex sum(0.0, 0.0);
        
        for (size_t n = 0; n < M; n++) {
            double delta = omega - ritz_values[n];
            // Lorentzian: (η/π) / ((ω - E)² + η²)
            double lorentzian = (eta / M_PI) / (delta * delta + eta * eta);
            sum += spectral_weights[n] * lorentzian;
        }
        
        results.spectral_function[iw] = sum.real();
        results.spectral_function_imag[iw] = sum.imag();
    }
    
    results.spectral_error.resize(params.num_omega_points, 0.0);
    results.spectral_error_imag.resize(params.num_omega_points, 0.0);
    
    return results;
}

/**
 * @brief Compute ground-state DSSF when O1, O2 lift |0> into a different
 *        magnetisation sector (audit item #1 -- full).
 *
 * The standard `compute_ground_state_dssf` / cross_correlation kernels
 * assume H, |0>, O1|0>, O2|0> all live in a single Hilbert space of
 * dimension N. For fixed-Sz workflows with raising/lowering observables
 * (S+, S-, Sx, Sy, ...) the resolvent (omega - H)^{-1} is block-diagonal
 * in n_up and the relevant matrix elements live in a destination sector
 * of dimension `dim_inner`, which is generally != `dim_outer` = dim of
 * the source sector hosting |0>.
 *
 * This routine therefore:
 *   1) applies O2 to |0> (size dim_outer) to obtain |phi> in the
 *      destination sector (size dim_inner);
 *   2) Lanczos-tridiagonalises H_inner restricted to dim_inner with
 *      |phi>/||phi|| as the start vector;
 *   3) applies O1 to |0> to obtain |chi> in dim_inner; computes the
 *      Krylov projection coefficients p[j] = <chi|v_j>;
 *   4) reconstructs the Lehmann sum via the Ritz eigendecomposition.
 *
 * Bug-for-bug compatible with `compute_ground_state_cross_correlation`
 * when dim_outer == dim_inner and O1, O2 preserve the sector.
 */
DynamicalResponseResults compute_ground_state_dssf_cross_sector(
    std::function<void(const Complex*, Complex*, int)> H_inner,
    std::function<void(const Complex*, Complex*, int)> O1_dagger_apply,
    std::function<void(const Complex*, Complex*, int)> O2_apply,
    const ComplexVector& ground_state,
    double ground_state_energy,
    uint64_t dim_outer,
    uint64_t dim_inner,
    const GroundStateDSSFParameters& params
) {
    const bool verbose = ed_dssf_verbose();
    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Cross-sector ground-state DSSF (audit #1 full)\n";
        std::cout << "  dim_outer = " << dim_outer
                  << ", dim_inner = " << dim_inner << "\n";
        std::cout << "==========================================\n";
    }

    DynamicalResponseResults results;
    results.total_samples = 1;
    results.omega_min = params.omega_min;
    results.omega_max = params.omega_max;
    results.frequencies.resize(params.num_omega_points);
    const double omega_step = (params.omega_max - params.omega_min) /
                              std::max<uint64_t>(1, params.num_omega_points - 1);
    for (size_t i = 0; i < params.num_omega_points; ++i) {
        results.frequencies[i] = params.omega_min + i * omega_step;
    }
    results.spectral_function.assign(params.num_omega_points, 0.0);
    results.spectral_function_imag.assign(params.num_omega_points, 0.0);
    results.spectral_error.assign(params.num_omega_points, 0.0);
    results.spectral_error_imag.assign(params.num_omega_points, 0.0);

    if (ground_state.size() != dim_outer) {
        std::cerr << "compute_ground_state_dssf_cross_sector: ground_state size "
                  << ground_state.size() << " != dim_outer " << dim_outer
                  << "; aborting kernel.\n";
        return results;
    }
    if (dim_inner == 0) {
        if (verbose) {
            std::cout << "  destination sector is empty; spectrum = 0.\n";
        }
        return results;
    }

    // |phi> = O2 |0> in destination sector.
    ComplexVector phi(dim_inner);
    O2_apply(ground_state.data(), phi.data(),
             static_cast<int>(dim_inner));
    const double phi_norm = cblas_dznrm2(static_cast<int>(dim_inner),
                                         phi.data(), 1);
    if (phi_norm < 1e-14) {
        if (verbose) std::cout << "  ||O2|0>|| ~ 0; spectrum = 0.\n";
        return results;
    }
    {
        const Complex scale(1.0 / phi_norm, 0.0);
        cblas_zscal(static_cast<int>(dim_inner), &scale, phi.data(), 1);
    }

    // Lanczos tridiagonal in destination sector with full basis storage.
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> basis_vectors;
    const int iters = build_lanczos_tridiagonal_with_basis(
        H_inner, phi, dim_inner, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta, &basis_vectors);
    if (verbose) {
        std::cout << "  Lanczos iterations: " << iters << "\n";
    }

    std::vector<double> ritz_values, weights, evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights, &evecs);
    const size_t M = ritz_values.size();
    const size_t M_basis = std::min(M, basis_vectors.size());
    if (M == 0) {
        if (verbose) std::cout << "  empty Ritz set; spectrum = 0.\n";
        return results;
    }

    // |chi> = O1 |0> in destination sector (note: caller passes the
    // same physical operator O1 -- the apply lambda is what makes it
    // act 'as O1', not the conjugate).
    ComplexVector chi(dim_inner);
    O1_dagger_apply(ground_state.data(), chi.data(),
                    static_cast<int>(dim_inner));

    // p[j] = <chi | v_j> via zdotc (returns chi^dagger v_j).
    std::vector<Complex> p(M_basis);
    for (size_t j = 0; j < M_basis; ++j) {
        cblas_zdotc_sub(static_cast<int>(dim_inner), chi.data(), 1,
                        basis_vectors[j].data(), 1, &p[j]);
    }
    std::vector<double> p_re(M, 0.0), p_im(M, 0.0);
    for (size_t j = 0; j < M_basis; ++j) {
        p_re[j] = p[j].real();
        p_im[j] = p[j].imag();
    }
    std::vector<double> overlap_re(M), overlap_im(M);
    cblas_dgemv(CblasColMajor, CblasTrans,
                static_cast<int>(M), static_cast<int>(M),
                1.0, evecs.data(), static_cast<int>(M),
                p_re.data(), 1, 0.0, overlap_re.data(), 1);
    cblas_dgemv(CblasColMajor, CblasTrans,
                static_cast<int>(M), static_cast<int>(M),
                1.0, evecs.data(), static_cast<int>(M),
                p_im.data(), 1, 0.0, overlap_im.data(), 1);

    std::vector<Complex> spectral_weights(M);
    for (size_t n = 0; n < M; ++n) {
        const Complex overlap_O1(overlap_re[n], overlap_im[n]);
        const Complex me_O2(phi_norm * evecs[n * M + 0], 0.0);
        spectral_weights[n] = overlap_O1 * me_O2;
    }

    basis_vectors.clear();
    basis_vectors.shrink_to_fit();

    for (size_t i = 0; i < M; ++i) {
        ritz_values[i] -= ground_state_energy;
    }

    const double eta = params.broadening;
    #pragma omp parallel for schedule(static)
    for (size_t iw = 0; iw < params.num_omega_points; ++iw) {
        const double omega = results.frequencies[iw];
        Complex sum(0.0, 0.0);
        for (size_t n = 0; n < M; ++n) {
            const double delta = omega - ritz_values[n];
            const double L = (eta / M_PI) / (delta * delta + eta * eta);
            sum += spectral_weights[n] * L;
        }
        results.spectral_function[iw] = sum.real();
        results.spectral_function_imag[iw] = sum.imag();
    }

    return results;
}

// load_ground_state_from_file was retired in the minimalist-architecture
// rev (May 2026): no external callers. Workflows that need a previously
// computed ground state read it back via HDF5IO::loadEigenvector(/0/) on
// the unified ed_results.h5 store; the legacy
// eigenvector_block0_0.dat / eigenvalues.txt sidecar paths are no
// longer written or supported.
