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

/**
 * @brief Build Krylov subspace and extract tridiagonal matrix coefficients.
 *
 * Phase 5.2 of the Krylov-unification gap-fill (May 2026 day 12+):
 * when full reorthogonalisation is requested, this function now bypasses
 * the legacy ``build_lanczos_tridiagonal_with_basis`` translation shim
 * entirely and calls ``ed::krylov::lanczos_tridiag`` (which delegates
 * to ``lanczos_kernel<CpuBackend>``) directly. The kernel's
 * ``UniqueVec`` basis is held only inside this function -- no
 * ``vector<ComplexVector>`` copy is materialised, since the
 * eigenvalues-only entry point (which this is) does not consume the
 * basis downstream. For the no-reorth or no-basis-storage path we still
 * route through the legacy entry to preserve the periodic-reorth /
 * non-reorth code paths that have not been ported to the kernel yet.
 *
 * History: previously a near-clone of
 * ``build_lanczos_tridiagonal_with_basis``, then a thin forwarder; now
 * a true facade over the unified kernel for the dominant call shape.
 */
int build_lanczos_tridiagonal(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v0,
    uint64_t N,
    uint64_t max_iter,
    double tol,
    bool full_reorth,
    uint64_t reorth_freq,
    std::vector<double>& alpha,
    std::vector<double>& beta
) {
    if (full_reorth) {
        (void)tol;
        ed::krylov::LanczosKernelOptions opts;
        opts.max_iter   = static_cast<std::size_t>(std::min<uint64_t>(N, max_iter));
        opts.reorth     = ed::krylov::ReorthPolicy::FullCGS2;
        opts.keep_basis = true;
        auto matvec = [&H](const Complex* in, Complex* out, std::size_t n) {
            H(in, out, static_cast<int>(n));
        };
        auto result = ed::krylov::lanczos_tridiag(
            matvec,
            static_cast<std::size_t>(N),
            v0.data(),
            opts);
        alpha = std::move(result.alpha);
        beta  = std::move(result.beta);
        return static_cast<int>(alpha.size());
    }

    std::vector<ComplexVector> basis_storage;
    std::vector<ComplexVector>* basis_ptr =
        (reorth_freq > 0) ? &basis_storage : nullptr;
    return build_lanczos_tridiagonal_with_basis(
        std::move(H), v0, N, max_iter, tol,
        full_reorth, reorth_freq,
        alpha, beta, basis_ptr);
}

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
            E_weighted_sum += ritz_values[i] * boltz;
            E2_weighted_sum += ritz_values[i] * ritz_values[i] * boltz;
        }
        
        // Store raw values for averaging
        thermo.Z_sample[t] = Z_sample;
        thermo.E_weighted[t] = E_weighted_sum;
        thermo.E2_weighted[t] = E2_weighted_sum;
        
        // Compute derived quantities for this sample
        if (Z_sample > 1e-300) {
            double E_avg = E_weighted_sum / Z_sample;
            double E2_avg = E2_weighted_sum / Z_sample;
            
            // Thermodynamic quantities
            thermo.energy[t] = E_avg;
            thermo.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
            
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
                // Rescale Z_sample to common reference energy
                double delta_e = sample_data[s].e_min - e_min_global;
                double rescale = std::exp(-beta * delta_e);
                
                Z_avg[t] += sample_data[s].Z_sample[t] * rescale;
                E_weighted_avg[t] += sample_data[s].E_weighted[t] * rescale;
                E2_weighted_avg[t] += sample_data[s].E2_weighted[t] * rescale;
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
                double E_avg = E_weighted_avg[t] / Z_avg[t];
                double E2_avg = E2_weighted_avg[t] / Z_avg[t];
                
                results.thermo_data.energy[t] = E_avg;
                results.thermo_data.specific_heat[t] = beta * beta * (E2_avg - E_avg * E_avg);
                
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

/**
 * @brief Main FTLM driver function
 */
FTLMResults finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const FTLMParameters& params,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins,
    const std::string& output_dir
) {
    // Phase 6.1: dim-aware OMP+BLAS thread cap (see lanczos() rationale).
    // FTLM runs ``num_samples`` independent Lanczos chains; each one is
    // limited by the same memory-bandwidth cliff that hits the standalone
    // ``lanczos()`` driver, so the same ``auto_threads_for_dim(N)`` bound
    // helps here too.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Finite Temperature Lanczos Method (FTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Temperature range: [" << temp_min << ", " << temp_max << "]" << std::endl;
        std::cout << "Temperature bins: " << num_temp_bins << std::endl;
    }

    // log() / 1/T below diverge at T=0; reject non-positive temperatures
    // before constructing the grid.
    if (!(temp_min > 0.0) || !(temp_max > 0.0)) {
        throw std::invalid_argument(
            "finite_temperature_lanczos: temp_min and temp_max must both "
            "be > 0 (got temp_min=" + std::to_string(temp_min) +
            ", temp_max=" + std::to_string(temp_max) + ").");
    }

    // Generate temperature grid (logarithmic spacing)
    std::vector<double> temperatures(num_temp_bins);
    double log_tmin = std::log(temp_min);
    double log_tmax = std::log(temp_max);
    double log_step = (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);
    
    for (int i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }
    
    // Reproducible base seed: 0 means "draw from random_device"; non-zero
    // is taken verbatim. Each sample derives its own per-thread RNG below
    // so the loop is safe to parallelize without giving up reproducibility.
    std::uint64_t base_seed = params.random_seed;
    if (base_seed == 0) {
        std::random_device rd;
        base_seed = (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }

    // Storage for results — pre-sized so OpenMP threads can write to
    // disjoint slots without locks. Empty slots (Lanczos failure) are
    // filtered out by the post-loop sweep.
    FTLMResults results;
    results.total_samples = params.num_samples;
    std::vector<ThermodynamicData> sample_data_indexed(params.num_samples);
    std::vector<bool>              sample_valid(params.num_samples, false);
    std::vector<double>            ground_state_indexed(
        params.num_samples, std::numeric_limits<double>::infinity());

    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/ftlm_samples";
        safe_system_call(cmd);
    }

    // Loop over samples — optionally parallelized across samples.
    //
    // Each sample is fully independent (random vector, Lanczos, thermo),
    // so the loop is embarrassingly parallel in principle. However the
    // Hv callback we receive is a std::function that may close over
    // operator internals which are NOT guaranteed to be thread-safe
    // (e.g., shared scratch buffers inside the Operator). Calling such
    // a callback from multiple threads concurrently corrupts state and
    // crashes. To stay correct by default while still letting users
    // who own thread-safe operators opt in, we gate parallelism on
    // ED_FTLM_PARALLEL=1.
    static const bool ftlm_omp_enabled = []() {
        const char* s = std::getenv("ED_FTLM_PARALLEL");
        return (s && s[0] == '1');
    }();
    const bool run_parallel = ftlm_omp_enabled && (params.num_samples > 1);

    #pragma omp parallel for schedule(dynamic) if(run_parallel)
    for (int sample = 0; sample < params.num_samples; sample++) {
        if (verbose) {
            #pragma omp critical(ftlm_log)
            std::cout << "\n--- FTLM Sample " << sample + 1 << " / "
                      << params.num_samples << " ---\n";
        }

        // Per-sample, per-thread RNG: splitmix-mixed (base_seed, sample)
        std::uint64_t z = base_seed + 0x9E3779B97F4A7C15ULL * (std::uint64_t)(sample + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z =  z ^ (z >> 31);
        std::mt19937 local_gen(static_cast<std::mt19937::result_type>(z));

        ComplexVector v0 = generateGaussianRandomVector(N, local_gen);

        std::vector<double> alpha, beta;
        uint64_t iterations = build_lanczos_tridiagonal(
            H, v0, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta
        );

        if (verbose) {
            #pragma omp critical(ftlm_log)
            std::cout << "  [sample " << sample << "] Lanczos iterations: "
                      << iterations << std::endl;
        }

        std::vector<double> ritz_values, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);

        if (ritz_values.empty()) {
            #pragma omp critical(ftlm_log)
            std::cerr << "  Warning: Tridiagonal diagonalization failed (sample "
                      << sample << ")" << std::endl;
            continue;
        }

        ground_state_indexed[sample] = ritz_values[0];

        ThermodynamicData sample_thermo = compute_ftlm_thermodynamics(
            ritz_values, weights, temperatures, N
        );
        sample_data_indexed[sample] = std::move(sample_thermo);
        sample_valid[sample]        = true;

        if (params.store_intermediate && !output_dir.empty()) {
            #pragma omp critical(ftlm_h5)
            {
                std::string h5_file = output_dir + "/ed_results.h5";
                if (!HDF5IO::fileExists(h5_file)) {
                    HDF5IO::createOrOpenFile(output_dir);
                }
                HDF5IO::FTLMThermodynamicSample h5_sample;
                h5_sample.temperatures   = temperatures;
                h5_sample.energy         = sample_data_indexed[sample].energy;
                h5_sample.specific_heat  = sample_data_indexed[sample].specific_heat;
                h5_sample.entropy        = sample_data_indexed[sample].entropy;
                h5_sample.free_energy    = sample_data_indexed[sample].free_energy;
                HDF5IO::saveFTLMThermodynamicSample(h5_file, sample, h5_sample);
                if (verbose) {
                    std::cout << "Saved FTLM sample " << sample << " to HDF5" << std::endl;
                }
            }
        }
    }

    // Compact valid samples into the dense vectors expected by the rest
    // of the routine. Order matches sample index, preserving determinism.
    std::vector<ThermodynamicData> sample_data;
    std::vector<double>            ground_state_estimates;
    sample_data.reserve(params.num_samples);
    ground_state_estimates.reserve(params.num_samples);
    for (int s = 0; s < params.num_samples; ++s) {
        if (sample_valid[s]) {
            sample_data.push_back(std::move(sample_data_indexed[s]));
            ground_state_estimates.push_back(ground_state_indexed[s]);
        }
    }

    // Average over all samples
    if (verbose) {
        std::cout << "\n--- Averaging over " << sample_data.size() << " samples ---\n";
    }

    if (params.compute_error_bars) {
        results.per_sample_data = sample_data;
    }

    average_ftlm_samples(sample_data, results);

    // Estimate ground state as minimum across all samples
    if (!ground_state_estimates.empty()) {
        results.ground_state_estimate = *std::min_element(
            ground_state_estimates.begin(), ground_state_estimates.end()
        );
        if (verbose) {
            std::cout << "Best ground state estimate: " << results.ground_state_estimate << std::endl;
        }
    }
    
    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "FTLM Calculation Complete\n";
        std::cout << "==========================================\n";
    }

    return results;
}

/**
 * @brief Save FTLM results to HDF5 file and unified text format
 */
void save_ftlm_results(
    const FTLMResults& results,
    const std::string& filename
) {
    // Extract directory from filename to create HDF5 file
    std::string directory = filename.substr(0, filename.find_last_of('/'));
    if (directory.empty()) directory = ".";
    
    try {
        std::string h5_path = HDF5IO::createOrOpenFile(directory);
        
        HDF5IO::saveFTLMThermodynamics(
            h5_path,
            results.thermo_data.temperatures,
            results.thermo_data.energy,
            results.energy_error,
            results.thermo_data.specific_heat,
            results.specific_heat_error,
            results.thermo_data.entropy,
            results.entropy_error,
            results.thermo_data.free_energy,
            results.free_energy_error,
            results.total_samples,
            "FTLM"
        );
        
        std::cout << "FTLM results saved to: " << h5_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error saving FTLM results to HDF5: " << e.what() << std::endl;
    }
}

/**
 * @brief Combine FTLM results from multiple symmetry sectors
 */
ThermodynamicData combine_ftlm_sector_results(
    const std::vector<FTLMResults>& sector_results,
    const std::vector<uint64_t>& sector_dims
) {
    if (sector_results.empty()) {
        throw std::runtime_error("combine_ftlm_sector_results: No sector results to combine");
    }
    
    if (sector_results.size() != sector_dims.size()) {
        throw std::runtime_error("combine_ftlm_sector_results: Mismatch between number of sectors and dimensions");
    }
    
    size_t n_sectors = sector_results.size();
    const bool verbose_combine = ed_dssf_verbose();
    if (verbose_combine) {
        std::cout << "\n=== Combining FTLM Results from " << n_sectors
                  << " Symmetry Sectors ===" << std::endl;
    }

    // All sectors should have the same temperature grid
    const auto& temps = sector_results[0].thermo_data.temperatures;
    size_t n_temps = temps.size();
    
    // Verify all sectors have same temperature grid
    for (size_t s = 1; s < n_sectors; ++s) {
        if (sector_results[s].thermo_data.temperatures.size() != n_temps) {
            throw std::runtime_error("combine_ftlm_sector_results: Sectors have different temperature grids");
        }
    }
    
    // Initialize combined results
    ThermodynamicData combined;
    combined.temperatures = temps;
    combined.energy.resize(n_temps, 0.0);
    combined.specific_heat.resize(n_temps, 0.0);
    combined.entropy.resize(n_temps, 0.0);
    combined.free_energy.resize(n_temps, 0.0);
    
    // Report sector dimensions
    uint64_t total_dim = 0;
    for (size_t s = 0; s < n_sectors; ++s) {
        if (verbose_combine) {
            std::cout << "  Sector " << s << ": dimension = " << sector_dims[s] << std::endl;
        }
        total_dim += sector_dims[s];
    }
    if (verbose_combine) {
        std::cout << "  Total dimension: " << total_dim << std::endl;
    }
    
    // For each temperature, combine sector contributions
    for (size_t t = 0; t < n_temps; ++t) {
        double T = temps[t];
        double beta = 1.0 / T;
        
        // Step 1: Compute partition function for each sector
        // Z_s(β) = exp(-β F_s)
        // Since sectors may have different ground state energies, we need to use a reference
        // to avoid numerical overflow/underflow
        
        // Find minimum free energy across all sectors for numerical stability
        double F_ref = sector_results[0].thermo_data.free_energy[t];
        for (size_t s = 1; s < n_sectors; ++s) {
            double F_s = sector_results[s].thermo_data.free_energy[t];
            if (F_s < F_ref) {
                F_ref = F_s;
            }
        }
        
        // Compute shifted partition functions: Z_s = exp(-β(F_s - F_ref))
        std::vector<double> Z_sectors;
        double Z_total = 0.0;
        
        for (size_t s = 0; s < n_sectors; ++s) {
            double F_s = sector_results[s].thermo_data.free_energy[t];
            double delta_F = F_s - F_ref;
            double Z_s = std::exp(-beta * delta_F);
            
            // Handle numerical overflow/underflow
            if (!std::isfinite(Z_s) || Z_s < 0.0) {
                std::cerr << "Warning: Numerical issue in sector " << s << " at T=" << T 
                          << ", F_s=" << F_s << ", delta_F=" << delta_F << std::endl;
                Z_s = 0.0;  // Will be handled below
            }
            
            Z_sectors.push_back(Z_s);
            Z_total += Z_s;
        }
        
        // Check for numerical issues
        if (Z_total <= 1e-300 || !std::isfinite(Z_total)) {
            std::cerr << "Error: Total partition function is zero or invalid at T=" << T << std::endl;
            std::cerr << "  This suggests all sectors have very high free energies." << std::endl;
            // Use the minimum free energy sector as fallback
            combined.free_energy[t] = F_ref;
            combined.energy[t] = sector_results[0].thermo_data.energy[t];  // Will be overwritten if Z_total > 0
            combined.specific_heat[t] = 0.0;
            combined.entropy[t] = 0.0;
            continue;  // Skip to next temperature
        }
        
        // Total free energy with reference shift: F_total = F_ref - T ln(Z_total)
        combined.free_energy[t] = F_ref - T * std::log(Z_total);
        
        // Step 2: Compute sector weights (normalized partition function contributions)
        std::vector<double> weights(n_sectors);
        for (size_t s = 0; s < n_sectors; ++s) {
            weights[s] = Z_sectors[s] / Z_total;
        }
        
        // Debug output for first and last temperature
        if (verbose_combine && (t == 0 || t == n_temps - 1)) {
            std::cout << "\n  T=" << T << " (beta=" << beta << "):" << std::endl;
            std::cout << "    F_ref=" << F_ref << std::endl;
            for (size_t s = 0; s < n_sectors; ++s) {
                std::cout << "    Sector " << s << ": F=" << sector_results[s].thermo_data.free_energy[t]
                          << ", Z_s/Z_total=" << weights[s] << ", <E>=" << sector_results[s].thermo_data.energy[t]
                          << std::endl;
            }
        }
        
        // Step 3: Combine observables with proper weights
        // For energy: <E>_total = Σ_s (Z_s/Z_total) <E>_s
        // For variance: Var[E]_total requires combining sector variances
        double E_total = 0.0;
        double E2_total = 0.0;
        
        for (size_t s = 0; s < n_sectors; ++s) {
            double w_s = weights[s];
            double E_s = sector_results[s].thermo_data.energy[t];
            double C_s = sector_results[s].thermo_data.specific_heat[t];
            
            // Weighted energy: <E> = Σ_s w_s <E>_s
            E_total += w_s * E_s;
            
            // For specific heat combination, we need <E²>:
            // C_s = β²(<E²>_s - <E>_s²) → <E²>_s = C_s/β² + <E>_s²
            // Then: <E²>_total = Σ_s w_s <E²>_s
            double E2_s = C_s / (beta * beta) + E_s * E_s;
            E2_total += w_s * E2_s;
        }
        
        // Step 4: Final thermodynamic quantities
        combined.energy[t] = E_total;
        
        // Combined specific heat: C = β²(<E²> - <E>²)
        combined.specific_heat[t] = beta * beta * (E2_total - E_total * E_total);
        
        // Entropy from thermodynamic relation: S = β(E - F)
        combined.entropy[t] = beta * (E_total - combined.free_energy[t]);
        
        // Additional diagnostic output for first/last temperature
        if (verbose_combine && (t == 0 || t == n_temps - 1)) {
            std::cout << "    Combined: F=" << combined.free_energy[t]
                      << ", <E>=" << combined.energy[t]
                      << ", C=" << combined.specific_heat[t]
                      << ", S=" << combined.entropy[t] << std::endl;
        }
    }

    // Final verification: check that combined results make physical sense
    if (verbose_combine) {
        std::cout << "\n=== Verification of Combined Results ===" << std::endl;
    }
    
    // Check a mid-range temperature for sanity
    size_t mid_t = n_temps / 2;
    double mid_T = temps[mid_t];
    double mid_E = combined.energy[mid_t];
    
    // Find min/max energies across sectors at this temperature
    double E_min = sector_results[0].thermo_data.energy[mid_t];
    double E_max = E_min;
    for (size_t s = 1; s < n_sectors; ++s) {
        double E_s = sector_results[s].thermo_data.energy[mid_t];
        if (E_s < E_min) E_min = E_s;
        if (E_s > E_max) E_max = E_s;
    }
    
    if (verbose_combine) {
        std::cout << "  At T=" << mid_T << ":" << std::endl;
        std::cout << "    Sector energy range: [" << E_min << ", " << E_max << "]" << std::endl;
        std::cout << "    Combined energy: " << mid_E << std::endl;
    }

    if (mid_E < E_min || mid_E > E_max) {
        // Always warn (sanity issue), even when verbose is off.
        std::cerr << "    WARNING: Combined energy at T=" << mid_T
                  << " is outside sector range [" << E_min << ", " << E_max
                  << "] (got " << mid_E << ")." << std::endl;
    } else if (verbose_combine) {
        std::cout << "    Combined energy is within expected sector range" << std::endl;
    }

    // Check that specific heat is non-negative
    bool all_positive_C = true;
    for (size_t t = 0; t < n_temps; ++t) {
        if (combined.specific_heat[t] < -1e-10) {  // Allow small numerical errors
            all_positive_C = false;
            std::cerr << "  WARNING: Negative specific heat at T=" << temps[t]
                      << ", C=" << combined.specific_heat[t] << std::endl;
        }
    }

    if (verbose_combine) {
        if (all_positive_C) {
            std::cout << "  All specific heat values are non-negative" << std::endl;
        }
        std::cout << "\nSuccessfully combined thermodynamic data from all sectors"
                  << std::endl;
        std::cout << "=== Sector Combination Complete ===" << std::endl;
    }

    return combined;
}

/**
 * @brief Helper function to compute spectral function from Ritz values and weights
 * 
 * @param ritz_values Eigenvalues (energies)
 * @param weights Statistical weights (without thermal factors)
 * @param frequencies Frequency grid
 * @param broadening Lorentzian broadening parameter
 * @param temperature Temperature (if <= 0, no thermal weighting applied)
 * @param spectral_function Output spectral function
 */
static void compute_spectral_function(
    const std::vector<double>& ritz_values,
    const std::vector<double>& weights,
    const std::vector<double>& frequencies,
    double broadening,
    double temperature,
    std::vector<double>& spectral_function
){
    uint64_t n_omega = frequencies.size();
    uint64_t n_states = ritz_values.size();
    
    spectral_function.resize(n_omega, 0.0);
    
    // Compute thermal weights if temperature > 0
    std::vector<double> thermal_weights = weights;
    
    if (temperature > 1e-14) {
        double beta = 1.0 / temperature;
        
        // Find minimum energy for numerical stability
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        // Compute partition function with shifted energies
        double Z = 0.0;
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            thermal_weights[i] = weights[i] * std::exp(-beta * shifted_energy);
            Z += thermal_weights[i];
        }
        
        // Normalize by partition function
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                thermal_weights[i] /= Z;
            }
        } else {
            // Very low temperature - only ground state contributes
            thermal_weights.assign(n_states, 0.0);
            uint64_t gs_idx = std::distance(ritz_values.begin(),
                                      std::min_element(ritz_values.begin(), ritz_values.end()));
            thermal_weights[gs_idx] = weights[gs_idx];
            // Normalize
            double sum = 0.0;
            for (double w : thermal_weights) sum += w;
            if (sum > 0) {
                for (double& w : thermal_weights) w /= sum;
            }
        }
    }
    
    // For each frequency, sum contributions from all states
    // S(ω,T) = Σ_i w_i * exp(-βE_i)/Z * δ(ω - E_i)
    // Using Lorentzian broadening: δ(ω - E) → (η/π) / ((ω - E)² + η²)
    double norm_factor = broadening / M_PI;
    
    for (int i_omega = 0; i_omega < n_omega; i_omega++) {
        double omega = frequencies[i_omega];
        
        for (int i = 0; i < n_states; i++) {
            double delta = omega - ritz_values[i];
            double lorentzian = norm_factor / (delta * delta + broadening * broadening);
            spectral_function[i_omega] += thermal_weights[i] * lorentzian;
        }
    }
}

/**
 * @brief Compute complex spectral function from complex weights
 * 
 * For cross-correlation S_{O1,O2}(ω) = ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩, the weights can be complex.
 * This function computes both real and imaginary parts of the spectral function.
 */
static void compute_spectral_function_complex(
    const std::vector<double>& ritz_values,
    const std::vector<Complex>& complex_weights,
    const std::vector<double>& frequencies,
    double broadening,
    double temperature,
    std::vector<double>& spectral_function_real,
    std::vector<double>& spectral_function_imag
){
    uint64_t n_omega = frequencies.size();
    uint64_t n_states = ritz_values.size();
    
    spectral_function_real.resize(n_omega, 0.0);
    spectral_function_imag.resize(n_omega, 0.0);
    
    // Compute thermal weights if temperature > 0
    std::vector<Complex> thermal_weights = complex_weights;
    
    if (temperature > 1e-14) {
        double beta = 1.0 / temperature;
        
        // Find minimum energy for numerical stability
        double e_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        // Compute partition function with shifted energies
        // Z = Σ_i exp(-βE_i) where the sum is over Krylov states
        // The weights already contain the ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ matrix elements
        double Z = 0.0;
        for (int i = 0; i < n_states; i++) {
            double shifted_energy = ritz_values[i] - e_min;
            double boltzmann_factor = std::exp(-beta * shifted_energy);
            Z += boltzmann_factor;
        }
        
        // Apply thermal weights: w_n → w_n * exp(-βE_n) / Z
        if (Z > 1e-300) {
            for (int i = 0; i < n_states; i++) {
                double shifted_energy = ritz_values[i] - e_min;
                double boltzmann_factor = std::exp(-beta * shifted_energy);
                thermal_weights[i] = complex_weights[i] * (boltzmann_factor / Z);
            }
        } else {
            // Very low temperature - only ground state contributes
            thermal_weights.assign(n_states, Complex(0.0, 0.0));
            uint64_t gs_idx = std::distance(ritz_values.begin(),
                                      std::min_element(ritz_values.begin(), ritz_values.end()));
            thermal_weights[gs_idx] = complex_weights[gs_idx];
            // Normalize by complex magnitude
            Complex sum = Complex(0.0, 0.0);
            for (const auto& w : thermal_weights) sum += w;
            if (std::abs(sum) > 1e-300) {
                for (auto& w : thermal_weights) w /= sum;
            }
        }
    }
    
    // For each frequency, sum contributions from all states
    // S(ω,T) = Σ_i w_i * exp(-βE_i)/Z * δ(ω - E_i)
    // Using Lorentzian broadening: δ(ω - E) → (η/π) / ((ω - E)² + η²)
    double norm_factor = broadening / M_PI;
    
    for (int i_omega = 0; i_omega < n_omega; i_omega++) {
        double omega = frequencies[i_omega];
        
        for (int i = 0; i < n_states; i++) {
            double delta = omega - ritz_values[i];
            double lorentzian = norm_factor / (delta * delta + broadening * broadening);
            Complex contribution = thermal_weights[i] * lorentzian;
            spectral_function_real[i_omega] += contribution.real();
            spectral_function_imag[i_omega] += contribution.imag();
        }
    }
}
/**
 * @brief Compute dynamical response with random initial states (finite temperature)
 *
 * The single-state `compute_dynamical_response(psi, ...)` overload was retired in
 * the minimalist-architecture rev (May 2026); its semantics live on through
 * `ed::observables::cf_dynamical_correlator` (see include/ed/observables/cf_dynamical.h)
 * and the GS DSSF entry points below.
 */
DynamicalResponseResults compute_dynamical_response_thermal(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature,
    const std::string& output_dir
){
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Thermal Dynamical Response (FTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
        std::cout << "Broadening: " << params.broadening << std::endl;
        if (temperature > 1e-14) {
            std::cout << "Temperature: " << temperature << std::endl;
        } else {
            std::cout << "Temperature: 0 (no thermal weighting)" << std::endl;
        }
    }
    
    DynamicalResponseResults results;
    results.total_samples = params.num_samples;
    
    // Generate frequency grid
    results.frequencies.resize(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
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
    
    // Storage for per-sample spectral functions
    std::vector<std::vector<double>> sample_spectra;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/dynamical_samples";
        safe_system_call(cmd);
    }

    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        if (verbose) {
            std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        }

        // Generate random initial state |ψ⟩ (Gaussian for unbiased trace estimate)
        ComplexVector psi = generateGaussianRandomVector(N, gen);

        // Apply operator O: |φ⟩ = O|ψ⟩
        ComplexVector phi(N);
        O(psi.data(), phi.data(), N);

        // Get norm of |φ⟩
        double phi_norm = cblas_dznrm2(N, phi.data(), 1);
        if (phi_norm < 1e-14) {
            if (verbose) {
                std::cout << "  Warning: O|ψ⟩ has zero norm, skipping sample\n";
            }
            continue;
        }

        if (verbose) {
            std::cout << "  Norm of O|ψ⟩: " << phi_norm << std::endl;
        }

        // Normalize |φ⟩
        Complex phi_scale(1.0/phi_norm, 0.0);
        cblas_zscal(N, &phi_scale, phi.data(), 1);

        // Build Lanczos tridiagonal
        std::vector<double> alpha, beta;
        uint64_t iterations = build_lanczos_tridiagonal(
            H, phi, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta
        );

        if (verbose) {
            std::cout << "  Lanczos iterations: " << iterations << std::endl;
        }
        
        // Diagonalize tridiagonal and extract Ritz values/weights
        std::vector<double> ritz_values, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, weights);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            continue;
        }
        
        // Scale weights by phi_norm squared
        for (int i = 0; i < weights.size(); i++) {
            weights[i] *= phi_norm * phi_norm;
        }
        
        // Compute spectral function for this sample
        std::vector<double> sample_spectrum;
        compute_spectral_function(ritz_values, weights, results.frequencies,
                                 params.broadening, temperature, sample_spectrum);
        
        sample_spectra.push_back(sample_spectrum);
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMDynamicalSample h5_sample;
            h5_sample.frequencies = results.frequencies;
            h5_sample.spectral_real = sample_spectrum;
            h5_sample.spectral_imag = std::vector<double>(sample_spectrum.size(), 0.0);  // Real for self-correlation
            
            HDF5IO::saveFTLMDynamicalSample(h5_file, sample, h5_sample, false);
        }
    }
    
    // Average over all samples (FTLM thermal)
    uint64_t n_valid_samples = sample_spectra.size();
    if (verbose) {
        std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    }
    
    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_function_imag.resize(num_omega_bins, 0.0);  // Self-correlation: imaginary part is zero
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained\n";
        return results;
    }
    
    // Compute mean
    for (int s = 0; s < n_valid_samples; s++) {
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_function[i] += sample_spectra[s][i];
        }
    }
    
    for (int i = 0; i < num_omega_bins; i++) {
        results.spectral_function[i] /= n_valid_samples;
    }
    
    // Compute standard error
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int i = 0; i < num_omega_bins; i++) {
                double diff = sample_spectra[s][i] - results.spectral_function[i];
                results.spectral_error[i] += diff * diff;
            }
        }
        
        double norm_factor = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_error[i] = std::sqrt(results.spectral_error[i]) / norm_factor;
        }
    }
    
    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Thermal Dynamical Response Complete\n";
        std::cout << "==========================================\n";
    }

    return results;
}

/**
 * @brief Save spectral function to text file in unified format
 * 
 * Unified format: 5 columns
 *   # Frequency  Re[S(ω)]  Im[S(ω)]  Re[Error]  Im[Error]
 * 
 * This provides consistent output across all spectral function methods:
// save_dynamical_response_results was retired in the minimalist-
// architecture rev (May 2026): no external callers. Workflows that need
// to persist a DynamicalResponseResults go through
// HDF5IO::saveDynamicalResponseFull directly with a pre-computed
// operator_name (which is the relevant metadata anyway, and saved as
// /dynamical/<operator_name>/ in the unified HDF5 store).

/**
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = <ψ|O1†δ(ω - H)O2|ψ>
 */
DynamicalResponseResults compute_dynamical_correlation(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double temperature,
    const std::string& output_dir,
    double energy_shift
){
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Dynamical Correlation: S(ω) = <O₁†δ(ω-H)O₂>\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
        std::cout << "Broadening: " << params.broadening << std::endl;
        if (temperature > 1e-14) {
            std::cout << "Temperature: " << temperature << std::endl;
        } else {
            std::cout << "Temperature: 0 (no thermal weighting)" << std::endl;
        }
    }
    
    DynamicalResponseResults results;
    results.total_samples = params.num_samples;
    
    // Generate frequency grid
    results.frequencies.resize(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        results.frequencies[i] = omega_min + i * omega_step;
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
    
    // Storage for per-sample spectral functions (real and imaginary parts)
    std::vector<std::vector<double>> sample_spectra_real;
    std::vector<std::vector<double>> sample_spectra_imag;
    
    // Create output directory if needed
    if (!output_dir.empty() && params.store_intermediate) {
        std::string cmd = "mkdir -p " + output_dir + "/dynamical_correlation_samples";
        safe_system_call(cmd);
    }
    
    // Loop over random samples
    for (int sample = 0; sample < params.num_samples; sample++) {
        if (verbose) {
            std::cout << "\n--- Sample " << sample + 1 << " / " << params.num_samples << " ---\n";
        }

        // Generate random initial state |ψ⟩ (Gaussian for unbiased trace estimate)
        ComplexVector psi = generateGaussianRandomVector(N, gen);
        
        // Apply operator O2: |φ⟩ = O₂|ψ⟩
        ComplexVector phi(N);
        O2(psi.data(), phi.data(), N);
        
        // Get norm of |φ⟩
        double phi_norm = cblas_dznrm2(N, phi.data(), 1);
        if (phi_norm < 1e-14) {
            if (verbose) {
                std::cout << "  Warning: O₂|ψ⟩ has zero norm, skipping sample\n";
            }
            continue;
        }

        if (verbose) {
            std::cout << "  Norm of O₂|ψ⟩: " << phi_norm << std::endl;
        }
        
        // Normalize |φ⟩
        Complex phi_scale(1.0/phi_norm, 0.0);
        cblas_zscal(N, &phi_scale, phi.data(), 1);
        
        // Build Lanczos tridiagonal for H starting from |φ⟩
        // Store basis vectors for computing matrix elements
        std::vector<double> alpha, beta;
        std::vector<ComplexVector> lanczos_vectors;
        
        uint64_t iterations = build_lanczos_tridiagonal_with_basis(
            H, phi, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha, beta, &lanczos_vectors
        );
        
        uint64_t m = alpha.size();
        if (verbose) {
            std::cout << "  Lanczos iterations: " << m << std::endl;
        }
        
        // Diagonalize tridiagonal (need eigenvectors for weight computation)
        std::vector<double> ritz_values, dummy_weights;
        std::vector<double> evecs;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, dummy_weights, &evecs);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Tridiagonal diagonalization failed\n";
            continue;
        }
        
        // For dynamical structure factors, shift energies so ground state is at E=0
        // This ensures spectral function has weight only at positive frequencies (excitation energies)
        if (sample == 0 && verbose) {
            double E_shift_announced;
            if (std::abs(energy_shift) > 1e-14) {
                E_shift_announced = energy_shift;
                std::cout << "  Using provided ground state energy shift: "
                          << E_shift_announced << std::endl;
            } else {
                E_shift_announced =
                    *std::min_element(ritz_values.begin(), ritz_values.end());
                std::cout << "  Ground state energy (auto-detected from Krylov): "
                          << E_shift_announced << std::endl;
            }
            std::cout << "  Shifting to excitation energies (E_gs = 0)" << std::endl;
        }
        
        // Apply energy shift for this sample
        double E_shift = (std::abs(energy_shift) > 1e-14) ? 
                         energy_shift : 
                         *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int i = 0; i < m; i++) {
            ritz_values[i] -= E_shift;
        }
        
        // Compute weights ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ for cross-correlation.
        //   |n⟩ = Σ_j V[j,n] |v_j⟩  with |v_0⟩ = O₂|ψ⟩/‖O₂|ψ⟩‖, V real.
        //
        // The matrix element ⟨ψ|O₁†|n⟩ = ⟨O₁ψ|n⟩ = Σ_j V[j,n] · ⟨O₁ψ|v_j⟩.
        // ⟨O₁ψ|v_j⟩ depends only on j, not on n -- precompute it once into
        // p[j] (m complex zdotc calls of length N) and reuse across n.
        // Then collapse the n-loop into a single BLAS-2 dgemv on the real and
        // imaginary parts of p separately (V is real, m×m).
        // Old: m × m zdotc(N) ⇒ O(m²·N) BLAS-1 traffic, dominated the loop.
        // New: m zdotc(N) plus two real m×m cblas_dgemv,
        //      ⇒ O(m·N) BLAS-1 + 2 BLAS-2(m²).
        ComplexVector O1_psi(N);
        O1(psi.data(), O1_psi.data(), N);

        std::vector<Complex> p(m);  // p[j] = ⟨O₁ψ|v_j⟩
        for (uint64_t j = 0; j < m; ++j) {
            cblas_zdotc_sub(N, O1_psi.data(), 1,
                            lanczos_vectors[j].data(), 1, &p[j]);
        }

        // Split p into real / imaginary parts and apply V^T to each.
        // evecs is row-major with V[j,n] = evecs[n*m + j], i.e. column-major
        // when viewed with leading dimension m. We compute
        //     overlap_O1_re[n] = Σ_j evecs[n*m + j] * p_re[j]
        //     overlap_O1_im[n] = Σ_j evecs[n*m + j] * p_im[j]
        // which is V^T · p_part with V column-major (m×m), so use CblasTrans.
        std::vector<double> p_re(m), p_im(m);
        for (uint64_t j = 0; j < m; ++j) {
            p_re[j] = p[j].real();
            p_im[j] = p[j].imag();
        }
        std::vector<double> overlap_O1_re(m), overlap_O1_im(m);
        cblas_dgemv(CblasColMajor, CblasTrans,
                    /*M=*/static_cast<int>(m), /*N=*/static_cast<int>(m),
                    1.0, evecs.data(), /*lda=*/static_cast<int>(m),
                    p_re.data(), 1, 0.0, overlap_O1_re.data(), 1);
        cblas_dgemv(CblasColMajor, CblasTrans,
                    /*M=*/static_cast<int>(m), /*N=*/static_cast<int>(m),
                    1.0, evecs.data(), /*lda=*/static_cast<int>(m),
                    p_im.data(), 1, 0.0, overlap_O1_im.data(), 1);

        std::vector<Complex> complex_weights(m);
        for (uint64_t n = 0; n < m; ++n) {
            const Complex overlap_O1(overlap_O1_re[n], overlap_O1_im[n]);
            // ⟨n|O₂|ψ⟩ = V[0,n] · ‖O₂|ψ‖   (since |v₀⟩ = O₂|ψ⟩/‖O₂|ψ‖)
            const Complex overlap_O2(evecs[n * m + 0] * phi_norm, 0.0);
            // Weight ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ = ⟨O₁ψ|n⟩ · ⟨n|O₂|ψ⟩.
            // Note: overlap_O1 already equals ⟨O₁ψ|n⟩ via zdotc (v†·u),
            // do NOT take an extra conjugate here.
            complex_weights[n] = overlap_O1 * overlap_O2;
        }
        
        // Compute spectral function for this sample (both real and imaginary parts)
        std::vector<double> sample_spectrum_real, sample_spectrum_imag;
        compute_spectral_function_complex(ritz_values, complex_weights, results.frequencies,
                                         params.broadening, temperature, 
                                         sample_spectrum_real, sample_spectrum_imag);
        
        sample_spectra_real.push_back(sample_spectrum_real);
        sample_spectra_imag.push_back(sample_spectrum_imag);
        
        // Save intermediate data if requested (to HDF5)
        if (params.store_intermediate && !output_dir.empty()) {
            std::string h5_file = output_dir + "/ed_results.h5";
            if (!HDF5IO::fileExists(h5_file)) {
                HDF5IO::createOrOpenFile(output_dir);
            }
            
            HDF5IO::FTLMDynamicalSample h5_sample;
            h5_sample.frequencies = results.frequencies;
            h5_sample.spectral_real = sample_spectrum_real;
            h5_sample.spectral_imag = sample_spectrum_imag;
            
            HDF5IO::saveFTLMDynamicalSample(h5_file, sample, h5_sample, true);  // is_correlation=true
        }
    }
    
    // Average over all samples (Dynamical Correlation FTLM)
    uint64_t n_valid_samples = sample_spectra_real.size();
    if (verbose) {
        std::cout << "\n--- Averaging over " << n_valid_samples << " samples ---\n";
    }

    results.spectral_function.resize(num_omega_bins, 0.0);
    results.spectral_function_imag.resize(num_omega_bins, 0.0);
    results.spectral_error.resize(num_omega_bins, 0.0);
    results.spectral_error_imag.resize(num_omega_bins, 0.0);
    
    if (n_valid_samples == 0) {
        std::cerr << "Error: No valid samples obtained\n";
        return results;
    }
    
    // Compute mean (real and imaginary parts)
    for (int s = 0; s < n_valid_samples; s++) {
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_function[i] += sample_spectra_real[s][i];
            results.spectral_function_imag[i] += sample_spectra_imag[s][i];
        }
    }
    
    for (int i = 0; i < num_omega_bins; i++) {
        results.spectral_function[i] /= n_valid_samples;
        results.spectral_function_imag[i] /= n_valid_samples;
    }
    
    // Compute standard error (real and imaginary parts)
    if (n_valid_samples > 1) {
        for (int s = 0; s < n_valid_samples; s++) {
            for (int i = 0; i < num_omega_bins; i++) {
                double diff_real = sample_spectra_real[s][i] - results.spectral_function[i];
                double diff_imag = sample_spectra_imag[s][i] - results.spectral_function_imag[i];
                results.spectral_error[i] += diff_real * diff_real;
                results.spectral_error_imag[i] += diff_imag * diff_imag;
            }
        }
        
        double norm_factor = std::sqrt(static_cast<double>(n_valid_samples * (n_valid_samples - 1)));
        for (int i = 0; i < num_omega_bins; i++) {
            results.spectral_error[i] = std::sqrt(results.spectral_error[i]) / norm_factor;
            results.spectral_error_imag[i] = std::sqrt(results.spectral_error_imag[i]) / norm_factor;
        }
    }

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Dynamical Correlation Complete\n";
        std::cout << "==========================================\n";
    }

    return results;
}

/**
 * @brief Compute dynamical correlation S_{O1,O2}(ω) = ⟨O₁†(ω)O₂⟩ for a given state
 * 
 * Computes the spectral function S(ω) = Σₙ ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩ δ(ω - Eₙ)
 * where |n⟩ are eigenstates of H with energy Eₙ, for a specific state |ψ⟩.
 * 
 * This function uses the Lehmann representation computed via Lanczos:
 * - Applies O₂ to the given state: |φ⟩ = O₂|ψ⟩
 * - Builds Krylov subspace starting from |φ⟩
 * - Diagonalizes H in the Krylov basis to get approximate eigenstates
 * - Computes weights: ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩
 * - Constructs spectral function with Lorentzian broadening
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param O1 First operator (O₁) matrix-vector product function
 * @param O2 Second operator (O₂) matrix-vector product function
 * @param state Input quantum state |ψ⟩ (must be normalized)
 * @param N Hilbert space dimension
 * @param params Parameters for dynamical response calculation
 * @param omega_min Minimum frequency
 * @param omega_max Maximum frequency
 * @param num_omega_bins Number of frequency points
 * @param temperature Temperature for Boltzmann weighting of eigenstates (0 = no weighting)
 * @return DynamicalResponseResults containing S_{O1,O2}(ω) vs frequency
 */
// The two-operator `compute_dynamical_correlation_state(O1, O2, state, ...)`
// driver was retired in the minimalist-architecture rev (May 2026); use
// `ed::observables::cf_dynamical_correlator` (self-correlator path) for the
// O1==O2 case and run twice for cross-correlator. The compute_lanczos_spectral_data
// (general-spectrum driver) below also covers the O1!=O2 multi-temperature
// case via compute_dynamical_correlation_*_multi_temperature.

/**
 * @brief MEMORY-EFFICIENT spectral function via continued fraction (O1=O2 case)
 * 
 * This version DOES NOT store Lanczos basis vectors, making it suitable for
 * very large Hilbert spaces (>16M states).
 * 
 * Memory: O(N) instead of O(krylov_dim × N)
 */
DynamicalResponseResults compute_dynamical_correlation_state_cf(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    double energy_shift
) {
    // Phase 2.5 orchestrator over `ed::observables::cf_spectral_kernel<CpuBackend>`.
    // The legacy hand-rolled body is preserved below in `#if 0` for archaeology.
    const bool verbose = ed_dssf_verbose();
    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Spectral Function via Continued Fraction (Memory-Efficient)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Memory mode: NO BASIS STORAGE (O(N) memory)" << std::endl;
        std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
        std::cout << "Broadening: " << params.broadening << std::endl;
    }

    DynamicalResponseResults results;
    results.total_samples = 1;
    results.omega_min = omega_min;
    results.omega_max = omega_max;

    std::vector<double> omega_grid(num_omega_bins);
    const double omega_step =
        (omega_max - omega_min) /
        static_cast<double>(std::max<uint64_t>(1, num_omega_bins - 1));
    for (uint64_t i = 0; i < num_omega_bins; ++i) {
        omega_grid[i] = omega_min + i * omega_step;
    }

    ed::matvec::CpuBackend backend;
    auto apply_H = [&H, N](const Complex* in, Complex* out, std::size_t /*n*/) {
        H(in, out, static_cast<int>(N));
    };
    auto apply_O = [&O, N](const Complex* in, Complex* out, std::size_t /*n*/) {
        O(in, out, static_cast<int>(N));
    };

    ed::observables::CfSpectralOptions kopts;
    kopts.krylov_dim   = params.krylov_dim;
    kopts.broadening   = params.broadening;
    kopts.energy_shift = energy_shift;
    kopts.tolerance    = params.tolerance;
    kopts.global_n     = N;
    kopts.verbose      = verbose;

    auto cf = ed::observables::cf_spectral_kernel(
        backend, apply_H, apply_O,
        static_cast<std::size_t>(N), state.data(), omega_grid, kopts);

    results.frequencies            = std::move(cf.frequencies);
    results.spectral_function      = std::move(cf.spectral_function);
    results.spectral_function_imag.assign(num_omega_bins, 0.0);
    results.spectral_error.assign(num_omega_bins, 0.0);
    results.spectral_error_imag.assign(num_omega_bins, 0.0);

    if (verbose) {
        std::cout << "  Tridiag size: " << cf.tridiag_size << "\n"
                  << "  ||O|psi>||  = " << cf.phi_norm << "\n"
                  << "  E_shift    = " << cf.energy_shift << "\n"
                  << "==========================================\n"
                  << "Continued Fraction Spectral Complete\n"
                  << "==========================================\n";
    }
    return results;
}


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
// TEMPERATURE-INDEPENDENT SPECTRAL DECOMPOSITION (OPTIMIZATION)
// ============================================================================

/**
 * @brief Compute temperature-independent spectral decomposition via Lanczos
 * 
 * This function runs the Lanczos iteration once to compute the spectral
 * decomposition (eigenvalues and weights) which is temperature-independent.
 * The results can then be reused to efficiently compute S(ω,T) at multiple
 * temperatures without re-running the expensive Lanczos iteration.
 */
LanczosSpectralData compute_lanczos_spectral_data(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double energy_shift
) {
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Computing Temperature-Independent Spectral Data\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Broadening: " << params.broadening << std::endl;
    }

    LanczosSpectralData spectral_data;

    // Verify state is normalized
    double state_norm = cblas_dznrm2(N, state.data(), 1);
    if (state_norm < 1e-14) {
        std::cerr << "  Error: input state has zero norm\n";
        return spectral_data;
    }
    ComplexVector psi = state;
    if (std::abs(state_norm - 1.0) > 1e-10) {
        if (verbose) {
            std::cout << "  Normalizing input state (norm = " << state_norm << ")\n";
        }
        Complex scale(1.0/state_norm, 0.0);
        cblas_zscal(N, &scale, psi.data(), 1);
    }

    // Apply operator O2: |φ⟩ = O₂|ψ⟩
    ComplexVector phi(N);
    O2(psi.data(), phi.data(), N);

    // Get norm of |φ⟩
    double phi_norm = cblas_dznrm2(N, phi.data(), 1);
    if (phi_norm < 1e-14) {
        std::cerr << "  Error: O₂|ψ⟩ has zero norm\n";
        return spectral_data;
    }

    if (verbose) {
        std::cout << "  Norm of O₂|ψ⟩: " << phi_norm << std::endl;
    }

    // Normalize |φ⟩
    Complex phi_scale(1.0/phi_norm, 0.0);
    cblas_zscal(N, &phi_scale, phi.data(), 1);

    // Build Lanczos tridiagonal for H starting from |φ⟩
    std::vector<double> alpha, beta;
    std::vector<ComplexVector> lanczos_vectors;

    uint64_t iterations = build_lanczos_tridiagonal_with_basis(
        H, phi, N, params.krylov_dim, params.tolerance,
        params.full_reorthogonalization, params.reorth_frequency,
        alpha, beta, &lanczos_vectors
    );

    uint64_t m = alpha.size();
    if (verbose) {
        std::cout << "  Lanczos iterations: " << m << std::endl;
    }

    spectral_data.krylov_dim = m;
    spectral_data.lanczos_iterations = iterations;

    // Diagonalize tridiagonal (need eigenvectors for weight computation)
    std::vector<double> ritz_values, dummy_weights;
    std::vector<double> evecs;
    diagonalize_tridiagonal_ritz(alpha, beta, ritz_values, dummy_weights, &evecs);

    if (ritz_values.empty()) {
        std::cerr << "  Error: Tridiagonal diagonalization failed\n";
        return spectral_data;
    }

    // Determine and apply energy shift
    double E_shift;
    if (std::abs(energy_shift) > 1e-14) {
        E_shift = energy_shift;
        if (verbose) {
            std::cout << "  Using provided ground state energy shift: " << E_shift << std::endl;
        }
    } else {
        E_shift = *std::min_element(ritz_values.begin(), ritz_values.end());
        if (verbose) {
            std::cout << "  Ground state energy (auto-detected from Krylov): " << E_shift << std::endl;
        }
    }

    spectral_data.ground_state_energy = E_shift;

    // Shift to excitation energies
    for (uint64_t i = 0; i < m; i++) {
        ritz_values[i] -= E_shift;
    }
    spectral_data.ritz_values = ritz_values;

    if (verbose) {
        std::cout << "  Shifted to excitation energies (E_gs = 0)" << std::endl;
        std::cout << "  Energy range: [" << *std::min_element(ritz_values.begin(), ritz_values.end())
                  << ", " << *std::max_element(ritz_values.begin(), ritz_values.end()) << "]" << std::endl;
    }

    // Compute temperature-independent spectral weights w_n = ⟨ψ|O₁†|n⟩⟨n|O₂|ψ⟩.
    // Same precomputation trick as the other Krylov-cross-correlation paths:
    // p[j] = ⟨O₁ψ|v_j⟩ depends only on j, factor it out of the n-loop to
    // turn an O(m²·N) inner pass into O(m·N) zdotc + O(m²) coeff sums.
    ComplexVector O1_psi(N);
    O1(psi.data(), O1_psi.data(), N);

    std::vector<Complex> p(m);
    for (uint64_t j = 0; j < m; ++j) {
        cblas_zdotc_sub(N, O1_psi.data(), 1,
                        lanczos_vectors[j].data(), 1, &p[j]);
    }

    // Free the basis -- the weight kernel below only needs p[] and evecs.
    lanczos_vectors.clear();
    lanczos_vectors.shrink_to_fit();

    spectral_data.spectral_weights.resize(m);
    for (uint64_t n = 0; n < m; ++n) {
        Complex overlap_O1(0.0, 0.0);
        for (uint64_t j = 0; j < m; ++j) {
            overlap_O1 += p[j] * evecs[n * m + j];
        }
        const Complex overlap_O2(evecs[n * m + 0] * phi_norm, 0.0);
        spectral_data.spectral_weights[n] = overlap_O1 * overlap_O2;
    }

    if (verbose) {
        std::cout << "==========================================\n";
        std::cout << "Spectral Data Computation Complete\n";
        std::cout << "==========================================\n";
    }

    return spectral_data;
}

/**
 * @brief Compute spectral function from Lanczos data for multiple temperatures
 * 
 * This function takes pre-computed spectral data and efficiently computes
 * S(ω,T) for multiple temperatures. This is much faster than re-running
 * Lanczos for each temperature.
 */
std::map<double, DynamicalResponseResults> compute_spectral_function_from_lanczos_data(
    const LanczosSpectralData& spectral_data,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double broadening,
    uint64_t num_samples,
    const std::vector<std::vector<Complex>>* per_sample_weights
) {
    const bool verbose = ed_dssf_verbose();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Computing Spectral Functions for Multiple Temperatures\n";
        std::cout << "==========================================\n";
        std::cout << "Number of temperatures: " << temperatures.size() << std::endl;
        std::cout << "Temperature range: [" << *std::min_element(temperatures.begin(), temperatures.end())
                  << ", " << *std::max_element(temperatures.begin(), temperatures.end()) << "]" << std::endl;
        std::cout << "Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
        std::cout << "Broadening: " << broadening << std::endl;
    }
    
    std::map<double, DynamicalResponseResults> results_map;
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (int i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    const auto& ritz_values = spectral_data.ritz_values;
    const auto& weights = spectral_data.spectral_weights;
    uint64_t m = ritz_values.size();
    
    // T → 0 limit diverges in 1/T below; reject early with a clear message
    // rather than producing inf/NaN spectra.
    for (double T : temperatures) {
        if (!(T > 0.0)) {
            throw std::invalid_argument(
                "compute_spectral_function_from_lanczos_data: temperatures must "
                "be strictly > 0 (got T = " + std::to_string(T) + ")");
        }
    }

    // Compute spectral function for each temperature
    for (double T : temperatures) {
        if (verbose) {
            std::cout << "  Computing for T = " << T << " ..." << std::endl;
        }

        DynamicalResponseResults results;
        results.frequencies = frequencies;
        results.total_samples = num_samples;
        results.omega_min = omega_min;
        results.omega_max = omega_max;
        
        // Initialize spectral function arrays
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        
        // Compute partition function and thermal weights
        double beta = 1.0 / T;
        std::vector<double> thermal_weights(m);
        double Z = 0.0;
        
        // Find minimum energy for numerical stability
        double E_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        for (int n = 0; n < m; n++) {
            double shifted_energy = ritz_values[n] - E_min;
            double boltzmann = std::exp(-beta * shifted_energy);
            thermal_weights[n] = boltzmann;
            Z += boltzmann;
        }
        
        // Normalize thermal weights
        if (Z > 0.0) {
            for (int n = 0; n < m; n++) {
                thermal_weights[n] /= Z;
            }
        }
        
        // Compute spectral function at each frequency. Parallelizing over
        // omega is embarrassingly parallel: each frequency bin does an
        // independent O(m) reduction. This is the dominant cost when
        // num_omega_bins is large and m is moderate.
        const double eta = broadening;
        const double norm_factor = eta / M_PI;
        const double eta_sq = eta * eta;
        const double* ritz_ptr = ritz_values.data();
        const Complex* w_ptr = weights.data();
        const double* tw_ptr = thermal_weights.data();
        const uint64_t m_local = m;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_omega_bins; i++) {
            const double omega = frequencies[i];
            double s_re = 0.0, s_im = 0.0;
            for (uint64_t n = 0; n < m_local; n++) {
                const double dE = omega - ritz_ptr[n];
                const double lorentzian = norm_factor / (dE * dE + eta_sq);
                const double scale = lorentzian * tw_ptr[n];
                s_re += w_ptr[n].real() * scale;
                s_im += w_ptr[n].imag() * scale;
            }
            results.spectral_function[i] = s_re;
            results.spectral_function_imag[i] = s_im;
        }

        // Compute error bars if per-sample data is available. We parallelise
        // the (sample × omega) outer-product evaluation by collapsing two
        // loops; each (s, i) is independent.
        if (per_sample_weights && num_samples > 1) {
            const uint64_t S = std::min<uint64_t>(num_samples, per_sample_weights->size());
            std::vector<std::vector<double>> per_sample_spectral_real(S, std::vector<double>(num_omega_bins, 0.0));
            std::vector<std::vector<double>> per_sample_spectral_imag(S, std::vector<double>(num_omega_bins, 0.0));

            #pragma omp parallel for collapse(2) schedule(static)
            for (uint64_t s = 0; s < S; s++) {
                for (int i = 0; i < num_omega_bins; i++) {
                    const auto& sample_weights = (*per_sample_weights)[s];
                    const uint64_t mn = std::min<uint64_t>(m_local, sample_weights.size());
                    const double omega = frequencies[i];
                    double ss_re = 0.0, ss_im = 0.0;
                    for (uint64_t n = 0; n < mn; n++) {
                        const double dE = omega - ritz_ptr[n];
                        const double lorentzian = norm_factor / (dE * dE + eta_sq);
                        const double scale = lorentzian * tw_ptr[n];
                        ss_re += sample_weights[n].real() * scale;
                        ss_im += sample_weights[n].imag() * scale;
                    }
                    per_sample_spectral_real[s][i] = ss_re;
                    per_sample_spectral_imag[s][i] = ss_im;
                }
            }

            // Standard error of the mean: SE = sqrt(variance / num_samples).
            // Guard against the (num_samples == 1) case implicitly via the
            // outer condition; here we additionally guard the divisor in
            // case S < num_samples (truncated sample buffer).
            const double denom = static_cast<double>(num_samples) * static_cast<double>(num_samples - 1);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < num_omega_bins; i++) {
                const double mean_real = results.spectral_function[i];
                const double mean_imag = results.spectral_function_imag[i];
                double var_real = 0.0, var_imag = 0.0;
                for (uint64_t s = 0; s < S; s++) {
                    const double dr = per_sample_spectral_real[s][i] - mean_real;
                    const double di = per_sample_spectral_imag[s][i] - mean_imag;
                    var_real += dr * dr;
                    var_imag += di * di;
                }
                results.spectral_error[i] = (denom > 0.0) ? std::sqrt(var_real / denom) : 0.0;
                results.spectral_error_imag[i] = (denom > 0.0) ? std::sqrt(var_imag / denom) : 0.0;
            }
        }
        
        results_map[T] = results;
    }
    
    if (verbose) {
        std::cout << "==========================================\n";
        std::cout << "Multi-Temperature Spectral Function Complete\n";
        std::cout << "==========================================\n";
    }
    
    return results_map;
}

/**
 * @brief Optimized version for computing dynamical correlation at multiple temperatures
 * 
 * This combines compute_lanczos_spectral_data and compute_spectral_function_from_lanczos_data
 * into a single convenient function for temperature scans.
 */
std::map<double, DynamicalResponseResults> compute_dynamical_correlation_state_multi_temperature(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    const ComplexVector& state,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift
) {
    const bool verbose = ed_dssf_verbose();
    if (verbose) {
        std::cout << "\n=========================================="  << std::endl;
        std::cout << "OPTIMIZED MULTI-TEMPERATURE DYNAMICAL CORRELATION" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Running Lanczos ONCE for " << temperatures.size() << " temperature points" << std::endl;
        std::cout << "==========================================" << std::endl;
    }
    
    // Step 1: Compute temperature-independent spectral data (Lanczos run)
    LanczosSpectralData spectral_data = compute_lanczos_spectral_data(
        H, O1, O2, state, N, params, energy_shift
    );
    
    if (spectral_data.ritz_values.empty()) {
        std::cerr << "Error: Failed to compute spectral data\n";
        return {};
    }
    
    // Step 2: Compute spectral functions for all temperatures (fast!)
    return compute_spectral_function_from_lanczos_data(
        spectral_data, omega_min, omega_max, num_omega_bins,
        temperatures, params.broadening, 1
    );
}

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
// Implementation core for both the legacy MPI_COMM_WORLD entry point and the
// audit-#4 MPI_Comm-aware overload. All MPI calls within use the `comm`
// parameter (replaced from MPI_COMM_WORLD via sed during the refactor).
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
    const bool verbose = ed_dssf_verbose();
    int mpi_rank_early = 0;
#ifdef WITH_MPI
    MPI_Comm_rank(comm, &mpi_rank_early);
#endif

    // Reject T <= 0 up-front: the inner thermal weight is exp(-β(E-E_min))
    // with β = 1/T, so T <= 0 produces inf/NaN propagated through the entire
    // spectral accumulation. Fail loud, not silent.
    for (double T : temperatures) {
        if (!(T > 0.0)) {
            throw std::invalid_argument(
                "compute_dynamical_correlation_multi_sample_multi_temperature: "
                "temperatures must be strictly > 0 (got T = " + std::to_string(T) + ")");
        }
    }

    if (verbose && mpi_rank_early == 0) {
        std::cout << "\n=========================================="  << std::endl;
        std::cout << "FTLM SPECTRAL FUNCTION (CORRECT FORMULATION)" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Samples: " << params.num_samples << std::endl;
        std::cout << "Temperatures: " << temperatures.size() << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Broadening: " << params.broadening << std::endl;
        std::cout << "==========================================" << std::endl;
        // Per-Ritz cross-correlation S_i(ω) is built via the Lehmann
        // representation over the sample's Krylov basis (see inner loop
        // around `overlap_O1_nk` below). The previous comment about a
        // continued-fraction inner kernel was stale.
        std::cout << "\nUsing FTLM formulation:" << std::endl;
        std::cout << "  S(ω,T) = (1/Z) × Σ_r Σ_i e^{-βε_i} |c_i|² S_i(ω)" << std::endl;
        std::cout << "  where S_i(ω) is the Lehmann-representation cross-spectrum" << std::endl;
        std::cout << "==========================================" << std::endl;
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
    
    // Ground state energy for shifting
    double E_gs = energy_shift;
    if (std::abs(E_gs) < 1e-14) {
        if (verbose) {
            std::cout << "\nDetermining ground state energy from Lanczos...\n";
        }
        // Plain Lanczos for E_gs: a real-only seed lets Operator::apply() take
        // its real-CSR fast path for the entire Krylov space when H is real.
        // The Hutchinson trace estimator below still uses Gaussian complex
        // vectors -- that's the FTLM probability distribution and is unrelated.
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
        
        std::vector<double> alpha, beta;
        build_lanczos_tridiagonal(H, test_state, N, std::min(params.krylov_dim, (uint64_t)100),
                                  params.tolerance, false, 10, alpha, beta);
        
        std::vector<double> ritz_vals, weights;
        diagonalize_tridiagonal_ritz(alpha, beta, ritz_vals, weights);
        
        if (!ritz_vals.empty()) {
            E_gs = *std::min_element(ritz_vals.begin(), ritz_vals.end());
            if (verbose) {
                std::cout << "Ground state energy (estimated): " << E_gs << std::endl;
            }
        }
    } else if (verbose) {
        std::cout << "Using provided ground state energy: " << E_gs << std::endl;
    }
    
    // Generate frequency grid
    std::vector<double> frequencies(num_omega_bins);
    double omega_step = (omega_max - omega_min) / std::max(uint64_t(1), num_omega_bins - 1);
    for (uint64_t i = 0; i < num_omega_bins; i++) {
        frequencies[i] = omega_min + i * omega_step;
    }
    
    // For each temperature, accumulate numerator (Σ_r Σ_i e^{-βε_i} |c_i|² S_i(ω))
    // and partition function (Σ_r Σ_i e^{-βε_i} |c_i|²)
    std::map<double, std::vector<double>> accumulated_spectral;
    std::map<double, std::vector<double>> accumulated_spectral_imag;
    std::map<double, double> accumulated_Z;
    std::map<double, std::vector<std::vector<double>>> per_sample_spectral;  // For error estimation
    std::map<double, std::vector<std::vector<double>>> per_sample_spectral_imag;
    
    for (double T : temperatures) {
        accumulated_spectral[T] = std::vector<double>(num_omega_bins, 0.0);
        accumulated_spectral_imag[T] = std::vector<double>(num_omega_bins, 0.0);
        accumulated_Z[T] = 0.0;
        per_sample_spectral[T] = std::vector<std::vector<double>>();
        per_sample_spectral_imag[T] = std::vector<std::vector<double>>();
    }
    
    // MPI parallelization: distribute samples across ranks
    int mpi_rank = 0, mpi_size = 1;
#ifdef WITH_MPI
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);
#endif
    
    // Calculate sample distribution for this rank
    uint64_t samples_per_rank = params.num_samples / mpi_size;
    uint64_t remainder = params.num_samples % mpi_size;
    uint64_t start_sample = mpi_rank * samples_per_rank + std::min((uint64_t)mpi_rank, remainder);
    uint64_t end_sample = start_sample + samples_per_rank + (mpi_rank < (int)remainder ? 1 : 0);
    uint64_t local_num_samples = end_sample - start_sample;
    
    if (mpi_rank == 0 && verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "FTLM Spectral Function\n";
        std::cout << "==========================================\n";
#ifdef WITH_MPI
        std::cout << "Total MPI ranks: " << mpi_size << "\n";
#endif
        std::cout << "Total samples: " << params.num_samples << "\n";
#ifdef WITH_MPI
        std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
#endif
        std::cout << "==========================================\n";
    }

#ifdef WITH_MPI
    if (verbose) {
        std::cout << "Rank " << mpi_rank << " processing samples ["
                  << start_sample << ", " << end_sample << ") - " << local_num_samples << " samples\n";
    }

    // Synchronize before starting (always; correctness, not chatter)
    MPI_Barrier(comm);

    if (mpi_rank == 0 && verbose) {
        std::cout << "\nStarting parallel sample processing across " << mpi_size << " ranks...\n";
        std::cout << "(Only rank 0 output shown for clarity)\n" << std::endl;
    }
#endif
    
    // How many Ritz states to use per sample for spectral function
    // Using all states is expensive; use states with significant thermal weight
    uint64_t max_ritz_states = std::min(params.krylov_dim, (uint64_t)50);  // Limit for efficiency
    
    // Pre-allocate working vectors to avoid repeated allocations in inner loop
    ComplexVector psi_work(N);  // For eigenstate construction
    ComplexVector phi_work(N);  // For O|psi>
    
#ifdef WITH_MPI
    double start_time = MPI_Wtime();
#else
    auto start_time = std::chrono::high_resolution_clock::now();
#endif
    
    // Loop over random samples assigned to this rank
    for (uint64_t sample_idx = start_sample; sample_idx < end_sample; sample_idx++) {
        if (mpi_rank == 0 && verbose) {
            uint64_t local_idx = sample_idx - start_sample + 1;
            std::cout << "\n--- Rank 0: Sample " << local_idx << "/" << local_num_samples
                      << " (Global: " << (sample_idx + 1) << "/" << params.num_samples << ") ---\n";
        }
        
        // Seed RNG deterministically based on sample index (not rank) for reproducibility
        std::mt19937 sample_gen(params.random_seed + sample_idx * 12345);
        
        // Generate random state |r⟩ with i.i.d. complex Gaussian components
        // (canonical Hutchinson trace-estimator distribution).
        ComplexVector r_state = generateGaussianRandomVector(static_cast<int>(N), sample_gen);
        
        // Step 1: Build Lanczos from |r⟩ to get approximate eigenstates
        std::vector<double> alpha_H, beta_H;
        std::vector<ComplexVector> lanczos_vectors;
        
        uint64_t H_iterations = build_lanczos_tridiagonal_with_basis(
            H, r_state, N, params.krylov_dim, params.tolerance,
            params.full_reorthogonalization, params.reorth_frequency,
            alpha_H, beta_H, &lanczos_vectors
        );
        
        uint64_t m_H = alpha_H.size();
        if (mpi_rank == 0 && verbose) {
            std::cout << "  Hamiltonian Lanczos: " << m_H << " iterations\n";
        }
        
        if (m_H == 0) {
            std::cerr << "  Warning: Lanczos failed, skipping sample\n";
            continue;
        }
        
        // Diagonalize tridiagonal to get Ritz values and vectors
        std::vector<double> ritz_values;
        std::vector<double> dummy_weights;
        std::vector<double> evecs;  // Row-major: evecs[i*m_H + j] = V[i,j]
        diagonalize_tridiagonal_ritz(alpha_H, beta_H, ritz_values, dummy_weights, &evecs);
        
        if (ritz_values.empty()) {
            std::cerr << "  Warning: Diagonalization failed, skipping sample\n";
            continue;
        }
        
        // Compute |c_i|² = |⟨ψ_i|r⟩|² = V[i,0]² (first Lanczos vector is |r⟩)
        std::vector<double> c_sq(m_H);
        for (uint64_t i = 0; i < m_H; i++) {
            c_sq[i] = evecs[i * m_H + 0] * evecs[i * m_H + 0];
        }
        
        // Find minimum energy for numerical stability
        double E_min = *std::min_element(ritz_values.begin(), ritz_values.end());
        
        if (mpi_rank == 0 && verbose && m_H > 0 && !ritz_values.empty()) {
            std::cout << "  Ritz values range: [" << *std::min_element(ritz_values.begin(), ritz_values.end())
                      << ", " << *std::max_element(ritz_values.begin(), ritz_values.end()) << "]\n";
        }
        
        // ============================================================
        // OPTIMIZATION: Precompute spectral functions for Ritz states
        // The Lanczos expansion and continued fraction are temperature-
        // independent, so we compute S_i(ω) once and reuse across all T
        // ============================================================
        
        // Step 2a: Determine which Ritz states are significant for ANY temperature
        // Use the highest temperature (smallest beta) for most inclusive threshold
        double T_max_local = *std::max_element(temperatures.begin(), temperatures.end());
        double beta_min = 1.0 / T_max_local;
        
        // Compute thermal weights at highest T to find potentially significant states
        std::vector<double> max_weights(m_H);
        double Z_max = 0.0;
        for (uint64_t i = 0; i < m_H; i++) {
            double boltzmann = std::exp(-beta_min * (ritz_values[i] - E_min));
            max_weights[i] = c_sq[i] * boltzmann;
            Z_max += max_weights[i];
        }
        
        // Identify significant Ritz states (union across all temperatures)
        double weight_threshold = 1e-10 * Z_max;  // Use looser threshold to catch all
        std::vector<uint64_t> significant_states;
        significant_states.reserve(max_ritz_states);
        
        for (uint64_t i = 0; i < std::min(m_H, max_ritz_states); i++) {
            if (max_weights[i] >= weight_threshold || c_sq[i] > 1e-12) {
                significant_states.push_back(i);
            }
        }
        
        if (mpi_rank == 0 && verbose) {
            std::cout << "  Identified " << significant_states.size() << " potentially significant Ritz states\n";
        }
        
        // Step 2b: Precompute S_i(ω) for each significant Ritz state
        // This is the expensive part - Lanczos + continued fraction per state
        std::vector<std::vector<double>> precomputed_S_i(significant_states.size());
        std::vector<std::vector<double>> precomputed_S_i_imag(significant_states.size());
        std::vector<double> precomputed_energies(significant_states.size());
        std::vector<double> precomputed_c_sq(significant_states.size());
        std::vector<bool> state_valid(significant_states.size(), false);
        
        // Process Ritz states sequentially. The inner operator::apply() and
        // build_lanczos_tridiagonal() use their own OpenMP parallel regions
        // which parallelize over the Hilbert space dimension. Using an outer
        // OpenMP parallel-for here would cause nested parallelism leading to
        // thread explosion, heap corruption, and segfaults.
        // MPI sample distribution handles the coarse-grained parallelism.
        for (size_t idx = 0; idx < significant_states.size(); idx++) {
            uint64_t i = significant_states[idx];
            
            // Working vectors
            ComplexVector psi_local(N, Complex(0.0, 0.0));
            ComplexVector phi2_local(N);
            
            // Construct approximate eigenstate |ψ_i⟩ = Σ_j V[i,j] |v_j⟩
            for (uint64_t j = 0; j < m_H; j++) {
                double coeff = evecs[i * m_H + j];
                Complex coeff_c(coeff, 0.0);
                cblas_zaxpy(N, &coeff_c, lanczos_vectors[j].data(), 1, psi_local.data(), 1);
            }
            
            // Normalize
            double psi_norm = cblas_dznrm2(N, psi_local.data(), 1);
            if (psi_norm < 1e-14) continue;
            Complex psi_scale(1.0/psi_norm, 0.0);
            cblas_zscal(N, &psi_scale, psi_local.data(), 1);
            
            // Apply operator O2: |φ₂⟩ = O₂|ψ_i⟩
            O2(psi_local.data(), phi2_local.data(), N);
            
            double phi2_norm = cblas_dznrm2(N, phi2_local.data(), 1);
            
            if (phi2_norm < 1e-14) continue;
            
            // Normalize for Lanczos
            Complex phi2_scale(1.0/phi2_norm, 0.0);
            cblas_zscal(N, &phi2_scale, phi2_local.data(), 1);
            
            // Build Lanczos from |φ₂⟩ with basis storage (needed for cross-correlation overlaps)
            std::vector<double> alpha_S, beta_S;
            std::vector<ComplexVector> lanczos_basis_S;
            build_lanczos_tridiagonal_with_basis(
                H, phi2_local, N, params.krylov_dim, params.tolerance,
                params.full_reorthogonalization, params.reorth_frequency,
                alpha_S, beta_S, &lanczos_basis_S
            );
            
            if (alpha_S.empty()) continue;
            
            uint64_t m_S = alpha_S.size();
            
            // Shift energies by E_gs
            for (size_t k = 0; k < m_S; k++) {
                alpha_S[k] -= E_gs;
            }
            
            // Apply O1 to the eigenstate: |φ₁⟩ = O₁|ψ_i⟩
            ComplexVector phi1_local(N);
            O1(psi_local.data(), phi1_local.data(), N);
            
            // Compute overlaps: p_j = ⟨φ₁|v_j⟩ where v_j are Lanczos basis vectors from |φ₂⟩
            std::vector<Complex> phi1_overlaps(m_S);
            for (uint64_t j = 0; j < m_S; j++) {
                Complex overlap;
                cblas_zdotc_sub(N, phi1_local.data(), 1, lanczos_basis_S[j].data(), 1, &overlap);
                phi1_overlaps[j] = overlap;
            }
            
            // Free the Lanczos basis vectors to save memory
            lanczos_basis_S.clear();
            lanczos_basis_S.shrink_to_fit();
            
            // Diagonalize the tridiagonal matrix to get Ritz eigenvalues and eigenvectors
            std::vector<double> ritz_vals_S, dummy_wts_S;
            std::vector<double> evecs_S;  // Row-major: evecs_S[k*m_S + j] = V[k,j]
            diagonalize_tridiagonal_ritz(alpha_S, beta_S, ritz_vals_S, dummy_wts_S, &evecs_S);
            
            if (ritz_vals_S.empty()) continue;
            
            // Compute cross-spectral function using Lehmann representation
            // S_cross(ω) = Σ_k w_k × L(ω - λ_k)
            // where w_k = (Σ_j V_{jk} ⟨φ₁|v_j⟩) × V_{0k} × ‖φ₂‖
            // and L(x) = (η/π) / (x² + η²) is the Lorentzian broadening kernel.
            // For cross-correlations, w_k is complex → S(ω) is complex.
            const uint64_t n_ritz = ritz_vals_S.size();
            std::vector<double> S_i(num_omega_bins, 0.0);
            std::vector<double> S_i_imag(num_omega_bins, 0.0);

            // Pre-compute the per-Ritz weights once (cheap, O(n_ritz × m_S))
            // and stage them as packed real/imag/E arrays so the omega loop
            // below is a clean memory-streamed reduction we can OpenMP-split.
            std::vector<double> w_re(n_ritz), w_im(n_ritz), E_arr(n_ritz);
            for (uint64_t k = 0; k < n_ritz; k++) {
                Complex overlap_O1_nk(0.0, 0.0);
                for (uint64_t j = 0; j < m_S; j++) {
                    overlap_O1_nk += Complex(evecs_S[k * m_S + j], 0.0) * phi1_overlaps[j];
                }
                const double V_0k = evecs_S[k * m_S + 0];
                const Complex w_k = overlap_O1_nk * Complex(V_0k * phi2_norm, 0.0);
                w_re[k] = w_k.real();
                w_im[k] = w_k.imag();
                E_arr[k] = ritz_vals_S[k];
            }

            // Lorentzian broadening of the Lehmann weights:
            //   S(ω) += w_k × (η/π) / ((ω - E_k)² + η²)
            // Parallelize over ω bins -- each is an independent O(n_ritz)
            // reduction. This is the hottest inner loop of the multi-T path.
            const double eta = params.broadening;
            const double eta_sq = eta * eta;
            const double inv_pi_eta = eta / M_PI;
            #pragma omp parallel for schedule(static)
            for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                const double omega = frequencies[iw];
                double s_re = 0.0, s_im = 0.0;
                for (uint64_t k = 0; k < n_ritz; k++) {
                    const double delta = omega - E_arr[k];
                    const double lorentzian = inv_pi_eta / (delta * delta + eta_sq);
                    s_re += w_re[k] * lorentzian;
                    s_im += w_im[k] * lorentzian;
                }
                S_i[iw] = s_re;
                S_i_imag[iw] = s_im;
            }
            
            // Store precomputed results
            precomputed_S_i[idx] = std::move(S_i);
            precomputed_S_i_imag[idx] = std::move(S_i_imag);
            precomputed_energies[idx] = ritz_values[i];
            precomputed_c_sq[idx] = c_sq[i];
            state_valid[idx] = true;
        }
        
        // Count valid states
        uint64_t n_valid = 0;
        for (size_t idx = 0; idx < significant_states.size(); idx++) {
            if (state_valid[idx]) n_valid++;
        }
        if (mpi_rank == 0 && verbose) {
            std::cout << "  Precomputed spectral functions for " << n_valid << " Ritz states\n";
        }
        
        // Step 3: For each temperature, apply thermal weights to precomputed spectra
        // This is now O(num_temps × num_states × num_omega) - no Lanczos!
        for (double T : temperatures) {
            double beta = 1.0 / T;
            
            // Compute partition function contribution for this sample
            double Z_sample = 0.0;
            for (size_t idx = 0; idx < significant_states.size(); idx++) {
                if (!state_valid[idx]) continue;
                double boltzmann = std::exp(-beta * (precomputed_energies[idx] - E_min));
                Z_sample += precomputed_c_sq[idx] * boltzmann;
            }
            
            accumulated_Z[T] += Z_sample;
            
            // Accumulate weighted spectral contributions (both real and imaginary)
            std::vector<double> sample_spectral(num_omega_bins, 0.0);
            std::vector<double> sample_spectral_imag(num_omega_bins, 0.0);
            
            for (size_t idx = 0; idx < significant_states.size(); idx++) {
                if (!state_valid[idx]) continue;
                
                double boltzmann = std::exp(-beta * (precomputed_energies[idx] - E_min));
                double thermal_weight = precomputed_c_sq[idx] * boltzmann;
                
                // Skip if negligible for this temperature
                if (thermal_weight < 1e-14 * Z_sample) continue;
                
                // Add contribution (vectorized)
                const std::vector<double>& S_i = precomputed_S_i[idx];
                const std::vector<double>& S_i_im = precomputed_S_i_imag[idx];
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    double contrib = thermal_weight * S_i[iw];
                    double contrib_im = thermal_weight * S_i_im[iw];
                    sample_spectral[iw] += contrib;
                    sample_spectral_imag[iw] += contrib_im;
                    accumulated_spectral[T][iw] += contrib;
                    accumulated_spectral_imag[T][iw] += contrib_im;
                }
            }
            
            // Store sample contribution for error estimation
            if (Z_sample > 1e-300) {
                std::vector<double> normalized_sample(num_omega_bins);
                std::vector<double> normalized_sample_imag(num_omega_bins);
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    normalized_sample[iw] = sample_spectral[iw] / Z_sample;
                    normalized_sample_imag[iw] = sample_spectral_imag[iw] / Z_sample;
                }
                per_sample_spectral[T].push_back(normalized_sample);
                per_sample_spectral_imag[T].push_back(normalized_sample_imag);
            }
        }
        
        if (mpi_rank == 0 && verbose) {
            std::cout << "  Applied thermal weights for " << temperatures.size() << " temperatures\n";
        }
    }
    
    // Report timing for this rank
#ifdef WITH_MPI
    double elapsed_time = MPI_Wtime() - start_time;
#else
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
#endif
    
    if (mpi_rank == 0 && verbose) {
        std::cout << "\nCompleted " << local_num_samples << " samples in "
                  << elapsed_time << " seconds (" << (elapsed_time / local_num_samples)
                  << " s/sample)\n";
    }
    
#ifdef WITH_MPI
    // MPI Reduce: gather accumulated results from all ranks
    MPI_Barrier(comm);
    
    if (mpi_rank == 0 && verbose) {
        std::cout << "\n--- Gathering results from all MPI ranks ---\n";
        std::cout << "All ranks have completed their sample processing.\n";
    }
    
    // Reduce accumulated_spectral, accumulated_spectral_imag, and accumulated_Z across all ranks
    for (double T : temperatures) {
        std::vector<double> global_spectral(num_omega_bins, 0.0);
        std::vector<double> global_spectral_imag(num_omega_bins, 0.0);
        double global_Z = 0.0;
        
        MPI_Reduce(accumulated_spectral[T].data(), global_spectral.data(), 
                   num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(accumulated_spectral_imag[T].data(), global_spectral_imag.data(), 
                   num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(&accumulated_Z[T], &global_Z, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        
        // Only rank 0 needs the final values
        if (mpi_rank == 0) {
            accumulated_spectral[T] = global_spectral;
            accumulated_spectral_imag[T] = global_spectral_imag;
            accumulated_Z[T] = global_Z;
        }
    }
    
    // Gather total sample count for error estimation
    uint64_t global_total_samples = 0;
    MPI_Reduce(&local_num_samples, &global_total_samples, 1, MPI_UINT64_T, MPI_SUM, 0, comm);
#else
    uint64_t global_total_samples = local_num_samples;
#endif
    
    // Compute final results: S(ω) = N × (Σ accumulated_spectral) / (Σ accumulated_Z)
    if (mpi_rank == 0 && verbose) {
        std::cout << "\n--- Computing final results ---\n";
    }
    
    std::map<double, DynamicalResponseResults> results_map;
    
    for (double T : temperatures) {
        DynamicalResponseResults results;
        results.frequencies = frequencies;
        results.omega_min = omega_min;
        results.omega_max = omega_max;
        results.total_samples = (mpi_rank == 0) ? global_total_samples : local_num_samples;
        
        results.spectral_function.resize(num_omega_bins, 0.0);
        results.spectral_function_imag.resize(num_omega_bins, 0.0);
        results.spectral_error.resize(num_omega_bins, 0.0);
        results.spectral_error_imag.resize(num_omega_bins, 0.0);
        
        double Z_total = accumulated_Z[T];
        if (mpi_rank == 0 && Z_total < 1e-300) {
            std::cerr << "  Warning: Z ≈ 0 for T = " << T << std::endl;
            results_map[T] = results;
            continue;
        }
        
        // Only rank 0 computes final results
        if (mpi_rank != 0) {
            results_map[T] = results;
            continue;
        }
        
        // Compute spectral function: S(ω) = accumulated_spectral / Z_total
        // Note: The trace sampling identity Tr[A] = N × E_r[⟨r|A|r⟩] means we should
        // multiply by N, but in FTLM the ratio of accumulated sums automatically gives
        // the correct thermal average without an additional factor.
        // 
        // accumulated_spectral = Σ_r Σ_i e^{-βε_i} |c_i|² S_i(ω)
        // accumulated_Z = Σ_r Σ_i e^{-βε_i} |c_i|²
        // 
        // The ratio gives: S(ω,T) = ⟨O† δ(ω-H+E₀) O⟩_β 
        // which is the thermal expectation value as desired.
        for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
            results.spectral_function[iw] = accumulated_spectral[T][iw] / Z_total;
            results.spectral_function_imag[iw] = accumulated_spectral_imag[T][iw] / Z_total;
        }
        
        // Note: Error estimation with MPI requires gathering per-sample data from all ranks
        // For now, skip detailed error estimation in MPI mode (use simpler estimate)
        uint64_t n_samples = per_sample_spectral[T].size();
        if (n_samples > 1 && mpi_size == 1) {
            // Compute mean of per-sample normalized spectra (serial only)
            std::vector<double> mean(num_omega_bins, 0.0);
            std::vector<double> mean_imag(num_omega_bins, 0.0);
            for (uint64_t s = 0; s < n_samples; s++) {
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    mean[iw] += per_sample_spectral[T][s][iw];
                    mean_imag[iw] += per_sample_spectral_imag[T][s][iw];
                }
            }
            for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                mean[iw] /= n_samples;
                mean_imag[iw] /= n_samples;
            }
            
            // Compute variance and standard error
            for (uint64_t s = 0; s < n_samples; s++) {
                for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                    double diff = per_sample_spectral[T][s][iw] - mean[iw];
                    results.spectral_error[iw] += diff * diff;
                    double diff_im = per_sample_spectral_imag[T][s][iw] - mean_imag[iw];
                    results.spectral_error_imag[iw] += diff_im * diff_im;
                }
            }
            
            double norm = std::sqrt(static_cast<double>(n_samples * (n_samples - 1)));
            for (uint64_t iw = 0; iw < num_omega_bins; iw++) {
                results.spectral_error[iw] = std::sqrt(results.spectral_error[iw]) / norm;
                results.spectral_error_imag[iw] = std::sqrt(results.spectral_error_imag[iw]) / norm;
            }
        }
        
        if (verbose) {
            std::cout << "  T = " << T << ": " << global_total_samples << " samples, Z = " << Z_total << std::endl;
        }
        results_map[T] = results;
    }

    if (mpi_rank == 0 && verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "FTLM Spectral Function Complete\n";
        std::cout << "==========================================" << std::endl;
    }

    return results_map;
}

// ============================================================================
// MULTI-OPERATOR FTLM: shares the per-sample H-Lanczos chain (and the cached
// Ritz eigenstates |psi_i>) across all (O1[p], O2[p]) pairs. The per-pair
// inner work (O2|psi_i>, sub-Lanczos, Lehmann weights) is unchanged. This is
// item #7 from the audit: only the operator-independent setup is hoisted.
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

// ============================================================================
// Public dispatch wrappers for the two FTLM dynamical-correlation kernels.
// Audit #4: the heavy bodies above are MPI-Comm-aware (`comm` parameter);
// these thin wrappers preserve the legacy MPI_COMM_WORLD-only API and
// expose a parallel `_comm` API in `ed::dssf` for distributed-DSSF
// orchestration via MPI_Comm_split.
// ============================================================================
std::map<double, DynamicalResponseResults>
compute_dynamical_correlation_multi_sample_multi_temperature(
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
) {
    return compute_dynamical_correlation_multi_sample_multi_temperature_impl(
        H, O1, O2, N, params, omega_min, omega_max, num_omega_bins,
        temperatures, energy_shift, output_dir
#ifdef WITH_MPI
        , MPI_COMM_WORLD
#endif
    );
}

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
