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
 * @brief Main FTLM driver function (explicit temperatures overload).
 *
 * The temperature grid is supplied verbatim by the caller; this preserves
 * whatever axis (linear / logarithmic / arbitrary) was requested so the
 * returned curves are index-aligned with it. The (temp_min, temp_max,
 * num_temp_bins) overload below forwards a log-spaced grid for backwards
 * compatibility with the legacy CLI / pybind call sites.
 */
FTLMResults finite_temperature_lanczos(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    const FTLMParameters& params,
    const std::vector<double>& temperatures,
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

    // 1/T below diverges at T=0; reject empty / non-positive grids.
    if (temperatures.empty()) {
        throw std::invalid_argument(
            "finite_temperature_lanczos: temperatures grid must be "
            "non-empty.");
    }
    for (double T : temperatures) {
        if (!(T > 0.0)) {
            throw std::invalid_argument(
                "finite_temperature_lanczos: every temperature must be "
                "> 0 (got T=" + std::to_string(T) + ").");
        }
    }
    const uint64_t num_temp_bins = temperatures.size();

    if (verbose) {
        std::cout << "\n==========================================\n";
        std::cout << "Finite Temperature Lanczos Method (FTLM)\n";
        std::cout << "==========================================\n";
        std::cout << "Hilbert space dimension: " << N << std::endl;
        std::cout << "Krylov dimension: " << params.krylov_dim << std::endl;
        std::cout << "Number of samples: " << params.num_samples << std::endl;
        std::cout << "Temperature range: [" << temperatures.front() << ", "
                  << temperatures.back() << "]" << std::endl;
        std::cout << "Temperature bins: " << num_temp_bins << std::endl;
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
 * @brief (temp_min, temp_max, num_temp_bins) overload -- builds a
 *        logarithmically-spaced temperature grid and forwards to the
 *        explicit-temperatures driver above. Kept verbatim for the
 *        existing CLI / pybind call sites that relied on the legacy
 *        log-spacing contract.
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
    // log() / 1/T diverge at T=0; reject non-positive temperatures
    // before constructing the grid.
    if (!(temp_min > 0.0) || !(temp_max > 0.0)) {
        throw std::invalid_argument(
            "finite_temperature_lanczos: temp_min and temp_max must both "
            "be > 0 (got temp_min=" + std::to_string(temp_min) +
            ", temp_max=" + std::to_string(temp_max) + ").");
    }

    // Generate temperature grid (logarithmic spacing).
    std::vector<double> temperatures(num_temp_bins);
    const double log_tmin = std::log(temp_min);
    const double log_tmax = std::log(temp_max);
    const double log_step =
        (log_tmax - log_tmin) / std::max(uint64_t(1), num_temp_bins - 1);
    for (uint64_t i = 0; i < num_temp_bins; i++) {
        temperatures[i] = std::exp(log_tmin + i * log_step);
    }

    return finite_temperature_lanczos(
        std::move(H), N, params, temperatures, output_dir);
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
