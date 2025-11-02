#include <iostream>
#include <chrono>
#include <iomanip>
#include "ed_config.h"
#include "ed_config_adapter.h"
#include "ed_wrapper.h"
#include "construct_ham.h"
#include "../cpu_solvers/ftlm.h"
#include "../cpu_solvers/ltlm.h"

/**
 * @file ed_main.cpp
 * @brief Elegant main entry point for exact diagonalization
 * 
 * This is a complete rewrite of ed_run.cpp with:
 * - Clean configuration management
 * - Separated concerns
 * - No massive if-else chains
 * - Support for config files
 */

// ============================================================================
// WORKFLOW FUNCTIONS
// ============================================================================

/**
 * @brief Run standard diagonalization workflow
 */
EDResults run_standard_workflow(const EDConfig& config) {
    std::cout << "\n==========================================\n";
    std::cout << "Standard Exact Diagonalization\n";
    std::cout << "==========================================\n";
    
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    safe_system_call("mkdir -p " + params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    // Check if fixed Sz mode is enabled
    if (config.system.use_fixed_sz) {
        uint64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        std::string interaction_file = config.system.hamiltonian_dir + "/" + config.system.interaction_file;
        std::string single_site_file = config.system.hamiltonian_dir + "/" + config.system.single_site_file;
        
        results = exact_diagonalization_fixed_sz(
            interaction_file,
            single_site_file,
            config.system.num_sites,
            config.system.spin_length,
            n_up,
            config.method,
            params
        );
    } else {
        results = exact_diagonalization_from_directory(
            config.system.hamiltonian_dir,
            config.method,
            params,
            HamiltonianFileFormat::STANDARD
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Completed in " << duration / 1000.0 << " seconds\n";
    
    // Save eigenvalues
    std::ofstream file(params.output_dir + "/eigenvalues.txt");
    if (file.is_open()) {
        file << std::setprecision(16);
        for (const auto& val : results.eigenvalues) {
            file << val << "\n";
        }
        std::cout << "Saved " << results.eigenvalues.size() << " eigenvalues\n";
    }
    
    return results;
}

/**
 * @brief Run symmetrized diagonalization workflow
 */
EDResults run_symmetrized_workflow(const EDConfig& config) {
    std::cout << "\n==========================================\n";
    std::cout << "Symmetrized Exact Diagonalization\n";
    std::cout << "==========================================\n";
    
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    safe_system_call("mkdir -p " + params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto results = exact_diagonalization_from_directory_symmetrized(
        config.system.hamiltonian_dir,
        config.method,
        params,
        HamiltonianFileFormat::STANDARD
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Completed in " << duration / 1000.0 << " seconds\n";
    
    // Save eigenvalues
    std::ofstream file(params.output_dir + "/eigenvalues.txt");
    if (file.is_open()) {
        file << std::setprecision(16);
        for (const auto& val : results.eigenvalues) {
            file << val << "\n";
        }
        std::cout << "Saved " << results.eigenvalues.size() << " eigenvalues\n";
    }
    
    return results;
}

/**
 * @brief Compute thermodynamics from eigenvalue spectrum
 */
void compute_thermodynamics(const std::vector<double>& eigenvalues, const EDConfig& config) {
    if (eigenvalues.empty()) return;
    
    std::cout << "\n==========================================\n";
    std::cout << "Computing Thermodynamics\n";
    std::cout << "==========================================\n";
    
    auto thermo_data = calculate_thermodynamics_from_spectrum(
        eigenvalues,
        config.thermal.temp_min,
        config.thermal.temp_max,
        config.thermal.num_temp_bins
    );
    
    // Save results
    std::string thermo_dir = config.workflow.output_dir + "/thermo";
    safe_system_call("mkdir -p " + thermo_dir);
    
    std::ofstream file(thermo_dir + "/thermo_data.txt");
    if (file.is_open()) {
        file << "# Temperature  Energy  Specific_Heat  Entropy  Free_Energy\n";
        for (size_t i = 0; i < thermo_data.temperatures.size(); i++) {
            file << thermo_data.temperatures[i] << " "
                 << thermo_data.energy[i] << " "
                 << thermo_data.specific_heat[i] << " "
                 << thermo_data.entropy[i] << " "
                 << thermo_data.free_energy[i] << "\n";
        }
        std::cout << "Saved thermodynamic data\n";
    }
}

/**
 * @brief Compute dynamical response (spectral functions)
 */
void compute_dynamical_response_workflow(const std::vector<double>& eigenvalues,
                                        const EDConfig& config) {
    if (eigenvalues.empty()) {
        std::cerr << "Error: Dynamical response requires eigenvalues\n";
        return;
    }
    
    // Note: Currently only thermal mode is supported in the integrated pipeline
    // For ground state dynamical response with eigenvectors, use the standalone
    // example in examples/dynamical_response_example.cpp
    if (!config.dynamical.thermal_average) {
        std::cerr << "Note: Only thermal mode (--dyn-thermal) is currently supported in the integrated pipeline.\n";
        std::cerr << "Setting thermal_average mode automatically.\n";
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Computing Dynamical Response\n";
    std::cout << "==========================================\n";
    
    // Load operator
    if (config.dynamical.operator_file.empty()) {
        std::cerr << "Error: --dyn-operator=<file> is required for dynamical response\n";
        return;
    }
    
    std::string op_path = config.system.hamiltonian_dir + "/" + config.dynamical.operator_file;
    Operator op(config.system.num_sites, config.system.spin_length);
    op.loadFromInterAllFile(op_path);
    
    // Setup parameters
    DynamicalResponseParameters params;
    params.num_samples = config.dynamical.num_random_states;
    params.krylov_dim = config.dynamical.krylov_dim;
    params.broadening = config.dynamical.broadening;
    params.random_seed = config.dynamical.random_seed;
    
    // Prepare Hamiltonian
    Operator ham(config.system.num_sites, config.system.spin_length);
    std::string interaction_file = config.system.hamiltonian_dir + "/" + config.system.interaction_file;
    std::string single_site_file = config.system.hamiltonian_dir + "/" + config.system.single_site_file;
    ham.loadFromInterAllFile(interaction_file);
    ham.loadFromFile(single_site_file);
    
    // Build sparse matrices
    ham.buildSparseMatrix();
    op.buildSparseMatrix();
    
    // Hilbert space dimension
    uint64_t N = 1ULL << config.system.num_sites;
    
    // Create function wrappers for matrix-vector products
    auto H_func = [&ham](const Complex* in, Complex* out, uint64_t dim) {
        ham.apply(in, out, dim);
    };
    
    auto O_func = [&op](const Complex* in, Complex* out, uint64_t dim) {
        op.apply(in, out, dim);
    };
    
    // Compute response (thermal mode) - default to two-operator correlation ⟨O†O⟩
    std::string output_subdir = config.workflow.output_dir + "/dynamical_response";
    safe_system_call("mkdir -p " + output_subdir);
    
    std::cout << "Computing thermal-averaged dynamical response...\n";
    std::cout << "  Random states: " << params.num_samples << "\n";
    std::cout << "  Krylov dimension: " << params.krylov_dim << "\n";
    std::cout << "  Temperature range: [" << config.dynamical.temp_min << ", " << config.dynamical.temp_max << "]\n";
    std::cout << "  Temperature bins: " << config.dynamical.num_temp_bins << "\n";
    
    // Generate temperature grid
    std::vector<double> temperatures(config.dynamical.num_temp_bins);
    if (config.dynamical.num_temp_bins == 1) {
        temperatures[0] = config.dynamical.temp_min;
    } else {
        double log_tmin = std::log(config.dynamical.temp_min);
        double log_tmax = std::log(config.dynamical.temp_max);
        double log_step = (log_tmax - log_tmin) / (config.dynamical.num_temp_bins - 1);
        for (int i = 0; i < config.dynamical.num_temp_bins; i++) {
            temperatures[i] = std::exp(log_tmin + i * log_step);
        }
    }
    
    // Compute for each temperature
    for (int t_idx = 0; t_idx < config.dynamical.num_temp_bins; t_idx++) {
        double temperature = temperatures[t_idx];
        
        std::cout << "\n--- Temperature " << (t_idx + 1) << " / " << config.dynamical.num_temp_bins 
                  << ": T = " << temperature << " ---\n";
    
        DynamicalResponseResults results;
    
        if (!config.dynamical.operator2_file.empty()) {
            // Two different operators: ⟨O₁†(t)O₂⟩
            std::cout << "Computing two-operator dynamical correlation ⟨O₁†(t)O₂⟩...\n";
            std::string op2_path = config.system.hamiltonian_dir + "/" + config.dynamical.operator2_file;
            Operator op2(config.system.num_sites, config.system.spin_length);
            op2.loadFromInterAllFile(op2_path);
            op2.buildSparseMatrix();
            
            auto O2_func = [&op2](const Complex* in, Complex* out, uint64_t dim) {
                op2.apply(in, out, dim);
            };
            
            results = compute_dynamical_correlation(
                H_func, O_func, O2_func, N, params,
                config.dynamical.omega_min,
                config.dynamical.omega_max,
                config.dynamical.num_omega_points,
                temperature,  // Use current temperature from loop
                output_subdir
            );
        } else {
            // Same operator: ⟨O†(t)O⟩ (default auto-correlation)
            std::cout << "Computing dynamical response ⟨O†(t)O⟩...\n";
            results = compute_dynamical_response_thermal(
                H_func, O_func, N, params,
                config.dynamical.omega_min,
                config.dynamical.omega_max,
                config.dynamical.num_omega_points,
                temperature,  // Use current temperature from loop
                output_subdir
            );
        }
        
        // Save results for this temperature
        std::string output_file = output_subdir + "/" + config.dynamical.output_prefix;
        if (config.dynamical.num_temp_bins > 1) {
            // Multi-temperature: add T to filename
            output_file += "_T" + std::to_string(temperature);
        }
        output_file += ".txt";
        
        save_dynamical_response_results(results, output_file);
        
        std::cout << "Results saved to: " << output_file << "\n";
    } // end temperature loop
    
    std::cout << "\nDynamical response complete.\n";
    std::cout << "Frequency range: [" << config.dynamical.omega_min << ", " << config.dynamical.omega_max << "]\n";
    std::cout << "Number of points: " << config.dynamical.num_omega_points << "\n";
}

/**
 * @brief Compute static response (thermal expectation values)
 */
void compute_static_response_workflow(const std::vector<double>& eigenvalues,
                                     const EDConfig& config) {
    if (eigenvalues.empty()) {
        std::cerr << "Error: Static response requires eigenvalues\n";
        return;
    }
    
    std::cout << "\n==========================================\n";
    std::cout << "Computing Static Response\n";
    std::cout << "==========================================\n";
    
    // Load operator
    if (config.static_resp.operator_file.empty()) {
        std::cerr << "Error: --static-operator=<file> is required for static response\n";
        return;
    }
    
    std::string op_path = config.system.hamiltonian_dir + "/" + config.static_resp.operator_file;
    Operator op(config.system.num_sites, config.system.spin_length);
    op.loadFromInterAllFile(op_path);
    
    // Setup parameters
    StaticResponseParameters params;
    params.num_samples = config.static_resp.num_random_states;
    params.krylov_dim = config.static_resp.krylov_dim;
    params.random_seed = config.static_resp.random_seed;
    
    // Prepare Hamiltonian
    Operator ham(config.system.num_sites, config.system.spin_length);
    std::string interaction_file = config.system.hamiltonian_dir + "/" + config.system.interaction_file;
    std::string single_site_file = config.system.hamiltonian_dir + "/" + config.system.single_site_file;
    ham.loadFromInterAllFile(interaction_file);
    ham.loadFromFile(single_site_file);
    
    // Build sparse matrices
    ham.buildSparseMatrix();
    op.buildSparseMatrix();
    
    // Hilbert space dimension
    uint64_t N = 1ULL << config.system.num_sites;
    
    // Create function wrappers for matrix-vector products
    auto H_func = [&ham](const Complex* in, Complex* out, uint64_t dim) {
        ham.apply(in, out, dim);
    };
    
    auto O_func = [&op](const Complex* in, Complex* out, uint64_t dim) {
        op.apply(in, out, dim);
    };
    
    std::cout << "Random states: " << params.num_samples << "\n";
    std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
    std::cout << "Temperature range: [" << config.static_resp.temp_min << ", " << config.static_resp.temp_max << "]\n";
    
    // Compute response
    StaticResponseResults results;
    std::string output_subdir = config.workflow.output_dir + "/static_response";
    
    if (config.static_resp.single_operator_mode) {
        // Single operator expectation value: ⟨O⟩
        std::cout << "Computing thermal expectation value ⟨O⟩...\n";
        results = compute_thermal_expectation_value(
            H_func, O_func, N, params,
            config.static_resp.temp_min,
            config.static_resp.temp_max,
            config.static_resp.num_temp_points,
            output_subdir
        );
    } else if (!config.static_resp.operator2_file.empty()) {
        // Two different operators: ⟨O₁†O₂⟩
        std::cout << "Computing two-operator static response ⟨O₁†O₂⟩...\n";
        std::string op2_path = config.system.hamiltonian_dir + "/" + config.static_resp.operator2_file;
        Operator op2(config.system.num_sites, config.system.spin_length);
        op2.loadFromInterAllFile(op2_path);
        op2.buildSparseMatrix();
        
        auto O2_func = [&op2](const Complex* in, Complex* out, uint64_t dim) {
            op2.apply(in, out, dim);
        };
        
        results = compute_static_response(
            H_func, O_func, O2_func, N, params,
            config.static_resp.temp_min,
            config.static_resp.temp_max,
            config.static_resp.num_temp_points,
            output_subdir
        );
    } else {
        // Same operator: ⟨O†O⟩ (default two-point correlation)
        std::cout << "Computing static response ⟨O†O⟩...\n";
        results = compute_static_response(
            H_func, O_func, O_func, N, params,
            config.static_resp.temp_min,
            config.static_resp.temp_max,
            config.static_resp.num_temp_points,
            output_subdir
        );
    }
    
    // Save results
    safe_system_call("mkdir -p " + output_subdir);
    
    std::string output_file = output_subdir + "/" + config.static_resp.output_prefix + ".txt";
    save_static_response_results(results, output_file);
    
    std::cout << "Static response saved to: " << output_file << "\n";
}

/**
 * @brief Print eigenvalue summary
 */
void print_eigenvalue_summary(const std::vector<double>& eigenvalues, uint64_t max_show = 10) {
    std::cout << "\nEigenvalues:\n";
    for (size_t i = 0; i < eigenvalues.size() && i < max_show; i++) {
        std::cout << "  " << i << ": " << std::setprecision(12) << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > max_show) {
        std::cout << "  ... (" << eigenvalues.size() - max_show << " more)\n";
    }
}

/**
 * @brief Print help message
 */
void print_help(const char* prog_name) {
    std::cout << "Exact Diagonalization Pipeline\n";
    std::cout << "==============================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " <directory> [options]\n";
    std::cout << "  " << prog_name << " --config=<file> [options]\n\n";
    
    std::cout << "Quick Examples:\n";
    std::cout << "  # Basic ground state calculation\n";
    std::cout << "  " << prog_name << " ./data --method=LANCZOS\n\n";
    std::cout << "  # Full spectrum with thermodynamics\n";
    std::cout << "  " << prog_name << " ./data --method=FULL --thermo\n\n";
    std::cout << "  # Symmetrized calculation\n";
    std::cout << "  " << prog_name << " ./data --symmetrized --eigenvalues=10\n\n";
    std::cout << "  # Use config file\n";
    std::cout << "  " << prog_name << " --config=ed_config.txt\n\n";
    
    std::cout << "General Options:\n";
    std::cout << "  --config=<file>         Load configuration from file\n";
    std::cout << "  --method=<name>         Diagonalization method (LANCZOS, FULL, mTPQ, etc.)\n";
    std::cout << "  --method-info=<name>    Show detailed parameters for specific method\n";
    std::cout << "  --num_sites=<n>         Number of sites (auto-detected if omitted)\n";
    std::cout << "  --output=<dir>          Output directory\n\n";
    
    std::cout << "Diagonalization Options:\n";
    std::cout << "  --eigenvalues=<n>       Number of eigenvalues (or FULL for complete spectrum)\n";
    std::cout << "  --eigenvectors          Compute eigenvectors\n";
    std::cout << "  --tolerance=<tol>       Convergence tolerance (default: 1e-10)\n";
    std::cout << "  --iterations=<n>        Maximum iterations\n\n";
    
    std::cout << "Workflow Options:\n";
    std::cout << "  --standard              Run standard diagonalization\n";
    std::cout << "  --symmetrized           Run symmetrized diagonalization (exploits symmetries)\n";
    std::cout << "  --thermo                Compute thermodynamic properties\n";
    std::cout << "  --dynamical-response    Compute dynamical response (spectral functions)\n";
    std::cout << "  --static-response       Compute static response (thermal expectation values)\n";
    std::cout << "  --calc_observables      Calculate custom observables\n";
    std::cout << "  --measure_spin          Measure spin expectations\n\n";
    
    std::cout << "Thermal Options (for mTPQ/cTPQ/FULL):\n";
    std::cout << "  --samples=<n>           Number of TPQ samples\n";
    std::cout << "  --temp_min=<T>          Minimum temperature\n";
    std::cout << "  --temp_max=<T>          Maximum temperature\n";
    std::cout << "  --temp_bins=<n>         Number of temperature bins\n\n";
    
    std::cout << "Dynamical Response Options:\n";
    std::cout << "  --dyn-thermal           Use thermal averaging (multiple random states)\n";
    std::cout << "  --dyn-samples=<n>       Number of random states (default: 20)\n";
    std::cout << "  --dyn-krylov=<n>        Krylov dimension per sample (default: 100)\n";
    std::cout << "  --dyn-omega-min=<ω>     Minimum frequency (default: -10)\n";
    std::cout << "  --dyn-omega-max=<ω>     Maximum frequency (default: 10)\n";
    std::cout << "  --dyn-omega-points=<n>  Number of frequency points (default: 1000)\n";
    std::cout << "  --dyn-broadening=<η>    Lorentzian broadening (default: 0.1)\n";
    std::cout << "  --dyn-correlation       Compute two-operator dynamical correlation\n";
    std::cout << "  --dyn-operator=<file>   Operator file to probe\n";
    std::cout << "  --dyn-operator2=<file>  Second operator for correlation\n";
    std::cout << "  --dyn-output=<prefix>   Output file prefix (default: dynamical_response)\n";
    std::cout << "  --dyn-seed=<n>          Random seed (0 = auto)\n\n";
    
    std::cout << "Static Response Options:\n";
    std::cout << "  --static-samples=<n>    Number of random states (default: 20)\n";
    std::cout << "  --static-krylov=<n>     Krylov dimension per sample (default: 100)\n";
    std::cout << "  --static-temp-min=<T>   Minimum temperature (default: 0.01)\n";
    std::cout << "  --static-temp-max=<T>   Maximum temperature (default: 10.0)\n";
    std::cout << "  --static-temp-points=<n> Number of temperature points (default: 100)\n";
    std::cout << "  --static-no-susceptibility  Don't compute susceptibility\n";
    std::cout << "  --static-correlation    Compute two-operator correlation\n";
    std::cout << "  --static-expectation    Compute single-operator <O> (implies --static-response)\n";
    std::cout << "  --static-operator=<file>    Operator file to probe\n";
    std::cout << "  --static-operator2=<file>   Second operator for correlation\n";
    std::cout << "  --static-output=<prefix>    Output file prefix (default: static_response)\n";
    std::cout << "  --static-seed=<n>       Random seed (0 = auto)\n\n";
    
    std::cout << "Available Methods:\n";
    std::cout << "  Lanczos Variants:\n";
    std::cout << "    LANCZOS                Standard Lanczos (default)\n";
    std::cout << "    LANCZOS_SELECTIVE      Lanczos with selective reorthogonalization\n";
    std::cout << "    LANCZOS_NO_ORTHO       Lanczos without reorthogonalization (fastest, least stable)\n";
    std::cout << "    BLOCK_LANCZOS          Block Lanczos for degenerate eigenvalues\n";
    std::cout << "    SHIFT_INVERT           Shift-invert Lanczos for interior eigenvalues\n";
    std::cout << "    SHIFT_INVERT_ROBUST    Robust shift-invert (fallback to standard)\n";
    std::cout << "    KRYLOV_SCHUR           Krylov-Schur method (restarted Lanczos)\n";
    std::cout << "\n";
    std::cout << "  Conjugate Gradient Variants:\n";
    std::cout << "    BICG                   Biconjugate gradient\n";
    std::cout << "    LOBPCG                 Locally optimal block preconditioned CG\n";
    std::cout << "\n";
    std::cout << "  Other Iterative Methods:\n";
    std::cout << "    DAVIDSON               Davidson method\n";
    std::cout << "\n";
    std::cout << "  Full Diagonalization:\n";
    std::cout << "    FULL                   Complete spectrum (exact, memory intensive)\n";
    std::cout << "    OSS                    Optimal spectrum solver (adaptive slicing)\n";
    std::cout << "\n";
    std::cout << "  Thermal Methods:\n";
    std::cout << "    mTPQ                   Microcanonical TPQ\n";
    std::cout << "    cTPQ                   Canonical TPQ\n";
    std::cout << "    mTPQ_MPI               MPI parallel mTPQ (requires MPI build)\n";
    std::cout << "    mTPQ_CUDA              GPU-accelerated mTPQ (requires CUDA build)\n";
    std::cout << "    FTLM                   Finite Temperature Lanczos Method\n";
    std::cout << "    LTLM                   Low Temperature Lanczos Method\n";
    std::cout << "    HYBRID                 Hybrid Thermal Method (LTLM+FTLM auto-switch)\n";
    std::cout << "\n";
    std::cout << "  ARPACK Methods:\n";
    std::cout << "    ARPACK_SM              ARPACK (smallest eigenvalues)\n";
    std::cout << "    ARPACK_LM              ARPACK (largest eigenvalues)\n";
    std::cout << "    ARPACK_SHIFT_INVERT    ARPACK with shift-invert\n";
    std::cout << "    ARPACK_ADVANCED        ARPACK with advanced multi-attempt strategy\n";
    std::cout << "\n";
    std::cout << "  GPU Methods (require CUDA build):\n";
    std::cout << "    LANCZOS_GPU            GPU-accelerated Lanczos\n";
    std::cout << "    LANCZOS_GPU_FIXED_SZ   GPU Lanczos for fixed Sz sector\n";
    std::cout << "    DAVIDSON_GPU           GPU-accelerated Davidson method\n";
    std::cout << "    LOBPCG_GPU             GPU-accelerated LOBPCG method\n";
    std::cout << "    mTPQ_GPU               GPU-accelerated microcanonical TPQ\n";
    std::cout << "    cTPQ_GPU               GPU-accelerated canonical TPQ\n\n";
    
    std::cout << "For detailed parameters of any method, use:\n";
    std::cout << "  " << prog_name << " --method-info=<METHOD_NAME>\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog_name << " --method-info=LANCZOS\n";
    std::cout << "  " << prog_name << " --method-info=LOBPCG\n";
    std::cout << "  " << prog_name << " --method-info=mTPQ\n";
    std::cout << "  " << prog_name << " --method-info=DAVIDSON_GPU\n\n";
    
    std::cout << "For more options, see documentation or generated config file.\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    // Check for help
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            return 0;
        }
        
        // Check for --method-info
        if (arg.find("--method-info=") == 0) {
            std::string method_name = arg.substr(14);
            auto method = ed_config::parseMethod(method_name);
            if (method.has_value()) {
                std::cout << ed_config::getMethodParameterInfo(method.value());
            } else {
                std::cerr << "Error: Unknown method '" << method_name << "'\n";
                std::cerr << "Use --help to see available methods.\n";
                return 1;
            }
            return 0;
        }
    }
    
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }
    
    // Parse configuration
    EDConfig config = EDConfig::fromCommandLine(argc, argv);

    
    // Validate
    if (!config.validate()) {
        std::cerr << "\nConfiguration validation failed. Use --help for usage.\n";
        return 1;
    }
    
    // Print configuration summary
    config.print();
    
    // Save configuration for reproducibility
    config.save(config.workflow.output_dir + "/ed_config.txt");
    
    // Create output directory
    safe_system_call("mkdir -p " + config.workflow.output_dir);
    
    // Execute workflows
    EDResults standard_results, sym_results;
    
    try {
        if (config.workflow.run_standard && !config.workflow.skip_ed) {
            standard_results = run_standard_workflow(config);
            print_eigenvalue_summary(standard_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !standard_results.eigenvalues.empty()) {
                compute_thermodynamics(standard_results.eigenvalues, config);
            }
            
            if (config.workflow.compute_dynamical_response && !standard_results.eigenvalues.empty()) {
                compute_dynamical_response_workflow(standard_results.eigenvalues, config);
            }
            
            if (config.workflow.compute_static_response && !standard_results.eigenvalues.empty()) {
                compute_static_response_workflow(standard_results.eigenvalues, config);
            }
        }
        
        if (config.workflow.run_symmetrized && !config.workflow.skip_ed) {
            sym_results = run_symmetrized_workflow(config);
            print_eigenvalue_summary(sym_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !sym_results.eigenvalues.empty()) {
                compute_thermodynamics(sym_results.eigenvalues, config);
            }
            
            if (config.workflow.compute_dynamical_response && !sym_results.eigenvalues.empty()) {
                compute_dynamical_response_workflow(sym_results.eigenvalues, config);
            }
            
            if (config.workflow.compute_static_response && !sym_results.eigenvalues.empty()) {
                compute_static_response_workflow(sym_results.eigenvalues, config);
            }
        }
        
        // Compare results if both were run
        if (config.workflow.run_standard && config.workflow.run_symmetrized) {
            std::cout << "\n==========================================\n";
            std::cout << "Comparison\n";
            std::cout << "==========================================\n";
            
            uint64_t n = std::min(standard_results.eigenvalues.size(), sym_results.eigenvalues.size());
            double max_diff = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = std::abs(standard_results.eigenvalues[i] - sym_results.eigenvalues[i]);
                max_diff = std::max(max_diff, diff);
            }
            std::cout << "Maximum difference: " << max_diff << "\n";
        }
        
        std::cout << "\n==========================================\n";
        std::cout << "Calculation Complete\n";
        std::cout << "Results saved to: " << config.workflow.output_dir << "\n";
        std::cout << "==========================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
