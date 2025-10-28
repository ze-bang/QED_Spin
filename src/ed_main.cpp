#include <iostream>
#include <chrono>
#include <iomanip>
#include "ed_config.h"
#include "ed_config_adapter.h"
#include "ed_wrapper.h"
#include "construct_ham.h"

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
    system(("mkdir -p " + params.output_dir).c_str());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    // Check if fixed Sz mode is enabled
    if (config.system.use_fixed_sz) {
        int n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
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
    system(("mkdir -p " + params.output_dir).c_str());
    
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
    system(("mkdir -p " + thermo_dir).c_str());
    
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
 * @brief Print eigenvalue summary
 */
void print_eigenvalue_summary(const std::vector<double>& eigenvalues, int max_show = 10) {
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
    std::cout << "  --calc_observables      Calculate custom observables\n";
    std::cout << "  --measure_spin          Measure spin expectations\n\n";
    
    std::cout << "Thermal Options (for mTPQ/cTPQ/FULL):\n";
    std::cout << "  --samples=<n>           Number of TPQ samples\n";
    std::cout << "  --temp_min=<T>          Minimum temperature\n";
    std::cout << "  --temp_max=<T>          Maximum temperature\n";
    std::cout << "  --temp_bins=<n>         Number of temperature bins\n\n";
    
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
    std::cout << "    FTLM                   Finite Temperature Lanczos Method (not yet implemented)\n";
    std::cout << "    LTLM                   Low Temperature Lanczos Method (not yet implemented)\n";
    std::cout << "\n";
    std::cout << "  ARPACK Methods:\n";
    std::cout << "    ARPACK_SM              ARPACK (smallest eigenvalues)\n";
    std::cout << "    ARPACK_LM              ARPACK (largest eigenvalues)\n";
    std::cout << "    ARPACK_SHIFT_INVERT    ARPACK with shift-invert\n";
    std::cout << "    ARPACK_ADVANCED        ARPACK with advanced multi-attempt strategy\n";
    std::cout << "\n";
    std::cout << "  GPU Methods (require CUDA build):\n";
    std::cout << "    LANCZOS_GPU            GPU-accelerated Lanczos\n";
    std::cout << "    LANCZOS_GPU_FIXED_SZ   GPU Lanczos for fixed Sz sector\n\n";
    
    std::cout << "For detailed parameters of any method, use:\n";
    std::cout << "  " << prog_name << " --method-info=<METHOD_NAME>\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog_name << " --method-info=LANCZOS\n";
    std::cout << "  " << prog_name << " --method-info=LOBPCG\n";
    std::cout << "  " << prog_name << " --method-info=mTPQ\n\n";
    
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
    system(("mkdir -p " + config.workflow.output_dir).c_str());
    
    // Execute workflows
    EDResults standard_results, sym_results;
    
    try {
        if (config.workflow.run_standard && !config.workflow.skip_ed) {
            standard_results = run_standard_workflow(config);
            print_eigenvalue_summary(standard_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !standard_results.eigenvalues.empty()) {
                compute_thermodynamics(standard_results.eigenvalues, config);
            }
        }
        
        if (config.workflow.run_symmetrized && !config.workflow.skip_ed) {
            sym_results = run_symmetrized_workflow(config);
            print_eigenvalue_summary(sym_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !sym_results.eigenvalues.empty()) {
                compute_thermodynamics(sym_results.eigenvalues, config);
            }
        }
        
        // Compare results if both were run
        if (config.workflow.run_standard && config.workflow.run_symmetrized) {
            std::cout << "\n==========================================\n";
            std::cout << "Comparison\n";
            std::cout << "==========================================\n";
            
            int n = std::min(standard_results.eigenvalues.size(), sym_results.eigenvalues.size());
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
