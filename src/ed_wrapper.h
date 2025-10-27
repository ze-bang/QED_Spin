#ifndef ED_WRAPPER_H
#define ED_WRAPPER_H

// ============================================================================
// INCLUDES
// ============================================================================
#include "TPQ.h"
#include "CG.h"
#include "arpack.h"
#include "lanczos.h"
#include "construct_ham.h"
#include "observables.h"
#include "ed_config.h"
#include <sys/stat.h>
#include <filesystem>
#include <algorithm>


// ============================================================================
// VECTOR OPERATIONS FOR COMPLEX VECTORS
// ============================================================================

/**
 * @brief Vector addition operator for complex vectors
 * @param a First vector
 * @param b Second vector
 * @return Element-wise sum of the two vectors
 * @throws std::invalid_argument if vectors have different sizes
 */
inline std::vector<Complex> operator+ (const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @brief Vector subtraction operator for complex vectors
 */
inline std::vector<Complex> operator- (const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

/**
 * @brief In-place addition operator for complex vectors
 */
inline std::vector<Complex> operator+= (std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
    return a;
}

/**
 * @brief In-place subtraction operator for complex vectors
 */
inline std::vector<Complex> operator-= (std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] -= b[i];
    }
    return a;
}

/**
 * @brief Scalar multiplication operator for complex vectors
 */
inline std::vector<Complex> operator* (const std::vector<Complex>& a, const Complex& b) {
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b;
    }
    return result;
}

// ============================================================================
// ENUMS AND STRUCTURES
// ============================================================================

/**
 * @brief Available diagonalization methods for exact diagonalization
 */
enum class DiagonalizationMethod {
    LANCZOS,               // Standard Lanczos algorithm
    LANCZOS_SELECTIVE,     // Lanczos with selective reorthogonalization
    LANCZOS_NO_ORTHO,      // Lanczos without reorthogonalization
    BLOCK_LANCZOS,         // Block Lanczos
    SHIFT_INVERT,          // Shift-invert Lanczos
    SHIFT_INVERT_ROBUST,   // Robust shift-invert Lanczos
    DAVIDSON,              // Davidson method
    BICG,                  // Biconjugate gradient
    LOBPCG,                // Locally optimal block preconditioned conjugate gradient
    KRYLOV_SCHUR,          // Krylov-Schur algorithm
    FULL,                  // Full diagonalization
    OSS,                   // Optimal spectrum solver
    
    // Thermal methods
    mTPQ,                  // Microcanonical Thermal Pure Quantum states
    mTPQ_MPI,              // MPI version of mTPQ
    cTPQ,                  // Canonical Thermal Pure Quantum states
    mTPQ_CUDA,             // CUDA microcanonical Thermal Pure Quantum states
    FTLM,                  // Finite Temperature Lanczos Method
    LTLM,                  // Low Temperature Lanczos Method
    
    // ARPACK methods
    ARPACK_SM,             // ARPACK smallest magnitude eigenvalues
    ARPACK_LM,             // ARPACK largest magnitude eigenvalues
    ARPACK_SHIFT_INVERT,   // ARPACK in shift-invert mode
    ARPACK_ADVANCED,       // ARPACK advanced multi-attempt strategy
    
    // GPU methods
    LANCZOS_GPU,           // GPU-accelerated Lanczos (full Hilbert space)
    LANCZOS_GPU_FIXED_SZ   // GPU-accelerated Lanczos (fixed Sz sector)
};

/**
 * @brief Structure to hold exact diagonalization results
 */
struct EDResults {
    std::vector<double> eigenvalues;
    bool eigenvectors_computed;
    std::string eigenvectors_path;
    ThermodynamicData thermo_data;  // For thermal calculations
};

/**
 * @brief Structure for exact diagonalization parameters
 * 
 * Contains all parameters needed to configure various diagonalization methods,
 * including convergence criteria, method-specific options, and observable calculations.
 */
struct EDParameters {
    // ========== General Parameters ==========
    int max_iterations = 100000;
    int num_eigenvalues = 1;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    std::string output_dir = "";
    
    // ========== Method-Specific Parameters ==========
    double shift = 0.0;        // For shift-invert methods
    int block_size = 4;       // For block methods
    int max_subspace = 100;    // For Davidson method
    
    // ========== TPQ-Specific Parameters ==========
    int num_samples = 1;
    double temp_min = 1e-3;
    double temp_max = 20;
    int num_temp_bins = 100;
    int num_order = 100;           // Order (steps) for canonical TPQ imaginary-time evolution
    int num_measure_freq = 100;    // Frequency of measurements
    double delta_tau = 1e-2;       // Time step for imaginary-time evolution (cTPQ)
    double large_value = 1e5;      // Large value for TPQ
    
    // ========== Observable Calculations ==========
    mutable std::vector<Operator> observables = {};             // Observables to calculate for TPQ
    mutable std::vector<std::string> observable_names = {};     // Names of observables
    double omega_min = -10.0;      // Minimum frequency for spectral function
    double omega_max = 10.0;       // Maximum frequency for spectral function
    int num_points = 1000;         // Number of points for spectral function
    double t_end = 50.0;           // End time for time evolution
    double dt = 0.01;              // Time step for time evolution
    
    // ========== Lattice Parameters ==========
    int num_sites = 0;             // Number of sites in the system
    float spin_length = 0.5;       // Spin length
    int sublattice_size = 1;       // Size of the sublattice
    
    bool calc_observables = false; // Calculate custom observables
    bool measure_spin = false;     // Measure spins
    
    // ========== ARPACK Advanced Options ==========
    // Used when method == ARPACK_ADVANCED
    // These mirror (a subset of) detail_arpack::ArpackAdvancedOptions
    bool arpack_advanced_verbose = false;
    std::string arpack_which = "SM";                            // SM|LM|SR|LR|... (Hermitian typical: SM/LM)
    int arpack_ncv = -1;                                        // Number of Lanczos vectors
    int arpack_max_restarts = 2;                                // Maximum number of restarts
    double arpack_ncv_growth = 1.5;                             // Growth factor for ncv
    bool arpack_auto_enlarge_ncv = true;                        // Automatically enlarge ncv
    bool arpack_two_phase_refine = true;                        // Use two-phase refinement
    double arpack_relaxed_tol = 1e-6;                           // Relaxed tolerance
    bool arpack_shift_invert = false;                           // Use shift-invert mode
    double arpack_sigma = 0.0;                                  // Shift value
    bool arpack_auto_switch_shift_invert = true;                // Auto switch to shift-invert
    double arpack_switch_sigma = 0.0;                           // Sigma for auto-switch
    bool arpack_adaptive_inner_tol = true;                      // Adaptive inner tolerance
    double arpack_inner_tol_factor = 1e-2;                      // Inner tolerance factor
    double arpack_inner_tol_min = 1e-14;                        // Minimum inner tolerance
    int arpack_inner_max_iter = 300;                            // Maximum inner iterations
};

/**
 * @brief Enum for Hamiltonian file formats
 */
enum class HamiltonianFileFormat {
    STANDARD,       // InterAll.dat and Trans.dat format
    SPARSE_MATRIX,  // Sparse matrix format
    CUSTOM          // Custom format requiring a parser function
};

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Core diagonalization function
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    int hilbert_space_dim,
    DiagonalizationMethod method,
    const EDParameters& params
);

// Helper functions
namespace ed_internal {
    void process_thermal_correlations(
        const EDParameters& params,
        int hilbert_space_dim
    );
    
    Operator load_hamiltonian_from_files(
        const std::string& interaction_file,
        const std::string& single_site_file,
        int num_sites,
        float spin_length,
        DiagonalizationMethod method,
        HamiltonianFileFormat format
    );
    
    std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
        Operator& hamiltonian
    );
    
    EDResults diagonalize_symmetry_block(
        Eigen::SparseMatrix<Complex>& block_matrix,
        int block_dim,
        DiagonalizationMethod method,
        const EDParameters& params,
        bool is_target_block = false,
        double large_value_override = 0.0
    );
    
    void transform_and_save_tpq_states(
        const std::string& block_output_dir,
        const std::string& main_output_dir,
        Operator& hamiltonian,
        const std::string& directory,
        int block_dim,
        int block_start_dim,
        int block_idx,
        int num_sites
    );
    
    void transform_and_save_eigenvectors(
        const std::string& block_output_dir,
        const std::string& main_output_dir,
        Operator& hamiltonian,
        const std::string& directory,
        const std::vector<double>& eigenvalues,
        int block_dim,
        int block_start_dim,
        int block_idx,
        int num_sites
    );
    
    struct GroundStateSectorInfo {
        int target_block;
        double min_energy;
        double max_energy;
    };
    
    GroundStateSectorInfo find_ground_state_sector(
        const std::vector<int>& block_sizes,
        const std::string& directory,
        Operator& hamiltonian,
        const EDParameters& params
    );
    
    void setup_symmetry_basis(
        const std::string& directory,
        Operator& hamiltonian
    );
}

// ============================================================================
// MAIN EXACT DIAGONALIZATION FUNCTION
// ============================================================================

/**
 * @brief Main wrapper function for exact diagonalization
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param hilbert_space_dim Dimension of the Hilbert space
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @return EDResults containing eigenvalues and metadata
 */
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    int hilbert_space_dim,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters()
) {
    EDResults results;
    
    // Initialize output directory if needed
    if (!params.output_dir.empty()) {
        std::string cmd = "mkdir -p " + params.output_dir;
        system(cmd.c_str());
    }
    
    // Set eigenvectors flag in results
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors && !params.output_dir.empty()) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Call the appropriate diagonalization method
    switch (method) {
        case DiagonalizationMethod::FULL:
            std::cout << "Using full diagonalization" << std::endl;
            full_diagonalization(H, hilbert_space_dim, params.num_eigenvalues, results.eigenvalues, 
                                 params.output_dir, params.compute_eigenvectors);
            break;


        case DiagonalizationMethod::LANCZOS:
            std::cout << "Using standard Lanczos method" << std::endl;
            lanczos(H, hilbert_space_dim, params.max_iterations, params.num_eigenvalues, 
                    params.tolerance, results.eigenvalues, params.output_dir, 
                    params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::LANCZOS_SELECTIVE:
            std::cout << "Using Lanczos with selective reorthogonalization" << std::endl;
            lanczos_selective_reorth(H, hilbert_space_dim, params.max_iterations, 
                                    params.num_eigenvalues, params.tolerance, 
                                    results.eigenvalues, params.output_dir, 
                                    params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::LANCZOS_NO_ORTHO:
            std::cout << "Using Lanczos without reorthogonalization" << std::endl;
            lanczos_no_ortho(H, hilbert_space_dim, params.max_iterations, 
                           params.num_eigenvalues, params.tolerance, 
                           results.eigenvalues, params.output_dir, 
                           params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::SHIFT_INVERT:
            std::cout << "Using shift-invert Lanczos method with shift = " << params.shift << std::endl;
            shift_invert_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                params.num_eigenvalues, params.shift, 
                                params.tolerance, results.eigenvalues, 
                                params.output_dir, params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::DAVIDSON:
            {
                std::cout << "Using Davidson method" << std::endl;
                std::vector<ComplexVector> eigenvectors;
                davidson_method(H, hilbert_space_dim, params.max_iterations, 
                             params.max_subspace, params.num_eigenvalues, 
                             params.tolerance, results.eigenvalues, 
                             eigenvectors, params.output_dir);
            }
            break;
            
        case DiagonalizationMethod::BICG:
            {
                std::cout << "Using biconjugate gradient method" << std::endl;
                std::vector<ComplexVector> eigenvectors;
                bicg_eigenvalues(H, hilbert_space_dim, params.max_iterations, 
                              params.tolerance, results.eigenvalues, 
                              eigenvectors, params.output_dir);
            }
            break;
            
        case DiagonalizationMethod::LOBPCG:
            std::cout << "Using LOBPCG method" << std::endl;
            lobpcg_diagonalization(H, hilbert_space_dim, params.max_iterations, 
                            params.num_eigenvalues, params.tolerance, 
                            results.eigenvalues, params.output_dir, 
                            params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::KRYLOV_SCHUR:
            std::cout << "Using Krylov-Schur method" << std::endl;
            krylov_schur(H, hilbert_space_dim, params.max_iterations, 
                       params.num_eigenvalues, params.tolerance, 
                       results.eigenvalues, params.output_dir, 
                       params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::OSS:
            std::cout << "Spectrum slicing for full diagonalization" << std::endl;
            optimal_spectrum_solver(
                H, hilbert_space_dim, params.max_iterations,
                results.eigenvalues, params.output_dir, 
                params.compute_eigenvectors
            );
            break;
            
        case DiagonalizationMethod::mTPQ:
            std::cout << "Using microcanonical TPQ method" << std::endl;
        
            microcanonical_tpq(H, hilbert_space_dim,
                            params.max_iterations, params.num_samples,
                            params.num_measure_freq,
                            results.eigenvalues,
                            params.output_dir,
                            params.compute_eigenvectors,
                            params.large_value,
                            params.calc_observables,params.observables, params.observable_names,
                            params.omega_min, params.omega_max,
                            params.num_points, params.t_end, params.dt, params.spin_length, 
                            params.measure_spin, params.sublattice_size, params.num_sites); 
            break;

        case DiagonalizationMethod::cTPQ:
            std::cout << "Using canonical TPQ method" << std::endl;
            // Invoke the canonical TPQ routine provided in TPQ.h
            // Assumes signature analogous to microcanonical_tpq with (num_order, delta_tau)
            canonical_tpq(
                H,                      // Hamiltonian matvec
                hilbert_space_dim,      // N
                params.temp_max,        // beta_max (use configured max inverse temperature)
                params.num_samples,     // num_samples
                params.num_measure_freq,// temp_interval / measurement frequency
                results.eigenvalues,    // energies output vector
                params.output_dir,      // output dir
                params.delta_tau,       // delta_beta (imaginary-time step)
                params.num_order,       // taylor_order
                params.calc_observables,// compute_observables
                params.observables,     // observables
                params.observable_names,// observable names
                params.omega_min,       // omega_min
                params.omega_max,       // omega_max
                params.num_points,      // num_points
                params.t_end,           // t_end
                params.dt,              // dt
                params.spin_length,     // spin length
                params.measure_spin,    // measure Sz and fluctuations
                params.sublattice_size, // sublattice size
                params.num_sites        // number of sites
            );
            break;


        case DiagonalizationMethod::mTPQ_CUDA:
            break;
        

        case DiagonalizationMethod::BLOCK_LANCZOS:
            std::cout << "Using block Lanczos method" << std::endl;
            block_lanczos(H, hilbert_space_dim, 
                        params.max_iterations, params.num_eigenvalues, params.block_size, 
                        params.tolerance, results.eigenvalues, 
                        params.output_dir, params.compute_eigenvectors);
            break;
            
        
        case DiagonalizationMethod::ARPACK_SM:
            std::cout << "Using ARPACK standard eigenvalue solver" << std::endl;
            arpack_ground_state(H, hilbert_space_dim,
                                params.max_iterations, params.num_eigenvalues, params.tolerance,
                                results.eigenvalues, params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::ARPACK_LM:
            std::cout << "Using ARPACK standard eigenvalue solver" << std::endl;
            arpack_largest(H, hilbert_space_dim,
                            params.max_iterations, params.num_eigenvalues, params.tolerance,
                            results.eigenvalues, params.output_dir, params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::ARPACK_SHIFT_INVERT:
            std::cout << "Using ARPACK shift-invert method with shift = " << params.shift << std::endl;
            arpack_shift_invert(H, hilbert_space_dim,
                                params.max_iterations, params.num_eigenvalues, params.tolerance,
                                params.shift,
                                results.eigenvalues, params.output_dir,
                                params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::ARPACK_ADVANCED: {
            std::cout << "Using ARPACK advanced multi-attempt solver" << std::endl;
            detail_arpack::ArpackAdvancedOptions opts;
            opts.nev = params.num_eigenvalues;
            opts.which = params.arpack_which;
            opts.tol = params.tolerance;
            opts.max_iter = params.max_iterations;
            opts.ncv = params.arpack_ncv;
            opts.auto_enlarge_ncv = params.arpack_auto_enlarge_ncv;
            opts.max_restarts = params.arpack_max_restarts;
            opts.ncv_growth = params.arpack_ncv_growth;
            opts.two_phase_refine = params.arpack_two_phase_refine;
            opts.relaxed_tol = params.arpack_relaxed_tol;
            opts.shift_invert = params.arpack_shift_invert;
            opts.sigma = params.arpack_sigma;
            opts.auto_switch_to_shift_invert = params.arpack_auto_switch_shift_invert;
            opts.switch_sigma = params.arpack_switch_sigma;
            opts.adaptive_inner_tol = params.arpack_adaptive_inner_tol;
            opts.inner_tol_factor = params.arpack_inner_tol_factor;
            opts.inner_tol_min = params.arpack_inner_tol_min;
            opts.inner_max_iter = params.arpack_inner_max_iter;
            opts.verbose = params.arpack_advanced_verbose;
            std::vector<Complex> evecs; // optionally capture
            int info = arpack_eigs_advanced(H, hilbert_space_dim, opts,
                                            results.eigenvalues,
                                            params.output_dir,
                                            params.compute_eigenvectors,
                                            params.compute_eigenvectors ? &evecs : nullptr);
            if (info != 0) {
                std::cerr << "ARPACK advanced solver returned info=" << info << std::endl;
            }
            break; }
        
        // Methods not yet fully implemented
        case DiagonalizationMethod::SHIFT_INVERT_ROBUST:
            std::cerr << "SHIFT_INVERT_ROBUST not yet implemented. Using standard SHIFT_INVERT instead." << std::endl;
            shift_invert_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                params.num_eigenvalues, params.shift, 
                                params.tolerance, results.eigenvalues, 
                                params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::mTPQ_MPI:
            std::cerr << "Error: mTPQ_MPI requires MPI build. Use standard mTPQ instead." << std::endl;
            throw std::runtime_error("mTPQ_MPI not available");
            break;
        
        case DiagonalizationMethod::FTLM:
            std::cerr << "Error: FTLM (Finite Temperature Lanczos Method) not yet implemented." << std::endl;
            std::cerr << "Please use FULL diagonalization with thermodynamics or mTPQ/cTPQ methods." << std::endl;
            throw std::runtime_error("FTLM not yet implemented");
            break;
        
        case DiagonalizationMethod::LTLM:
            std::cerr << "Error: LTLM (Low Temperature Lanczos Method) not yet implemented." << std::endl;
            std::cerr << "Please use standard Lanczos or mTPQ/cTPQ methods." << std::endl;
            throw std::runtime_error("LTLM not yet implemented");
            break;
        
        case DiagonalizationMethod::LANCZOS_GPU:
            std::cerr << "Error: LANCZOS_GPU requires CUDA build." << std::endl;
            throw std::runtime_error("GPU methods not available in this build");
            break;
        
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
            std::cerr << "Error: LANCZOS_GPU_FIXED_SZ requires CUDA build." << std::endl;
            throw std::runtime_error("GPU methods not available in this build");
            break;

        default:
            std::cerr << "Unknown diagonalization method selected" << std::endl;
            break;
    }
    
    if (params.compute_eigenvectors) {
        std::cout << "Eigenvectors computed and saved to " << params.output_dir << std::endl;
    }

    // Calculate thermal observables if requested
    if (params.calc_observables) {
        ed_internal::process_thermal_correlations(params, hilbert_space_dim);
    }

    return results;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

namespace ed_internal {

/**
 * @brief Process thermal correlation functions
 * 
 * Searches for correlation files, loads operators, and calculates
 * thermal expectation values at different temperatures.
 * 
 * @param params ED parameters containing temperature ranges and observables
 * @param hilbert_space_dim Dimension of the Hilbert space
 */
void process_thermal_correlations(
    const EDParameters& params,
    int hilbert_space_dim
) {
    std::cout << "Calculating custom observables..." << std::endl;
    std::cout << "Calculating thermal expectation values for correlation operators..." << std::endl;

    // Create output directory for thermal correlation results
    std::string output_correlations_dir = params.output_dir + "/thermal_correlations";
    std::string cmd_mkdir = "mkdir -p " + output_correlations_dir;
    system(cmd_mkdir.c_str());

    // Determine base directory where correlation files might be located
    std::string base_dir;
    if (!params.output_dir.empty()) {
        size_t pos = params.output_dir.find_last_of("/\\");
        base_dir = (pos != std::string::npos) ? params.output_dir.substr(0, pos) : ".";
    } else {
        base_dir = ".";
    }

    std::cout << "Looking for correlation files in: " << base_dir << std::endl;

    // Define correlation file patterns to search for
    std::vector<std::pair<std::string, std::string>> patterns = {
        {"one_body_correlations", "one_body_correlations*.dat"},
        {"two_body_correlations", "two_body_correlations*.dat"}
    };

    // Process each type of correlation file
    for (const auto& [prefix, pattern] : patterns) {
            // Find matching files
            std::string temp_list_file = output_correlations_dir + "/" + prefix + "_files.txt";
            std::string find_command = "find \"" + base_dir + "\" -name \"" + pattern + "\" 2>/dev/null > \"" + temp_list_file + "\"";
            system(find_command.c_str());
            
            // Read the list of files
            std::ifstream file_list(temp_list_file);
            if (!file_list.is_open()) continue;
            
            std::string correlation_file;
            int file_count = 0;

            // Compute thermal expectations at different temperatures
            std::string results_file_path = output_correlations_dir + "/thermal_expectation_" + 
            prefix + ".dat";
            std::ofstream results_file(results_file_path);
            
            if (!results_file.is_open()) {
                std::cerr << "Error: Could not open output file: " << results_file_path << std::endl;
                continue;
            }
                        
            while (std::getline(file_list, correlation_file)) {
                if (correlation_file.empty()) continue;
                file_count++;
                
                // Extract operator type from filename
                size_t prefix_pos = correlation_file.find(prefix);
                if (prefix_pos == std::string::npos) continue;
                
                
                std::cout << "Processing " << prefix << " file: " << correlation_file << std::endl;
                
                try {
                    // Load the operator
                    

                    std::ifstream file(correlation_file);
                    if (!file.is_open()) {
                        throw std::runtime_error("Could not open file: " + correlation_file);
                    }
                    std::cout << "Reading file: " << correlation_file << std::endl;
                    std::string line;
                    
                    // Skip the first line (header)
                    std::getline(file, line);
                    
                    // Read the number of lines
            
                    std::getline(file, line);
                    std::istringstream iss(line);
                    int numLines;
                    std::string m;
                    iss >> m >> numLines;
                    // std::cout << "Number of lines: " << numLines << std::endl;
                    
                    // Skip the next 3 lines (separators/headers)
                    for (int i = 0; i < 3; ++i) {
                        std::getline(file, line);
                    }
                                            
                    if (prefix == "one_body_correlations") {
                        results_file << std::setw(12) << "Temperatures" << " "
                                    << std::setw(12) << "Beta" << " "
                                    << std::setw(12) << "Op1" << " "
                                    << std::setw(12) << "Index1" << " "
                                    << std::setw(12) << "Expectation" << std::endl;
                    } else if (prefix == "two_body_correlations") {
                        results_file << std::setw(12) << "Temperatures" << " "
                                    << std::setw(12) << "Beta" << " "
                                    << std::setw(12) << "Op1" << " "
                                    << std::setw(12) << "Op2" << " "
                                    << std::setw(12) << "Index1" << " "
                                    << std::setw(12) << "Index2" << " "
                                    << std::setw(12) << "Expectation" << std::endl;
                    }

                    // Process transform data
                    int lineCount = 0;
                    while (std::getline(file, line) && lineCount < numLines) {
                        Operator correlation_op(params.num_sites, params.spin_length);
                        std::istringstream lineStream(line);
                        int Op1, indx1, Op2, indx2;
                        double E, F;
                        if (prefix == "one_body_correlations") {

                            // std::cout << "Reading line: " << line << std::endl;
                            if (!(lineStream >> Op1 >> indx1 >> E >> F)) {
                                continue; // Skip invalid lines
                            }

                            correlation_op.loadonebodycorrelation(Op1, indx1);
                        } else if (prefix == "two_body_correlations") {

                            // std::cout << "Reading line: " << line << std::endl;
                            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                                continue; // Skip invalid lines
                            }
                            correlation_op.loadtwobodycorrelation(Op1, indx1, Op2, indx2);
                        }

                        // Create a lambda to apply the operator
                        auto apply_correlation_op = [&correlation_op](const Complex* in, Complex* out, int n) {
                            std::vector<Complex> in_vec(in, in + n);
                            std::vector<Complex> out_vec = correlation_op.apply(in_vec);
                            std::copy(out_vec.begin(), out_vec.end(), out);
                        };
                        

                        // Calculate thermal expectations at temperature points
                        int num_temps = std::min(params.num_temp_bins, 20);
                        double log_temp_min = std::log(params.temp_min);
                        double log_temp_max = std::log(params.temp_max);
                        double log_temp_step = (log_temp_max - log_temp_min) / std::max(1, num_temps - 1);

                        for (int i = 0; i < num_temps; i++) {
                            double T = std::exp(log_temp_min + i * log_temp_step);
                            double beta = 1.0 / T;
                                // Compute thermal expectation
                            Complex expectation = calculate_thermal_expectation(
                                apply_correlation_op, hilbert_space_dim, beta, params.output_dir + "/eigenvectors/");
                            
                            std::cout << "T: " << T << ", beta: " << beta << ", expectation: " 
                                        << expectation.real() << " + " << expectation.imag() << "i" << std::endl;

                            // Write to file
                            if (prefix == "one_body_correlations") {
                                results_file << std::setw(12) << std::setprecision(6) << T << " "
                                            << std::setw(12) << std::setprecision(6) << beta << " "
                                            << std::setw(12) << std::setprecision(6) << Op1 << " "
                                            << std::setw(12) << std::setprecision(6) << indx1 << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.real() << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.imag() << std::endl;
                            } else if (prefix == "two_body_correlations") {
                                results_file << std::setw(12) << std::setprecision(6) << T << " "
                                            << std::setw(12) << std::setprecision(6) << beta << " "
                                            << std::setw(12) << std::setprecision(6) << Op1 << " "
                                            << std::setw(12) << std::setprecision(6) << Op2 << " "
                                            << std::setw(12) << std::setprecision(6) << indx1 << " "
                                            << std::setw(12) << std::setprecision(6) << indx2 << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.real() << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.imag() << std::endl;
                            }
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing " << correlation_file << ": " << e.what() << std::endl;
                }
            }
            results_file.close();
            std::cout << "Thermal expectations saved to: " << results_file_path << std::endl;
            file_list.close();
            std::cout << "Processed " << file_count << " " << prefix << " files" << std::endl;
            std::remove(temp_list_file.c_str());
        }

    std::cout << "Thermal expectation calculations complete!" << std::endl;
}

/**
 * @brief Factory function to create the appropriate operator type
 * Creates either a standard Operator or FixedSzOperator based on config
 */
template<typename OperatorType = Operator>
OperatorType* create_operator(const SystemConfig& config) {
    if (config.use_fixed_sz) {
        int n_up = (config.n_up >= 0) ? config.n_up : config.num_sites / 2;
        return new FixedSzOperator(config.num_sites, config.spin_length, n_up);
    } else {
        return new OperatorType(config.num_sites, config.spin_length);
    }
}

/**
 * @brief Load Hamiltonian from files based on format
 */
Operator load_hamiltonian_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file,
    int num_sites,
    float spin_length,
    DiagonalizationMethod method,
    HamiltonianFileFormat format
) {
    Operator hamiltonian(num_sites, spin_length);
    
    switch (format) {
        case HamiltonianFileFormat::STANDARD:
            // Load terms from files
            if (!single_site_file.empty()) {
                hamiltonian.loadFromFile(single_site_file);
            }
            if (!interaction_file.empty()) {
                hamiltonian.loadFromInterAllFile(interaction_file);
            }
            // Build sparse matrix (except for full diagonalization)
            if (method != DiagonalizationMethod::FULL) {
                hamiltonian.buildSparseMatrix();
            }
            break;
            
        case HamiltonianFileFormat::SPARSE_MATRIX:
            throw std::runtime_error("Sparse matrix format not yet implemented");
            
        case HamiltonianFileFormat::CUSTOM:
            throw std::runtime_error("Custom format requires a parser function");
            
        default:
            throw std::runtime_error("Unknown Hamiltonian file format");
    }
    
    return hamiltonian;
}

/**
 * @brief Create a lambda function to apply the Hamiltonian
 */
std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
    Operator& hamiltonian
) {
    return [&hamiltonian](const Complex* in, Complex* out, int n) {
        std::vector<Complex> in_vec(in, in + n);
        std::vector<Complex> out_vec = hamiltonian.apply(in_vec);
        std::copy(out_vec.begin(), out_vec.end(), out);
    };
}

/**
 * @brief Diagonalize a single symmetry block
 */
EDResults diagonalize_symmetry_block(
    Eigen::SparseMatrix<Complex>& block_matrix,
    int block_dim,
    DiagonalizationMethod method,
    const EDParameters& params,
    bool is_target_block,
    double large_value_override
) {
    // Define matrix-vector product for this block
    std::function<void(const Complex*, Complex*, int)> apply_block =
        [block_matrix](const Complex* in, Complex* out, int n) {
            if (n != block_matrix.rows()) {
                throw std::invalid_argument("Block apply: dimension mismatch");
            }
            Eigen::Map<const Eigen::VectorXcd> vin(in, n);
            Eigen::Map<Eigen::VectorXcd> vout(out, n);
            vout = block_matrix * vin;
        };

    // Configure parameters for block diagonalization
    EDParameters block_params = params;
    block_params.num_eigenvalues = std::min(params.num_eigenvalues, block_dim);
    
    if (is_target_block && large_value_override > 0) {
        block_params.large_value = large_value_override;
    }
    
    return exact_diagonalization_core(apply_block, block_dim, method, block_params);
}

/**
 * @brief Transform and save TPQ states from block to full basis
 */
void transform_and_save_tpq_states(
    const std::string& block_output_dir,
    const std::string& main_output_dir,
    Operator& hamiltonian,
    const std::string& directory,
    int block_dim,
    int block_start_dim,
    int block_idx,
    int num_sites
) {
    std::cout << "Transforming TPQ states for block " << block_idx << std::endl;
    
    // Find all TPQ state files
    std::string temp_list_file = main_output_dir + "/tpq_state_files_" + std::to_string(block_idx) + ".txt";
    std::string find_command = "find \"" + block_output_dir + "\" -name \"tpq_state_*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
    system(find_command.c_str());
    
    std::ifstream file_list(temp_list_file);
    if (!file_list.is_open()) return;
    
    std::string tpq_state_file;
    while (std::getline(file_list, tpq_state_file)) {
        if (tpq_state_file.empty()) continue;
        
        // Extract sample and beta from filename
        size_t sample_pos = tpq_state_file.find("tpq_state_");
        size_t beta_pos = tpq_state_file.find("_beta=");
        size_t dat_pos = tpq_state_file.find(".dat");
        
        if (sample_pos == std::string::npos || beta_pos == std::string::npos || dat_pos == std::string::npos) {
            continue;
        }
        
        std::string sample_str = tpq_state_file.substr(sample_pos + 10, beta_pos - sample_pos - 10);
        std::string beta_str = tpq_state_file.substr(beta_pos + 6, dat_pos - beta_pos - 6);
        
        std::cout << "Processing TPQ state: sample=" << sample_str << ", beta=" << beta_str << std::endl;
        
        // Read block TPQ state
        std::ifstream tpq_file(tpq_state_file, std::ios::binary);
        if (!tpq_file.is_open()) {
            std::cerr << "Warning: Could not open TPQ state file: " << tpq_state_file << std::endl;
            continue;
        }

        size_t stored_size = 0;
        tpq_file.read(reinterpret_cast<char*>(&stored_size), sizeof(size_t));
        if (!tpq_file || stored_size != static_cast<size_t>(block_dim)) {
            std::cerr << "Warning: Invalid TPQ state file: " << tpq_state_file << std::endl;
            tpq_file.close();
            continue;
        }

        ComplexVector block_tpq_state(block_dim);
        tpq_file.read(reinterpret_cast<char*>(block_tpq_state.data()), stored_size * sizeof(Complex));
        tpq_file.close();
        
        if (!tpq_file) {
            std::cerr << "Warning: Failed to read TPQ state data" << std::endl;
            continue;
        }

        // Normalize
        double norm = std::sqrt(std::inner_product(block_tpq_state.begin(), block_tpq_state.end(),
                                                     block_tpq_state.begin(), 0.0,
                                                     std::plus<double>(),
                                                     [](const Complex& a, const Complex& b) { return std::norm(a); }));
        if (norm < 1e-10) {
            std::cerr << "Warning: TPQ state has zero norm" << std::endl;
            continue;
        }
        
        for (auto& val : block_tpq_state) val /= norm;
        
        // Transform to full Hilbert space
        ComplexVector full_tpq_state(1ULL << num_sites, Complex(0.0, 0.0));
        for (int i = 0; i < block_dim; ++i) {
            std::vector<Complex> basis_vector = hamiltonian.read_sym_basis(i + block_start_dim, directory);
            full_tpq_state += basis_vector * block_tpq_state[i];
        }

        // Save transformed state
        std::string transformed_file = main_output_dir + "/tpq_state_" + sample_str + "_beta=" + beta_str + ".dat";
        std::ofstream out_file(transformed_file);
        out_file << full_tpq_state.size() << std::endl;
        for (size_t i = 0; i < full_tpq_state.size(); ++i) {
            if (std::abs(full_tpq_state[i]) > 1e-10) {
                out_file << i << " " << full_tpq_state[i].real() << " " << full_tpq_state[i].imag() << std::endl;
            }
        }
        out_file.close();
    }
    
    file_list.close();
    std::remove(temp_list_file.c_str());
}

/**
 * @brief Transform and save eigenvectors from block to full basis
 */
void transform_and_save_eigenvectors(
    const std::string& block_output_dir,
    const std::string& main_output_dir,
    Operator& hamiltonian,
    const std::string& directory,
    const std::vector<double>& eigenvalues,
    int block_dim,
    int block_start_dim,
    int block_idx,
    int num_sites
) {
    std::cout << "Processing eigenvectors for block " << block_idx << std::endl;
    
    for (size_t eigen_idx = 0; eigen_idx < eigenvalues.size(); ++eigen_idx) {
        std::string block_eigenvector_file = block_output_dir + "/eigenvector_" + std::to_string(eigen_idx) + ".dat";
        
        // Read block eigenvector
        std::vector<Complex> block_eigenvector(block_dim);
        std::ifstream eigen_file(block_eigenvector_file);
        if (!eigen_file.is_open()) {
            std::cerr << "Warning: Could not open eigenvector file: " << block_eigenvector_file << std::endl;
            continue;
        }
        
        std::string line;
        int num_entries = 0;
        while (std::getline(eigen_file, line) && num_entries < block_dim) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            double real_part, imag_part;
            if (!(iss >> real_part >> imag_part)) continue;
            block_eigenvector[num_entries++] = Complex(real_part, imag_part);
        }
        eigen_file.close();

        // Validate eigenvector
        double norm = 0.0;
        bool has_invalid = false;
        for (const auto& val : block_eigenvector) {
            if (!std::isfinite(val.real()) || !std::isfinite(val.imag())) {
                has_invalid = true;
                break;
            }
            norm += std::norm(val);
        }
        
        if (has_invalid || norm < 1e-10) {
            std::cerr << "Warning: Invalid or zero-norm eigenvector in file: " << block_eigenvector_file << std::endl;
            continue;
        }
        
        norm = std::sqrt(norm);
        
        // Transform to full basis
        std::vector<Complex> transformed_eigenvector(1ULL << num_sites, Complex(0.0, 0.0));
        for (size_t i = 0; i < block_dim; ++i) {
            std::vector<Complex> temp = hamiltonian.read_sym_basis(i + block_start_dim, directory);
            transformed_eigenvector += temp * block_eigenvector[i];
        }
        
        // Normalize
        for (auto& val : transformed_eigenvector) val /= norm;
        
        // Save
        std::string transformed_file = main_output_dir + "/eigenvector_block" + 
                                       std::to_string(block_idx) + "_" + std::to_string(eigen_idx) + ".dat";
        save_tpq_state(transformed_eigenvector, transformed_file);
    }
}

/**
 * @brief Find the symmetry sector containing the ground state
 */
GroundStateSectorInfo find_ground_state_sector(
    const std::vector<int>& block_sizes,
    const std::string& directory,
    Operator& hamiltonian,
    const EDParameters& params
) {
    std::cout << "=== Scanning blocks to find ground state sector ===" << std::endl;
    
    GroundStateSectorInfo info;
    info.min_energy = std::numeric_limits<double>::max();
    info.max_energy = std::numeric_limits<double>::lowest();
    info.target_block = 0;
    
    for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
        int block_dim = block_sizes[block_idx];
        std::cout << "Scanning block " << block_idx + 1 << "/" << block_sizes.size() 
                  << " (dimension: " << block_dim << ")" << std::endl;
        
        // Load block Hamiltonian
        std::string block_file = directory + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
        Eigen::SparseMatrix<Complex> block_matrix = hamiltonian.loadSymmetrizedBlock(block_file);
        
        // Quick Lanczos scan
        EDParameters scan_params = params;
        scan_params.num_eigenvalues = std::min(10, block_dim);
        scan_params.max_iterations = scan_params.num_eigenvalues * 4 + 20;
        scan_params.compute_eigenvectors = false;
        scan_params.calc_observables = false;
        scan_params.measure_spin = false;
        
        EDResults block_results = diagonalize_symmetry_block(
            block_matrix, block_dim, DiagonalizationMethod::LANCZOS, scan_params
        );
        
        double min_block = *std::min_element(block_results.eigenvalues.begin(), block_results.eigenvalues.end());
        double max_block = *std::max_element(block_results.eigenvalues.begin(), block_results.eigenvalues.end());
        std::cout << "  Block " << block_idx << ": E_min=" << min_block << ", E_max=" << max_block << std::endl;
        
        if (min_block < info.min_energy) {
            info.min_energy = min_block;
            info.max_energy = max_block;
            info.target_block = block_idx;
        }
    }
    
    std::cout << "\nGround state sector identified:" << std::endl;
    std::cout << "  Block index: " << info.target_block << std::endl;
    std::cout << "  E_min (estimated ground state): " << info.min_energy << std::endl;
    std::cout << "  E_max in sector: " << info.max_energy << std::endl;
    
    return info;
}

/**
 * @brief Setup symmetrized basis (generate or load)
 */
void setup_symmetry_basis(
    const std::string& directory,
    Operator& hamiltonian
) {
    std::string sym_basis_dir = directory + "/sym_basis";
    std::string sym_blocks_dir = directory + "/sym_blocks";
    std::string block_sizes_file = sym_basis_dir + "/sym_block_sizes.txt";
    
    struct stat buffer;
    // Check if the actual block sizes file exists, not just the directory
    bool sym_basis_exists = (stat(block_sizes_file.c_str(), &buffer) == 0);
    
    // Check if at least one block file exists (block_0.dat)
    std::string first_block_file = sym_blocks_dir + "/block_0.dat";
    bool sym_blocks_exist = (stat(first_block_file.c_str(), &buffer) == 0);
    
    if (!sym_basis_exists) {
        std::cout << "Symmetrized basis not found. Generating..." << std::endl;
        hamiltonian.generateSymmetrizedBasis(directory);
    } else {
        std::cout << "Using existing symmetrized basis from " << sym_basis_dir << std::endl;
    }
    
    if (!sym_blocks_exist) {
        hamiltonian.buildAndSaveSymmetrizedBlocks(directory);
    } else {
        hamiltonian.loadAllSymmetrizedBlocks(directory);
    }
}

} // namespace ed_internal

// ============================================================================
// FIXED SZ EXACT DIAGONALIZATION
// ============================================================================

/**
 * @brief Exact diagonalization in fixed Sz sector
 * 
 * Performs diagonalization on a restricted Hilbert space with fixed
 * total Sz quantum number, significantly reducing memory and computational cost.
 * 
 * @param interaction_file Path to interaction file
 * @param single_site_file Path to single-site file
 * @param num_sites Number of sites
 * @param spin_length Spin length (usually 0.5)
 * @param n_up Number of up spins (determines Sz sector)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_fixed_sz(
    const std::string& interaction_file,
    const std::string& single_site_file,
    int num_sites,
    float spin_length,
    int n_up,
    DiagonalizationMethod method,
    const EDParameters& params
) {
    std::cout << "\n=== Fixed Sz Exact Diagonalization ===" << std::endl;
    std::cout << "System: " << num_sites << " sites, spin = " << spin_length << std::endl;
    std::cout << "Fixed Sz sector: n_up = " << n_up << " (Sz = " << (n_up - num_sites/2.0) << ")" << std::endl;
    
    // Create Fixed Sz operator
    FixedSzOperator hamiltonian(num_sites, spin_length, n_up);
    
    // Load Hamiltonian terms
    if (!single_site_file.empty()) {
        std::ifstream file(single_site_file);
        if (file.is_open()) {
            hamiltonian.loadFromFile(single_site_file);
        }
    }
    if (!interaction_file.empty()) {
        std::ifstream file(interaction_file);
        if (file.is_open()) {
            hamiltonian.loadFromInterAllFile(interaction_file);
        }
    }
    
    // Get dimension of fixed Sz sector
    int fixed_sz_dim = hamiltonian.getFixedSzDim();
    int full_dim = 1 << num_sites;
    
    std::cout << "Hilbert space dimension:" << std::endl;
    std::cout << "  Full space: " << full_dim << std::endl;
    std::cout << "  Fixed Sz:   " << fixed_sz_dim << std::endl;
    std::cout << "  Reduction:  " << (double)full_dim / fixed_sz_dim << "x" << std::endl;
    
    // Build sparse matrix
    if (method != DiagonalizationMethod::FULL) {
        hamiltonian.buildFixedSzMatrix();
    }
    
    // Create apply function
    auto apply_hamiltonian = [&hamiltonian, fixed_sz_dim](const Complex* in, Complex* out, int n) {
        if (n != fixed_sz_dim) {
            throw std::runtime_error("Dimension mismatch in fixed Sz apply");
        }
        hamiltonian.apply(in, out, n);
    };
    
    // Perform diagonalization
    std::cout << "\nDiagonalizing..." << std::endl;
    auto results = exact_diagonalization_core(apply_hamiltonian, fixed_sz_dim, method, params);
    
    std::cout << "=== Fixed Sz Diagonalization Complete ===" << std::endl;
    return results;
}

// ============================================================================
// FILE-BASED WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Wrapper function to perform exact diagonalization from Hamiltonian files
 * 
 * Loads Hamiltonian from input files and performs diagonalization.
 * 
 * @param interaction_file Path to interaction file (e.g., InterAll.dat)
 * @param single_site_file Path to single-site file (e.g., Trans.dat)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @return EDResults containing eigenvalues and metadata
 */
EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD
) {
    std::cerr << "[DEBUG] exact_diagonalization_from_files: num_sites=" << params.num_sites 
              << ", method=" << static_cast<int>(method) << std::endl;
    
    // Load Hamiltonian (for CPU methods)
    Operator hamiltonian = ed_internal::load_hamiltonian_from_files(
        interaction_file, single_site_file, params.num_sites, params.spin_length, method, format
    );
    
    // Calculate Hilbert space dimension
    int hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
    std::cerr << "[DEBUG] hilbert_space_dim=" << hilbert_space_dim << std::endl;
    
    // Create Hamiltonian apply function
    auto apply_hamiltonian = ed_internal::create_hamiltonian_apply_function(hamiltonian);
    
    // Perform diagonalization
    return exact_diagonalization_core(apply_hamiltonian, hilbert_space_dim, method, params);
}

/**
 * @brief Wrapper function to perform exact diagonalization from a directory
 * 
 * Convenience function that constructs file paths from a directory and
 * calls exact_diagonalization_from_files.
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @param interaction_filename Name of interaction file (default: "InterAll.dat")
 * @param single_site_filename Name of single-site file (default: "Trans.dat")
 * @return EDResults containing eigenvalues and metadata
 */
EDResults exact_diagonalization_from_directory(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    // Call the file-based wrapper
    return exact_diagonalization_from_files(
        interaction_file, single_site_file, method, params, format
    );
}

// ============================================================================
// SYMMETRIZED BASIS WRAPPER
// ============================================================================

/**
 * @brief Perform exact diagonalization using symmetrized basis
 * 
 * This function exploits lattice symmetries to block-diagonalize the Hamiltonian,
 * significantly reducing computational cost for symmetric systems.
 * 
 * Steps:
 * 1. Generate or load automorphisms (lattice symmetries)
 * 2. Construct symmetrized basis
 * 3. Block-diagonalize Hamiltonian
 * 4. Diagonalize each block separately
 * 5. Transform eigenvectors back to full basis
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @param interaction_filename Name of interaction file (default: "InterAll.dat")
 * @param single_site_filename Name of single-site file (default: "Trans.dat")
 * @return EDResults containing eigenvalues and metadata
 */
EDResults exact_diagonalization_from_directory_symmetrized(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cerr << "[DEBUG] exact_diagonalization_from_directory_symmetrized: num_sites=" 
              << params.num_sites << ", method=" << static_cast<int>(method) << std::endl;

    // ========== Step 1: Generate or Load Automorphisms ==========
    std::string automorphism_file = directory + "/automorphism_results/automorphisms.json";
    struct stat automorphism_buffer;
    bool automorphisms_exist = (stat(automorphism_file.c_str(), &automorphism_buffer) == 0);

    if (!automorphisms_exist) {
        std::string automorphism_finder_path = std::string(__FILE__);
        automorphism_finder_path = automorphism_finder_path.substr(0, automorphism_finder_path.find_last_of("/\\"));
        automorphism_finder_path += "/automorphism_finder.py";
        std::string cmd = "python " + automorphism_finder_path + " --data_dir=\"" + directory + "\"";
        std::cout << "Running automorphism finder: " << cmd << std::endl;
        int result = system(cmd.c_str());
        if (result != 0) {
            std::cerr << "Warning: Automorphism finder returned non-zero code: " << result << std::endl;
        }
    } else {
        std::cout << "Using existing automorphism results from: " << automorphism_file << std::endl;
    }
    
    // ========== Step 2: Load Hamiltonian ==========
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    EDResults results;
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    Operator hamiltonian(params.num_sites, params.spin_length);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    // ========== Step 3: Setup Symmetrized Basis ==========
    ed_internal::setup_symmetry_basis(directory, hamiltonian);
    
    std::vector<int> block_sizes = hamiltonian.symmetrized_block_ham_sizes;
    std::cout << "Found " << block_sizes.size() << " symmetrized blocks with sizes: ";
    for (const auto& size : block_sizes) std::cout << size << " ";
    std::cout << std::endl;

    if (block_sizes.empty()) {
        throw std::runtime_error("No symmetrized blocks found. Failed to generate symmetrized basis.");
    }
    
    if (!params.output_dir.empty()) {
        system(("mkdir -p " + params.output_dir).c_str());
    }
    
    // ========== Step 4: For TPQ - Find Ground State Sector ==========
    bool is_tpq_method = (method == DiagonalizationMethod::mTPQ || 
                          method == DiagonalizationMethod::mTPQ_CUDA || 
                          method == DiagonalizationMethod::cTPQ);
    
    ed_internal::GroundStateSectorInfo gs_info;
    if (is_tpq_method) {
        gs_info = ed_internal::find_ground_state_sector(block_sizes, directory, hamiltonian, params);
    }

    // ========== Step 5: Diagonalize Each Block ==========
    struct EigenInfo {
        double value;
        int block_idx;
        int eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;
    
    int block_start_dim = 0;
    for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
        int block_dim = block_sizes[block_idx];
        std::cout << "Diagonalizing block " << block_idx + 1 << "/" << block_sizes.size() 
                  << " (dimension: " << block_dim << ")" << std::endl;
        
        // Load block Hamiltonian
        std::string block_file = directory + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
        Eigen::SparseMatrix<Complex> block_matrix = hamiltonian.loadSymmetrizedBlock(block_file);

        // Configure block parameters
        EDParameters block_params = params;
        // Compute at least params.num_eigenvalues for each block (up to block dimension)
        block_params.num_eigenvalues = std::min(params.num_eigenvalues, block_dim);
        if (params.compute_eigenvectors) {
            block_params.output_dir = params.output_dir + "/min_sector";
            system(("mkdir -p " + block_params.output_dir).c_str());
        }
        
        // Diagonalize (only target block for TPQ)
        EDResults block_results;
        bool is_target_block = (is_tpq_method && gs_info.target_block == static_cast<int>(block_idx));
        
        if (is_tpq_method && !is_target_block) {
            // Skip non-target blocks for TPQ
        } else {
            double large_val = is_target_block ? std::max(gs_info.max_energy * 10, params.large_value) : 0.0;
            if (is_target_block) {
                std::cout << "Running TPQ in ground state sector with large value " << large_val << std::endl;
            }
            block_results = ed_internal::diagonalize_symmetry_block(
                block_matrix, block_dim, method, block_params, is_target_block, large_val
            );
        }
            
        // Store eigenvalues
        for (size_t i = 0; i < block_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({block_results.eigenvalues[i], static_cast<int>(block_idx), static_cast<int>(i)});
        }
        
        // Transform eigenvectors/states if needed
        if (params.compute_eigenvectors || (is_tpq_method && is_target_block)) {
            std::string eigenvector_dir = params.output_dir + "/eigenvectors";
            system(("mkdir -p " + eigenvector_dir).c_str());
            
            if (is_tpq_method && is_target_block) {
                ed_internal::transform_and_save_tpq_states(
                    block_params.output_dir, params.output_dir, hamiltonian, directory,
                    block_dim, block_start_dim, block_idx, params.num_sites
                );
            }
            
            if (params.compute_eigenvectors && method != DiagonalizationMethod::mTPQ && 
                method != DiagonalizationMethod::mTPQ_CUDA && method != DiagonalizationMethod::cTPQ) {
                ed_internal::transform_and_save_eigenvectors(
                    block_params.output_dir, params.output_dir, hamiltonian, directory,
                    block_results.eigenvalues, block_dim, block_start_dim, block_idx, params.num_sites
                );
            }
        }
        
        block_start_dim += block_dim;
    }
    
    // ========== Step 6: Finalize Results ==========
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    if (all_eigen_info.size() > static_cast<size_t>(params.num_eigenvalues)) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    // Create eigenvector mapping file
    if (params.compute_eigenvectors && !params.output_dir.empty()) {
        std::ofstream map_file(params.output_dir + "/eigenvector_mapping.txt");
        if (map_file.is_open()) {
            map_file << "# Global Index, Eigenvalue, Block Index, Block Eigenvalue Index, Filename" << std::endl;
            for (size_t i = 0; i < all_eigen_info.size(); ++i) {
                const auto& info = all_eigen_info[i];
                map_file << i << " " << info.value << " " << info.block_idx << " " << info.eigen_idx 
                        << " eigenvector_block" << info.block_idx << "_" << info.eigen_idx << ".dat" << std::endl;
            }
        }
    }
    
    return results;
}



#endif