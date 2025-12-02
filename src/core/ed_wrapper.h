#ifndef ED_WRAPPER_H
#define ED_WRAPPER_H

// ============================================================================
// INCLUDES
// ============================================================================
#include "../cpu_solvers/TPQ.h"
#include "../cpu_solvers/CG.h"
#include "../cpu_solvers/arpack.h"
#include "../cpu_solvers/lanczos.h"
#include "../cpu_solvers/ftlm.h"
#include "../cpu_solvers/ltlm.h"
#include "../cpu_solvers/hybrid_thermal.h"
#include "construct_ham.h"
#include "../cpu_solvers/observables.h"
#include "ed_config.h"
#include "system_utils.h"
#include <sys/stat.h>
#include <filesystem>
#include <algorithm>

// GPU support
#ifdef WITH_CUDA
#include "gpu/gpu_ed_wrapper.h"
#endif


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
    CHEBYSHEV_FILTERED,    // Chebyshev filtered Lanczos for spectral slicing
    SHIFT_INVERT,          // Shift-invert Lanczos
    SHIFT_INVERT_ROBUST,   // Robust shift-invert Lanczos
    DAVIDSON,              // Davidson method
    BICG,                  // Biconjugate gradient
    LOBPCG,                // Locally optimal block preconditioned conjugate gradient
    KRYLOV_SCHUR,          // Krylov-Schur algorithm
    IMPLICIT_RESTART_LANCZOS,  // Implicitly restarted Lanczos algorithm
    THICK_RESTART_LANCZOS,     // Thick restart Lanczos algorithm with locking
    FULL,                  // Full diagonalization
    OSS,                   // Optimal spectrum solver
    
    // Thermal methods
    mTPQ,                  // Microcanonical Thermal Pure Quantum states
    mTPQ_MPI,              // MPI version of mTPQ
    cTPQ,                  // Canonical Thermal Pure Quantum states
    mTPQ_CUDA,             // CUDA microcanonical Thermal Pure Quantum states
    FTLM,                  // Finite Temperature Lanczos Method
    LTLM,                  // Low Temperature Lanczos Method
    HYBRID,                // Hybrid Thermal Method (LTLM + FTLM with automatic crossover)
    
    // ARPACK methods
    ARPACK_SM,             // ARPACK smallest magnitude eigenvalues
    ARPACK_LM,             // ARPACK largest magnitude eigenvalues
    ARPACK_SHIFT_INVERT,   // ARPACK in shift-invert mode
    ARPACK_ADVANCED,       // ARPACK advanced multi-attempt strategy
    
    // GPU methods
    LANCZOS_GPU,           // GPU-accelerated Lanczos (full Hilbert space)
    LANCZOS_GPU_FIXED_SZ,  // GPU-accelerated Lanczos (fixed Sz sector)
    DAVIDSON_GPU,          // GPU-accelerated Davidson method
    LOBPCG_GPU,            // GPU-accelerated LOBPCG method
    mTPQ_GPU,              // GPU-accelerated microcanonical TPQ
    cTPQ_GPU,              // GPU-accelerated canonical TPQ
    FTLM_GPU,              // GPU-accelerated Finite Temperature Lanczos Method
    FTLM_GPU_FIXED_SZ      // GPU-accelerated FTLM (fixed Sz sector)
};

/**
 * @brief Structure to hold exact diagonalization results
 */
struct EDResults {
    std::vector<double> eigenvalues;
    bool eigenvectors_computed;
    std::string eigenvectors_path;
    ThermodynamicData thermo_data;  // For thermal calculations
    FTLMResults ftlm_results;       // For FTLM calculations (includes per-sector data)
};

/**
 * @brief Structure for exact diagonalization parameters
 * 
 * Contains all parameters needed to configure various diagonalization methods,
 * including convergence criteria, method-specific options, and observable calculations.
 */
struct EDParameters {
    // ========== General Parameters ==========
    uint64_t max_iterations = 10000;
    uint64_t num_eigenvalues = 1;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    std::string output_dir = "";
    
    // ========== Method-Specific Parameters ==========
    double shift = 0.0;        // For shift-invert methods
    uint64_t block_size = 4;       // For block methods
    uint64_t max_subspace = 100;    // For Davidson method
    double target_lower = 0.0; // Lower energy bound for Chebyshev filtered (0 = auto)
    double target_upper = 0.0; // Upper energy bound for Chebyshev filtered (0 = auto)
    
    // ========== TPQ-Specific Parameters ==========
    uint64_t num_samples = 1;
    double temp_min = 1e-3;
    double temp_max = 20;
    uint64_t num_temp_bins = 100;
    uint64_t num_order = 100;           // Order (steps) for canonical TPQ imaginary-time evolution
    uint64_t num_measure_freq = 100;    // Frequency of measurements
    double delta_tau = 1e-2;       // Time step for imaginary-time evolution (cTPQ)
    double large_value = 1e5;      // Large value for TPQ
    bool continue_quenching = false;  // Continue quenching from saved state
    uint64_t continue_sample = 0;          // Sample to continue from (0 = auto-detect lowest energy)
    double continue_beta = 0.0;       // Beta to continue from (0.0 = use saved beta)
    
    // ========== FTLM-Specific Parameters ==========
    uint64_t ftlm_krylov_dim = 100;     // Krylov subspace dimension per sample
    bool ftlm_full_reorth = true; // Use full reorthogonalization
    uint64_t ftlm_reorth_freq = 10;     // Reorthogonalization frequency
    uint64_t ftlm_seed = 0;    // Random seed (0 = auto)
    bool ftlm_store_samples = false; // Store per-sample intermediate data
    bool ftlm_error_bars = true;   // Compute error bars
    
    // ========== LTLM-Specific Parameters ==========
    uint64_t ltlm_krylov_dim = 200;     // Krylov subspace dimension for excitations
    uint64_t ltlm_ground_krylov = 100;  // Krylov dimension for finding ground state
    bool ltlm_full_reorth = true; // Use full reorthogonalization
    uint64_t ltlm_reorth_freq = 10;     // Reorthogonalization frequency
    uint64_t ltlm_seed = 0;    // Random seed (0 = auto)
    bool ltlm_store_data = false;  // Store intermediate data
    bool use_hybrid_method = false; // Use hybrid LTLM/FTLM (deprecated, use method=HYBRID)
    double hybrid_crossover = 1.0; // Temperature crossover for hybrid
    bool hybrid_auto_crossover = false; // Auto-determine crossover temperature
    
    // ========== Observable Calculations ==========
    mutable std::vector<Operator> observables = {};             // Observables to calculate for TPQ
    mutable std::vector<std::string> observable_names = {};     // Names of observables
    double omega_min = -10.0;      // Minimum frequency for spectral function
    double omega_max = 10.0;       // Maximum frequency for spectral function
    uint64_t num_points = 1000;         // Number of points for spectral function
    double t_end = 50.0;           // End time for time evolution
    double dt = 0.01;              // Time step for time evolution
    
    // ========== Lattice Parameters ==========
    uint64_t num_sites = 0;             // Number of sites in the system
    float spin_length = 0.5;       // Spin length
    uint64_t sublattice_size = 1;       // Size of the sublattice
    
    bool calc_observables = false; // Calculate custom observables
    bool measure_spin = false;     // Measure spins
    
    // ========== Fixed-Sz Parameters ==========
    mutable class FixedSzOperator* fixed_sz_op = nullptr;  // If using fixed-Sz, pointer to operator for embedding
    
    // ========== ARPACK Advanced Options ==========
    // Used when method == ARPACK_ADVANCED
    // These mirror (a subset of) detail_arpack::ArpackAdvancedOptions
    bool arpack_advanced_verbose = false;
    std::string arpack_which = "SR";                            // SR=Smallest Real (ground state), LR=Largest Real, SM/LM=by magnitude
    int64_t arpack_ncv = -1;                                        // Number of Lanczos vectors
    uint64_t arpack_max_restarts = 2;                                // Maximum number of restarts
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
    uint64_t arpack_inner_max_iter = 300;                            // Maximum inner iterations
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
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method,
    const EDParameters& params
);

// Helper functions
namespace ed_internal {
    void process_thermal_correlations(
        const EDParameters& params,
        uint64_t hilbert_space_dim
    );
    
    Operator load_hamiltonian_from_files(
        const std::string& interaction_file,
        const std::string& single_site_file,
        const std::string& counterterm_file,
        const std::string& three_body_file,
        uint64_t num_sites,
        float spin_length,
        DiagonalizationMethod method,
        HamiltonianFileFormat format
    );
    
    std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
        Operator& hamiltonian
    );
    
    EDResults diagonalize_symmetry_block(
        Eigen::SparseMatrix<Complex>& block_matrix,
        uint64_t block_dim,
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
        uint64_t block_dim,
        uint64_t block_start_dim,
        uint64_t block_idx,
        uint64_t num_sites
    );
    
    void transform_and_save_eigenvectors(
        const std::string& block_output_dir,
        const std::string& main_output_dir,
        Operator& hamiltonian,
        const std::string& directory,
        const std::vector<double>& eigenvalues,
        uint64_t block_dim,
        uint64_t block_start_dim,
        uint64_t block_idx,
        uint64_t num_sites
    );
    
    struct GroundStateSectorInfo {
        uint64_t target_block;
        double min_energy;
        double max_energy;
    };
    
    GroundStateSectorInfo find_ground_state_sector(
        const std::vector<int>& block_sizes,
        const std::string& directory,
        Operator& hamiltonian,
        const EDParameters& params,
        bool use_hdf5
    );
    
    void setup_symmetry_basis(
        const std::string& directory,
        Operator& hamiltonian,
        bool use_hdf5
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
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters()
) {
    EDResults results;
    
    // Initialize output directory if needed
    if (!params.output_dir.empty()) {
        std::string cmd = "mkdir -p " + params.output_dir;
        safe_system_call(cmd);
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
            
        case DiagonalizationMethod::IMPLICIT_RESTART_LANCZOS:
            std::cout << "Using Implicitly Restarted Lanczos method" << std::endl;
            implicitly_restarted_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                       params.num_eigenvalues, params.tolerance, 
                                       results.eigenvalues, params.output_dir, 
                                       params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::THICK_RESTART_LANCZOS:
            std::cout << "Using Thick Restart Lanczos method" << std::endl;
            thick_restart_lanczos(H, hilbert_space_dim, params.max_iterations, 
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
                            params.measure_spin, params.sublattice_size, params.num_sites,
                            params.fixed_sz_op,
                            params.continue_quenching,
                            params.continue_sample,
                            params.continue_beta); 
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
                params.num_sites,       // number of sites
                params.fixed_sz_op      // Fixed-Sz operator for embedding
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
            
        case DiagonalizationMethod::CHEBYSHEV_FILTERED:
            std::cout << "Using Chebyshev filtered Lanczos method" << std::endl;
            if (params.target_lower != 0.0 || params.target_upper != 0.0) {
                std::cout << "  Target energy range: [" << params.target_lower 
                          << ", " << params.target_upper << "]" << std::endl;
            } else {
                std::cout << "  Auto-detecting spectral range" << std::endl;
            }
            chebyshev_filtered_lanczos(H, hilbert_space_dim, 
                                     params.max_iterations, params.num_eigenvalues, 
                                     params.tolerance, results.eigenvalues, 
                                     params.output_dir, params.compute_eigenvectors,
                                     params.target_lower, params.target_upper);
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
            uint64_t info = arpack_eigs_advanced(H, hilbert_space_dim, opts,
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
            std::cout << "Using Finite Temperature Lanczos Method (FTLM)" << std::endl;
            {
                // Setup FTLM parameters
                FTLMParameters ftlm_params;
                ftlm_params.krylov_dim = params.ftlm_krylov_dim;
                ftlm_params.num_samples = params.num_samples;
                ftlm_params.max_iterations = params.max_iterations;
                ftlm_params.tolerance = params.tolerance;
                ftlm_params.full_reorthogonalization = params.ftlm_full_reorth;
                ftlm_params.reorth_frequency = params.ftlm_reorth_freq;
                ftlm_params.random_seed = params.ftlm_seed;
                ftlm_params.store_intermediate = params.ftlm_store_samples;
                ftlm_params.compute_error_bars = params.ftlm_error_bars;
                
                // Run FTLM
                FTLMResults ftlm_results = finite_temperature_lanczos(
                    H, hilbert_space_dim, ftlm_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    params.output_dir
                );
                
                // Store results (both thermodynamics and full FTLM data)
                results.thermo_data = ftlm_results.thermo_data;
                results.ftlm_results = ftlm_results;  // Store for sector combination
                
                // Store ground state estimate as eigenvalue
                if (ftlm_results.ground_state_estimate != 0.0) {
                    results.eigenvalues.push_back(ftlm_results.ground_state_estimate);
                }
                
                // Save FTLM results to file
                if (!params.output_dir.empty()) {
                    std::string ftlm_dir = params.output_dir + "/thermo";
                    safe_system_call("mkdir -p " + ftlm_dir);
                    save_ftlm_results(ftlm_results, ftlm_dir + "/ftlm_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::LTLM:
            std::cout << "Using Low Temperature Lanczos Method (LTLM)" << std::endl;
            {
                if (params.use_hybrid_method) {
                    // Deprecated: use method=HYBRID instead
                    std::cerr << "Warning: use_hybrid_method flag is deprecated. Use --method=HYBRID instead.\n";
                    std::cerr << "Falling back to standard LTLM.\n";
                }
                
                // Standard LTLM
                // Setup LTLM parameters
                LTLMParameters ltlm_params;
                ltlm_params.krylov_dim = params.ltlm_krylov_dim;
                ltlm_params.ground_state_krylov = params.ltlm_ground_krylov;
                ltlm_params.num_samples = 1;  // LTLM typically uses 1 sample
                ltlm_params.max_iterations = params.max_iterations;
                ltlm_params.tolerance = params.tolerance;
                ltlm_params.full_reorthogonalization = params.ltlm_full_reorth;
                ltlm_params.reorth_frequency = params.ltlm_reorth_freq;
                ltlm_params.random_seed = params.ltlm_seed;
                ltlm_params.store_intermediate = params.ltlm_store_data;
                ltlm_params.compute_error_bars = false;  // Not needed for single sample
                
                // Run LTLM
                LTLMResults ltlm_results = low_temperature_lanczos(
                    H, hilbert_space_dim, ltlm_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    nullptr, params.output_dir
                );
                
                // Store results
                results.thermo_data = ltlm_results.thermo_data;
                results.eigenvalues.push_back(ltlm_results.ground_state_energy);
                
                // Save LTLM results to file
                if (!params.output_dir.empty()) {
                    std::string ltlm_dir = params.output_dir + "/thermo";
                    safe_system_call("mkdir -p " + ltlm_dir);
                    save_ltlm_results(ltlm_results, ltlm_dir + "/ltlm_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::HYBRID:
            std::cout << "Using Hybrid Thermal Method (LTLM+FTLM)" << std::endl;
            {
                // Setup Hybrid Thermal parameters
                HybridThermalParameters hybrid_params;
                
                // Temperature and crossover settings
                hybrid_params.crossover_temperature = params.hybrid_crossover;
                hybrid_params.auto_crossover = params.hybrid_auto_crossover;
                
                // LTLM parameters (low temperature)
                hybrid_params.ltlm_krylov_dim = params.ltlm_krylov_dim;
                hybrid_params.ltlm_ground_krylov = params.ltlm_ground_krylov;
                hybrid_params.ltlm_full_reorth = params.ltlm_full_reorth;
                hybrid_params.ltlm_reorth_freq = params.ltlm_reorth_freq;
                hybrid_params.ltlm_seed = params.ltlm_seed;
                hybrid_params.ltlm_store_data = params.ltlm_store_data;
                
                // FTLM parameters (high temperature)
                hybrid_params.ftlm_num_samples = params.num_samples;
                hybrid_params.ftlm_krylov_dim = params.ftlm_krylov_dim;
                hybrid_params.ftlm_full_reorth = params.ftlm_full_reorth;
                hybrid_params.ftlm_reorth_freq = params.ftlm_reorth_freq;
                hybrid_params.ftlm_seed = params.ftlm_seed;
                hybrid_params.ftlm_store_samples = params.ftlm_store_samples;
                hybrid_params.ftlm_error_bars = params.ftlm_error_bars;
                
                // General parameters
                hybrid_params.max_iterations = params.max_iterations;
                hybrid_params.tolerance = params.tolerance;
                
                // Run hybrid thermal method
                HybridThermalResults hybrid_results = hybrid_thermal_method(
                    H, hilbert_space_dim, hybrid_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    params.output_dir
                );
                
                // Store results
                results.thermo_data = hybrid_results.thermo_data;
                results.eigenvalues.push_back(hybrid_results.ground_state_energy);
                
                // Save results to file
                if (!params.output_dir.empty()) {
                    std::string thermo_dir = params.output_dir + "/thermo";
                    safe_system_call("mkdir -p " + thermo_dir);
                    save_hybrid_thermal_results(hybrid_results, thermo_dir + "/hybrid_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::LANCZOS_GPU:
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
        case DiagonalizationMethod::DAVIDSON_GPU:
        case DiagonalizationMethod::LOBPCG_GPU:
        case DiagonalizationMethod::mTPQ_GPU:
        case DiagonalizationMethod::cTPQ_GPU:
        case DiagonalizationMethod::FTLM_GPU:
        case DiagonalizationMethod::FTLM_GPU_FIXED_SZ:
            // These should be handled in exact_diagonalization_from_files
            // If we reach here, it means they were called incorrectly
            std::cerr << "Error: GPU methods must be called via exact_diagonalization_from_files" << std::endl;
            std::cerr << "Use: ED <directory> --method=<GPU_METHOD>" << std::endl;
            throw std::runtime_error("GPU methods require file-based interface");
            break;

        default:
            std::cerr << "Unknown diagonalization method selected" << std::endl;
            break;
    }
    
    if (params.compute_eigenvectors) {
        std::cout << "Eigenvectors computed and saved to " << params.output_dir << std::endl;
    }

    // Calculate thermal observables if requested
    // if (params.calc_observables) {
    //     ed_internal::process_thermal_correlations(params, hilbert_space_dim);
    // }

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
    uint64_t hilbert_space_dim
) {
    std::cout << "Calculating custom observables..." << std::endl;
    std::cout << "Calculating thermal expectation values for correlation operators..." << std::endl;

    // Create output directory for thermal correlation results
    std::string output_correlations_dir = params.output_dir + "/thermal_correlations";
    std::string cmd_mkdir = "mkdir -p " + output_correlations_dir;
    safe_system_call(cmd_mkdir);

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
            safe_system_call(find_command);
            
            // Read the list of files
            std::ifstream file_list(temp_list_file);
            if (!file_list.is_open()) continue;
            
            std::string correlation_file;
            uint64_t file_count = 0;

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
                    uint64_t numLines;
                    std::string m;
                    iss >> m >> numLines;
                    // std::cout << "Number of lines: " << numLines << std::endl;
                    
                    // Skip the next 3 lines (separators/headers)
                    for (uint64_t i = 0; i < 3; ++i) {
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
                    uint64_t lineCount = 0;
                    while (std::getline(file, line) && lineCount < numLines) {
                        Operator correlation_op(params.num_sites, params.spin_length);
                        std::istringstream lineStream(line);
                        uint64_t Op1, indx1, Op2, indx2;
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
                        auto apply_correlation_op = [&correlation_op](const Complex* in, Complex* out, uint64_t n) {
                            correlation_op.apply(in, out, n);
                        };
                        

                        // Calculate thermal expectations at temperature points
                        uint64_t num_temps = std::min(params.num_temp_bins, static_cast<uint64_t>(20));
                        double log_temp_min = std::log(params.temp_min);
                        double log_temp_max = std::log(params.temp_max);
                        double log_temp_step = (log_temp_max - log_temp_min) / std::max(1, static_cast<int>(num_temps - 1));

                        for (uint64_t i = 0; i < num_temps; i++) {
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
        int64_t n_up = (config.n_up >= 0) ? config.n_up : config.num_sites / 2;
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
    const std::string& counterterm_file,
    const std::string& three_body_file,
    uint64_t num_sites,
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
            // Load three-body terms if provided
            if (!three_body_file.empty() && std::filesystem::exists(three_body_file)) {
                std::cout << "Loading three-body terms from: " << three_body_file << std::endl;
                hamiltonian.loadThreeBodyTerm(three_body_file);
            }
            // COUNTERTERM DISABLED
            // if (!counterterm_file.empty()){
            //     hamiltonian.loadCounterTerm(counterterm_file);
            // }
            // Build sparse matrix (except for full diagonalization)
            // if (method == DiagonalizationMethod::FULL) {
            //     hamiltonian.buildSparseMatrix();
            // }
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
    return [&hamiltonian](const Complex* in, Complex* out, uint64_t n) {
        // Directly use pointer-based apply to avoid temporary vector allocations
        hamiltonian.apply(in, out, n);
    };
}

/**
 * @brief Diagonalize a single symmetry block
 */
EDResults diagonalize_symmetry_block(
    Eigen::SparseMatrix<Complex>& block_matrix,
    uint64_t block_dim,
    DiagonalizationMethod method,
    const EDParameters& params,
    bool is_target_block,
    double large_value_override
) {
    // If a GPU method was requested for a single symmetry block, attempt
    // to convert the Eigen sparse block into CSR and dispatch to the
    // GPU implementation. If CUDA is not available, or the conversion
    // / upload fails, gracefully fall back to the CPU Lanczos path.
#ifdef WITH_CUDA
    if (method == DiagonalizationMethod::LANCZOS_GPU ||
        method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ ||
        method == DiagonalizationMethod::DAVIDSON_GPU ||
        method == DiagonalizationMethod::LOBPCG_GPU) {
        if (!GPUEDWrapper::isGPUAvailable()) {
            std::cerr << "Warning: No CUDA-capable GPU found. Falling back to CPU for this block (dim="
                      << block_dim << ").\n";
            // Map GPU methods to appropriate CPU methods
            if (method == DiagonalizationMethod::DAVIDSON_GPU) {
                method = DiagonalizationMethod::DAVIDSON;
            } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
                method = DiagonalizationMethod::LOBPCG;
            } else {
                method = DiagonalizationMethod::LANCZOS;
            }
        } else {
            // Convert Eigen sparse matrix (which may be column-major) to
            // row-major CSR arrays expected by the GPU upload path.
            try {
                uint64_t N = block_dim;
                uint64_t nnz = static_cast<int>(block_matrix.nonZeros());

                std::vector<std::vector<std::pair<int, std::complex<double>>>> rows(N);
                rows.assign(N, {});

                for (uint64_t k = 0; k < block_matrix.outerSize(); ++k) {
                    for (Eigen::SparseMatrix<Complex>::InnerIterator it(block_matrix, k); it; ++it) {
                        uint64_t r = it.row();
                        uint64_t c = it.col();
                        std::complex<double> v = it.value();
                        if (r < 0 || r >= N || c < 0 || c >= N) continue;
                        rows[r].emplace_back(c, v);
                    }
                }

                std::vector<int> row_ptr(N + 1, 0);
                std::vector<int> col_ind; col_ind.reserve(nnz);
                std::vector<std::complex<double>> values; values.reserve(nnz);

                uint64_t counter = 0;
                for (uint64_t i = 0; i < N; ++i) {
                    row_ptr[i] = counter;
                    // Optionally, keep columns sorted for cuSPARSE efficiency
                    auto &row = rows[i];
                    if (row.size() > 1) std::sort(row.begin(), row.end(), [](const auto &a, const auto &b){ return a.first < b.first; });
                    for (const auto &p : row) {
                        col_ind.push_back(p.first);
                        values.push_back(p.second);
                        ++counter;
                    }
                }
                row_ptr[N] = counter;

                // Create GPU operator and run appropriate method
                void* gpu_op = GPUEDWrapper::createGPUOperatorFromCSR(params.num_sites, N, row_ptr, col_ind, values);
                if (!gpu_op) {
                    std::cerr << "Warning: Failed to create GPU operator from CSR. Falling back to CPU for this block (dim="
                              << block_dim << ").\n";
                    if (method == DiagonalizationMethod::DAVIDSON_GPU) {
                        method = DiagonalizationMethod::DAVIDSON;
                    } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
                        method = DiagonalizationMethod::LOBPCG;
                    } else {
                        method = DiagonalizationMethod::LANCZOS;
                    }
                } else {
                    std::vector<double> eigenvalues;
                    GPUEDWrapper::printGPUInfo();
                    
                    if (method == DiagonalizationMethod::DAVIDSON_GPU) {
                        GPUEDWrapper::runGPUDavidson(
                            gpu_op, N,
                            std::min(params.num_eigenvalues, block_dim),
                            params.max_iterations,
                            params.max_subspace,
                            params.tolerance,
                            eigenvalues,
                            params.output_dir,
                            params.compute_eigenvectors
                        );
                    } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
                        GPUEDWrapper::runGPULOBPCG(
                            gpu_op, N,
                            std::min(params.num_eigenvalues, block_dim),
                            params.max_iterations,
                            params.tolerance,
                            eigenvalues,
                            params.output_dir,
                            params.compute_eigenvectors
                        );
                    } else {
                        // LANCZOS_GPU or LANCZOS_GPU_FIXED_SZ
                        GPUEDWrapper::runGPULanczos(
                            gpu_op, N,
                            params.max_iterations,
                            std::min(params.num_eigenvalues, block_dim),
                            params.tolerance,
                            eigenvalues,
                            params.output_dir,
                            params.compute_eigenvectors
                        );
                    }

                    // Fill results and clean up
                    EDResults results;
                    results.eigenvectors_computed = params.compute_eigenvectors;
                    results.eigenvalues = eigenvalues;
                    GPUEDWrapper::destroyGPUOperator(gpu_op);

                    return results;
                }
            } catch (const std::exception &e) {
                std::cerr << "Warning: Exception during GPU dispatch for symmetry block: " << e.what()
                          << "\nFalling back to CPU for this block (dim=" << block_dim << ").\n";
                if (method == DiagonalizationMethod::DAVIDSON_GPU) {
                    method = DiagonalizationMethod::DAVIDSON;
                } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
                    method = DiagonalizationMethod::LOBPCG;
                } else {
                    method = DiagonalizationMethod::LANCZOS;
                }
            }
        }
    }
    
    // TPQ methods shouldn't be used for individual blocks - fall back to Lanczos
    if (method == DiagonalizationMethod::mTPQ_GPU || method == DiagonalizationMethod::cTPQ_GPU) {
        std::cerr << "Warning: GPU TPQ methods not applicable to individual symmetry blocks.\n";
        std::cerr << "         Falling back to CPU Lanczos for this block (dim=" << block_dim << ").\n";
        method = DiagonalizationMethod::LANCZOS;
    }
#else
    if (method == DiagonalizationMethod::LANCZOS_GPU ||
        method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ ||
        method == DiagonalizationMethod::DAVIDSON_GPU ||
        method == DiagonalizationMethod::LOBPCG_GPU ||
        method == DiagonalizationMethod::mTPQ_GPU ||
        method == DiagonalizationMethod::cTPQ_GPU) {
        std::cerr << "Warning: GPU methods requested but CUDA not available.\n";
        std::cerr << "         Falling back to CPU for this block (dim=" << block_dim << ").\n";
        if (method == DiagonalizationMethod::DAVIDSON_GPU) {
            method = DiagonalizationMethod::DAVIDSON;
        } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
            method = DiagonalizationMethod::LOBPCG;
        } else {
            method = DiagonalizationMethod::LANCZOS;
        }
    }
#endif

    // Define matrix-vector product for this block
    std::function<void(const Complex*, Complex*, int)> apply_block =
        [block_matrix](const Complex* in, Complex* out, uint64_t n) {
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
    uint64_t block_dim,
    uint64_t block_start_dim,
    uint64_t block_idx,
    uint64_t num_sites
) {
    std::cout << "Transforming TPQ states for block " << block_idx << std::endl;
    
    // Find all TPQ state files
    std::string temp_list_file = main_output_dir + "/tpq_state_files_" + std::to_string(block_idx) + ".txt";
    std::string find_command = "find \"" + block_output_dir + "\" -name \"tpq_state_*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
    safe_system_call(find_command);
    
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
        for (uint64_t i = 0; i < block_dim; ++i) {
            std::vector<Complex> basis_vector = hamiltonian.read_sym_basis(i + block_start_dim, directory);
            full_tpq_state += basis_vector * block_tpq_state[i];
        }

        // Save transformed state in binary format (same as original TPQ saves)
        std::string transformed_file = main_output_dir + "/tpq_state_" + sample_str + "_beta=" + beta_str + ".dat";
        std::ofstream out_file(transformed_file, std::ios::binary);
        if (!out_file.is_open()) {
            std::cerr << "Warning: Could not open file for writing: " << transformed_file << std::endl;
            continue;
        }
        
        size_t full_dim = full_tpq_state.size();
        out_file.write(reinterpret_cast<const char*>(&full_dim), sizeof(size_t));
        out_file.write(reinterpret_cast<const char*>(full_tpq_state.data()), full_dim * sizeof(Complex));
        out_file.close();
        
        std::cout << "  Saved transformed TPQ state (dim: " << block_dim << " -> " << full_dim << ")" << std::endl;
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
    uint64_t block_dim,
    uint64_t block_start_dim,
    uint64_t block_idx,
    uint64_t num_sites
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
        uint64_t num_entries = 0;
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
    const EDParameters& params,
    bool use_hdf5 = true
) {
    std::cout << "\n=== Quick Scan: Finding Target Sector ===" << std::endl;
    std::cout << "Performing lightweight scan of all sectors to identify which contains lowest eigenvalues..." << std::endl;
    
    GroundStateSectorInfo info;
    info.min_energy = std::numeric_limits<double>::max();
    info.max_energy = std::numeric_limits<double>::lowest();
    info.target_block = 0;
    
    for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
        uint64_t block_dim = block_sizes[block_idx];
        
        if (block_dim == 0) {
            std::cout << "  Block " << block_idx << ": empty (skipping)" << std::endl;
            continue;
        }
        
        std::cout << "  Scanning block " << block_idx << " (dim=" << block_dim << ")..." << std::flush;
        
        // Load block Hamiltonian
        Eigen::SparseMatrix<Complex> block_matrix;
        if (use_hdf5) {
            block_matrix = hamiltonian.loadSymmetrizedBlockHDF5(directory, block_idx);
        } else {
            std::string block_file = directory + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
            block_matrix = hamiltonian.loadSymmetrizedBlock(block_file);
        }
        
        // Quick Lanczos scan with minimal eigenvalues
        EDParameters scan_params = params;
        scan_params.num_eigenvalues = std::min(static_cast<uint64_t>(5), block_dim);  // Only need a few
        scan_params.max_iterations = scan_params.num_eigenvalues * 3 + 15;  // Minimal iterations
        scan_params.compute_eigenvectors = false;
        scan_params.calc_observables = false;
        scan_params.measure_spin = false;
        
        EDResults block_results = diagonalize_symmetry_block(
            block_matrix, block_dim, DiagonalizationMethod::LANCZOS, scan_params
        );
        
        if (block_results.eigenvalues.empty()) {
            std::cout << " no eigenvalues found" << std::endl;
            continue;
        }
        
        double min_block = *std::min_element(block_results.eigenvalues.begin(), block_results.eigenvalues.end());
        double max_block = *std::max_element(block_results.eigenvalues.begin(), block_results.eigenvalues.end());
        std::cout << " E_min=" << std::fixed << std::setprecision(6) << min_block 
                  << ", E_max=" << max_block << std::endl;
        
        if (min_block < info.min_energy) {
            info.min_energy = min_block;
            info.max_energy = max_block;
            info.target_block = block_idx;
        }
    }
    
    std::cout << "\n=== Target Sector Identified ===" << std::endl;
    std::cout << "  Block index: " << info.target_block << std::endl;
    std::cout << "  Block dimension: " << block_sizes[info.target_block] << std::endl;
    std::cout << "  Estimated E_min: " << std::fixed << std::setprecision(6) << info.min_energy << std::endl;
    std::cout << "  Estimated E_max: " << info.max_energy << std::endl;
    std::cout << "=================================\n" << std::endl;
    
    return info;
}

/**
 * @brief Setup symmetrized basis (generate or load)
 */
void setup_symmetry_basis(
    const std::string& directory,
    Operator& hamiltonian,
    bool use_hdf5 = true  // Use HDF5 by default for better file management
) {
    std::string hdf5_file = directory + "/symmetry_data.h5";
    std::string sym_basis_dir = directory + "/sym_basis";
    std::string sym_blocks_dir = directory + "/sym_blocks";
    std::string block_sizes_file = sym_basis_dir + "/sym_block_sizes.txt";
    
    struct stat buffer;
    
    if (use_hdf5) {
        // HDF5 workflow
        bool hdf5_exists = HDF5SymmetryIO::fileExists(hdf5_file);
        
        if (!hdf5_exists) {
            std::cout << "Symmetrized basis (HDF5) not found. Generating..." << std::endl;
            hamiltonian.generateSymmetrizedBasisHDF5(directory);
            hamiltonian.buildAndSaveSymmetrizedBlocksHDF5(directory);
        } else {
            std::cout << "Using existing symmetrized basis from HDF5: " << hdf5_file << std::endl;
            // Load block sizes from HDF5
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            hamiltonian.symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
    } else {
        // Legacy text file workflow
        bool sym_basis_exists = (stat(block_sizes_file.c_str(), &buffer) == 0);
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
    uint64_t num_sites,
    float spin_length,
    int64_t n_up,
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
    uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
    uint64_t full_dim = 1ULL << num_sites;
    
    std::cout << "Hilbert space dimension:" << std::endl;
    std::cout << "  Full space: " << full_dim << std::endl;
    std::cout << "  Fixed Sz:   " << fixed_sz_dim << std::endl;
    std::cout << "  Reduction:  " << (double)full_dim / fixed_sz_dim << "x" << std::endl;
    
    // Check if GPU method requested
    bool is_gpu_method = (method == DiagonalizationMethod::DAVIDSON_GPU ||
                          method == DiagonalizationMethod::LOBPCG_GPU ||
                          method == DiagonalizationMethod::LANCZOS_GPU ||
                          method == DiagonalizationMethod::mTPQ_GPU ||
                          method == DiagonalizationMethod::cTPQ_GPU ||
                          method == DiagonalizationMethod::FTLM_GPU_FIXED_SZ);
    
    EDResults results;
    
    if (is_gpu_method) {
#ifdef WITH_CUDA
        std::cout << "\n=== GPU Fixed-Sz Diagonalization ===" << std::endl;
        
        // Prepare interactions and single-site operators
        std::vector<std::tuple<int, int, char, char, double>> gpu_interactions;
        std::vector<std::tuple<int, char, double>> gpu_single_site_ops;
        
        // Load from files
        std::ifstream inter_file(interaction_file);
        if (inter_file.is_open()) {
            std::string line;
            std::getline(inter_file, line);
            std::getline(inter_file, line);
            std::istringstream iss(line);
            uint64_t numLines;
            std::string m;
            iss >> m >> numLines;
            
            for (uint64_t i = 0; i < 3; ++i) std::getline(inter_file, line);
            
            uint64_t lineCount = 0;
            while (std::getline(inter_file, line) && lineCount < numLines) {
                std::istringstream lineStream(line);
                uint64_t Op_i, indx_i, Op_j, indx_j;
                double E, F;
                
                if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
                
                // File operator codes: 0=S+, 1=S-, 2=Sz
                // Map to chars: '+'=S+, '-'=S-, 'z'=Sz
                auto mapOp = [](uint64_t op) -> char {
                    if (op == 0) return '+';  // S+
                    if (op == 1) return '-';  // S-
                    return 'z';  // Sz
                };
                
                gpu_interactions.push_back(std::make_tuple(indx_i, indx_j, mapOp(Op_i), mapOp(Op_j), E));
                lineCount++;
            }
        }
        
        // Load single-site terms if present
        if (!single_site_file.empty()) {
            std::ifstream ss_file(single_site_file);
            if (ss_file.is_open()) {
                std::string line;
                std::getline(ss_file, line);
                std::getline(ss_file, line);
                std::istringstream iss(line);
                uint64_t numLines;
                std::string m;
                iss >> m >> numLines;
                
                for (uint64_t i = 0; i < 3; ++i) std::getline(ss_file, line);
                
                uint64_t lineCount = 0;
                while (std::getline(ss_file, line) && lineCount < numLines) {
                    std::istringstream lineStream(line);
                    uint64_t Op_i, indx_i;
                    double E, F;
                    
                    if (!(lineStream >> Op_i >> indx_i >> E >> F)) continue;
                    
                    // File operator codes: 0=S+, 1=S-, 2=Sz
                    // Map to chars: '+'=S+, '-'=S-, 'z'=Sz
                    auto mapOp = [](uint64_t op) -> char {
                        if (op == 0) return '+';  // S+
                        if (op == 1) return '-';  // S-
                        return 'z';  // Sz
                    };
                    
                    gpu_single_site_ops.push_back(std::make_tuple(indx_i, mapOp(Op_i), E));
                    lineCount++;
                }
            }
        }
        
        // Create GPU operator
        void* gpu_op_handle = GPUEDWrapper::createGPUFixedSzOperatorDirect(
            num_sites, n_up, spin_length,
            gpu_interactions, gpu_single_site_ops);
        
        std::cout << "Loaded " << gpu_interactions.size() << " interactions and " 
                  << gpu_single_site_ops.size() << " single-site terms\n";
        
        // Run appropriate GPU method
        std::vector<double> eigenvalues;
        
        if (method == DiagonalizationMethod::DAVIDSON_GPU || method == DiagonalizationMethod::LOBPCG_GPU) {
            GPUEDWrapper::runGPUDavidsonFixedSz(
                gpu_op_handle, n_up,
                params.num_eigenvalues,
                params.max_iterations,
                params.max_subspace,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::LANCZOS_GPU) {
            GPUEDWrapper::runGPULanczosFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_eigenvalues,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::mTPQ_GPU) {
            GPUEDWrapper::runGPUMicrocanonicalTPQFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_samples,
                params.num_measure_freq,
                eigenvalues,
                params.output_dir,
                params.large_value,
                params.continue_quenching,
                params.continue_sample,
                params.continue_beta);
        } else if (method == DiagonalizationMethod::cTPQ_GPU) {
            GPUEDWrapper::runGPUCanonicalTPQFixedSz(
                gpu_op_handle, n_up,
                params.temp_max,  // beta_max
                params.num_samples,
                params.num_measure_freq,
                eigenvalues,
                params.output_dir,
                params.delta_tau,  // delta_beta
                params.num_order);  // taylor_order
        }
        
        results.eigenvalues = eigenvalues;
        
        // Note: GPU TPQ states are now automatically transformed during save (via saveTPQState)
        // No post-processing transformation needed
        if (method == DiagonalizationMethod::mTPQ_GPU || method == DiagonalizationMethod::cTPQ_GPU) {
            std::cout << "\nGPU TPQ states were automatically transformed to full Hilbert space during save." << std::endl;
        }
        
        // Cleanup
        GPUEDWrapper::destroyGPUOperator(gpu_op_handle);
        
        std::cout << "GPU diagonalization complete\n";
#else
        throw std::runtime_error("GPU methods require CUDA support (compile with -DWITH_CUDA=ON)");
#endif
    } else {

        
        // Create apply function
        auto apply_hamiltonian = [&hamiltonian, fixed_sz_dim](const Complex* in, Complex* out, uint64_t n) {
            if (n != fixed_sz_dim) {
                throw std::runtime_error("Dimension mismatch in fixed Sz apply");
            }
            hamiltonian.apply(in, out, n);
        };
        
        // Set the fixed_sz_op in params so TPQ can transform states before saving
        params.fixed_sz_op = &hamiltonian;
        
        // Perform diagonalization
        std::cout << "\nDiagonalizing..." << std::endl;
        results = exact_diagonalization_core(apply_hamiltonian, fixed_sz_dim, method, params);
    }

    // Check if this is a TPQ method
    bool is_tpq_method = (method == DiagonalizationMethod::mTPQ || 
                          method == DiagonalizationMethod::mTPQ_CUDA || 
                          method == DiagonalizationMethod::cTPQ);

    // Transform eigenvectors from fixed-Sz basis to full basis
    // Note: TPQ states are now transformed during save, so no post-processing needed for them
    if (!params.output_dir.empty() && params.compute_eigenvectors) {
        std::cout << "Transforming eigenvectors from fixed-Sz basis to full Hilbert space..." << std::endl;
        std::cout << "  Fixed Sz dim: " << fixed_sz_dim << ", Full dim: " << full_dim << std::endl;
        std::cout << "  Output directory: " << params.output_dir << std::endl;
        
        // Transform eigenvectors if computed
        if (params.compute_eigenvectors) {
            std::cout << "  Number of eigenvectors: " << results.eigenvalues.size() << std::endl;
            for (size_t i = 0; i < results.eigenvalues.size(); ++i) {
                std::string eigvec_file = params.output_dir + "/eigenvectors/eigenvector_" + std::to_string(i) + ".dat";
                std::cout << "  Processing eigenvector file: " << eigvec_file << std::endl;
                std::ifstream infile(eigvec_file, std::ios::binary);
                
                if (infile.is_open()) {
                    // Read eigenvector in fixed-Sz basis
                    std::vector<Complex> fixed_sz_vec(fixed_sz_dim);
                    infile.read(reinterpret_cast<char*>(fixed_sz_vec.data()), fixed_sz_dim * sizeof(Complex));
                    infile.close();
                    
                    // Transform to full basis
                    std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                    
                    // Overwrite file with full-space eigenvector
                    std::ofstream outfile(eigvec_file, std::ios::binary);
                    outfile.write(reinterpret_cast<const char*>(full_vec.data()), full_dim * sizeof(Complex));
                    outfile.close();
                    
                    std::cout << "  Transformed eigenvector " << i << " to full space (dim: " 
                              << fixed_sz_dim << " -> " << full_dim << ")" << std::endl;
                }
            }
            std::cout << "All eigenvectors successfully transformed to full Hilbert space." << std::endl;
        }
        
        // Note: TPQ states are now automatically transformed during save (via save_tpq_state)
        // No post-processing transformation needed for TPQ states
        if (is_tpq_method) {
            std::cout << "\nTPQ states were automatically transformed to full Hilbert space during save." << std::endl;
        }
    }

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
 * @param counterterm_file Path to counter term file (optional)
 * @param three_body_file Path to three-body interaction file (e.g., ThreeBodyG.dat, optional)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @return EDResults containing eigenvalues and metadata
 */
EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD
) {
    std::cerr << "[DEBUG] exact_diagonalization_from_files: num_sites=" << params.num_sites 
              << ", method=" << static_cast<int>(method) << std::endl;
    
    // Handle GPU methods separately (they don't need CPU Operator)
#ifdef WITH_CUDA
    if (method == DiagonalizationMethod::LANCZOS_GPU || 
        method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ ||
        method == DiagonalizationMethod::DAVIDSON_GPU ||
        method == DiagonalizationMethod::LOBPCG_GPU ||
        method == DiagonalizationMethod::mTPQ_GPU ||
        method == DiagonalizationMethod::cTPQ_GPU ||
        method == DiagonalizationMethod::FTLM_GPU ||
        method == DiagonalizationMethod::FTLM_GPU_FIXED_SZ) {
        
        std::cout << "Running GPU-accelerated algorithm..." << std::endl;
        
        // Check if GPU is available
        if (!GPUEDWrapper::isGPUAvailable()) {
            std::cerr << "Error: No CUDA-capable GPU found!" << std::endl;
            throw std::runtime_error("GPU not available");
        }
        
        GPUEDWrapper::printGPUInfo();
        
        EDResults results;
        uint64_t hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
        
        if (method == DiagonalizationMethod::LANCZOS_GPU) {
            // Check if files exist
            if (!std::filesystem::exists(interaction_file)) {
                std::cerr << "Error: " << interaction_file << " not found!" << std::endl;
                throw std::runtime_error("InterAll.dat file not found");
            }
            
            // Create GPU operator from files
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            // Run GPU Lanczos
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPULanczos(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_eigenvalues,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            // Store results
            results.eigenvalues = eigenvalues;
            
            // Clean up
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Lanczos completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ) {
            std::cerr << "Error: LANCZOS_GPU_FIXED_SZ file interface not yet implemented." << std::endl;
            std::cerr << "Please use GPU_FILE_TEST for now." << std::endl;
            throw std::runtime_error("Fixed Sz GPU method not yet integrated with file interface");
            
        } else if (method == DiagonalizationMethod::DAVIDSON_GPU) {
            std::cout << "Running GPU Davidson method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUDavidson(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.max_subspace,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Davidson completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
            std::cout << "Running GPU LOBPCG method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPULOBPCG(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU LOBPCG completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::mTPQ_GPU) {
            std::cout << "Running GPU microcanonical TPQ..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUMicrocanonicalTPQ(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_samples,
                params.num_measure_freq,
                eigenvalues,
                params.output_dir,
                params.large_value,
                params.continue_quenching,
                params.continue_sample,
                params.continue_beta
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU mTPQ completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::cTPQ_GPU) {
            std::cout << "Running GPU canonical TPQ..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUCanonicalTPQ(
                gpu_op,
                hilbert_space_dim,
                params.temp_max,
                params.num_samples,
                params.num_measure_freq,
                eigenvalues,
                params.output_dir,
                params.delta_tau,
                params.num_order
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU cTPQ completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::FTLM_GPU) {
            std::cout << "Running GPU Finite Temperature Lanczos Method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            GPUEDWrapper::runGPUFTLM(
                gpu_op,
                hilbert_space_dim,
                params.ftlm_krylov_dim,
                params.num_samples,
                params.temp_min,
                params.temp_max,
                params.num_temp_bins,
                params.tolerance,
                params.output_dir,
                params.ftlm_full_reorth,
                params.ftlm_reorth_freq,
                params.ftlm_seed
            );
            
            // FTLM doesn't return eigenvalues in the traditional sense
            // Results are thermodynamic quantities written to files
            
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU FTLM completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::FTLM_GPU_FIXED_SZ) {
            std::cerr << "Error: FTLM_GPU_FIXED_SZ file interface not yet implemented." << std::endl;
            std::cerr << "Please use the fixed_sz wrapper function directly." << std::endl;
            throw std::runtime_error("Fixed Sz GPU FTLM not yet integrated with file interface");
        }
        
        return results;
    }
#endif
    
    // Load Hamiltonian (for CPU methods)
    Operator hamiltonian = ed_internal::load_hamiltonian_from_files(
        interaction_file, single_site_file, counterterm_file, three_body_file, 
        params.num_sites, params.spin_length, method, format
    );
    
    // Calculate Hilbert space dimension
    uint64_t hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
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
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat"
) {
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    std::string counterterm_file = directory + "/" + counterterm_filename;
    std::string three_body_file = directory + "/" + three_body_filename;
    
    // Check if counter term file exists
    struct stat buffer;
    if (stat(counterterm_file.c_str(), &buffer) != 0) {
        counterterm_file = "";  // File doesn't exist, pass empty string
    }
    
    // Check if three-body file exists
    if (stat(three_body_file.c_str(), &buffer) != 0) {
        three_body_file = "";  // File doesn't exist, pass empty string
    }
    
    // Call the file-based wrapper
    return exact_diagonalization_from_files(
        interaction_file, single_site_file, counterterm_file, three_body_file, method, params, format
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
 * @param counterterm_filename Name of counter term file (default: "CounterTerm.dat")
 * @return EDResults containing eigenvalues and metadata
 */
EDResults exact_diagonalization_from_directory_symmetrized(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat"
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
        if (!safe_system_call(cmd)) {
            std::cerr << "Warning: Automorphism finder failed" << std::endl;
        }
    } else {
        std::cout << "Using existing automorphism results from: " << automorphism_file << std::endl;
    }
    
    // ========== Step 2: Load Hamiltonian ==========
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    std::string counterterm_file = directory + "/" + counterterm_filename;
    std::string three_body_file = directory + "/" + three_body_filename;
    
    // Check if counter term file exists
    struct stat counterterm_buffer;
    bool counterterm_exists = (stat(counterterm_file.c_str(), &counterterm_buffer) == 0);
    
    // Check if three-body file exists
    struct stat three_body_buffer;
    bool three_body_exists = (stat(three_body_file.c_str(), &three_body_buffer) == 0);
    
    EDResults results;
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    Operator hamiltonian(params.num_sites, params.spin_length);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    // Load three-body terms if available
    if (three_body_exists) {
        std::cout << "Loading three-body terms from: " << three_body_file << std::endl;
        hamiltonian.loadThreeBodyTerm(three_body_file);
    }
    
    // COUNTERTERM DISABLED
    // if (counterterm_exists) {
    //     hamiltonian.loadCounterTerm(counterterm_file);
    // }
    // ========== Step 3: Setup Symmetrized Basis ==========
    bool use_hdf5 = true;  // Use HDF5 by default
    ed_internal::setup_symmetry_basis(directory, hamiltonian, use_hdf5);
    
    std::vector<int> block_sizes = hamiltonian.symmetrized_block_ham_sizes;
    std::cout << "Found " << block_sizes.size() << " symmetrized blocks with sizes: ";
    for (const auto& size : block_sizes) std::cout << size << " ";
    std::cout << std::endl;

    if (block_sizes.empty()) {
        throw std::runtime_error("No symmetrized blocks found. Failed to generate symmetrized basis.");
    }
    
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
    }
    
    // ========== Step 4: Determine if we need targeted diagonalization ==========
    bool is_tpq_method = (method == DiagonalizationMethod::mTPQ || 
                          method == DiagonalizationMethod::mTPQ_CUDA || 
                          method == DiagonalizationMethod::cTPQ);
    
    // Optimization: if we only need a few eigenvalues, find the target sector first
    bool use_targeted_diagonalization = false;
    ed_internal::GroundStateSectorInfo gs_info;
    
    // Use targeted approach if:
    // 1. TPQ method (always needs ground state sector)
    // 2. Requesting small number of eigenvalues compared to total dimension
    uint64_t total_dimension = 0;
    for (const auto& size : block_sizes) total_dimension += size;
    
    // Use targeted if requesting < 10 eigenvalues OR < 1% of total dimension
    bool small_eigenvalue_request = (params.num_eigenvalues < 10) || 
                                   (params.num_eigenvalues < total_dimension / 100);
    
    if (is_tpq_method || small_eigenvalue_request) {
        use_targeted_diagonalization = true;
        std::cout << "\n=== Using Targeted Diagonalization ===" << std::endl;
        std::cout << "Requested eigenvalues: " << params.num_eigenvalues << std::endl;
        std::cout << "Total Hilbert space dimension: " << total_dimension 
                  << " (across " << block_sizes.size() << " sectors)" << std::endl;
        std::cout << "Scanning blocks to identify target sector(s)..." << std::endl;
        
        gs_info = ed_internal::find_ground_state_sector(block_sizes, directory, hamiltonian, params, use_hdf5);
        
        std::cout << "Target sector: block " << gs_info.target_block 
                  << " (dimension: " << block_sizes[gs_info.target_block] << ")" << std::endl;
        std::cout << "Will only diagonalize this sector to save computation." << std::endl;
    } else {
        std::cout << "\n=== Using Full Diagonalization ===" << std::endl;
        std::cout << "Requested eigenvalues: " << params.num_eigenvalues << std::endl;
        std::cout << "Total dimension: " << total_dimension 
                  << " (across " << block_sizes.size() << " sectors)" << std::endl;
        std::cout << "Will diagonalize all sectors." << std::endl;
    }

    // ========== Step 5: Diagonalize Block(s) ==========
    struct EigenInfo {
        double value;
        uint64_t block_idx;
        uint64_t eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;
    
    // For FTLM: collect results from all sectors for proper combination
    bool is_ftlm = (method == DiagonalizationMethod::FTLM);
    std::vector<FTLMResults> sector_ftlm_results;
    std::vector<uint64_t> sector_dimensions;
    
    uint64_t block_start_dim = 0;
    for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
        uint64_t block_dim = block_sizes[block_idx];
        
        // Skip this block if using targeted diagonalization and it's not the target
        bool is_target_block = (use_targeted_diagonalization && 
                               gs_info.target_block == static_cast<int>(block_idx));
        
        if (use_targeted_diagonalization && !is_target_block) {
            std::cout << "Skipping block " << block_idx + 1 << "/" << block_sizes.size() 
                      << " (not the target sector)" << std::endl;
            block_start_dim += block_dim;
            continue;
        }
        
        std::cout << "Diagonalizing block " << block_idx + 1 << "/" << block_sizes.size() 
                  << " (dimension: " << block_dim << ")" << std::endl;
        
        // Load block Hamiltonian
        Eigen::SparseMatrix<Complex> block_matrix;
        if (use_hdf5) {
            block_matrix = hamiltonian.loadSymmetrizedBlockHDF5(directory, block_idx);
        } else {
            std::string block_file = directory + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
            block_matrix = hamiltonian.loadSymmetrizedBlock(block_file);
        }

        // Configure block parameters
        EDParameters block_params = params;
        // Compute at least params.num_eigenvalues for each block (up to block dimension)
        block_params.num_eigenvalues = std::min(params.num_eigenvalues, block_dim);
        if (params.compute_eigenvectors) {
            block_params.output_dir = params.output_dir + "/min_sector";
            safe_system_call("mkdir -p " + block_params.output_dir);
        }
        
        // For FTLM with multiple sectors, set output dir for individual sectors
        if (is_ftlm && block_sizes.size() > 1) {
            block_params.output_dir = params.output_dir + "/sector_" + std::to_string(block_idx);
            safe_system_call("mkdir -p " + block_params.output_dir);
        }
        
        // Diagonalize the block
        EDResults block_results;
        double large_val = (is_tpq_method && is_target_block) ? 
                          std::max(gs_info.max_energy * 10, params.large_value) : 0.0;
        
        if (is_tpq_method && is_target_block) {
            std::cout << "Running TPQ in ground state sector with large value " << large_val << std::endl;
        }
        
        block_results = ed_internal::diagonalize_symmetry_block(
            block_matrix, block_dim, method, block_params, is_target_block, large_val
        );
        
        // For FTLM: store sector results for later combination
        if (is_ftlm) {
            sector_ftlm_results.push_back(block_results.ftlm_results);
            sector_dimensions.push_back(block_dim);
        }
            
        // Store eigenvalues
        for (size_t i = 0; i < block_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({block_results.eigenvalues[i], block_idx, static_cast<uint64_t>(i)});
        }
        
        // Transform eigenvectors/states if needed
        if (params.compute_eigenvectors || (is_tpq_method && is_target_block)) {
            std::string eigenvector_dir = params.output_dir + "/eigenvectors";
            safe_system_call("mkdir -p " + eigenvector_dir);
            
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
    
    // For FTLM with multiple sectors: combine thermodynamic results properly
    if (is_ftlm && sector_ftlm_results.size() > 1) {
        std::cout << "\n=== Combining FTLM Results from " << sector_ftlm_results.size() 
                  << " Symmetry Sectors ===" << std::endl;
        
        // Combine sector results with proper statistical weights
        results.thermo_data = combine_ftlm_sector_results(
            sector_ftlm_results, sector_dimensions
        );
        
        // Save combined results
        if (!params.output_dir.empty()) {
            std::string ftlm_dir = params.output_dir + "/thermo";
            safe_system_call("mkdir -p " + ftlm_dir);
            
            // Create a combined FTLMResults for saving
            FTLMResults combined_results;
            combined_results.thermo_data = results.thermo_data;
            combined_results.ground_state_estimate = all_eigen_info.empty() ? 0.0 : all_eigen_info[0].value;
            combined_results.total_samples = sector_ftlm_results[0].total_samples;
            
            // Initialize error arrays with zeros (no error bars for combined results)
            size_t n_temps = combined_results.thermo_data.temperatures.size();
            combined_results.energy_error.assign(n_temps, 0.0);
            combined_results.specific_heat_error.assign(n_temps, 0.0);
            combined_results.entropy_error.assign(n_temps, 0.0);
            combined_results.free_energy_error.assign(n_temps, 0.0);
            
            // Save combined thermodynamics
            save_ftlm_results(combined_results, ftlm_dir + "/ftlm_thermo_combined.txt");
            
            std::cout << "Combined FTLM results saved to: " << ftlm_dir << "/ftlm_thermo_combined.txt" << std::endl;
            
            // Also save individual sector results for debugging
            for (size_t s = 0; s < sector_ftlm_results.size(); ++s) {
                std::string sector_file = ftlm_dir + "/ftlm_thermo_sector_" + 
                                         std::to_string(s) + ".txt";
                save_ftlm_results(sector_ftlm_results[s], sector_file);
            }
            std::cout << "Individual sector results saved to: " << ftlm_dir << "/ftlm_thermo_sector_*.txt" << std::endl;
        }
        
        std::cout << "=== FTLM Sector Combination Complete ===" << std::endl;
    } else if (is_ftlm && sector_ftlm_results.size() == 1) {
        // Single sector - use it directly
        std::cout << "\nNote: Only one symmetry sector computed. Results represent this sector only." << std::endl;
        results.thermo_data = sector_ftlm_results[0].thermo_data;
    }
    
    // Sort and store eigenvalues
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


/**
 * @brief Run exact diagonalization with both Fixed Sz and Spatial Symmetries (HDF5)
 * 
 * Combines U(1) charge conservation (fixed Sz) with spatial lattice symmetries.
 * This provides maximal dimension reduction: 2^N → C(N,N_up) → C(N,N_up)/|G|
 * 
 * @param directory Directory containing Hamiltonian files and automorphism data
 * @param n_up Number of up spins (determines Sz sector)
 * @param method Diagonalization method to use
 * @param params ED parameters (num_sites, spin_length, num_eigenvalues, etc.)
 * @param format Hamiltonian file format
 * @param interaction_filename Interaction file name
 * @param single_site_filename Single-site file name
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_fixed_sz_symmetrized(
    const std::string& directory,
    int64_t n_up,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Fixed-Sz + Symmetrized ED (HDF5)" << std::endl;
    std::cout << "  Sz sector: N_up = " << n_up << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    std::string automorphism_file = directory + "/automorphism_results/automorphisms.json";
    struct stat automorphism_buffer;
    bool automorphisms_exist = (stat(automorphism_file.c_str(), &automorphism_buffer) == 0);

    if (!automorphisms_exist) {
        std::cout << "Generating automorphisms..." << std::endl;
        std::string automorphism_finder_path = std::string(__FILE__);
        automorphism_finder_path = automorphism_finder_path.substr(0, automorphism_finder_path.find_last_of("/\\"));
        automorphism_finder_path += "/automorphism_finder.py";
        std::string cmd = "python3 " + automorphism_finder_path + " --data_dir=\"" + directory + "\"";
        if (!safe_system_call(cmd)) {
            std::cerr << "Warning: Automorphism finder failed" << std::endl;
            return EDResults();
        }
    }
    
    // ========== Step 2: Load Hamiltonian ==========
    std::cout << "\nLoading Hamiltonian..." << std::endl;
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    std::string three_body_file = directory + "/" + three_body_filename;
    
    FixedSzOperator hamiltonian(params.num_sites, params.spin_length, n_up);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    // Load three-body terms if available
    struct stat three_body_buffer;
    if (stat(three_body_file.c_str(), &three_body_buffer) == 0) {
        std::cout << "Loading three-body terms from: " << three_body_file << std::endl;
        hamiltonian.loadThreeBodyTerm(three_body_file);
    }
    
    // COUNTERTERM DISABLED
    // hamiltonian.loadCounterTerm(directory + "/CounterTerm.dat");

    std::cout << "Fixed Sz dimension: " << hamiltonian.getFixedSzDim() << std::endl;
    
    // ========== Step 3: Generate/Load Symmetrized Basis (HDF5) ==========
    std::string hdf5_file = directory + "/symmetry_data_fixed_sz.h5";
    bool basis_exists = HDF5SymmetryIO::fileExists(hdf5_file);
    
    if (!basis_exists) {
        std::cout << "\nGenerating symmetrized basis (Fixed Sz, HDF5)..." << std::endl;
        hamiltonian.generateSymmetrizedBasisFixedSzHDF5(directory);
        
        std::cout << "\nBuilding block-diagonal Hamiltonian (Fixed Sz, HDF5)..." << std::endl;
        hamiltonian.buildAndSaveSymmetrizedBlocksFixedSzHDF5(directory);
    } else {
        std::cout << "\nUsing existing symmetrized basis from: " << hdf5_file << std::endl;
        // Load block sizes
        auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
        hamiltonian.symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        
        // Check if blocks exist
        bool blocks_exist = false;
        try {
            H5::H5File file(hdf5_file, H5F_ACC_RDONLY);
            blocks_exist = file.nameExists("/blocks/block_0");
            file.close();
        } catch (...) {
            blocks_exist = false;
        }
        
        if (!blocks_exist) {
            std::cout << "Blocks not found. Building block-diagonal Hamiltonian..." << std::endl;
            hamiltonian.buildAndSaveSymmetrizedBlocksFixedSzHDF5(directory);
        }
    }
    
    std::vector<int> block_sizes = hamiltonian.symmetrized_block_ham_sizes;
    std::cout << "\nFound " << block_sizes.size() << " symmetry sectors within fixed-Sz" << std::endl;
    
    if (block_sizes.empty()) {
        throw std::runtime_error("No symmetrized blocks found in fixed-Sz sector");
    }
    
    // Calculate total dimension
    uint64_t total_dim = 0;
    uint64_t non_empty_sectors = 0;
    for (const auto& size : block_sizes) {
        if (size > 0) {
            total_dim += size;
            non_empty_sectors++;
        }
    }
    
    std::cout << "Non-empty sectors: " << non_empty_sectors << std::endl;
    std::cout << "Total symmetrized dimension: " << total_dim << std::endl;
    std::cout << "Fixed-Sz dimension: " << hamiltonian.getFixedSzDim() << std::endl;
    std::cout << "Dimension reduction: " 
              << static_cast<double>(hamiltonian.getFixedSzDim()) / total_dim << "x\n" << std::endl;
    
    // ========== Step 4: Determine if we need targeted diagonalization ==========
    bool is_tpq_method = (method == DiagonalizationMethod::mTPQ || 
                          method == DiagonalizationMethod::mTPQ_CUDA || 
                          method == DiagonalizationMethod::cTPQ);
    
    // For TPQ methods, find the ground state sector
    bool use_targeted_diagonalization = false;
    size_t target_sector = 0;
    double min_energy = 0.0;
    double max_energy = 0.0;
    
    if (is_tpq_method) {
        use_targeted_diagonalization = true;
        std::cout << "\n=== Finding Ground State Sector for TPQ ===" << std::endl;
        std::cout << "Scanning all sectors to identify ground state sector..." << std::endl;
        
        min_energy = std::numeric_limits<double>::max();
        max_energy = -std::numeric_limits<double>::max();
        
        for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
            uint64_t block_dim = block_sizes[block_idx];
            if (block_dim == 0) continue;
            
            std::cout << "  Scanning sector " << (block_idx + 1) << "/" << block_sizes.size() 
                      << " (dim=" << block_dim << ")" << std::endl;
            
            Eigen::SparseMatrix<Complex> block_matrix = 
                hamiltonian.loadSymmetrizedBlockFixedSzHDF5(directory, block_idx);
            
            EDParameters scan_params = params;
            scan_params.num_eigenvalues = std::min(uint64_t(5), block_dim);
            scan_params.compute_eigenvectors = false;
            scan_params.output_dir = "";
            
            EDResults scan_results = ed_internal::diagonalize_symmetry_block(
                block_matrix, block_dim, DiagonalizationMethod::LANCZOS, scan_params, false, 0.0
            );
            
            if (!scan_results.eigenvalues.empty()) {
                double sector_min = scan_results.eigenvalues[0];
                double sector_max = scan_results.eigenvalues.back();
                
                std::cout << "    Energy range: [" << sector_min << ", " << sector_max << "]" << std::endl;
                
                if (sector_min < min_energy) {
                    min_energy = sector_min;
                    target_sector = block_idx;
                }
                if (sector_max > max_energy) {
                    max_energy = sector_max;
                }
            }
        }
        
        std::cout << "\nTarget sector: " << (target_sector + 1) 
                  << " (dimension: " << block_sizes[target_sector] << ")" << std::endl;
        std::cout << "Ground state energy: " << min_energy << std::endl;
        std::cout << "Maximum energy found: " << max_energy << std::endl;
        std::cout << "Will only diagonalize target sector for TPQ.\n" << std::endl;
    }
    
    // ========== Step 5: Diagonalize Sector(s) ==========
    std::cout << "========== Diagonalizing Sectors ==========\n" << std::endl;
    
    struct EigenInfo {
        double value;
        uint64_t sector_idx;
        uint64_t eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;
    
    // For FTLM: collect results from all sectors for proper combination
    bool is_ftlm = (method == DiagonalizationMethod::FTLM);
    std::vector<FTLMResults> sector_ftlm_results;
    std::vector<uint64_t> sector_dimensions;
    
    EDResults results;
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
    }
    
    uint64_t block_start_dim = 0;
    for (size_t block_idx = 0; block_idx < block_sizes.size(); ++block_idx) {
        uint64_t block_dim = block_sizes[block_idx];
        
        // Skip empty blocks
        if (block_dim == 0) {
            std::cout << "Sector " << (block_idx + 1) << "/" << block_sizes.size() 
                     << ": empty (skipping)" << std::endl;
            continue;
        }
        
        // Skip non-target blocks for TPQ
        bool is_target_block = (use_targeted_diagonalization && block_idx == target_sector);
        
        if (use_targeted_diagonalization && !is_target_block) {
            std::cout << "Sector " << (block_idx + 1) << "/" << block_sizes.size() 
                     << " (dim=" << block_dim << "): skipping (not target sector)" << std::endl;
            block_start_dim += block_dim;
            continue;
        }
        
        std::cout << "Sector " << (block_idx + 1) << "/" << block_sizes.size() 
                  << " (dim=" << block_dim << ")" << std::endl;
        
        // Load block matrix from HDF5
        Eigen::SparseMatrix<Complex> block_matrix = 
            hamiltonian.loadSymmetrizedBlockFixedSzHDF5(directory, block_idx);
        
        // Configure parameters for this block
        EDParameters block_params = params;
        block_params.num_eigenvalues = std::min(params.num_eigenvalues, block_dim);
        
        if (params.compute_eigenvectors || (is_tpq_method && is_target_block)) {
            block_params.output_dir = params.output_dir + "/sector_" + std::to_string(block_idx);
            safe_system_call("mkdir -p " + block_params.output_dir);
        }
        
        // Set large value for TPQ in target block
        double large_val = 0.0;
        if (is_tpq_method && is_target_block) {
            large_val = std::max(max_energy * 10, params.large_value);
            std::cout << "  Running TPQ in ground state sector with large value " << large_val << std::endl;
        }
        
        // Diagonalize this block (with GPU support)
        std::cout << "  Diagonalizing..." << std::endl;
        EDResults block_results = ed_internal::diagonalize_symmetry_block(
            block_matrix, block_dim, method, block_params, is_target_block, large_val
        );
        
        // For FTLM: store sector results for later combination
        if (is_ftlm) {
            sector_ftlm_results.push_back(block_results.ftlm_results);
            sector_dimensions.push_back(block_dim);
        }
        
        // Store eigenvalue information
        for (size_t i = 0; i < block_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                block_results.eigenvalues[i],
                block_idx,
                static_cast<uint64_t>(i)
            });
        }
        
        // Transform TPQ states or eigenvectors if needed
        if (params.compute_eigenvectors || (is_tpq_method && is_target_block)) {
            std::string eigenvector_dir = params.output_dir + "/eigenvectors";
            safe_system_call("mkdir -p " + eigenvector_dir);
            
            if (is_tpq_method && is_target_block) {
                std::cout << "  Transforming TPQ states from sector basis to fixed-Sz basis..." << std::endl;
                // Cast FixedSzOperator to Operator (safe since FixedSzOperator inherits from Operator)
                Operator& op_ref = static_cast<Operator&>(hamiltonian);
                ed_internal::transform_and_save_tpq_states(
                    block_params.output_dir, params.output_dir, 
                    op_ref, directory,
                    block_dim, block_start_dim, block_idx, params.num_sites
                );
                
                // Additional transformation: fixed-Sz basis -> full Hilbert space for TPQ states
                std::cout << "  Transforming TPQ states from fixed-Sz basis to full Hilbert space..." << std::endl;
                uint64_t full_dim = 1ULL << params.num_sites;
                uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
                
                // Find all TPQ state files in output directory
                std::string temp_list_file = params.output_dir + "/tpq_state_files.txt";
                std::string find_command = "find \"" + params.output_dir + "\" -name \"tpq_state_*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
                safe_system_call(find_command);
                
                std::ifstream file_list(temp_list_file);
                if (file_list.is_open()) {
                    std::string tpq_state_file;
                    while (std::getline(file_list, tpq_state_file)) {
                        if (tpq_state_file.empty()) continue;
                        
                        std::ifstream infile(tpq_state_file, std::ios::binary);
                        if (infile.is_open()) {
                            // Read TPQ state in fixed-Sz basis
                            std::vector<Complex> fixed_sz_vec(fixed_sz_dim);
                            infile.read(reinterpret_cast<char*>(fixed_sz_vec.data()), fixed_sz_dim * sizeof(Complex));
                            infile.close();
                            
                            // Transform to full Hilbert space basis
                            std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                            
                            // Overwrite file with full-space TPQ state
                            std::ofstream outfile(tpq_state_file, std::ios::binary);
                            outfile.write(reinterpret_cast<const char*>(full_vec.data()), full_dim * sizeof(Complex));
                            outfile.close();
                        }
                    }
                    file_list.close();
                }
                std::remove(temp_list_file.c_str());
                std::cout << "  All TPQ states transformed to full Hilbert space." << std::endl;
            }
            
            if (params.compute_eigenvectors && method != DiagonalizationMethod::mTPQ && 
                method != DiagonalizationMethod::mTPQ_CUDA && method != DiagonalizationMethod::cTPQ) {
                std::cout << "  Transforming eigenvectors from sector basis to fixed-Sz basis..." << std::endl;
                Operator& op_ref = static_cast<Operator&>(hamiltonian);
                ed_internal::transform_and_save_eigenvectors(
                    block_params.output_dir, params.output_dir,
                    op_ref, directory,
                    block_results.eigenvalues, block_dim, block_start_dim, block_idx, params.num_sites
                );
                
                // Additional transformation: fixed-Sz basis -> full Hilbert space
                std::cout << "  Transforming eigenvectors from fixed-Sz basis to full Hilbert space..." << std::endl;
                uint64_t full_dim = 1ULL << params.num_sites;
                uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
                
                for (size_t i = 0; i < block_results.eigenvalues.size(); ++i) {
                    std::string eigvec_file = params.output_dir + "/eigenvector_sector" + 
                                             std::to_string(block_idx) + "_" + std::to_string(i) + ".dat";
                    
                    std::ifstream infile(eigvec_file, std::ios::binary);
                    if (infile.is_open()) {
                        // Read eigenvector in fixed-Sz basis
                        std::vector<Complex> fixed_sz_vec(fixed_sz_dim);
                        infile.read(reinterpret_cast<char*>(fixed_sz_vec.data()), fixed_sz_dim * sizeof(Complex));
                        infile.close();
                        
                        // Transform to full Hilbert space basis
                        std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                        
                        // Overwrite file with full-space eigenvector
                        std::ofstream outfile(eigvec_file, std::ios::binary);
                        outfile.write(reinterpret_cast<const char*>(full_vec.data()), full_dim * sizeof(Complex));
                        outfile.close();
                        
                        std::cout << "    Transformed eigenvector " << i << " to full space (dim: " 
                                  << fixed_sz_dim << " -> " << full_dim << ")" << std::endl;
                    } else {
                        std::cerr << "    Warning: Could not open eigenvector file: " << eigvec_file << std::endl;
                    }
                }
            }
        }
        
        std::cout << "  Found " << block_results.eigenvalues.size() << " eigenvalues" << std::endl;
        if (!block_results.eigenvalues.empty()) {
            std::cout << "  Lowest: " << block_results.eigenvalues[0] << std::endl;
        }
        std::cout << std::endl;
        
        block_start_dim += block_dim;
    }
    
    // ========== Step 5.5: Sort Eigenvalues First ==========
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    if (all_eigen_info.size() > static_cast<size_t>(params.num_eigenvalues)) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    // ========== Step 5.6: Combine FTLM Results from Multiple Sectors ==========
    
    // For FTLM with multiple sectors: combine thermodynamic results properly
    if (is_ftlm && sector_ftlm_results.size() > 1) {
        std::cout << "\n=== Combining FTLM Results from " << sector_ftlm_results.size() 
                  << " Symmetry Sectors ===" << std::endl;
        
        // Combine sector results with proper statistical weights
        results.thermo_data = combine_ftlm_sector_results(
            sector_ftlm_results, sector_dimensions
        );
        
        // Save combined results
        if (!params.output_dir.empty()) {
            std::string ftlm_dir = params.output_dir + "/thermo";
            safe_system_call("mkdir -p " + ftlm_dir);
            
            // Create a combined FTLMResults for saving
            FTLMResults combined_results;
            combined_results.thermo_data = results.thermo_data;
            combined_results.ground_state_estimate = results.eigenvalues.empty() ? 0.0 : results.eigenvalues[0];
            combined_results.total_samples = sector_ftlm_results[0].total_samples;
            
            // Initialize error arrays with zeros (no error bars for combined results)
            size_t n_temps = combined_results.thermo_data.temperatures.size();
            combined_results.energy_error.assign(n_temps, 0.0);
            combined_results.specific_heat_error.assign(n_temps, 0.0);
            combined_results.entropy_error.assign(n_temps, 0.0);
            combined_results.free_energy_error.assign(n_temps, 0.0);
            
            // Save combined thermodynamics
            save_ftlm_results(combined_results, ftlm_dir + "/ftlm_thermo_combined.txt");
            
            std::cout << "Combined FTLM results saved to: " << ftlm_dir << "/ftlm_thermo_combined.txt" << std::endl;
            
            // Also save individual sector results for debugging
            for (size_t s = 0; s < sector_ftlm_results.size(); ++s) {
                std::string sector_file = ftlm_dir + "/ftlm_thermo_sector_" + 
                                         std::to_string(s) + ".txt";
                save_ftlm_results(sector_ftlm_results[s], sector_file);
            }
            std::cout << "Individual sector results saved to: " << ftlm_dir << "/ftlm_thermo_sector_*.txt" << std::endl;
        }
        
        std::cout << "=== FTLM Sector Combination Complete ===" << std::endl;
    } else if (is_ftlm && sector_ftlm_results.size() == 1) {
        // Single sector - use it directly
        std::cout << "\nNote: Only one symmetry sector computed. Results represent this sector only." << std::endl;
        results.thermo_data = sector_ftlm_results[0].thermo_data;
        results.ftlm_results = sector_ftlm_results[0];
    }
    
    // ========== Step 6: Create Eigenvector Mapping ==========
    if (params.compute_eigenvectors && !params.output_dir.empty()) {
        std::ofstream map_file(params.output_dir + "/eigenvector_mapping.txt");
        if (map_file.is_open()) {
            map_file << "# Global_Index Eigenvalue Sector_Index Sector_Eigenvalue_Index Filename" << std::endl;
            for (size_t i = 0; i < all_eigen_info.size(); ++i) {
                const auto& info = all_eigen_info[i];
                map_file << i << " " << info.value << " " << info.sector_idx << " " << info.eigen_idx 
                        << " eigenvector_sector" << info.sector_idx << "_" << info.eigen_idx << ".dat" << std::endl;
            }
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Fixed-Sz Symmetrized ED Complete" << std::endl;
    std::cout << "  Total eigenvalues: " << results.eigenvalues.size() << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "  Ground state: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    return results;
}


#endif