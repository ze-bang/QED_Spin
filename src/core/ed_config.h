#ifndef ED_CONFIG_H
#define ED_CONFIG_H

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <variant>
#include <fstream>
#include <sstream>
#include <iostream>
#include "construct_ham.h"  // For Operator class (needed by ObservableConfig)

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Forward declare to avoid including ed_wrapper.h (which has inline functions that cause linking issues)
enum class DiagonalizationMethod;

// ============================================================================
// HIERARCHICAL PARAMETER STRUCTURES
// ============================================================================

/**
 * @brief Core diagonalization parameters
 */
struct DiagonalizationConfig {
    int num_eigenvalues = 1;
    int max_iterations = 10000;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    
    // Method-specific
    double shift = 0.0;           // For shift-invert
    int block_size = 4;          // For block methods
    int max_subspace = 100;       // For Davidson
    double target_lower = 0.0;    // For Chebyshev filtered (lower energy bound)
    double target_upper = 0.0;    // For Chebyshev filtered (upper energy bound)
};

/**
 * @brief Thermal calculation parameters
 */
struct ThermalConfig {
    int num_samples = 1;
    double temp_min = 1e-3;
    double temp_max = 20.0;
    int num_temp_bins = 100;
    
    // TPQ-specific
    int num_order = 100;          // Order for cTPQ
    int num_measure_freq = 100;   // Measurement frequency
    double delta_tau = 1e-2;      // Time step for cTPQ
    double large_value = 1e5;     // Large value for mTPQ
    
    // FTLM-specific
    int ftlm_krylov_dim = 100;    // Krylov subspace dimension per sample
    bool ftlm_full_reorth = false; // Use full reorthogonalization
    int ftlm_reorth_freq = 10;    // Reorthogonalization frequency
    unsigned int ftlm_seed = 0;   // Random seed (0 = auto)
    bool ftlm_store_samples = false; // Store per-sample intermediate data
    bool ftlm_error_bars = true;  // Compute error bars
    
    // LTLM-specific
    int ltlm_num_eigenstates = 100;   // Number of low-energy eigenstates to compute
    int ltlm_krylov_dim = 300;        // Krylov subspace dimension (should be > num_eigenstates)
    double ltlm_tolerance = 1e-12;    // Eigenvalue convergence tolerance
    bool ltlm_full_reorth = true;     // Use full reorthogonalization
    int ltlm_reorth_freq = 1;         // Reorthogonalization frequency (if not full)
    unsigned int ltlm_seed = 0;       // Random seed (0 = auto)
    bool ltlm_store_eigenvectors = false;  // Store eigenvectors (memory intensive)
    bool ltlm_verify_eigenvalues = true;   // Verify using residual test
    double ltlm_residual_tol = 1e-10;      // Residual tolerance for verification
    double ltlm_degeneracy_threshold = 1e-10; // Threshold for degeneracy detection
    
    // HYBRID mode (automatic LTLM + FTLM stitching)
    bool hybrid_mode = false;         // Enable hybrid mode
    double hybrid_crossover_temp = -1.0; // Crossover temperature (auto if < 0)
    int hybrid_overlap_bins = 10;     // Number of overlapping temperature points
    double hybrid_ltlm_temp_max = 0.5; // Max temperature for LTLM in hybrid mode
};

/**
 * @brief Observable calculation parameters
 */
struct ObservableConfig {
    bool calculate = false;
    bool measure_spin = false;
    
    // Spectral functions
    double omega_min = -10.0;
    double omega_max = 10.0;
    int num_points = 1000;
    
    // Time evolution
    double t_end = 50.0;
    double dt = 0.01;
    
    // Custom observables (loaded from files)
    mutable std::vector<Operator> operators = {};
    mutable std::vector<std::string> names = {};
};

/**
 * @brief Dynamical response calculation parameters
 */
struct DynamicalResponseConfig {
    bool calculate = false;
    bool thermal_average = false;     // If true, compute thermal-averaged response
    int num_random_states = 20;       // Number of random states for thermal averaging
    int krylov_dim = 100;             // Krylov subspace dimension
    double omega_min = -10.0;         // Minimum frequency
    double omega_max = 10.0;          // Maximum frequency
    int num_omega_points = 1000;      // Number of frequency points
    double broadening = 0.1;          // Lorentzian broadening parameter
    double temp_min = 0.01;           // Minimum temperature (for temperature scan)
    double temp_max = 10.0;           // Maximum temperature (for temperature scan)
    int num_temp_bins = 1;            // Number of temperature points (1 = single temperature at temp_min)
    bool compute_correlation = false; // Compute two-operator correlation
    std::string operator_file = "";   // File containing operator to probe
    std::string operator2_file = "";  // Second operator file (for correlation)
    std::string output_prefix = "dynamical_response";
    unsigned int random_seed = 0;     // Random seed (0 = auto)
};

/**
 * @brief Static response calculation parameters
 */
struct StaticResponseConfig {
    bool calculate = false;
    int num_random_states = 20;       // Number of random states for thermal averaging
    int krylov_dim = 100;             // Krylov subspace dimension
    double temp_min = 0.01;           // Minimum temperature
    double temp_max = 10.0;           // Maximum temperature
    int num_temp_points = 100;        // Number of temperature points
    bool compute_susceptibility = true; // Compute susceptibility dO/dT
    bool compute_correlation = false; // Compute two-operator correlation
    bool single_operator_mode = false; // Compute single-operator ⟨O⟩ instead of ⟨O†O⟩
    std::string operator_file = "";   // File containing operator(s) to probe
    std::string operator2_file = "";  // Second operator file (for correlation)
    std::string output_prefix = "static_response";
    unsigned int random_seed = 0;     // Random seed (0 = auto)
};

/**
 * @brief System/lattice parameters
 */
struct SystemConfig {
    int num_sites = 0;
    float spin_length = 0.5;
    int sublattice_size = 1;
    
    // Fixed Sz mode
    bool use_fixed_sz = false;
    int n_up = -1;  // Number of up spins (-1 = not set, will use num_sites/2)
    
    std::string hamiltonian_dir = "";
    std::string interaction_file = "InterAll.dat";
    std::string single_site_file = "Trans.dat";
};

/**
 * @brief ARPACK advanced options (only used for ARPACK_ADVANCED)
 */
struct ArpackConfig {
    bool verbose = false;
    std::string which = "SR";  // For Hermitian matrices: SR=Smallest Real (ground state), LR=Largest Real
    int ncv = -1;
    int max_restarts = 2;
    double ncv_growth = 1.5;
    bool auto_enlarge_ncv = true;
    bool two_phase_refine = true;
    double relaxed_tol = 1e-6;
    bool shift_invert = false;
    double sigma = 0.0;
    bool auto_switch_shift_invert = true;
    double switch_sigma = 0.0;
    bool adaptive_inner_tol = true;
    double inner_tol_factor = 1e-2;
    double inner_tol_min = 1e-14;
    int inner_max_iter = 300;
};

/**
 * @brief Workflow control flags
 */
struct WorkflowConfig {
    bool run_standard = false;
    bool run_symmetrized = false;
    bool compute_thermo = false;
    bool compute_dynamical_response = false;
    bool compute_static_response = false;
    bool skip_ed = false;
    std::string output_dir = "output";
};

// ============================================================================
// UNIFIED CONFIGURATION CLASS
// ============================================================================

/**
 * @brief Main configuration class using builder pattern
 * 
 * Organizes all parameters into logical groups and provides
 * fluent interface for configuration.
 */
class EDConfig {
public:
    DiagonalizationMethod method;
    DiagonalizationConfig diag;
    ThermalConfig thermal;
    ObservableConfig observable;
    DynamicalResponseConfig dynamical;
    StaticResponseConfig static_resp;
    SystemConfig system;
    ArpackConfig arpack;
    WorkflowConfig workflow;
    
    // ========== Constructors ==========
    EDConfig();  // Default constructor (implemented in .cpp with default method)
    EDConfig(DiagonalizationMethod m) : method(m) {}  // Explicit method constructor
    
    // ========== Builder Methods ==========
    
    // Diagonalization
    EDConfig& eigenvalues(int n) { diag.num_eigenvalues = n; return *this; }
    EDConfig& iterations(int n) { diag.max_iterations = n; return *this; }
    EDConfig& tolerance(double t) { diag.tolerance = t; return *this; }
    EDConfig& eigenvectors(bool b = true) { diag.compute_eigenvectors = b; return *this; }
    EDConfig& shift(double s) { diag.shift = s; return *this; }
    EDConfig& blockSize(int b) { diag.block_size = b; return *this; }
    EDConfig& targetRange(double lower, double upper) { 
        diag.target_lower = lower; 
        diag.target_upper = upper; 
        return *this; 
    }
    
    // Thermal
    EDConfig& samples(int n) { thermal.num_samples = n; return *this; }
    EDConfig& tempRange(double min, double max) { 
        thermal.temp_min = min; 
        thermal.temp_max = max; 
        return *this; 
    }
    EDConfig& tempBins(int n) { thermal.num_temp_bins = n; return *this; }
    
    // Observable
    EDConfig& calcObservables(bool b = true) { observable.calculate = b; return *this; }
    EDConfig& measureSpin(bool b = true) { observable.measure_spin = b; return *this; }
    
    // System
    EDConfig& numSites(int n) { system.num_sites = n; return *this; }
    EDConfig& spinLength(float s) { system.spin_length = s; return *this; }
    EDConfig& hamiltonianDir(const std::string& dir) { 
        system.hamiltonian_dir = dir; 
        return *this; 
    }
    EDConfig& fixedSz(bool use = true) { system.use_fixed_sz = use; return *this; }
    EDConfig& numUp(int n) { system.n_up = n; return *this; }
    
    // Workflow
    EDConfig& standard(bool b = true) { workflow.run_standard = b; return *this; }
    EDConfig& symmetrized(bool b = true) { workflow.run_symmetrized = b; return *this; }
    EDConfig& thermo(bool b = true) { workflow.compute_thermo = b; return *this; }
    EDConfig& dynamicalResponse(bool b = true) { workflow.compute_dynamical_response = b; return *this; }
    EDConfig& staticResponse(bool b = true) { workflow.compute_static_response = b; return *this; }
    EDConfig& outputDir(const std::string& dir) { workflow.output_dir = dir; return *this; }
    
    // ========== Configuration Loading ==========
    
    /**
     * @brief Load configuration from file
     * Supports simple key=value format
     */
    static EDConfig fromFile(const std::string& filename);
    
    /**
     * @brief Parse command line arguments
     */
    static EDConfig fromCommandLine(int argc, char* argv[]);
    
    /**
     * @brief Merge configurations (command line overrides file)
     */
    EDConfig& merge(const EDConfig& other);
    
    /**
     * @brief Validate configuration
     */
    bool validate(std::ostream& err = std::cerr) const;
    
    /**
     * @brief Save configuration to file
     */
    void save(const std::string& filename) const;
    
    /**
     * @brief Print configuration summary
     */
    void print(std::ostream& out = std::cout) const;
    
    /**
     * @brief Auto-detect num_sites from positions.dat
     */
    bool autoDetectNumSites();
};

// ============================================================================
// COMMAND LINE PARSER
// ============================================================================

/**
 * @brief Modern command-line parser with automatic help generation
 */
class CommandLineParser {
public:
    struct Option {
        std::string long_name;
        std::string short_name;
        std::string description;
        std::string value_name;
        bool has_value;
        bool required;
        std::string category;
    };
    
    CommandLineParser& addOption(
        const std::string& long_name,
        const std::string& short_name,
        const std::string& description,
        bool has_value = true,
        bool required = false,
        const std::string& category = "General"
    );
    
    bool parse(int argc, char* argv[]);
    
    std::optional<std::string> get(const std::string& name) const;
    bool has(const std::string& name) const;
    
    void printHelp(std::ostream& out = std::cout) const;
    
private:
    std::vector<Option> options_;
    std::map<std::string, std::string> values_;
    std::string program_name_;
};

// ============================================================================
// CONVERSION UTILITIES
// ============================================================================

namespace ed_config {
    /**
     * @brief Convert string to DiagonalizationMethod
     */
    std::optional<DiagonalizationMethod> parseMethod(const std::string& str);
    
    /**
     * @brief Convert DiagonalizationMethod to string
     */
    std::string methodToString(DiagonalizationMethod method);
    
    /**
     * @brief Setup default configuration for a method
     */
    EDConfig defaultConfigFor(DiagonalizationMethod method);
    
    /**
     * @brief Get detailed parameter information for a method
     */
    std::string getMethodParameterInfo(DiagonalizationMethod method);
}

#endif // ED_CONFIG_H
