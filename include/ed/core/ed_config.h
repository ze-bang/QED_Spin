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
#include <ed/core/construct_ham.h>  // For Operator class (needed by ObservableConfig)

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
    uint64_t num_eigenvalues = 1;
    uint64_t max_iterations = 10000;
    double tolerance = 1e-10;
    bool compute_eigenvectors = false;
    
    // Method-specific
    double shift = 0.0;           // For shift-invert
    uint64_t block_size = 4;          // For block methods
    uint64_t max_subspace = 100;       // For Davidson
    double target_lower = 0.0;    // For Chebyshev filtered (lower energy bound)
    double target_upper = 0.0;    // For Chebyshev filtered (upper energy bound)
};

/**
 * @brief Thermal calculation parameters
 * 
 * Unified thermal parameters for all methods: TPQ (mTPQ, cTPQ), FTLM, LTLM, Hybrid.
 * 
 * Naming convention for TPQ parameters:
 * - Old name -> New name (use new names in new code)
 * - num_order -> tpq_taylor_order
 * - num_measure_freq -> tpq_measurement_interval
 * - delta_tau -> tpq_delta_beta
 * - large_value -> tpq_energy_shift
 * - continue_quenching -> tpq_continue
 * - continue_sample -> tpq_continue_sample
 * - continue_beta -> tpq_continue_beta
 */
struct ThermalConfig {
    // Common parameters
    uint64_t num_samples = 1;          // Number of random samples for thermal averaging
    double temp_min = 1e-3;            // Minimum temperature (for output grid)
    double temp_max = 20.0;            // Maximum temperature (for output grid)
    uint64_t num_temp_bins = 100;      // Number of temperature bins for output
    
    // ========== TPQ-specific parameters ==========
    // mTPQ (microcanonical) parameters
    uint64_t tpq_max_steps = 10000;        // Maximum number of mTPQ evolution steps
    uint64_t tpq_measurement_interval = 100; // Interval between measurements (in steps)
    double tpq_energy_shift = 1e5;         // Large energy shift for mTPQ
    
    // cTPQ (canonical) parameters  
    double tpq_beta_max = 20.0;            // Maximum inverse temperature (1/T_min)
    double tpq_delta_beta = 1e-2;          // Imaginary-time step for cTPQ evolution
    uint64_t tpq_taylor_order = 100;       // Taylor expansion order for e^{-delta_beta*H}
    
    // Continue quenching options
    bool tpq_continue = false;             // Continue quenching from saved state
    uint64_t tpq_continue_sample = 0;      // Sample to continue from (0 = auto-detect)
    double tpq_continue_beta = 0.0;        // Beta to continue from (0.0 = use saved)
    
    // ========== FTLM-specific parameters ==========
    uint64_t ftlm_krylov_dim = 100;    // Krylov subspace dimension per sample
    bool ftlm_full_reorth = false;     // Use full reorthogonalization
    uint64_t ftlm_reorth_freq = 10;    // Reorthogonalization frequency
    uint64_t ftlm_seed = 0;            // Random seed (0 = auto)
    bool ftlm_store_samples = false;   // Store per-sample intermediate data
    bool ftlm_error_bars = true;       // Compute error bars
    
    // ========== LTLM-specific parameters ==========
    uint64_t ltlm_krylov_dim = 200;    // Krylov subspace dimension for excitations
    uint64_t ltlm_ground_krylov = 100; // Krylov dimension for finding ground state
    bool ltlm_full_reorth = false;     // Use full reorthogonalization
    uint64_t ltlm_reorth_freq = 10;    // Reorthogonalization frequency
    uint64_t ltlm_seed = 0;            // Random seed (0 = auto)
    bool ltlm_store_data = false;      // Store intermediate data
    
    // ========== Hybrid Thermal parameters ==========
    bool use_hybrid_method = false;    // DEPRECATED: use method=HYBRID instead
    double hybrid_crossover = 1.0;     // Temperature crossover for hybrid method
    bool hybrid_auto_crossover = false;// Automatically determine crossover temperature
    
    // Getter methods for backwards compatibility with old parameter names
    uint64_t get_num_order() const { return tpq_taylor_order; }
    uint64_t get_num_measure_freq() const { return tpq_measurement_interval; }
    double get_delta_tau() const { return tpq_delta_beta; }
    double get_large_value() const { return tpq_energy_shift; }
    bool get_continue_quenching() const { return tpq_continue; }
    uint64_t get_continue_sample() const { return tpq_continue_sample; }
    double get_continue_beta() const { return tpq_continue_beta; }
    
    // Setter methods for backwards compatibility
    void set_num_order(uint64_t v) { tpq_taylor_order = v; }
    void set_num_measure_freq(uint64_t v) { tpq_measurement_interval = v; }
    void set_delta_tau(double v) { tpq_delta_beta = v; }
    void set_large_value(double v) { tpq_energy_shift = v; }
    void set_continue_quenching(bool v) { tpq_continue = v; }
    void set_continue_sample(uint64_t v) { tpq_continue_sample = v; }
    void set_continue_beta(double v) { tpq_continue_beta = v; }
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
    uint64_t num_points = 1000;
    
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
    bool use_gpu = false;             // Use GPU acceleration (requires CUDA)
    uint64_t num_random_states = 20;       // Number of random states for thermal averaging
    uint64_t krylov_dim = 400;             // Krylov subspace dimension
    double omega_min = -5.0;         // Minimum frequency
    double omega_max = 5.0;          // Maximum frequency
    uint64_t num_omega_points = 1000;      // Number of frequency points
    double broadening = 0.1;          // Lorentzian broadening parameter
    double temp_min = 0.001;           // Minimum temperature (for temperature scan)
    double temp_max = 1.0;           // Maximum temperature (for temperature scan)
    uint64_t num_temp_bins = 4;            // Number of temperature points (1 = single temperature at temp_min)
    bool compute_correlation = false; // Compute two-operator correlation
    std::string operator_file = "";   // File containing operator to probe (legacy mode)
    std::string operator2_file = "";  // Second operator file (for correlation, legacy mode)
    std::string output_prefix = "dynamical_response";
    uint64_t random_seed = 0;     // Random seed (0 = auto)
    
    // Configuration-based operator construction (like TPQ_DSSF)
    std::string operator_type = "sum";  // sum | transverse | sublattice | experimental | transverse_experimental
    std::string basis = "ladder";       // ladder | xyz
    std::string spin_combinations = "0,0;2,2";  // Format: "op1,op2;op3,op4;..." (0=Sp/Sx, 1=Sm/Sy, 2=Sz)
    uint64_t unit_cell_size = 4;            // For sublattice operators
    std::string momentum_points = "0,0,0;0,0,2";  // Format: "Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;..." (in units of π)
    std::string polarization = "1,-1,0";  // For transverse operators (normalized)
    double theta = 0.0;                 // For experimental operators (in radians)
};

/**
 * @brief Static response calculation parameters
 */
struct StaticResponseConfig {
    bool calculate = false;
    bool use_gpu = false;             // Use GPU acceleration (requires CUDA)
    uint64_t num_random_states = 20;       // Number of random states for thermal averaging
    uint64_t krylov_dim = 400;             // Krylov subspace dimension
    double temp_min = 0.001;           // Minimum temperature
    double temp_max = 1.0;           // Maximum temperature
    uint64_t num_temp_points = 4;        // Number of temperature points
    bool compute_susceptibility = true; // Compute susceptibility dO/dT
    bool compute_correlation = false; // Compute two-operator correlation
    bool single_operator_mode = false; // Compute single-operator ⟨O⟩ instead of ⟨O†O⟩
    std::string operator_file = "";   // File containing operator(s) to probe (legacy mode)
    std::string operator2_file = "";  // Second operator file (for correlation, legacy mode)
    std::string output_prefix = "static_response";
    uint64_t random_seed = 0;     // Random seed (0 = auto)
    
    // Configuration-based operator construction (like TPQ_DSSF)
    std::string operator_type = "sum";  // sum | transverse | sublattice | experimental | transverse_experimental
    std::string basis = "ladder";       // ladder | xyz
    std::string spin_combinations = "0,0;2,2";  // Format: "op1,op2;op3,op4;..." (0=Sp/Sx, 1=Sm/Sy, 2=Sz)
    uint64_t unit_cell_size = 4;            // For sublattice operators
    std::string momentum_points = "0,0,0;0,0,2";  // Format: "Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;..." (in units of π)
    std::string polarization = "1,-1,0";  // For transverse operators (normalized)
    double theta = 0.0;                 // For experimental operators (in radians)
};

/**
 * @brief System/lattice parameters
 */
struct SystemConfig {
    uint64_t num_sites = 0;
    float spin_length = 0.5;
    uint64_t sublattice_size = 1;
    
    // Fixed Sz mode
    bool use_fixed_sz = false;
    int64_t n_up = -1;  // Number of up spins (-1 = not set, will use num_sites/2)
    
    std::string hamiltonian_dir = "";
    std::string interaction_file = "InterAll.dat";
    std::string single_site_file = "Trans.dat";
    std::string three_body_file = "ThreeBodyG.dat";  // Optional: three-body interaction file (e.g., ThreeBodyG.dat)
};

/**
 * @brief ARPACK advanced options (only used for ARPACK_ADVANCED)
 */
struct ArpackConfig {
    bool verbose = false;
    std::string which = "SR";  // For Hermitian matrices: SR=Smallest Real (ground state), LR=Largest Real
    int64_t ncv = -1;
    uint64_t max_restarts = 2;
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
    uint64_t inner_max_iter = 300;
};

/**
 * @brief Workflow control flags
 */
struct WorkflowConfig {
    bool run_standard = false;
    bool run_symmetrized = false;
    bool run_streaming_symmetry = false;  // New streaming symmetry mode
    bool compute_thermo = false;
    bool compute_dynamical_response = false;
    bool compute_static_response = false;
    bool compute_ground_state_dssf = false;  // Ground state dynamical structure factor (continued fraction)
    bool skip_ed = false;
    std::string output_dir = "output";
    std::string eigenvector_dir = "";  // Directory with pre-computed eigenvectors (for ground state DSSF)
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
    EDConfig& eigenvalues(uint64_t n) { diag.num_eigenvalues = n; return *this; }
    EDConfig& iterations(uint64_t n) { diag.max_iterations = n; return *this; }
    EDConfig& tolerance(double t) { diag.tolerance = t; return *this; }
    EDConfig& eigenvectors(bool b = true) { diag.compute_eigenvectors = b; return *this; }
    EDConfig& shift(double s) { diag.shift = s; return *this; }
    EDConfig& blockSize(uint64_t b) { diag.block_size = b; return *this; }
    EDConfig& targetRange(double lower, double upper) { 
        diag.target_lower = lower; 
        diag.target_upper = upper; 
        return *this; 
    }
    
    // Thermal
    EDConfig& samples(uint64_t n) { thermal.num_samples = n; return *this; }
    EDConfig& tempRange(double min, double max) { 
        thermal.temp_min = min; 
        thermal.temp_max = max; 
        return *this; 
    }
    EDConfig& tempBins(uint64_t n) { thermal.num_temp_bins = n; return *this; }
    
    // Observable
    EDConfig& calcObservables(bool b = true) { observable.calculate = b; return *this; }
    EDConfig& measureSpin(bool b = true) { observable.measure_spin = b; return *this; }
    
    // System
    EDConfig& numSites(uint64_t n) { system.num_sites = n; return *this; }
    EDConfig& spinLength(float s) { system.spin_length = s; return *this; }
    EDConfig& hamiltonianDir(const std::string& dir) { 
        system.hamiltonian_dir = dir; 
        return *this; 
    }
    EDConfig& fixedSz(bool use = true) { system.use_fixed_sz = use; return *this; }
    EDConfig& numUp(uint64_t n) { system.n_up = n; return *this; }
    
    // Workflow
    EDConfig& standard(bool b = true) { workflow.run_standard = b; return *this; }
    EDConfig& symmetrized(bool b = true) { workflow.run_symmetrized = b; return *this; }
    EDConfig& streamingSymmetry(bool b = true) { workflow.run_streaming_symmetry = b; return *this; }
    EDConfig& thermo(bool b = true) { workflow.compute_thermo = b; return *this; }
    EDConfig& dynamicalResponse(bool b = true) { workflow.compute_dynamical_response = b; return *this; }
    EDConfig& staticResponse(bool b = true) { workflow.compute_static_response = b; return *this; }
    EDConfig& groundStateDSSF(bool b = true) { workflow.compute_ground_state_dssf = b; return *this; }
    EDConfig& outputDir(const std::string& dir) { workflow.output_dir = dir; return *this; }
    EDConfig& eigenvectorDir(const std::string& dir) { workflow.eigenvector_dir = dir; return *this; }
    
    // ========== Configuration Loading ==========
    
    /**
     * @brief Load configuration from file
     * Supports simple key=value format
     */
    static EDConfig fromFile(const std::string& filename);

    
    /**
     * @brief Parse command line arguments
     */
    static EDConfig fromCommandLine(uint64_t argc, char* argv[]);
    
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
     * @brief Pruint64_t configuration summary
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
    
    bool parse(uint64_t argc, char* argv[]);
    
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
