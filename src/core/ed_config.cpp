#include "ed_config.h"
// NOTE: We include ed_wrapper.h ONLY in the implementation of conversion functions
// This is at the end of the file to avoid including it globally
#include <algorithm>
#include <cctype>

// ============================================================================
// Temporary forward declaration to avoid including ed_wrapper.h here
// The actual enum is defined in ed_wrapper.h
// ============================================================================
enum class DiagonalizationMethod {
    LANCZOS,
    LANCZOS_SELECTIVE,
    LANCZOS_NO_ORTHO,
    BLOCK_LANCZOS,
    CHEBYSHEV_FILTERED,
    SHIFT_INVERT,
    SHIFT_INVERT_ROBUST,
    DAVIDSON,
    BICG,
    LOBPCG,
    KRYLOV_SCHUR,
    IMPLICIT_RESTART_LANCZOS,
    THICK_RESTART_LANCZOS,
    FULL,
    OSS,
    mTPQ,
    mTPQ_MPI,
    cTPQ,
    mTPQ_CUDA,
    FTLM,
    LTLM,
    HYBRID,
    ARPACK_SM,
    ARPACK_LM,
    ARPACK_SHIFT_INVERT,
    ARPACK_ADVANCED,
    LANCZOS_GPU,
    LANCZOS_GPU_FIXED_SZ,
    DAVIDSON_GPU,
    LOBPCG_GPU,
    mTPQ_GPU,
    cTPQ_GPU,
    FTLM_GPU,
    FTLM_GPU_FIXED_SZ
};

// ============================================================================
// EDConfig Implementation
// ============================================================================

// Default constructor with LANCZOS as default method
EDConfig::EDConfig() : method(DiagonalizationMethod::LANCZOS) {}

EDConfig EDConfig::fromFile(const std::string& filename) {
    EDConfig config;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open config file: " << filename << std::endl;
        return config;
    }
    
    std::string line;
    uint64_t line_num = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Parse key=value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Parse based on key
        try {
            if (key == "method") {
                auto m = ed_config::parseMethod(value);
                if (m) config.method = *m;
            }
            else if (key == "num_eigenvalues") config.diag.num_eigenvalues = std::stoi(value);
            else if (key == "max_iterations") config.diag.max_iterations = std::stoi(value);
            else if (key == "tolerance") config.diag.tolerance = std::stod(value);
            else if (key == "compute_eigenvectors") config.diag.compute_eigenvectors = (value == "true" || value == "1");
            else if (key == "shift") config.diag.shift = std::stod(value);
            else if (key == "block_size") config.diag.block_size = std::stoi(value);
            else if (key == "target_lower") config.diag.target_lower = std::stod(value);
            else if (key == "target_upper") config.diag.target_upper = std::stod(value);
            else if (key == "num_sites") config.system.num_sites = std::stoi(value);
            else if (key == "spin_length") config.system.spin_length = std::stof(value);
            else if (key == "hamiltonian_dir") config.system.hamiltonian_dir = value;
            else if (key == "use_fixed_sz") config.system.use_fixed_sz = (value == "true" || value == "1");
            else if (key == "n_up") config.system.n_up = std::stoi(value);
            else if (key == "output_dir") config.workflow.output_dir = value;
            else if (key == "num_samples") config.thermal.num_samples = std::stoi(value);
            else if (key == "temp_min") config.thermal.temp_min = std::stod(value);
            else if (key == "temp_max") config.thermal.temp_max = std::stod(value);
            else if (key == "temp_bins") config.thermal.num_temp_bins = std::stoi(value);
            else if (key == "ftlm_krylov_dim") config.thermal.ftlm_krylov_dim = std::stoi(value);
            else if (key == "ftlm_full_reorth") config.thermal.ftlm_full_reorth = (value == "true" || value == "1");
            else if (key == "ftlm_reorth_freq") config.thermal.ftlm_reorth_freq = std::stoi(value);
            else if (key == "ftlm_seed") config.thermal.ftlm_seed = std::stoul(value);
            else if (key == "ftlm_store_samples") config.thermal.ftlm_store_samples = (value == "true" || value == "1");
            else if (key == "ftlm_error_bars") config.thermal.ftlm_error_bars = (value == "true" || value == "1");
            // LTLM parameters
            else if (key == "ltlm_krylov_dim") config.thermal.ltlm_krylov_dim = std::stoi(value);
            else if (key == "ltlm_ground_krylov") config.thermal.ltlm_ground_krylov = std::stoi(value);
            else if (key == "ltlm_full_reorth") config.thermal.ltlm_full_reorth = (value == "true" || value == "1");
            else if (key == "ltlm_reorth_freq") config.thermal.ltlm_reorth_freq = std::stoi(value);
            else if (key == "ltlm_seed") config.thermal.ltlm_seed = std::stoul(value);
            else if (key == "ltlm_store_data") config.thermal.ltlm_store_data = (value == "true" || value == "1");
            // Hybrid method parameters
            else if (key == "use_hybrid_method") config.thermal.use_hybrid_method = (value == "true" || value == "1");
            else if (key == "hybrid_crossover") config.thermal.hybrid_crossover = std::stod(value);
            else if (key == "hybrid_auto_crossover") config.thermal.hybrid_auto_crossover = (value == "true" || value == "1");
            // TPQ continue-quenching parameters
            else if (key == "continue_quenching") config.thermal.continue_quenching = (value == "true" || value == "1");
            else if (key == "continue_sample") config.thermal.continue_sample = std::stoi(value);
            else if (key == "continue_beta") config.thermal.continue_beta = std::stod(value);
            else if (key == "calc_observables") config.observable.calculate = (value == "true" || value == "1");
            else if (key == "measure_spin") config.observable.measure_spin = (value == "true" || value == "1");
            else if (key == "run_standard") config.workflow.run_standard = (value == "true" || value == "1");
            else if (key == "run_symmetrized") config.workflow.run_symmetrized = (value == "true" || value == "1");
            else if (key == "run_streaming_symmetry") config.workflow.run_streaming_symmetry = (value == "true" || value == "1");
            else if (key == "compute_thermo") config.workflow.compute_thermo = (value == "true" || value == "1");
            else if (key == "compute_dynamical_response") config.workflow.compute_dynamical_response = (value == "true" || value == "1");
            else if (key == "compute_static_response") config.workflow.compute_static_response = (value == "true" || value == "1");
            // Dynamical response parameters
            else if (key == "dynamical_thermal_average") config.dynamical.thermal_average = (value == "true" || value == "1");
            else if (key == "dynamical_num_random_states") config.dynamical.num_random_states = std::stoi(value);
            else if (key == "dynamical_krylov_dim") config.dynamical.krylov_dim = std::stoi(value);
            else if (key == "dynamical_omega_min") config.dynamical.omega_min = std::stod(value);
            else if (key == "dynamical_omega_max") config.dynamical.omega_max = std::stod(value);
            else if (key == "dynamical_num_omega_points") config.dynamical.num_omega_points = std::stoi(value);
            else if (key == "dynamical_broadening") config.dynamical.broadening = std::stod(value);
            else if (key == "dynamical_temp_min") config.dynamical.temp_min = std::stod(value);
            else if (key == "dynamical_temp_max") config.dynamical.temp_max = std::stod(value);
            else if (key == "dynamical_num_temp_bins") config.dynamical.num_temp_bins = std::stoi(value);
            else if (key == "dynamical_compute_correlation") config.dynamical.compute_correlation = (value == "true" || value == "1");
            else if (key == "dynamical_operator_file") config.dynamical.operator_file = value;
            else if (key == "dynamical_operator2_file") config.dynamical.operator2_file = value;
            else if (key == "dynamical_output_prefix") config.dynamical.output_prefix = value;
            else if (key == "dynamical_random_seed") config.dynamical.random_seed = std::stoul(value);
            // Static response parameters
            else if (key == "static_num_random_states") config.static_resp.num_random_states = std::stoi(value);
            else if (key == "static_krylov_dim") config.static_resp.krylov_dim = std::stoi(value);
            else if (key == "static_temp_min") config.static_resp.temp_min = std::stod(value);
            else if (key == "static_temp_max") config.static_resp.temp_max = std::stod(value);
            else if (key == "static_num_temp_points") config.static_resp.num_temp_points = std::stoi(value);
            else if (key == "static_compute_susceptibility") config.static_resp.compute_susceptibility = (value == "true" || value == "1");
            else if (key == "static_compute_correlation") config.static_resp.compute_correlation = (value == "true" || value == "1");
            else if (key == "static_single_operator_mode") config.static_resp.single_operator_mode = (value == "true" || value == "1");
            else if (key == "static_operator_file") config.static_resp.operator_file = value;
            else if (key == "static_operator2_file") config.static_resp.operator2_file = value;
            else if (key == "static_output_prefix") config.static_resp.output_prefix = value;
            else if (key == "static_random_seed") config.static_resp.random_seed = std::stoul(value);
            else {
                std::cerr << "Warning: Unknown config key '" << key << "' at line " << line_num << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_num << ": " << e.what() << std::endl;
        }
    }
    
    return config;
}

EDConfig EDConfig::fromCommandLine(uint64_t argc, char* argv[]) {
    EDConfig config;
    
    if (argc < 2) {
        return config;  // Return default config
    }
    
    // First argument is directory
    config.system.hamiltonian_dir = argv[1];
    config.workflow.output_dir = config.system.hamiltonian_dir + "/output";
    
    // Parse remaining arguments
    for (uint64_t i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        auto parse_value = [&](const std::string& prefix) -> std::string {
            return arg.substr(prefix.length());
        };
        
        try {
            if (arg.find("--method=") == 0) {
                auto m = ed_config::parseMethod(parse_value("--method="));
                if (m) config.method = *m;
            }
            else if (arg.find("--eigenvalues=") == 0) {
                auto val = parse_value("--eigenvalues=");
                if (val == "FULL") {
                    // Will be set after num_sites is known
                    config.diag.num_eigenvalues = -1; // Special marker
                } else {
                    config.diag.num_eigenvalues = std::stoi(val);
                }
            }
            else if (arg.find("--iterations=") == 0) config.diag.max_iterations = std::stoi(parse_value("--iterations="));
            else if (arg.find("--tolerance=") == 0) config.diag.tolerance = std::stod(parse_value("--tolerance="));
            else if (arg == "--eigenvectors") config.diag.compute_eigenvectors = true;
            else if (arg.find("--shift=") == 0) config.diag.shift = std::stod(parse_value("--shift="));
            else if (arg.find("--block-size=") == 0) config.diag.block_size = std::stoi(parse_value("--block-size="));
            else if (arg.find("--target-lower=") == 0) config.diag.target_lower = std::stod(parse_value("--target-lower="));
            else if (arg.find("--target-upper=") == 0) config.diag.target_upper = std::stod(parse_value("--target-upper="));
            else if (arg.find("--num_sites=") == 0) config.system.num_sites = std::stoi(parse_value("--num_sites="));
            else if (arg.find("--spin_length=") == 0) config.system.spin_length = std::stof(parse_value("--spin_length="));
            else if (arg == "--fixed-sz") config.system.use_fixed_sz = true;
            else if (arg.find("--n-up=") == 0) config.system.n_up = std::stoi(parse_value("--n-up="));
            else if (arg.find("--output=") == 0) config.workflow.output_dir = parse_value("--output=");
            else if (arg.find("--samples=") == 0) config.thermal.num_samples = std::stoi(parse_value("--samples="));
            else if (arg.find("--temp_min=") == 0) config.thermal.temp_min = std::stod(parse_value("--temp_min="));
            else if (arg.find("--temp_max=") == 0) config.thermal.temp_max = std::stod(parse_value("--temp_max="));
            else if (arg.find("--temp_bins=") == 0) config.thermal.num_temp_bins = std::stoi(parse_value("--temp_bins="));
            else if (arg.find("--num_order=") == 0) config.thermal.num_order = std::stoi(parse_value("--num_order="));
            else if (arg.find("--measure-freq=") == 0) config.thermal.num_measure_freq = std::stoi(parse_value("--measure-freq="));
            else if (arg.find("--num_measure_freq=") == 0) config.thermal.num_measure_freq = std::stoi(parse_value("--num_measure_freq=")); // Deprecated: use --measure-freq
            else if (arg.find("--delta_tau=") == 0) config.thermal.delta_tau = std::stod(parse_value("--delta_tau="));
            else if (arg.find("--large_value=") == 0) config.thermal.large_value = std::stod(parse_value("--large_value="));
            // TPQ continue-quenching options
            else if (arg == "--continue_quenching") config.thermal.continue_quenching = true;
            else if (arg.find("--continue_sample=") == 0) config.thermal.continue_sample = std::stoi(parse_value("--continue_sample="));
            else if (arg.find("--continue_beta=") == 0) config.thermal.continue_beta = std::stod(parse_value("--continue_beta="));
            else if (arg == "--calc_observables") config.observable.calculate = true;
            else if (arg == "--measure_spin") config.observable.measure_spin = true;
            else if (arg == "--standard") config.workflow.run_standard = true;
            else if (arg == "--symmetrized") config.workflow.run_symmetrized = true;
            else if (arg == "--streaming-symmetry") config.workflow.run_streaming_symmetry = true;
            else if (arg == "--thermo") config.workflow.compute_thermo = true;
            else if (arg == "--dynamical-response") config.workflow.compute_dynamical_response = true;
            else if (arg == "--static-response") config.workflow.compute_static_response = true;
            else if (arg == "--skip_ED") config.workflow.skip_ed = true;
            else if (arg.find("--sublattice_size=") == 0) config.system.sublattice_size = std::stoi(parse_value("--sublattice_size="));
            else if (arg.find("--omega_min=") == 0) config.observable.omega_min = std::stod(parse_value("--omega_min="));
            else if (arg.find("--omega_max=") == 0) config.observable.omega_max = std::stod(parse_value("--omega_max="));
            else if (arg.find("--num_points=") == 0) config.observable.num_points = std::stoi(parse_value("--num_points="));
            else if (arg.find("--t_end=") == 0) config.observable.t_end = std::stod(parse_value("--t_end="));
            else if (arg.find("--dt=") == 0) config.observable.dt = std::stod(parse_value("--dt="));
            else if (arg.find("--max_subspace=") == 0) config.diag.max_subspace = std::stoi(parse_value("--max_subspace="));
            // FTLM options
            else if (arg.find("--ftlm-krylov=") == 0) config.thermal.ftlm_krylov_dim = std::stoi(parse_value("--ftlm-krylov="));
            else if (arg == "--ftlm-full-reorth") config.thermal.ftlm_full_reorth = true;
            else if (arg.find("--ftlm-reorth-freq=") == 0) config.thermal.ftlm_reorth_freq = std::stoi(parse_value("--ftlm-reorth-freq="));
            else if (arg.find("--ftlm-seed=") == 0) config.thermal.ftlm_seed = std::stoul(parse_value("--ftlm-seed="));
            else if (arg == "--ftlm-store-samples") config.thermal.ftlm_store_samples = true;
            else if (arg == "--ftlm-no-error-bars") config.thermal.ftlm_error_bars = false;
            // LTLM options
            else if (arg.find("--ltlm-krylov=") == 0) config.thermal.ltlm_krylov_dim = std::stoi(parse_value("--ltlm-krylov="));
            else if (arg.find("--ltlm-ground-krylov=") == 0) config.thermal.ltlm_ground_krylov = std::stoi(parse_value("--ltlm-ground-krylov="));
            else if (arg == "--ltlm-full-reorth") config.thermal.ltlm_full_reorth = true;
            else if (arg.find("--ltlm-reorth-freq=") == 0) config.thermal.ltlm_reorth_freq = std::stoi(parse_value("--ltlm-reorth-freq="));
            else if (arg.find("--ltlm-seed=") == 0) config.thermal.ltlm_seed = std::stoul(parse_value("--ltlm-seed="));
            else if (arg == "--ltlm-store-data") config.thermal.ltlm_store_data = true;
            // Hybrid LTLM/FTLM options (DEPRECATED: use --method=HYBRID instead)
            else if (arg == "--hybrid-thermal") config.thermal.use_hybrid_method = true;  // Deprecated: use --method=HYBRID
            else if (arg.find("--hybrid-crossover=") == 0) config.thermal.hybrid_crossover = std::stod(parse_value("--hybrid-crossover="));
            else if (arg == "--hybrid-auto-crossover") config.thermal.hybrid_auto_crossover = true;
            // Dynamical response options
            else if (arg == "--dyn-thermal") config.dynamical.thermal_average = true;
            else if (arg.find("--dyn-samples=") == 0) config.dynamical.num_random_states = std::stoi(parse_value("--dyn-samples="));
            else if (arg.find("--dyn-krylov=") == 0) config.dynamical.krylov_dim = std::stoi(parse_value("--dyn-krylov="));
            else if (arg.find("--dyn-omega-min=") == 0) config.dynamical.omega_min = std::stod(parse_value("--dyn-omega-min="));
            else if (arg.find("--dyn-omega-max=") == 0) config.dynamical.omega_max = std::stod(parse_value("--dyn-omega-max="));
            else if (arg.find("--dyn-omega-points=") == 0) config.dynamical.num_omega_points = std::stoi(parse_value("--dyn-omega-points="));
            else if (arg.find("--dyn-broadening=") == 0) config.dynamical.broadening = std::stod(parse_value("--dyn-broadening="));
            else if (arg.find("--dyn-temp-min=") == 0) config.dynamical.temp_min = std::stod(parse_value("--dyn-temp-min="));
            else if (arg.find("--dyn-temp-max=") == 0) config.dynamical.temp_max = std::stod(parse_value("--dyn-temp-max="));
            else if (arg.find("--dyn-temp-bins=") == 0) config.dynamical.num_temp_bins = std::stoi(parse_value("--dyn-temp-bins="));
            else if (arg == "--dyn-correlation") config.dynamical.compute_correlation = true;
            else if (arg.find("--dyn-operator=") == 0) config.dynamical.operator_file = parse_value("--dyn-operator=");
            else if (arg.find("--dyn-operator2=") == 0) config.dynamical.operator2_file = parse_value("--dyn-operator2=");
            else if (arg.find("--dyn-output=") == 0) config.dynamical.output_prefix = parse_value("--dyn-output=");
            else if (arg.find("--dyn-seed=") == 0) config.dynamical.random_seed = std::stoul(parse_value("--dyn-seed="));
            // Dynamical response configuration-based operator options
            else if (arg.find("--dyn-operator-type=") == 0) config.dynamical.operator_type = parse_value("--dyn-operator-type=");
            else if (arg.find("--dyn-basis=") == 0) config.dynamical.basis = parse_value("--dyn-basis=");
            else if (arg.find("--dyn-spin-combinations=") == 0) config.dynamical.spin_combinations = parse_value("--dyn-spin-combinations=");
            else if (arg.find("--dyn-unit-cell-size=") == 0) config.dynamical.unit_cell_size = std::stoi(parse_value("--dyn-unit-cell-size="));
            else if (arg.find("--dyn-momentum-points=") == 0) config.dynamical.momentum_points = parse_value("--dyn-momentum-points=");
            else if (arg.find("--dyn-polarization=") == 0) config.dynamical.polarization = parse_value("--dyn-polarization=");
            else if (arg.find("--dyn-theta=") == 0) config.dynamical.theta = std::stod(parse_value("--dyn-theta="));
            // GPU acceleration options
            else if (arg == "--dyn-use-gpu") config.dynamical.use_gpu = true;
            else if (arg == "--static-use-gpu") config.static_resp.use_gpu = true;
            else if (arg == "--use-gpu") {
                config.dynamical.use_gpu = true;
                config.static_resp.use_gpu = true;
            }
            // Static response options
            else if (arg.find("--static-samples=") == 0) config.static_resp.num_random_states = std::stoi(parse_value("--static-samples="));
            else if (arg.find("--static-krylov=") == 0) config.static_resp.krylov_dim = std::stoi(parse_value("--static-krylov="));
            else if (arg.find("--static-temp-min=") == 0) config.static_resp.temp_min = std::stod(parse_value("--static-temp-min="));
            else if (arg.find("--static-temp-max=") == 0) config.static_resp.temp_max = std::stod(parse_value("--static-temp-max="));
            else if (arg.find("--static-temp-points=") == 0) config.static_resp.num_temp_points = std::stoi(parse_value("--static-temp-points="));
            else if (arg == "--static-no-susceptibility") config.static_resp.compute_susceptibility = false;
            else if (arg == "--static-correlation") config.static_resp.compute_correlation = true;
            else if (arg == "--static-expectation") {
                config.static_resp.single_operator_mode = true;
                config.workflow.compute_static_response = true;
            }
            else if (arg.find("--static-operator=") == 0) config.static_resp.operator_file = parse_value("--static-operator=");
            else if (arg.find("--static-operator2=") == 0) config.static_resp.operator2_file = parse_value("--static-operator2=");
            else if (arg.find("--static-output=") == 0) config.static_resp.output_prefix = parse_value("--static-output=");
            else if (arg.find("--static-seed=") == 0) config.static_resp.random_seed = std::stoul(parse_value("--static-seed="));
            // Static response configuration-based operator options
            else if (arg.find("--static-operator-type=") == 0) config.static_resp.operator_type = parse_value("--static-operator-type=");
            else if (arg.find("--static-basis=") == 0) config.static_resp.basis = parse_value("--static-basis=");
            else if (arg.find("--static-spin-combinations=") == 0) config.static_resp.spin_combinations = parse_value("--static-spin-combinations=");
            else if (arg.find("--static-unit-cell-size=") == 0) config.static_resp.unit_cell_size = std::stoi(parse_value("--static-unit-cell-size="));
            else if (arg.find("--static-momentum-points=") == 0) config.static_resp.momentum_points = parse_value("--static-momentum-points=");
            else if (arg.find("--static-polarization=") == 0) config.static_resp.polarization = parse_value("--static-polarization=");
            else if (arg.find("--static-theta=") == 0) config.static_resp.theta = std::stod(parse_value("--static-theta="));
            // ARPACK options
            else if (arg.find("--arpack-which=") == 0) config.arpack.which = parse_value("--arpack-which=");
            else if (arg.find("--arpack-ncv=") == 0) config.arpack.ncv = std::stoi(parse_value("--arpack-ncv="));
            else if (arg.find("--arpack-max-restarts=") == 0) config.arpack.max_restarts = std::stoi(parse_value("--arpack-max-restarts="));
            else if (arg == "--arpack-shift-invert") config.arpack.shift_invert = true;
            else if (arg.find("--arpack-sigma=") == 0) config.arpack.sigma = std::stod(parse_value("--arpack-sigma="));
            else if (arg == "--arpack-verbose") config.arpack.verbose = true;
            else if (arg.find("--config=") == 0) {
                // Load from config file and merge
                auto file_config = EDConfig::fromFile(parse_value("--config="));
                config = file_config.merge(config); // Command line takes precedence
            }
            else if (arg != "--help") {
                std::cerr << "Warning: Unknown option: " << arg << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing argument '" << arg << "': " << e.what() << std::endl;
        }
    }
    
    // Auto-detect num_sites if not specified
    if (config.system.num_sites == 0) {
        config.autoDetectNumSites();
    }
    
    // Handle FULL spectrum case
    if (config.diag.num_eigenvalues == -1 && config.system.num_sites > 0) {
        config.diag.num_eigenvalues = (1ULL << config.system.num_sites);
    }
    
    // Auto-enable skip_ed if only response calculations are requested
    bool only_response = (config.workflow.compute_dynamical_response || config.workflow.compute_static_response) &&
                        !config.workflow.run_standard && 
                        !config.workflow.run_symmetrized && 
                        !config.workflow.run_streaming_symmetry &&
                        !config.workflow.compute_thermo;
    
    if (only_response && !config.workflow.skip_ed) {
        std::cout << "Note: Only response calculations requested. Skipping diagonalization (use --standard/--symmetrized to override).\n";
        config.workflow.skip_ed = true;
    }
    
    // Default to standard workflow if nothing specified (and skip_ed not set)
    if (!config.workflow.run_standard && !config.workflow.run_symmetrized && 
        !config.workflow.run_streaming_symmetry && !config.workflow.skip_ed) {
        config.workflow.run_standard = true;
    }
    
    return config;
}

EDConfig& EDConfig::merge(const EDConfig& other) {
    // Simple merge: other overrides this where values differ from defaults
    // This is a simplified version - could be more sophisticated
    method = other.method;
    
    // Merge diag
    if (other.diag.num_eigenvalues != 1) diag.num_eigenvalues = other.diag.num_eigenvalues;
    if (other.diag.max_iterations != 10000) diag.max_iterations = other.diag.max_iterations;
    if (other.diag.tolerance != 1e-10) diag.tolerance = other.diag.tolerance;
    if (other.diag.compute_eigenvectors) diag.compute_eigenvectors = true;
    
    // Merge system
    if (other.system.num_sites != 0) system.num_sites = other.system.num_sites;
    if (other.system.spin_length != 0.5f) system.spin_length = other.system.spin_length;
    if (!other.system.hamiltonian_dir.empty()) system.hamiltonian_dir = other.system.hamiltonian_dir;
    
    // Merge workflow
    if (other.workflow.run_standard) workflow.run_standard = true;
    if (other.workflow.run_symmetrized) workflow.run_symmetrized = true;
    if (other.workflow.run_streaming_symmetry) workflow.run_streaming_symmetry = true;
    if (other.workflow.compute_thermo) workflow.compute_thermo = true;
    if (!other.workflow.output_dir.empty()) workflow.output_dir = other.workflow.output_dir;
    
    return *this;
}

bool EDConfig::validate(std::ostream& err) const {
    bool valid = true;
    
    if (system.num_sites == 0) {
        err << "Error: num_sites must be specified or auto-detected\n";
        valid = false;
    }
    
    if (system.hamiltonian_dir.empty()) {
        err << "Error: hamiltonian_dir must be specified\n";
        valid = false;
    }
    
    if (diag.num_eigenvalues < 1) {
        err << "Error: num_eigenvalues must be >= 1\n";
        valid = false;
    }
    
    if (diag.tolerance <= 0) {
        err << "Error: tolerance must be positive\n";
        valid = false;
    }
    
    return valid;
}

void EDConfig::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not write config to " << filename << std::endl;
        return;
    }
    
    file << "# ED Configuration\n";
    file << "# Generated configuration file\n\n";
    
    file << "[Diagonalization]\n";
    file << "method = " << ed_config::methodToString(method) << "\n";
    file << "num_eigenvalues = " << diag.num_eigenvalues << "\n";
    file << "max_iterations = " << diag.max_iterations << "\n";
    file << "tolerance = " << diag.tolerance << "\n";
    file << "compute_eigenvectors = " << (diag.compute_eigenvectors ? "true" : "false") << "\n\n";
    
    file << "[System]\n";
    file << "num_sites = " << system.num_sites << "\n";
    file << "spin_length = " << system.spin_length << "\n";
    file << "hamiltonian_dir = " << system.hamiltonian_dir << "\n\n";
    
    file << "[Workflow]\n";
    file << "output_dir = " << workflow.output_dir << "\n";
    file << "run_standard = " << (workflow.run_standard ? "true" : "false") << "\n";
    file << "run_symmetrized = " << (workflow.run_symmetrized ? "true" : "false") << "\n";
    file << "run_streaming_symmetry = " << (workflow.run_streaming_symmetry ? "true" : "false") << "\n";
    file << "compute_thermo = " << (workflow.compute_thermo ? "true" : "false") << "\n";
}

void EDConfig::print(std::ostream& out) const {
    out << "========================================\n";
    out << "ED Configuration Summary\n";
    out << "========================================\n\n";
    
    out << "Method: " << ed_config::methodToString(method) << "\n";
    out << "System: " << system.num_sites << " sites, spin = " << system.spin_length << "\n";
    
    if (system.use_fixed_sz) {
        int64_t n_up_actual = (system.n_up >= 0) ? system.n_up : system.num_sites / 2;
        double sz = n_up_actual - system.num_sites / 2.0;
        out << "Fixed Sz: n_up = " << n_up_actual << " (Sz = " << sz << ")\n";
        
        // Calculate dimension reduction
        auto binomial = [](uint64_t n, uint64_t k) {
            if (k > n || k < 0) return 0;
            if (k == 0 || k == n) return 1;
            long long result = 1;
            for (uint64_t i = 1; i <= k; ++i) {
                result = result * (n - k + i) / i;
            }
            return (int)result;
        };
        uint64_t full_dim = 1 << system.num_sites;
        uint64_t fixed_dim = binomial(system.num_sites, n_up_actual);
        out << "Hilbert space: " << fixed_dim << " (reduced from " << full_dim 
            << ", factor: " << (double)full_dim / fixed_dim << "x)\n";
    }
    
    out << "Eigenvalues: " << diag.num_eigenvalues << " (tol=" << diag.tolerance << ")\n";
    out << "Output: " << workflow.output_dir << "\n";
    
    if (workflow.run_standard) out << "  - Running standard diagonalization\n";
    if (workflow.run_symmetrized) out << "  - Running symmetrized diagonalization\n";
    if (workflow.run_streaming_symmetry) out << "  - Running streaming symmetry diagonalization (memory-efficient)\n";
    if (workflow.compute_thermo) out << "  - Computing thermodynamics\n";
    if (observable.calculate) out << "  - Calculating observables\n";
    if (observable.measure_spin) out << "  - Measuring spin expectations\n";
    
    out << "========================================\n";
}

bool EDConfig::autoDetectNumSites() {
    std::string positions_file = system.hamiltonian_dir + "/positions.dat";
    std::ifstream file(positions_file);
    
    if (!file.is_open()) {
        return false;
    }
    
    uint64_t max_site_id = 0;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        uint64_t site_id;
        if (iss >> site_id) {
            max_site_id = std::max(max_site_id, site_id);
        }
    }
    
    if (max_site_id >= 0) {
        system.num_sites = max_site_id + 1;
        std::cout << "Auto-detected num_sites = " << system.num_sites << " from positions.dat\n";
        return true;
    }
    
    return false;
}

// ============================================================================
// Conversion Utilities
// ============================================================================

namespace ed_config {

std::optional<DiagonalizationMethod> parseMethod(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Standard Lanczos variants
    if (lower == "lanczos") return DiagonalizationMethod::LANCZOS;
    if (lower == "lanczos_selective") return DiagonalizationMethod::LANCZOS_SELECTIVE;
    if (lower == "lanczos_no_ortho") return DiagonalizationMethod::LANCZOS_NO_ORTHO;
    if (lower == "block_lanczos") return DiagonalizationMethod::BLOCK_LANCZOS;
    if (lower == "chebyshev_filtered") return DiagonalizationMethod::CHEBYSHEV_FILTERED;
    if (lower == "chebyshev") return DiagonalizationMethod::CHEBYSHEV_FILTERED;
    if (lower == "shift_invert") return DiagonalizationMethod::SHIFT_INVERT;
    if (lower == "shift_invert_robust") return DiagonalizationMethod::SHIFT_INVERT_ROBUST;
    if (lower == "krylov_schur") return DiagonalizationMethod::KRYLOV_SCHUR;
    if (lower == "irl") return DiagonalizationMethod::IMPLICIT_RESTART_LANCZOS;
    if (lower == "trlan") return DiagonalizationMethod::THICK_RESTART_LANCZOS;
    
    // Conjugate Gradient variants
    if (lower == "bicg") return DiagonalizationMethod::BICG;
    if (lower == "lobpcg") return DiagonalizationMethod::LOBPCG;
    
    // Other iterative methods
    if (lower == "davidson") return DiagonalizationMethod::DAVIDSON;
    
    // Full diagonalization
    if (lower == "full") return DiagonalizationMethod::FULL;
    if (lower == "oss") return DiagonalizationMethod::OSS;
    
    // Thermal methods
    if (lower == "mtpq") return DiagonalizationMethod::mTPQ;
    if (lower == "mtpq_mpi") return DiagonalizationMethod::mTPQ_MPI;
    if (lower == "ctpq") return DiagonalizationMethod::cTPQ;
    if (lower == "mtpq_cuda") return DiagonalizationMethod::mTPQ_CUDA;
    if (lower == "ftlm") return DiagonalizationMethod::FTLM;
    if (lower == "ltlm") return DiagonalizationMethod::LTLM;
    if (lower == "hybrid") return DiagonalizationMethod::HYBRID;
    
    // ARPACK methods
    if (lower == "arpack" || lower == "arpack_sm") return DiagonalizationMethod::ARPACK_SM;
    if (lower == "arpack_lm") return DiagonalizationMethod::ARPACK_LM;
    if (lower == "arpack_shift_invert") return DiagonalizationMethod::ARPACK_SHIFT_INVERT;
    if (lower == "arpack_advanced") return DiagonalizationMethod::ARPACK_ADVANCED;
    
    // GPU methods
    if (lower == "lanczos_gpu") return DiagonalizationMethod::LANCZOS_GPU;
    if (lower == "lanczos_gpu_fixed_sz") return DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ;
    if (lower == "davidson_gpu") return DiagonalizationMethod::DAVIDSON_GPU;
    if (lower == "lobpcg_gpu") return DiagonalizationMethod::LOBPCG_GPU;
    if (lower == "mtpq_gpu") return DiagonalizationMethod::mTPQ_GPU;
    if (lower == "ctpq_gpu") return DiagonalizationMethod::cTPQ_GPU;
    if (lower == "ftlm_gpu") return DiagonalizationMethod::FTLM_GPU;
    if (lower == "ftlm_gpu_fixed_sz") return DiagonalizationMethod::FTLM_GPU_FIXED_SZ;
    std::cerr << "Warning: Unknown method '" << str << "', using LANCZOS\n";
    return std::nullopt;
}

std::string methodToString(DiagonalizationMethod method) {
    switch (method) {
        // Standard Lanczos variants
        case DiagonalizationMethod::LANCZOS: return "LANCZOS";
        case DiagonalizationMethod::LANCZOS_SELECTIVE: return "LANCZOS_SELECTIVE";
        case DiagonalizationMethod::LANCZOS_NO_ORTHO: return "LANCZOS_NO_ORTHO";
        case DiagonalizationMethod::BLOCK_LANCZOS: return "BLOCK_LANCZOS";
        case DiagonalizationMethod::CHEBYSHEV_FILTERED: return "CHEBYSHEV_FILTERED";
        case DiagonalizationMethod::SHIFT_INVERT: return "SHIFT_INVERT";
        case DiagonalizationMethod::SHIFT_INVERT_ROBUST: return "SHIFT_INVERT_ROBUST";
        case DiagonalizationMethod::KRYLOV_SCHUR: return "KRYLOV_SCHUR";
        case DiagonalizationMethod::IMPLICIT_RESTART_LANCZOS: return "IMPLICIT_RESTART_LANCZOS";
        case DiagonalizationMethod::THICK_RESTART_LANCZOS: return "THICK_RESTART_LANCZOS";
        
        // Conjugate Gradient variants
        case DiagonalizationMethod::BICG: return "BICG";
        case DiagonalizationMethod::LOBPCG: return "LOBPCG";
        
        // Other iterative methods
        case DiagonalizationMethod::DAVIDSON: return "DAVIDSON";
        
        // Full diagonalization
        case DiagonalizationMethod::FULL: return "FULL";
        case DiagonalizationMethod::OSS: return "OSS";
        
        // Thermal methods
        case DiagonalizationMethod::mTPQ: return "mTPQ";
        case DiagonalizationMethod::mTPQ_MPI: return "mTPQ_MPI";
        case DiagonalizationMethod::cTPQ: return "cTPQ";
        case DiagonalizationMethod::mTPQ_CUDA: return "mTPQ_CUDA";
        case DiagonalizationMethod::FTLM: return "FTLM";
        case DiagonalizationMethod::LTLM: return "LTLM";
        case DiagonalizationMethod::HYBRID: return "HYBRID";
        
        // ARPACK methods
        case DiagonalizationMethod::ARPACK_SM: return "ARPACK_SM";
        case DiagonalizationMethod::ARPACK_LM: return "ARPACK_LM";
        case DiagonalizationMethod::ARPACK_SHIFT_INVERT: return "ARPACK_SHIFT_INVERT";
        case DiagonalizationMethod::ARPACK_ADVANCED: return "ARPACK_ADVANCED";
        
        // GPU methods
        case DiagonalizationMethod::LANCZOS_GPU: return "LANCZOS_GPU";
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ: return "LANCZOS_GPU_FIXED_SZ";
        case DiagonalizationMethod::DAVIDSON_GPU: return "DAVIDSON_GPU";
        case DiagonalizationMethod::LOBPCG_GPU: return "LOBPCG_GPU";
        case DiagonalizationMethod::mTPQ_GPU: return "mTPQ_GPU";
        case DiagonalizationMethod::cTPQ_GPU: return "cTPQ_GPU";
        case DiagonalizationMethod::FTLM_GPU: return "FTLM_GPU";
        case DiagonalizationMethod::FTLM_GPU_FIXED_SZ: return "FTLM_GPU_FIXED_SZ";
        
        default: return "UNKNOWN";
    }
}

EDConfig defaultConfigFor(DiagonalizationMethod method) {
    EDConfig config(method);
    
    switch (method) {
        case DiagonalizationMethod::mTPQ:
        case DiagonalizationMethod::cTPQ:
            config.thermal.num_samples = 10;
            config.workflow.compute_thermo = true;
            break;
            
        case DiagonalizationMethod::FULL:
            config.diag.num_eigenvalues = -1; // Will be set based on system size
            config.workflow.compute_thermo = true;
            break;
            
        case DiagonalizationMethod::ARPACK_ADVANCED:
            config.arpack.max_restarts = 3;
            config.arpack.two_phase_refine = true;
            break;
            
        default:
            break;
    }
    
    return config;
}

/**
 * @brief Get detailed parameter information for a diagonalization method
 */
std::string getMethodParameterInfo(DiagonalizationMethod method) {
    std::ostringstream info;
    
    info << "\n========================================\n";
    info << "Method: " << methodToString(method) << "\n";
    info << "========================================\n\n";
    
    switch (method) {
        case DiagonalizationMethod::LANCZOS:
            info << "Standard Lanczos algorithm with full reorthogonalization.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Ground state and low-lying excited states\n";
            break;
            
        case DiagonalizationMethod::LANCZOS_SELECTIVE:
            info << "Lanczos with selective reorthogonalization for improved stability.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: When standard Lanczos has convergence issues\n";
            break;
            
        case DiagonalizationMethod::LANCZOS_NO_ORTHO:
            info << "Lanczos without reorthogonalization (fastest but may lose orthogonality).\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Quick estimates with well-conditioned Hamiltonians\n";
            info << "Warning: May produce inaccurate results for ill-conditioned problems\n";
            break;
            
        case DiagonalizationMethod::BLOCK_LANCZOS:
            info << "Block Lanczos for finding multiple eigenvalues simultaneously.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --block_size=<b>      Block size (default: 10)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Degenerate or near-degenerate eigenvalues\n";
            break;
            
        case DiagonalizationMethod::CHEBYSHEV_FILTERED:
            info << "Chebyshev Filtered Lanczos for spectral slicing and interior eigenvalues.\n\n";
            info << "Uses Chebyshev polynomial filtering to focus on eigenvalues within a target\n";
            info << "energy window. Automatically estimates spectral bounds if not provided.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --target_lower=<E>    Lower bound of target energy window (default: auto)\n";
            info << "  --target_upper=<E>    Upper bound of target energy window (default: auto)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Computing eigenvalues in specific energy ranges\n";
            info << "          Interior spectrum without shift-invert\n";
            info << "          Avoiding eigenvalues outside target window\n";
            info << "\nNote: If target range not specified, computes lowest eigenvalues\n";
            info << "      Filter degree automatically determined from spectral properties\n";
            break;
            
        case DiagonalizationMethod::SHIFT_INVERT:
            info << "Shift-invert Lanczos for finding eigenvalues near a target value.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --shift=<sigma>       Target shift value (default: 0.0)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Interior eigenvalues, excited states at specific energies\n";
            info << "Note: Requires solving linear systems (H - sigma*I)x = b\n";
            break;
            
        case DiagonalizationMethod::SHIFT_INVERT_ROBUST:
            info << "Robust shift-invert (currently falls back to standard SHIFT_INVERT).\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --shift=<sigma>       Target shift value (default: 0.0)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            break;
            
        case DiagonalizationMethod::KRYLOV_SCHUR:
            info << "Krylov-Schur method (implicitly restarted Lanczos with Schur form).\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Large-scale problems requiring multiple restarts\n";
            break;
            
        case DiagonalizationMethod::IMPLICIT_RESTART_LANCZOS:
            info << "Implicitly Restarted Lanczos Algorithm (IRLA).\n\n";
            info << "Uses implicit filtering with polynomial restarts to compute eigenvalues.\n";
            info << "More memory efficient than thick restart but doesn't preserve converged vectors.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum Krylov space dimension (default: 100)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Memory-constrained problems, fast convergence to few eigenvalues\n";
            break;
            
        case DiagonalizationMethod::THICK_RESTART_LANCZOS:
            info << "Thick Restart Lanczos Algorithm with Locking.\n\n";
            info << "Preserves converged eigenvectors and uses refined Ritz vectors for restart.\n";
            info << "Superior stability and convergence compared to implicit restart.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum Krylov space dimension (default: 100)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Computing many eigenvalues, better stability, problems with clusters\n";
            info << "Features: Converged vector locking, Rayleigh quotient refinement\n";
            break;
            
            
        case DiagonalizationMethod::BICG:
            info << "Biconjugate gradient method.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "\nBest for: Specialized applications\n";
            break;
            
        case DiagonalizationMethod::LOBPCG:
            info << "Locally optimal block preconditioned conjugate gradient.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Multiple eigenvalues with good preconditioning\n";
            break;
            
        case DiagonalizationMethod::DAVIDSON:
            info << "Davidson method for interior and exterior eigenvalues.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --max_subspace=<m>    Maximum subspace size (default: 100)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "\nBest for: Low-lying eigenvalues with controlled memory usage\n";
            break;
            
        case DiagonalizationMethod::FULL:
            info << "Full diagonalization using dense eigensolver.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues (default: all)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Complete spectrum, small systems (< 10^5 dimension)\n";
            info << "Warning: Memory intensive - stores full matrix\n";
            break;
            
        case DiagonalizationMethod::OSS:
            info << "Optimal spectrum solver with adaptive slicing.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --iterations=<n>      Maximum iterations per slice (default: 100000)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Complete spectrum with memory constraints\n";
            info << "Note: Automatically determines energy slicing\n";
            break;
            
        case DiagonalizationMethod::mTPQ:
            info << "Microcanonical Thermal Pure Quantum states method.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --samples=<n>         Number of random samples (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --temp_min=<T>        Minimum temperature (default: 1e-3)\n";
            info << "  --temp_max=<T>        Maximum temperature (default: 20.0)\n";
            info << "  --temp_bins=<n>       Number of temperature points (default: 100)\n";
            info << "  --calc_observables    Calculate thermal expectation values\n";
            info << "  --measure_spin        Measure spin correlations\n";
            info << "\nBest for: Thermal properties at finite temperature\n";
            break;
            
        case DiagonalizationMethod::cTPQ:
            info << "Canonical Thermal Pure Quantum states method.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --samples=<n>         Number of random samples (default: 1)\n";
            info << "  --num_order=<k>       Taylor expansion order (default: 100)\n";
            info << "  --delta_tau=<dt>      Imaginary time step (default: 1e-2)\n";
            info << "  --temp_min=<T>        Minimum temperature (default: 1e-3)\n";
            info << "  --temp_max=<T>        Maximum temperature (default: 20.0)\n";
            info << "  --temp_bins=<n>       Number of temperature points (default: 100)\n";
            info << "  --calc_observables    Calculate thermal expectation values\n";
            info << "  --measure_spin        Measure spin correlations\n";
            info << "\nBest for: Canonical ensemble thermal properties\n";
            break;
            
        case DiagonalizationMethod::mTPQ_MPI:
            info << "MPI-parallel microcanonical TPQ (requires MPI build).\n\n";
            info << "Requires: MPI-enabled build\n";
            info << "Status: Not available in current build\n";
            break;
            
        case DiagonalizationMethod::mTPQ_CUDA:
            info << "GPU-accelerated microcanonical TPQ (requires CUDA build).\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Status: Not available in current build\n";
            break;
            
        case DiagonalizationMethod::FTLM:
            info << "Finite Temperature Lanczos Method (FTLM).\n\n";
            info << "Computes thermodynamic properties (energy, entropy, specific heat, free energy)\n";
            info << "at finite temperature without computing the full spectrum. Uses random sampling\n";
            info << "and Krylov subspace projections.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --samples=<n>         Number of random samples (default: 10)\n";
            info << "  --ftlm-krylov=<m>     Krylov dimension per sample (default: 100)\n";
            info << "  --temp_min=<T>        Minimum temperature (default: 1e-3)\n";
            info << "  --temp_max=<T>        Maximum temperature (default: 20.0)\n";
            info << "  --temp_bins=<n>       Number of temperature points (default: 100)\n";
            info << "  --ftlm-full-reorth    Use full reorthogonalization (slower, more stable)\n";
            info << "  --ftlm-reorth-freq=<k>  Reorthogonalization frequency (default: 10)\n";
            info << "  --ftlm-seed=<seed>    Random seed (0 = random, default: 0)\n";
            info << "  --ftlm-store-samples  Store per-sample intermediate data for debugging\n";
            info << "  --ftlm-no-error-bars  Disable error bar computation\n";
            info << "\nOutput:\n";
            info << "  Saves to: output_dir/thermo/ftlm_thermo.txt\n";
            info << "  Format: Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error\n";
            info << "\nBest for: Finite-temperature thermodynamics without full spectrum\n";
            info << "Advantages: Memory efficient, scales to larger systems than FULL diagonalization\n";
            info << "Note: Accuracy improves with more samples and larger Krylov dimension\n";
            break;
        
        case DiagonalizationMethod::LTLM:
            info << "Low Temperature Lanczos Method (LTLM).\n\n";
            info << "Specialized method for low-temperature thermodynamics. First finds the ground\n";
            info << "state, then builds Krylov subspace from it to capture low-lying excitations.\n";
            info << "More accurate than FTLM at low temperatures.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --ltlm-krylov=<m>        Krylov dimension for excitations (default: 200)\n";
            info << "  --ltlm-ground-krylov=<m> Krylov dimension for ground state (default: 100)\n";
            info << "  --temp_min=<T>           Minimum temperature (default: 1e-3)\n";
            info << "  --temp_max=<T>           Maximum temperature (default: 20.0)\n";
            info << "  --temp_bins=<n>          Number of temperature points (default: 100)\n";
            info << "  --ltlm-full-reorth       Use full reorthogonalization (slower, more stable)\n";
            info << "  --ltlm-reorth-freq=<k>   Reorthogonalization frequency (default: 10)\n";
            info << "  --ltlm-seed=<seed>       Random seed for initial state (0 = random, default: 0)\n";
            info << "  --ltlm-store-data        Store intermediate data (spectrum, etc.)\n";
            info << "\nHybrid LTLM/FTLM Mode:\n";
            info << "  --hybrid-thermal         Use hybrid method (LTLM at low T, FTLM at high T) [DEPRECATED: use --method=HYBRID]\n";
            info << "  --hybrid-crossover=<T>   Temperature crossover (default: 1.0)\n";
            info << "  --hybrid-auto-crossover  Automatically determine optimal crossover temperature\n";
            info << "\nOutput:\n";
            info << "  Saves to: output_dir/thermo/ltlm_thermo.txt (or hybrid_thermo.txt for hybrid)\n";
            info << "  Format: Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error\n";
            info << "\nBest for: Low-temperature thermodynamics where ground state dominates\n";
            info << "Advantages: More accurate than FTLM at low T, deterministic ground state\n";
            info << "Note: Combine with FTLM using --hybrid-thermal for full temperature range\n";
            break;
            
        case DiagonalizationMethod::ARPACK_SM:
            info << "ARPACK smallest real (algebraically smallest) eigenvalues.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Ground state (most negative eigenvalues) using ARPACK\n";
            info << "Note: Uses SR (Smallest Real) internally for Hermitian matrices\n";
            break;
            
        case DiagonalizationMethod::ARPACK_LM:
            info << "ARPACK largest real (algebraically largest) eigenvalues.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Highest energy states (most positive eigenvalues) using ARPACK\n";
            info << "Note: Uses LR (Largest Real) internally for Hermitian matrices\n";
            break;
            
        case DiagonalizationMethod::ARPACK_SHIFT_INVERT:
            info << "ARPACK with shift-invert mode.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues to compute (default: 1)\n";
            info << "  --shift=<sigma>       Shift value (default: 0.0)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nBest for: Eigenvalues near shift value using ARPACK\n";
            break;
            
        case DiagonalizationMethod::ARPACK_ADVANCED:
            info << "ARPACK with advanced multi-attempt strategy.\n\n";
            info << "Configurable Parameters:\n";
            info << "  --eigenvalues=<n>     Number of eigenvalues (default: 1)\n";
            info << "  --iterations=<n>      Maximum iterations (default: 100000)\n";
            info << "  --tolerance=<tol>     Convergence tolerance (default: 1e-10)\n";
            info << "  --eigenvectors        Compute and save eigenvectors\n";
            info << "\nAdvanced ARPACK Options (requires config file):\n";
            info << "  arpack_which          Which eigenvalues: SR (ground state), LR (excited), SM, LM (default: SR)\n";
            info << "  arpack_ncv            Number of Lanczos vectors (default: auto)\n";
            info << "  arpack_max_restarts   Maximum restarts (default: 2)\n";
            info << "  arpack_ncv_growth     NCV growth factor (default: 1.5)\n";
            info << "  arpack_auto_enlarge_ncv      Auto-enlarge NCV (default: true)\n";
            info << "  arpack_two_phase_refine      Two-phase refinement (default: true)\n";
            info << "  arpack_relaxed_tol           Relaxed tolerance (default: 1e-6)\n";
            info << "  arpack_shift_invert          Use shift-invert (default: false)\n";
            info << "  arpack_sigma                 Shift value (default: 0.0)\n";
            info << "  arpack_auto_switch_shift_invert  Auto-switch mode (default: true)\n";
            info << "  arpack_adaptive_inner_tol    Adaptive tolerance (default: true)\n";
            info << "\nBest for: Difficult convergence cases requiring fine-tuning\n";
            break;
            
        case DiagonalizationMethod::LANCZOS_GPU:
            info << "GPU-accelerated Lanczos (requires CUDA build).\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  - num_eigenvalues: Number of eigenvalues to compute\n";
            info << "  - max_iterations: Maximum Lanczos iterations\n";
            info << "  - tolerance: Convergence tolerance\n";
            info << "  - compute_eigenvectors: Whether to compute eigenvectors\n\n";
            info << "Best for: Large systems requiring GPU acceleration\n";
            break;
            
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
            info << "GPU Lanczos for fixed Sz sector (requires CUDA build).\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  - n_up: Number of up spins (determines Sz sector)\n";
            info << "  - num_eigenvalues: Number of eigenvalues to compute\n";
            info << "  - max_iterations: Maximum Lanczos iterations\n";
            info << "  - tolerance: Convergence tolerance\n\n";
            info << "Best for: Fixed Sz calculations with GPU acceleration\n";
            break;
            
        case DiagonalizationMethod::DAVIDSON_GPU:
            info << "GPU-accelerated Davidson method (requires CUDA build).\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  - num_eigenvalues: Number of eigenvalues to compute\n";
            info << "  - max_iterations: Maximum iterations\n";
            info << "  - max_subspace: Maximum subspace dimension\n";
            info << "  - tolerance: Convergence tolerance\n";
            info << "  - compute_eigenvectors: Whether to compute eigenvectors\n\n";
            info << "Best for: GPU-accelerated eigenvalue calculations with subspace expansion\n";
            break;
            
        case DiagonalizationMethod::LOBPCG_GPU:
            info << "GPU-accelerated LOBPCG method (requires CUDA build).\n";
            info << "*** DEPRECATED: This method now redirects to DAVIDSON_GPU ***\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  - num_eigenvalues: Number of eigenvalues to compute\n";
            info << "  - max_iterations: Maximum iterations\n";
            info << "  - tolerance: Convergence tolerance\n";
            info << "  - compute_eigenvectors: Whether to compute eigenvectors\n\n";
            info << "Note: LOBPCG_GPU has been retired due to numerical stability issues.\n";
            info << "      It now redirects to DAVIDSON_GPU which provides superior accuracy.\n";
            info << "      Please use DAVIDSON_GPU directly for new projects.\n";
            break;
            
        case DiagonalizationMethod::mTPQ_GPU:
            info << "GPU-accelerated microcanonical TPQ (requires CUDA build).\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  - num_samples: Number of TPQ samples\n";
            info << "  - max_iterations: Maximum iterations per sample\n";
            info << "  - num_measure_freq: Measurement frequency\n";
            info << "  - large_value: Large value parameter for TPQ\n\n";
            info << "Best for: GPU-accelerated finite-temperature calculations (microcanonical)\n";
            break;
            
        case DiagonalizationMethod::cTPQ_GPU:
            info << "GPU-accelerated canonical TPQ (requires CUDA build).\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  - num_samples: Number of TPQ samples\n";
            info << "  - temp_max: Maximum temperature\n";
            info << "  - num_measure_freq: Measurement frequency\n";
            info << "  - delta_tau: Imaginary time step\n";
            info << "  - num_order: Order parameter for TPQ\n\n";
            info << "Best for: GPU-accelerated finite-temperature calculations (canonical)\n";
            break;
            
        case DiagonalizationMethod::FTLM_GPU:
            info << "GPU-accelerated Finite Temperature Lanczos Method (requires CUDA build).\n\n";
            info << "Computes thermodynamic properties (energy, entropy, specific heat, free energy)\n";
            info << "at finite temperature using GPU-accelerated Lanczos iterations. Significantly\n";
            info << "faster than CPU FTLM for large systems.\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  --samples=<n>         Number of random samples (default: 10)\n";
            info << "  --ftlm-krylov=<m>     Krylov dimension per sample (default: 100)\n";
            info << "  --temp_min=<T>        Minimum temperature (default: 1e-3)\n";
            info << "  --temp_max=<T>        Maximum temperature (default: 20.0)\n";
            info << "  --temp_bins=<n>       Number of temperature points (default: 100)\n";
            info << "  --ftlm-full-reorth    Use full reorthogonalization (slower, more stable)\n";
            info << "  --ftlm-reorth-freq=<k>  Reorthogonalization frequency (default: 10)\n";
            info << "  --ftlm-seed=<seed>    Random seed (0 = random, default: 0)\n";
            info << "\nOutput:\n";
            info << "  Saves to: output_dir/thermo/ftlm_gpu_thermo.txt\n";
            info << "  Format: Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error\n";
            info << "\nBest for: GPU-accelerated finite-temperature thermodynamics\n";
            info << "Advantages: Much faster than CPU FTLM, scales to larger systems\n";
            info << "Note: Requires sufficient GPU memory for Hamiltonian and Lanczos vectors\n";
            break;
            
        case DiagonalizationMethod::FTLM_GPU_FIXED_SZ:
            info << "GPU-accelerated FTLM for fixed Sz sector (requires CUDA build).\n\n";
            info << "Computes thermodynamic properties in a specific Sz sector using GPU acceleration.\n";
            info << "Useful when only a particular spin sector is of interest.\n\n";
            info << "Requires: CUDA-enabled build\n";
            info << "Configurable Parameters:\n";
            info << "  --n_up=<n>            Number of up spins (determines Sz sector)\n";
            info << "  --samples=<n>         Number of random samples (default: 10)\n";
            info << "  --ftlm-krylov=<m>     Krylov dimension per sample (default: 100)\n";
            info << "  --temp_min=<T>        Minimum temperature (default: 1e-3)\n";
            info << "  --temp_max=<T>        Maximum temperature (default: 20.0)\n";
            info << "  --temp_bins=<n>       Number of temperature points (default: 100)\n";
            info << "  --ftlm-full-reorth    Use full reorthogonalization (slower, more stable)\n";
            info << "  --ftlm-reorth-freq=<k>  Reorthogonalization frequency (default: 10)\n";
            info << "  --ftlm-seed=<seed>    Random seed (0 = random, default: 0)\n";
            info << "\nOutput:\n";
            info << "  Saves to: output_dir/thermo/ftlm_gpu_fixedsz_thermo.txt\n";
            info << "  Format: Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error\n";
            info << "\nBest for: Fixed Sz sector thermodynamics with GPU acceleration\n";
            info << "Advantages: Reduced Hilbert space, faster than full space calculations\n";
            break;
            
        default:
            info << "No detailed information available for this method.\n";
            break;
    }
    
    info << "\n========================================\n";
    return info.str();
}

} // namespace ed_config

// ============================================================================
// CommandLineParser Implementation
// ============================================================================

CommandLineParser& CommandLineParser::addOption(
    const std::string& long_name,
    const std::string& short_name,
    const std::string& description,
    bool has_value,
    bool required,
    const std::string& category
) {
    Option opt;
    opt.long_name = long_name;
    opt.short_name = short_name;
    opt.description = description;
    opt.has_value = has_value;
    opt.required = required;
    opt.category = category;
    options_.push_back(opt);
    return *this;
}

bool CommandLineParser::parse(uint64_t argc, char* argv[]) {
    program_name_ = argv[0];
    
    for (uint64_t i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        // Handle --key=value
        if (arg.find("--") == 0) {
            size_t eq = arg.find('=');
            if (eq != std::string::npos) {
                std::string key = arg.substr(2, eq - 2);
                std::string value = arg.substr(eq + 1);
                values_[key] = value;
            } else {
                // Boolean flag
                std::string key = arg.substr(2);
                values_[key] = "true";
            }
        }
    }
    
    // Check required options
    for (const auto& opt : options_) {
        if (opt.required && values_.find(opt.long_name) == values_.end()) {
            std::cerr << "Error: Required option --" << opt.long_name << " not provided\n";
            return false;
        }
    }
    
    return true;
}

std::optional<std::string> CommandLineParser::get(const std::string& name) const {
    auto it = values_.find(name);
    if (it != values_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool CommandLineParser::has(const std::string& name) const {
    return values_.find(name) != values_.end();
}

void CommandLineParser::printHelp(std::ostream& out) const {
    out << "Usage: " << program_name_ << " <directory> [options]\n\n";
    
    // Group by category
    std::map<std::string, std::vector<Option>> grouped;
    for (const auto& opt : options_) {
        grouped[opt.category].push_back(opt);
    }
    
    for (const auto& [category, opts] : grouped) {
        out << category << ":\n";
        for (const auto& opt : opts) {
            out << "  --" << opt.long_name;
            if (opt.has_value) out << "=<value>";
            if (opt.required) out << " (required)";
            out << "\n      " << opt.description << "\n";
        }
        out << "\n";
    }
}
