#include <ed/core/ed_config.h>
#include <ed/core/ed_types.h>  // Single source of truth for DiagonalizationMethod (P0.14).
#include <algorithm>
#include <cctype>
#include <sstream>

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
    std::string current_section = "";  // Track current section
    
    // Helper to check if a bool value is true
    auto parse_bool = [](const std::string& value) -> bool {
        return (value == "true" || value == "1" || value == "yes" || value == "True" || value == "TRUE");
    };
    
    while (std::getline(file, line)) {
        line_num++;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Check for section header [SectionName]
        if (line[0] == '[') {
            size_t end_bracket = line.find(']');
            if (end_bracket != std::string::npos) {
                current_section = line.substr(1, end_bracket - 1);
                // Convert to lowercase for easier matching
                std::transform(current_section.begin(), current_section.end(), current_section.begin(), ::tolower);
            }
            continue;
        }
        
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
        
        // Parse based on section and key
        try {
            // ========== [System] section ==========
            if (current_section == "system") {
                if (key == "hamiltonian_dir") config.system.hamiltonian_dir = value;
                else if (key == "num_sites") config.system.num_sites = std::stoi(value);
                else if (key == "spin_length") config.system.spin_length = std::stof(value);
                else if (key == "use_fixed_sz") config.system.use_fixed_sz = parse_bool(value);
                else if (key == "full_sz_split") config.system.full_sz_split = parse_bool(value);
                else if (key == "n_up") config.system.n_up = std::stoi(value);
                else if (key == "sublattice_size") config.system.sublattice_size = std::stoi(value);
                else if (key == "interaction_file") config.system.interaction_file = value;
                else if (key == "single_site_file") config.system.single_site_file = value;
                else if (key == "three_body_file") config.system.three_body_file = value;
            }
            // ========== [Diagonalization] section ==========
            else if (current_section == "diagonalization") {
                if (key == "method") {
                    auto m = ed_config::parseMethod(value);
                    if (m) config.method = *m;
                }
                else if (key == "num_eigenvalues") config.diag.num_eigenvalues = std::stoi(value);
                else if (key == "max_iterations") config.diag.max_iterations = std::stoi(value);
                else if (key == "tolerance") config.diag.tolerance = std::stod(value);
                else if (key == "compute_eigenvectors") config.diag.compute_eigenvectors = parse_bool(value);
                else if (key == "block_size") config.diag.block_size = std::stoi(value);
            }
            // ========== [Output] section ==========
            else if (current_section == "output") {
                if (key == "output_dir") config.workflow.output_dir = value;
                else if (key == "output_prefix") {
                    config.dynamical.output_prefix = value;
                    config.static_resp.output_prefix = value;
                }
            }
            // ========== [Workflow] section ==========
            else if (current_section == "workflow") {
                if (key == "run_standard") config.workflow.run_standard = parse_bool(value);
                else if (key == "run_symmetrized") config.workflow.run_symm_auto = parse_bool(value);  // Legacy alias
                else if (key == "run_streaming_symmetry") config.workflow.run_symm_auto = parse_bool(value);  // Legacy alias
                else if (key == "run_symm") config.workflow.run_symm_auto = parse_bool(value);
                else if (key == "compute_thermo") config.workflow.compute_thermo = parse_bool(value);
                else if (key == "compute_dynamical_response") config.workflow.compute_dynamical_response = parse_bool(value);
                else if (key == "compute_static_response") config.workflow.compute_static_response = parse_bool(value);
                else if (key == "compute_ground_state_dssf") config.workflow.compute_ground_state_dssf = parse_bool(value);
                else if (key == "skip_ed") config.workflow.skip_ed = parse_bool(value);
                else if (key == "translation_only") config.workflow.translation_only = parse_bool(value);
            }
            // ========== [Thermodynamics] section ==========
            else if (current_section == "thermodynamics") {
                if (key == "compute_thermo") config.workflow.compute_thermo = parse_bool(value);
                else if (key == "temp_min") config.thermal.temp_min = std::stod(value);
                else if (key == "temp_max") config.thermal.temp_max = std::stod(value);
                else if (key == "num_temp_bins") config.thermal.num_temp_bins = std::stoi(value);
            }
            // ========== [FTLM] section ==========
            else if (current_section == "ftlm") {
                if (key == "num_samples") config.thermal.num_samples = std::stoi(value);
                else if (key == "krylov_dim") config.thermal.ftlm_krylov_dim = std::stoi(value);
                else if (key == "full_reorth") config.thermal.ftlm_full_reorth = parse_bool(value);
                else if (key == "reorth_freq") config.thermal.ftlm_reorth_freq = std::stoi(value);
                else if (key == "random_seed") config.thermal.ftlm_seed = std::stoul(value);
                else if (key == "store_samples") config.thermal.ftlm_store_samples = parse_bool(value);
                else if (key == "error_bars") config.thermal.ftlm_error_bars = parse_bool(value);
            }
            // ========== [LTLM] section ==========
            else if (current_section == "ltlm") {
                if (key == "krylov_dim") config.thermal.ltlm_krylov_dim = std::stoi(value);
                else if (key == "ground_krylov") config.thermal.ltlm_ground_krylov = std::stoi(value);
                else if (key == "full_reorth") config.thermal.ltlm_full_reorth = parse_bool(value);
                else if (key == "reorth_freq") config.thermal.ltlm_reorth_freq = std::stoi(value);
                else if (key == "random_seed") config.thermal.ltlm_seed = std::stoul(value);
                else if (key == "store_data") config.thermal.ltlm_store_data = parse_bool(value);
            }
            // ========== [TPQ] section ==========
            else if (current_section == "tpq") {
                if (key == "num_samples") config.thermal.num_samples = std::stoi(value);
                // New names (preferred)
                else if (key == "tpq_max_steps" || key == "max_steps") config.thermal.tpq_max_steps = std::stoi(value);
                else if (key == "taylor_order") config.thermal.tpq_taylor_order = std::stoi(value);
                else if (key == "tpq_measurement_interval" || key == "measurement_interval") config.thermal.tpq_measurement_interval = std::stoi(value);
                else if (key == "delta_beta") config.thermal.tpq_delta_beta = std::stod(value);
                else if (key == "tpq_energy_shift" || key == "energy_shift") config.thermal.tpq_energy_shift = std::stod(value);
                // Legacy names (for backwards compatibility)
                else if (key == "num_order") config.thermal.tpq_taylor_order = std::stoi(value);
                else if (key == "measure_freq") config.thermal.tpq_measurement_interval = std::stoi(value);
                else if (key == "delta_tau") config.thermal.tpq_delta_beta = std::stod(value);
                else if (key == "large_value") config.thermal.tpq_energy_shift = std::stod(value);
                // Continue quenching options
                else if (key == "continue_quenching" || key == "tpq_continue") config.thermal.tpq_continue = parse_bool(value);
                else if (key == "continue_sample" || key == "tpq_continue_sample") config.thermal.tpq_continue_sample = std::stoi(value);
                else if (key == "continue_beta" || key == "tpq_continue_beta") config.thermal.tpq_continue_beta = std::stod(value);
                else if (key == "target_beta" || key == "tpq_target_beta") config.thermal.tpq_target_beta = std::stod(value);
                else if (key == "tpq_num_measure_points" || key == "num_measure_points") config.thermal.tpq_num_measure_points = std::stoi(value);
                else if (key == "tpq_measure_beta_min" || key == "measure_beta_min") config.thermal.tpq_measure_beta_min = std::stod(value);
                else if (key == "tpq_measure_beta_max" || key == "measure_beta_max") config.thermal.tpq_measure_beta_max = std::stod(value);
            }
            // ========== [DynamicalResponse] section ==========
            else if (current_section == "dynamicalresponse") {
                if (key == "compute") config.workflow.compute_dynamical_response = parse_bool(value);
                else if (key == "thermal_average") config.dynamical.thermal_average = parse_bool(value);
                else if (key == "use_gpu") config.dynamical.use_gpu = parse_bool(value);
                else if (key == "num_samples") config.dynamical.num_random_states = std::stoi(value);
                else if (key == "krylov_dim") config.dynamical.krylov_dim = std::stoi(value);
                else if (key == "omega_min") config.dynamical.omega_min = std::stod(value);
                else if (key == "omega_max") config.dynamical.omega_max = std::stod(value);
                else if (key == "num_omega_points") config.dynamical.num_omega_points = std::stoi(value);
                else if (key == "broadening") config.dynamical.broadening = std::stod(value);
                else if (key == "temp_min") config.dynamical.temp_min = std::stod(value);
                else if (key == "temp_max") config.dynamical.temp_max = std::stod(value);
                else if (key == "num_temp_bins") config.dynamical.num_temp_bins = std::stoi(value);
                else if (key == "random_seed") config.dynamical.random_seed = std::stoul(value);
                else if (key == "output_prefix") config.dynamical.output_prefix = value;
            }
            // ========== [GroundStateDSSF] section ==========
            else if (current_section == "groundstatedssf") {
                if (key == "compute") config.workflow.compute_ground_state_dssf = parse_bool(value);
                else if (key == "krylov_dim") config.dynamical.krylov_dim = std::stoi(value);
                else if (key == "omega_min") config.dynamical.omega_min = std::stod(value);
                else if (key == "omega_max") config.dynamical.omega_max = std::stod(value);
                else if (key == "num_omega_points") config.dynamical.num_omega_points = std::stoi(value);
                else if (key == "broadening") config.dynamical.broadening = std::stod(value);
                else if (key == "tolerance") config.diag.tolerance = std::stod(value);
            }
            // ========== [StaticResponse] section ==========
            else if (current_section == "staticresponse") {
                if (key == "compute") config.workflow.compute_static_response = parse_bool(value);
                else if (key == "use_gpu") config.static_resp.use_gpu = parse_bool(value);
                else if (key == "num_samples") config.static_resp.num_random_states = std::stoi(value);
                else if (key == "krylov_dim") config.static_resp.krylov_dim = std::stoi(value);
                else if (key == "temp_min") config.static_resp.temp_min = std::stod(value);
                else if (key == "temp_max") config.static_resp.temp_max = std::stod(value);
                else if (key == "num_temp_points") config.static_resp.num_temp_points = std::stoi(value);
                else if (key == "compute_susceptibility") config.static_resp.compute_susceptibility = parse_bool(value);
                else if (key == "single_operator_mode") config.static_resp.single_operator_mode = parse_bool(value);
                else if (key == "random_seed") config.static_resp.random_seed = std::stoul(value);
                else if (key == "output_prefix") config.static_resp.output_prefix = value;
            }
            // ========== [Operators] section (shared by dynamical and static) ==========
            else if (current_section == "operators") {
                if (key == "operator_type") {
                    config.dynamical.operator_type = value;
                    config.static_resp.operator_type = value;
                }
                else if (key == "basis") {
                    config.dynamical.basis = value;
                    config.static_resp.basis = value;
                }
                else if (key == "spin_combinations") {
                    config.dynamical.spin_combinations = value;
                    config.static_resp.spin_combinations = value;
                }
                else if (key == "momentum_points") {
                    config.dynamical.momentum_points = value;
                    config.static_resp.momentum_points = value;
                }
                else if (key == "polarization") {
                    config.dynamical.polarization = value;
                    config.static_resp.polarization = value;
                }
                else if (key == "theta") {
                    config.dynamical.theta = std::stod(value);
                    config.static_resp.theta = std::stod(value);
                }
                else if (key == "unit_cell_size") {
                    config.dynamical.unit_cell_size = std::stoi(value);
                    config.static_resp.unit_cell_size = std::stoi(value);
                }
            }
            // ========== Legacy flat format (no section) ==========
            else {
                // Support for legacy flat config files without sections
                if (key == "method") {
                    auto m = ed_config::parseMethod(value);
                    if (m) config.method = *m;
                }
                else if (key == "num_eigenvalues") config.diag.num_eigenvalues = std::stoi(value);
                else if (key == "max_iterations") config.diag.max_iterations = std::stoi(value);
                else if (key == "tolerance") config.diag.tolerance = std::stod(value);
                else if (key == "compute_eigenvectors") config.diag.compute_eigenvectors = parse_bool(value);
                else if (key == "block_size") config.diag.block_size = std::stoi(value);
                else if (key == "num_sites") config.system.num_sites = std::stoi(value);
                else if (key == "spin_length") config.system.spin_length = std::stof(value);
                else if (key == "hamiltonian_dir") config.system.hamiltonian_dir = value;
                else if (key == "use_fixed_sz") config.system.use_fixed_sz = parse_bool(value);
                else if (key == "full_sz_split") config.system.full_sz_split = parse_bool(value);
                else if (key == "n_up") config.system.n_up = std::stoi(value);
                else if (key == "output_dir") config.workflow.output_dir = value;
                else if (key == "num_samples") config.thermal.num_samples = std::stoi(value);
                else if (key == "temp_min") config.thermal.temp_min = std::stod(value);
                else if (key == "temp_max") config.thermal.temp_max = std::stod(value);
                else if (key == "temp_bins") config.thermal.num_temp_bins = std::stoi(value);
                else if (key == "run_standard") config.workflow.run_standard = parse_bool(value);
                else if (key == "run_symmetrized") config.workflow.run_symm_auto = parse_bool(value);  // Legacy alias
                else if (key == "run_streaming_symmetry") config.workflow.run_symm_auto = parse_bool(value);  // Legacy alias
                else if (key == "run_symm") config.workflow.run_symm_auto = parse_bool(value);
                else if (key == "compute_thermo") config.workflow.compute_thermo = parse_bool(value);
                else if (key == "compute_dynamical_response") config.workflow.compute_dynamical_response = parse_bool(value);
                else if (key == "compute_static_response") config.workflow.compute_static_response = parse_bool(value);
                else if (key == "translation_only") config.workflow.translation_only = parse_bool(value);
                // Note: many legacy keys omitted for brevity - the section-based format is preferred
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing line " << line_num << " (section: " << current_section << "): " << e.what() << std::endl;
        }
    }
    
    return config;
}

EDConfig EDConfig::fromCommandLine(uint64_t argc, char* argv[]) {
    EDConfig config;
    
    if (argc < 2) {
        return config;  // Return default config
    }
    
    // Check if first argument is a config file (ends with .cfg, .ini, .txt, or .conf)
    std::string first_arg = argv[1];
    bool first_is_config = false;
    
    // Check for explicit --config= anywhere in args first
    for (uint64_t i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--config=") == 0) {
            std::string config_file = arg.substr(9);
            config = EDConfig::fromFile(config_file);
            first_is_config = true;  // Skip first arg processing below
            break;
        }
    }
    
    // If first argument looks like a config file, load it
    if (!first_is_config && (first_arg.find(".cfg") != std::string::npos || 
                              first_arg.find(".ini") != std::string::npos ||
                              first_arg.find(".conf") != std::string::npos ||
                              (first_arg.find(".txt") != std::string::npos && first_arg.find("config") != std::string::npos))) {
        config = EDConfig::fromFile(first_arg);
        first_is_config = true;
    }
    
    // If first argument is a directory (no config file found)
    if (!first_is_config && first_arg[0] != '-') {
        config.system.hamiltonian_dir = first_arg;
        config.workflow.output_dir = first_arg + "/output";
    }
    
    // Parse remaining arguments (command line overrides config file)
    // Start from index 2 if first arg was a config file or directory, otherwise 1
    uint64_t start_idx = (first_arg[0] != '-') ? 2 : 1;
    for (uint64_t i = start_idx; i < argc; i++) {
        std::string arg = argv[i];
        
        // Skip --config= since we already processed it
        if (arg.find("--config=") == 0) continue;
        
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
            else if (arg.find("--block-size=") == 0) config.diag.block_size = std::stoi(parse_value("--block-size="));
            else if (arg.find("--num_sites=") == 0) config.system.num_sites = std::stoi(parse_value("--num_sites="));
            else if (arg.find("--spin_length=") == 0) config.system.spin_length = std::stof(parse_value("--spin_length="));
            else if (arg == "--fixed-sz") config.system.use_fixed_sz = true;
            // Phase 7: orthogonal device / parallelism flags. Combine with
            // any base method (e.g. `--method=lanczos --gpu` is the canonical
            // replacement for the deprecated `--method=lanczos_gpu`).
            else if (arg == "--gpu") config.system.use_gpu = true;
            else if (arg == "--mpi") config.system.use_mpi = true;
            else if (arg == "--full-sz-split") config.system.full_sz_split = true;
            else if (arg.find("--n-up=") == 0) config.system.n_up = std::stoi(parse_value("--n-up="));
            else if (arg.find("--output=") == 0) config.workflow.output_dir = parse_value("--output=");
            else if (arg.find("--samples=") == 0) config.thermal.num_samples = std::stoi(parse_value("--samples="));
            else if (arg.find("--temp_min=") == 0) config.thermal.temp_min = std::stod(parse_value("--temp_min="));
            else if (arg.find("--temp_max=") == 0) config.thermal.temp_max = std::stod(parse_value("--temp_max="));
            else if (arg.find("--temp_bins=") == 0) config.thermal.num_temp_bins = std::stoi(parse_value("--temp_bins="));
            // New TPQ parameter names (preferred)
            else if (arg.find("--taylor_order=") == 0) config.thermal.tpq_taylor_order = std::stoi(parse_value("--taylor_order="));
            else if (arg.find("--measurement_interval=") == 0) config.thermal.tpq_measurement_interval = std::stoi(parse_value("--measurement_interval="));
            else if (arg.find("--delta_beta=") == 0) config.thermal.tpq_delta_beta = std::stod(parse_value("--delta_beta="));
            else if (arg.find("--energy_shift=") == 0) config.thermal.tpq_energy_shift = std::stod(parse_value("--energy_shift="));
            // Legacy TPQ parameter names (for backwards compatibility)
            else if (arg.find("--num_order=") == 0) config.thermal.tpq_taylor_order = std::stoi(parse_value("--num_order="));
            else if (arg.find("--measure-freq=") == 0) config.thermal.tpq_measurement_interval = std::stoi(parse_value("--measure-freq="));
            else if (arg.find("--num_measure_freq=") == 0) config.thermal.tpq_measurement_interval = std::stoi(parse_value("--num_measure_freq=")); // Deprecated: use --measurement_interval
            else if (arg.find("--delta_tau=") == 0) config.thermal.tpq_delta_beta = std::stod(parse_value("--delta_tau="));
            else if (arg.find("--large_value=") == 0) config.thermal.tpq_energy_shift = std::stod(parse_value("--large_value="));
            // TPQ continue-quenching options (new and legacy names)
            else if (arg == "--continue_quenching" || arg == "--tpq_continue") config.thermal.tpq_continue = true;
            else if (arg.find("--continue_sample=") == 0) config.thermal.tpq_continue_sample = std::stoi(parse_value("--continue_sample="));
            else if (arg.find("--tpq_continue_sample=") == 0) config.thermal.tpq_continue_sample = std::stoi(parse_value("--tpq_continue_sample="));
            else if (arg.find("--continue_beta=") == 0) config.thermal.tpq_continue_beta = std::stod(parse_value("--continue_beta="));
            else if (arg.find("--tpq_continue_beta=") == 0) config.thermal.tpq_continue_beta = std::stod(parse_value("--tpq_continue_beta="));
            else if (arg.find("--target_beta=") == 0) config.thermal.tpq_target_beta = std::stod(parse_value("--target_beta="));
            else if (arg.find("--tpq_target_beta=") == 0) config.thermal.tpq_target_beta = std::stod(parse_value("--tpq_target_beta="));
            else if (arg.find("--tpq_num_measure_points=") == 0) config.thermal.tpq_num_measure_points = std::stoi(parse_value("--tpq_num_measure_points="));
            else if (arg.find("--tpq_measure_beta_min=") == 0) config.thermal.tpq_measure_beta_min = std::stod(parse_value("--tpq_measure_beta_min="));
            else if (arg.find("--tpq_measure_beta_max=") == 0) config.thermal.tpq_measure_beta_max = std::stod(parse_value("--tpq_measure_beta_max="));
            // TPQ observable options (new names + deprecated aliases)
            else if (arg == "--save-thermal-states" || arg == "--calc_observables") config.observable.save_thermal_states = true;
            else if (arg == "--compute-spin-correlations" || arg == "--measure_spin") config.observable.compute_spin_correlations = true;
            else if (arg == "--standard") config.workflow.run_standard = true;
            // Phase 7.1: --symm and its deprecated aliases set BOTH the legacy
            // workflow flag (which dispatches in ed_main.cpp) and the canonical
            // SystemConfig::use_symmetry flag (the 5th orthogonal axis seen by
            // exact_diagonalization_from_directory through toEDParameters).
            else if (arg == "--symmetrized") {  // deprecated alias
                config.workflow.run_symm_auto = true;
                config.system.use_symmetry = true;
            }
            else if (arg == "--streaming-symmetry") {  // deprecated alias
                config.workflow.run_symm_auto = true;
                config.system.use_symmetry = true;
            }
            else if (arg == "--symm") {
                config.workflow.run_symm_auto = true;
                config.system.use_symmetry = true;
            }
            else if (arg == "--no-symm") {
                config.workflow.run_symm_auto = false;
                config.system.use_symmetry = false;
            }
            // --disk-streaming / --chunked-symm / --disk-threshold= / --chunked-threshold=
            // were retired in matvec-unification Phase 7.2. They were
            // single-node CPU-only ultra-low-memory specialisations. Use
            // the distributed/MPI build for those scales.
            else if (arg == "--disk-streaming" || arg == "--chunked-symm"
                  || arg.find("--disk-threshold=") == 0
                  || arg.find("--chunked-threshold=") == 0) {
                std::cerr << "[ED] WARNING: '" << arg << "' has been removed "
                          << "(matvec-unification Phase 7.2). Use the "
                          << "distributed/MPI build for very large Hilbert "
                          << "spaces. Ignoring.\n";
            }
            else if (arg == "--thermo") config.workflow.compute_thermo = true;
            else if (arg == "--dynamical-response") config.workflow.compute_dynamical_response = true;
            else if (arg == "--static-response") config.workflow.compute_static_response = true;
            else if (arg == "--ground-state-dssf") config.workflow.compute_ground_state_dssf = true;
            else if (arg == "--precompute-basis") config.workflow.precompute_basis_only = true;
            else if (arg == "--translation-only") config.workflow.translation_only = true;
            else if (arg.find("--basis-cache-dir=") == 0) config.workflow.basis_cache_dir = parse_value("--basis-cache-dir=");
            else if (arg.find("--sectors=") == 0) {
                std::string sectors_str = parse_value("--sectors=");
                std::istringstream ss(sectors_str);
                std::string token;
                config.workflow.selected_sectors.clear();
                while (std::getline(ss, token, ',')) {
                    config.workflow.selected_sectors.push_back(std::stoi(token));
                }
            }
            else if (arg == "--skip_ED") config.workflow.skip_ed = true;
            else if (arg.find("--sublattice_size=") == 0) config.system.sublattice_size = std::stoi(parse_value("--sublattice_size="));
            else if (arg.find("--omega_min=") == 0) config.observable.omega_min = std::stod(parse_value("--omega_min="));
            else if (arg.find("--omega_max=") == 0) config.observable.omega_max = std::stod(parse_value("--omega_max="));
            else if (arg.find("--num_points=") == 0) config.observable.num_points = std::stoi(parse_value("--num_points="));
            else if (arg.find("--t_end=") == 0) config.observable.t_end = std::stod(parse_value("--t_end="));
            else if (arg.find("--dt=") == 0) config.observable.dt = std::stod(parse_value("--dt="));
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
    bool only_response = (config.workflow.compute_dynamical_response || 
                          config.workflow.compute_static_response ||
                          config.workflow.compute_ground_state_dssf) &&
                        !config.workflow.run_standard && 
                        !config.workflow.run_symm_auto &&
                        !config.workflow.compute_thermo;
    
    if (only_response && !config.workflow.skip_ed) {
        std::cout << "Note: Only response calculations requested. Skipping diagonalization (use --standard/--symm to override).\n";
        config.workflow.skip_ed = true;
    }
    
    // Default to standard workflow if nothing specified (and skip_ed not set)
    if (!config.workflow.run_standard && !config.workflow.run_symm_auto && !config.workflow.skip_ed) {
        config.workflow.run_standard = true;
    }
    
    return config;
}

EDConfig& EDConfig::merge(const EDConfig& other) {
    // Simple merge: other overrides this where values differ from defaults
    // This is a simplified version - could be more sophisticated
    // Only override method if it was explicitly changed from default (LANCZOS)
    if (other.method != DiagonalizationMethod::LANCZOS) {
        method = other.method;
    }
    
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
    if (other.workflow.run_symm_auto) workflow.run_symm_auto = true;
    if (other.workflow.compute_thermo) workflow.compute_thermo = true;
    if (!other.workflow.output_dir.empty()) workflow.output_dir = other.workflow.output_dir;
    
    return *this;
}

bool EDConfig::validate(std::ostream& err) const {
    bool valid = true;
    
    // ========== System validation ==========
    if (system.num_sites == 0) {
        err << "Error: num_sites must be specified or auto-detected\n";
        valid = false;
    } else if (system.num_sites >= 64) {
        err << "Error: num_sites must be < 64 (bit-shift 1ULL << num_sites would overflow)\n";
        valid = false;
    }
    
    if (system.hamiltonian_dir.empty()) {
        err << "Error: hamiltonian_dir must be specified\n";
        valid = false;
    }
    
    if (system.spin_length <= 0) {
        err << "Error: spin_length must be positive\n";
        valid = false;
    }
    
    // Validate n_up for fixed-Sz mode
    if (system.use_fixed_sz && system.n_up >= 0) {
        if (static_cast<uint64_t>(system.n_up) > system.num_sites) {
            err << "Error: n_up (" << system.n_up << ") cannot exceed num_sites (" << system.num_sites << ")\n";
            valid = false;
        }
    }
    
    // ========== Diagonalization validation ==========
    if (diag.num_eigenvalues == 0) {
        err << "Error: num_eigenvalues must be >= 1\n";
        valid = false;
    }
    
    if (diag.tolerance <= 0) {
        err << "Error: tolerance must be positive\n";
        valid = false;
    }
    
    if (diag.max_iterations < 1) {
        err << "Error: max_iterations must be >= 1\n";
        valid = false;
    }
    
    // ========== Thermal validation ==========
    if (thermal.temp_min <= 0) {
        err << "Error: temp_min must be positive\n";
        valid = false;
    }
    
    if (thermal.temp_max < thermal.temp_min) {
        err << "Error: temp_max must be >= temp_min\n";
        valid = false;
    }
    
    if (thermal.num_temp_bins < 1) {
        err << "Error: num_temp_bins must be >= 1\n";
        valid = false;
    }
    
    if (thermal.num_samples < 1) {
        err << "Error: num_samples must be >= 1\n";
        valid = false;
    }
    
    // TPQ-specific validation
    if (thermal.tpq_taylor_order < 1) {
        err << "Error: tpq_taylor_order must be >= 1\n";
        valid = false;
    }
    
    if (thermal.tpq_delta_beta <= 0) {
        err << "Error: tpq_delta_beta must be positive\n";
        valid = false;
    }
    
    // ========== Dynamical response validation ==========
    if (workflow.compute_dynamical_response) {
        if (dynamical.num_omega_points < 1) {
            err << "Error: num_omega_points must be >= 1\n";
            valid = false;
        }
        
        if (dynamical.broadening < 0) {
            err << "Error: broadening must be non-negative\n";
            valid = false;
        }
        
        if (dynamical.krylov_dim < 1) {
            err << "Error: dynamical krylov_dim must be >= 1\n";
            valid = false;
        }
    }
    
    // ========== Static response validation ==========
    if (workflow.compute_static_response) {
        if (static_resp.krylov_dim < 1) {
            err << "Error: static_resp krylov_dim must be >= 1\n";
            valid = false;
        }
        
        if (static_resp.num_temp_points < 1) {
            err << "Error: static_resp num_temp_points must be >= 1\n";
            valid = false;
        }
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
    file << "run_symm = " << (workflow.run_symm_auto ? "true" : "false") << "\n";
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
        
        // Calculate dimension reduction using overflow-safe binomial coefficient
        // Uses uint64_t throughout and divides early to prevent overflow
        auto binomial_safe = [](uint64_t n, uint64_t k) -> uint64_t {
            if (k > n) return 0;
            if (k == 0 || k == n) return 1;
            // Use symmetry: C(n,k) = C(n, n-k)
            if (k > n - k) k = n - k;
            
            uint64_t result = 1;
            for (uint64_t i = 0; i < k; ++i) {
                // Multiply first, then divide to maintain integer arithmetic
                // The division is always exact due to properties of binomial coefficients
                result = result * (n - i) / (i + 1);
            }
            return result;
        };
        uint64_t full_dim = 1ULL << system.num_sites;
        uint64_t fixed_dim = binomial_safe(system.num_sites, static_cast<uint64_t>(n_up_actual));
        out << "Hilbert space: " << fixed_dim << " (reduced from " << full_dim 
            << ", factor: " << (double)full_dim / fixed_dim << "x)\n";
    }
    
    out << "Eigenvalues: " << diag.num_eigenvalues << " (tol=" << diag.tolerance << ")\n";
    out << "Output: " << workflow.output_dir << "\n";
    
    if (workflow.run_standard) out << "  - Running standard diagonalization\n";
    if (workflow.run_symm_auto) out << "  - Running symmetry-exploiting diagonalization (auto-select mode)\n";
    if (workflow.compute_thermo) out << "  - Computing thermodynamics\n";
    if (observable.save_thermal_states) out << "  - Saving TPQ states at target temperatures\n";
    if (observable.compute_spin_correlations) out << "  - Computing spin correlations ⟨Si⟩ and ⟨Si·Sj⟩\n";
    
    out << "========================================\n";
}

bool EDConfig::autoDetectNumSites() {
    std::string positions_file = system.hamiltonian_dir + "/positions.dat";
    std::ifstream file(positions_file);

    if (!file.is_open()) {
        return false;
    }

    // The canonical positions.dat format (written by write_positions_file) is
    //   x y z
    // one line per site, with comment lines beginning with '#'. Count the
    // number of data lines; that equals num_sites.
    uint64_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double x;
        if (iss >> x) {
            ++count;
        }
    }

    if (count > 0) {
        system.num_sites = count;
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

    if (lower == "lanczos")        return DiagonalizationMethod::LANCZOS;
    if (lower == "block_lanczos")  return DiagonalizationMethod::BLOCK_LANCZOS;
    if (lower == "krylov_schur")   return DiagonalizationMethod::KRYLOV_SCHUR;
    if (lower == "full")           return DiagonalizationMethod::FULL;

    if (lower == "mtpq")           return DiagonalizationMethod::mTPQ;
    if (lower == "ctpq")           return DiagonalizationMethod::cTPQ;
    if (lower == "ftlm")           return DiagonalizationMethod::FTLM;
    if (lower == "ltlm")           return DiagonalizationMethod::LTLM;
    if (lower == "kpm_dos" || lower == "kpmdos" || lower == "kpm")
        return DiagonalizationMethod::KPM_DOS;

    std::cerr << "Warning: Unknown method '" << str
              << "' (retired or never supported); falling back to LANCZOS.\n";
    return std::nullopt;
}

std::string methodToString(DiagonalizationMethod method) {
    switch (method) {
        case DiagonalizationMethod::LANCZOS:       return "LANCZOS";
        case DiagonalizationMethod::BLOCK_LANCZOS: return "BLOCK_LANCZOS";
        case DiagonalizationMethod::KRYLOV_SCHUR:  return "KRYLOV_SCHUR";
        case DiagonalizationMethod::FULL:          return "FULL";
        case DiagonalizationMethod::mTPQ:          return "mTPQ";
        case DiagonalizationMethod::cTPQ:          return "cTPQ";
        case DiagonalizationMethod::FTLM:          return "FTLM";
        case DiagonalizationMethod::LTLM:          return "LTLM";
        case DiagonalizationMethod::KPM_DOS:       return "KPM_DOS";
    }
    return "UNKNOWN";
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

        default:
            break;
    }

    return config;
}

/**
 * @brief Compact parameter-info helper used by `--info <method>`.
 * The exhaustive per-method text was retired in the minimalist
 * refactor; for the kept solvers we point users at the relevant
 * docs / header instead.
 */
std::string getMethodParameterInfo(DiagonalizationMethod method) {
    std::ostringstream info;
    info << "\n========================================\n";
    info << "Method: " << methodToString(method) << "\n";
    info << "========================================\n\n";
    switch (method) {
        case DiagonalizationMethod::LANCZOS:
        case DiagonalizationMethod::BLOCK_LANCZOS:
        case DiagonalizationMethod::KRYLOV_SCHUR:
        case DiagonalizationMethod::FULL:
            info << "Ground-state / low-spectrum solver. Common knobs:\n"
                 << "  --eigenvalues=<n>   number of eigenpairs\n"
                 << "  --iterations=<n>    max Krylov dimension\n"
                 << "  --tolerance=<tol>   convergence tolerance\n"
                 << "  --block-size=<b>    block size (BLOCK_LANCZOS only)\n";
            break;
        case DiagonalizationMethod::FTLM:
        case DiagonalizationMethod::LTLM:
        case DiagonalizationMethod::mTPQ:
        case DiagonalizationMethod::cTPQ:
        case DiagonalizationMethod::KPM_DOS:
            info << "Finite-temperature solver. Common knobs:\n"
                 << "  --samples=<n>       number of random samples\n"
                 << "  --temp_min/--temp_max/--temp_bins\n"
                 << "  method-specific knobs in docs/guides/usage.md\n";
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
