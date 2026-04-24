// =============================================================================
// src/cli/workflows.cpp
//
// Implementations of the CLI-driven workflow entry points declared in
// `include/ed/cli/workflows.h`. Extracted verbatim from `src/apps/ed_main.cpp`
// in P1.11 (DSSF PR-B / audit §3.10): pure structural lift-and-shift, no
// behaviour change. Audit dashboards and unit tests should be bit-identical
// across this commit.
//
// What lives here today:
//   * String parsing helpers (parse_spin_combinations / parse_momentum_points
//     / parse_polarization).
//   * `construct_operators_from_config` — thin wrapper over the canonical
//     `ed::dssf::build_observable_pairs` (P1.10), preserved so the four
//     historical call sites in this file (and the `run_dssf_mode` shim
//     still living in ed_main.cpp) keep compiling.
//   * The four `run_*_workflow` (standard / streaming-symmetry /
//     disk-streaming / chunked-symmetry) entry points.
//   * `compute_thermodynamics`.
//   * The three `compute_*_workflow` entry points (dynamical response,
//     static response, ground-state DSSF) — these are the principal
//     reason this TU exists; they account for ~1.5 kLOC of nearly
//     identical operator-construction + Lanczos/FTLM dispatch + HDF5 save
//     logic that future PRs (P2.2 dssf_engine, P2.3 unified /dssf/ HDF5
//     schema, P2.4 `ED dssf` subcommand) will collapse onto a single
//     `ed::dssf::run(...)` call.
//   * `print_eigenvalue_summary`.
//
// The include list mirrors what `ed_main.cpp` used to require for these
// functions (no more, no less). When the future DSSF refactor pulls the
// guts of the `compute_*_workflow` functions into `ed::dssf`, we can
// trim this list aggressively.
// =============================================================================

#include <ed/cli/workflows.h>

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <filesystem>
#include <random>
#include <algorithm>
#include <map>
#include <fstream>

#include <ed/core/ed_config.h>
#include <ed/core/ed_config_adapter.h>
#include <ed/core/ed_wrapper.h>
#include <ed/core/ed_wrapper_streaming.h>
#include <ed/core/disk_streaming_symmetry.h>
#include <ed/core/ed_wrapper_chunked.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/dssf/operator_spec.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_CUDA
#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_ed_wrapper.h>
#include <cuda_runtime.h>
#endif

/**
 * @brief Parse spin combinations from string format
 * Format: "op1,op2;op3,op4;..." where op is 0=Sp/Sx, 1=Sm/Sy, 2=Sz
 */
std::vector<std::pair<int, int>> parse_spin_combinations(const std::string& spin_combinations_str) {
    std::vector<std::pair<int, int>> spin_combinations;
    std::stringstream ss(spin_combinations_str);
    std::string pair_str;
    
    while (std::getline(ss, pair_str, ';')) {
        std::stringstream pair_ss(pair_str);
        std::string op1_str, op2_str;
        
        if (std::getline(pair_ss, op1_str, ',') && std::getline(pair_ss, op2_str)) {
            try {
                int op1 = std::stoi(op1_str);
                int op2 = std::stoi(op2_str);
                
                if (op1 >= 0 && op1 <= 2 && op2 >= 0 && op2 <= 2) {
                    spin_combinations.push_back({op1, op2});
                } else {
                    std::cerr << "Warning: Invalid spin operator " << op1 << "," << op2 
                              << ". Operators must be 0, 1, or 2." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse spin combination: " << pair_str << std::endl;
            }
        }
    }
    
    if (spin_combinations.empty()) {
        std::cerr << "Warning: No valid spin combinations provided. Using default SzSz." << std::endl;
        spin_combinations = {{2, 2}};
    }
    
    return spin_combinations;
}

/**
 * @brief Parse momentum points from string format
 * Format: "Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;..." (values are multiplied by π)
 */
std::vector<std::vector<double>> parse_momentum_points(const std::string& momentum_str) {
    std::vector<std::vector<double>> momentum_points;
    std::stringstream mom_ss(momentum_str);
    std::string point_str;
    
    while (std::getline(mom_ss, point_str, ';')) {
        std::stringstream point_ss(point_str);
        std::string coord_str;
        std::vector<double> point;
        
        while (std::getline(point_ss, coord_str, ',')) {
            try {
                double coord = std::stod(coord_str);
                coord *= M_PI;  // Scale to π
                point.push_back(coord);
            } catch (...) {
                std::cerr << "Warning: Failed to parse momentum coordinate: " << coord_str << std::endl;
            }
        }
        
        if (point.size() == 3) {
            momentum_points.push_back(point);
        } else {
            std::cerr << "Warning: Momentum point must have 3 coordinates, got " << point.size() << std::endl;
        }
    }
    
    // Use default momentum points if none provided or parsing failed
    if (momentum_points.empty()) {
        momentum_points = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 2.0 * M_PI}
        };
        std::cout << "Using default momentum points: (0,0,0) and (0,0,2π)" << std::endl;
    }
    
    return momentum_points;
}

/**
 * @brief Parse polarization vector from string format
 * Format: "px,py,pz" (will be normalized)
 */
std::vector<double> parse_polarization(const std::string& pol_str) {
    std::vector<double> polarization = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};  // default
    
    std::stringstream pol_ss(pol_str);
    std::string coord_str;
    std::vector<double> pol_temp;
    
    while (std::getline(pol_ss, coord_str, ',')) {
        try {
            double coord = std::stod(coord_str);
            pol_temp.push_back(coord);
        } catch (...) {
            std::cerr << "Warning: Failed to parse polarization coordinate: " << coord_str << std::endl;
        }
    }
    
    if (pol_temp.size() == 3) {
        // Normalize the polarization vector
        double norm = std::sqrt(pol_temp[0]*pol_temp[0] + pol_temp[1]*pol_temp[1] + pol_temp[2]*pol_temp[2]);
        if (norm > 1e-10) {
            polarization = {pol_temp[0]/norm, pol_temp[1]/norm, pol_temp[2]/norm};
            std::cout << "Using custom polarization: (" << polarization[0] << "," 
                      << polarization[1] << "," << polarization[2] << ")" << std::endl;
        } else {
            std::cerr << "Warning: Polarization vector has zero norm, using default" << std::endl;
        }
    } else {
        std::cerr << "Warning: Polarization must have 3 coordinates, got " << pol_temp.size() << std::endl;
    }
    
    return polarization;
}

/**
 * @brief Construct operators based on configuration.
 *
 * Thin wrapper around ed::dssf::build_observable_pairs (P1.10). The
 * implementation moved to src/dssf/operator_spec.cpp; this function exists
 * purely so the four legacy call sites in ed_main.cpp keep compiling.
 * New code should call ed::dssf::build_observable_pairs directly.
 */
void construct_operators_from_config(
    const std::string& operator_type,
    const std::string& basis,
    const std::vector<std::pair<int, int>>& spin_combinations,
    const std::vector<std::vector<double>>& momentum_points,
    const std::vector<double>& polarization,
    double theta,
    uint64_t unit_cell_size,
    uint64_t num_sites,
    float spin_length,
    bool use_fixed_sz,
    int64_t n_up,
    const std::string& positions_file,
    std::vector<Operator>& obs_1_out,
    std::vector<Operator>& obs_2_out,
    std::vector<std::string>& names_out
) {
    ed::dssf::OperatorSpec spec;
    spec.operator_type    = operator_type;
    spec.basis            = basis;
    spec.spin_combinations = spin_combinations;
    spec.momentum_points  = momentum_points;
    spec.polarization     = polarization;
    spec.theta            = theta;
    spec.unit_cell_size   = unit_cell_size;
    spec.num_sites        = num_sites;
    spec.spin_length      = spin_length;
    spec.use_fixed_sz     = use_fixed_sz;
    spec.n_up             = n_up;
    spec.positions_file   = positions_file;
    auto pairs = ed::dssf::build_observable_pairs(spec);
    obs_1_out = std::move(pairs.obs_1);
    obs_2_out = std::move(pairs.obs_2);
    names_out = std::move(pairs.names);
}


// ============================================================================
// WORKFLOW FUNCTIONS
// ============================================================================

/**
 * @brief Run standard diagonalization workflow
 */
EDResults run_standard_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    // Check if fixed Sz mode is enabled
    if (config.system.use_fixed_sz) {
        int64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
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
    
    // Print results summary
    if (!results.eigenvalues.empty()) {
        std::cout << "\n  Lowest eigenvalues:\n";
        size_t show = std::min(results.eigenvalues.size(), (size_t)5);
        for (size_t i = 0; i < show; i++) {
            std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) 
                      << results.eigenvalues[i] << "\n";
        }
        if (results.eigenvalues.size() > 5) {
            std::cout << "    ... (" << (results.eigenvalues.size() - 5) << " more)\n";
        }
    }
    
    std::cout << "\n  Time: " << std::fixed << std::setprecision(2) << duration / 1000.0 << " s\n";
    
    // Eigenvalues are saved to HDF5 by the underlying diagonalization functions
    
    return results;
}

/**
 * @brief Run symmetry-exploiting diagonalization workflow
 *
 * Uses streaming symmetry path which handles both CPU and GPU
 * uniformly.  The streaming approach never materialises explicit block
 * matrices — it keeps the per-sector orbit data in memory so the GPU
 * symmetrized matvec kernel can use it directly.  On the CPU side the
 * same matrix-free operator is wrapped in a lambda and forwarded to the
 * standard solver dispatch (Lanczos, Block Lanczos, Davidson, etc.).
 */
EDResults run_streaming_symmetry_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    if (config.system.use_fixed_sz) {
        int64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        results = exact_diagonalization_streaming_symmetry_fixed_sz(
            config.system.hamiltonian_dir,
            n_up,
            config.method,
            params,
            "InterAll.dat",
            "Trans.dat",
            config.workflow.basis_cache_dir,
            config.workflow.precompute_basis_only
        );
    } else {
        results = exact_diagonalization_streaming_symmetry(
            config.system.hamiltonian_dir,
            config.method,
            params,
            "InterAll.dat",
            "Trans.dat",
            config.workflow.basis_cache_dir,
            config.workflow.precompute_basis_only
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Print results summary
    if (!results.eigenvalues.empty()) {
        std::cout << "\n  Lowest eigenvalues:\n";
        size_t show = std::min(results.eigenvalues.size(), (size_t)5);
        for (size_t i = 0; i < show; i++) {
            std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) 
                      << results.eigenvalues[i] << "\n";
        }
        if (results.eigenvalues.size() > 5) {
            std::cout << "    ... (" << (results.eigenvalues.size() - 5) << " more)\n";
        }
    }
    
    std::cout << "\n  Time: " << std::fixed << std::setprecision(2) << duration / 1000.0 << " s\n";
    
    // Eigenvalues are saved to HDF5 by the underlying diagonalization functions
    
    return results;
}

/**
 * @brief Run disk-based streaming symmetry diagonalization workflow
 * 
 * This ultra-low-memory mode processes sectors one at a time,
 * storing sector data on disk. Suitable for very large Hilbert spaces
 * (>64M states) where standard streaming would OOM.
 * 
 * NOTE: GPU methods are NOT supported - this uses matrix-free operations
 * which require CPU Lanczos. GPU methods will be automatically converted.
 */
EDResults run_disk_streaming_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);
    
    // Warn about GPU method override
    DiagonalizationMethod method = config.method;
    if (method == DiagonalizationMethod::LANCZOS_GPU || 
        method == DiagonalizationMethod::BLOCK_LANCZOS_GPU ||
        method == DiagonalizationMethod::DAVIDSON_GPU ||
        method == DiagonalizationMethod::LOBPCG_GPU ||
        method == DiagonalizationMethod::mTPQ_GPU ||
        method == DiagonalizationMethod::cTPQ_GPU ||
        method == DiagonalizationMethod::FTLM_GPU) {
        std::cout << "\n  WARNING: Disk-streaming mode uses matrix-free operations.\n";
        std::cout << "           GPU methods are not supported - using CPU Lanczos instead.\n\n";
        method = DiagonalizationMethod::LANCZOS;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results = exact_diagonalization_disk_streaming(
        config.system.hamiltonian_dir,
        method,
        params
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Print results summary
    if (!results.eigenvalues.empty()) {
        std::cout << "\n  Lowest eigenvalues:\n";
        size_t show = std::min(results.eigenvalues.size(), (size_t)5);
        for (size_t i = 0; i < show; i++) {
            std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) 
                      << results.eigenvalues[i] << "\n";
        }
        if (results.eigenvalues.size() > 5) {
            std::cout << "    ... (" << (results.eigenvalues.size() - 5) << " more)\n";
        }
    }
    
    std::cout << "\n  Time: " << std::fixed << std::setprecision(2) << duration / 1000.0 << " s\n";
    
    return results;
}

/**
 * @brief Run ultra-low-memory chunked symmetry diagonalization workflow
 * 
 * This mode uses a two-pass algorithm to minimize memory during basis construction:
 * 1. Discover orbit representatives without caching (O(1) memory per state)
 * 2. Build sectors one at a time from the orbit representatives
 * 
 * Use this when standard streaming modes run out of memory during the
 * symmetry sector building phase.
 * 
 * NOTE: This trades speed for memory efficiency - it's slower than standard
 * streaming because it doesn't use orbit lookup caching.
 */
EDResults run_chunked_symmetry_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);
    
    // Warn about GPU method override
    DiagonalizationMethod method = config.method;
    if (method == DiagonalizationMethod::LANCZOS_GPU || 
        method == DiagonalizationMethod::BLOCK_LANCZOS_GPU ||
        method == DiagonalizationMethod::DAVIDSON_GPU ||
        method == DiagonalizationMethod::LOBPCG_GPU ||
        method == DiagonalizationMethod::mTPQ_GPU ||
        method == DiagonalizationMethod::cTPQ_GPU ||
        method == DiagonalizationMethod::FTLM_GPU) {
        std::cout << "\n  WARNING: Chunked-symmetry mode uses matrix-free operations.\n";
        std::cout << "           GPU methods are not supported - using CPU Lanczos instead.\n\n";
        method = DiagonalizationMethod::LANCZOS;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    if (config.system.use_fixed_sz) {
        int64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        results = exact_diagonalization_chunked_symmetry_fixed_sz(
            config.system.hamiltonian_dir,
            n_up,
            method,
            params
        );
    } else {
        results = exact_diagonalization_chunked_symmetry(
            config.system.hamiltonian_dir,
            method,
            params
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // Print results summary
    if (!results.eigenvalues.empty()) {
        std::cout << "\n  Lowest eigenvalues:\n";
        size_t show = std::min(results.eigenvalues.size(), (size_t)5);
        for (size_t i = 0; i < show; i++) {
            std::cout << "    E[" << i << "] = " << std::fixed << std::setprecision(10) 
                      << results.eigenvalues[i] << "\n";
        }
        if (results.eigenvalues.size() > 5) {
            std::cout << "    ... (" << (results.eigenvalues.size() - 5) << " more)\n";
        }
    }
    
    std::cout << "\n  Time: " << std::fixed << std::setprecision(2) << duration / 1000.0 << " s\n";
    
    return results;
}

/**
 * @brief Compute thermodynamics from eigenvalue spectrum
 */
void compute_thermodynamics(const std::vector<double>& eigenvalues, const EDConfig& config) {
    if (eigenvalues.empty()) return;
    
    auto thermo_data = calculate_thermodynamics_from_spectrum(
        eigenvalues,
        config.thermal.temp_min,
        config.thermal.temp_max,
        config.thermal.num_temp_bins
    );
    
    // Save results to HDF5
    try {
        std::string hdf5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "energy", thermo_data.energy);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "specific_heat", thermo_data.specific_heat);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "entropy", thermo_data.entropy);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "free_energy", thermo_data.free_energy);
        std::cout << "  Saved thermodynamic data to HDF5\n";
        
    } catch (const std::exception& e) {
        std::cerr << "  Error: Failed to save thermodynamics to HDF5: " << e.what() << std::endl;
    }
}

/**
 * @brief Compute dynamical response (spectral functions)
 */
void compute_dynamical_response_workflow(const EDConfig& config) {
    // Get MPI rank and size
    int rank = 0, size = 1;
    #ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif
    
    // Note: Currently only thermal mode is supported in the integrated pipeline
    if (!config.dynamical.thermal_average) {
        if (rank == 0) {
            std::cerr << "Note: Only thermal mode is supported. Setting thermal_average mode.\n";
        }
    }
    
    if (rank == 0) {
        std::cout << "\nDynamical Response Calculation\n";
        
#ifdef WITH_CUDA
        if (config.dynamical.use_gpu) {
            std::cout << "  GPU: enabled";
            if (config.system.use_fixed_sz) {
                std::cout << " (disabled for fixed-Sz)";
            }
            std::cout << "\n";
        }
#endif
    }
    
    // Check if using configuration-based or legacy file-based operator loading
    bool use_config_operators = config.dynamical.operator_file.empty() || 
                                config.dynamical.operator_type != "sum";
    
    // Prepare Hamiltonian
    Operator ham(config.system.num_sites, config.system.spin_length);
    std::string interaction_file = config.system.hamiltonian_dir + "/" + config.system.interaction_file;
    std::string single_site_file = config.system.hamiltonian_dir + "/" + config.system.single_site_file;
    ham.loadFromInterAllFile(interaction_file);
    ham.loadFromFile(single_site_file);
    
    // Load three-body terms if specified
    if (!config.system.three_body_file.empty()) {
        std::string three_body_file = config.system.hamiltonian_dir + "/" + config.system.three_body_file;
        if (std::filesystem::exists(three_body_file)) {
            if (rank == 0) std::cout << "Loading three-body terms from: " << three_body_file << "\n";
            ham.loadThreeBodyTerm(three_body_file);
        }
    }
    
    // Hilbert space dimension
    uint64_t N;
    if (config.system.use_fixed_sz) {
        // Use binomial coefficient C(num_sites, n_up) for fixed-Sz sector
        int64_t n_up_dim = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        N = 1;
        for (int64_t i = 0; i < n_up_dim; i++) {
            N = N * (config.system.num_sites - i) / (i + 1);
        }
        if (rank == 0) std::cout << "Fixed-Sz dynamical response: dim=" << N << " (n_up=" << n_up_dim << ")\n";
    } else {
        N = 1ULL << config.system.num_sites;
    }
    
    // Create function wrapper for Hamiltonian
    auto H_func = [&ham](const Complex* in, Complex* out, uint64_t dim) {
        ham.apply(in, out, dim);
    };
    
    // Setup parameters
    DynamicalResponseParameters params;
    params.num_samples = config.dynamical.num_random_states;
    params.krylov_dim = config.dynamical.krylov_dim;
    params.broadening = config.dynamical.broadening;
    params.random_seed = config.dynamical.random_seed;
    
    // Ensure output directory exists
    create_directory_mpi_safe(config.workflow.output_dir);
    
    if (rank == 0) {
        std::cout << "Random states: " << params.num_samples << "\n";
        std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
        std::cout << "Temperature range: [" << config.dynamical.temp_min << ", " << config.dynamical.temp_max << "]\n";
        std::cout << "Temperature bins: " << config.dynamical.num_temp_bins << "\n";
    }
    
    // Find ground state energy for proper energy shifting
    double ground_state_energy = 0.0;
    bool found_ground_state = false;
    
    if (rank == 0) {
        std::string h5_file = config.workflow.output_dir + "/ed_results.h5";
        
        // Method 1: Try to load eigenvalues from HDF5
        if (HDF5IO::fileExists(h5_file)) {
            try {
                auto eigenvalues = HDF5IO::loadEigenvalues(h5_file);
                if (!eigenvalues.empty()) {
                    ground_state_energy = eigenvalues[0];
                    found_ground_state = true;
                    std::cout << "  Loaded ground state energy from HDF5 eigenvalues\n";
                }
            } catch (const std::exception& e) {
                // Continue to next method
            }
            
            // Method 2: Try TPQ thermodynamics from HDF5
            if (!found_ground_state) {
                try {
                    auto points = HDF5IO::loadTPQThermodynamics(h5_file, 0);
                    if (!points.empty()) {
                        double min_energy = std::numeric_limits<double>::max();
                        for (size_t i = 1; i < points.size(); ++i) {  // Skip first entry
                            if (points[i].energy < min_energy) {
                                min_energy = points[i].energy;
                            }
                        }
                        if (min_energy < std::numeric_limits<double>::max()) {
                            ground_state_energy = min_energy;
                            found_ground_state = true;
                            std::cout << "  Loaded ground state energy from HDF5 TPQ data\n";
                        }
                    }
                } catch (const std::exception& e) {
                    // Continue to fallback
                }
            }
        }
        
        // Method 3 (fallback): Compute using Lanczos
        if (!found_ground_state) {
            std::cout << "  Computing ground state energy using Lanczos...\n";
            ComplexVector ground_state(N);
            ground_state_energy = find_ground_state_lanczos(
                H_func, N, params.krylov_dim, params.tolerance,
                params.full_reorthogonalization, params.reorth_frequency,
                ground_state
            );
            found_ground_state = true;
            
            // Save computed ground state energy to HDF5
            try {
                std::string h5_path = HDF5IO::createOrOpenFile(config.workflow.output_dir);
                HDF5IO::saveEigenvalues(h5_path, {ground_state_energy});
            } catch (...) {
                // Ignore save errors
            }
        }
        
        std::cout << "  Ground state energy: " << std::fixed << std::setprecision(10) 
                  << ground_state_energy << "\n";
    }
    
    #ifdef WITH_MPI
    // Broadcast ground state energy to all ranks
    MPI_Bcast(&ground_state_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&found_ground_state, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    #endif
    
    if (!found_ground_state) {
        if (rank == 0) {
            std::cerr << "Error: Failed to obtain ground state energy\n";
        }
        return;
    }
    
    // Generate temperature grid
    std::vector<double> temperatures(config.dynamical.num_temp_bins);
    if (config.dynamical.num_temp_bins == 1) {
        temperatures[0] = config.dynamical.temp_min;
    } else {
        double log_tmin = std::log(config.dynamical.temp_min);
        double log_tmax = std::log(config.dynamical.temp_max);
        double log_step = (log_tmax - log_tmin) / (config.dynamical.num_temp_bins - 1);
        for (uint64_t i = 0; i < config.dynamical.num_temp_bins; i++) {
            temperatures[i] = std::exp(log_tmin + i * log_step);
        }
    }
    
    if (use_config_operators) {
        // Configuration-based operator construction
        if (rank == 0) {
            std::cout << "  Operator type: " << config.dynamical.operator_type 
                      << " (" << config.dynamical.basis << " basis)\n";
        }
        
        // Parse configuration
        auto spin_combinations = parse_spin_combinations(config.dynamical.spin_combinations);
        auto momentum_points = parse_momentum_points(config.dynamical.momentum_points);
        auto polarization = parse_polarization(config.dynamical.polarization);
        
        // Get positions file
        std::string positions_file = config.system.hamiltonian_dir + "/positions.dat";
        
        // Determine fixed-Sz parameters
        bool use_fixed_sz = config.system.use_fixed_sz;
        int64_t n_up = (use_fixed_sz && config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        // Construct operators
        std::vector<Operator> obs_1, obs_2;
        std::vector<std::string> names;
        
        construct_operators_from_config(
            config.dynamical.operator_type,
            config.dynamical.basis,
            spin_combinations,
            momentum_points,
            polarization,
            config.dynamical.theta,
            config.dynamical.unit_cell_size,
            config.system.num_sites,
            config.system.spin_length,
            use_fixed_sz,
            n_up,
            positions_file,
            obs_1,
            obs_2,
            names
        );
        
        if (rank == 0) {
            std::cout << "  Operators: " << obs_1.size() << " pair(s)\n";
        }
        
        // ============================================================
        // MPI Task Distribution
        // ============================================================
        
        // Decide whether to use optimized multi-temperature workflow
        int num_operators = obs_1.size();
        int num_temps = config.dynamical.num_temp_bins;
        bool use_optimized_multi_temp = (num_temps > 1);
        
        if (rank == 0 && use_optimized_multi_temp) {
            std::cout << "  Multi-temperature optimization enabled (" << num_temps << " temps)\n";
        }
        
        struct DynTask {
            int temp_idx;
            int op_idx;
            size_t weight;  // estimated cost (number of operators * samples)
            bool is_multi_temp;  // True if this task handles all temperatures for one operator
        };
        
        std::vector<DynTask> all_tasks;
        
        if (rank == 0) {
            if (use_optimized_multi_temp) {
                // OPTIMIZED: Create one task per operator (handles all temperatures)
                for (int o = 0; o < num_operators; o++) {
                    size_t weight = params.num_samples * params.krylov_dim * num_temps;
                    all_tasks.push_back({0, o, weight, true});
                }
            } else {
                // Standard: Create one task per (temperature, operator) pair
                for (int t = 0; t < num_temps; t++) {
                    for (int o = 0; o < num_operators; o++) {
                        size_t weight = params.num_samples * params.krylov_dim;
                        all_tasks.push_back({t, o, weight, false});
                    }
                }
                std::cout << "\nStandard Mode: " << all_tasks.size() << " tasks = "
                          << num_temps << " temperatures × " << num_operators << " operators\n";
            }
            
            // Sort by weight (descending) for better load balance
            std::sort(all_tasks.begin(), all_tasks.end(), 
                      [](const DynTask& a, const DynTask& b) { return a.weight > b.weight; });
            
            std::cout << "Running on " << size << " MPI rank(s)\n";
        }
        
        // Broadcast optimization flag and task count
        int num_tasks = all_tasks.size();
        #ifdef WITH_MPI
        MPI_Bcast(&use_optimized_multi_temp, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            all_tasks.resize(num_tasks);
        }
        
        // Broadcast all tasks
        for (int i = 0; i < num_tasks; i++) {
            int buf[3] = {all_tasks[i].temp_idx, all_tasks[i].op_idx, all_tasks[i].is_multi_temp ? 1 : 0};
            size_t w = all_tasks[i].weight;
            MPI_Bcast(buf, 3, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                all_tasks[i] = {buf[0], buf[1], w, buf[2] != 0};
            }
        }
        #endif
        
        // Lambda to process a single task (single temperature, single operator)
        auto process_task_single = [&](const DynTask& task) -> bool {
            int t_idx = task.temp_idx;
            int op_idx = task.op_idx;
            double temperature = temperatures[t_idx];
            
            DynamicalResponseResults results;
            
#ifdef WITH_CUDA
            if (config.dynamical.use_gpu) {
                // Check for Fixed-Sz mode (not yet supported on GPU)
                if (config.system.use_fixed_sz) {
                    if (rank == 0) {
                        std::cout << "  Note: Fixed-Sz GPU support not yet implemented, using CPU" << std::endl;
                    }
                    // Fall through to CPU path
                } else {
                    // GPU acceleration path
                    try {
                        // Convert operators to GPU
                        GPUOperator gpu_ham(config.system.num_sites, config.system.spin_length);
                        GPUOperator gpu_obs1(config.system.num_sites, config.system.spin_length);
                        GPUOperator gpu_obs2(config.system.num_sites, config.system.spin_length);
                    
                    if (!convertOperatorToGPU(ham, gpu_ham) || 
                        !convertOperatorToGPU(obs_1[op_idx], gpu_obs1) ||
                        !convertOperatorToGPU(obs_2[op_idx], gpu_obs2)) {
                        std::cerr << "  GPU operator conversion failed, falling back to CPU" << std::endl;
                        throw std::runtime_error("GPU conversion failed");
                    }
                    
                    // Call GPU FTLM thermal expectation
                    auto [temps, exps, suscept, exp_err, sus_err] = GPUEDWrapper::runGPUThermalExpectation(
                        &gpu_ham, &gpu_obs1,
                        N, params.num_samples, params.krylov_dim,
                        temperature, temperature, 1,  // Single temperature
                        params.random_seed
                    );
                    
                        // Package results for dynamical correlation
                        // Note: This is thermal expectation, not full dynamical correlation
                        // For full dynamical correlation with GPU, need different approach
                        std::cout << "  Note: GPU currently supports thermal expectation only" << std::endl;
                        throw std::runtime_error("Full GPU dynamical correlation not implemented for multi-sample");
                        
                    } catch (const std::exception& e) {
                        if (rank == 0) {
                            std::cerr << "  GPU computation failed: " << e.what() << ", using CPU" << std::endl;
                        }
                        // Fall through to CPU path
                    }
                }
            }
#endif
            
            // CPU computation path
            {
                // Create function wrappers for this operator pair
                auto O1_func = [&obs_1, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                    obs_1[op_idx].apply(in, out, dim);
                };
                
                auto O2_func = [&obs_2, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                    obs_2[op_idx].apply(in, out, dim);
                };
                
                // Compute response on CPU
                results = compute_dynamical_correlation(
                    H_func, O1_func, O2_func, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperature,
                    config.workflow.output_dir,
                    ground_state_energy
                );
            }
            
            // Save results to HDF5
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            std::string op_name = names[op_idx];
            if (config.dynamical.num_temp_bins > 1) {
                op_name += "_T" + std::to_string(temperature);
            }
            HDF5IO::saveDynamicalResponseFull(
                h5_file, op_name,
                results.frequencies, results.spectral_function, results.spectral_function_imag,
                results.spectral_error, results.spectral_error_imag,
                results.total_samples, temperature
            );
            
            return true;
        };
        
        // Lambda to process all temperatures for one operator (OPTIMIZED!)
        auto process_operator_all_temps = [&](int op_idx) -> bool {
            if (rank == 0) {
                std::cout << "\n=== OPTIMIZED: Processing operator " << names[op_idx] 
                          << " for ALL " << temperatures.size() << " temperatures with SINGLE Lanczos run ===\n";
            }
            
            // Use optimized multi-temperature function
            // This runs Lanczos once per sample, then computes all temperatures efficiently
            std::map<double, DynamicalResponseResults> results_map;
            
#ifdef WITH_CUDA
            if (config.dynamical.use_gpu) {
                // Check for Fixed-Sz mode (not yet supported on GPU)
                if (config.system.use_fixed_sz) {
                    if (rank == 0) {
                        std::cout << "  Note: Fixed-Sz GPU support not yet implemented, using CPU" << std::endl;
                    }
                    // Fall through to CPU path
                } else {
                    // GPU acceleration path
                    try {
                        if (rank == 0) {
                            std::cout << "Using GPU for multi-temperature computation\n";
                        }
                        
                        // Convert operators to GPU
                        GPUOperator gpu_ham(config.system.num_sites, config.system.spin_length);
                        GPUOperator gpu_obs1(config.system.num_sites, config.system.spin_length);
                        GPUOperator gpu_obs2(config.system.num_sites, config.system.spin_length);
                        
                        if (!convertOperatorToGPU(ham, gpu_ham) || 
                            !convertOperatorToGPU(obs_1[op_idx], gpu_obs1) ||
                            !convertOperatorToGPU(obs_2[op_idx], gpu_obs2)) {
                            throw std::runtime_error("GPU operator conversion failed");
                        }
                        
                        // Call optimized GPU multi-temperature dynamical correlation
                        auto gpu_results = GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp(
                            &gpu_ham, &gpu_obs1, &gpu_obs2,
                            N, params.num_samples, params.krylov_dim,
                            config.dynamical.omega_min,
                            config.dynamical.omega_max,
                            config.dynamical.num_omega_points,
                            params.broadening,
                            temperatures,
                            params.random_seed,
                            ground_state_energy
                        );
                        
                        // Convert GPU results to DynamicalResponseResults format
                        for (const auto& [temp, result_tuple] : gpu_results) {
                            auto [freqs, S_real, S_imag] = result_tuple;
                            
                            DynamicalResponseResults result;
                            result.frequencies = freqs;
                            result.spectral_function = S_real;
                            result.spectral_function_imag = S_imag;
                            // Initialize error vectors to zero (GPU computation doesn't provide errors yet)
                            result.spectral_error.resize(freqs.size(), 0.0);
                            result.spectral_error_imag.resize(freqs.size(), 0.0);
                            result.total_samples = params.num_samples;
                            
                            results_map[temp] = result;
                        }
                        
                        if (rank == 0) {
                            std::cout << "  GPU multi-temperature computation successful!" << std::endl;
                        }
                        
                    } catch (const std::exception& e) {
                        if (rank == 0) {
                            std::cerr << "  GPU computation failed: " << e.what() << ", using CPU" << std::endl;
                        }
                        // Fall through to CPU path
                    }
                }
            }
#endif
            
            // CPU computation path (only if GPU didn't produce results)
            if (results_map.empty()) {
                // Create function wrappers for this operator pair
                auto O1_func = [&obs_1, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                    obs_1[op_idx].apply(in, out, dim);
                };
                
                auto O2_func = [&obs_2, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                    obs_2[op_idx].apply(in, out, dim);
                };
                
                if (params.num_samples == 1) {
                    // Single sample mode - use state-based optimization
                    // Generate a random state
                    ComplexVector state(N);
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<double> dist(-1.0, 1.0);
                    for (uint64_t i = 0; i < N; i++) {
                        state[i] = Complex(dist(gen), dist(gen));
                    }
                    double norm = cblas_dznrm2(N, state.data(), 1);
                    Complex scale(1.0/norm, 0.0);
                    cblas_zscal(N, &scale, state.data(), 1);
                    
                    results_map = compute_dynamical_correlation_state_multi_temperature(
                        H_func, O1_func, O2_func, state, N, params,
                        config.dynamical.omega_min,
                        config.dynamical.omega_max,
                        config.dynamical.num_omega_points,
                        temperatures,
                        ground_state_energy
                    );
                } else {
                    // Multiple samples - use multi-sample multi-temperature optimization!
                    if (rank == 0) {
                        std::cout << "Using multi-sample multi-temperature optimization\n";
                        std::cout << "Lanczos will run " << params.num_samples 
                                  << " times, then compute " << temperatures.size() 
                                  << " temperatures from cached data\n";
                    }
                    
                    results_map = compute_dynamical_correlation_multi_sample_multi_temperature(
                        H_func, O1_func, O2_func, N, params,
                        config.dynamical.omega_min,
                        config.dynamical.omega_max,
                        config.dynamical.num_omega_points,
                        temperatures,
                        ground_state_energy,
                        config.workflow.output_dir
                    );
                }
            }
            
            // Save results for all temperatures to HDF5
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            for (const auto& [temperature, results] : results_map) {
                std::string op_name = names[op_idx];
                if (temperatures.size() > 1) {
                    op_name += "_T" + std::to_string(temperature);
                }
                HDF5IO::saveDynamicalResponseFull(
                    h5_file, op_name,
                    results.frequencies, results.spectral_function, results.spectral_function_imag,
                    results.spectral_error, results.spectral_error_imag,
                    results.total_samples, temperature
                );
            }
            
            return true;
        };
        
        // Execute tasks with dynamic work distribution
        int local_processed_count = 0;
        
        #ifdef WITH_MPI
        if (size > 1 && use_optimized_multi_temp) {
            // ============================================================
            // SYNCHRONIZED MODE: All ranks process the same operator at once
            // Required because compute_dynamical_correlation_multi_sample_multi_temperature
            // uses MPI collectives (Barrier, Reduce) internally for sample distribution.
            // The master-worker pattern would cause collective mismatches since
            // different ranks would be processing different operators.
            // ============================================================
            for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
                const auto& task = all_tasks[task_idx];
                if (rank == 0) {
                    std::cout << "\n--- Task " << (task_idx + 1) << " / " << num_tasks
                              << ": Operator " << names[task.op_idx] << " (ALL temperatures, " << size << " MPI ranks) ---\n";
                }
                if (process_operator_all_temps(task.op_idx)) {
                    local_processed_count++;
                }
            }
        } else if (size > 1 && !use_optimized_multi_temp) {
            // Master-worker pattern: safe for single-temperature tasks
            // (process_task_single does NOT use MPI collectives internally)
            const int TASK_TAG = 1;
            const int DONE_TAG = 2;
            const int STOP_TAG = 3;
            
            if (rank == 0) {
                // Master: distribute tasks dynamically
                int next_task = 0;
                
                // Send initial tasks to all workers
                int first_idle_worker = size;  // track workers that got no task
                for (int r = 1; r < size && next_task < num_tasks; r++) {
                    MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                    next_task++;
                    first_idle_worker = r + 1;
                }
                
                // Send STOP_TAG to workers that didn't get any task
                for (int r = first_idle_worker; r < size; r++) {
                    int dummy = -1;
                    MPI_Send(&dummy, 1, MPI_INT, r, STOP_TAG, MPI_COMM_WORLD);
                }
                
                // Process tasks on rank 0 while managing other workers
                int completed = 0;
                while (completed < num_tasks) {
                    // Check if rank 0 can grab a task
                    if (next_task < num_tasks) {
                        int my_task = next_task;
                        next_task++;
                        
                        const auto& task = all_tasks[my_task];
                        std::cout << "Rank 0 processing task " << (my_task + 1) << "/" << num_tasks
                                  << " (T=" << temperatures[task.temp_idx]
                                  << ", op=" << names[task.op_idx] << ")\n";
                        if (process_task_single(task)) {
                            local_processed_count++;
                        }
                        completed++;
                    }
                    
                    // Check for completed tasks from other workers (non-blocking)
                    int flag;
                    MPI_Status status;
                    MPI_Iprobe(MPI_ANY_SOURCE, DONE_TAG, MPI_COMM_WORLD, &flag, &status);
                    
                    if (flag) {
                        int done_task;
                        MPI_Recv(&done_task, 1, MPI_INT, status.MPI_SOURCE, DONE_TAG, MPI_COMM_WORLD, &status);
                        completed++;
                        
                        if (next_task < num_tasks) {
                            MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                            next_task++;
                        } else {
                            int dummy = -1;
                            MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE, STOP_TAG, MPI_COMM_WORLD);
                        }
                    }
                }
            } else {
                // Worker: request and process tasks
                while (true) {
                    int task_id;
                    MPI_Status status;
                    MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    
                    if (status.MPI_TAG == STOP_TAG) {
                        break;
                    }
                    
                    const auto& task = all_tasks[task_id];
                    std::cout << "Rank " << rank << " processing task " << (task_id + 1) << "/" << num_tasks
                              << " (T=" << temperatures[task.temp_idx]
                              << ", op=" << names[task.op_idx] << ")\n";
                    if (process_task_single(task)) {
                        local_processed_count++;
                    }
                    
                    MPI_Send(&task_id, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
                }
            }
        } else
        #endif
        {
            // Sequential execution (no MPI or single rank)
            for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
                const auto& task = all_tasks[task_idx];
                
                if (task.is_multi_temp) {
                    if (rank == 0) {
                        std::cout << "\n--- Task " << (task_idx + 1) << " / " << num_tasks
                                  << ": Operator " << names[task.op_idx] << " (ALL temperatures) ---\n";
                    }
                    if (process_operator_all_temps(task.op_idx)) {
                        local_processed_count++;
                    }
                } else {
                    if (rank == 0) {
                        std::cout << "\n--- Task " << (task_idx + 1) << " / " << num_tasks
                                  << ": T = " << temperatures[task.temp_idx]
                                  << ", operator: " << names[task.op_idx] << " ---\n";
                    }
                    if (process_task_single(task)) {
                        local_processed_count++;
                    }
                }
            }
        }
        
        #ifdef WITH_MPI
        // Gather statistics
        int total_processed_count;
        MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "\nProcessed " << total_processed_count << "/" << num_tasks << " tasks successfully.\n";
        }
        #else
        if (rank == 0) {
            std::cout << "\nProcessed " << local_processed_count << "/" << num_tasks << " tasks successfully.\n";
        }
        #endif
        
    } else {
        // ============================================================
        // Legacy file-based operator loading
        // ============================================================
        if (rank == 0) std::cout << "\nUsing legacy file-based operator loading\n";
        
        if (config.dynamical.operator_file.empty()) {
            std::cerr << "Error: --dyn-operator=<file> is required for dynamical response\n";
            return;
        }
        
        std::string op_path = config.system.hamiltonian_dir + "/" + config.dynamical.operator_file;
        Operator op(config.system.num_sites, config.system.spin_length);
        op.loadFromInterAllFile(op_path);
        // Also load three-body terms if a companion file exists
        {
            std::string op_3body = op_path + ".3body";
            std::ifstream test_3b(op_3body);
            if (test_3b.good()) {
                op.loadThreeBodyTerm(op_3body);
            }
        }
        
        auto O_func = [&op](const Complex* in, Complex* out, uint64_t dim) {
            op.apply(in, out, dim);
        };
        
        // Compute for each temperature
        for (uint64_t t_idx = 0; t_idx < config.dynamical.num_temp_bins; t_idx++) {
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
                // Also load three-body terms for second operator
                {
                    std::string op2_3body = op2_path + ".3body";
                    std::ifstream test_3b2(op2_3body);
                    if (test_3b2.good()) {
                        op2.loadThreeBodyTerm(op2_3body);
                    }
                }
                
                auto O2_func = [&op2](const Complex* in, Complex* out, uint64_t dim) {
                    op2.apply(in, out, dim);
                };
                
                results = compute_dynamical_correlation(
                    H_func, O_func, O2_func, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperature,
                    config.workflow.output_dir,
                    ground_state_energy
                );
            } else {
                // Same operator: ⟨O†(t)O⟩ (default auto-correlation)
                std::cout << "Computing dynamical response ⟨O†(t)O⟩...\n";
                results = compute_dynamical_response_thermal(
                    H_func, O_func, N, params,
                    config.dynamical.omega_min,
                    config.dynamical.omega_max,
                    config.dynamical.num_omega_points,
                    temperature,
                    config.workflow.output_dir
                );
            }
            
            // Save results for this temperature to HDF5
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            std::string op_name = config.dynamical.output_prefix;
            if (config.dynamical.num_temp_bins > 1) {
                op_name += "_T" + std::to_string(temperature);
            }
            HDF5IO::saveDynamicalResponseFull(
                h5_file, op_name,
                results.frequencies, results.spectral_function, results.spectral_function_imag,
                results.spectral_error, results.spectral_error_imag,
                results.total_samples, temperature
            );
            if (rank == 0) std::cout << "Results saved to HDF5: " << h5_file << " (" << op_name << ")\n";
        }
    }
    
    if (rank == 0) {
        std::cout << "\nDynamical response complete.\n";
        std::cout << "Frequency range: [" << config.dynamical.omega_min << ", " << config.dynamical.omega_max << "]\n";
        std::cout << "Number of points: " << config.dynamical.num_omega_points << "\n";
    }
}

/**
 * @brief Compute static response (thermal expectation values)
 */
void compute_static_response_workflow(const EDConfig& config) {
    // Get MPI rank and size
    int rank = 0, size = 1;
    #ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif
    
    if (rank == 0) {
        std::cout << "\nStatic Response Calculation\n";
        
#ifdef WITH_CUDA
        if (config.static_resp.use_gpu) {
            std::cout << "  GPU: enabled";
            if (config.system.use_fixed_sz) {
                std::cout << " (disabled for fixed-Sz)";
            }
            std::cout << "\n";
        }
#endif
    }
    
    // Check if using configuration-based or legacy file-based operator loading
    bool use_config_operators = config.static_resp.operator_file.empty() || 
                                config.static_resp.operator_type != "sum";
    
    // Prepare Hamiltonian
    Operator ham(config.system.num_sites, config.system.spin_length);
    std::string interaction_file = config.system.hamiltonian_dir + "/" + config.system.interaction_file;
    std::string single_site_file = config.system.hamiltonian_dir + "/" + config.system.single_site_file;
    ham.loadFromInterAllFile(interaction_file);
    ham.loadFromFile(single_site_file);
    
    // Load three-body terms if specified
    if (!config.system.three_body_file.empty()) {
        std::string three_body_file = config.system.hamiltonian_dir + "/" + config.system.three_body_file;
        if (std::filesystem::exists(three_body_file)) {
            if (rank == 0) std::cout << "Loading three-body terms from: " << three_body_file << "\n";
            ham.loadThreeBodyTerm(three_body_file);
        }
    }
    
    // Hilbert space dimension
    uint64_t N;
    if (config.system.use_fixed_sz) {
        // Use binomial coefficient C(num_sites, n_up) for fixed-Sz sector
        int64_t n_up_dim = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        N = 1;
        for (int64_t i = 0; i < n_up_dim; i++) {
            N = N * (config.system.num_sites - i) / (i + 1);
        }
        if (rank == 0) std::cout << "Fixed-Sz static response: dim=" << N << " (n_up=" << n_up_dim << ")\n";
    } else {
        N = 1ULL << config.system.num_sites;
    }
    
    // Create function wrapper for Hamiltonian
    auto H_func = [&ham](const Complex* in, Complex* out, uint64_t dim) {
        ham.apply(in, out, dim);
    };
    
    // Setup parameters
    StaticResponseParameters params;
    params.num_samples = config.static_resp.num_random_states;
    params.krylov_dim = config.static_resp.krylov_dim;
    params.random_seed = config.static_resp.random_seed;
    
    // Ensure output directory exists
    create_directory_mpi_safe(config.workflow.output_dir);
    
    if (rank == 0) {
        std::cout << "Random states: " << params.num_samples << "\n";
        std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
        std::cout << "Temperature range: [" << config.static_resp.temp_min << ", " << config.static_resp.temp_max << "]\n";
    }
    
    if (use_config_operators) {
        // ============================================================
        // Configuration-based operator construction (canonical `ED dssf` knobs)
        // ============================================================
        if (rank == 0) {
            std::cout << "\nUsing configuration-based operator construction\n";
            std::cout << "  Operator type: " << config.static_resp.operator_type << "\n";
            std::cout << "  Basis: " << config.static_resp.basis << "\n";
            std::cout << "  Spin combinations: " << config.static_resp.spin_combinations << "\n";
        }
        
        // Parse configuration
        auto spin_combinations = parse_spin_combinations(config.static_resp.spin_combinations);
        auto momentum_points = parse_momentum_points(config.static_resp.momentum_points);
        auto polarization = parse_polarization(config.static_resp.polarization);
        
        // Get positions file
        std::string positions_file = config.system.hamiltonian_dir + "/positions.dat";
        
        // Determine fixed-Sz parameters
        bool use_fixed_sz = config.system.use_fixed_sz;
        int64_t n_up = (use_fixed_sz && config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        // Construct operators
        std::vector<Operator> obs_1, obs_2;
        std::vector<std::string> names;
        
        construct_operators_from_config(
            config.static_resp.operator_type,
            config.static_resp.basis,
            spin_combinations,
            momentum_points,
            polarization,
            config.static_resp.theta,
            config.static_resp.unit_cell_size,
            config.system.num_sites,
            config.system.spin_length,
            use_fixed_sz,
            n_up,
            positions_file,
            obs_1,
            obs_2,
            names
        );
        
        if (rank == 0) {
            std::cout << "Constructed " << obs_1.size() << " operator pair(s)\n";
        }
        
        // ============================================================
        // MPI Task Distribution (per-operator-pair sharding)
        // ============================================================
        
        // Build task list: each task is an operator pair
        struct StaticTask {
            int op_idx;
            size_t weight;  // estimated cost (number of samples * krylov dimension)
        };
        
        std::vector<StaticTask> all_tasks;
        int num_operators = obs_1.size();
        
        if (rank == 0) {
            // Create tasks
            for (int o = 0; o < num_operators; o++) {
                // Weight is proportional to samples, krylov dimension, and temperature points
                size_t weight = params.num_samples * params.krylov_dim * config.static_resp.num_temp_points;
                all_tasks.push_back({o, weight});
            }
            
            // Sort by weight (descending) for better load balance
            std::sort(all_tasks.begin(), all_tasks.end(), 
                      [](const StaticTask& a, const StaticTask& b) { return a.weight > b.weight; });
            
            std::cout << "\nMPI Parallelization: " << all_tasks.size() << " tasks = "
                      << num_operators << " operators\n";
            std::cout << "Running on " << size << " MPI rank(s)\n";
        }
        
        // Broadcast task count
        int num_tasks = all_tasks.size();
        #ifdef WITH_MPI
        MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            all_tasks.resize(num_tasks);
        }
        
        // Broadcast all tasks
        for (int i = 0; i < num_tasks; i++) {
            int op = all_tasks[i].op_idx;
            size_t w = all_tasks[i].weight;
            MPI_Bcast(&op, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                all_tasks[i] = {op, w};
            }
        }
        #endif
        
        // Lambda to process a single task
        auto process_task = [&](const StaticTask& task) -> bool {
            int op_idx = task.op_idx;
            
            StaticResponseResults results;
            
#ifdef WITH_CUDA
            if (config.static_resp.use_gpu) {
                // Check for Fixed-Sz mode (not yet supported on GPU)
                if (config.system.use_fixed_sz) {
                    if (rank == 0) {
                        std::cout << "  Note: Fixed-Sz GPU support not yet implemented, using CPU" << std::endl;
                    }
                    // Fall through to CPU path
                } else {
                    // GPU acceleration path
                    try {
                        // Convert operators to GPU
                        GPUOperator gpu_ham(config.system.num_sites, config.system.spin_length);
                        GPUOperator gpu_obs1(config.system.num_sites, config.system.spin_length);
                        GPUOperator gpu_obs2(config.system.num_sites, config.system.spin_length);
                    
                    if (!convertOperatorToGPU(ham, gpu_ham) || 
                        !convertOperatorToGPU(obs_1[op_idx], gpu_obs1) ||
                        !convertOperatorToGPU(obs_2[op_idx], gpu_obs2)) {
                        throw std::runtime_error("GPU operator conversion failed");
                    }
                    
                    // Call GPU static correlation - returns tuple
                    auto [temps, corr_real, corr_imag, err_real, err_imag] = 
                        GPUEDWrapper::runGPUStaticCorrelation(
                            &gpu_ham, &gpu_obs1, &gpu_obs2,
                            N, params.num_samples, params.krylov_dim,
                            config.static_resp.temp_min,
                            config.static_resp.temp_max,
                            config.static_resp.num_temp_points,
                            params.random_seed
                        );
                    
                    // Package into results struct
                    // Note: GPU returns complex correlation (real, imag parts)
                    // CPU returns expectation value and susceptibility
                    // For now, store real part as expectation
                    results.temperatures = temps;
                    results.expectation = corr_real;
                    results.expectation_error = err_real;
                    // TODO: Map imag part appropriately or compute susceptibility on GPU
                    results.total_samples = params.num_samples;
                    
                        if (rank == 0) {
                            std::cout << "  GPU computation successful for operator " << names[op_idx] << std::endl;
                        }
                        
                    } catch (const std::exception& e) {
                        if (rank == 0) {
                            std::cerr << "  GPU computation failed: " << e.what() << ", using CPU" << std::endl;
                        }
                        // Fall through to CPU path
                    }
                }
            }
#endif
            
            // CPU computation path
            if (results.temperatures.empty()) {  // Only compute on CPU if GPU didn't succeed
                // Create function wrappers for this operator pair
                auto O1_func = [&obs_1, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                    obs_1[op_idx].apply(in, out, dim);
                };
                
                auto O2_func = [&obs_2, op_idx](const Complex* in, Complex* out, uint64_t dim) {
                    obs_2[op_idx].apply(in, out, dim);
                };
                
                // Compute response on CPU
                results = compute_static_response(
                    H_func, O1_func, O2_func, N, params,
                    config.static_resp.temp_min,
                    config.static_resp.temp_max,
                    config.static_resp.num_temp_points,
                    config.workflow.output_dir
                );
            }
            
            // Save results to HDF5
            std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
            HDF5IO::saveStaticResponse(
                h5_file, names[op_idx],
                results.temperatures, results.expectation, results.expectation_error,
                results.variance, results.variance_error,
                results.susceptibility, results.susceptibility_error,
                results.total_samples
            );
            
            return true;
        };
        
        // Execute tasks with dynamic work distribution
        int local_processed_count = 0;
        
        #ifdef WITH_MPI
        if (size > 1) {
            // MPI tags for communication
            const int TASK_TAG = 1;
            const int DONE_TAG = 2;
            const int STOP_TAG = 3;
            
            if (rank == 0) {
                // Master: distribute tasks dynamically
                int next_task = 0;
                
                // Send initial tasks to all workers
                int first_idle_worker = size;  // track workers that got no task
                for (int r = 1; r < size && next_task < num_tasks; r++) {
                    MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                    next_task++;
                    first_idle_worker = r + 1;
                }
                
                // Send STOP_TAG to workers that didn't get any task
                for (int r = first_idle_worker; r < size; r++) {
                    int dummy = -1;
                    MPI_Send(&dummy, 1, MPI_INT, r, STOP_TAG, MPI_COMM_WORLD);
                }
                
                // Process tasks on rank 0 while managing other workers
                int completed = 0;
                while (completed < num_tasks) {
                    // Check if rank 0 can grab a task
                    if (next_task < num_tasks) {
                        int my_task = next_task;
                        next_task++;
                        
                        std::cout << "Rank 0 processing task " << (my_task + 1) << "/" << num_tasks
                                  << " (op=" << names[all_tasks[my_task].op_idx] << ")\n";
                        
                        if (process_task(all_tasks[my_task])) {
                            local_processed_count++;
                        }
                        completed++;
                    }
                    
                    // Check for completed tasks from other workers (non-blocking)
                    int flag;
                    MPI_Status status;
                    MPI_Iprobe(MPI_ANY_SOURCE, DONE_TAG, MPI_COMM_WORLD, &flag, &status);
                    
                    if (flag) {
                        int done_task;
                        MPI_Recv(&done_task, 1, MPI_INT, status.MPI_SOURCE, DONE_TAG, MPI_COMM_WORLD, &status);
                        completed++;
                        
                        if (next_task < num_tasks) {
                            MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                            next_task++;
                        } else {
                            int dummy = -1;
                            MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE, STOP_TAG, MPI_COMM_WORLD);
                        }
                    }
                }
            } else {
                // Worker: request and process tasks
                while (true) {
                    int task_id;
                    MPI_Status status;
                    MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    
                    if (status.MPI_TAG == STOP_TAG) {
                        break;
                    }
                    
                    std::cout << "Rank " << rank << " processing task " << (task_id + 1) << "/" << num_tasks
                              << " (op=" << names[all_tasks[task_id].op_idx] << ")\n";
                    
                    if (process_task(all_tasks[task_id])) {
                        local_processed_count++;
                    }
                    
                    MPI_Send(&task_id, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
                }
            }
        } else
        #endif
        {
            // Sequential execution (no MPI or single rank)
            for (int task_idx = 0; task_idx < num_tasks; task_idx++) {
                if (rank == 0) {
                    std::cout << "  Processing operator: " << names[all_tasks[task_idx].op_idx] << "\n";
                }
                
                if (process_task(all_tasks[task_idx])) {
                    local_processed_count++;
                }
            }
        }
        
        #ifdef WITH_MPI
        // Gather statistics
        int total_processed_count;
        MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "\nProcessed " << total_processed_count << "/" << num_tasks << " tasks successfully.\n";
        }
        #else
        if (rank == 0) {
            std::cout << "\nProcessed " << local_processed_count << "/" << num_tasks << " tasks successfully.\n";
        }
        #endif
        
    } else {
        // ============================================================
        // Legacy file-based operator loading
        // ============================================================
        if (rank == 0) std::cout << "\nUsing legacy file-based operator loading\n";
        
        if (config.static_resp.operator_file.empty()) {
            std::cerr << "Error: --static-operator=<file> is required for static response\n";
            return;
        }
        
        std::string op_path = config.system.hamiltonian_dir + "/" + config.static_resp.operator_file;
        Operator op(config.system.num_sites, config.system.spin_length);
        op.loadFromInterAllFile(op_path);
        
        auto O_func = [&op](const Complex* in, Complex* out, uint64_t dim) {
            op.apply(in, out, dim);
        };
        
        // Compute response
        StaticResponseResults results;
        
        if (config.static_resp.single_operator_mode) {
            // Single operator expectation value: ⟨O⟩
            std::cout << "Computing thermal expectation value ⟨O⟩...\n";
            results = compute_thermal_expectation_value(
                H_func, O_func, N, params,
                config.static_resp.temp_min,
                config.static_resp.temp_max,
                config.static_resp.num_temp_points,
                config.workflow.output_dir
            );
        } else if (!config.static_resp.operator2_file.empty()) {
            // Two different operators: ⟨O₁†O₂⟩
            std::cout << "Computing two-operator static response ⟨O₁†O₂⟩...\n";
            std::string op2_path = config.system.hamiltonian_dir + "/" + config.static_resp.operator2_file;
            Operator op2(config.system.num_sites, config.system.spin_length);
            op2.loadFromInterAllFile(op2_path);
            
            auto O2_func = [&op2](const Complex* in, Complex* out, uint64_t dim) {
                op2.apply(in, out, dim);
            };
            
            results = compute_static_response(
                H_func, O_func, O2_func, N, params,
                config.static_resp.temp_min,
                config.static_resp.temp_max,
                config.static_resp.num_temp_points,
                config.workflow.output_dir
            );
        } else {
            // Same operator: ⟨O†O⟩ (default two-point correlation)
            std::cout << "Computing static response ⟨O†O⟩...\n";
            results = compute_static_response(
                H_func, O_func, O_func, N, params,
                config.static_resp.temp_min,
                config.static_resp.temp_max,
                config.static_resp.num_temp_points,
                config.workflow.output_dir
            );
        }
        
        // Save results to HDF5
        std::string h5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
        HDF5IO::saveStaticResponse(
            h5_file, config.static_resp.output_prefix,
            results.temperatures, results.expectation, results.expectation_error,
            results.variance, results.variance_error,
            results.susceptibility, results.susceptibility_error,
            results.total_samples
        );
        std::cout << "Static response saved to HDF5: " << h5_file << "\n";
    }
}

/**
 * @brief Compute ground state dynamical spin structure factor (T=0 DSSF)
 * 
 * Uses the continued fraction method for efficient ground state dynamics:
 * S(q,ω) = -1/π Im⟨GS| O†(-q) 1/(ω + E₀ - H + iη) O(q) |GS⟩
 * 
 * This is optimal for 32-site ED where:
 * - Fixed-Sz sector has 601M states (~9GB per vector)
 * - Only need to store 2-3 Lanczos vectors (not full spectrum)
 * - Continued fraction avoids explicit eigendecomposition
 */
void compute_ground_state_dssf_workflow(const EDConfig& config) {
    // Get MPI rank and size
    int rank = 0, size = 1;
    #ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "Computing Ground State DSSF (T=0)\n";
        std::cout << "==========================================\n";
        std::cout << "Using continued fraction method for optimal efficiency\n";
    }
    
    // Prepare Hamiltonian
    Operator ham(config.system.num_sites, config.system.spin_length);
    std::string interaction_file = config.system.hamiltonian_dir + "/" + config.system.interaction_file;
    std::string single_site_file = config.system.hamiltonian_dir + "/" + config.system.single_site_file;
    ham.loadFromInterAllFile(interaction_file);
    ham.loadFromFile(single_site_file);
    
    // Load three-body terms if specified
    if (!config.system.three_body_file.empty()) {
        std::string three_body_file = config.system.hamiltonian_dir + "/" + config.system.three_body_file;
        if (std::filesystem::exists(three_body_file)) {
            if (rank == 0) {
                std::cout << "Loading three-body terms from: " << three_body_file << "\n";
            }
            ham.loadThreeBodyTerm(three_body_file);
        }
    }
    
    // Hilbert space dimension
    bool use_fixed_sz = config.system.use_fixed_sz;
    int64_t n_up = (use_fixed_sz && config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
    uint64_t N;
    
    if (use_fixed_sz) {
        // Binomial coefficient for fixed Sz
        uint64_t num_sites = config.system.num_sites;
        N = 1;
        for (uint64_t i = 0; i < n_up; i++) {
            N = N * (num_sites - i) / (i + 1);
        }
        if (rank == 0) {
            std::cout << "Fixed-Sz sector: N_sites=" << num_sites << ", n_up=" << n_up 
                      << ", dim=" << N << "\n";
        }
    } else {
        N = 1ULL << config.system.num_sites;
        if (rank == 0) {
            std::cout << "Full Hilbert space: dim=" << N << "\n";
        }
    }
    
    // Create function wrapper for Hamiltonian
    auto H_func = [&ham](const Complex* in, Complex* out, uint64_t dim) {
        ham.apply(in, out, dim);
    };
    
    // Ensure output directory exists
    create_directory_mpi_safe(config.workflow.output_dir);
    
    // Setup ground state DSSF parameters
    GroundStateDSSFParameters gs_params;
    gs_params.krylov_dim = config.dynamical.krylov_dim > 0 ? config.dynamical.krylov_dim : 300;
    gs_params.omega_min = config.dynamical.omega_min;
    gs_params.omega_max = config.dynamical.omega_max;
    gs_params.num_omega_points = config.dynamical.num_omega_points;
    gs_params.broadening = config.dynamical.broadening;
    gs_params.tolerance = config.diag.tolerance;
    gs_params.full_reorthogonalization = true;  // Important for accuracy
    
    if (rank == 0) {
        std::cout << "Krylov dimension: " << gs_params.krylov_dim << "\n";
        std::cout << "Frequency range: [" << gs_params.omega_min << ", " << gs_params.omega_max << "]\n";
        std::cout << "Frequency points: " << gs_params.num_omega_points << "\n";
        std::cout << "Broadening (eta): " << gs_params.broadening << "\n";
    }
    
    // Parse configuration for operators
    auto spin_combinations = parse_spin_combinations(config.dynamical.spin_combinations);
    auto momentum_points = parse_momentum_points(config.dynamical.momentum_points);
    auto polarization = parse_polarization(config.dynamical.polarization);
    std::string positions_file = config.system.hamiltonian_dir + "/positions.dat";
    
    if (rank == 0) {
        std::cout << "Operator type: " << config.dynamical.operator_type << "\n";
        std::cout << "Basis: " << config.dynamical.basis << "\n";
        std::cout << "Momentum points: " << momentum_points.size() << "\n";
        std::cout << "Spin combinations: " << spin_combinations.size() << "\n";
    }
    
    // Construct operators
    std::vector<Operator> obs_1, obs_2;
    std::vector<std::string> names;
    
    construct_operators_from_config(
        config.dynamical.operator_type,
        config.dynamical.basis,
        spin_combinations,
        momentum_points,
        polarization,
        config.dynamical.theta,
        config.dynamical.unit_cell_size,
        config.system.num_sites,
        config.system.spin_length,
        use_fixed_sz,
        n_up,
        positions_file,
        obs_1, obs_2, names
    );
    
    if (rank == 0) {
        std::cout << "Constructed " << names.size() << " operator pairs\n";
    }
    
    // Find ground state using Lanczos
    if (rank == 0) {
        std::cout << "\n--- Finding ground state ---\n";
    }
    
    // Validate dimension fits in int for solver function signatures
    if (N > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "Error: Hilbert space dimension " << N
                      << " exceeds INT_MAX. The current solver API uses int for "
                      << "dimension parameters. Use fixed-Sz or symmetry reduction."
                      << std::endl;
        }
        return;
    }
    
    ComplexVector ground_state(N);
    double ground_state_energy;
    
    // Check if ground state is already saved in HDF5
    std::string h5_file = config.workflow.output_dir + "/ed_results.h5";
    bool gs_loaded = false;
    
    if (HDF5IO::fileExists(h5_file)) {
        try {
            // Try to load eigenvalue (ground state energy)
            auto eigenvalues = HDF5IO::loadEigenvalues(h5_file);
            if (!eigenvalues.empty()) {
                ground_state_energy = eigenvalues[0];
                
                // Try to load eigenvector (ground state)
                auto gs_vec = HDF5IO::loadEigenvector(h5_file, 0);
                if (gs_vec.size() == N) {
                    std::copy(gs_vec.begin(), gs_vec.end(), ground_state.begin());
                    gs_loaded = true;
                    if (rank == 0) {
                        std::cout << "Loaded ground state from HDF5: E0 = " << ground_state_energy << "\n";
                    }
                }
            }
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cout << "Could not load ground state from HDF5, will compute...\n";
            }
        }
    }
    
    if (!gs_loaded) {
        // Compute ground state
        // Create int-based wrapper for find_ground_state_lanczos
        auto H_func_int = [&ham](const Complex* in, Complex* out, int dim) {
            ham.apply(in, out, static_cast<uint64_t>(dim));
        };
        
        ground_state_energy = find_ground_state_lanczos(
            H_func_int, N, gs_params.krylov_dim, gs_params.tolerance,
            gs_params.full_reorthogonalization, gs_params.reorth_frequency,
            ground_state
        );
        
        if (rank == 0) {
            std::cout << "Computed ground state: E0 = " << ground_state_energy << "\n";
            
            // Save ground state to HDF5
            try {
                std::string h5_path = HDF5IO::createOrOpenFile(config.workflow.output_dir);
                HDF5IO::saveEigenvalues(h5_path, {ground_state_energy});
                std::vector<Complex> gs_vec(ground_state.begin(), ground_state.end());
                HDF5IO::saveEigenvector(h5_path, 0, gs_vec);
                std::cout << "Saved ground state to HDF5: " << h5_path << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to save ground state to HDF5: " << e.what() << "\n";
            }
        }
    }
    
    // Compute DSSF for each operator pair
    // Distribute work across MPI ranks
    std::vector<int> my_tasks;
    for (int i = rank; i < (int)names.size(); i += size) {
        my_tasks.push_back(i);
    }
    
    if (rank == 0) {
        std::cout << "\n--- Computing S(q,ω) for " << names.size() << " operators ---\n";
    }
    
    for (int op_idx : my_tasks) {
        std::cout << "[Rank " << rank << "] Processing: " << names[op_idx] << "\n";
        
        // Create function wrappers (with int signature for FTLM functions)
        auto H_func_int = [&ham](const Complex* in, Complex* out, int dim) {
            ham.apply(in, out, static_cast<uint64_t>(dim));
        };
        
        auto O1_func = [&obs_1, op_idx](const Complex* in, Complex* out, int dim) {
            obs_1[op_idx].apply(in, out, static_cast<uint64_t>(dim));
        };
        
        auto O2_func = [&obs_2, op_idx](const Complex* in, Complex* out, int dim) {
            obs_2[op_idx].apply(in, out, static_cast<uint64_t>(dim));
        };
        
        // Compute ground state DSSF using continued fraction method
        auto results = compute_ground_state_cross_correlation(
            H_func_int, O1_func, O2_func, ground_state, ground_state_energy, N, gs_params
        );
        
        // Save results to unified HDF5 file
        std::string h5_path = HDF5IO::createOrOpenFile(config.workflow.output_dir);
        std::string op_name = "ground_state_dssf/" + names[op_idx];
        HDF5IO::saveDynamicalResponseFull(
            h5_path, op_name,
            results.frequencies, results.spectral_function, results.spectral_function_imag,
            results.spectral_error, results.spectral_error_imag,
            1, 0.0  // T=0 ground state
        );
        std::cout << "[Rank " << rank << "] Saved to HDF5: " << op_name << "\n";
    }
    
    #ifdef WITH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "Ground State DSSF Complete\n";
        std::cout << "Results saved to: " << config.workflow.output_dir << "/ed_results.h5\n";
        std::cout << "==========================================\n";
    }
}

/**
 * @brief Print eigenvalue summary
 */
void print_eigenvalue_summary(const std::vector<double>& eigenvalues, uint64_t max_show) {
    std::cout << "\nEigenvalues:\n";
    for (size_t i = 0; i < eigenvalues.size() && i < max_show; i++) {
        std::cout << "  " << i << ": " << std::setprecision(12) << eigenvalues[i] << "\n";
    }
    if (eigenvalues.size() > max_show) {
        std::cout << "  ... (" << eigenvalues.size() - max_show << " more)\n";
    }
}

