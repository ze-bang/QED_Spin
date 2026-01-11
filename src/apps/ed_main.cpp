#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <filesystem>
#include <ed/core/ed_config.h>
#include <ed/core/ed_config_adapter.h>
#include <ed/core/ed_wrapper.h>
#include <ed/core/ed_wrapper_streaming.h>
#include <ed/core/disk_streaming_symmetry.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
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
 * @file ed_main.cpp
 * @brief Elegant main entry point for exact diagonalization
 * 
 * This is a complete rewrite of ed_run.cpp with:
 * - Clean configuration management
 * - Separated concerns
 * - No massive if-else chains
 * - Support for config files
 * - MPI support for parallel TPQ sample execution
 */

// ============================================================================
// HELPER FUNCTIONS FOR OPERATOR CONSTRUCTION
// ============================================================================

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
 * @brief Helper function for cross product
 */
std::array<double, 3> cross_product(const std::vector<double>& a, const std::vector<double>& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

/**
 * @brief Helper function to normalize a vector
 */
std::array<double, 3> normalize(const std::array<double, 3>& v) {
    double norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm < 1e-10) {
        return {0.0, 0.0, 0.0};
    }
    return {v[0]/norm, v[1]/norm, v[2]/norm};
}

/**
 * @brief Construct operators based on configuration (like TPQ_DSSF)
 * Returns pairs of operators (obs_1, obs_2) and their names
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
    bool use_xyz_basis = (basis == "xyz");
    
    // Helper to get spin operator name
    auto spin_combination_name = [use_xyz_basis](int op) {
        if (use_xyz_basis) {
            switch (op) {
                case 0: return "Sx";
                case 1: return "Sy";
                case 2: return "Sz";
                default: return "Unknown";
            }
        } else {
            switch (op) {
                case 2: return "Sz";
                case 0: return "Sp";
                case 1: return "Sm";
                default: return "Unknown";
            }
        }
    };
    
    // Iterate over momentum points and spin combinations
    for (size_t qi = 0; qi < momentum_points.size(); ++qi) {
        const auto& Q = momentum_points[qi];
        
        // Pre-compute transverse bases if needed
        std::array<double, 3> transverse_basis_1, transverse_basis_2;
        if (operator_type == "transverse" || operator_type == "transverse_experimental") {
            std::array<double, 3> pol_array = {polarization[0], polarization[1], polarization[2]};
            transverse_basis_1 = pol_array;
            
            // transverse_basis_2 is Q × polarization (cross product)
            auto cross = cross_product(Q, polarization);
            transverse_basis_2 = normalize(cross);
            
            // Handle special case: if Q is parallel to polarization, cross product is zero
            double cross_norm = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
            if (cross_norm < 1e-10) {
                if (std::abs(pol_array[0]) > 0.5) {
                    auto alt_cross = cross_product({0.0, 1.0, 0.0}, polarization);
                    transverse_basis_2 = normalize(alt_cross);
                } else {
                    auto alt_cross = cross_product({1.0, 0.0, 0.0}, polarization);
                    transverse_basis_2 = normalize(alt_cross);
                }
                std::cout << "Warning: Q parallel to polarization, using alternative basis" << std::endl;
            }
        }
        
        for (const auto& combo : spin_combinations) {
            int op_type_1 = combo.first;
            int op_type_2 = combo.second;
            
            // Convert operator indices for ladder basis
            int first = op_type_1;
            int second = op_type_2;
            if (!use_xyz_basis) {
                first = first == 2 ? 2 : 1 - first;  // Convert 0->1(Sp), 1->0(Sm) for first operator
            }
            
            std::string base_name = std::string(spin_combination_name(first)) + std::string(spin_combination_name(second));
            
            if (operator_type == "sum") {
                // Standard sum operators: S^{op1}(Q) S^{op2}(-Q)
                std::stringstream name_ss;
                name_ss << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                
                if (use_fixed_sz) {
                    if (use_xyz_basis) {
                        FixedSzSumOperatorXYZ sum_op_1(num_sites, spin_length, n_up, op_type_1, Q, positions_file);
                        FixedSzSumOperatorXYZ sum_op_2(num_sites, spin_length, n_up, op_type_2, Q, positions_file);
                        obs_1_out.push_back(Operator(sum_op_1));
                        obs_2_out.push_back(Operator(sum_op_2));
                    } else {
                        FixedSzSumOperator sum_op_1(num_sites, spin_length, n_up, op_type_1, Q, positions_file);
                        FixedSzSumOperator sum_op_2(num_sites, spin_length, n_up, op_type_2, Q, positions_file);
                        obs_1_out.push_back(Operator(sum_op_1));
                        obs_2_out.push_back(Operator(sum_op_2));
                    }
                } else {
                    if (use_xyz_basis) {
                        SumOperatorXYZ sum_op_1(num_sites, spin_length, op_type_1, Q, positions_file);
                        SumOperatorXYZ sum_op_2(num_sites, spin_length, op_type_2, Q, positions_file);
                        obs_1_out.push_back(Operator(sum_op_1));
                        obs_2_out.push_back(Operator(sum_op_2));
                    } else {
                        SumOperator sum_op_1(num_sites, spin_length, op_type_1, Q, positions_file);
                        SumOperator sum_op_2(num_sites, spin_length, op_type_2, Q, positions_file);
                        obs_1_out.push_back(Operator(sum_op_1));
                        obs_2_out.push_back(Operator(sum_op_2));
                    }
                }
                
                names_out.push_back(name_ss.str());
                
            } else if (operator_type == "transverse") {
                // Transverse operators for SF/NSF separation
                std::vector<double> e1_vec = {transverse_basis_1[0], transverse_basis_1[1], transverse_basis_1[2]};
                std::vector<double> e2_vec = {transverse_basis_2[0], transverse_basis_2[1], transverse_basis_2[2]};
                
                std::stringstream name_sf, name_nsf;
                name_sf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_NSF";
                name_nsf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_SF";
                
                if (use_fixed_sz) {
                    if (use_xyz_basis) {
                        FixedSzTransverseOperatorXYZ op1_sf(num_sites, spin_length, n_up, op_type_1, Q, e1_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op2_sf(num_sites, spin_length, n_up, op_type_2, Q, e1_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op1_nsf(num_sites, spin_length, n_up, op_type_1, Q, e2_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op2_nsf(num_sites, spin_length, n_up, op_type_2, Q, e2_vec, positions_file);
                        
                        obs_1_out.push_back(Operator(op1_sf));
                        obs_2_out.push_back(Operator(op2_sf));
                        obs_1_out.push_back(Operator(op1_nsf));
                        obs_2_out.push_back(Operator(op2_nsf));
                    } else {
                        FixedSzTransverseOperator op1_sf(num_sites, spin_length, n_up, op_type_1, Q, e1_vec, positions_file);
                        FixedSzTransverseOperator op2_sf(num_sites, spin_length, n_up, op_type_2, Q, e1_vec, positions_file);
                        FixedSzTransverseOperator op1_nsf(num_sites, spin_length, n_up, op_type_1, Q, e2_vec, positions_file);
                        FixedSzTransverseOperator op2_nsf(num_sites, spin_length, n_up, op_type_2, Q, e2_vec, positions_file);
                        
                        obs_1_out.push_back(Operator(op1_sf));
                        obs_2_out.push_back(Operator(op2_sf));
                        obs_1_out.push_back(Operator(op1_nsf));
                        obs_2_out.push_back(Operator(op2_nsf));
                    }
                } else {
                    if (use_xyz_basis) {
                        TransverseOperatorXYZ op1_sf(num_sites, spin_length, op_type_1, Q, e1_vec, positions_file);
                        TransverseOperatorXYZ op2_sf(num_sites, spin_length, op_type_2, Q, e1_vec, positions_file);
                        TransverseOperatorXYZ op1_nsf(num_sites, spin_length, op_type_1, Q, e2_vec, positions_file);
                        TransverseOperatorXYZ op2_nsf(num_sites, spin_length, op_type_2, Q, e2_vec, positions_file);
                        
                        obs_1_out.push_back(Operator(op1_sf));
                        obs_2_out.push_back(Operator(op2_sf));
                        obs_1_out.push_back(Operator(op1_nsf));
                        obs_2_out.push_back(Operator(op2_nsf));
                    } else {
                        TransverseOperator op1_sf(num_sites, spin_length, op_type_1, Q, e1_vec, positions_file);
                        TransverseOperator op2_sf(num_sites, spin_length, op_type_2, Q, e1_vec, positions_file);
                        TransverseOperator op1_nsf(num_sites, spin_length, op_type_1, Q, e2_vec, positions_file);
                        TransverseOperator op2_nsf(num_sites, spin_length, op_type_2, Q, e2_vec, positions_file);
                        
                        obs_1_out.push_back(Operator(op1_sf));
                        obs_2_out.push_back(Operator(op2_sf));
                        obs_1_out.push_back(Operator(op1_nsf));
                        obs_2_out.push_back(Operator(op2_nsf));
                    }
                }
                
                names_out.push_back(name_sf.str());
                names_out.push_back(name_nsf.str());
                
            } else if (operator_type == "sublattice") {
                // Sublattice-resolved operators (only compute upper triangle)
                for (uint64_t sub_i = 0; sub_i < unit_cell_size; ++sub_i) {
                    for (uint64_t sub_j = sub_i; sub_j < unit_cell_size; ++sub_j) {
                        std::stringstream name_ss;
                        name_ss << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2]
                                << "_sub" << sub_i << "_sub" << sub_j;
                        
                        if (use_fixed_sz) {
                            FixedSzSublatticeOperator sub_op_1(sub_i, unit_cell_size, num_sites, spin_length, n_up, op_type_1, Q, positions_file);
                            FixedSzSublatticeOperator sub_op_2(sub_j, unit_cell_size, num_sites, spin_length, n_up, op_type_2, Q, positions_file);
                            obs_1_out.push_back(Operator(sub_op_1));
                            obs_2_out.push_back(Operator(sub_op_2));
                        } else {
                            SublatticeOperator sub_op_1(sub_i, unit_cell_size, num_sites, spin_length, op_type_1, Q, positions_file);
                            SublatticeOperator sub_op_2(sub_j, unit_cell_size, num_sites, spin_length, op_type_2, Q, positions_file);
                            obs_1_out.push_back(Operator(sub_op_1));
                            obs_2_out.push_back(Operator(sub_op_2));
                        }
                        
                        names_out.push_back(name_ss.str());
                    }
                }
                
            } else if (operator_type == "experimental") {
                // Experimental operators: cos(θ)Sz + sin(θ)Sx (only one per momentum point, independent of spin combo)
                if (combo.first == spin_combinations[0].first && combo.second == spin_combinations[0].second) {
                    std::stringstream name_ss;
                    name_ss << "Experimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_theta" << theta;
                    
                    if (use_fixed_sz) {
                        FixedSzExperimentalOperator exp_op_1(num_sites, spin_length, n_up, theta, Q, positions_file);
                        FixedSzExperimentalOperator exp_op_2(num_sites, spin_length, n_up, theta, Q, positions_file);
                        obs_1_out.push_back(Operator(exp_op_1));
                        obs_2_out.push_back(Operator(exp_op_2));
                    } else {
                        ExperimentalOperator exp_op_1(num_sites, spin_length, theta, Q, positions_file);
                        ExperimentalOperator exp_op_2(num_sites, spin_length, theta, Q, positions_file);
                        obs_1_out.push_back(Operator(exp_op_1));
                        obs_2_out.push_back(Operator(exp_op_2));
                    }
                    
                    names_out.push_back(name_ss.str());
                }
                
            } else if (operator_type == "transverse_experimental") {
                // Transverse experimental with SF/NSF separation (only one per momentum point)
                if (combo.first == spin_combinations[0].first && combo.second == spin_combinations[0].second) {
                    std::vector<double> e1_vec = {transverse_basis_1[0], transverse_basis_1[1], transverse_basis_1[2]};
                    std::vector<double> e2_vec = {transverse_basis_2[0], transverse_basis_2[1], transverse_basis_2[2]};
                    
                    std::stringstream name_sf, name_nsf;
                    name_sf << "TransverseExperimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] 
                            << "_theta" << theta << "_NSF";
                    name_nsf << "TransverseExperimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] 
                             << "_theta" << theta << "_SF";
                    
                    if (use_fixed_sz) {
                        FixedSzTransverseExperimentalOperator op1_sf(num_sites, spin_length, n_up, theta, Q, e1_vec, positions_file);
                        FixedSzTransverseExperimentalOperator op2_sf(num_sites, spin_length, n_up, theta, Q, e1_vec, positions_file);
                        FixedSzTransverseExperimentalOperator op1_nsf(num_sites, spin_length, n_up, theta, Q, e2_vec, positions_file);
                        FixedSzTransverseExperimentalOperator op2_nsf(num_sites, spin_length, n_up, theta, Q, e2_vec, positions_file);
                        
                        obs_1_out.push_back(Operator(op1_sf));
                        obs_2_out.push_back(Operator(op2_sf));
                        obs_1_out.push_back(Operator(op1_nsf));
                        obs_2_out.push_back(Operator(op2_nsf));
                    } else {
                        TransverseExperimentalOperator op1_sf(num_sites, spin_length, theta, Q, e1_vec, positions_file);
                        TransverseExperimentalOperator op2_sf(num_sites, spin_length, theta, Q, e1_vec, positions_file);
                        TransverseExperimentalOperator op1_nsf(num_sites, spin_length, theta, Q, e2_vec, positions_file);
                        TransverseExperimentalOperator op2_nsf(num_sites, spin_length, theta, Q, e2_vec, positions_file);
                        
                        obs_1_out.push_back(Operator(op1_sf));
                        obs_2_out.push_back(Operator(op2_sf));
                        obs_1_out.push_back(Operator(op1_nsf));
                        obs_2_out.push_back(Operator(op2_nsf));
                    }
                    
                    names_out.push_back(name_sf.str());
                    names_out.push_back(name_nsf.str());
                }
            }
        }
    }
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
 * @brief Run symmetrized diagonalization workflow
 */
EDResults run_symmetrized_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    // Check if fixed Sz mode is enabled
    if (config.system.use_fixed_sz) {
        int64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        results = exact_diagonalization_fixed_sz_symmetrized(
            config.system.hamiltonian_dir,
            n_up,
            config.method,
            params,
            HamiltonianFileFormat::STANDARD
        );
    } else {
        results = exact_diagonalization_from_directory_symmetrized(
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
 * @brief Run streaming symmetry diagonalization workflow
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
            params
        );
    } else {
        results = exact_diagonalization_streaming_symmetry(
            config.system.hamiltonian_dir,
            config.method,
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
    
    // Eigenvalues are saved to HDF5 by the underlying diagonalization functions
    
    return results;
}

/**
 * @brief Run disk-based streaming symmetry diagonalization workflow
 * 
 * This ultra-low-memory mode processes sectors one at a time,
 * storing sector data on disk. Suitable for very large Hilbert spaces
 * (>64M states) where standard streaming would OOM.
 */
EDResults run_disk_streaming_workflow(const EDConfig& config) {
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    create_directory_mpi_safe(params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results = exact_diagonalization_disk_streaming(
        config.system.hamiltonian_dir,
        config.method,
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
            std::cout << "Loading three-body terms from: " << three_body_file << "\n";
            ham.loadThreeBodyTerm(three_body_file);
        }
    }
    
    // Hilbert space dimension
    uint64_t N = 1ULL << config.system.num_sites;
    
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
    
    std::cout << "Random states: " << params.num_samples << "\n";
    std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
    std::cout << "Temperature range: [" << config.dynamical.temp_min << ", " << config.dynamical.temp_max << "]\n";
    std::cout << "Temperature bins: " << config.dynamical.num_temp_bins << "\n";
    
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
        if (size > 1) {
            // MPI tags for communication
            const int TASK_TAG = 1;
            const int DONE_TAG = 2;
            const int STOP_TAG = 3;
            
            if (rank == 0) {
                // Master: distribute tasks dynamically
                int next_task = 0;
                
                // Send initial tasks to all workers
                for (int r = 1; r < size && next_task < num_tasks; r++) {
                    MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                    next_task++;
                }
                
                // Process tasks on rank 0 while managing other workers
                int completed = 0;
                while (completed < num_tasks) {
                    // Check if rank 0 can grab a task
                    if (next_task < num_tasks) {
                        int my_task = next_task;
                        next_task++;
                        
                        const auto& task = all_tasks[my_task];
                        if (task.is_multi_temp) {
                            std::cout << "Rank 0 processing task " << (my_task + 1) << "/" << num_tasks
                                      << " (op=" << names[task.op_idx] << ", ALL temperatures)\n";
                            if (process_operator_all_temps(task.op_idx)) {
                                local_processed_count++;
                            }
                        } else {
                            std::cout << "Rank 0 processing task " << (my_task + 1) << "/" << num_tasks
                                      << " (T=" << temperatures[task.temp_idx]
                                      << ", op=" << names[task.op_idx] << ")\n";
                            if (process_task_single(task)) {
                                local_processed_count++;
                            }
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
                    if (task.is_multi_temp) {
                        std::cout << "Rank " << rank << " processing task " << (task_id + 1) << "/" << num_tasks
                                  << " (op=" << names[task.op_idx] << ", ALL temperatures)\n";
                        if (process_operator_all_temps(task.op_idx)) {
                            local_processed_count++;
                        }
                    } else {
                        std::cout << "Rank " << rank << " processing task " << (task_id + 1) << "/" << num_tasks
                                  << " (T=" << temperatures[task.temp_idx]
                                  << ", op=" << names[task.op_idx] << ")\n";
                        if (process_task_single(task)) {
                            local_processed_count++;
                        }
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
        std::cout << "\nUsing legacy file-based operator loading\n";
        
        if (config.dynamical.operator_file.empty()) {
            std::cerr << "Error: --dyn-operator=<file> is required for dynamical response\n";
            return;
        }
        
        std::string op_path = config.system.hamiltonian_dir + "/" + config.dynamical.operator_file;
        Operator op(config.system.num_sites, config.system.spin_length);
        op.loadFromInterAllFile(op_path);
        
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
            std::cout << "Results saved to HDF5: " << h5_file << " (" << op_name << ")\n";
        }
    }
    
    std::cout << "\nDynamical response complete.\n";
    std::cout << "Frequency range: [" << config.dynamical.omega_min << ", " << config.dynamical.omega_max << "]\n";
    std::cout << "Number of points: " << config.dynamical.num_omega_points << "\n";
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
            std::cout << "Loading three-body terms from: " << three_body_file << "\n";
            ham.loadThreeBodyTerm(three_body_file);
        }
    }
    
    // Hilbert space dimension
    uint64_t N = 1ULL << config.system.num_sites;
    
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
    
    std::cout << "Random states: " << params.num_samples << "\n";
    std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
    std::cout << "Temperature range: [" << config.static_resp.temp_min << ", " << config.static_resp.temp_max << "]\n";
    
    if (use_config_operators) {
        // ============================================================
        // Configuration-based operator construction (like TPQ_DSSF)
        // ============================================================
        std::cout << "\nUsing configuration-based operator construction\n";
        std::cout << "  Operator type: " << config.static_resp.operator_type << "\n";
        std::cout << "  Basis: " << config.static_resp.basis << "\n";
        std::cout << "  Spin combinations: " << config.static_resp.spin_combinations << "\n";
        
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
        // MPI Task Distribution (like TPQ_DSSF.cpp)
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
                for (int r = 1; r < size && next_task < num_tasks; r++) {
                    MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                    next_task++;
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
        std::cout << "\nUsing legacy file-based operator loading\n";
        
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
    std::cout << "  # Symmetry-exploiting calculation (auto-selects best mode)\n";
    std::cout << "  " << prog_name << " ./data --symm --eigenvalues=10\n\n";
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
    std::cout << "  --symm                  Run symmetry-exploiting diagonalization (auto-selects best mode)\n";
    std::cout << "  --symm-threshold=<n>    Hilbert dim threshold for streaming mode (default: 4096)\n";
    std::cout << "  --disk-threshold=<n>    Hilbert dim threshold for disk-streaming mode (default: 67108864)\n";
    std::cout << "  --symmetrized           Run symmetrized diagonalization (exploits symmetries)\n";
    std::cout << "  --streaming-symmetry    Run streaming symmetry diagonalization (memory-efficient,\n";
    std::cout << "                          recommended for systems ≥12 sites)\n";
    std::cout << "  --disk-streaming        Run ultra-low-memory disk-based symmetry diagonalization\n";
    std::cout << "                          (processes one sector at a time, uses disk cache)\n";
    std::cout << "  --thermo                Compute thermodynamic properties\n";
    std::cout << "  --dynamical-response    Compute dynamical response (spectral functions)\n";
    std::cout << "  --static-response       Compute static response (thermal expectation values)\n";
    std::cout << "  --ground-state-dssf     Compute T=0 DSSF using continued fraction (optimal for 32-site ED)\n\n";
    
    std::cout << "TPQ Observable Options:\n";
    std::cout << "  --save-thermal-states   Save TPQ states at target temperatures (for TPQ_DSSF post-processing)\n";
    std::cout << "  --compute-spin-correlations  Compute ⟨Si⟩ and ⟨Si·Sj⟩ correlations during TPQ evolution\n";
    std::cout << "  --calc_observables      (deprecated) Alias for --save-thermal-states\n";
    std::cout << "  --measure_spin          (deprecated) Alias for --compute-spin-correlations\n\n";
    
    std::cout << "Thermal Options (for mTPQ/cTPQ/FULL):\n";
    std::cout << "  --samples=<n>           Number of TPQ samples\n";
    std::cout << "  --temp_min=<T>          Minimum temperature\n";
    std::cout << "  --temp_max=<T>          Maximum temperature\n";
    std::cout << "  --temp_bins=<n>         Number of temperature bins\n";
    std::cout << "  --continue_quenching    Continue TPQ from saved state (requires prior run)\n";
    std::cout << "  --continue_sample=<n>   Sample to continue from (0 = auto-detect lowest energy)\n";
    std::cout << "  --continue_beta=<β>     Beta to continue from (0.0 = use saved beta)\n\n";
    
    std::cout << "Dynamical Response Options:\n";
    std::cout << "  --dyn-thermal           Use thermal averaging (multiple random states)\n";
    std::cout << "  --dyn-samples=<n>       Number of random states (default: 20)\n";
    std::cout << "  --dyn-krylov=<n>        Krylov dimension per sample (default: 100)\n";
    std::cout << "  --dyn-omega-min=<ω>     Minimum frequency (default: -10)\n";
    std::cout << "  --dyn-omega-max=<ω>     Maximum frequency (default: 10)\n";
    std::cout << "  --dyn-omega-points=<n>  Number of frequency points (default: 1000)\n";
    std::cout << "  --dyn-broadening=<η>    Lorentzian broadening (default: 0.1)\n";
    std::cout << "  --dyn-temp-min=<T>      Minimum temperature (default: 0.001)\n";
    std::cout << "  --dyn-temp-max=<T>      Maximum temperature (default: 1.0)\n";
    std::cout << "  --dyn-temp-bins=<n>     Number of temperature points (default: 50)\n";
    std::cout << "  --dyn-correlation       Compute two-operator dynamical correlation\n";
    std::cout << "  --dyn-operator=<file>   Operator file to probe (legacy mode)\n";
    std::cout << "  --dyn-operator2=<file>  Second operator for correlation (legacy mode)\n";
    std::cout << "  --dyn-output=<prefix>   Output file prefix (default: dynamical_response)\n";
    std::cout << "  --dyn-seed=<n>          Random seed (0 = auto)\n\n";
    
    std::cout << "Dynamical Response Operator Configuration (like TPQ_DSSF):\n";
    std::cout << "  --dyn-operator-type=<type>     Operator type: sum | transverse | sublattice |\n";
    std::cout << "                                 experimental | transverse_experimental (default: sum)\n";
    std::cout << "  --dyn-basis=<basis>            Basis: ladder (Sp,Sm,Sz) | xyz (Sx,Sy,Sz) (default: ladder)\n";
    std::cout << "  --dyn-spin-combinations=<str>  Spin operators: \"op1,op2;op3,op4\" (default: \"0,0;2,2\")\n";
    std::cout << "                                 For ladder: 0=Sp, 1=Sm, 2=Sz\n";
    std::cout << "                                 For xyz: 0=Sx, 1=Sy, 2=Sz\n";
    std::cout << "  --dyn-unit-cell-size=<n>       Unit cell size for sublattice operators (default: 1)\n";
    std::cout << "  --dyn-momentum-points=<str>    Q-points: \"Qx,Qy,Qz;...\" (default: \"0,0,0;0,0,2\")\n";
    std::cout << "                                 Values are multiples of π\n";
    std::cout << "  --dyn-polarization=<str>       Polarization vector: \"ex,ey,ez\" (default: auto)\n";
    std::cout << "  --dyn-theta=<θ>                Rotation angle for experimental operators (radians)\n\n";
    
    std::cout << "GPU Acceleration Options:\n";
    std::cout << "  --use-gpu               Enable GPU acceleration for both dynamical and static response\n";
    std::cout << "  --dyn-use-gpu           Enable GPU acceleration for dynamical response only\n";
    std::cout << "  --static-use-gpu        Enable GPU acceleration for static response only\n";
    std::cout << "                          Note: Fixed-Sz GPU support is not yet implemented\n\n";
    
    std::cout << "Static Response Options:\n";
    std::cout << "  --static-samples=<n>    Number of random states (default: 20)\n";
    std::cout << "  --static-krylov=<n>     Krylov dimension per sample (default: 400)\n";
    std::cout << "  --static-temp-min=<T>   Minimum temperature (default: 0.001)\n";
    std::cout << "  --static-temp-max=<T>   Maximum temperature (default: 1.0)\n";
    std::cout << "  --static-temp-points=<n> Number of temperature points (default: 100)\n";
    std::cout << "  --static-no-susceptibility  Don't compute susceptibility\n";
    std::cout << "  --static-correlation    Compute two-operator correlation\n";
    std::cout << "  --static-expectation    Compute single-operator <O> (implies --static-response)\n";
    std::cout << "  --static-operator=<file>    Operator file to probe (legacy mode)\n";
    std::cout << "  --static-operator2=<file>   Second operator for correlation (legacy mode)\n";
    std::cout << "  --static-output=<prefix>    Output file prefix (default: static_response)\n";
    std::cout << "  --static-seed=<n>       Random seed (0 = auto)\n\n";
    
    std::cout << "Static Response Operator Configuration (like TPQ_DSSF):\n";
    std::cout << "  --static-operator-type=<type>     Operator type: sum | transverse | sublattice |\n";
    std::cout << "                                    experimental | transverse_experimental (default: sum)\n";
    std::cout << "  --static-basis=<basis>            Basis: ladder (Sp,Sm,Sz) | xyz (Sx,Sy,Sz) (default: ladder)\n";
    std::cout << "  --static-spin-combinations=<str>  Spin operators: \"op1,op2;op3,op4\" (default: \"0,0;2,2\")\n";
    std::cout << "                                    For ladder: 0=Sp, 1=Sm, 2=Sz\n";
    std::cout << "                                    For xyz: 0=Sx, 1=Sy, 2=Sz\n";
    std::cout << "  --static-unit-cell-size=<n>       Unit cell size for sublattice operators (default: 1)\n";
    std::cout << "  --static-momentum-points=<str>    Q-points: \"Qx,Qy,Qz;...\" (default: \"0,0,0;0,0,2\")\n";
    std::cout << "                                    Values are multiples of π\n";
    std::cout << "  --static-polarization=<str>       Polarization vector: \"ex,ey,ez\" (default: auto)\n";
    std::cout << "  --static-theta=<θ>                Rotation angle for experimental operators (radians)\n\n";
    
    std::cout << "Ground State DSSF Options (T=0 Dynamical Correlations):\n";
    std::cout << "  --ground-state-dssf     Compute T=0 DSSF using continued fraction method\n";
    std::cout << "                          Uses same operator options as --dynamical-response\n";
    std::cout << "                          Optimal for 32-site ED with fixed-Sz sector\n";
    std::cout << "                          (Only needs 2-3 Lanczos vectors instead of full spectrum)\n\n";
    
    std::cout << "Fixed-Sz Sector Options:\n";
    std::cout << "  --fixed-sz              Use fixed-Sz sector (reduced Hilbert space)\n";
    std::cout << "  --n-up=<n>              Number of up spins (determines Sz sector, default: N/2)\n\n";
    
    std::cout << "MPI Options:\n";
    std::cout << "  Use mpirun/mpiexec to run with multiple ranks:\n";
    std::cout << "    mpirun -np <N> " << prog_name << " <directory> [options]\n";
    std::cout << "  MPI parallelization automatically applies to:\n";
    std::cout << "    - TPQ samples (mTPQ, cTPQ)\n";
    std::cout << "    - Dynamical response (temperature × operator tasks)\n";
    std::cout << "    - Static response (operator tasks)\n\n";
    
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
    std::cout << "    DAVIDSON_GPU           GPU-accelerated Davidson method (recommended)\n";
    std::cout << "    LOBPCG_GPU             [DEPRECATED] Redirects to DAVIDSON_GPU\n";
    std::cout << "    FTLM_GPU               GPU-accelerated Finite Temperature Lanczos\n";
    std::cout << "    FTLM_GPU_FIXED_SZ      GPU FTLM for fixed Sz sector\n";
    std::cout << "    mTPQ_GPU               GPU-accelerated microcanonical TPQ\n";
    std::cout << "    cTPQ_GPU               GPU-accelerated canonical TPQ\n\n";
    
    std::cout << "For detailed parameters of any method, use:\n";
    std::cout << "  " << prog_name << " --method-info=<METHOD_NAME>\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog_name << " --method-info=LANCZOS\n";
    std::cout << "  " << prog_name << " --method-info=LOBPCG\n";
    std::cout << "  " << prog_name << " --method-info=mTPQ\n";
    std::cout << "  " << prog_name << " --method-info=DAVIDSON_GPU\n\n";
    
    std::cout << "For more options, see documentation or generated config file.\n\n";
    
    std::cout << "================================================================================\n";
    std::cout << "DSSF MODE (TPQ_DSSF-style interface)\n";
    std::cout << "================================================================================\n";
    std::cout << "For spectral function calculations with simpler command-line interface:\n\n";
    std::cout << "  " << prog_name << " --dssf <directory> <krylov_dim> <spin_combinations> [options]\n\n";
    std::cout << "  Required arguments:\n";
    std::cout << "    <directory>          Path containing InterAll.dat, Trans.dat, positions.dat\n";
    std::cout << "    <krylov_dim>         Krylov subspace dimension (30-100 typical)\n";
    std::cout << "    <spin_combinations>  \"op1,op2;op3,op4\" (0=Sp/Sx, 1=Sm/Sy, 2=Sz)\n\n";
    std::cout << "  DSSF-specific options:\n";
    std::cout << "    --dssf-method=<m>    Method: spectral | ftlm_thermal | static | ground_state\n";
    std::cout << "    --dssf-operator=<o>  Operator: sum | transverse | sublattice | experimental\n";
    std::cout << "    --dssf-basis=<b>     Basis: ladder | xyz (default: ladder)\n";
    std::cout << "    --dssf-omega=<min,max,bins,eta>  Frequency grid and broadening\n";
    std::cout << "    --dssf-temps=<min,max,steps>     Temperature range (log spacing)\n";
    std::cout << "    --dssf-momentum=<Qx,Qy,Qz;...>   Momentum points (in units of π)\n";
    std::cout << "    --dssf-samples=<n>   Number of FTLM random samples (default: 40)\n\n";
    std::cout << "  Examples:\n";
    std::cout << "    # SzSz spectral function at Q=0\n";
    std::cout << "    " << prog_name << " --dssf ./data 50 \"2,2\" --dssf-method=spectral\n\n";
    std::cout << "    # Finite-T DSSF with FTLM averaging\n";
    std::cout << "    " << prog_name << " --dssf ./data 50 \"2,2\" --dssf-method=ftlm_thermal \\\n";
    std::cout << "                   --dssf-temps=0.1,10.0,20\n\n";
    std::cout << "    # Static structure factor (SSSF)\n";
    std::cout << "    " << prog_name << " --dssf ./data 50 \"2,2\" --dssf-method=static\n\n";
    std::cout << "  Note: DSSF mode uses TPQ states from ed_results.h5 if available.\n";
    std::cout << "        Run diagonalization/mTPQ first, then use --dssf for post-processing.\n";
}

// ============================================================================
// DSSF MODE (TPQ_DSSF-style interface)
// ============================================================================

/**
 * @brief Run DSSF mode with TPQ_DSSF-style arguments
 * 
 * This provides a simpler interface for spectral function calculations,
 * using positional arguments like TPQ_DSSF.
 */
int run_dssf_mode(int argc, char* argv[]) {
    int rank = 0, size = 1;
    #ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif
    
    // Find positional arguments after --dssf
    int dssf_idx = -1;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dssf") {
            dssf_idx = i;
            break;
        }
    }
    
    if (dssf_idx < 0 || dssf_idx + 3 >= argc) {
        if (rank == 0) {
            std::cerr << "DSSF mode requires: --dssf <directory> <krylov_dim> <spin_combinations>\n";
            std::cerr << "Use --help for more information.\n";
        }
        return 1;
    }
    
    // Parse positional arguments
    std::string directory = argv[dssf_idx + 1];
    int krylov_dim = std::stoi(argv[dssf_idx + 2]);
    std::string spin_combinations_str = argv[dssf_idx + 3];
    
    // Parse optional arguments
    std::string method = "spectral";
    std::string operator_type = "sum";
    std::string basis = "ladder";
    double omega_min = -5.0, omega_max = 5.0;
    int num_omega_bins = 200;
    double broadening = 0.1;
    double T_min = 0.1, T_max = 10.0;
    int T_steps = 20;
    bool use_temperature_scan = false;
    std::string momentum_str = "0,0,0";
    std::string polarization_str = "";
    double theta = 0.0;
    int num_samples = 40;
    int n_up = -1;
    bool use_fixed_sz = false;
    int unit_cell_size = 4;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--dssf-method=") == 0) {
            method = arg.substr(14);
        } else if (arg.find("--dssf-operator=") == 0) {
            operator_type = arg.substr(16);
        } else if (arg.find("--dssf-basis=") == 0) {
            basis = arg.substr(13);
        } else if (arg.find("--dssf-omega=") == 0) {
            std::string omega_str = arg.substr(13);
            std::stringstream ss(omega_str);
            std::string val;
            std::vector<double> vals;
            while (std::getline(ss, val, ',')) {
                vals.push_back(std::stod(val));
            }
            if (vals.size() >= 1) omega_min = vals[0];
            if (vals.size() >= 2) omega_max = vals[1];
            if (vals.size() >= 3) num_omega_bins = static_cast<int>(vals[2]);
            if (vals.size() >= 4) broadening = vals[3];
        } else if (arg.find("--dssf-temps=") == 0) {
            std::string temps_str = arg.substr(13);
            std::stringstream ss(temps_str);
            std::string val;
            std::vector<double> vals;
            while (std::getline(ss, val, ',')) {
                vals.push_back(std::stod(val));
            }
            if (vals.size() >= 1) T_min = vals[0];
            if (vals.size() >= 2) T_max = vals[1];
            if (vals.size() >= 3) T_steps = static_cast<int>(vals[2]);
            use_temperature_scan = true;
        } else if (arg.find("--dssf-momentum=") == 0) {
            momentum_str = arg.substr(16);
        } else if (arg.find("--dssf-polarization=") == 0) {
            polarization_str = arg.substr(20);
        } else if (arg.find("--dssf-theta=") == 0) {
            theta = std::stod(arg.substr(13));
        } else if (arg.find("--dssf-samples=") == 0) {
            num_samples = std::stoi(arg.substr(15));
        } else if (arg.find("--dssf-n-up=") == 0 || arg.find("--n-up=") == 0) {
            std::string val = (arg.find("--dssf-n-up=") == 0) ? arg.substr(12) : arg.substr(7);
            n_up = std::stoi(val);
            use_fixed_sz = (n_up >= 0);
        } else if (arg == "--fixed-sz") {
            use_fixed_sz = true;
        } else if (arg.find("--dssf-unit-cell=") == 0) {
            unit_cell_size = std::stoi(arg.substr(17));
        } else if (arg.find("--output=") == 0) {
            // output directory can be specified
        }
    }
    
    // Read num_sites from positions.dat
    std::string positions_file = directory + "/positions.dat";
    int num_sites = 0;
    {
        std::ifstream file(positions_file);
        if (!file.is_open()) {
            if (rank == 0) {
                std::cerr << "Error: Cannot open positions.dat at " << positions_file << "\n";
            }
            return 1;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') num_sites++;
        }
    }
    
    float spin_length = 0.5f;
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "ED DSSF MODE (TPQ_DSSF-style interface)\n";
        std::cout << "==========================================\n";
        std::cout << "Directory: " << directory << "\n";
        std::cout << "Sites: " << num_sites << ", Spin: " << spin_length << "\n";
        std::cout << "Krylov dimension: " << krylov_dim << "\n";
        std::cout << "Spin combinations: " << spin_combinations_str << "\n";
        std::cout << "Method: " << method << "\n";
        std::cout << "Operator type: " << operator_type << "\n";
        std::cout << "Basis: " << basis << "\n";
        if (method != "static") {
            std::cout << "Omega range: [" << omega_min << ", " << omega_max << "]\n";
            std::cout << "Omega bins: " << num_omega_bins << "\n";
            std::cout << "Broadening: " << broadening << "\n";
        }
        if (use_temperature_scan) {
            std::cout << "Temperature range: [" << T_min << ", " << T_max << "]\n";
            std::cout << "Temperature steps: " << T_steps << "\n";
        }
        if (use_fixed_sz) {
            std::cout << "Fixed-Sz sector: n_up = " << (n_up >= 0 ? std::to_string(n_up) : "N/2") << "\n";
        }
    }
    
    // Load Hamiltonian
    Operator ham_op(num_sites, spin_length);
    std::string interall_file = directory + "/InterAll.dat";
    std::string trans_file = directory + "/Trans.dat";
    std::string three_body_file = directory + "/ThreeBodyG.dat";
    
    if (!std::filesystem::exists(interall_file) || !std::filesystem::exists(trans_file)) {
        if (rank == 0) {
            std::cerr << "Error: Missing Hamiltonian files (InterAll.dat, Trans.dat)\n";
        }
        return 1;
    }
    
    ham_op.loadFromInterAllFile(interall_file);
    ham_op.loadFromFile(trans_file);
    if (std::filesystem::exists(three_body_file)) {
        ham_op.loadThreeBodyTerm(three_body_file);
    }
    
    // Determine Hilbert space dimension
    if (n_up < 0 && use_fixed_sz) {
        n_up = num_sites / 2;
    }
    
    uint64_t N;
    if (use_fixed_sz) {
        N = 1;
        for (uint64_t i = 0; i < static_cast<uint64_t>(n_up); i++) {
            N = N * (num_sites - i) / (i + 1);
        }
    } else {
        N = 1ULL << num_sites;
    }
    
    if (rank == 0) {
        std::cout << "Hilbert space dimension: " << N << "\n";
    }
    
    // Create Hamiltonian function wrapper
    auto H = [&ham_op](const Complex* in, Complex* out, int size) {
        ham_op.apply(in, out, size);
    };
    
    // Read ground state energy for energy shift
    double ground_state_energy = 0.0;
    std::string h5_file = directory + "/output/ed_results.h5";
    if (HDF5IO::fileExists(h5_file)) {
        try {
            auto eigenvalues = HDF5IO::loadEigenvalues(h5_file);
            if (!eigenvalues.empty()) {
                ground_state_energy = eigenvalues[0];
                if (rank == 0) {
                    std::cout << "Ground state energy: " << ground_state_energy << "\n";
                }
            }
        } catch (...) {}
    }
    
    // Parse operators
    auto spin_combinations = parse_spin_combinations(spin_combinations_str);
    auto momentum_points = parse_momentum_points(momentum_str);
    auto polarization = polarization_str.empty() ? 
        std::vector<double>{1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0} :
        parse_polarization(polarization_str);
    
    // Construct operators
    std::vector<Operator> obs_1, obs_2;
    std::vector<std::string> obs_names;
    
    construct_operators_from_config(
        operator_type, basis, spin_combinations, momentum_points,
        polarization, theta, unit_cell_size, num_sites, spin_length,
        use_fixed_sz, n_up, positions_file,
        obs_1, obs_2, obs_names
    );
    
    if (rank == 0) {
        std::cout << "Number of operator pairs: " << obs_1.size() << "\n";
    }
    
    // Create output directory
    std::string output_dir = directory + "/output/dssf_" + method;
    create_directory_mpi_safe(output_dir);
    
    // Generate temperature grid if needed
    std::vector<double> temperatures;
    if (use_temperature_scan || method == "ftlm_thermal" || method == "static") {
        double log_T_min = std::log(T_min);
        double log_T_max = std::log(T_max);
        double log_step = (log_T_max - log_T_min) / std::max(1, T_steps - 1);
        for (int i = 0; i < T_steps; i++) {
            temperatures.push_back(std::exp(log_T_min + i * log_step));
        }
    }
    
    // Execute the requested method
    if (method == "spectral") {
        // Single-state spectral function
        // Load ground state or TPQ state
        ComplexVector state;
        bool loaded = load_ground_state_from_file(directory + "/output", state, ground_state_energy, N);
        
        if (!loaded) {
            if (rank == 0) {
                std::cerr << "Error: Could not load state for spectral method\n";
                std::cerr << "Run diagonalization first with --eigenvectors\n";
            }
            return 1;
        }
        
        DynamicalResponseParameters params;
        params.krylov_dim = krylov_dim;
        params.broadening = broadening;
        params.tolerance = 1e-10;
        params.full_reorthogonalization = true;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results = compute_dynamical_correlation_state(
                H, O1, O2, state, N, params,
                omega_min, omega_max, num_omega_bins, 0.0, ground_state_energy
            );
            
            std::string filename = output_dir + "/" + obs_names[i] + "_spectral.txt";
            save_dynamical_response_results(results, filename);
            
            if (rank == 0) {
                std::cout << "Saved: " << filename << "\n";
            }
        }
        
    } else if (method == "ftlm_thermal") {
        // TRUE FTLM thermal averaging
        DynamicalResponseParameters params;
        params.krylov_dim = krylov_dim;
        params.broadening = broadening;
        params.tolerance = 1e-10;
        params.full_reorthogonalization = true;
        params.num_samples = num_samples;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results_map = compute_dynamical_correlation_multi_sample_multi_temperature(
                H, O1, O2, N, params,
                omega_min, omega_max, num_omega_bins,
                temperatures, ground_state_energy, output_dir
            );
            
            for (const auto& [T, results] : results_map) {
                std::stringstream ss;
                ss << output_dir << "/" << obs_names[i] << "_ftlm_T" << std::fixed << std::setprecision(4) << T << ".txt";
                save_dynamical_response_results(results, ss.str());
                
                if (rank == 0) {
                    std::cout << "Saved: " << ss.str() << "\n";
                }
            }
        }
        
    } else if (method == "static") {
        // Static structure factor
        StaticResponseParameters params;
        params.krylov_dim = krylov_dim;
        params.tolerance = 1e-10;
        params.full_reorthogonalization = true;
        params.num_samples = num_samples;
        params.compute_error_bars = true;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results = compute_static_response(
                H, O1, O2, N, params,
                T_min, T_max, T_steps, output_dir
            );
            
            std::string filename = output_dir + "/" + obs_names[i] + "_static.txt";
            save_static_response_results(results, filename);
            
            if (rank == 0) {
                std::cout << "Saved: " << filename << "\n";
            }
        }
        
    } else if (method == "ground_state") {
        // Ground state DSSF using continued fraction
        ComplexVector ground_state;
        bool loaded = load_ground_state_from_file(directory + "/output", ground_state, ground_state_energy, N);
        
        if (!loaded) {
            if (rank == 0) {
                std::cerr << "Error: Could not load ground state for ground_state method\n";
            }
            return 1;
        }
        
        GroundStateDSSFParameters gs_params;
        gs_params.krylov_dim = krylov_dim;
        gs_params.omega_min = omega_min;
        gs_params.omega_max = omega_max;
        gs_params.num_omega_points = num_omega_bins;
        gs_params.broadening = broadening;
        gs_params.tolerance = 1e-10;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results = compute_ground_state_cross_correlation(
                H, O1, O2, ground_state, ground_state_energy, N, gs_params
            );
            
            std::string filename = output_dir + "/" + obs_names[i] + "_ground_state_dssf.txt";
            save_dynamical_response_results(results, filename);
            
            if (rank == 0) {
                std::cout << "Saved: " << filename << "\n";
            }
        }
        
    } else {
        if (rank == 0) {
            std::cerr << "Unknown DSSF method: " << method << "\n";
            std::cerr << "Available methods: spectral, ftlm_thermal, static, ground_state\n";
        }
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "\nDSSF calculation complete.\n";
        std::cout << "Results saved in: " << output_dir << "\n";
    }
    
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    #ifdef WITH_MPI
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0 && size > 1) {
        std::cout << "ED: MPI enabled (" << size << " ranks)\n";
    }
    #endif
    
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
    
    // Check for DSSF mode (TPQ_DSSF-style interface)
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dssf") {
            int result = run_dssf_mode(argc, argv);
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return result;
        }
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
    create_directory_mpi_safe(config.workflow.output_dir);
    
    // Execute workflows
    EDResults standard_results, sym_results;
    
    try {
        // Handle unified --symm flag: auto-select between symmetrized, streaming-symmetry, or disk-streaming
        if (config.workflow.run_symm_auto && !config.workflow.skip_ed) {
            // Calculate Hilbert space dimension for threshold decision
            uint64_t hilbert_dim = 1ULL << config.system.num_sites;  // 2^N for spin-1/2
            
            bool use_disk_streaming = (hilbert_dim >= config.workflow.disk_streaming_threshold);
            bool use_streaming = (hilbert_dim >= config.workflow.symm_streaming_threshold);
            
            std::cout << "========================================\n";
            std::cout << "  Auto-Symmetry Mode Selection\n";
            std::cout << "  Hilbert space dimension: " << hilbert_dim << "\n";
            std::cout << "  Threshold for streaming: " << config.workflow.symm_streaming_threshold << "\n";
            std::cout << "  Threshold for disk-streaming: " << config.workflow.disk_streaming_threshold << "\n";
            if (use_disk_streaming) {
                std::cout << "  Selected: disk-streaming (ultra-low-memory)\n";
            } else {
                std::cout << "  Selected: " << (use_streaming ? "streaming-symmetry" : "symmetrized") << "\n";
            }
            std::cout << "========================================\n\n";
            
            if (use_disk_streaming) {
                EDResults disk_results = run_disk_streaming_workflow(config);
                print_eigenvalue_summary(disk_results.eigenvalues);
                
                if (config.workflow.compute_thermo && !disk_results.eigenvalues.empty()) {
                    compute_thermodynamics(disk_results.eigenvalues, config);
                }
            } else if (use_streaming) {
                EDResults streaming_results = run_streaming_symmetry_workflow(config);
                print_eigenvalue_summary(streaming_results.eigenvalues);
                
                if (config.workflow.compute_thermo && !streaming_results.eigenvalues.empty()) {
                    compute_thermodynamics(streaming_results.eigenvalues, config);
                }
            } else {
                sym_results = run_symmetrized_workflow(config);
                print_eigenvalue_summary(sym_results.eigenvalues);
                
                if (config.workflow.compute_thermo && !sym_results.eigenvalues.empty()) {
                    compute_thermodynamics(sym_results.eigenvalues, config);
                }
            }
        }
        
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
        
        if (config.workflow.run_streaming_symmetry && !config.workflow.skip_ed) {
            EDResults streaming_results = run_streaming_symmetry_workflow(config);
            print_eigenvalue_summary(streaming_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !streaming_results.eigenvalues.empty()) {
                compute_thermodynamics(streaming_results.eigenvalues, config);
            }
        }
        
        if (config.workflow.run_disk_streaming && !config.workflow.skip_ed) {
            EDResults disk_results = run_disk_streaming_workflow(config);
            print_eigenvalue_summary(disk_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !disk_results.eigenvalues.empty()) {
                compute_thermodynamics(disk_results.eigenvalues, config);
            }
        }
        
        // Standalone response calculations (don't require prior diagonalization)
        if (config.workflow.compute_dynamical_response) {
            compute_dynamical_response_workflow(config);
        }
        
        if (config.workflow.compute_static_response) {
            compute_static_response_workflow(config);
        }
        
        // Ground state DSSF (T=0 dynamical correlations using continued fraction)
        if (config.workflow.compute_ground_state_dssf) {
            compute_ground_state_dssf_workflow(config);
        }
        
        // Compare results if both were run
        if (config.workflow.run_standard && config.workflow.run_symmetrized) {
            uint64_t n = std::min(standard_results.eigenvalues.size(), sym_results.eigenvalues.size());
            double max_diff = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = std::abs(standard_results.eigenvalues[i] - sym_results.eigenvalues[i]);
                max_diff = std::max(max_diff, diff);
            }
            std::cout << "\nStandard vs Symmetrized max difference: " << max_diff << "\n";
        }
        
        std::cout << "\nComplete. Results: " << config.workflow.output_dir << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        #ifdef WITH_MPI
        MPI_Finalize();
        #endif
        return 1;
    }
    
    #ifdef WITH_MPI
    // Finalize MPI
    MPI_Finalize();
    #endif
    
    return 0;
}
