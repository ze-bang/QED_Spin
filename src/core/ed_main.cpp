#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include "ed_config.h"
#include "ed_config_adapter.h"
#include "ed_wrapper.h"
#include "ed_wrapper_streaming.h"
#include "construct_ham.h"
#include "hdf5_io.h"
#include "../cpu_solvers/ftlm.h"
#include "../cpu_solvers/ltlm.h"
#include "observables.h"

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_CUDA
#include "../gpu/gpu_operator.cuh"
#include "../gpu/gpu_ed_wrapper.h"
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
    
    EDResults results;
    
    // Check if fixed Sz mode is enabled
    if (config.system.use_fixed_sz) {
        int64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        std::cout << "Using Fixed Sz + Symmetrized mode\n";
        std::cout << "  Number of up spins: " << n_up << "\n";
        
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
 * @brief Run streaming symmetry diagonalization workflow
 */
EDResults run_streaming_symmetry_workflow(const EDConfig& config) {
    std::cout << "\n==========================================\n";
    std::cout << "Streaming Symmetry Exact Diagonalization\n";
    std::cout << "==========================================\n";
    
    auto params = ed_adapter::toEDParameters(config);
    params.output_dir = config.workflow.output_dir;
    safe_system_call("mkdir -p " + params.output_dir);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    EDResults results;
    
    // Check if fixed Sz mode is enabled
    if (config.system.use_fixed_sz) {
        int64_t n_up = (config.system.n_up >= 0) ? config.system.n_up : config.system.num_sites / 2;
        
        std::cout << "Using Fixed Sz + Streaming Symmetry mode\n";
        std::cout << "  Number of up spins: " << n_up << "\n";
        
        results = exact_diagonalization_streaming_symmetry_fixed_sz(
            config.system.hamiltonian_dir,
            n_up,
            config.method,
            params
        );
    } else {
        std::cout << "Using Streaming Symmetry mode (full Hilbert space)\n";
        
        results = exact_diagonalization_streaming_symmetry(
            config.system.hamiltonian_dir,
            config.method,
            params
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
    
    // Try to save to HDF5 first
    try {
        std::string hdf5_file = HDF5IO::createOrOpenFile(config.workflow.output_dir);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "energy", thermo_data.energy);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "specific_heat", thermo_data.specific_heat);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "entropy", thermo_data.entropy);
        HDF5IO::saveThermodynamics(hdf5_file, thermo_data.temperatures, "free_energy", thermo_data.free_energy);
        std::cout << "Saved thermodynamic data to HDF5\n";
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to save thermodynamics to HDF5: " << e.what() << std::endl;
        std::cerr << "Falling back to text format..." << std::endl;
    }
    
    // Also save text format for backward compatibility
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
    // For ground state dynamical response with eigenvectors, use the standalone
    // example in examples/dynamical_response_example.cpp
    if (!config.dynamical.thermal_average) {
        if (rank == 0) {
            std::cerr << "Note: Only thermal mode (--dyn-thermal) is currently supported in the integrated pipeline.\n";
            std::cerr << "Setting thermal_average mode automatically.\n";
        }
    }
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "Computing Dynamical Response\n";
        std::cout << "==========================================\n";
        
        // Print GPU status
#ifdef WITH_CUDA
        if (config.dynamical.use_gpu) {
            std::cout << "GPU Acceleration: ENABLED\n";
            if (config.system.use_fixed_sz) {
                std::cout << "  Warning: Fixed-Sz mode detected - GPU will be disabled (not yet supported)\n";
            }
            // Print GPU device info
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            if (device_count > 0) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                std::cout << "  GPU Device: " << prop.name << "\n";
                std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
                std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB\n";
            }
        } else {
            std::cout << "GPU Acceleration: DISABLED (use --dyn-use-gpu to enable)\n";
        }
#else
        std::cout << "GPU Acceleration: NOT AVAILABLE (compiled without CUDA support)\n";
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
    
    std::string output_subdir = config.workflow.output_dir + "/dynamical_response";
    safe_system_call("mkdir -p " + output_subdir);
    
    std::cout << "Random states: " << params.num_samples << "\n";
    std::cout << "Krylov dimension: " << params.krylov_dim << "\n";
    std::cout << "Temperature range: [" << config.dynamical.temp_min << ", " << config.dynamical.temp_max << "]\n";
    std::cout << "Temperature bins: " << config.dynamical.num_temp_bins << "\n";
    
    // Find ground state energy for proper energy shifting
    double ground_state_energy = 0.0;
    bool found_ground_state = false;
    
    if (rank == 0) {
        std::cout << "\n--- Finding ground state energy for spectrum normalization ---\n";
        
        // First, try to read from eigenvalues.dat if it exists
        std::string eigenvalues_file = config.workflow.output_dir + "/eigenvectors/eigenvalues.dat";
        std::cout << "Checking for eigenvalues file: " << eigenvalues_file << std::endl;
        std::ifstream check_file(eigenvalues_file, std::ios::binary);
        
        if (check_file.is_open()) {
            check_file.close();
            try {
                // Read number of eigenvalues
                std::ifstream infile(eigenvalues_file, std::ios::binary);
                size_t num_eigenvalues;
                infile.read(reinterpret_cast<char*>(&num_eigenvalues), sizeof(size_t));
                
                if (num_eigenvalues > 0) {
                    // Read the first eigenvalue (ground state energy)
                    infile.read(reinterpret_cast<char*>(&ground_state_energy), sizeof(double));
                    infile.close();
                    
                    std::cout << "Ground state energy read from eigenvalues.dat: " 
                              << std::fixed << std::setprecision(10) << ground_state_energy << std::endl;
                    found_ground_state = true;
                } else {
                    infile.close();
                }
            } catch (const std::exception& e) {
                std::cout << "Warning: Failed to read eigenvalues.dat: " << e.what() << std::endl;
                std::cout << "Will compute ground state energy using Lanczos...\n";
            }
        }
        
        // If eigenvalues.dat doesn't exist or failed to read, use Lanczos
        if (!found_ground_state) {
            std::cout << "eigenvalues.dat not found. Computing ground state energy using Lanczos...\n";
            ComplexVector ground_state(N);
            ground_state_energy = find_ground_state_lanczos(
                H_func, N, params.krylov_dim, params.tolerance,
                params.full_reorthogonalization, params.reorth_frequency,
                ground_state
            );
            found_ground_state = true;
            std::cout << "Ground state energy from Lanczos: " << ground_state_energy << std::endl;
        }
        
        std::cout << "Dynamical correlations will be shifted to excitation energies (E_gs = 0)\n";
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
        // ============================================================
        // Configuration-based operator construction (like TPQ_DSSF)
        // ============================================================
        std::cout << "\nUsing configuration-based operator construction\n";
        std::cout << "  Operator type: " << config.dynamical.operator_type << "\n";
        std::cout << "  Basis: " << config.dynamical.basis << "\n";
        std::cout << "  Spin combinations: " << config.dynamical.spin_combinations << "\n";
        
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
            std::cout << "Constructed " << obs_1.size() << " operator pair(s)\n";
        }
        
        // ============================================================
        // MPI Task Distribution (like TPQ_DSSF.cpp)
        // ============================================================
        
        // Build task list: each task is (temperature_idx, operator_idx)
        // Decide whether to use optimized multi-temperature workflow
        int num_operators = obs_1.size();
        int num_temps = config.dynamical.num_temp_bins;
        // Optimization now works for ANY number of samples!
        bool use_optimized_multi_temp = (num_temps > 1);
        
        if (rank == 0 && use_optimized_multi_temp) {
            std::cout << "\n========================================\n";
            std::cout << "TEMPERATURE SCAN OPTIMIZATION ENABLED\n";
            std::cout << "========================================\n";
            std::cout << "Samples: " << params.num_samples << std::endl;
            std::cout << "Computing " << num_temps << " temperatures from " << params.num_samples 
                      << " Lanczos runs per operator\n";
            std::cout << "Expected speedup: ~" << (num_temps * 0.9) << "× compared to separate runs\n";
            std::cout << "========================================\n\n";
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
                    // Weight is higher since we're doing all temperatures
                    size_t weight = params.num_samples * params.krylov_dim * num_temps;
                    all_tasks.push_back({0, o, weight, true});  // temp_idx unused for multi-temp
                }
                std::cout << "\nOptimized Mode: " << all_tasks.size() << " tasks = "
                          << num_operators << " operators (each processes all " << num_temps << " temperatures)\n";
            } else {
                // Standard: Create one task per (temperature, operator) pair
                for (int t = 0; t < num_temps; t++) {
                    for (int o = 0; o < num_operators; o++) {
                        // Weight is proportional to samples and krylov dimension
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
                    output_subdir,
                    ground_state_energy
                );
            }
            
            // Save results
            std::string output_file = output_subdir + "/" + names[op_idx];
            if (config.dynamical.num_temp_bins > 1) {
                output_file += "_T" + std::to_string(temperature);
            }
            output_file += ".txt";
            
            save_dynamical_response_results(results, output_file);
            
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
                        output_subdir
                    );
                }
            }
            
            // Save results for all temperatures
            for (const auto& [temperature, results] : results_map) {
                std::string output_file = output_subdir + "/" + names[op_idx];
                if (temperatures.size() > 1) {
                    output_file += "_T" + std::to_string(temperature);
                }
                output_file += ".txt";
                
                save_dynamical_response_results(results, output_file);
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
                    output_subdir,
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
                    output_subdir
                );
            }
            
            // Save results for this temperature
            std::string output_file = output_subdir + "/" + config.dynamical.output_prefix;
            if (config.dynamical.num_temp_bins > 1) {
                output_file += "_T" + std::to_string(temperature);
            }
            output_file += ".txt";
            
            save_dynamical_response_results(results, output_file);
            std::cout << "Results saved to: " << output_file << "\n";
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
        std::cout << "\n==========================================\n";
        std::cout << "Computing Static Response\n";
        std::cout << "==========================================\n";
        
        // Print GPU status
#ifdef WITH_CUDA
        if (config.static_resp.use_gpu) {
            std::cout << "GPU Acceleration: ENABLED\n";
            if (config.system.use_fixed_sz) {
                std::cout << "  Warning: Fixed-Sz mode detected - GPU will be disabled (not yet supported)\n";
            }
            // Print GPU device info
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            if (device_count > 0) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                std::cout << "  GPU Device: " << prop.name << "\n";
                std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
                std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024*1024*1024.0)) << " GB\n";
            }
        } else {
            std::cout << "GPU Acceleration: DISABLED (use --static-use-gpu to enable)\n";
        }
#else
        std::cout << "GPU Acceleration: NOT AVAILABLE (compiled without CUDA support)\n";
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
    
    std::string output_subdir = config.workflow.output_dir + "/static_response";
    safe_system_call("mkdir -p " + output_subdir);
    
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
                    output_subdir
                );
            }
            
            // Save results
            std::string output_file = output_subdir + "/" + names[op_idx] + ".txt";
            save_static_response_results(results, output_file);
            
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
                output_subdir
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
        std::string output_file = output_subdir + "/" + config.static_resp.output_prefix + ".txt";
        save_static_response_results(results, output_file);
        std::cout << "Static response saved to: " << output_file << "\n";
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
    std::cout << "  # Symmetrized calculation\n";
    std::cout << "  " << prog_name << " ./data --symmetrized --eigenvalues=10\n\n";
    std::cout << "  # Streaming symmetry (memory-efficient)\n";
    std::cout << "  " << prog_name << " ./data --streaming-symmetry --eigenvalues=10\n\n";
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
    std::cout << "  --streaming-symmetry    Run streaming symmetry diagonalization (memory-efficient,\n";
    std::cout << "                          recommended for systems ≥12 sites)\n";
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
    
    std::cout << "For more options, see documentation or generated config file.\n";
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
        std::cout << "\n===========================================\n";
        std::cout << "Exact Diagonalization with MPI Support\n";
        std::cout << "===========================================\n";
        std::cout << "Running on " << size << " MPI ranks\n";
        std::cout << "MPI parallelization enabled for TPQ samples\n";
        std::cout << "===========================================\n\n";
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
        
        // Standalone response calculations (don't require prior diagonalization)
        if (config.workflow.compute_dynamical_response) {
            compute_dynamical_response_workflow(config);
        }
        
        if (config.workflow.compute_static_response) {
            compute_static_response_workflow(config);
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
