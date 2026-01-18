#ifndef ED_WRAPPER_CHUNKED_H
#define ED_WRAPPER_CHUNKED_H

#include <ed/core/ed_wrapper.h>
#include <ed/core/chunked_symmetry_builder.h>
#include <ed/core/disk_streaming_symmetry.h>

/**
 * @file ed_wrapper_chunked.h
 * @brief Ultra-low-memory chunked symmetry exact diagonalization
 * 
 * This file provides wrapper functions for exact diagonalization using
 * the chunked symmetry builder that minimizes memory usage during the
 * basis construction phase.
 * 
 * Key advantages over standard streaming symmetry:
 * - Memory usage during basis construction: O(num_orbits) instead of O(2^N)
 * - Can handle systems where standard methods run out of memory
 * - Two-pass algorithm trades speed for memory efficiency
 * - Works with both full Hilbert space and fixed-Sz sectors
 * 
 * Trade-offs:
 * - Slower than cached methods (no orbit lookup cache)
 * - Requires disk space for intermediate files
 * - Better suited for one-off calculations than repeated runs
 */

// ============================================================================
// CHUNKED SYMMETRY ED FUNCTIONS
// ============================================================================

/**
 * @brief Ultra-low-memory symmetry ED using chunked basis construction
 * 
 * This function uses a two-pass algorithm to minimize memory usage:
 * 
 * Pass 1 (Discovery): Iterate through Hilbert space to find unique orbit
 *                     representatives. No caching - O(1) memory per state.
 *                     Total memory: O(num_unique_orbits)
 * 
 * Pass 2 (Build): For each orbit representative, compute full orbit data
 *                 and assign to appropriate sector. Sectors saved to disk.
 *                 Memory: O(sector_size) per sector.
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method
 * @param params Parameters for diagonalization
 * @param interaction_filename Name of interaction file
 * @param single_site_filename Name of single-site file
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_chunked_symmetry(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Ultra-Low-Memory Chunked Symmetry ED" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // ========== Step 2: Build Sectors Using Chunked Algorithm ==========
    std::cout << "\nBuilding symmetry sectors (ultra-low-memory mode)..." << std::endl;
    
    ChunkedSymmetryBuilder builder(params.num_sites, params.spin_length);
    
    // Pass 1: Discover orbit representatives
    builder.discoverOrbits(directory);
    
    // Pass 2: Build and save sectors
    builder.buildSectors(directory);
    
    std::string cache_dir = builder.getCacheDir();
    
    // ========== Step 3: Load Hamiltonian ==========
    std::cout << "\nLoading Hamiltonian for diagonalization..." << std::endl;
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    DiskStreamingSymmetryOperator hamiltonian(params.num_sites, params.spin_length);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    // Load symmetry information (needed for applying Hamiltonian)
    hamiltonian.symmetry_info.loadFromDirectory(directory);
    
    std::cout << "Loaded " << hamiltonian.getTransformData().size() << " Hamiltonian terms" << std::endl;
    
    // Load metadata and set up for diagonalization
    std::ifstream meta_file(cache_dir + "/metadata.txt");
    std::string line;
    size_t num_sectors = 0;
    std::vector<uint64_t> sector_dims;
    
    while (std::getline(meta_file, line)) {
        if (line.find("num_sectors") == 0) {
            sscanf(line.c_str(), "num_sectors %zu", &num_sectors);
            sector_dims.resize(num_sectors);
        } else if (line.find("sector_") == 0) {
            size_t idx;
            uint64_t dim;
            sscanf(line.c_str(), "sector_%zu_dim %lu", &idx, &dim);
            if (idx < num_sectors) {
                sector_dims[idx] = dim;
            }
        }
    }
    meta_file.close();
    
    // Set up hamiltonian with sector info
    hamiltonian.symmetrized_block_ham_sizes.assign(sector_dims.begin(), sector_dims.end());
    
    std::cout << "\nFound " << num_sectors << " symmetry sectors" << std::endl;
    
    // ========== Step 4: Diagonalize Each Sector ==========
    std::cout << "\n========== Diagonalizing Sectors ==========\n" << std::endl;
    
    struct EigenInfo {
        double value;
        uint64_t sector_idx;
        uint64_t eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;
    
    for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
        uint64_t sector_dim = sector_dims[sector_idx];
        
        if (sector_dim == 0) {
            std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                     << ": empty (skipping)" << std::endl;
            continue;
        }
        
        // Load this sector from the chunked cache
        std::string sector_file = cache_dir + "/sector_" + std::to_string(sector_idx) + ".bin";
        SymmetrySector sector = loadSectorFromDisk(sector_file);
        
        std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                  << " (dim=" << sector_dim << ", QN=[";
        for (size_t i = 0; i < sector.quantum_numbers.size(); ++i) {
            std::cout << sector.quantum_numbers[i];
            if (i < sector.quantum_numbers.size() - 1) std::cout << ",";
        }
        std::cout << "])" << std::endl;
        
        // Build lookup table for this sector
        std::unordered_map<uint64_t, size_t> sector_lookup;
        for (size_t i = 0; i < sector.basis_states.size(); ++i) {
            for (uint64_t elem : sector.basis_states[i].orbit_elements) {
                sector_lookup[elem] = i;
            }
        }
        
        // Create matrix-free operator
        const auto& symmetry_info = hamiltonian.symmetry_info;
        const double group_size = static_cast<double>(symmetry_info.max_clique.size());
        const double group_norm = 1.0 / group_size;
        
        auto matvec = [&](const Complex* in, Complex* out, int size) {
            std::fill(out, out + sector_dim, Complex(0.0, 0.0));
            
            const auto& transform_data = hamiltonian.getTransformData();
            
            for (size_t j = 0; j < sector_dim; ++j) {
                Complex c_j = in[j];
                if (std::abs(c_j) < 1e-15) continue;
                
                const auto& state_j = sector.basis_states[j];
                const double norm_j = state_j.norm;
                
                for (size_t orbit_idx = 0; orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                    uint64_t s = state_j.orbit_elements[orbit_idx];
                    Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                    
                    if (std::abs(alpha_s) < 1e-15) continue;
                    
                    Complex prefactor = c_j * alpha_s / norm_j;
                    
                    // Apply Hamiltonian terms
                    for (const auto& tdata : transform_data) {
                        uint64_t s_prime = s;
                        Complex h_element = tdata.coefficient;
                        bool valid = true;
                        
                        // Apply operators (same logic as in DiskStreamingSymmetryOperator)
                        if (!tdata.is_two_body) {
                            if (tdata.op_type == 2) {
                                double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                                h_element *= hamiltonian.getSpin() * sign;
                            } else {
                                uint64_t bit = (s >> tdata.site_index) & 1;
                                if (bit != tdata.op_type) {
                                    s_prime ^= (1ULL << tdata.site_index);
                                } else {
                                    valid = false;
                                }
                            }
                        } else {
                            if (tdata.op_type == 2) {
                                double sign = ((s_prime >> tdata.site_index) & 1) ? -1.0 : 1.0;
                                h_element *= hamiltonian.getSpin() * sign;
                            } else {
                                uint64_t bit = (s_prime >> tdata.site_index) & 1;
                                if (bit != tdata.op_type) {
                                    s_prime ^= (1ULL << tdata.site_index);
                                } else {
                                    valid = false;
                                }
                            }
                            
                            if (valid) {
                                if (tdata.op_type_2 == 2) {
                                    double sign = ((s_prime >> tdata.site_index_2) & 1) ? -1.0 : 1.0;
                                    h_element *= hamiltonian.getSpin() * sign;
                                } else {
                                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                                    if (bit != tdata.op_type_2) {
                                        s_prime ^= (1ULL << tdata.site_index_2);
                                    } else {
                                        valid = false;
                                    }
                                }
                            }
                        }
                        
                        if (valid && std::abs(h_element) > 1e-15) {
                            auto it = sector_lookup.find(s_prime);
                            if (it != sector_lookup.end()) {
                                size_t k = it->second;
                                const auto& state_k = sector.basis_states[k];
                                
                                Complex beta_s_prime(0.0, 0.0);
                                for (size_t idx = 0; idx < state_k.orbit_elements.size(); ++idx) {
                                    if (state_k.orbit_elements[idx] == s_prime) {
                                        beta_s_prime = state_k.orbit_coefficients[idx];
                                        break;
                                    }
                                }
                                
                                out[k] += prefactor * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
                            }
                        }
                    }
                }
            }
        };
        
        // Diagonalize
        EDParameters sector_params = params;
        sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
        
        EDResults sector_results = exact_diagonalization_core(
            matvec, sector_dim, method, sector_params
        );
        
        // Store eigenvalues
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }
        
        std::cout << "  Lowest: " << sector_results.eigenvalues[0] << std::endl;
    }
    
    // Sort and collect results
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    if (all_eigen_info.size() > params.num_eigenvalues) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    EDResults results;
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    // Save results
    if (!params.output_dir.empty() && !results.eigenvalues.empty()) {
        try {
            safe_system_call("mkdir -p " + params.output_dir);
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save eigenvalues: " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Ultra-Low-Memory Symmetry ED Complete" << std::endl;
    std::cout << "  Ground state energy: " << results.eigenvalues[0] << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return results;
}

// ============================================================================
// FIXED-SZ CHUNKED SYMMETRY ED
// ============================================================================

/**
 * @brief Ultra-low-memory symmetry ED for fixed-Sz sector
 * 
 * This is more efficient than the full Hilbert space version since
 * it only iterates over C(N, k) states instead of 2^N.
 * 
 * For spin-1/2 systems at Sz=0:
 *   N=30: C(30,15) = 155 million vs 2^30 = 1 billion (6.4x reduction)
 *   N=36: C(36,18) = 9 billion vs 2^36 = 69 billion (7.6x reduction)
 * 
 * @param directory Directory containing Hamiltonian files
 * @param n_up Number of up spins (determines Sz sector)
 * @param method Diagonalization method
 * @param params Parameters for diagonalization
 * @param interaction_filename Name of interaction file
 * @param single_site_filename Name of single-site file
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_chunked_symmetry_fixed_sz(
    const std::string& directory,
    int64_t n_up,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Ultra-Low-Memory Chunked Symmetry ED" << std::endl;
    std::cout << "  (Fixed Sz Mode: n_up=" << n_up << ")" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // ========== Step 2: Build Sectors Using Chunked Algorithm ==========
    std::cout << "\nBuilding symmetry sectors (fixed-Sz, ultra-low-memory mode)..." << std::endl;
    
    ChunkedSymmetryBuilderFixedSz builder(params.num_sites, params.spin_length, n_up);
    
    // Combined discovery and build
    builder.generateSectors(directory);
    
    std::string cache_dir = builder.getCacheDir();
    
    // ========== Step 3: Load Hamiltonian ==========
    std::cout << "\nLoading Hamiltonian for diagonalization..." << std::endl;
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    FixedSzOperator base_hamiltonian(params.num_sites, params.spin_length, n_up);
    base_hamiltonian.loadFromFile(single_site_file);
    base_hamiltonian.loadFromInterAllFile(interaction_file);
    
    // Load symmetry information (needed for applying Hamiltonian)
    base_hamiltonian.symmetry_info.loadFromDirectory(directory);
    
    std::cout << "Loaded " << base_hamiltonian.getTransformData().size() << " Hamiltonian terms" << std::endl;
    
    // Load metadata
    std::ifstream meta_file(cache_dir + "/metadata.txt");
    std::string line;
    size_t num_sectors = 0;
    std::vector<uint64_t> sector_dims;
    
    while (std::getline(meta_file, line)) {
        if (line.find("num_sectors") == 0) {
            sscanf(line.c_str(), "num_sectors %zu", &num_sectors);
            sector_dims.resize(num_sectors);
        } else if (line.find("sector_") == 0) {
            size_t idx;
            uint64_t dim;
            sscanf(line.c_str(), "sector_%zu_dim %lu", &idx, &dim);
            if (idx < num_sectors) {
                sector_dims[idx] = dim;
            }
        }
    }
    meta_file.close();
    
    std::cout << "\nFound " << num_sectors << " symmetry sectors" << std::endl;
    
    // ========== Step 4: Diagonalize Each Sector ==========
    std::cout << "\n========== Diagonalizing Sectors ==========\n" << std::endl;
    
    struct EigenInfo {
        double value;
        uint64_t sector_idx;
        uint64_t eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;
    
    for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
        uint64_t sector_dim = sector_dims[sector_idx];
        
        if (sector_dim == 0) {
            std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                     << ": empty (skipping)" << std::endl;
            continue;
        }
        
        // Load this sector
        std::string sector_file = cache_dir + "/sector_" + std::to_string(sector_idx) + ".bin";
        SymmetrySector sector = loadSectorFromDisk(sector_file);
        
        std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                  << " (dim=" << sector_dim << ", QN=[";
        for (size_t i = 0; i < sector.quantum_numbers.size(); ++i) {
            std::cout << sector.quantum_numbers[i];
            if (i < sector.quantum_numbers.size() - 1) std::cout << ",";
        }
        std::cout << "])" << std::endl;
        
        // Build lookup table
        std::unordered_map<uint64_t, size_t> sector_lookup;
        for (size_t i = 0; i < sector.basis_states.size(); ++i) {
            for (uint64_t elem : sector.basis_states[i].orbit_elements) {
                sector_lookup[elem] = i;
            }
        }
        
        // Get symmetry info
        const auto& symmetry_info = base_hamiltonian.symmetry_info;
        const double group_size = static_cast<double>(symmetry_info.max_clique.size());
        const double group_norm = 1.0 / group_size;
        const double spin_l = base_hamiltonian.getSpin();
        const auto& transform_data = base_hamiltonian.getTransformData();
        
        // Create matrix-free operator using fixed-Sz Hamiltonian
        auto matvec = [&](const Complex* in, Complex* out, int size) {
            std::fill(out, out + sector_dim, Complex(0.0, 0.0));
            
            for (size_t j = 0; j < sector_dim; ++j) {
                Complex c_j = in[j];
                if (std::abs(c_j) < 1e-15) continue;
                
                const auto& state_j = sector.basis_states[j];
                const double norm_j = state_j.norm;
                
                for (size_t orbit_idx = 0; orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                    uint64_t s = state_j.orbit_elements[orbit_idx];
                    Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                    
                    if (std::abs(alpha_s) < 1e-15) continue;
                    
                    Complex prefactor = c_j * alpha_s / norm_j;
                    
                    // Apply Hamiltonian terms
                    for (const auto& tdata : transform_data) {
                        uint64_t s_prime = s;
                        Complex h_element = tdata.coefficient;
                        bool valid = true;
                        
                        if (!tdata.is_two_body) {
                            if (tdata.op_type == 2) {
                                double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                                h_element *= spin_l * sign;
                            } else {
                                uint64_t bit = (s >> tdata.site_index) & 1;
                                if (bit != tdata.op_type) {
                                    s_prime ^= (1ULL << tdata.site_index);
                                } else {
                                    valid = false;
                                }
                            }
                        } else {
                            if (tdata.op_type == 2) {
                                double sign = ((s_prime >> tdata.site_index) & 1) ? -1.0 : 1.0;
                                h_element *= spin_l * sign;
                            } else {
                                uint64_t bit = (s_prime >> tdata.site_index) & 1;
                                if (bit != tdata.op_type) {
                                    s_prime ^= (1ULL << tdata.site_index);
                                } else {
                                    valid = false;
                                }
                            }
                            
                            if (valid) {
                                if (tdata.op_type_2 == 2) {
                                    double sign = ((s_prime >> tdata.site_index_2) & 1) ? -1.0 : 1.0;
                                    h_element *= spin_l * sign;
                                } else {
                                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                                    if (bit != tdata.op_type_2) {
                                        s_prime ^= (1ULL << tdata.site_index_2);
                                    } else {
                                        valid = false;
                                    }
                                }
                            }
                        }
                        
                        if (valid && std::abs(h_element) > 1e-15) {
                            auto it = sector_lookup.find(s_prime);
                            if (it != sector_lookup.end()) {
                                size_t k = it->second;
                                const auto& state_k = sector.basis_states[k];
                                
                                Complex beta_s_prime(0.0, 0.0);
                                for (size_t idx = 0; idx < state_k.orbit_elements.size(); ++idx) {
                                    if (state_k.orbit_elements[idx] == s_prime) {
                                        beta_s_prime = state_k.orbit_coefficients[idx];
                                        break;
                                    }
                                }
                                
                                out[k] += prefactor * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
                            }
                        }
                    }
                }
            }
        };
        
        // Diagonalize
        EDParameters sector_params = params;
        sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
        
        EDResults sector_results = exact_diagonalization_core(
            matvec, sector_dim, method, sector_params
        );
        
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }
        
        std::cout << "  Lowest: " << sector_results.eigenvalues[0] << std::endl;
    }
    
    // Sort and collect results
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    if (all_eigen_info.size() > params.num_eigenvalues) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    EDResults results;
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Ultra-Low-Memory Symmetry ED Complete" << std::endl;
    std::cout << "  Ground state energy: " << results.eigenvalues[0] << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return results;
}

// ============================================================================
// DISK-BASED CHUNKED SYMMETRY ED (MAXIMUM MEMORY EFFICIENCY)
// ============================================================================

/**
 * @brief Maximum memory efficiency ED using disk-based chunked algorithm
 * 
 * This version minimizes memory even further by:
 * 1. Processing Hilbert space in chunks
 * 2. Writing orbit representatives to temp files
 * 3. External merge-sort to combine results
 * 4. Building sectors from disk-stored representatives
 * 
 * Memory: O(chunk_size) during discovery
 * Disk: O(num_orbits) for temporary storage
 * 
 * Use this when even storing all orbit representatives in memory is too much.
 */
inline EDResults exact_diagonalization_disk_chunked_symmetry(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    size_t chunk_size = 10000000,  // 10M states per chunk
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Disk-Based Chunked Symmetry ED" << std::endl;
    std::cout << "  (Maximum Memory Efficiency Mode)" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // ========== Step 2: Disk-Based Orbit Discovery ==========
    DiskBasedChunkedSymmetryBuilder builder(params.num_sites, params.spin_length);
    builder.discoverOrbitsDiskBased(directory, chunk_size);
    
    // ========== Step 3: Build Sectors from Disk ==========
    builder.buildSectorsFromDisk(directory);
    
    // ========== Step 4: Diagonalize using disk streaming ==========
    // Use the existing disk streaming ED function
    return exact_diagonalization_disk_streaming(
        directory, method, params, interaction_filename, single_site_filename
    );
}

#endif // ED_WRAPPER_CHUNKED_H
