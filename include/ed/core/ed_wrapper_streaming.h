#ifndef ED_WRAPPER_STREAMING_H
#define ED_WRAPPER_STREAMING_H

#include <ed/core/ed_wrapper.h>
#include <ed/core/streaming_symmetry.h>

/**
 * @file ed_wrapper_streaming.h
 * @brief Streaming symmetry-adapted exact diagonalization wrapper
 * 
 * This file provides wrapper functions for exact diagonalization using
 * the streaming symmetry implementation that avoids disk storage of
 * basis vectors and block matrices.
 * 
 * Advantages over standard symmetrized ED:
 * - No disk I/O for basis vectors (faster)
 * - No disk storage for block matrices (saves GBs of disk space)
 * - Lower memory footprint
 * - Can handle larger systems
 * - Simpler workflow (no intermediate file generation)
 */

// ============================================================================
// STREAMING SYMMETRY ED FUNCTIONS
// ============================================================================

/**
 * @brief Exact diagonalization using streaming symmetry (no disk storage)
 * 
 * This function performs symmetry-adapted ED without storing basis vectors
 * or block matrices on disk. It uses on-the-fly computation of matrix-vector
 * products in the symmetrized basis.
 * 
 * Workflow:
 * 1. Generate automorphisms (if not present)
 * 2. Generate orbit representatives for each symmetry sector (in memory)
 * 3. Diagonalize each sector using matrix-free methods
 * 4. Collect and sort eigenvalues
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method (must support matrix-free operations)
 * @param params Parameters for diagonalization
 * @param interaction_filename Name of interaction file
 * @param single_site_filename Name of single-site file
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_streaming_symmetry(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Streaming Symmetry Exact Diagonalization" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // ========== Step 2: Load Hamiltonian ==========
    std::cout << "\nLoading Hamiltonian..." << std::endl;
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    StreamingSymmetryOperator hamiltonian(params.num_sites, params.spin_length);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    // ========== Step 3: Generate Symmetry Sectors (Streaming) ==========
    std::cout << "\nGenerating symmetry sectors (streaming mode)..." << std::endl;
    hamiltonian.generateSymmetrySectorsStreaming(directory);
    
    size_t num_sectors = hamiltonian.getNumSectors();
    std::cout << "\nFound " << num_sectors << " symmetry sectors" << std::endl;
    
    if (num_sectors == 0) {
        throw std::runtime_error("No symmetry sectors found");
    }
    
    // Optional: Save sector metadata for later use
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
        hamiltonian.saveSectorMetadata(params.output_dir);
    }
    
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
        uint64_t sector_dim = hamiltonian.getSectorDimension(sector_idx);
        
        if (sector_dim == 0) {
            std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                     << ": empty (skipping)" << std::endl;
            continue;
        }
        
        const auto& sector = hamiltonian.getSector(sector_idx);
        std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                  << " (dim=" << sector_dim << ", QN=[";
        for (size_t i = 0; i < sector.quantum_numbers.size(); ++i) {
            std::cout << sector.quantum_numbers[i];
            if (i < sector.quantum_numbers.size() - 1) std::cout << ",";
        }
        std::cout << "])" << std::endl;
        
        // Create matrix-free operator for this sector
        auto matvec = [&hamiltonian, sector_idx](const Complex* in, Complex* out, int size) {
            hamiltonian.applySymmetrized(sector_idx, in, out);
        };
        
        // Configure parameters for this sector
        EDParameters sector_params = params;
        sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
        
        if (params.compute_eigenvectors && !params.output_dir.empty()) {
            sector_params.output_dir = params.output_dir + "/sector_" + std::to_string(sector_idx);
            safe_system_call("mkdir -p " + sector_params.output_dir);
        }
        
        // Diagonalize this sector
        std::cout << "  Diagonalizing with " << sector_params.num_eigenvalues 
                  << " eigenvalues..." << std::endl;
        
        EDResults sector_results = exact_diagonalization_core(
            matvec, sector_dim, method, sector_params
        );
        
        // Store eigenvalue info
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }
        
        std::cout << "  Found " << sector_results.eigenvalues.size() << " eigenvalues" << std::endl;
        if (!sector_results.eigenvalues.empty()) {
            std::cout << "  Lowest eigenvalue: " << sector_results.eigenvalues[0] << std::endl;
        }
        std::cout << std::endl;
    }
    
    // ========== Step 5: Collect and Sort Results ==========
    std::cout << "Collecting results..." << std::endl;
    
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    // Limit to requested number of eigenvalues
    if (all_eigen_info.size() > params.num_eigenvalues) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    EDResults results;
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Save combined eigenvalues to HDF5
    if (!params.output_dir.empty() && !results.eigenvalues.empty()) {
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                     << " combined eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save combined eigenvalues: " << e.what() << std::endl;
        }
    }
    
    // Save eigenvalue mapping
    if (!params.output_dir.empty()) {
        std::ofstream map_file(params.output_dir + "/eigenvalue_mapping.txt");
        if (map_file.is_open()) {
            map_file << "# Global_Index Eigenvalue Sector_Index Sector_Eigenvalue_Index\n";
            for (size_t i = 0; i < all_eigen_info.size(); ++i) {
                const auto& info = all_eigen_info[i];
                map_file << i << " " << info.value << " " 
                        << info.sector_idx << " " << info.eigen_idx << "\n";
            }
            map_file.close();
            std::cout << "Saved eigenvalue mapping to " 
                     << params.output_dir << "/eigenvalue_mapping.txt" << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Streaming Symmetry ED Complete" << std::endl;
    std::cout << "  Total eigenvalues found: " << results.eigenvalues.size() << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "  Ground state energy: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    return results;
}

/**
 * @brief Exact diagonalization using streaming symmetry with fixed Sz
 * 
 * Combines Sz conservation with spatial symmetries in streaming mode.
 * This is the most efficient approach for systems with both symmetries.
 * 
 * MATHEMATICAL BACKGROUND:
 * ========================
 * The total Sz operator commutes with all lattice symmetry operators:
 *   [S^z_total, P_g] = 0  for all permutations g in the automorphism group
 * 
 * This means we can simultaneously diagonalize:
 *   1. The Hamiltonian H
 *   2. The total Sz operator (eigenvalue: n_up - n_down)
 *   3. The symmetry group representation (irrep labels)
 * 
 * SECTOR STRUCTURE:
 * =================
 * The Hilbert space decomposes as:
 *   H = ⊕_{Sz} ⊕_{irrep} H_{Sz,irrep}
 * 
 * Not all (Sz, irrep) sectors are non-empty:
 *   - Some irreps may have no states in a given Sz sector
 *   - Empty sectors are automatically skipped
 *   - The code counts and reports non-empty sectors
 * 
 * Example: For a 4-site chain with C4 symmetry and Sz=0:
 *   - k=0 (trivial) sector: contains identity representation states
 *   - k=π sector: may have states or be empty depending on n_up
 * 
 * @param directory Directory containing Hamiltonian files
 * @param n_up Number of up spins (determines Sz sector: Sz = n_up - N/2 for spin-1/2)
 * @param method Diagonalization method
 * @param params Parameters for diagonalization
 * @param interaction_filename Name of interaction file
 * @param single_site_filename Name of single-site file
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_streaming_symmetry_fixed_sz(
    const std::string& directory,
    int64_t n_up,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Streaming Symmetry ED (Fixed Sz)" << std::endl;
    std::cout << "  Sz sector: N_up = " << n_up << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // ========== Step 2: Load Hamiltonian ==========
    std::cout << "\nLoading Hamiltonian..." << std::endl;
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    FixedSzStreamingSymmetryOperator hamiltonian(params.num_sites, params.spin_length, n_up);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
    std::cout << "Fixed Sz dimension: " << fixed_sz_dim << std::endl;
    
    // ========== Step 3: Generate Symmetry Sectors (Streaming) ==========
    std::cout << "\nGenerating symmetry sectors (streaming mode, fixed Sz)..." << std::endl;
    hamiltonian.generateSymmetrySectorsStreamingFixedSz(directory);
    
    size_t num_sectors = hamiltonian.getNumSectors();
    
    // Count non-empty sectors
    size_t non_empty_sectors = 0;
    uint64_t total_sym_dim = 0;
    for (size_t i = 0; i < num_sectors; ++i) {
        uint64_t dim = hamiltonian.getSectorDimension(i);
        if (dim > 0) {
            non_empty_sectors++;
            total_sym_dim += dim;
        }
    }
    
    std::cout << "\n=== Fixed Sz × Symmetry Sector Statistics ===" << std::endl;
    std::cout << "Total symmetry irreps: " << num_sectors << std::endl;
    std::cout << "Non-empty sectors in Sz=" << (n_up - static_cast<int64_t>(params.num_sites)/2) 
              << " (N_up=" << n_up << "): " << non_empty_sectors << std::endl;
    std::cout << "Total symmetrized dimension: " << total_sym_dim << std::endl;
    std::cout << "Dimension reduction: " << fixed_sz_dim << " -> " << total_sym_dim 
              << " (" << std::fixed << std::setprecision(1) 
              << (100.0 * (1.0 - double(total_sym_dim) / fixed_sz_dim)) << "% reduction)" 
              << std::defaultfloat << std::endl;
    
    if (non_empty_sectors == 0) {
        std::cerr << "Error: All symmetry sectors are empty in this Sz sector.\n";
        std::cerr << "This can happen if the Sz value is incompatible with the symmetry irreps.\n";
        throw std::runtime_error("No non-empty symmetry sectors found in fixed Sz sector");
    }
    
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
        uint64_t sector_dim = hamiltonian.getSectorDimension(sector_idx);
        
        if (sector_dim == 0) {
            std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                     << ": empty (skipping)" << std::endl;
            continue;
        }
        
        const auto& sector = hamiltonian.getSector(sector_idx);
        std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors 
                  << " (dim=" << sector_dim << ", QN=[";
        for (size_t i = 0; i < sector.quantum_numbers.size(); ++i) {
            std::cout << sector.quantum_numbers[i];
            if (i < sector.quantum_numbers.size() - 1) std::cout << ",";
        }
        std::cout << "])" << std::endl;
        
        // Create matrix-free operator
        auto matvec = [&hamiltonian, sector_idx](const Complex* in, Complex* out, int size) {
            hamiltonian.applySymmetrizedFixedSz(sector_idx, in, out);
        };
        
        // Configure parameters
        EDParameters sector_params = params;
        sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
        
        if (params.compute_eigenvectors && !params.output_dir.empty()) {
            sector_params.output_dir = params.output_dir + "/sector_" + std::to_string(sector_idx);
            safe_system_call("mkdir -p " + sector_params.output_dir);
        }
        
        // Diagonalize
        std::cout << "  Diagonalizing..." << std::endl;
        EDResults sector_results = exact_diagonalization_core(
            matvec, sector_dim, method, sector_params
        );
        
        // Store results
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }
        
        std::cout << "  Found " << sector_results.eigenvalues.size() << " eigenvalues" << std::endl;
        if (!sector_results.eigenvalues.empty()) {
            std::cout << "  Lowest: " << sector_results.eigenvalues[0] << std::endl;
        }
        std::cout << std::endl;
    }
    
    // ========== Step 5: Collect Results ==========
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    if (all_eigen_info.size() > params.num_eigenvalues) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    EDResults results;
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Save combined eigenvalues to HDF5
    if (!params.output_dir.empty() && !results.eigenvalues.empty()) {
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                     << " combined eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save combined eigenvalues: " << e.what() << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Streaming Symmetry ED Complete" << std::endl;
    std::cout << "  Total eigenvalues: " << results.eigenvalues.size() << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "  Ground state: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    return results;
}

#endif // ED_WRAPPER_STREAMING_H
