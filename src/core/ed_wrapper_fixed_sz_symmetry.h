#ifndef ED_WRAPPER_FIXED_SZ_SYMMETRY_H
#define ED_WRAPPER_FIXED_SZ_SYMMETRY_H

#include "ed_wrapper.h"
#include "construct_ham.h"
#include "hdf5_symmetry_io.h"
#include <sys/stat.h>

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
    const std::string& single_site_filename = "Trans.dat"
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
    
    FixedSzOperator hamiltonian(params.num_sites, params.spin_length, n_up);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
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
                std::cout << "  Transforming TPQ states from sector basis to full basis..." << std::endl;
                // Cast FixedSzOperator to Operator (safe since FixedSzOperator inherits from Operator)
                Operator& op_ref = static_cast<Operator&>(hamiltonian);
                ed_internal::transform_and_save_tpq_states(
                    block_params.output_dir, params.output_dir, 
                    op_ref, directory,
                    block_dim, block_start_dim, block_idx, params.num_sites
                );
            }
            
            if (params.compute_eigenvectors && method != DiagonalizationMethod::mTPQ && 
                method != DiagonalizationMethod::mTPQ_CUDA && method != DiagonalizationMethod::cTPQ) {
                std::cout << "  Transforming eigenvectors from sector basis to full basis..." << std::endl;
                Operator& op_ref = static_cast<Operator&>(hamiltonian);
                ed_internal::transform_and_save_eigenvectors(
                    block_params.output_dir, params.output_dir,
                    op_ref, directory,
                    block_results.eigenvalues, block_dim, block_start_dim, block_idx, params.num_sites
                );
            }
        }
        
        std::cout << "  Found " << block_results.eigenvalues.size() << " eigenvalues" << std::endl;
        if (!block_results.eigenvalues.empty()) {
            std::cout << "  Lowest: " << block_results.eigenvalues[0] << std::endl;
        }
        std::cout << std::endl;
        
        block_start_dim += block_dim;
    }
    
    // ========== Step 5: Collect and Sort Results ==========
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    if (all_eigen_info.size() > static_cast<size_t>(params.num_eigenvalues)) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    // Create eigenvector mapping file if eigenvectors were computed
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

#endif // ED_WRAPPER_FIXED_SZ_SYMMETRY_H
