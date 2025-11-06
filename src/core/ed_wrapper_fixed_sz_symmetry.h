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
    
    // For FTLM: collect results from all sectors for proper combination
    bool is_ftlm = (method == DiagonalizationMethod::FTLM);
    std::vector<FTLMResults> sector_ftlm_results;
    std::vector<uint64_t> sector_dimensions;
    
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
        
        // For FTLM: store sector results for later combination
        if (is_ftlm) {
            sector_ftlm_results.push_back(block_results.ftlm_results);
            sector_dimensions.push_back(block_dim);
        }
        
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
                std::cout << "  Transforming TPQ states from sector basis to fixed-Sz basis..." << std::endl;
                // Cast FixedSzOperator to Operator (safe since FixedSzOperator inherits from Operator)
                Operator& op_ref = static_cast<Operator&>(hamiltonian);
                ed_internal::transform_and_save_tpq_states(
                    block_params.output_dir, params.output_dir, 
                    op_ref, directory,
                    block_dim, block_start_dim, block_idx, params.num_sites
                );
                
                // Additional transformation: fixed-Sz basis -> full Hilbert space for TPQ states
                std::cout << "  Transforming TPQ states from fixed-Sz basis to full Hilbert space..." << std::endl;
                uint64_t full_dim = 1ULL << params.num_sites;
                uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
                
                // Find all TPQ state files in output directory
                std::string temp_list_file = params.output_dir + "/tpq_state_files.txt";
                std::string find_command = "find \"" + params.output_dir + "\" -name \"tpq_state_*.dat\" 2>/dev/null > \"" + temp_list_file + "\"";
                safe_system_call(find_command);
                
                std::ifstream file_list(temp_list_file);
                if (file_list.is_open()) {
                    std::string tpq_state_file;
                    while (std::getline(file_list, tpq_state_file)) {
                        if (tpq_state_file.empty()) continue;
                        
                        std::ifstream infile(tpq_state_file, std::ios::binary);
                        if (infile.is_open()) {
                            // Read TPQ state in fixed-Sz basis
                            std::vector<Complex> fixed_sz_vec(fixed_sz_dim);
                            infile.read(reinterpret_cast<char*>(fixed_sz_vec.data()), fixed_sz_dim * sizeof(Complex));
                            infile.close();
                            
                            // Transform to full Hilbert space basis
                            std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                            
                            // Overwrite file with full-space TPQ state
                            std::ofstream outfile(tpq_state_file, std::ios::binary);
                            outfile.write(reinterpret_cast<const char*>(full_vec.data()), full_dim * sizeof(Complex));
                            outfile.close();
                        }
                    }
                    file_list.close();
                }
                std::remove(temp_list_file.c_str());
                std::cout << "  All TPQ states transformed to full Hilbert space." << std::endl;
            }
            
            if (params.compute_eigenvectors && method != DiagonalizationMethod::mTPQ && 
                method != DiagonalizationMethod::mTPQ_CUDA && method != DiagonalizationMethod::cTPQ) {
                std::cout << "  Transforming eigenvectors from sector basis to fixed-Sz basis..." << std::endl;
                Operator& op_ref = static_cast<Operator&>(hamiltonian);
                ed_internal::transform_and_save_eigenvectors(
                    block_params.output_dir, params.output_dir,
                    op_ref, directory,
                    block_results.eigenvalues, block_dim, block_start_dim, block_idx, params.num_sites
                );
                
                // Additional transformation: fixed-Sz basis -> full Hilbert space
                std::cout << "  Transforming eigenvectors from fixed-Sz basis to full Hilbert space..." << std::endl;
                uint64_t full_dim = 1ULL << params.num_sites;
                uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
                
                for (size_t i = 0; i < block_results.eigenvalues.size(); ++i) {
                    std::string eigvec_file = params.output_dir + "/eigenvector_sector" + 
                                             std::to_string(block_idx) + "_" + std::to_string(i) + ".dat";
                    
                    std::ifstream infile(eigvec_file, std::ios::binary);
                    if (infile.is_open()) {
                        // Read eigenvector in fixed-Sz basis
                        std::vector<Complex> fixed_sz_vec(fixed_sz_dim);
                        infile.read(reinterpret_cast<char*>(fixed_sz_vec.data()), fixed_sz_dim * sizeof(Complex));
                        infile.close();
                        
                        // Transform to full Hilbert space basis
                        std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                        
                        // Overwrite file with full-space eigenvector
                        std::ofstream outfile(eigvec_file, std::ios::binary);
                        outfile.write(reinterpret_cast<const char*>(full_vec.data()), full_dim * sizeof(Complex));
                        outfile.close();
                        
                        std::cout << "    Transformed eigenvector " << i << " to full space (dim: " 
                                  << fixed_sz_dim << " -> " << full_dim << ")" << std::endl;
                    } else {
                        std::cerr << "    Warning: Could not open eigenvector file: " << eigvec_file << std::endl;
                    }
                }
            }
        }
        
        std::cout << "  Found " << block_results.eigenvalues.size() << " eigenvalues" << std::endl;
        if (!block_results.eigenvalues.empty()) {
            std::cout << "  Lowest: " << block_results.eigenvalues[0] << std::endl;
        }
        std::cout << std::endl;
        
        block_start_dim += block_dim;
    }
    
    // ========== Step 5.5: Sort Eigenvalues First ==========
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    if (all_eigen_info.size() > static_cast<size_t>(params.num_eigenvalues)) {
        all_eigen_info.resize(params.num_eigenvalues);
    }
    
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    // ========== Step 5.6: Combine FTLM Results from Multiple Sectors ==========
    
    // For FTLM with multiple sectors: combine thermodynamic results properly
    if (is_ftlm && sector_ftlm_results.size() > 1) {
        std::cout << "\n=== Combining FTLM Results from " << sector_ftlm_results.size() 
                  << " Symmetry Sectors ===" << std::endl;
        
        // Combine sector results with proper statistical weights
        results.thermo_data = combine_ftlm_sector_results(
            sector_ftlm_results, sector_dimensions
        );
        
        // Save combined results
        if (!params.output_dir.empty()) {
            std::string ftlm_dir = params.output_dir + "/thermo";
            safe_system_call("mkdir -p " + ftlm_dir);
            
            // Create a combined FTLMResults for saving
            FTLMResults combined_results;
            combined_results.thermo_data = results.thermo_data;
            combined_results.ground_state_estimate = results.eigenvalues.empty() ? 0.0 : results.eigenvalues[0];
            combined_results.total_samples = sector_ftlm_results[0].total_samples;
            
            // Initialize error arrays with zeros (no error bars for combined results)
            size_t n_temps = combined_results.thermo_data.temperatures.size();
            combined_results.energy_error.assign(n_temps, 0.0);
            combined_results.specific_heat_error.assign(n_temps, 0.0);
            combined_results.entropy_error.assign(n_temps, 0.0);
            combined_results.free_energy_error.assign(n_temps, 0.0);
            
            // Save combined thermodynamics
            save_ftlm_results(combined_results, ftlm_dir + "/ftlm_thermo_combined.txt");
            
            std::cout << "Combined FTLM results saved to: " << ftlm_dir << "/ftlm_thermo_combined.txt" << std::endl;
            
            // Also save individual sector results for debugging
            for (size_t s = 0; s < sector_ftlm_results.size(); ++s) {
                std::string sector_file = ftlm_dir + "/ftlm_thermo_sector_" + 
                                         std::to_string(s) + ".txt";
                save_ftlm_results(sector_ftlm_results[s], sector_file);
            }
            std::cout << "Individual sector results saved to: " << ftlm_dir << "/ftlm_thermo_sector_*.txt" << std::endl;
        }
        
        std::cout << "=== FTLM Sector Combination Complete ===" << std::endl;
    } else if (is_ftlm && sector_ftlm_results.size() == 1) {
        // Single sector - use it directly
        std::cout << "\nNote: Only one symmetry sector computed. Results represent this sector only." << std::endl;
        results.thermo_data = sector_ftlm_results[0].thermo_data;
        results.ftlm_results = sector_ftlm_results[0];
    }
    
    // ========== Step 6: Create Eigenvector Mapping ==========
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
