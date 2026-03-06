#ifndef ED_WRAPPER_STREAMING_H
#define ED_WRAPPER_STREAMING_H

#include <ed/core/ed_wrapper.h>
#include <ed/core/streaming_symmetry.h>
#include <set>

#ifdef WITH_CUDA
#include <ed/gpu/gpu_ed_wrapper.h>
#endif

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

#ifdef WITH_CUDA
namespace ed_streaming_internal {

/**
 * @brief Extract flattened orbit data from a symmetry sector for GPU transfer
 *
 * Converts the per-basis-state orbit data (vector-of-vectors) into CSR-like
 * flattened arrays suitable for GPU upload.
 */
inline void extractOrbitData(
    const SymmetrySector& sector,
    std::vector<uint64_t>& orbit_elements,
    std::vector<std::complex<double>>& orbit_coefficients,
    std::vector<int>& orbit_offsets,
    std::vector<double>& orbit_norms
) {
    size_t num_basis = sector.basis_states.size();
    orbit_offsets.resize(num_basis + 1);
    orbit_norms.resize(num_basis);

    // First pass: compute offsets
    orbit_offsets[0] = 0;
    for (size_t j = 0; j < num_basis; ++j) {
        orbit_offsets[j + 1] = orbit_offsets[j] +
            static_cast<int>(sector.basis_states[j].orbit_elements.size());
    }

    size_t total_elements = orbit_offsets[num_basis];
    orbit_elements.resize(total_elements);
    orbit_coefficients.resize(total_elements);

    // Second pass: flatten
    for (size_t j = 0; j < num_basis; ++j) {
        const auto& bs = sector.basis_states[j];
        orbit_norms[j] = bs.norm;
        int offset = orbit_offsets[j];
        for (size_t e = 0; e < bs.orbit_elements.size(); ++e) {
            orbit_elements[offset + e] = bs.orbit_elements[e];
            orbit_coefficients[offset + e] = bs.orbit_coefficients[e];
        }
    }
}

/**
 * @brief Dispatch GPU solver for a symmetrized sector
 *
 * Creates a GPUSymmetrizedOperator for the sector, runs the appropriate
 * GPU solver, and collects eigenvalues.
 */
inline EDResults dispatchGPUSymmetrizedSector(
    DiagonalizationMethod method,
    const EDParameters& params,
    int n_sites, float spin_l,
    uint64_t sector_dim,
    const SymmetrySector& sector,
    int group_size,
    const std::string& interall_file,
    const std::string& trans_file
) {
    // Extract orbit data
    std::vector<uint64_t> orbit_elements;
    std::vector<std::complex<double>> orbit_coefficients;
    std::vector<int> orbit_offsets;
    std::vector<double> orbit_norms;
    extractOrbitData(sector, orbit_elements, orbit_coefficients, orbit_offsets, orbit_norms);

    // Create GPU operator
    void* gpu_op = GPUEDWrapper::createGPUSymmetrizedOperator(
        n_sites, spin_l, static_cast<int>(sector_dim),
        orbit_elements, orbit_coefficients, orbit_offsets, orbit_norms,
        group_size, interall_file, trans_file
    );

    if (!gpu_op) {
        throw std::runtime_error("Failed to create GPU symmetrized operator");
    }

    std::vector<double> eigenvalues;
    int num_eigs = static_cast<int>(std::min(params.num_eigenvalues, sector_dim));

    // Dispatch to appropriate GPU solver
    // Note: all symmetrized methods use the full-space kernel variants since
    // the symmetrized operator already handles the projection internally.
    if (method == DiagonalizationMethod::LANCZOS_GPU ||
        method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ) {
        GPUEDWrapper::runGPULanczos(
            gpu_op, static_cast<int>(sector_dim),
            params.max_iterations, num_eigs, params.tolerance,
            eigenvalues, params.output_dir, params.compute_eigenvectors);
    } else if (method == DiagonalizationMethod::BLOCK_LANCZOS_GPU ||
               method == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ) {
        GPUEDWrapper::runGPUBlockLanczos(
            gpu_op, static_cast<int>(sector_dim),
            params.max_iterations, num_eigs, params.block_size,
            params.tolerance, eigenvalues, params.output_dir,
            params.compute_eigenvectors);
    } else if (method == DiagonalizationMethod::DAVIDSON_GPU) {
        GPUEDWrapper::runGPUDavidson(
            gpu_op, static_cast<int>(sector_dim),
            num_eigs, params.max_iterations, params.max_subspace,
            params.tolerance, eigenvalues, params.output_dir,
            params.compute_eigenvectors);
    } else if (method == DiagonalizationMethod::KRYLOV_SCHUR_GPU) {
        GPUEDWrapper::runGPUKrylovSchur(
            gpu_op, static_cast<int>(sector_dim),
            num_eigs, params.max_iterations, params.tolerance,
            eigenvalues, params.output_dir, params.compute_eigenvectors);
    } else if (method == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU) {
        GPUEDWrapper::runGPUBlockKrylovSchur(
            gpu_op, static_cast<int>(sector_dim),
            num_eigs, params.max_iterations, params.block_size,
            params.tolerance, eigenvalues, params.output_dir,
            params.compute_eigenvectors);
    } else if (method == DiagonalizationMethod::FULL_GPU) {
        GPUEDWrapper::runGPUFullDiag(
            gpu_op, static_cast<int>(sector_dim),
            num_eigs, eigenvalues, params.output_dir,
            params.compute_eigenvectors);
    } else {
        GPUEDWrapper::destroyGPUOperator(gpu_op);
        throw std::runtime_error("Unsupported GPU method for symmetrized diagonalization: use Lanczos, Block Lanczos, Davidson, or Krylov-Schur GPU variants");
    }

    GPUEDWrapper::destroyGPUOperator(gpu_op);

    EDResults results;
    results.eigenvalues = eigenvalues;
    return results;
}

} // namespace ed_streaming_internal
#endif // WITH_CUDA

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
    const std::string& single_site_filename = "Trans.dat",
    const std::string& basis_cache_dir = "",
    bool precompute_basis_only = false
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
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    StreamingSymmetryOperator hamiltonian(params.num_sites, params.spin_length);

    if (!precompute_basis_only) {
        std::cout << "\nLoading Hamiltonian..." << std::endl;
        hamiltonian.loadFromFile(single_site_file);
        hamiltonian.loadFromInterAllFile(interaction_file);
    } else {
        // Only need operator metadata (n_bits_, etc.) for basis generation
        std::cout << "\nLoading operator metadata (precompute mode)..." << std::endl;
        hamiltonian.loadFromFile(single_site_file);
    }
    
    // ========== Step 3: Generate or Load Symmetry Sectors ==========
    std::string effective_cache_dir = basis_cache_dir;
    if (effective_cache_dir.empty() && precompute_basis_only) {
        effective_cache_dir = directory + "/basis_cache";
    }

    bool loaded_from_cache = false;
    if (!effective_cache_dir.empty()) {
        if (!precompute_basis_only) {
            std::cout << "\nChecking for cached orbit basis in "
                      << effective_cache_dir << "..." << std::endl;
            hamiltonian.symmetry_info.loadFromDirectory(directory);
            loaded_from_cache = hamiltonian.loadOrbitBasisHDF5(effective_cache_dir);
            if (loaded_from_cache) {
                std::cout << "*** Orbit basis loaded from cache — "
                          << "skipping sector generation ***" << std::endl;
            }
        }
    }

    if (!loaded_from_cache) {
        std::cout << "\nGenerating symmetry sectors (streaming mode)..." << std::endl;
        hamiltonian.generateSymmetrySectorsStreaming(directory);

        // Only save to cache in precompute mode — never during regular runs.
        // A regular run may have a different symmetry group (e.g. Jpm=0
        // enlarges the group) and overwriting the shared cache would corrupt
        // it for all other concurrent tasks.
        if (!effective_cache_dir.empty() && precompute_basis_only) {
            hamiltonian.saveOrbitBasisHDF5(effective_cache_dir);
        }
    }

    // If precompute-only, we're done
    if (precompute_basis_only) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Precompute Basis Complete (full-space)" << std::endl;
        std::cout << "  Cached to: " << effective_cache_dir << std::endl;
        std::cout << "========================================\n" << std::endl;
        return EDResults();
    }
    
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
    
    // Build set of selected sectors for filtering
    std::set<size_t> sector_filter;
    if (!params.selected_sectors.empty()) {
        for (int s : params.selected_sectors) {
            if (s >= 0 && static_cast<size_t>(s) < num_sectors) {
                sector_filter.insert(static_cast<size_t>(s));
            } else {
                std::cerr << "Warning: --sectors value " << s 
                          << " out of range [0, " << num_sectors << "), ignoring" << std::endl;
            }
        }
        std::cout << "Sector filter active: running " << sector_filter.size() 
                  << " of " << num_sectors << " sectors" << std::endl;
    }
    
    struct EigenInfo {
        double value;
        uint64_t sector_idx;
        uint64_t eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;

#ifdef WITH_CUDA
    bool use_gpu = ed_internal::is_gpu_method(method);
    int group_size = 0;
    if (use_gpu) {
        if (!GPUEDWrapper::isGPUAvailable()) {
            throw std::runtime_error("GPU method requested but no CUDA-capable GPU found");
        }
        GPUEDWrapper::printGPUInfo();
        group_size = static_cast<int>(hamiltonian.symmetry_info.max_clique.size());
        std::cout << "Using GPU symmetrized matvec (group_size=" << group_size << ")" << std::endl;
    }
#endif
    
    for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
        // Skip sectors not in the filter (if filter is active)
        if (!sector_filter.empty() && sector_filter.find(sector_idx) == sector_filter.end()) {
            continue;
        }
        
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

        EDResults sector_results;

#ifdef WITH_CUDA
        if (use_gpu) {
            // GPU dispatch: create symmetrized operator per sector
            EDParameters sector_params = params;
            sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
            if (!params.output_dir.empty()) {
                sector_params.output_dir = params.output_dir + "/sector_" + std::to_string(sector_idx);
                safe_system_call("mkdir -p " + sector_params.output_dir);
            }
            std::cout << "  GPU diagonalizing with " << sector_params.num_eigenvalues 
                      << " eigenvalues..." << std::endl;
            sector_results = ed_streaming_internal::dispatchGPUSymmetrizedSector(
                method, sector_params,
                params.num_sites, params.spin_length,
                sector_dim, sector, group_size,
                interaction_file, single_site_file
            );
        } else
#endif
        {
            // CPU dispatch: matrix-free operator
            auto matvec = [&hamiltonian, sector_idx](const Complex* in, Complex* out, int size) {
                hamiltonian.applySymmetrized(sector_idx, in, out);
            };
            EDParameters sector_params = params;
            sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
            if (!params.output_dir.empty()) {
                sector_params.output_dir = params.output_dir + "/sector_" + std::to_string(sector_idx);
                safe_system_call("mkdir -p " + sector_params.output_dir);
            }
            std::cout << "  Diagonalizing with " << sector_params.num_eigenvalues 
                      << " eigenvalues..." << std::endl;
            sector_results = exact_diagonalization_core(
                matvec, sector_dim, method, sector_params
            );
        }
        
        // Store eigenvalue info
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }

        // Save per-sector eigenvalues to HDF5
        if (!params.output_dir.empty() && !sector_results.eigenvalues.empty()) {
            std::string sector_out = params.output_dir + "/sector_" + std::to_string(sector_idx);
            try {
                std::string sector_hdf5 = HDF5IO::createOrOpenFile(sector_out);
                HDF5IO::saveEigenvalues(sector_hdf5, sector_results.eigenvalues);
            } catch (const std::exception& e) {
                std::cerr << "  Warning: Could not save sector eigenvalues: " << e.what() << std::endl;
            }
        }

        // Expand eigenvectors from symmetrized basis → full 2^N computational basis
        if (params.compute_eigenvectors && !sector_results.eigenvalues.empty()
            && !params.output_dir.empty()) {
            std::string sector_out = params.output_dir + "/sector_" + std::to_string(sector_idx);
            std::string sector_hdf5 = HDF5IO::createOrOpenFile(sector_out);
            for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
                try {
                    auto sym_vec = HDF5IO::loadEigenvector(sector_hdf5, i);
                    auto full_vec = hamiltonian.expandToComputationalBasis(
                        sector_idx, sym_vec);

                    std::string main_hdf5 = HDF5IO::createOrOpenFile(params.output_dir);
                    // Encode (sector, eigen_idx) into a unique global ID
                    size_t global_id = sector_idx * 10000 + i;
                    HDF5IO::saveEigenvector(main_hdf5, global_id, full_vec);
                } catch (const std::exception& e) {
                    std::cerr << "  Warning: Could not expand eigenvector " << i
                              << " from sector " << sector_idx << ": "
                              << e.what() << std::endl;
                }
            }
            std::cout << "  Expanded " << sector_results.eigenvalues.size()
                      << " eigenvectors to full basis" << std::endl;
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
    
    EDResults results;
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Save combined eigenvalues to HDF5 (only when not using sector filter)
    if (!params.output_dir.empty() && !results.eigenvalues.empty() && sector_filter.empty()) {
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
            map_file << "# Global_Index Eigenvalue Sector_Index Sector_Eigenvalue_Index";
            if (params.compute_eigenvectors) {
                map_file << " Eigenvector_Dataset";
            }
            map_file << "\n";
            for (size_t i = 0; i < all_eigen_info.size(); ++i) {
                const auto& info = all_eigen_info[i];
                map_file << i << " " << std::setprecision(15) << info.value << " " 
                        << info.sector_idx << " " << info.eigen_idx;
                if (params.compute_eigenvectors) {
                    size_t global_id = info.sector_idx * 10000 + info.eigen_idx;
                    map_file << " eigenvector_" << global_id;
                }
                map_file << "\n";
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
 * @param basis_cache_dir If non-empty, cache orbit basis to/from this directory
 * @param precompute_basis_only If true, generate & cache orbit basis then return (skip Lanczos)
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_streaming_symmetry_fixed_sz(
    const std::string& directory,
    int64_t n_up,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& basis_cache_dir = "",
    bool precompute_basis_only = false
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Streaming Symmetry ED (Fixed Sz)" << std::endl;
    std::cout << "  Sz sector: N_up = " << n_up << std::endl;
    if (!basis_cache_dir.empty())
        std::cout << "  Basis cache: " << basis_cache_dir << std::endl;
    if (precompute_basis_only)
        std::cout << "  Mode: PRECOMPUTE BASIS ONLY" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // ========== Step 1: Generate or Load Automorphisms ==========
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // ========== Step 2: Construct operator and load Hamiltonian ==========
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    FixedSzStreamingSymmetryOperator hamiltonian(params.num_sites, params.spin_length, n_up);

    // Hamiltonian (InterAll.dat + Trans.dat) is only needed for the Lanczos solve,
    // not for orbit basis construction.  Load it unless we're only precomputing.
    if (!precompute_basis_only) {
        std::cout << "\nLoading Hamiltonian..." << std::endl;
        hamiltonian.loadFromFile(single_site_file);
        hamiltonian.loadFromInterAllFile(interaction_file);
    } else {
        // Trans.dat contains single-site terms AND symmetry transforms.
        // We need to load it for operator metadata (n_bits_, basis_states_).
        // loadFromFile populates basis_states_ which is needed for sector generation.
        std::cout << "\nLoading operator metadata (precompute mode)..." << std::endl;
        hamiltonian.loadFromFile(single_site_file);
    }
    
    uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
    std::cout << "Fixed Sz dimension: " << fixed_sz_dim << std::endl;
    
    // ========== Step 3: Generate or Load Symmetry Sectors ==========
    // Determine effective cache directory
    std::string effective_cache_dir = basis_cache_dir;
    if (effective_cache_dir.empty() && precompute_basis_only) {
        effective_cache_dir = directory + "/basis_cache";
    }

    bool loaded_from_cache = false;
    if (!effective_cache_dir.empty()) {
        // Try loading from cache first (unless we're explicitly precomputing)
        if (!precompute_basis_only) {
            std::cout << "\nChecking for cached orbit basis in " << effective_cache_dir << "..." << std::endl;
            // Load symmetry info so the operator can verify group_size
            hamiltonian.symmetry_info.loadFromDirectory(directory);
            loaded_from_cache = hamiltonian.loadOrbitBasisHDF5(effective_cache_dir);
            if (loaded_from_cache) {
                std::cout << "*** Orbit basis loaded from cache — skipping sector generation ***" << std::endl;
            }
        }
    }

    if (!loaded_from_cache) {
        std::cout << "\nGenerating symmetry sectors (streaming mode, fixed Sz)..." << std::endl;
        hamiltonian.generateSymmetrySectorsStreamingFixedSz(directory);

        // Only save to cache in precompute mode — never during regular runs.
        // A regular run may have a different symmetry group (e.g. Jpm=0
        // enlarges the group) and overwriting the shared cache would corrupt
        // it for all other concurrent tasks.
        if (!effective_cache_dir.empty() && precompute_basis_only) {
            hamiltonian.saveOrbitBasisHDF5(effective_cache_dir);
        }
    }

    // If precompute-only, we're done
    if (precompute_basis_only) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "  Precompute Basis Complete" << std::endl;
        std::cout << "  Cached to: " << effective_cache_dir << std::endl;
        std::cout << "========================================\n" << std::endl;
        return EDResults();
    }
    
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
    
    // Build set of selected sectors for filtering
    std::set<size_t> sector_filter;
    if (!params.selected_sectors.empty()) {
        for (int s : params.selected_sectors) {
            if (s >= 0 && static_cast<size_t>(s) < num_sectors) {
                sector_filter.insert(static_cast<size_t>(s));
            } else {
                std::cerr << "Warning: --sectors value " << s 
                          << " out of range [0, " << num_sectors << "), ignoring" << std::endl;
            }
        }
        std::cout << "Sector filter active: running " << sector_filter.size() 
                  << " of " << num_sectors << " sectors" << std::endl;
    }
    
    struct EigenInfo {
        double value;
        uint64_t sector_idx;
        uint64_t eigen_idx;
        bool operator<(const EigenInfo& other) const { return value < other.value; }
    };
    std::vector<EigenInfo> all_eigen_info;

#ifdef WITH_CUDA
    bool use_gpu = ed_internal::is_gpu_method(method);
    int group_size = 0;
    if (use_gpu) {
        if (!GPUEDWrapper::isGPUAvailable()) {
            throw std::runtime_error("GPU method requested but no CUDA-capable GPU found");
        }
        GPUEDWrapper::printGPUInfo();
        group_size = static_cast<int>(hamiltonian.symmetry_info.max_clique.size());
        std::cout << "Using GPU symmetrized matvec (group_size=" << group_size 
                  << ", fixed Sz, N_up=" << n_up << ")" << std::endl;
    }
#endif
    
    for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
        // Skip sectors not in the filter (if filter is active)
        if (!sector_filter.empty() && sector_filter.find(sector_idx) == sector_filter.end()) {
            continue;
        }
        
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

        EDResults sector_results;

#ifdef WITH_CUDA
        if (use_gpu) {
            // GPU dispatch: create symmetrized operator per sector
            EDParameters sector_params = params;
            sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
            if (!params.output_dir.empty()) {
                sector_params.output_dir = params.output_dir + "/sector_" + std::to_string(sector_idx);
                safe_system_call("mkdir -p " + sector_params.output_dir);
            }
            std::cout << "  GPU diagonalizing..." << std::endl;
            sector_results = ed_streaming_internal::dispatchGPUSymmetrizedSector(
                method, sector_params,
                params.num_sites, params.spin_length,
                sector_dim, sector, group_size,
                interaction_file, single_site_file
            );
        } else
#endif
        {
            // CPU dispatch: matrix-free operator
            auto matvec = [&hamiltonian, sector_idx](const Complex* in, Complex* out, int size) {
                hamiltonian.applySymmetrizedFixedSz(sector_idx, in, out);
            };
            EDParameters sector_params = params;
            sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
            if (!params.output_dir.empty()) {
                sector_params.output_dir = params.output_dir + "/sector_" + std::to_string(sector_idx);
                safe_system_call("mkdir -p " + sector_params.output_dir);
            }
            std::cout << "  Diagonalizing..." << std::endl;
            sector_results = exact_diagonalization_core(
                matvec, sector_dim, method, sector_params
            );
        }
        
        // Store results
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }

        // Save per-sector eigenvalues to HDF5
        if (!params.output_dir.empty() && !sector_results.eigenvalues.empty()) {
            std::string sector_out = params.output_dir + "/sector_" + std::to_string(sector_idx);
            try {
                std::string sector_hdf5 = HDF5IO::createOrOpenFile(sector_out);
                HDF5IO::saveEigenvalues(sector_hdf5, sector_results.eigenvalues);
            } catch (const std::exception& e) {
                std::cerr << "  Warning: Could not save sector eigenvalues: " << e.what() << std::endl;
            }
        }

        // Expand eigenvectors from symmetrized basis → full 2^N computational basis
        if (params.compute_eigenvectors && !sector_results.eigenvalues.empty()
            && !params.output_dir.empty()) {
            std::string sector_out = params.output_dir + "/sector_" + std::to_string(sector_idx);
            std::string sector_hdf5 = HDF5IO::createOrOpenFile(sector_out);
            for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
                try {
                    auto sym_vec = HDF5IO::loadEigenvector(sector_hdf5, i);
                    auto full_vec = hamiltonian.expandToComputationalBasis(
                        sector_idx, sym_vec);

                    std::string main_hdf5 = HDF5IO::createOrOpenFile(params.output_dir);
                    size_t global_id = sector_idx * 10000 + i;
                    HDF5IO::saveEigenvector(main_hdf5, global_id, full_vec);
                } catch (const std::exception& e) {
                    std::cerr << "  Warning: Could not expand eigenvector " << i
                              << " from sector " << sector_idx << ": "
                              << e.what() << std::endl;
                }
            }
            std::cout << "  Expanded " << sector_results.eigenvalues.size()
                      << " eigenvectors to full basis" << std::endl;
        }
        
        std::cout << "  Found " << sector_results.eigenvalues.size() << " eigenvalues" << std::endl;
        if (!sector_results.eigenvalues.empty()) {
            std::cout << "  Lowest: " << sector_results.eigenvalues[0] << std::endl;
        }
        std::cout << std::endl;
    }
    
    // ========== Step 5: Collect Results ==========
    std::sort(all_eigen_info.begin(), all_eigen_info.end());
    
    EDResults results;
    results.eigenvalues.resize(all_eigen_info.size());
    for (size_t i = 0; i < all_eigen_info.size(); ++i) {
        results.eigenvalues[i] = all_eigen_info[i].value;
    }
    
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Save combined eigenvalues to HDF5 (only when not using sector filter)
    if (!params.output_dir.empty() && !results.eigenvalues.empty() && sector_filter.empty()) {
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                     << " combined eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save combined eigenvalues: " << e.what() << std::endl;
        }
    }
    
    // Save eigenvalue mapping (fixed-Sz)
    if (!params.output_dir.empty()) {
        std::ofstream map_file(params.output_dir + "/eigenvalue_mapping.txt");
        if (map_file.is_open()) {
            map_file << "# Global_Index Eigenvalue Sector_Index Sector_Eigenvalue_Index";
            if (params.compute_eigenvectors) {
                map_file << " Eigenvector_Dataset";
            }
            map_file << "\n";
            for (size_t i = 0; i < all_eigen_info.size(); ++i) {
                const auto& info = all_eigen_info[i];
                map_file << i << " " << std::setprecision(15) << info.value << " "
                        << info.sector_idx << " " << info.eigen_idx;
                if (params.compute_eigenvectors) {
                    size_t global_id = info.sector_idx * 10000 + info.eigen_idx;
                    map_file << " eigenvector_" << global_id;
                }
                map_file << "\n";
            }
            map_file.close();
            std::cout << "Saved eigenvalue mapping to "
                     << params.output_dir << "/eigenvalue_mapping.txt" << std::endl;
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
