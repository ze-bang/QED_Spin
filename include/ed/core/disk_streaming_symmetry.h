#ifndef DISK_STREAMING_SYMMETRY_H
#define DISK_STREAMING_SYMMETRY_H

#include <ed/core/streaming_symmetry.h>
#include <fstream>
#include <filesystem>

/**
 * @file disk_streaming_symmetry.h
 * @brief Ultra-low-memory disk-based symmetry exact diagonalization
 * 
 * This implementation minimizes RAM usage by:
 * 1. Processing ONE sector at a time
 * 2. Saving sector data to disk immediately after generation
 * 3. Clearing memory before moving to next sector
 * 4. Loading only the needed sector for diagonalization
 * 
 * Memory complexity: O(largest_sector_dim × orbit_size) instead of O(total_basis × orbit_size)
 * Disk usage: O(total_orbit_elements) - binary format for efficiency
 * 
 * Trade-off: Slower due to disk I/O, but can handle much larger systems
 * 
 * Typical memory savings for 27-site kagome with 27 sectors:
 *   Standard streaming: ~10-15 GB (all sectors in memory)
 *   Disk streaming: ~0.5-1 GB (one sector at a time)
 */

// ============================================================================
// Disk-Based Sector Storage Format
// ============================================================================

/**
 * @brief Save a single sector to disk in binary format
 * 
 * Format (binary):
 *   - uint64_t: sector_id
 *   - uint64_t: num_quantum_numbers
 *   - int[num_quantum_numbers]: quantum_numbers
 *   - uint64_t: num_phase_factors
 *   - Complex[num_phase_factors]: phase_factors
 *   - uint64_t: num_basis_states
 *   For each basis state:
 *     - uint64_t: orbit_rep
 *     - double: norm
 *     - uint64_t: num_orbit_elements
 *     - uint64_t[num_orbit_elements]: orbit_elements
 *     - Complex[num_orbit_elements]: orbit_coefficients
 */
inline void saveSectorToDisk(const std::string& filepath, const SymmetrySector& sector) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open sector file for writing: " + filepath);
    }
    
    // Write sector metadata
    file.write(reinterpret_cast<const char*>(&sector.sector_id), sizeof(uint64_t));
    
    uint64_t num_qn = sector.quantum_numbers.size();
    file.write(reinterpret_cast<const char*>(&num_qn), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(sector.quantum_numbers.data()), 
               num_qn * sizeof(int));
    
    uint64_t num_pf = sector.phase_factors.size();
    file.write(reinterpret_cast<const char*>(&num_pf), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(sector.phase_factors.data()), 
               num_pf * sizeof(Complex));
    
    // Write basis states
    uint64_t num_states = sector.basis_states.size();
    file.write(reinterpret_cast<const char*>(&num_states), sizeof(uint64_t));
    
    for (const auto& state : sector.basis_states) {
        file.write(reinterpret_cast<const char*>(&state.orbit_rep), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(&state.norm), sizeof(double));
        
        uint64_t num_orbit = state.orbit_elements.size();
        file.write(reinterpret_cast<const char*>(&num_orbit), sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(state.orbit_elements.data()), 
                   num_orbit * sizeof(uint64_t));
        file.write(reinterpret_cast<const char*>(state.orbit_coefficients.data()), 
                   num_orbit * sizeof(Complex));
    }
    
    file.close();
}

/**
 * @brief Load a single sector from disk
 */
inline SymmetrySector loadSectorFromDisk(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open sector file for reading: " + filepath);
    }
    
    SymmetrySector sector;
    
    // Read sector metadata
    file.read(reinterpret_cast<char*>(&sector.sector_id), sizeof(uint64_t));
    
    uint64_t num_qn;
    file.read(reinterpret_cast<char*>(&num_qn), sizeof(uint64_t));
    sector.quantum_numbers.resize(num_qn);
    file.read(reinterpret_cast<char*>(sector.quantum_numbers.data()), 
              num_qn * sizeof(int));
    
    uint64_t num_pf;
    file.read(reinterpret_cast<char*>(&num_pf), sizeof(uint64_t));
    sector.phase_factors.resize(num_pf);
    file.read(reinterpret_cast<char*>(sector.phase_factors.data()), 
              num_pf * sizeof(Complex));
    
    // Read basis states
    uint64_t num_states;
    file.read(reinterpret_cast<char*>(&num_states), sizeof(uint64_t));
    sector.basis_states.resize(num_states);
    
    for (auto& state : sector.basis_states) {
        file.read(reinterpret_cast<char*>(&state.orbit_rep), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&state.norm), sizeof(double));
        state.quantum_numbers = sector.quantum_numbers;
        
        uint64_t num_orbit;
        file.read(reinterpret_cast<char*>(&num_orbit), sizeof(uint64_t));
        state.orbit_elements.resize(num_orbit);
        state.orbit_coefficients.resize(num_orbit);
        file.read(reinterpret_cast<char*>(state.orbit_elements.data()), 
                  num_orbit * sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(state.orbit_coefficients.data()), 
                  num_orbit * sizeof(Complex));
    }
    
    file.close();
    return sector;
}

// ============================================================================
// Disk-Based Streaming Symmetry Operator
// ============================================================================

/**
 * @brief Ultra-low-memory operator that stores sectors on disk
 * 
 * Only ONE sector is loaded into memory at a time.
 * This allows handling systems that would otherwise OOM.
 */
class DiskStreamingSymmetryOperator : public Operator {
private:
    std::string disk_cache_dir_;
    size_t num_sectors_ = 0;
    std::vector<uint64_t> sector_dimensions_;
    
    // Currently loaded sector (only one at a time!)
    mutable size_t loaded_sector_idx_ = SIZE_MAX;
    mutable SymmetrySector loaded_sector_;
    mutable std::unordered_map<uint64_t, size_t> loaded_sector_lookup_;
    
public:
    DiskStreamingSymmetryOperator(uint64_t n_bits, float spin_l) 
        : Operator(n_bits, spin_l) {}
    
    /**
     * @brief Generate all sectors, saving each to disk immediately
     * 
     * TWO-PASS ALGORITHM (cache-free, low memory):
     * Pass 1: Discover unique orbit representatives without caching
     * Pass 2: Build sectors from orbit representatives and save to disk
     * 
     * Memory usage during generation: O(num_orbits) instead of O(2^N)
     * After generation: O(1) - all data on disk
     */
    void generateSectorsToDisk(const std::string& dir) {
        std::cout << "\n=== Generating Sectors (Disk-Based Low-Memory Mode) ===" << std::endl;
        std::cout << "Using two-pass cache-free algorithm for minimal memory usage" << std::endl;
        
        // Create cache directory
        disk_cache_dir_ = dir + "/sector_cache";
        std::filesystem::create_directories(disk_cache_dir_);
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const size_t dim = 1ULL << n_bits_;
        const size_t group_size = symmetry_info.max_clique.size();
        num_sectors_ = symmetry_info.sectors.size();
        sector_dimensions_.resize(num_sectors_, 0);
        symmetrized_block_ham_sizes.resize(num_sectors_, 0);
        
        // ========== PASS 1: Discover orbit representatives (NO CACHING) ==========
        std::cout << "\n--- Pass 1: Discovering orbit representatives ---" << std::endl;
        std::cout << "Hilbert space dimension: " << dim << std::endl;
        std::cout << "Expected reduction factor: ~" << group_size << "x" << std::endl;
        
        std::vector<uint64_t> orbit_reps;
        orbit_reps.reserve(dim / group_size + 1000);
        
        uint64_t progress_interval = std::max(dim / 100, 1UL);
        
        for (uint64_t basis = 0; basis < dim; ++basis) {
            // Compute orbit representative WITHOUT caching
            uint64_t orbit_rep = computeOrbitRepNocache(basis);
            
            // Only keep if this IS the representative
            if (basis == orbit_rep) {
                orbit_reps.push_back(orbit_rep);
            }
            
            // Progress reporting
            if ((basis + 1) % progress_interval == 0 || basis == dim - 1) {
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                          << (100.0 * (basis + 1) / dim) << "%   " << std::flush;
            }
        }
        std::cout << std::endl;
        
        std::cout << "Found " << orbit_reps.size() << " unique orbit representatives" << std::endl;
        std::cout << "Memory for orbit reps: " 
                  << (orbit_reps.size() * sizeof(uint64_t)) / (1024.0 * 1024.0)
                  << " MB" << std::endl;
        
        // ========== PASS 2: Build sectors from orbit representatives ==========
        std::cout << "\n--- Pass 2: Building sectors ---" << std::endl;
        
        size_t total_orbit_elements = 0;
        size_t total_basis_states = 0;
        size_t max_sector_dim = 0;
        
        // Process each sector ONE AT A TIME
        for (size_t sector_idx = 0; sector_idx < num_sectors_; ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << num_sectors_ << " (QN: ";
            for (auto qn : sector_meta.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::flush;
            
            // Create sector in temporary memory
            SymmetrySector sector;
            sector.sector_id = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors = sector_meta.phase_factors;
            
            // For each orbit representative, check if it belongs to this sector
            for (uint64_t orbit_rep : orbit_reps) {
                std::vector<uint64_t> orbit_elements;
                std::vector<Complex> orbit_coefficients;
                double norm_sq = 0.0;
                
                computeOrbitData(orbit_rep, sector.phase_factors, 
                                orbit_elements, orbit_coefficients, norm_sq);
                
                if (norm_sq > 1e-10) {
                    SymBasisState state(orbit_rep, sector.quantum_numbers, std::sqrt(norm_sq));
                    state.orbit_elements = std::move(orbit_elements);
                    state.orbit_coefficients = std::move(orbit_coefficients);
                    
                    total_orbit_elements += state.orbit_elements.size();
                    sector.basis_states.push_back(std::move(state));
                }
            }
            
            uint64_t sector_dim = sector.basis_states.size();
            sector_dimensions_[sector_idx] = sector_dim;
            symmetrized_block_ham_sizes[sector_idx] = sector_dim;
            total_basis_states += sector_dim;
            max_sector_dim = std::max(max_sector_dim, static_cast<size_t>(sector_dim));
            
            std::cout << " -> " << sector_dim << " basis states";
            
            // SAVE TO DISK IMMEDIATELY
            std::string sector_file = disk_cache_dir_ + "/sector_" + std::to_string(sector_idx) + ".bin";
            saveSectorToDisk(sector_file, sector);
            std::cout << " [saved]" << std::endl;
            
            // CLEAR MEMORY - sector goes out of scope and is deallocated
        }
        
        // Clear orbit reps to free memory
        orbit_reps.clear();
        orbit_reps.shrink_to_fit();
        
        // Save metadata
        std::ofstream meta_file(disk_cache_dir_ + "/metadata.txt");
        meta_file << "num_sectors " << num_sectors_ << "\n";
        for (size_t i = 0; i < num_sectors_; ++i) {
            meta_file << "sector_" << i << "_dim " << sector_dimensions_[i] << "\n";
        }
        meta_file.close();
        
        std::cout << "\n=== Disk-Based Sector Generation Complete ===" << std::endl;
        std::cout << "Total sectors: " << num_sectors_ << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis_states << std::endl;
        std::cout << "Largest sector dimension: " << max_sector_dim << std::endl;
        std::cout << "Total orbit elements: " << total_orbit_elements << std::endl;
        std::cout << "Cache directory: " << disk_cache_dir_ << std::endl;
        
        // Estimate memory savings
        size_t full_memory_mb = (total_basis_states * group_size * (8 + 16)) / (1024 * 1024);
        size_t disk_memory_mb = (max_sector_dim * group_size * (8 + 16)) / (1024 * 1024);
        std::cout << "\nMemory comparison:" << std::endl;
        std::cout << "  All sectors in RAM: ~" << full_memory_mb << " MB" << std::endl;
        std::cout << "  Disk-based (one sector): ~" << disk_memory_mb << " MB" << std::endl;
        std::cout << "  Savings: ~" << (full_memory_mb - disk_memory_mb) << " MB" << std::endl;
    }
    
    /**
     * @brief Load a specific sector from disk
     */
    void loadSector(size_t sector_idx) const {
        if (sector_idx == loaded_sector_idx_) {
            return;  // Already loaded
        }
        
        if (sector_idx >= num_sectors_) {
            throw std::runtime_error("Invalid sector index: " + std::to_string(sector_idx));
        }
        
        // Clear previous sector data
        loaded_sector_.basis_states.clear();
        loaded_sector_.basis_states.shrink_to_fit();
        loaded_sector_lookup_.clear();
        
        // Load new sector
        std::string sector_file = disk_cache_dir_ + "/sector_" + std::to_string(sector_idx) + ".bin";
        loaded_sector_ = loadSectorFromDisk(sector_file);
        loaded_sector_idx_ = sector_idx;
        
        // Build lookup table for this sector
        for (size_t i = 0; i < loaded_sector_.basis_states.size(); ++i) {
            const auto& state = loaded_sector_.basis_states[i];
            for (uint64_t elem : state.orbit_elements) {
                loaded_sector_lookup_[elem] = i;
            }
        }
    }
    
    /**
     * @brief Apply Hamiltonian in currently loaded sector
     */
    void applySymmetrized(size_t sector_idx, const Complex* in, Complex* out) const {
        // Ensure sector is loaded
        loadSector(sector_idx);
        
        const size_t sector_dim = loaded_sector_.basis_states.size();
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));
        
        const double group_size = static_cast<double>(symmetry_info.max_clique.size());
        const double group_norm = 1.0 / group_size;
        
        // Process each input basis state
        for (size_t j = 0; j < sector_dim; ++j) {
            Complex c_j = in[j];
            if (std::abs(c_j) < 1e-15) continue;
            
            const auto& state_j = loaded_sector_.basis_states[j];
            const double norm_j = state_j.norm;
            
            // Iterate over orbit elements
            for (size_t orbit_idx = 0; orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                uint64_t s = state_j.orbit_elements[orbit_idx];
                Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                
                if (std::abs(alpha_s) < 1e-15) continue;
                
                // Apply Hamiltonian terms
                applyHamiltonianTerms(s, c_j * alpha_s / norm_j, group_norm, out);
            }
        }
    }
    
    size_t getNumSectors() const { return num_sectors_; }
    
    uint64_t getSectorDimension(size_t sector_idx) const {
        if (sector_idx >= num_sectors_) {
            throw std::runtime_error("Invalid sector index");
        }
        return sector_dimensions_[sector_idx];
    }
    
    const SymmetrySector& getSector(size_t sector_idx) const {
        loadSector(sector_idx);
        return loaded_sector_;
    }
    
    /**
     * @brief Unload current sector to free memory
     */
    void unloadCurrentSector() const {
        loaded_sector_.basis_states.clear();
        loaded_sector_.basis_states.shrink_to_fit();
        loaded_sector_lookup_.clear();
        loaded_sector_idx_ = SIZE_MAX;
    }
    
    /**
     * @brief Clean up disk cache
     */
    void cleanupDiskCache() {
        if (!disk_cache_dir_.empty() && std::filesystem::exists(disk_cache_dir_)) {
            std::filesystem::remove_all(disk_cache_dir_);
            std::cout << "Cleaned up disk cache: " << disk_cache_dir_ << std::endl;
        }
    }
    
private:
    mutable std::unordered_map<uint64_t, uint64_t> state_to_orbit_cache_;
    
    uint64_t getOrbitRepresentative(uint64_t basis) const {
        auto it = state_to_orbit_cache_.find(basis);
        if (it != state_to_orbit_cache_.end()) {
            return it->second;
        }
        
        uint64_t rep = basis;
        for (const auto& perm : symmetry_info.max_clique) {
            uint64_t permuted = applyPermutation(basis, perm);
            if (permuted < rep) rep = permuted;
        }
        
        state_to_orbit_cache_[basis] = rep;
        return rep;
    }
    
    /**
     * @brief Compute orbit representative WITHOUT caching (cache-free, O(|G|) time, O(1) memory)
     * 
     * This is essential for large systems where storing the cache would exceed memory.
     * Each state's orbit representative is computed on-the-fly in O(|G|) time.
     */
    uint64_t computeOrbitRepNocache(uint64_t basis) const {
        uint64_t rep = basis;
        for (const auto& perm : symmetry_info.max_clique) {
            uint64_t permuted = applyPermutation(basis, perm);
            if (permuted < rep) rep = permuted;
        }
        return rep;
    }
    
    void computeOrbitData(uint64_t basis, const std::vector<Complex>& phase_factors,
                         std::vector<uint64_t>& orbit_elements,
                         std::vector<Complex>& orbit_coefficients,
                         double& norm_sq) const {
        orbit_elements.clear();
        orbit_coefficients.clear();
        norm_sq = 0.0;
        
        std::unordered_map<uint64_t, Complex> orbit_map;
        
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            uint64_t permuted = applyPermutation(basis, perm);
            orbit_map[permuted] += std::conj(character);
        }
        
        for (const auto& [state, coeff] : orbit_map) {
            if (std::abs(coeff) > 1e-10) {
                orbit_elements.push_back(state);
                orbit_coefficients.push_back(coeff);
                norm_sq += std::norm(coeff);
            }
        }
        
        // Critical: normalize by group size (same as streaming_symmetry.h)
        norm_sq /= symmetry_info.max_clique.size();
    }
    
    void applyHamiltonianTerms(uint64_t s, Complex prefactor, double group_norm,
                               Complex* out) const {
        // Apply all Hamiltonian terms using the transform_data_ storage
        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            if (!tdata.is_two_body) {
                // One-body term
                if (tdata.op_type == 2) {  // Sz
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {  // S+ or S-
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                // Two-body term
                if (tdata.op_type == 2) {  // Sz_i
                    double sign = ((s_prime >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
                
                if (valid) {
                    if (tdata.op_type_2 == 2) {  // Sz_j
                        double sign = ((s_prime >> tdata.site_index_2) & 1) ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign;
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
                // Project s_prime back to symmetrized basis
                auto it = loaded_sector_lookup_.find(s_prime);
                if (it != loaded_sector_lookup_.end()) {
                    size_t k = it->second;
                    const auto& state_k = loaded_sector_.basis_states[k];
                    
                    // Find coefficient of s_prime in state_k
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
};

// ============================================================================
// Disk-Based ED Wrapper Function
// ============================================================================

/**
 * @brief Ultra-low-memory ED using disk-based sector storage
 * 
 * This function processes sectors one at a time, keeping memory usage
 * proportional to the LARGEST sector rather than ALL sectors combined.
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method
 * @param params ED parameters
 * @return EDResults with eigenvalues
 */
inline EDResults exact_diagonalization_disk_streaming(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat"
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Disk-Based Low-Memory Symmetry ED" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Generate automorphisms if needed
    if (!generate_automorphisms(directory)) {
        std::cerr << "Warning: Automorphism generation failed" << std::endl;
        return EDResults();
    }
    
    // Load Hamiltonian
    std::cout << "\nLoading Hamiltonian..." << std::endl;
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    
    DiskStreamingSymmetryOperator hamiltonian(params.num_sites, params.spin_length);
    hamiltonian.loadFromFile(single_site_file);
    hamiltonian.loadFromInterAllFile(interaction_file);
    
    // Generate sectors to disk (low memory!)
    std::cout << "\nGenerating symmetry sectors to disk..." << std::endl;
    hamiltonian.generateSectorsToDisk(directory);
    
    size_t num_sectors = hamiltonian.getNumSectors();
    
    // Diagonalize each sector ONE AT A TIME
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
        
        // Load this sector from disk
        hamiltonian.loadSector(sector_idx);
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
            hamiltonian.applySymmetrized(sector_idx, in, out);
        };
        
        // Diagonalize - FORCE CPU LANCZOS (GPU doesn't support matrix-free operations)
        EDParameters sector_params = params;
        sector_params.num_eigenvalues = std::min(params.num_eigenvalues, sector_dim);
        
        // Override method to CPU Lanczos since disk-streaming uses matrix-free matvec
        DiagonalizationMethod actual_method = DiagonalizationMethod::LANCZOS;
        if (method == DiagonalizationMethod::LANCZOS || 
            method == DiagonalizationMethod::DAVIDSON ||
            method == DiagonalizationMethod::ARPACK_SM ||
            method == DiagonalizationMethod::ARPACK_LM ||
            method == DiagonalizationMethod::ARPACK_SHIFT_INVERT ||
            method == DiagonalizationMethod::ARPACK_ADVANCED) {
            actual_method = method;  // Use requested CPU method
        } else {
            std::cout << "  Note: Using CPU Lanczos (disk-streaming requires matrix-free ops)" << std::endl;
        }
        
        EDResults sector_results = exact_diagonalization_core(
            matvec, sector_dim, actual_method, sector_params
        );
        
        // Store eigenvalues (small, stays in memory)
        for (size_t i = 0; i < sector_results.eigenvalues.size(); ++i) {
            all_eigen_info.push_back({
                sector_results.eigenvalues[i], 
                sector_idx, 
                static_cast<uint64_t>(i)
            });
        }
        
        std::cout << "  Lowest: " << sector_results.eigenvalues[0] << std::endl;
        
        // UNLOAD SECTOR TO FREE MEMORY before next iteration
        hamiltonian.unloadCurrentSector();
        std::cout << "  [Memory freed]" << std::endl << std::endl;
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
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save eigenvalues: " << e.what() << std::endl;
        }
    }
    
    // Clean up disk cache (optional - can keep for debugging)
    // hamiltonian.cleanupDiskCache();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Disk-Based ED Complete" << std::endl;
    std::cout << "  Ground state energy: " << results.eigenvalues[0] << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return results;
}

#endif // DISK_STREAMING_SYMMETRY_H
