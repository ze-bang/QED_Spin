#ifndef CHUNKED_SYMMETRY_BUILDER_H
#define CHUNKED_SYMMETRY_BUILDER_H

#include <ed/core/streaming_symmetry.h>
#include <ed/core/disk_streaming_symmetry.h>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <bitset>

/**
 * @file chunked_symmetry_builder.h
 * @brief Ultra-low-memory chunked symmetry sector generation
 * 
 * PROBLEM:
 * The standard symmetry sector generation iterates over all 2^N states (or all
 * fixed-Sz states) while maintaining:
 *   - state_to_orbit_cache_: O(2^N) entries for full space
 *   - processed_orbits set: O(num_orbits) entries
 *   - All sector data simultaneously
 * 
 * For N=30+, this requires 10+ GB of RAM just for the cache.
 * 
 * SOLUTION: Two-pass chunked algorithm
 * 
 * Pass 1: "Discovery" - Find all unique orbit representatives
 *   - Process states in chunks (e.g., 1M states at a time)
 *   - For each state, compute orbit rep WITHOUT caching
 *   - Keep only unique representatives in a sorted vector
 *   - Memory: O(num_unique_orbits) which is ~2^N / |G|
 * 
 * Pass 2: "Build" - Generate orbit data for each representative
 *   - For each unique orbit representative, compute full orbit data
 *   - Assign to appropriate sector based on character
 *   - Save sectors to disk one at a time
 *   - Memory: O(largest_sector_size)
 * 
 * Key insight: The orbit representative of a state can be computed in O(|G|) time
 * without needing a cache. The cache was an optimization that becomes a liability
 * for large systems.
 */

// ============================================================================
// Helper Functions
// ============================================================================

namespace chunked_symmetry {

/**
 * @brief Compute orbit representative WITHOUT caching
 * 
 * This is O(|G|) per call, but uses O(1) memory.
 * For large systems, the memory savings outweigh the time cost.
 */
inline uint64_t computeOrbitRepNocache(
    uint64_t basis,
    const std::vector<std::vector<int>>& group_elements,
    uint64_t n_bits
) {
    uint64_t rep = basis;
    for (const auto& perm : group_elements) {
        // Apply permutation
        uint64_t permuted = 0;
        for (size_t i = 0; i < n_bits; ++i) {
            if ((basis >> i) & 1) {
                permuted |= (1ULL << perm[i]);
            }
        }
        if (permuted < rep) {
            rep = permuted;
        }
    }
    return rep;
}

/**
 * @brief Compute orbit representative for fixed-Sz basis (combinadic iteration)
 */
inline uint64_t computeOrbitRepFixedSzNocache(
    uint64_t basis,
    const std::vector<std::vector<int>>& group_elements,
    uint64_t n_bits
) {
    return computeOrbitRepNocache(basis, group_elements, n_bits);
}

/**
 * @brief Generate next fixed-Sz state using Gosper's hack
 * 
 * This generates states with exactly k bits set in lexicographic order.
 */
inline uint64_t nextFixedSzState(uint64_t v) {
    if (v == 0) return 0;
    uint64_t t = v | (v - 1);
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctzll(v) + 1));
}

/**
 * @brief Compute first state with k bits set
 */
inline uint64_t firstFixedSzState(int k) {
    return (1ULL << k) - 1;
}

/**
 * @brief Check if state is the last fixed-Sz state for given n, k
 */
inline bool isLastFixedSzState(uint64_t state, uint64_t n_bits, int k) {
    // Last state has all k bits in the highest positions
    uint64_t last_state = ((1ULL << k) - 1) << (n_bits - k);
    return state >= last_state;
}

/**
 * @brief Count number of states with k bits set in n bits
 */
inline uint64_t binomialCoeff(uint64_t n, uint64_t k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;
    
    uint64_t result = 1;
    for (uint64_t i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

} // namespace chunked_symmetry

// ============================================================================
// Chunked Symmetry Builder for Full Hilbert Space
// ============================================================================

/**
 * @brief Ultra-low-memory chunked symmetry builder
 * 
 * Two-pass algorithm:
 * 1. Discover unique orbit representatives (streamed, O(num_orbits) memory)
 * 2. Build sectors one at a time (O(sector_size) memory)
 */
class ChunkedSymmetryBuilder {
private:
    uint64_t n_bits_;
    float spin_l_;
    SymmetryGroupInfo symmetry_info;
    std::string cache_dir_;
    
    // Discovered orbit representatives (sorted)
    std::vector<uint64_t> orbit_reps_;
    
    // Memory limit for chunk processing (default 100 MB worth of states)
    static constexpr size_t DEFAULT_CHUNK_SIZE = 100 * 1024 * 1024 / sizeof(uint64_t);
    
public:
    ChunkedSymmetryBuilder(uint64_t n_bits, float spin_l)
        : n_bits_(n_bits), spin_l_(spin_l) {}
    
    /**
     * @brief Pass 1: Discover all unique orbit representatives
     * 
     * This processes the Hilbert space in chunks to find unique orbits.
     * Memory usage: O(num_unique_orbits)
     * Time: O(2^N × |G|) - no caching but low memory
     * 
     * @param dir Directory containing symmetry data
     * @param chunk_size Number of states to process per chunk
     */
    void discoverOrbits(const std::string& dir, size_t chunk_size = DEFAULT_CHUNK_SIZE) {
        std::cout << "\n=== Pass 1: Discovering Orbit Representatives ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const uint64_t dim = 1ULL << n_bits_;
        const size_t group_size = symmetry_info.max_clique.size();
        
        std::cout << "Hilbert space dimension: " << dim << std::endl;
        std::cout << "Symmetry group size: " << group_size << std::endl;
        std::cout << "Expected reduction factor: ~" << group_size << "x" << std::endl;
        std::cout << "Processing in chunks of " << chunk_size << " states" << std::endl;
        
        // Use a sorted vector to collect unique representatives
        // This uses less memory than unordered_set for large numbers
        std::vector<uint64_t> local_reps;
        local_reps.reserve(dim / group_size + 1000);  // Estimate with buffer
        
        size_t num_chunks = (dim + chunk_size - 1) / chunk_size;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            uint64_t chunk_start = chunk_idx * chunk_size;
            uint64_t chunk_end = std::min(chunk_start + chunk_size, dim);
            
            if (chunk_idx % 10 == 0 || chunk_idx == num_chunks - 1) {
                std::cout << "\rProcessing chunk " << (chunk_idx + 1) << "/" << num_chunks
                          << " (" << std::fixed << std::setprecision(1)
                          << (100.0 * chunk_end / dim) << "%)   " << std::flush;
            }
            
            // Process this chunk - find orbit reps WITHOUT caching
            for (uint64_t basis = chunk_start; basis < chunk_end; ++basis) {
                uint64_t orbit_rep = chunked_symmetry::computeOrbitRepNocache(
                    basis, symmetry_info.max_clique, n_bits_);
                
                // Only keep if this is the representative (i.e., basis == orbit_rep)
                // This way we only store representatives, not all states
                if (basis == orbit_rep) {
                    local_reps.push_back(orbit_rep);
                }
            }
        }
        
        std::cout << std::endl;
        
        // Sort and deduplicate (though if we only kept basis==orbit_rep, already unique)
        std::sort(local_reps.begin(), local_reps.end());
        local_reps.erase(std::unique(local_reps.begin(), local_reps.end()), local_reps.end());
        
        orbit_reps_ = std::move(local_reps);
        orbit_reps_.shrink_to_fit();
        
        std::cout << "Found " << orbit_reps_.size() << " unique orbit representatives" << std::endl;
        std::cout << "Memory for representatives: " 
                  << (orbit_reps_.size() * sizeof(uint64_t)) / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }
    
    /**
     * @brief Pass 2: Build sectors from discovered orbit representatives
     * 
     * For each sector, iterate through orbit representatives and assign those
     * whose character is non-zero.
     * 
     * @param dir Directory to save sectors
     */
    void buildSectors(const std::string& dir) {
        std::cout << "\n=== Pass 2: Building Sectors ===" << std::endl;
        
        if (orbit_reps_.empty()) {
            throw std::runtime_error("No orbit representatives discovered. Call discoverOrbits first.");
        }
        
        cache_dir_ = dir + "/sector_cache_chunked";
        std::filesystem::create_directories(cache_dir_);
        
        size_t num_sectors = symmetry_info.sectors.size();
        std::cout << "Building " << num_sectors << " sectors" << std::endl;
        
        std::vector<uint64_t> sector_dimensions(num_sectors, 0);
        size_t total_basis_states = 0;
        size_t total_orbit_elements = 0;
        
        // Process each sector
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nSector " << (sector_idx + 1) << "/" << num_sectors
                      << " (QN: ";
            for (auto qn : sector_meta.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::flush;
            
            SymmetrySector sector;
            sector.sector_id = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors = sector_meta.phase_factors;
            
            // For each orbit representative, check if it belongs to this sector
            for (uint64_t orbit_rep : orbit_reps_) {
                // Compute full orbit data for this representative
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
            
            sector_dimensions[sector_idx] = sector.basis_states.size();
            total_basis_states += sector.basis_states.size();
            
            std::cout << " -> " << sector.basis_states.size() << " basis states";
            
            // Save sector to disk
            std::string sector_file = cache_dir_ + "/sector_" + std::to_string(sector_idx) + ".bin";
            saveSectorToDisk(sector_file, sector);
            std::cout << " [saved]" << std::endl;
            
            // Sector goes out of scope - memory freed
        }
        
        // Save metadata
        std::ofstream meta_file(cache_dir_ + "/metadata.txt");
        meta_file << "num_sectors " << num_sectors << "\n";
        meta_file << "total_basis " << total_basis_states << "\n";
        meta_file << "total_orbits " << total_orbit_elements << "\n";
        for (size_t i = 0; i < num_sectors; ++i) {
            meta_file << "sector_" << i << "_dim " << sector_dimensions[i] << "\n";
        }
        meta_file.close();
        
        // Clear orbit reps to free memory
        orbit_reps_.clear();
        orbit_reps_.shrink_to_fit();
        
        std::cout << "\n=== Chunked Sector Building Complete ===" << std::endl;
        std::cout << "Total sectors: " << num_sectors << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis_states << std::endl;
        std::cout << "Cache directory: " << cache_dir_ << std::endl;
    }
    
    /**
     * @brief Get cache directory path
     */
    const std::string& getCacheDir() const { return cache_dir_; }
    
private:
    /**
     * @brief Compute orbit data for a given basis state
     */
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
            
            // Compute character for this group element
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            // Apply permutation to basis state
            uint64_t permuted = 0;
            for (size_t i = 0; i < n_bits_; ++i) {
                if ((basis >> i) & 1) {
                    permuted |= (1ULL << perm[i]);
                }
            }
            
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
};

// ============================================================================
// Chunked Symmetry Builder for Fixed-Sz Sector
// ============================================================================

/**
 * @brief Ultra-low-memory chunked symmetry builder for fixed-Sz sector
 * 
 * This is more memory-efficient than the full Hilbert space version since
 * we only iterate over C(N, k) states instead of 2^N.
 * 
 * For N=36, Sz=0 (k=18):
 *   2^36 = 68 billion states
 *   C(36,18) = 9 billion states (still large, but much better)
 * 
 * Further optimizations:
 * - Use Gosper's hack to iterate fixed-Sz states without storing them
 * - Process in chunks to bound memory usage
 */
class ChunkedSymmetryBuilderFixedSz {
private:
    uint64_t n_bits_;
    float spin_l_;
    int64_t n_up_;
    SymmetryGroupInfo symmetry_info;
    std::string cache_dir_;
    
    // Discovered orbit representatives (sorted)
    std::vector<uint64_t> orbit_reps_;
    
public:
    ChunkedSymmetryBuilderFixedSz(uint64_t n_bits, float spin_l, int64_t n_up)
        : n_bits_(n_bits), spin_l_(spin_l), n_up_(n_up) {}
    
    /**
     * @brief Pass 1: Discover all unique orbit representatives in fixed-Sz sector
     * 
     * Uses Gosper's hack to iterate through fixed-Sz states without
     * storing all of them in memory.
     * 
     * Memory usage: O(num_unique_orbits in fixed-Sz sector)
     */
    void discoverOrbits(const std::string& dir) {
        std::cout << "\n=== Pass 1: Discovering Orbit Representatives (Fixed Sz) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const uint64_t fixed_sz_dim = chunked_symmetry::binomialCoeff(n_bits_, n_up_);
        const size_t group_size = symmetry_info.max_clique.size();
        
        std::cout << "Fixed-Sz dimension: " << fixed_sz_dim 
                  << " (N=" << n_bits_ << ", n_up=" << n_up_ << ")" << std::endl;
        std::cout << "Symmetry group size: " << group_size << std::endl;
        std::cout << "Expected reduction factor: ~" << group_size << "x" << std::endl;
        
        // Reserve space for orbit representatives
        orbit_reps_.reserve(fixed_sz_dim / group_size + 1000);
        
        // Iterate through fixed-Sz states using Gosper's hack
        uint64_t state = chunked_symmetry::firstFixedSzState(n_up_);
        uint64_t max_state = 1ULL << n_bits_;
        uint64_t count = 0;
        uint64_t progress_interval = std::max(fixed_sz_dim / 100, 1UL);
        
        while (state < max_state) {
            // Compute orbit representative
            uint64_t orbit_rep = chunked_symmetry::computeOrbitRepNocache(
                state, symmetry_info.max_clique, n_bits_);
            
            // Only keep if this is the representative
            if (state == orbit_rep) {
                orbit_reps_.push_back(orbit_rep);
            }
            
            // Progress reporting
            if (++count % progress_interval == 0) {
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1)
                          << (100.0 * count / fixed_sz_dim) << "%   " << std::flush;
            }
            
            // Get next fixed-Sz state
            state = chunked_symmetry::nextFixedSzState(state);
        }
        
        std::cout << "\rProgress: 100.0%   " << std::endl;
        
        // Sort (already mostly sorted due to iteration order)
        std::sort(orbit_reps_.begin(), orbit_reps_.end());
        orbit_reps_.shrink_to_fit();
        
        std::cout << "Found " << orbit_reps_.size() << " unique orbit representatives" << std::endl;
        std::cout << "Memory for representatives: " 
                  << (orbit_reps_.size() * sizeof(uint64_t)) / (1024.0 * 1024.0)
                  << " MB" << std::endl;
    }
    
    /**
     * @brief Pass 2: Build sectors from discovered orbit representatives
     */
    void buildSectors(const std::string& dir) {
        std::cout << "\n=== Pass 2: Building Sectors (Fixed Sz) ===" << std::endl;
        
        if (orbit_reps_.empty()) {
            throw std::runtime_error("No orbit representatives discovered. Call discoverOrbits first.");
        }
        
        cache_dir_ = dir + "/sector_cache_chunked_fixed_sz";
        std::filesystem::create_directories(cache_dir_);
        
        size_t num_sectors = symmetry_info.sectors.size();
        std::cout << "Building " << num_sectors << " sectors" << std::endl;
        
        std::vector<uint64_t> sector_dimensions(num_sectors, 0);
        size_t total_basis_states = 0;
        size_t total_orbit_elements = 0;
        
        // Process each sector
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nSector " << (sector_idx + 1) << "/" << num_sectors
                      << " (QN: ";
            for (auto qn : sector_meta.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::flush;
            
            SymmetrySector sector;
            sector.sector_id = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors = sector_meta.phase_factors;
            
            // For each orbit representative, check if it belongs to this sector
            for (uint64_t orbit_rep : orbit_reps_) {
                // Compute full orbit data for this representative
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
            
            sector_dimensions[sector_idx] = sector.basis_states.size();
            total_basis_states += sector.basis_states.size();
            
            std::cout << " -> " << sector.basis_states.size() << " basis states";
            
            // Save sector to disk
            std::string sector_file = cache_dir_ + "/sector_" + std::to_string(sector_idx) + ".bin";
            saveSectorToDisk(sector_file, sector);
            std::cout << " [saved]" << std::endl;
        }
        
        // Save metadata
        std::ofstream meta_file(cache_dir_ + "/metadata.txt");
        meta_file << "num_sectors " << num_sectors << "\n";
        meta_file << "total_basis " << total_basis_states << "\n";
        meta_file << "n_bits " << n_bits_ << "\n";
        meta_file << "n_up " << n_up_ << "\n";
        for (size_t i = 0; i < num_sectors; ++i) {
            meta_file << "sector_" << i << "_dim " << sector_dimensions[i] << "\n";
        }
        meta_file.close();
        
        // Clear orbit reps to free memory
        orbit_reps_.clear();
        orbit_reps_.shrink_to_fit();
        
        std::cout << "\n=== Chunked Sector Building Complete ===" << std::endl;
        std::cout << "Total sectors: " << num_sectors << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis_states << std::endl;
        std::cout << "Cache directory: " << cache_dir_ << std::endl;
    }
    
    /**
     * @brief Combined discovery and build in one call
     */
    void generateSectors(const std::string& dir) {
        discoverOrbits(dir);
        buildSectors(dir);
    }
    
    const std::string& getCacheDir() const { return cache_dir_; }
    
private:
    /**
     * @brief Compute orbit data for a given basis state
     */
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
            
            // Compute character for this group element
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            // Apply permutation to basis state
            uint64_t permuted = 0;
            for (size_t i = 0; i < n_bits_; ++i) {
                if ((basis >> i) & 1) {
                    permuted |= (1ULL << perm[i]);
                }
            }
            
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
};

// ============================================================================
// Even More Memory-Efficient: Parallel Chunked with Disk-Based Merge
// ============================================================================

/**
 * @brief Parallel chunked orbit discovery with disk-based merging
 * 
 * For truly massive systems, even storing all orbit representatives in memory
 * may be too much. This version:
 * 1. Splits the Hilbert space into chunks
 * 2. Each chunk discovers its orbit reps and writes to a temp file
 * 3. External merge-sort combines all temp files
 * 4. Final pass builds sectors from disk-stored reps
 * 
 * Memory: O(chunk_size) during discovery, O(merge_buffer) during merge
 * Disk: O(num_orbits × sizeof(uint64_t))
 */
class DiskBasedChunkedSymmetryBuilder {
private:
    uint64_t n_bits_;
    float spin_l_;
    SymmetryGroupInfo symmetry_info;
    std::string work_dir_;
    std::string orbit_file_;  // File containing sorted orbit representatives
    uint64_t num_orbits_ = 0;
    
public:
    DiskBasedChunkedSymmetryBuilder(uint64_t n_bits, float spin_l)
        : n_bits_(n_bits), spin_l_(spin_l) {}
    
    /**
     * @brief Discover orbits using disk-based chunked algorithm
     * 
     * @param dir Base directory
     * @param chunk_size Number of states per chunk (controls memory usage)
     * @param num_parallel_chunks Number of chunks to process in parallel
     */
    void discoverOrbitsDiskBased(const std::string& dir, 
                                  size_t chunk_size = 10000000,  // 10M states
                                  size_t num_parallel_chunks = 1) {
        std::cout << "\n=== Disk-Based Orbit Discovery ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        work_dir_ = dir + "/chunked_work";
        std::filesystem::create_directories(work_dir_);
        
        const uint64_t dim = 1ULL << n_bits_;
        const size_t num_chunks = (dim + chunk_size - 1) / chunk_size;
        
        std::cout << "Hilbert space: " << dim << " states" << std::endl;
        std::cout << "Chunk size: " << chunk_size << " states" << std::endl;
        std::cout << "Number of chunks: " << num_chunks << std::endl;
        
        // Phase 1: Process chunks and write temporary files
        std::cout << "\nPhase 1: Processing chunks..." << std::endl;
        
        std::vector<std::string> temp_files;
        
        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            uint64_t chunk_start = chunk_idx * chunk_size;
            uint64_t chunk_end = std::min(chunk_start + chunk_size, dim);
            
            std::cout << "\rChunk " << (chunk_idx + 1) << "/" << num_chunks 
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * chunk_end / dim) << "%)   " << std::flush;
            
            // Find orbit representatives in this chunk
            std::vector<uint64_t> chunk_reps;
            chunk_reps.reserve((chunk_end - chunk_start) / symmetry_info.max_clique.size() + 100);
            
            for (uint64_t basis = chunk_start; basis < chunk_end; ++basis) {
                uint64_t orbit_rep = chunked_symmetry::computeOrbitRepNocache(
                    basis, symmetry_info.max_clique, n_bits_);
                
                if (basis == orbit_rep) {
                    chunk_reps.push_back(orbit_rep);
                }
            }
            
            // Sort chunk
            std::sort(chunk_reps.begin(), chunk_reps.end());
            
            // Write to temp file
            std::string temp_file = work_dir_ + "/chunk_" + std::to_string(chunk_idx) + ".bin";
            std::ofstream out(temp_file, std::ios::binary);
            uint64_t num_reps = chunk_reps.size();
            out.write(reinterpret_cast<const char*>(&num_reps), sizeof(uint64_t));
            out.write(reinterpret_cast<const char*>(chunk_reps.data()), 
                     num_reps * sizeof(uint64_t));
            out.close();
            
            temp_files.push_back(temp_file);
        }
        std::cout << std::endl;
        
        // Phase 2: Merge sorted files
        std::cout << "\nPhase 2: Merging sorted chunks..." << std::endl;
        orbit_file_ = work_dir_ + "/all_orbits.bin";
        mergeSortedFiles(temp_files, orbit_file_);
        
        // Count orbits
        std::ifstream count_file(orbit_file_, std::ios::binary);
        count_file.read(reinterpret_cast<char*>(&num_orbits_), sizeof(uint64_t));
        count_file.close();
        
        std::cout << "Total unique orbit representatives: " << num_orbits_ << std::endl;
        
        // Clean up temp files
        for (const auto& tf : temp_files) {
            std::filesystem::remove(tf);
        }
    }
    
    /**
     * @brief Build sectors from disk-stored orbit representatives
     */
    void buildSectorsFromDisk(const std::string& dir) {
        std::cout << "\n=== Building Sectors from Disk ===" << std::endl;
        
        if (orbit_file_.empty() || num_orbits_ == 0) {
            throw std::runtime_error("No orbits discovered. Call discoverOrbitsDiskBased first.");
        }
        
        std::string cache_dir = dir + "/sector_cache_disk_chunked";
        std::filesystem::create_directories(cache_dir);
        
        size_t num_sectors = symmetry_info.sectors.size();
        std::cout << "Building " << num_sectors << " sectors from " 
                  << num_orbits_ << " orbit representatives" << std::endl;
        
        // Open orbit file for streaming read
        std::ifstream orbit_in(orbit_file_, std::ios::binary);
        uint64_t stored_num_orbits;
        orbit_in.read(reinterpret_cast<char*>(&stored_num_orbits), sizeof(uint64_t));
        
        std::vector<uint64_t> sector_dimensions(num_sectors, 0);
        std::vector<std::vector<SymBasisState>> sector_states(num_sectors);
        
        // Read orbits in chunks and assign to sectors
        constexpr size_t READ_CHUNK = 100000;
        std::vector<uint64_t> orbit_buffer(READ_CHUNK);
        uint64_t orbits_read = 0;
        
        while (orbits_read < stored_num_orbits) {
            size_t to_read = std::min(READ_CHUNK, static_cast<size_t>(stored_num_orbits - orbits_read));
            orbit_in.read(reinterpret_cast<char*>(orbit_buffer.data()), 
                         to_read * sizeof(uint64_t));
            
            // Process each orbit
            for (size_t i = 0; i < to_read; ++i) {
                uint64_t orbit_rep = orbit_buffer[i];
                
                // Try each sector
                for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
                    const auto& sector_meta = symmetry_info.sectors[sector_idx];
                    
                    std::vector<uint64_t> orbit_elements;
                    std::vector<Complex> orbit_coefficients;
                    double norm_sq = 0.0;
                    
                    computeOrbitData(orbit_rep, sector_meta.phase_factors,
                                   orbit_elements, orbit_coefficients, norm_sq);
                    
                    if (norm_sq > 1e-10) {
                        SymBasisState state(orbit_rep, sector_meta.quantum_numbers, std::sqrt(norm_sq));
                        state.orbit_elements = std::move(orbit_elements);
                        state.orbit_coefficients = std::move(orbit_coefficients);
                        sector_states[sector_idx].push_back(std::move(state));
                        break;  // Each orbit belongs to exactly one sector
                    }
                }
            }
            
            orbits_read += to_read;
            
            if (orbits_read % 1000000 == 0 || orbits_read == stored_num_orbits) {
                std::cout << "\rProcessed " << orbits_read << "/" << stored_num_orbits 
                          << " orbits" << std::flush;
            }
        }
        orbit_in.close();
        std::cout << std::endl;
        
        // Save sectors to disk
        size_t total_basis = 0;
        for (size_t sector_idx = 0; sector_idx < num_sectors; ++sector_idx) {
            SymmetrySector sector;
            sector.sector_id = symmetry_info.sectors[sector_idx].sector_id;
            sector.quantum_numbers = symmetry_info.sectors[sector_idx].quantum_numbers;
            sector.phase_factors = symmetry_info.sectors[sector_idx].phase_factors;
            sector.basis_states = std::move(sector_states[sector_idx]);
            
            sector_dimensions[sector_idx] = sector.basis_states.size();
            total_basis += sector.basis_states.size();
            
            std::string sector_file = cache_dir + "/sector_" + std::to_string(sector_idx) + ".bin";
            saveSectorToDisk(sector_file, sector);
            
            std::cout << "Sector " << (sector_idx + 1) << "/" << num_sectors
                      << ": " << sector.basis_states.size() << " states [saved]" << std::endl;
        }
        
        // Save metadata
        std::ofstream meta_file(cache_dir + "/metadata.txt");
        meta_file << "num_sectors " << num_sectors << "\n";
        meta_file << "total_basis " << total_basis << "\n";
        for (size_t i = 0; i < num_sectors; ++i) {
            meta_file << "sector_" << i << "_dim " << sector_dimensions[i] << "\n";
        }
        meta_file.close();
        
        std::cout << "\nTotal symmetrized basis: " << total_basis << std::endl;
    }
    
private:
    /**
     * @brief K-way merge of sorted files
     */
    void mergeSortedFiles(const std::vector<std::string>& input_files,
                          const std::string& output_file) {
        struct FileReader {
            std::ifstream file;
            uint64_t remaining;
            uint64_t current;
            bool valid;
            
            bool readNext() {
                if (remaining == 0) {
                    valid = false;
                    return false;
                }
                file.read(reinterpret_cast<char*>(&current), sizeof(uint64_t));
                remaining--;
                valid = true;
                return true;
            }
        };
        
        // Open all input files
        std::vector<FileReader> readers(input_files.size());
        for (size_t i = 0; i < input_files.size(); ++i) {
            readers[i].file.open(input_files[i], std::ios::binary);
            readers[i].file.read(reinterpret_cast<char*>(&readers[i].remaining), sizeof(uint64_t));
            readers[i].readNext();
        }
        
        // Open output file
        std::ofstream out(output_file, std::ios::binary);
        uint64_t placeholder = 0;
        out.write(reinterpret_cast<const char*>(&placeholder), sizeof(uint64_t));  // Will update later
        
        // K-way merge with deduplication
        uint64_t num_written = 0;
        uint64_t last_written = UINT64_MAX;
        
        while (true) {
            // Find minimum among all readers
            uint64_t min_val = UINT64_MAX;
            int min_idx = -1;
            
            for (size_t i = 0; i < readers.size(); ++i) {
                if (readers[i].valid && readers[i].current < min_val) {
                    min_val = readers[i].current;
                    min_idx = static_cast<int>(i);
                }
            }
            
            if (min_idx < 0) break;  // All files exhausted
            
            // Write if not duplicate
            if (min_val != last_written) {
                out.write(reinterpret_cast<const char*>(&min_val), sizeof(uint64_t));
                last_written = min_val;
                num_written++;
            }
            
            // Advance the reader that had the minimum
            readers[min_idx].readNext();
        }
        
        // Update count at beginning of file
        out.seekp(0);
        out.write(reinterpret_cast<const char*>(&num_written), sizeof(uint64_t));
        out.close();
        
        // Close input files
        for (auto& r : readers) {
            r.file.close();
        }
        
        std::cout << "Merged " << input_files.size() << " files into " 
                  << num_written << " unique orbits" << std::endl;
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
            
            uint64_t permuted = 0;
            for (size_t i = 0; i < n_bits_; ++i) {
                if ((basis >> i) & 1) {
                    permuted |= (1ULL << perm[i]);
                }
            }
            
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
};

#endif // CHUNKED_SYMMETRY_BUILDER_H
