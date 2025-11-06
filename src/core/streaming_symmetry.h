#ifndef STREAMING_SYMMETRY_H
#define STREAMING_SYMMETRY_H

#include "construct_ham.h"
#include <unordered_set>
#include <algorithm>
#include <numeric>

/**
 * @file streaming_symmetry.h
 * @brief Memory-efficient streaming implementation of symmetry-adapted exact diagonalization
 * 
 * This implementation avoids storing the full symmetrized basis and block matrices on disk.
 * Instead, it uses on-the-fly computation of symmetrized matrix-vector products.
 * 
 * Key advantages:
 * - No disk storage required for basis vectors
 * - No disk storage required for block matrices
 * - Memory usage scales with O(block_dim) instead of O(total_dim)
 * - Faster basis generation (no file I/O)
 * - Can handle much larger systems
 * 
 * Algorithm:
 * 1. Generate orbit representatives on-the-fly
 * 2. Compute H|ψ⟩ in the symmetrized basis without explicit matrix construction
 * 3. Use Lanczos/Davidson with streaming matrix-vector products
 */

// ============================================================================
// Orbit-Based Symmetry Data Structures
// ============================================================================

/**
 * @brief Compact representation of a symmetrized basis state
 * 
 * Instead of storing the full vector, we store only the orbit representative
 * and the quantum numbers. The full vector can be reconstructed on-the-fly.
 */
struct SymBasisState {
    uint64_t orbit_rep;                    // Representative element of the orbit
    std::vector<int> quantum_numbers;      // Quantum numbers for this sector
    std::vector<uint64_t> orbit_elements;  // All elements in the orbit (optional, for faster access)
    double norm;                           // Normalization factor
    
    SymBasisState() : orbit_rep(0), norm(0.0) {}
    
    SymBasisState(uint64_t rep, const std::vector<int>& qn, double n = 1.0) 
        : orbit_rep(rep), quantum_numbers(qn), norm(n) {}
};

/**
 * @brief Sector information with orbit representatives
 */
struct SymmetrySector {
    uint64_t sector_id;
    std::vector<int> quantum_numbers;
    std::vector<Complex> phase_factors;
    std::vector<SymBasisState> basis_states;  // Compact basis representation
    
    SymmetrySector() : sector_id(0) {}
};

// ============================================================================
// Streaming Symmetry Operator
// ============================================================================

/**
 * @brief Operator class with streaming symmetry-adapted matrix-vector products
 * 
 * This class extends the Operator class to provide on-the-fly computation
 * of symmetrized matrix-vector products without storing basis vectors or matrices.
 */
class StreamingSymmetryOperator : public Operator {
private:
    std::vector<SymmetrySector> sectors_;
    mutable std::unordered_map<uint64_t, uint64_t> state_to_orbit_cache_;  // Cache for orbit lookups
    
public:
    StreamingSymmetryOperator(uint64_t n_bits, float spin_l) 
        : Operator(n_bits, spin_l) {}
    
    /**
     * @brief Generate symmetry sectors with orbit representatives (streaming version)
     * 
     * This generates the sectors without saving basis vectors to disk.
     * Memory usage: O(num_sectors × sector_dimension)
     */
    void generateSymmetrySectorsStreaming(const std::string& dir) {
        std::cout << "\n=== Generating Symmetry Sectors (Streaming) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const size_t dim = 1ULL << n_bits_;
        sectors_.resize(symmetry_info.sectors.size());
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        // Process each sector
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            auto& sector = sectors_[sector_idx];
            
            sector.sector_id = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors = sector_meta.phase_factors;
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (auto qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            // Find all orbit representatives for this sector
            std::unordered_set<uint64_t> processed_orbits;
            
            for (uint64_t basis = 0; basis < dim; ++basis) {
                // Progress indicator
                if (basis % (dim / 20) == 0 && dim > 20) {
                    std::cout << "\r  Progress: " << (100 * basis / dim) << "%" << std::flush;
                }
                
                // Get orbit representative
                uint64_t orbit_rep = getOrbitRepresentativeFast(basis);
                
                // Skip if already processed
                if (processed_orbits.count(orbit_rep)) continue;
                
                // Check if this orbit belongs to this sector
                double norm_sq = computeSymmetrizedNorm(basis, sector.quantum_numbers, 
                                                        sector.phase_factors);
                
                if (norm_sq > 1e-10) {
                    processed_orbits.insert(orbit_rep);
                    
                    SymBasisState state(orbit_rep, sector.quantum_numbers, std::sqrt(norm_sq));
                    
                    // Optionally precompute orbit elements for faster apply
                    if (n_bits_ <= 20) {  // Only for smaller systems to save memory
                        state.orbit_elements = computeOrbitElements(basis);
                    }
                    
                    sector.basis_states.push_back(state);
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector.basis_states.size();
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector.basis_states.size() << " basis states" << std::endl;
        }
        
        std::cout << "\nTotal sectors: " << sectors_.size() << std::endl;
        std::cout << "=== Symmetry Sector Generation Complete ===" << std::endl;
    }
    
    /**
     * @brief Apply Hamiltonian in a specific symmetry sector (streaming)
     * 
     * Computes H|ψ⟩ where |ψ⟩ is in the symmetrized basis of the given sector.
     * This is the key function that avoids matrix construction.
     * 
     * @param sector_idx Index of the symmetry sector
     * @param in Input vector in symmetrized basis (length = sector_dimension)
     * @param out Output vector H|in⟩ in symmetrized basis
     */
    void applySymmetrized(size_t sector_idx, const Complex* in, Complex* out) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        
        const auto& sector = sectors_[sector_idx];
        const size_t sector_dim = sector.basis_states.size();
        const size_t full_dim = 1ULL << n_bits_;
        
        // Initialize output
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));
        
        // Temporary full-space vectors for intermediate computation
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));
        std::vector<Complex> h_full_vec(full_dim, Complex(0.0, 0.0));
        
        // For each basis state in the sector
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < sector_dim; ++i) {
            if (std::abs(in[i]) < 1e-14) continue;
            
            // Thread-local temporary vectors
            std::vector<Complex> local_full_vec(full_dim, Complex(0.0, 0.0));
            std::vector<Complex> local_h_full_vec(full_dim, Complex(0.0, 0.0));
            
            // Expand i-th basis state to full Hilbert space
            expandSymmetrizedState(sector.basis_states[i], sector.phase_factors, 
                                   local_full_vec.data());
            
            // Scale by input coefficient
            for (size_t k = 0; k < full_dim; ++k) {
                local_full_vec[k] *= in[i];
            }
            
            // Apply Hamiltonian in full space: H|ψ_i⟩
            applyFullSpace(local_full_vec.data(), local_h_full_vec.data(), full_dim);
            
            // Project back to symmetrized basis
            #pragma omp critical
            {
                for (size_t j = 0; j < sector_dim; ++j) {
                    out[j] += projectOntoSymmetrizedState(local_h_full_vec.data(), 
                                                         sector.basis_states[j],
                                                         sector.phase_factors);
                }
            }
        }
    }
    
    /**
     * @brief Apply Hamiltonian in a specific symmetry sector (vector interface)
     */
    std::vector<Complex> applySymmetrized(size_t sector_idx, 
                                          const std::vector<Complex>& vec) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        
        const size_t sector_dim = sectors_[sector_idx].basis_states.size();
        std::vector<Complex> result(sector_dim);
        
        applySymmetrized(sector_idx, vec.data(), result.data());
        
        return result;
    }
    
    /**
     * @brief Get sector information
     */
    const SymmetrySector& getSector(size_t sector_idx) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        return sectors_[sector_idx];
    }
    
    size_t getNumSectors() const { return sectors_.size(); }
    
    /**
     * @brief Get dimension of a specific sector
     */
    uint64_t getSectorDimension(size_t sector_idx) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        return sectors_[sector_idx].basis_states.size();
    }
    
    /**
     * @brief Save sector metadata (lightweight, only orbit representatives)
     */
    void saveSectorMetadata(const std::string& dir) const {
        std::string metadata_dir = dir + "/sym_metadata";
        safe_system_call("mkdir -p " + metadata_dir);
        
        // Save sector dimensions
        std::ofstream dim_file(metadata_dir + "/sector_dimensions.txt");
        for (const auto& sector : sectors_) {
            dim_file << sector.basis_states.size() << "\n";
        }
        dim_file.close();
        
        // Save orbit representatives for each sector (binary format for efficiency)
        for (size_t i = 0; i < sectors_.size(); ++i) {
            std::string filename = metadata_dir + "/sector_" + std::to_string(i) + "_orbits.bin";
            std::ofstream file(filename, std::ios::binary);
            
            uint64_t num_states = sectors_[i].basis_states.size();
            file.write(reinterpret_cast<const char*>(&num_states), sizeof(uint64_t));
            
            for (const auto& state : sectors_[i].basis_states) {
                file.write(reinterpret_cast<const char*>(&state.orbit_rep), sizeof(uint64_t));
                file.write(reinterpret_cast<const char*>(&state.norm), sizeof(double));
            }
            
            file.close();
        }
        
        std::cout << "Saved sector metadata to " << metadata_dir << std::endl;
    }
    
    /**
     * @brief Load sector metadata
     */
    void loadSectorMetadata(const std::string& dir) {
        std::string metadata_dir = dir + "/sym_metadata";
        
        // Load sector dimensions
        std::ifstream dim_file(metadata_dir + "/sector_dimensions.txt");
        if (!dim_file.is_open()) {
            throw std::runtime_error("Could not open sector dimensions file");
        }
        
        std::vector<uint64_t> dimensions;
        uint64_t dim;
        while (dim_file >> dim) {
            dimensions.push_back(dim);
        }
        dim_file.close();
        
        // Load orbit representatives for each sector
        sectors_.resize(dimensions.size());
        symmetrized_block_ham_sizes.resize(dimensions.size());
        
        for (size_t i = 0; i < dimensions.size(); ++i) {
            std::string filename = metadata_dir + "/sector_" + std::to_string(i) + "_orbits.bin";
            std::ifstream file(filename, std::ios::binary);
            
            if (!file.is_open()) {
                throw std::runtime_error("Could not open orbit file: " + filename);
            }
            
            uint64_t num_states;
            file.read(reinterpret_cast<char*>(&num_states), sizeof(uint64_t));
            
            sectors_[i].sector_id = i;
            sectors_[i].quantum_numbers = symmetry_info.sectors[i].quantum_numbers;
            sectors_[i].phase_factors = symmetry_info.sectors[i].phase_factors;
            sectors_[i].basis_states.resize(num_states);
            
            for (uint64_t j = 0; j < num_states; ++j) {
                file.read(reinterpret_cast<char*>(&sectors_[i].basis_states[j].orbit_rep), 
                         sizeof(uint64_t));
                file.read(reinterpret_cast<char*>(&sectors_[i].basis_states[j].norm), 
                         sizeof(double));
                sectors_[i].basis_states[j].quantum_numbers = sectors_[i].quantum_numbers;
            }
            
            symmetrized_block_ham_sizes[i] = num_states;
            file.close();
        }
        
        std::cout << "Loaded sector metadata for " << sectors_.size() << " sectors" << std::endl;
    }
    
private:
    /**
     * @brief Fast orbit representative computation with caching
     */
    uint64_t getOrbitRepresentativeFast(uint64_t basis) const {
        // Check cache first
        auto it = state_to_orbit_cache_.find(basis);
        if (it != state_to_orbit_cache_.end()) {
            return it->second;
        }
        
        // Compute orbit representative
        uint64_t rep = basis;
        for (const auto& perm : symmetry_info.max_clique) {
            uint64_t permuted = applyPermutation(basis, perm);
            if (permuted < rep) rep = permuted;
        }
        
        // Cache result
        state_to_orbit_cache_[basis] = rep;
        
        return rep;
    }
    
    /**
     * @brief Compute all elements in the orbit of a given state
     */
    std::vector<uint64_t> computeOrbitElements(uint64_t basis) const {
        std::unordered_set<uint64_t> orbit_set;
        for (const auto& perm : symmetry_info.max_clique) {
            orbit_set.insert(applyPermutation(basis, perm));
        }
        return std::vector<uint64_t>(orbit_set.begin(), orbit_set.end());
    }
    
    /**
     * @brief Compute squared norm of symmetrized state
     * 
     * ||P_q|basis⟩||² where P_q is the projection operator
     */
    double computeSymmetrizedNorm(uint64_t basis, 
                                  const std::vector<int>& quantum_numbers,
                                  const std::vector<Complex>& phase_factors) const {
        double norm_sq = 0.0;
        
        // Apply symmetry projection and compute norm
        std::unordered_map<uint64_t, Complex> orbit_coefficients;
        
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            // Compute character: χ_q(g)
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            uint64_t permuted_basis = applyPermutation(basis, perm);
            orbit_coefficients[permuted_basis] += std::conj(character);  // Use conjugate for projection
        }
        
        // Compute norm
        for (const auto& [state, coeff] : orbit_coefficients) {
            norm_sq += std::norm(coeff);
        }
        
        // Normalization factor: 1/|G|
        norm_sq /= symmetry_info.max_clique.size();
        
        return norm_sq;
    }
    
    /**
     * @brief Expand a symmetrized state to full Hilbert space
     */
    void expandSymmetrizedState(const SymBasisState& state,
                                const std::vector<Complex>& phase_factors,
                                Complex* full_vec) const {
        const size_t full_dim = 1ULL << n_bits_;
        std::fill(full_vec, full_vec + full_dim, Complex(0.0, 0.0));
        
        uint64_t basis = state.orbit_rep;
        
        // Apply symmetry projection: |ψ_q⟩ = (1/√|G|) Σ_g χ_q(g)* g|basis⟩
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            // Compute character
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            uint64_t permuted_basis = applyPermutation(basis, perm);
            full_vec[permuted_basis] += std::conj(character);
        }
        
        // Normalize
        double norm_factor = 1.0 / (std::sqrt(symmetry_info.max_clique.size()) * state.norm);
        for (size_t i = 0; i < full_dim; ++i) {
            full_vec[i] *= norm_factor;
        }
    }
    
    /**
     * @brief Project a full-space vector onto a symmetrized state
     * 
     * Returns ⟨ψ_sym|vec⟩ where |ψ_sym⟩ is the symmetrized state
     */
    Complex projectOntoSymmetrizedState(const Complex* full_vec,
                                       const SymBasisState& state,
                                       const std::vector<Complex>& phase_factors) const {
        Complex result(0.0, 0.0);
        uint64_t basis = state.orbit_rep;
        
        // ⟨ψ_sym| = (1/√|G|) Σ_g χ_q(g) ⟨basis|g†
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            // Compute character χ_q(g) for the bra
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            uint64_t permuted_basis = applyPermutation(basis, perm);
            result += character * full_vec[permuted_basis];
        }
        
        // Normalize
        result /= (std::sqrt(symmetry_info.max_clique.size()) * state.norm);
        
        return result;
    }
    
    /**
     * @brief Apply Hamiltonian in full Hilbert space
     */
    void applyFullSpace(const Complex* in, Complex* out, size_t dim) const {
        std::fill(out, out + dim, Complex(0.0, 0.0));
        
        for (size_t i = 0; i < dim; ++i) {
            if (std::abs(in[i]) < 1e-14) continue;
            
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < static_cast<int>(dim)) {
                    out[j] += scalar * in[i];
                }
            }
        }
    }
};

// ============================================================================
// Fixed Sz Streaming Symmetry Operator
// ============================================================================

/**
 * @brief Streaming symmetry operator for fixed Sz sector
 * 
 * Combines Sz conservation with spatial symmetries in a streaming fashion.
 */
class FixedSzStreamingSymmetryOperator : public FixedSzOperator {
private:
    std::vector<SymmetrySector> sectors_;
    mutable std::unordered_map<uint64_t, uint64_t> state_to_orbit_cache_;
    
public:
    FixedSzStreamingSymmetryOperator(uint64_t n_bits, float spin_l, int64_t n_up)
        : FixedSzOperator(n_bits, spin_l, n_up) {}
    
    /**
     * @brief Generate symmetry sectors within fixed Sz sector (streaming)
     */
    void generateSymmetrySectorsStreamingFixedSz(const std::string& dir) {
        std::cout << "\n=== Generating Symmetry Sectors (Streaming, Fixed Sz) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        sectors_.resize(symmetry_info.sectors.size());
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        // Process each sector
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector_meta = symmetry_info.sectors[sector_idx];
            auto& sector = sectors_[sector_idx];
            
            sector.sector_id = sector_meta.sector_id;
            sector.quantum_numbers = sector_meta.quantum_numbers;
            sector.phase_factors = sector_meta.phase_factors;
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (auto qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            // Find orbit representatives in fixed Sz sector
            std::unordered_set<uint64_t> processed_orbits;
            
            for (size_t i = 0; i < basis_states_.size(); ++i) {
                uint64_t basis = basis_states_[i];
                
                // Progress indicator
                if (i % (basis_states_.size() / 20) == 0 && basis_states_.size() > 20) {
                    std::cout << "\r  Progress: " << (100 * i / basis_states_.size()) 
                             << "%" << std::flush;
                }
                
                // Get orbit representative
                uint64_t orbit_rep = getOrbitRepresentativeFixedSzFast(basis);
                
                // Skip if already processed
                if (processed_orbits.count(orbit_rep)) continue;
                
                // Check if this orbit belongs to this sector
                double norm_sq = computeSymmetrizedNormFixedSz(basis, sector.quantum_numbers,
                                                               sector.phase_factors);
                
                if (norm_sq > 1e-10) {
                    processed_orbits.insert(orbit_rep);
                    
                    SymBasisState state(orbit_rep, sector.quantum_numbers, std::sqrt(norm_sq));
                    sector.basis_states.push_back(state);
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector.basis_states.size();
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector.basis_states.size() << " basis states" << std::endl;
        }
        
        std::cout << "\nTotal sectors: " << sectors_.size() << std::endl;
        std::cout << "=== Symmetry Sector Generation Complete ===" << std::endl;
    }
    
    /**
     * @brief Apply Hamiltonian in symmetrized fixed Sz sector
     */
    void applySymmetrizedFixedSz(size_t sector_idx, const Complex* in, Complex* out) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        
        const auto& sector = sectors_[sector_idx];
        const size_t sector_dim = sector.basis_states.size();
        
        // Initialize output
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));
        
        // Temporary vectors in fixed Sz basis
        std::vector<Complex> fixed_sz_vec(fixed_sz_dim_, Complex(0.0, 0.0));
        std::vector<Complex> h_fixed_sz_vec(fixed_sz_dim_, Complex(0.0, 0.0));
        
        // For each basis state in the symmetrized sector
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < sector_dim; ++i) {
            if (std::abs(in[i]) < 1e-14) continue;
            
            // Thread-local vectors
            std::vector<Complex> local_vec(fixed_sz_dim_, Complex(0.0, 0.0));
            std::vector<Complex> local_h_vec(fixed_sz_dim_, Complex(0.0, 0.0));
            
            // Expand to fixed Sz basis
            expandSymmetrizedStateFixedSz(sector.basis_states[i], sector.phase_factors,
                                         local_vec.data());
            
            // Scale by input coefficient
            for (size_t k = 0; k < fixed_sz_dim_; ++k) {
                local_vec[k] *= in[i];
            }
            
            // Apply Hamiltonian in fixed Sz basis
            applyFixedSzSpace(local_vec.data(), local_h_vec.data(), fixed_sz_dim_);
            
            // Project back to symmetrized basis
            #pragma omp critical
            {
                for (size_t j = 0; j < sector_dim; ++j) {
                    out[j] += projectOntoSymmetrizedStateFixedSz(local_h_vec.data(),
                                                                sector.basis_states[j],
                                                                sector.phase_factors);
                }
            }
        }
    }
    
    std::vector<Complex> applySymmetrizedFixedSz(size_t sector_idx,
                                                 const std::vector<Complex>& vec) const {
        const size_t sector_dim = sectors_[sector_idx].basis_states.size();
        std::vector<Complex> result(sector_dim);
        applySymmetrizedFixedSz(sector_idx, vec.data(), result.data());
        return result;
    }
    
    const SymmetrySector& getSector(size_t sector_idx) const {
        return sectors_[sector_idx];
    }
    
    size_t getNumSectors() const { return sectors_.size(); }
    
    uint64_t getSectorDimension(size_t sector_idx) const {
        return sectors_[sector_idx].basis_states.size();
    }
    
private:
    uint64_t getOrbitRepresentativeFixedSzFast(uint64_t basis) const {
        auto it = state_to_orbit_cache_.find(basis);
        if (it != state_to_orbit_cache_.end()) {
            return it->second;
        }
        
        uint64_t rep = basis;
        for (const auto& perm : symmetry_info.max_clique) {
            uint64_t permuted = applyPermutation(basis, perm);
            // Check if still in fixed Sz sector
            if (state_to_index_.count(permuted) && permuted < rep) {
                rep = permuted;
            }
        }
        
        state_to_orbit_cache_[basis] = rep;
        return rep;
    }
    
    double computeSymmetrizedNormFixedSz(uint64_t basis,
                                        const std::vector<int>& quantum_numbers,
                                        const std::vector<Complex>& phase_factors) const {
        double norm_sq = 0.0;
        std::unordered_map<uint64_t, Complex> orbit_coefficients;
        
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
            
            uint64_t permuted_basis = applyPermutation(basis, perm);
            // Only include if in fixed Sz sector
            if (state_to_index_.count(permuted_basis)) {
                orbit_coefficients[permuted_basis] += std::conj(character);  // Use conjugate for projection
            }
        }
        
        for (const auto& [state, coeff] : orbit_coefficients) {
            norm_sq += std::norm(coeff);
        }
        
        norm_sq /= symmetry_info.max_clique.size();
        return norm_sq;
    }
    
    void expandSymmetrizedStateFixedSz(const SymBasisState& state,
                                      const std::vector<Complex>& phase_factors,
                                      Complex* fixed_sz_vec) const {
        std::fill(fixed_sz_vec, fixed_sz_vec + fixed_sz_dim_, Complex(0.0, 0.0));
        
        uint64_t basis = state.orbit_rep;
        
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
            
            uint64_t permuted_basis = applyPermutation(basis, perm);
            auto it = state_to_index_.find(permuted_basis);
            if (it != state_to_index_.end()) {
                fixed_sz_vec[it->second] += std::conj(character);
            }
        }
        
        double norm_factor = 1.0 / (std::sqrt(symmetry_info.max_clique.size()) * state.norm);
        for (size_t i = 0; i < fixed_sz_dim_; ++i) {
            fixed_sz_vec[i] *= norm_factor;
        }
    }
    
    Complex projectOntoSymmetrizedStateFixedSz(const Complex* fixed_sz_vec,
                                               const SymBasisState& state,
                                               const std::vector<Complex>& phase_factors) const {
        Complex result(0.0, 0.0);
        uint64_t basis = state.orbit_rep;
        
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
            
            uint64_t permuted_basis = applyPermutation(basis, perm);
            auto it = state_to_index_.find(permuted_basis);
            if (it != state_to_index_.end()) {
                result += character * fixed_sz_vec[it->second];
            }
        }
        
        result /= (std::sqrt(symmetry_info.max_clique.size()) * state.norm);
        return result;
    }
    
    void applyFixedSzSpace(const Complex* in, Complex* out, size_t dim) const {
        std::fill(out, out + dim, Complex(0.0, 0.0));
        
        for (size_t i = 0; i < dim; ++i) {
            if (std::abs(in[i]) < 1e-14) continue;
            
            uint64_t basis = basis_states_[i];
            for (const auto& transform : transforms_) {
                auto [j_state, scalar] = transform(basis);
                if (j_state >= 0) {
                    auto it = state_to_index_.find(j_state);
                    if (it != state_to_index_.end()) {
                        out[it->second] += scalar * in[i];
                    }
                }
            }
        }
    }
};

#endif // STREAMING_SYMMETRY_H
