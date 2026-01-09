#ifndef STREAMING_SYMMETRY_H
#define STREAMING_SYMMETRY_H

#include <ed/core/construct_ham.h>
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
 * - Memory usage scales with O(sector_dim × orbit_size) instead of O(full_dim)
 * - Faster basis generation (no file I/O)
 * - Can handle much larger systems
 * 
 * Algorithm (Matrix-Free):
 * 1. Generate orbit representatives on-the-fly
 * 2. Pre-compute orbit elements and phase factors per basis state
 * 3. Build lookup table: computational_state -> (sector_basis_index, orbit_index)
 * 4. Apply H term-by-term on orbit elements, project to symmetrized basis
 * 
 * This is truly matrix-free: H|ψ⟩ computed without expanding to full Hilbert space.
 */

// ============================================================================
// Orbit-Based Symmetry Data Structures
// ============================================================================

/**
 * @brief Compact representation of a symmetrized basis state
 * 
 * Stores orbit elements and their phase coefficients for efficient H*v computation.
 * Memory: O(orbit_size) per basis state instead of O(full_dim).
 */
struct SymBasisState {
    uint64_t orbit_rep;                    // Representative element (smallest in orbit)
    std::vector<int> quantum_numbers;      // Quantum numbers for this sector
    std::vector<uint64_t> orbit_elements;  // All states in the orbit
    std::vector<Complex> orbit_coefficients;  // Coefficient of each orbit element in symmetrized state
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
 * 
 * MATRIX-FREE APPROACH:
 * - Pre-computes orbit elements and coefficients for each symmetrized basis state
 * - Builds lookup table: computational_state -> (sector_idx, basis_idx)
 * - Apply H term-by-term on orbit elements, project using lookup (no 2^N expansion)
 * 
 * Memory: O(total_orbit_elements) instead of O(2^N)
 */
class StreamingSymmetryOperator : public Operator {
private:
    std::vector<SymmetrySector> sectors_;
    mutable std::unordered_map<uint64_t, uint64_t> state_to_orbit_cache_;  // Cache for orbit lookups
    
    // Lookup table: computational_state -> (sector_idx, basis_idx_in_sector)
    // This enables O(1) projection of H-transformed states
    mutable std::vector<std::unordered_map<uint64_t, size_t>> state_to_sector_basis_;
    
public:
    StreamingSymmetryOperator(uint64_t n_bits, float spin_l) 
        : Operator(n_bits, spin_l) {}
    
    /**
     * @brief Generate symmetry sectors with orbit representatives (streaming version)
     * 
     * This generates the sectors without saving basis vectors to disk.
     * Pre-computes orbit elements/coefficients and builds lookup table for
     * matrix-free H*v computation.
     * 
     * Memory usage: O(total_orbit_elements) - much smaller than O(2^N)
     */
    void generateSymmetrySectorsStreaming(const std::string& dir) {
        std::cout << "\n=== Generating Symmetry Sectors (Matrix-Free Streaming) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        const size_t dim = 1ULL << n_bits_;
        sectors_.resize(symmetry_info.sectors.size());
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        state_to_sector_basis_.resize(symmetry_info.sectors.size());
        
        size_t total_orbit_elements = 0;
        
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
            std::cout << ")" << std::flush;
            
            // Find all orbit representatives for this sector
            std::unordered_set<uint64_t> processed_orbits;
            
            for (uint64_t basis = 0; basis < dim; ++basis) {
                // Get orbit representative
                uint64_t orbit_rep = getOrbitRepresentativeFast(basis);
                
                // Skip if already processed
                if (processed_orbits.count(orbit_rep)) continue;
                
                // Compute orbit elements and coefficients for this sector
                std::vector<uint64_t> orbit_elements;
                std::vector<Complex> orbit_coefficients;
                double norm_sq = 0.0;
                
                computeOrbitData(basis, sector.phase_factors, 
                                orbit_elements, orbit_coefficients, norm_sq);
                
                if (norm_sq > 1e-10) {
                    processed_orbits.insert(orbit_rep);
                    
                    SymBasisState state(orbit_rep, sector.quantum_numbers, std::sqrt(norm_sq));
                    state.orbit_elements = std::move(orbit_elements);
                    state.orbit_coefficients = std::move(orbit_coefficients);
                    
                    // Build lookup: each orbit element maps to this basis state
                    size_t basis_idx = sector.basis_states.size();
                    for (uint64_t elem : state.orbit_elements) {
                        state_to_sector_basis_[sector_idx][elem] = basis_idx;
                    }
                    
                    total_orbit_elements += state.orbit_elements.size();
                    sector.basis_states.push_back(std::move(state));
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector.basis_states.size();
            std::cout << " -> " << sector.basis_states.size() << " basis states" << std::endl;
        }
        
        size_t total_basis = 0;
        for (const auto& sector : sectors_) {
            total_basis += sector.basis_states.size();
        }
        
        std::cout << "\n=== Matrix-Free Sector Generation Complete ===" << std::endl;
        std::cout << "Total sectors: " << sectors_.size() << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis << std::endl;
        std::cout << "Total orbit elements stored: " << total_orbit_elements << std::endl;
        std::cout << "Memory saved vs full expansion: " 
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - double(total_orbit_elements) / (total_basis * dim)))
                  << "%" << std::endl;
    }
    
    /**
     * @brief Matrix-free Hamiltonian application in symmetrized sector
     * 
     * This is the key function that avoids expanding to 2^N dimension.
     * 
     * Algorithm:
     * 1. For each input coefficient c_j with basis state |φ_j⟩
     * 2. For each orbit element |s⟩ in |φ_j⟩ with coefficient α_s
     * 3. Apply each Hamiltonian term to |s⟩ -> |s'⟩ with matrix element h
     * 4. Look up which basis state |φ_k⟩ contains |s'⟩
     * 5. Accumulate: out[k] += c_j * α_s * h * conj(β_{s'}) / (norm_j * norm_k * |G|)
     *    where β_{s'} is the coefficient of |s'⟩ in |φ_k⟩
     * 
     * Memory: O(sector_dim) for output, no 2^N intermediates
     */
    void applySymmetrized(size_t sector_idx, const Complex* in, Complex* out) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        
        const auto& sector = sectors_[sector_idx];
        const size_t sector_dim = sector.basis_states.size();
        const auto& lookup = state_to_sector_basis_[sector_idx];
        
        // Initialize output
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));
        
        // Pre-compute normalization factors
        const double group_size = static_cast<double>(symmetry_info.max_clique.size());
        const double group_norm = 1.0 / group_size;
        
        // Thread-local accumulator for sequential case
        std::vector<Complex> local_out(sector_dim, Complex(0.0, 0.0));
        
        // Process each input basis state
        for (size_t j = 0; j < sector_dim; ++j) {
            Complex c_j = in[j];
            if (std::abs(c_j) < 1e-15) continue;
            
            const auto& state_j = sector.basis_states[j];
            const double norm_j = state_j.norm;
            
            // Iterate over orbit elements of |φ_j⟩
            for (size_t orbit_idx = 0; orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                uint64_t s = state_j.orbit_elements[orbit_idx];
                Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                
                // Skip if coefficient is negligible
                if (std::abs(alpha_s) < 1e-15) continue;
                
                // Apply Hamiltonian terms to |s⟩
                applyHamiltonianTermsFullSpace(s, c_j * alpha_s / norm_j, 
                                               sector, lookup, group_norm, local_out);
            }
        }
        
        // Copy to output
        for (size_t k = 0; k < sector_dim; ++k) {
            out[k] = local_out[k];
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
     * @brief Compute orbit elements and coefficients for a basis state in a sector
     * 
     * Returns the orbit elements (computational basis states) and their
     * corresponding complex coefficients in the symmetrized state.
     * This is the full-space version (no fixed-Sz restriction).
     */
    void computeOrbitData(uint64_t basis,
                          const std::vector<Complex>& phase_factors,
                          std::vector<uint64_t>& orbit_elements,
                          std::vector<Complex>& orbit_coefficients,
                          double& norm_sq) const {
        
        std::unordered_map<uint64_t, Complex> coeff_map;
        
        // Apply all group elements and accumulate coefficients
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            // Compute character χ_q(g)*
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            uint64_t permuted = applyPermutation(basis, perm);
            coeff_map[permuted] += std::conj(character);
        }
        
        // Convert to vectors and compute norm
        orbit_elements.clear();
        orbit_coefficients.clear();
        norm_sq = 0.0;
        
        for (const auto& [state, coeff] : coeff_map) {
            if (std::abs(coeff) > 1e-15) {
                orbit_elements.push_back(state);
                orbit_coefficients.push_back(coeff);
                norm_sq += std::norm(coeff);
            }
        }
        
        norm_sq /= symmetry_info.max_clique.size();
    }
    
    /**
     * @brief Apply all Hamiltonian terms to a single computational basis state
     * 
     * This is the inner loop of the matrix-free multiplication.
     * Projects results onto the symmetrized basis using the lookup table.
     * Full-space version (no fixed-Sz restriction).
     */
    void applyHamiltonianTermsFullSpace(uint64_t s, Complex weighted_coeff,
                                        const SymmetrySector& sector,
                                        const std::unordered_map<uint64_t, size_t>& lookup,
                                        double group_norm,
                                        std::vector<Complex>& local_out) const {
        
        // Helper lambda to project result onto sector
        auto projectResult = [&](uint64_t s_prime, Complex h_element) {
            auto it = lookup.find(s_prime);
            if (it == lookup.end()) return;
            
            size_t k = it->second;
            const auto& state_k = sector.basis_states[k];
            
            // Find coefficient of s_prime in |φ_k⟩
            Complex beta_s_prime(0.0, 0.0);
            for (size_t orbit_idx = 0; orbit_idx < state_k.orbit_elements.size(); ++orbit_idx) {
                if (state_k.orbit_elements[orbit_idx] == s_prime) {
                    beta_s_prime = state_k.orbit_coefficients[orbit_idx];
                    break;
                }
            }
            
            // Accumulate: out[k] += weighted_coeff * h * conj(β_{s'}) / norm_k
            local_out[k] += weighted_coeff * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
        };
        
        // Apply each one/two-body term from transform_data_
        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            if (!tdata.is_two_body) {
                // One-body: S^α_i
                if (tdata.op_type == 2) {
                    // Sz: diagonal
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    // S+ or S-: flip bit
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                // Two-body: S^α_i S^β_j
                uint64_t bit_i = (s >> tdata.site_index) & 1;
                uint64_t bit_j = (s >> tdata.site_index_2) & 1;
                
                if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                    // Sz_i Sz_j: diagonal
                    double sign_i = bit_i ? -1.0 : 1.0;
                    double sign_j = bit_j ? -1.0 : 1.0;
                    h_element *= spin_l_ * spin_l_ * sign_i * sign_j;
                } else {
                    // Mixed terms
                    if (tdata.op_type != 2) {
                        if (bit_i != tdata.op_type) {
                            s_prime ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    } else {
                        double sign_i = bit_i ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_i;
                    }
                    
                    if (valid && tdata.op_type_2 != 2) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        if (new_bit_j != tdata.op_type_2) {
                            s_prime ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    } else if (valid) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        double sign_j = new_bit_j ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_j;
                    }
                }
            }
            
            if (valid) {
                projectResult(s_prime, h_element);
            }
        }
        
        // Apply three-body terms from three_body_data_
        for (const auto& tdata : three_body_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            // Apply first operator
            if (tdata.op_type_1 == 2) {
                uint64_t bit = (s_prime >> tdata.site_index_1) & 1;
                double sign = bit ? -1.0 : 1.0;
                h_element *= spin_l_ * sign;
            } else {
                uint64_t bit = (s_prime >> tdata.site_index_1) & 1;
                if (bit != tdata.op_type_1) {
                    s_prime ^= (1ULL << tdata.site_index_1);
                } else {
                    valid = false;
                }
            }
            
            // Apply second operator
            if (valid) {
                if (tdata.op_type_2 == 2) {
                    uint64_t bit = (s_prime >> tdata.site_index_2) & 1;
                    double sign = bit ? -1.0 : 1.0;
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
            
            // Apply third operator
            if (valid) {
                if (tdata.op_type_3 == 2) {
                    uint64_t bit = (s_prime >> tdata.site_index_3) & 1;
                    double sign = bit ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    uint64_t bit = (s_prime >> tdata.site_index_3) & 1;
                    if (bit != tdata.op_type_3) {
                        s_prime ^= (1ULL << tdata.site_index_3);
                    } else {
                        valid = false;
                    }
                }
            }
            
            if (valid) {
                projectResult(s_prime, h_element);
            }
        }
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
     * Uses optimized transform_data_ storage (same as standard ED)
     */
    void applyFullSpace(const Complex* in, Complex* out, size_t dim) const {
        std::fill(out, out + dim, Complex(0.0, 0.0));
        
        for (size_t basis = 0; basis < dim; ++basis) {
            if (std::abs(in[basis]) < 1e-14) continue;
            
            Complex coeff = in[basis];
            
            // Process all transforms using optimized transform_data_ representation
            for (const auto& tdata : transform_data_) {
                uint64_t new_basis = basis;
                Complex scalar = tdata.coefficient;
                bool valid = true;

                if (!tdata.is_two_body) {
                    // One-body operator: S^α_i
                    if (tdata.op_type == 2) {
                        // Sz: diagonal, just multiply by eigenvalue
                        double sign = ((basis >> tdata.site_index) & 1) ? -1.0 : 1.0;
                        scalar *= spin_l_ * sign;
                    } else {
                        // S+ or S-: off-diagonal, flip bit
                        uint64_t bit = (basis >> tdata.site_index) & 1;
                        if (bit != tdata.op_type) {
                            new_basis ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    }
                } else {
                    // Two-body operator: S^α_i S^β_j
                    uint64_t bit_i = (basis >> tdata.site_index) & 1;
                    uint64_t bit_j = (basis >> tdata.site_index_2) & 1;

                    if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                        // Sz_i Sz_j: purely diagonal
                        double sign_i = bit_i ? -1.0 : 1.0;
                        double sign_j = bit_j ? -1.0 : 1.0;
                        scalar *= spin_l_ * spin_l_ * sign_i * sign_j;
                    } else {
                        // Mixed or off-diagonal terms
                        if (tdata.op_type != 2) {
                            if (bit_i != tdata.op_type) {
                                new_basis ^= (1ULL << tdata.site_index);
                            } else {
                                valid = false;
                            }
                        } else {
                            double sign_i = bit_i ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_i;
                        }

                        if (valid && tdata.op_type_2 != 2) {
                            uint64_t new_bit_j = (new_basis >> tdata.site_index_2) & 1;
                            if (new_bit_j != tdata.op_type_2) {
                                new_basis ^= (1ULL << tdata.site_index_2);
                            } else {
                                valid = false;
                            }
                        } else if (valid) {
                            uint64_t new_bit_j = (new_basis >> tdata.site_index_2) & 1;
                            double sign_j = new_bit_j ? -1.0 : 1.0;
                            scalar *= spin_l_ * sign_j;
                        }
                    }
                }

                if (valid && std::abs(scalar) > 1e-15) {
                    out[new_basis] += scalar * coeff;
                }
            }
            
            // Process three-body terms
            for (const auto& tdata : three_body_data_) {
                uint64_t new_basis = basis;
                Complex scalar = tdata.coefficient;
                bool valid = true;
                
                // Apply first operator
                if (tdata.op_type_1 == 2) {
                    uint64_t bit_1 = (new_basis >> tdata.site_index_1) & 1;
                    double sign_1 = bit_1 ? -1.0 : 1.0;
                    scalar *= spin_l_ * sign_1;
                } else {
                    uint64_t bit_1 = (new_basis >> tdata.site_index_1) & 1;
                    if (bit_1 != tdata.op_type_1) {
                        new_basis ^= (1ULL << tdata.site_index_1);
                    } else {
                        valid = false;
                    }
                }
                
                // Apply second operator
                if (valid) {
                    if (tdata.op_type_2 == 2) {
                        uint64_t bit_2 = (new_basis >> tdata.site_index_2) & 1;
                        double sign_2 = bit_2 ? -1.0 : 1.0;
                        scalar *= spin_l_ * sign_2;
                    } else {
                        uint64_t bit_2 = (new_basis >> tdata.site_index_2) & 1;
                        if (bit_2 != tdata.op_type_2) {
                            new_basis ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    }
                }
                
                // Apply third operator
                if (valid) {
                    if (tdata.op_type_3 == 2) {
                        uint64_t bit_3 = (new_basis >> tdata.site_index_3) & 1;
                        double sign_3 = bit_3 ? -1.0 : 1.0;
                        scalar *= spin_l_ * sign_3;
                    } else {
                        uint64_t bit_3 = (new_basis >> tdata.site_index_3) & 1;
                        if (bit_3 != tdata.op_type_3) {
                            new_basis ^= (1ULL << tdata.site_index_3);
                        } else {
                            valid = false;
                        }
                    }
                }
                
                if (valid && std::abs(scalar) > 1e-15) {
                    out[new_basis] += scalar * coeff;
                }
            }
        }
    }
};

// ============================================================================
// Fixed Sz Streaming Symmetry Operator (MATRIX-FREE)
// ============================================================================

/**
 * @brief Truly matrix-free symmetry operator for fixed Sz sector
 * 
 * This implementation is memory-efficient: it doesn't expand to the full
 * fixed-Sz dimension. Instead, it works directly with orbit elements.
 * 
 * Memory complexity: O(sector_dim × average_orbit_size)
 * Time complexity: O(sector_dim × orbit_size × num_hamiltonian_terms)
 * 
 * Key optimizations:
 * 1. Pre-compute orbit elements and coefficients during sector generation
 * 2. Build lookup table: computational_state -> (sector_basis_index)
 * 3. Apply H term-by-term on orbit elements only
 * 4. Project results using orbit lookup (no full-dim vectors)
 */
class FixedSzStreamingSymmetryOperator : public FixedSzOperator {
private:
    std::vector<SymmetrySector> sectors_;
    
    // Lookup table: computational_state -> (sector_index, basis_index_in_sector)
    // This enables O(1) projection of H-transformed states
    mutable std::vector<std::unordered_map<uint64_t, size_t>> state_to_sector_basis_;
    
    mutable std::unordered_map<uint64_t, uint64_t> state_to_orbit_cache_;
    
public:
    FixedSzStreamingSymmetryOperator(uint64_t n_bits, float spin_l, int64_t n_up)
        : FixedSzOperator(n_bits, spin_l, n_up) {}
    
    /**
     * @brief Generate symmetry sectors with pre-computed orbit data (optimized)
     * 
     * This version pre-computes orbit elements and coefficients, and builds
     * lookup tables for efficient matrix-free H*v computation.
     */
    void generateSymmetrySectorsStreamingFixedSz(const std::string& dir) {
        std::cout << "\n=== Generating Symmetry Sectors (Matrix-Free, Fixed Sz) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        sectors_.resize(symmetry_info.sectors.size());
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        state_to_sector_basis_.resize(symmetry_info.sectors.size());
        
        size_t total_orbit_elements = 0;
        
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
            std::cout << ")" << std::flush;
            
            // Find orbit representatives in fixed Sz sector
            std::unordered_set<uint64_t> processed_orbits;
            
            for (size_t i = 0; i < basis_states_.size(); ++i) {
                uint64_t basis = basis_states_[i];
                
                // Get orbit representative
                uint64_t orbit_rep = getOrbitRepresentativeFixedSzFast(basis);
                
                // Skip if already processed
                if (processed_orbits.count(orbit_rep)) continue;
                
                // Compute orbit elements and coefficients for this sector
                std::vector<uint64_t> orbit_elements;
                std::vector<Complex> orbit_coefficients;
                double norm_sq = 0.0;
                
                computeOrbitDataFixedSz(basis, sector.phase_factors, 
                                       orbit_elements, orbit_coefficients, norm_sq);
                
                if (norm_sq > 1e-10) {
                    processed_orbits.insert(orbit_rep);
                    
                    SymBasisState state(orbit_rep, sector.quantum_numbers, std::sqrt(norm_sq));
                    state.orbit_elements = std::move(orbit_elements);
                    state.orbit_coefficients = std::move(orbit_coefficients);
                    
                    // Build lookup: each orbit element maps to this basis state
                    size_t basis_idx = sector.basis_states.size();
                    for (uint64_t elem : state.orbit_elements) {
                        state_to_sector_basis_[sector_idx][elem] = basis_idx;
                    }
                    
                    total_orbit_elements += state.orbit_elements.size();
                    sector.basis_states.push_back(std::move(state));
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector.basis_states.size();
            std::cout << " -> " << sector.basis_states.size() << " basis states" << std::endl;
        }
        
        size_t total_basis = 0;
        for (const auto& sector : sectors_) {
            total_basis += sector.basis_states.size();
        }
        
        std::cout << "\n=== Matrix-Free Sector Generation Complete ===" << std::endl;
        std::cout << "Total sectors: " << sectors_.size() << std::endl;
        std::cout << "Total symmetrized basis: " << total_basis << std::endl;
        std::cout << "Total orbit elements stored: " << total_orbit_elements << std::endl;
        std::cout << "Memory saved vs full expansion: " 
                  << std::fixed << std::setprecision(1)
                  << (100.0 * (1.0 - double(total_orbit_elements) / (total_basis * fixed_sz_dim_)))
                  << "%" << std::endl;
    }
    
    /**
     * @brief Matrix-free Hamiltonian application in symmetrized sector
     * 
     * This is the key function that avoids expanding to fixed-Sz dimension.
     * 
     * Algorithm:
     * 1. For each input coefficient c_j with basis state |φ_j⟩
     * 2. For each orbit element |s⟩ in |φ_j⟩ with coefficient α_s
     * 3. Apply each Hamiltonian term to |s⟩ -> |s'⟩ with matrix element h
     * 4. Look up which basis state |φ_k⟩ contains |s'⟩
     * 5. Accumulate: out[k] += c_j * α_s * h * conj(β_{s'})
     *    where β_{s'} is the coefficient of |s'⟩ in |φ_k⟩
     * 
     * Memory: O(sector_dim) for output, no full-dim intermediates
     */
    void applySymmetrizedFixedSz(size_t sector_idx, const Complex* in, Complex* out) const {
        if (sector_idx >= sectors_.size()) {
            throw std::runtime_error("Invalid sector index");
        }
        
        const auto& sector = sectors_[sector_idx];
        const size_t sector_dim = sector.basis_states.size();
        const auto& lookup = state_to_sector_basis_[sector_idx];
        
        // Initialize output
        std::fill(out, out + sector_dim, Complex(0.0, 0.0));
        
        // Pre-compute normalization factors
        const double group_size = static_cast<double>(symmetry_info.max_clique.size());
        const double group_norm = 1.0 / group_size;
        
        // Process each input basis state
        #pragma omp parallel if(sector_dim > 100)
        {
            // Thread-local accumulator to avoid atomic operations
            std::vector<Complex> local_out(sector_dim, Complex(0.0, 0.0));
            
            #pragma omp for schedule(dynamic, 4)
            for (size_t j = 0; j < sector_dim; ++j) {
                Complex c_j = in[j];
                if (std::abs(c_j) < 1e-15) continue;
                
                const auto& state_j = sector.basis_states[j];
                const double norm_j = state_j.norm;
                
                // Iterate over orbit elements of |φ_j⟩
                for (size_t orbit_idx = 0; orbit_idx < state_j.orbit_elements.size(); ++orbit_idx) {
                    uint64_t s = state_j.orbit_elements[orbit_idx];
                    Complex alpha_s = state_j.orbit_coefficients[orbit_idx];
                    
                    // Skip if coefficient is negligible
                    if (std::abs(alpha_s) < 1e-15) continue;
                    
                    // Apply Hamiltonian terms to |s⟩
                    // Use optimized term-by-term application
                    applyHamiltonianTerms(s, c_j * alpha_s / norm_j, 
                                         sector, lookup, group_norm, local_out);
                }
            }
            
            // Merge thread-local results
            #pragma omp critical
            {
                for (size_t k = 0; k < sector_dim; ++k) {
                    out[k] += local_out[k];
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
    /**
     * @brief Apply all Hamiltonian terms to a single computational basis state
     * 
     * This is the inner loop of the matrix-free multiplication.
     * Projects results onto the symmetrized basis using the lookup table.
     */
    void applyHamiltonianTerms(uint64_t s, Complex weighted_coeff,
                               const SymmetrySector& sector,
                               const std::unordered_map<uint64_t, size_t>& lookup,
                               double group_norm,
                               std::vector<Complex>& local_out) const {
        
        // Apply each one/two-body term from transform_data_
        for (const auto& tdata : transform_data_) {
            uint64_t s_prime = s;
            Complex h_element = tdata.coefficient;
            bool valid = true;
            
            if (!tdata.is_two_body) {
                // One-body: S^α_i
                if (tdata.op_type == 2) {
                    // Sz: diagonal
                    double sign = ((s >> tdata.site_index) & 1) ? -1.0 : 1.0;
                    h_element *= spin_l_ * sign;
                } else {
                    // S+ or S-: flip bit
                    uint64_t bit = (s >> tdata.site_index) & 1;
                    if (bit != tdata.op_type) {
                        s_prime ^= (1ULL << tdata.site_index);
                    } else {
                        valid = false;
                    }
                }
            } else {
                // Two-body: S^α_i S^β_j
                uint64_t bit_i = (s >> tdata.site_index) & 1;
                uint64_t bit_j = (s >> tdata.site_index_2) & 1;
                
                if (tdata.op_type == 2 && tdata.op_type_2 == 2) {
                    // Sz_i Sz_j: diagonal
                    double sign_i = bit_i ? -1.0 : 1.0;
                    double sign_j = bit_j ? -1.0 : 1.0;
                    h_element *= spin_l_ * spin_l_ * sign_i * sign_j;
                } else {
                    // Mixed terms
                    if (tdata.op_type != 2) {
                        if (bit_i != tdata.op_type) {
                            s_prime ^= (1ULL << tdata.site_index);
                        } else {
                            valid = false;
                        }
                    } else {
                        double sign_i = bit_i ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_i;
                    }
                    
                    if (valid && tdata.op_type_2 != 2) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        if (new_bit_j != tdata.op_type_2) {
                            s_prime ^= (1ULL << tdata.site_index_2);
                        } else {
                            valid = false;
                        }
                    } else if (valid) {
                        uint64_t new_bit_j = (s_prime >> tdata.site_index_2) & 1;
                        double sign_j = new_bit_j ? -1.0 : 1.0;
                        h_element *= spin_l_ * sign_j;
                    }
                }
            }
            
            if (!valid) continue;
            
            // Check if s_prime is in this sector (via lookup)
            auto it = lookup.find(s_prime);
            if (it == lookup.end()) continue;
            
            size_t k = it->second;  // Index of target basis state
            const auto& state_k = sector.basis_states[k];
            
            // Find coefficient of s_prime in |φ_k⟩
            Complex beta_s_prime(0.0, 0.0);
            for (size_t orbit_idx = 0; orbit_idx < state_k.orbit_elements.size(); ++orbit_idx) {
                if (state_k.orbit_elements[orbit_idx] == s_prime) {
                    beta_s_prime = state_k.orbit_coefficients[orbit_idx];
                    break;
                }
            }
            
            // Accumulate: out[k] += weighted_coeff * h * conj(β_{s'}) / norm_k
            local_out[k] += weighted_coeff * h_element * std::conj(beta_s_prime) * group_norm / state_k.norm;
        }
    }
    
    uint64_t getOrbitRepresentativeFixedSzFast(uint64_t basis) const {
        auto it = state_to_orbit_cache_.find(basis);
        if (it != state_to_orbit_cache_.end()) {
            return it->second;
        }
        
        uint64_t rep = basis;
        for (const auto& perm : symmetry_info.max_clique) {
            uint64_t permuted = applyPermutation(basis, perm);
            if (state_to_index_.count(permuted) && permuted < rep) {
                rep = permuted;
            }
        }
        
        state_to_orbit_cache_[basis] = rep;
        return rep;
    }
    
    /**
     * @brief Compute orbit elements and coefficients for a basis state in a sector
     * 
     * Returns the orbit elements (computational basis states) and their
     * corresponding complex coefficients in the symmetrized state.
     */
    void computeOrbitDataFixedSz(uint64_t basis,
                                 const std::vector<Complex>& phase_factors,
                                 std::vector<uint64_t>& orbit_elements,
                                 std::vector<Complex>& orbit_coefficients,
                                 double& norm_sq) const {
        
        std::unordered_map<uint64_t, Complex> coeff_map;
        
        // Apply all group elements and accumulate coefficients
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            // Compute character χ_q(g)*
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (int p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            uint64_t permuted = applyPermutation(basis, perm);
            
            // Only include if in fixed Sz sector
            if (state_to_index_.count(permuted)) {
                coeff_map[permuted] += std::conj(character);
            }
        }
        
        // Convert to vectors and compute norm
        orbit_elements.clear();
        orbit_coefficients.clear();
        norm_sq = 0.0;
        
        for (const auto& [state, coeff] : coeff_map) {
            if (std::abs(coeff) > 1e-15) {
                orbit_elements.push_back(state);
                orbit_coefficients.push_back(coeff);
                norm_sq += std::norm(coeff);
            }
        }
        
        norm_sq /= symmetry_info.max_clique.size();
    }
    
    // Legacy methods kept for backward compatibility
    double computeSymmetrizedNormFixedSz(uint64_t basis,
                                        const std::vector<int>& quantum_numbers,
                                        const std::vector<Complex>& phase_factors) const {
        std::vector<uint64_t> dummy_elements;
        std::vector<Complex> dummy_coeffs;
        double norm_sq;
        computeOrbitDataFixedSz(basis, phase_factors, dummy_elements, dummy_coeffs, norm_sq);
        return norm_sq;
    }
};

#endif // STREAMING_SYMMETRY_H
