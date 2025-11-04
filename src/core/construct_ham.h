#ifndef CONSTRUCT_HAM_H
#define CONSTRUCT_HAM_H

#include <vector>
#include <complex>
#include <functional>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <set>
#include <queue>
#include <map>
#include <tuple>
#include <unordered_map>
#include "system_utils.h"
#include "hdf5_symmetry_io.h"

// Define complex number type for convenience
using Complex = std::complex<double>;



// Calculate thermodynamic quantities directly from eigenvalues
struct ThermodynamicData {
    std::vector<double> temperatures;
    std::vector<double> energy;
    std::vector<double> specific_heat;
    std::vector<double> entropy;
    std::vector<double> free_energy;
};


// Calculate dynamical susceptibility χ(ω) for operator A
// χ(ω) = ∑_{n,m} (p_m - p_n) * |<n|A|m>|^2 / (ω - (E_n - E_m) + iη)
struct DynamicalSusceptibilityData {
    std::vector<double> frequencies;         // ω values
    std::vector<std::complex<double>> chi;   // χ(ω) values (complex)
};

// Calculate spectral function A(ω) for operator O using all eigenstates
// A(ω) = Σ_n,m |<n|O|m>|^2 δ(ω - (E_n - E_m)) * weight(m)
// where δ is approximated by a broadening function (Gaussian or Lorentzian)
struct SpectralFunctionData {
    std::vector<double> frequencies;     // ω values
    std::vector<std::complex<double>> spectral_function;  // A(ω) values
};


// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Count number of bits set in an integer (population count)
 * @param x Integer to count bits in
 * @return Number of bits set to 1
 */
inline uint64_t popcount(uint64_t x) {
    return __builtin_popcountll(x);
}

/**
 * Generate all basis states with exactly n_up bits set
 * Returns states in lexicographic order
 * @param n_bits Total number of bits
 * @param n_up Number of bits that should be 1
 * @return Vector of basis states (as integers)
 */
inline std::vector<uint64_t> generateFixedSzBasis(uint64_t n_bits, int64_t n_up) {
    std::vector<uint64_t> basis;
    if (n_up > n_bits) return basis;
    
    // Start with lowest n_up bits set
    uint64_t state = (1ULL << n_up) - 1;
    uint64_t limit = 1ULL << n_bits;
    
    while (state < limit) {
        basis.push_back(state);
        
        // Gosper's hack: generate next combination
        uint64_t c = state & -state;  // rightmost bit
        uint64_t r = state + c;        // add 1 to rightmost bit
        uint64_t new_state = (((r ^ state) >> 2) / c) | r;
        
        if (new_state >= limit) break;
        state = new_state;
    }
    
    return basis;
}

/**
 * Build inverse mapping: basis state (integer) -> index in fixed-Sz basis
 * @param basis Vector of basis states
 * @return Unordered map from state to index
 */
inline std::unordered_map<uint64_t, int> buildBasisIndexMap(const std::vector<uint64_t>& basis) {
    std::unordered_map<uint64_t, int> index_map;
    for (size_t i = 0; i < basis.size(); ++i) {
        index_map[basis[i]] = i;
    }
    return index_map;
}

/**
 * Apply a permutation to a basis state (represented as an integer)
 * @param basis The basis state as a bit string
 * @param perm The permutation to apply
 * @return The permuted basis state
 */
inline uint64_t applyPermutation(uint64_t basis, const std::vector<int>& perm) {
    uint64_t result = 0;
    for (size_t i = 0; i < perm.size(); ++i) {
        result |= ((basis >> perm[i]) & 1) << i;
    }
    return result;
}

// ============================================================================
// Symmetry Data Structures
// ============================================================================

/**
 * Structure to hold symmetry sector metadata
 */
struct SectorMetadata {
    uint64_t sector_id;
    std::vector<int> quantum_numbers;
    std::vector<Complex> phase_factors;
    uint64_t dimension;  // Computed during basis generation
    
    SectorMetadata() : sector_id(0), dimension(0) {}
};

/**
 * Structure to hold complete symmetry group information
 * Loads data from JSON files produced by automorphism_finder.py
 */
struct SymmetryGroupInfo {
    uint64_t num_generators;
    std::vector<int> generator_orders;
    std::vector<std::vector<int>> generators;
    std::vector<std::vector<int>> max_clique;
    std::vector<std::vector<int>> power_representation;
    std::vector<SectorMetadata> sectors;
    
    /**
     * Load all symmetry information from directory
     * @param dir Directory containing automorphism_results/
     */
    void loadFromDirectory(const std::string& dir) {
        std::string auto_dir = dir + "/automorphism_results";
        loadMaxClique(auto_dir);
        loadMinimalGenerators(auto_dir);
        loadSectorMetadata(auto_dir);
        computePowerRepresentation();
        
        std::cout << "Loaded symmetry group information:" << std::endl;
        std::cout << "  Number of generators: " << num_generators << std::endl;
        std::cout << "  Generator orders: ";
        for (uint64_t order : generator_orders) std::cout << order << " ";
        std::cout << std::endl;
        std::cout << "  Number of symmetry sectors: " << sectors.size() << std::endl;
        std::cout << "  Group size: " << max_clique.size() << std::endl;
    }
    
private:
    void loadMaxClique(const std::string& auto_dir) {
        std::string filepath = auto_dir + "/max_clique.json";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open max_clique.json: " + filepath);
        }
        
        std::string json((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
        max_clique = parseJsonIntArrays(json);
        
        std::cout << "Loaded max clique with " << max_clique.size() << " automorphisms" << std::endl;
    }
    
    void loadMinimalGenerators(const std::string& auto_dir) {
        std::string filepath = auto_dir + "/minimal_generators.json";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open minimal_generators.json: " + filepath);
        }
        
        std::string json((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
        
        // Parse JSON manually (simple parser for our specific format)
        generators.clear();
        generator_orders.clear();
        
        size_t pos = 0;
        while ((pos = json.find("\"permutation\":", pos)) != std::string::npos) {
            pos += 14;
            size_t start = json.find('[', pos);
            size_t end = json.find(']', start);
            if (start == std::string::npos || end == std::string::npos) break;
            
            std::string perm_str = json.substr(start + 1, end - start - 1);
            std::vector<int> perm;
            std::istringstream iss(perm_str);
            std::string num;
            while (std::getline(iss, num, ',')) {
                num.erase(0, num.find_first_not_of(" \t\n\r"));
                num.erase(num.find_last_not_of(" \t\n\r") + 1);
                if (!num.empty()) perm.push_back(std::stoi(num));
            }
            generators.push_back(perm);
            
            // Find order
            size_t order_pos = json.find("\"order\":", pos);
            if (order_pos != std::string::npos && order_pos < json.find('{', pos)) {
                order_pos += 8;
                size_t comma_pos = json.find_first_of(",}", order_pos);
                std::string order_str = json.substr(order_pos, comma_pos - order_pos);
                order_str.erase(0, order_str.find_first_not_of(" \t\n\r"));
                order_str.erase(order_str.find_last_not_of(" \t\n\r") + 1);
                generator_orders.push_back(std::stoi(order_str));
            }
            
            pos = end;
        }
        
        num_generators = generators.size();
        std::cout << "Loaded " << num_generators << " minimal generators" << std::endl;
    }
    
    void loadSectorMetadata(const std::string& auto_dir) {
        std::string filepath = auto_dir + "/sector_metadata.json";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open sector_metadata.json: " + filepath);
        }
        
        std::string json((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
        
        sectors.clear();
        
        // Parse sectors array
        size_t sectors_pos = json.find("\"sectors\":");
        if (sectors_pos == std::string::npos) {
            throw std::runtime_error("Could not find 'sectors' in metadata");
        }
        
        size_t pos = json.find('[', sectors_pos);
        if (pos == std::string::npos) {
            throw std::runtime_error("Could not find sectors array start");
        }
        pos++; // Move past the opening bracket
        
        // Find the end of the sectors array by counting brackets
        size_t array_end = pos;
        uint64_t bracket_depth = 1;
        while (array_end < json.size() && bracket_depth > 0) {
            if (json[array_end] == '[') bracket_depth++;
            else if (json[array_end] == ']') bracket_depth--;
            array_end++;
        }
        
        // Parse each sector object
        while (pos < array_end) {
            // Skip whitespace and commas
            while (pos < array_end && (json[pos] == ' ' || json[pos] == '\n' || 
                   json[pos] == '\r' || json[pos] == '\t' || json[pos] == ',')) {
                pos++;
            }
            
            if (pos >= array_end || json[pos] == ']') break;
            
            if (json[pos] == '{') {
                // Find the matching closing brace for this sector
                size_t sector_start = pos;
                size_t sector_end = sector_start + 1;
                uint64_t brace_depth = 1;
                while (sector_end < array_end && brace_depth > 0) {
                    if (json[sector_end] == '{') brace_depth++;
                    else if (json[sector_end] == '}') brace_depth--;
                    sector_end++;
                }
                
                // Extract this sector's JSON
                std::string sector_json = json.substr(sector_start, sector_end - sector_start);
                
                // Parse one sector
                SectorMetadata sector;
                
                // Find sector_id
                size_t id_pos = sector_json.find("\"sector_id\":");
                if (id_pos != std::string::npos) {
                    id_pos += 12;
                    size_t comma = sector_json.find_first_of(",}", id_pos);
                    std::string id_str = sector_json.substr(id_pos, comma - id_pos);
                    id_str.erase(0, id_str.find_first_not_of(" \t\n\r"));
                    sector.sector_id = std::stoi(id_str);
                }
                
                // Find quantum_numbers array
                size_t qn_pos = sector_json.find("\"quantum_numbers\":");
                if (qn_pos != std::string::npos) {
                    size_t qn_start = sector_json.find('[', qn_pos);
                    size_t qn_end = sector_json.find(']', qn_start);
                    std::string qn_str = sector_json.substr(qn_start + 1, qn_end - qn_start - 1);
                    
                    std::istringstream iss(qn_str);
                    std::string num;
                    while (std::getline(iss, num, ',')) {
                        num.erase(0, num.find_first_not_of(" \t\n\r"));
                        num.erase(num.find_last_not_of(" \t\n\r") + 1);
                        if (!num.empty()) sector.quantum_numbers.push_back(std::stoi(num));
                    }
                }
                
                // Parse phase_factors
                size_t pf_pos = sector_json.find("\"phase_factors\":");
                if (pf_pos != std::string::npos) {
                    size_t pf_start = sector_json.find('[', pf_pos);
                    size_t pf_end = sector_json.find(']', pf_start);
                    
                    std::string pf_section = sector_json.substr(pf_start, pf_end - pf_start + 1);
                    size_t temp_pos = 0;
                    while ((temp_pos = pf_section.find('{', temp_pos)) != std::string::npos) {
                        size_t real_pos = pf_section.find("\"real\":", temp_pos);
                        size_t imag_pos = pf_section.find("\"imag\":", temp_pos);
                        
                        if (real_pos != std::string::npos && imag_pos != std::string::npos) {
                            real_pos += 7;
                            size_t real_end = pf_section.find_first_of(",}", real_pos);
                            std::string real_str = pf_section.substr(real_pos, real_end - real_pos);
                            real_str.erase(0, real_str.find_first_not_of(" \t\n\r"));
                            real_str.erase(real_str.find_last_not_of(" \t\n\r") + 1);
                            
                            imag_pos += 7;
                            size_t imag_end = pf_section.find_first_of(",}", imag_pos);
                            std::string imag_str = pf_section.substr(imag_pos, imag_end - imag_pos);
                            imag_str.erase(0, imag_str.find_first_not_of(" \t\n\r"));
                            imag_str.erase(imag_str.find_last_not_of(" \t\n\r") + 1);
                            
                            Complex phase(std::stod(real_str), std::stod(imag_str));
                            sector.phase_factors.push_back(phase);
                        }
                        
                        temp_pos = pf_section.find('}', temp_pos) + 1;
                    }
                }
                
                sector.dimension = 0;  // Will be set during basis generation
                sectors.push_back(sector);
                
                // Move to the position after this sector
                pos = sector_end;
            } else {
                pos++;
            }
        }
        
        std::cout << "Loaded metadata for " << sectors.size() << " symmetry sectors" << std::endl;
    }
    
    void computePowerRepresentation() {
        power_representation.clear();
        
        // Use BFS to represent each automorphism as powers of generators
        for (const auto& automorphism : max_clique) {
            std::vector<int> powers = representAsGeneratorPowers(automorphism);
            power_representation.push_back(powers);
        }
        
        std::cout << "Computed power representation for all automorphisms" << std::endl;
    }
    
    std::vector<int> representAsGeneratorPowers(const std::vector<int>& automorphism) {
        if (generators.empty()) return std::vector<int>();
        
        uint64_t n = automorphism.size();
        std::vector<int> identity(n);
        for (uint64_t i = 0; i < n; ++i) identity[i] = i;
        
        // Check if it's identity
        if (automorphism == identity) {
            return std::vector<int>(generators.size(), 0);
        }
        
        // BFS to find representation
        struct State {
            std::vector<int> powers;
            std::vector<int> perm;
        };
        
        std::queue<State> queue;
        std::set<std::vector<int>> visited;
        
        State init;
        init.powers = std::vector<int>(generators.size(), 0);
        init.perm = identity;
        queue.push(init);
        visited.insert(identity);
        
        while (!queue.empty()) {
            State curr = queue.front();
            queue.pop();
            
            for (size_t g = 0; g < generators.size(); ++g) {
                std::vector<int> new_perm(n);
                for (uint64_t i = 0; i < n; ++i) {
                    new_perm[i] = curr.perm[generators[g][i]];
                }
                
                if (new_perm == automorphism) {
                    std::vector<int> result = curr.powers;
                    result[g]++;
                    return result;
                }
                
                if (visited.find(new_perm) == visited.end()) {
                    visited.insert(new_perm);
                    State next;
                    next.perm = new_perm;
                    next.powers = curr.powers;
                    next.powers[g]++;
                    
                    // Only explore if powers are reasonable
                    bool valid = true;
                    for (size_t i = 0; i < generators.size(); ++i) {
                        if (next.powers[i] >= generator_orders[i]) {
                            valid = false;
                            break;
                        }
                    }
                    
                    if (valid) queue.push(next);
                }
            }
        }
        
        return std::vector<int>();  // Not found
    }
    
    std::vector<std::vector<int>> parseJsonIntArrays(const std::string& json) {
        std::vector<std::vector<int>> result;
        size_t pos = json.find('[');
        if (pos == std::string::npos) return result;
        ++pos;
        
        while (pos < json.size()) {
            while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\t')) ++pos;
            if (pos >= json.size() || json[pos] == ']') break;
            
            if (json[pos] == '[') {
                size_t end = json.find(']', pos);
                if (end == std::string::npos) break;
                
                std::string array_str = json.substr(pos + 1, end - pos - 1);
                std::vector<int> array;
                std::istringstream iss(array_str);
                std::string num;
                
                while (std::getline(iss, num, ',')) {
                    num.erase(0, num.find_first_not_of(" \t\n\r"));
                    num.erase(num.find_last_not_of(" \t\n\r") + 1);
                    if (!num.empty()) array.push_back(std::stoi(num));
                }
                
                result.push_back(array);
                pos = end + 1;
            } else {
                ++pos;
            }
        }
        
        return result;
    }
};

// ============================================================================
// Operator Class
// ============================================================================

/**
 * Operator class that represents quantum operators through bit flip operations
 * Supports symmetry-adapted basis and block diagonalization
 */
class Operator {
public:
    using TransformFunction = std::function<std::pair<int, Complex>(int)>;
    
    // Public member variables
    std::vector<int> symmetrized_block_ham_sizes;
    SymmetryGroupInfo symmetry_info;
    
    // Constructor
    Operator(uint64_t n_bits, float spin_l) : n_bits_(n_bits), spin_l_(spin_l), matrixBuilt_(false) {}
    
    // Assignment operator
    Operator& operator=(const Operator& other) {
        if (this != &other) {
            n_bits_ = other.n_bits_;
            spin_l_ = other.spin_l_;
            transforms_ = other.transforms_;
            sparseMatrix_ = other.sparseMatrix_;
            matrixBuilt_ = other.matrixBuilt_;
            symmetrized_block_ham_sizes = other.symmetrized_block_ham_sizes;
            symmetry_info = other.symmetry_info;
        }
        return *this;
    }
    
    // ========================================================================
    // Core Operator Functions
    // ========================================================================
    
    void addTransform(TransformFunction transform) {
        transforms_.push_back(transform);
        matrixBuilt_ = false;
    }
    
    /**
     * Matrix-free apply: H|vec⟩ using on-the-fly transform evaluation
     * Faster and more memory-efficient than building sparse matrix
     * Parallelized with OpenMP for multi-core performance
     */
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        uint64_t dim = 1ULL << n_bits_;
        if (vec.size() != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input vector size mismatch");
        }
        
        std::vector<Complex> result(dim, Complex(0.0, 0.0));
        
        // Parallel reduction: each thread accumulates to local buffer, then combine
        #pragma omp parallel if(dim > 1024)
        {
            std::vector<Complex> local_result(dim, Complex(0.0, 0.0));
            
            #pragma omp for schedule(static) nowait
            for (uint64_t i = 0; i < dim; ++i) {
                Complex coeff = vec[i];
                if (std::abs(coeff) < 1e-15) continue;
                
                // Prefetch next input (helps if vec is large)
                if (i + 8 < dim) {
                    __builtin_prefetch(&vec[i + 8], 0, 1);
                }
                
                // Apply all transforms to this basis state
                for (const auto& transform : transforms_) {
                    auto [j, scalar] = transform(i);
                    if (j >= 0 && j < dim && std::abs(scalar) > 1e-15) {
                        local_result[j] += scalar * coeff;
                    }
                }
            }
            
            // Combine local results with critical section
            #pragma omp critical
            {
                for (uint64_t i = 0; i < dim; ++i) {
                    result[i] += local_result[i];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Matrix-free apply for raw arrays (optimized for Lanczos/Davidson solvers)
     * Parallelized with OpenMP and optimized memory access patterns
     */
    void apply(const Complex* in, Complex* out, size_t size) const {
        uint64_t dim = 1ULL << n_bits_;
        if (size != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input/output vector size mismatch");
        }
        
        // Zero output
        std::fill(out, out + dim, Complex(0.0, 0.0));
        
        // Parallel with thread-local buffers to avoid race conditions
        #pragma omp parallel if(dim > 1024)
        {
            std::vector<Complex> local_out(dim, Complex(0.0, 0.0));
            
            #pragma omp for schedule(static) nowait
            for (uint64_t i = 0; i < dim; ++i) {
                Complex coeff = in[i];
                if (std::abs(coeff) < 1e-15) continue;
                
                // Prefetch ahead
                if (i + 8 < dim) {
                    __builtin_prefetch(&in[i + 8], 0, 1);
                }
                
                // Apply all transforms
                for (const auto& transform : transforms_) {
                    auto [j, scalar] = transform(i);
                    if (j >= 0 && j < dim && std::abs(scalar) > 1e-15) {
                        local_out[j] += scalar * coeff;
                    }
                }
            }
            
            // Reduce: combine thread-local results
            #pragma omp critical
            {
                for (uint64_t i = 0; i < dim; ++i) {
                    out[i] += local_out[i];
                }
            }
        }
    }
    
    /**
     * Original apply methods (use sparse matrix - kept for compatibility)
     * For matrix-free operation, use apply() instead
     */
    std::vector<Complex> apply_sparse(const std::vector<Complex>& vec) const {
        uint64_t dim = 1ULL << n_bits_;
        if (vec.size() != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input vector size mismatch");
        }
        
        buildSparseMatrix();
        Eigen::VectorXcd eigenVec(dim);
        for (uint64_t i = 0; i < dim; ++i) {
            eigenVec(i) = vec[i];
        }
        
        Eigen::VectorXcd result = sparseMatrix_ * eigenVec;
        
        std::vector<Complex> resultVec(dim);
        for (uint64_t i = 0; i < dim; ++i) {
            resultVec[i] = result(i);
        }
        return resultVec;
    }
    
    void apply_sparse(const Complex* in, Complex* out, size_t size) const {
        uint64_t dim = 1ULL << n_bits_;
        if (size != static_cast<size_t>(dim)) {
            throw std::invalid_argument("Input/output vector size mismatch");
        }
        
        buildSparseMatrix();
        Eigen::Map<const Eigen::VectorXcd> eigenIn(in, dim);
        Eigen::Map<Eigen::VectorXcd> eigenOut(out, dim);
        eigenOut = sparseMatrix_ * eigenIn;
    }
    
    void buildSparseMatrix() const {
        if (matrixBuilt_) return;
        
        uint64_t dim = 1ULL << n_bits_;
        sparseMatrix_.resize(dim, dim);
        
        std::vector<Eigen::Triplet<Complex>> triplets;
        for (uint64_t i = 0; i < dim; ++i) {
            for (const auto& transform : transforms_) {
                auto [j, scalar] = transform(i);
                if (j >= 0 && j < dim) {
                    triplets.emplace_back(j, i, scalar);
                }
            }
        }
        
        sparseMatrix_.setFromTriplets(triplets.begin(), triplets.end());
        matrixBuilt_ = true;
    }
    
    Eigen::SparseMatrix<Complex> getSparseMatrix() const {
        buildSparseMatrix();
        return sparseMatrix_;
    }
    
    // ========================================================================
    // File I/O for Hamiltonian Parameters
    // ========================================================================
    
    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        uint64_t numLines;
        std::string m;
        iss >> m >> numLines;
        
        for (uint64_t i = 0; i < 3; ++i) std::getline(file, line);
        
        uint64_t lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            uint64_t Op, indx;
            double E, F;
            
            if (!(lineStream >> Op >> indx >> E >> F)) continue;
            Complex coeff(E, F);
            if (std::abs(coeff) < 1e-15) continue;
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (Op == 2) {
                    return {basis, coeff * double(spin_l_) * pow(-1, (basis >> indx) & 1)};
                } else {
                    if (((basis >> indx) & 1) != Op) {
                        uint64_t flipped_basis = basis ^ (1 << indx);
                        return {flipped_basis, coeff};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
            
            lineCount++;
        }
    }
    
    void loadFromInterAllFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }
        
        std::string line;
        std::getline(file, line);
        std::getline(file, line);
        std::istringstream iss(line);
        uint64_t numLines;
        std::string m;
        iss >> m >> numLines;
        
        for (uint64_t i = 0; i < 3; ++i) std::getline(file, line);
        
        uint64_t lineCount = 0;
        while (std::getline(file, line) && lineCount < numLines) {
            std::istringstream lineStream(line);
            uint64_t Op_i, indx_i, Op_j, indx_j;
            double E, F;
            
            if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
            Complex coeff(E, F);
            if (std::abs(coeff) < 1e-15) continue;

            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t bit_i = (basis >> indx_i) & 1;
                uint64_t bit_j = (basis >> indx_j) & 1;
                
                if (Op_i == 2 && Op_j == 2) {
                    double sign_i = pow(-1, bit_i);
                    double sign_j = pow(-1, bit_j);
                    return {basis, coeff * double(spin_l_) * double(spin_l_) * sign_i * sign_j};
                }
                
                Complex local_coeff = coeff;
                uint64_t new_basis = basis;
                bool valid = true;
                
                if (Op_i != 2) {
                    if (bit_i != Op_i) {
                        new_basis ^= (1 << indx_i);
                    } else {
                        valid = false;
                    }
                } else {
                    local_coeff *= double(spin_l_) * pow(-1, bit_i);
                }
                
                if (valid && Op_j != 2) {
                    uint64_t new_bit_j = (new_basis >> indx_j) & 1;
                    if (new_bit_j != Op_j) {
                        new_basis ^= (1 << indx_j);
                    } else {
                        valid = false;
                    }
                } else if (valid) {
                    uint64_t new_bit_j = (new_basis >> indx_j) & 1;
                    local_coeff *= double(spin_l_) * pow(-1, new_bit_j);
                }
                
                if (valid) {
                    return {new_basis, local_coeff};
                }
                return {basis, Complex(0.0, 0.0)};
            });
            
            lineCount++;
        }
    }
    
    void loadonebodycorrelation(const uint64_t Op, const uint64_t indx) {
        addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
            if (Op == 2) {
                return {basis, Complex(double(spin_l_) * pow(-1, (basis >> indx) & 1), 0.0)};
            } else {
                if (((basis >> indx) & 1) != Op) {
                    uint64_t flipped_basis = basis ^ (1 << indx);
                    return {flipped_basis, Complex(1.0, 0.0)};
                }
            }
            return {basis, Complex(0.0, 0.0)};
        });
    }
    
    void loadtwobodycorrelation(const uint64_t Op1, const uint64_t indx1, const uint64_t Op2, const uint64_t indx2) {
        addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
            uint64_t bit1 = (basis >> indx1) & 1;
            uint64_t bit2 = (basis >> indx2) & 1;
            
            Complex factor(1.0, 0.0);
            
            if (Op1 == 2 && Op2 == 2) {
                return {basis, Complex(double(spin_l_) * double(spin_l_) * pow(-1, bit1) * pow(-1, bit2), 0.0)};
            }
            
            uint64_t new_basis = basis;
            bool valid = true;
            
            if (Op1 != 2) {
                if (bit1 != Op1) {
                    new_basis ^= (1 << indx1);
                } else {
                    valid = false;
                }
            } else {
                factor *= Complex(double(spin_l_) * pow(-1, bit1), 0.0);
            }
            
            if (valid && Op2 != 2) {
                uint64_t new_bit2 = (new_basis >> indx2) & 1;
                if (new_bit2 != Op2) {
                    new_basis ^= (1 << indx2);
                } else {
                    valid = false;
                }
            } else if (valid) {
                uint64_t new_bit2 = (new_basis >> indx2) & 1;
                factor *= Complex(double(spin_l_) * pow(-1, new_bit2), 0.0);
            }
            
            if (valid) return {new_basis, factor};
            return {basis, Complex(0.0, 0.0)};
        });
    }
    
    std::vector<Complex> read_sym_basis(uint64_t index, const std::string& dir) const {
        return readSymBasisVector(dir, index);
    }
    
    // ========================================================================
    // Symmetry-Adapted Basis Generation
    // ========================================================================
    
    // HDF5-based methods (recommended for better file management)
    
    /**
     * Generate symmetrized basis vectors using HDF5 storage
     * More efficient than individual text files for large systems
     */
    void generateSymmetrizedBasisHDF5(const std::string& dir) {
        std::cout << "\n=== Generating Symmetrized Basis (HDF5) ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        // Create HDF5 file
        std::string hdf5_file = HDF5SymmetryIO::createFile(dir);
        
        // Generate basis for each sector
        const size_t dim = 1ULL << n_bits_;
        size_t total_written = 0;
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (uint64_t qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            std::set<size_t> processed_orbits;
            size_t sector_basis_count = 0;
            
            for (size_t basis = 0; basis < dim; ++basis) {
                size_t progress_interval = dim / 20;
                if (progress_interval > 0 && basis % progress_interval == 0 && dim > 20) {
                    std::cout << "\r  Progress: " << (100 * basis / dim) << "%" << std::flush;
                }
                
                // Check if this basis state's orbit was already processed
                size_t orbit_rep = getOrbitRepresentative(basis);
                if (processed_orbits.count(orbit_rep)) continue;
                processed_orbits.insert(orbit_rep);
                
                // Create symmetrized vector for this sector
                std::vector<Complex> sym_vec = createSymmetrizedVector(
                    basis, sector.quantum_numbers, sector.phase_factors);
                
                // Check if vector is valid (non-zero norm)
                double norm_sq = 0.0;
                for (const auto& v : sym_vec) norm_sq += std::norm(v);
                
                if (norm_sq > 1e-10) {
                    // Normalize
                    double norm = std::sqrt(norm_sq);
                    for (auto& v : sym_vec) v /= norm;
                    
                    // Save vector to HDF5
                    HDF5SymmetryIO::saveBasisVector(hdf5_file, total_written, sym_vec);
                    sector_basis_count++;
                    total_written++;
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector_basis_count;
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector_basis_count << " basis vectors" << std::endl;
        }
        
        // Save block sizes to HDF5
        HDF5SymmetryIO::saveSectorDimensions(hdf5_file, 
            std::vector<uint64_t>(symmetrized_block_ham_sizes.begin(), 
                                  symmetrized_block_ham_sizes.end()));
        
        std::cout << "\nTotal symmetrized basis vectors: " << total_written << std::endl;
        std::cout << "=== Symmetrized Basis Generation Complete (HDF5) ===" << std::endl;
    }
    
    /**
     * Build and save block-diagonal Hamiltonian matrices using HDF5
     * All blocks are stored in a single HDF5 file for efficient access
     * OPTIMIZED: Caches basis vectors, parallelizes columns, minimizes I/O
     */
    void buildAndSaveSymmetrizedBlocksHDF5(const std::string& dir) {
        std::cout << "\n=== Building Symmetrized Hamiltonian Blocks (HDF5) ===" << std::endl;
        
        std::string hdf5_file = dir + "/symmetry_data.h5";
        
        // Load block sizes from HDF5 if not already loaded
        if (symmetrized_block_ham_sizes.empty()) {
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
                
        uint64_t block_start = 0;
        const size_t dim = 1ULL << n_bits_;
        
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            uint64_t block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                std::cout << "  Skipping empty block " << block_idx << std::endl;
                continue;
            }
            
            std::cout << "  Building block " << block_idx << " ("
                      << block_size << "x" << block_size << ")..." << std::flush;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // OPTIMIZATION 1: Load ALL basis vectors for this block ONCE (batch I/O)
            std::cout << " [loading basis]" << std::flush;
            std::vector<std::vector<Complex>> basis_vectors(block_size);
            for (uint64_t i = 0; i < block_size; ++i) {
                basis_vectors[i] = HDF5SymmetryIO::loadBasisVector(hdf5_file, block_start + i, dim);
            }
            
            // OPTIMIZATION 2: Parallel computation over columns
            std::cout << " [computing]" << std::flush;
            std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(block_size);
            
            #pragma omp parallel for schedule(dynamic, 1) if(block_size > 4)
            for (uint64_t col = 0; col < block_size; ++col) {
                const auto& basis_col = basis_vectors[col];
                
                // Apply Hamiltonian: H|ψ_j⟩ (matrix-free)
                std::vector<Complex> H_psi_j = apply(basis_col);
                
                // Compute matrix elements with all rows (row-wise)
                // OPTIMIZATION 3: Use conjugate symmetry for Hermitian operators
                for (uint64_t row = 0; row <= col; ++row) {  // Only compute lower triangle + diagonal
                    const auto& basis_row = basis_vectors[row];
                    
                    // H_ij = ⟨ψ_i|H|ψ_j⟩ = Σ_k ψ_i*(k) * (H|ψ_j⟩)(k)
                    Complex element(0.0, 0.0);
                    for (uint64_t k = 0; k < dim; ++k) {
                        if (std::abs(basis_row[k]) > 1e-15 && std::abs(H_psi_j[k]) > 1e-15) {
                            element += std::conj(basis_row[k]) * H_psi_j[k];
                        }
                    }
                    
                    if (std::abs(element) > 1e-12) {
                        thread_triplets[col].emplace_back(row, col, element);
                        // Add conjugate transpose element (if not diagonal)
                        if (row != col) {
                            thread_triplets[col].emplace_back(col, row, std::conj(element));
                        }
                    }
                }
            }
            
            // OPTIMIZATION 4: Merge triplets efficiently
            std::vector<Eigen::Triplet<Complex>> triplets;
            size_t total_nnz = 0;
            for (const auto& t : thread_triplets) total_nnz += t.size();
            triplets.reserve(total_nnz);
            
            for (auto& t : thread_triplets) {
                triplets.insert(triplets.end(), 
                               std::make_move_iterator(t.begin()), 
                               std::make_move_iterator(t.end()));
            }
            
            // Create sparse matrix
            Eigen::SparseMatrix<Complex> block(block_size, block_size);
            block.setFromTriplets(triplets.begin(), triplets.end());
            block.makeCompressed();
            
            // Save to HDF5
            HDF5SymmetryIO::saveBlockMatrix(hdf5_file, block_idx, block);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            double fill_percent = 100.0 * triplets.size() / (block_size * block_size);
            std::cout << " done (" << triplets.size() << " nnz, "
                      << std::fixed << std::setprecision(2) << fill_percent << "% fill, "
                      << duration.count() << " ms)" << std::endl;
            
            block_start += block_size;
        }
        
        std::cout << "=== Block Construction Complete (HDF5) ===" << std::endl;
    }
    
    /**
     * Load a specific symmetrized block matrix from HDF5
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockHDF5(const std::string& dir, size_t block_idx) {
        std::string hdf5_file = dir + "/symmetry_data.h5";
        return HDF5SymmetryIO::loadBlockMatrix(hdf5_file, block_idx);
    }
    
    /**
     * Load all symmetrized blocks from HDF5
     */
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocksHDF5(const std::string& dir) {
        std::string hdf5_file = dir + "/symmetry_data.h5";
        
        // Load block sizes if not already loaded
        if (symmetrized_block_ham_sizes.empty()) {
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            if (symmetrized_block_ham_sizes[i] > 0) {
                blocks.push_back(HDF5SymmetryIO::loadBlockMatrix(hdf5_file, i));
            } else {
                blocks.emplace_back(0, 0);
            }
        }
        
        return blocks;
    }
    
    // ========================================================================
    // Legacy text-based methods (kept for backward compatibility)
    // ========================================================================
    
    /**
     * Generate symmetrized basis vectors for all symmetry sectors
     * Uses symmetry group information to project onto irreducible representations
     */
    void generateSymmetrizedBasis(const std::string& dir) {
        std::cout << "\n=== Generating Symmetrized Basis ===" << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        // Setup output directory
        std::string sym_basis_dir = dir + "/sym_basis";
        safe_system_call("mkdir -p " + sym_basis_dir);
        
        // Generate basis for each sector
        const size_t dim = 1ULL << n_bits_;
        size_t total_written = 0;
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (uint64_t qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            std::set<size_t> processed_orbits;
            size_t sector_basis_count = 0;
            
            for (size_t basis = 0; basis < dim; ++basis) {
                size_t progress_interval = dim / 20;
                if (progress_interval > 0 && basis % progress_interval == 0 && dim > 20) {
                    std::cout << "\r  Progress: " << (100 * basis / dim) << "%" << std::flush;
                }
                
                // Check if this basis state's orbit was already processed
                size_t orbit_rep = getOrbitRepresentative(basis);
                if (processed_orbits.count(orbit_rep)) continue;
                processed_orbits.insert(orbit_rep);
                
                // Create symmetrized vector for this sector
                std::vector<Complex> sym_vec = createSymmetrizedVector(
                    basis, sector.quantum_numbers, sector.phase_factors);
                
                // Check if vector is valid (non-zero norm)
                double norm_sq = 0.0;
                for (const auto& v : sym_vec) norm_sq += std::norm(v);
                
                if (norm_sq > 1e-10) {
                    // Normalize
                    double norm = std::sqrt(norm_sq);
                    for (auto& v : sym_vec) v /= norm;
                    
                    // Save vector
                    saveSymBasisVector(sym_basis_dir, total_written, sym_vec);
                    sector_basis_count++;
                    total_written++;
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector_basis_count;
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector_basis_count << " basis vectors" << std::endl;
        }
        
        // Save block sizes
        saveBlockSizes(dir);
        
        std::cout << "\nTotal symmetrized basis vectors: " << total_written << std::endl;
        std::cout << "=== Symmetrized Basis Generation Complete ===" << std::endl;
    }
    
    /**
     * Build and save block-diagonal Hamiltonian matrices
     * Each block corresponds to one symmetry sector
     */
    void buildAndSaveSymmetrizedBlocks(const std::string& dir) {
        std::cout << "\n=== Building Symmetrized Hamiltonian Blocks ===" << std::endl;
        
        loadBlockSizesIfNeeded(dir);
        
        std::string block_dir = dir + "/sym_blocks";
        safe_system_call("mkdir -p " + block_dir);
        
        uint64_t block_start = 0;
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            uint64_t block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                std::cout << "  Block " << block_idx << ": empty (skipping)" << std::endl;
                continue;
            }
            
            std::cout << "  Building block " << block_idx << " ("
                      << block_size << "x" << block_size << ")..." << std::flush;
            
            buildSingleBlock(dir, block_dir, block_idx, block_start, block_size);
            
            block_start += block_size;
        }
        
        std::cout << "=== Block Construction Complete ===" << std::endl;
    }
    
    /**
     * Load a specific symmetrized block matrix from disk
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlock(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open block file: " + filepath);
        }
        
        // Read dimensions as int to match what was written
        int rows_int, cols_int;
        size_t nnz;
        file.read(reinterpret_cast<char*>(&rows_int), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols_int), sizeof(int));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(size_t));
        
        uint64_t rows = rows_int;
        uint64_t cols = cols_int;
        
        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(nnz);
        
        for (size_t i = 0; i < nnz; ++i) {
            int row_int, col_int;
            Complex val;
            file.read(reinterpret_cast<char*>(&row_int), sizeof(int));
            file.read(reinterpret_cast<char*>(&col_int), sizeof(int));
            file.read(reinterpret_cast<char*>(&val), sizeof(Complex));
            triplets.emplace_back(row_int, col_int, val);
        }
        
        Eigen::SparseMatrix<Complex> matrix(rows, cols);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        
        return matrix;
    }
    
    /**
     * Load a block by its index
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockByIndex(const std::string& dir, size_t block_idx) {
        loadBlockSizesIfNeeded(dir);
        
        if (block_idx >= symmetrized_block_ham_sizes.size()) {
            throw std::runtime_error("Block index out of range");
        }
        
        if (symmetrized_block_ham_sizes[block_idx] == 0) {
            return Eigen::SparseMatrix<Complex>(0, 0);
        }
        
        std::string filepath = dir + "/sym_blocks/block_" + std::to_string(block_idx) + ".dat";
        return loadSymmetrizedBlock(filepath);
    }
    
    /**
     * Load all symmetrized blocks
     */
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocks(const std::string& dir) {
        loadBlockSizesIfNeeded(dir);
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            if (symmetrized_block_ham_sizes[i] > 0) {
                blocks.push_back(loadSymmetrizedBlockByIndex(dir, i));
            } else {
                blocks.push_back(Eigen::SparseMatrix<Complex>(0, 0));
            }
        }
        
        return blocks;
    }
    
protected:
    // Member variables (protected so derived classes can access)
    std::vector<TransformFunction> transforms_;
    uint64_t n_bits_;
    float spin_l_;
    mutable Eigen::SparseMatrix<Complex> sparseMatrix_;
    mutable bool matrixBuilt_;
    
    const std::array<std::array<double, 4>, 3> operators_ = {
        {{0, 1, 0, 0}, {0, 0, 1, 0}, {1, 0, 0, -1}}
    };
    
private:
    // ========================================================================
    // Private Helper Functions
    // ========================================================================
    
    size_t getOrbitRepresentative(size_t basis) const {
        size_t rep = basis;
        for (const auto& perm : symmetry_info.max_clique) {
            size_t permuted = applyPermutation(basis, perm);
            if (permuted < rep) rep = permuted;
        }
        return rep;
    }
    
    std::vector<Complex> createSymmetrizedVector(
        size_t basis,
        const std::vector<int>& quantum_numbers,
        const std::vector<Complex>& phase_factors) const {
        
        const size_t dim = 1ULL << n_bits_;
        std::vector<Complex> result(dim, Complex(0.0, 0.0));
        
        // Apply symmetry projection: |ψ_q⟩ = (1/|G|) Σ_g χ_q(g)* g|basis⟩
        for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
            const auto& perm = symmetry_info.max_clique[g];
            const auto& powers = symmetry_info.power_representation[g];
            
            // Compute character: χ_q(g) = exp(2πi Σ_k q_k * n_k / order_k)
            Complex character(1.0, 0.0);
            for (size_t k = 0; k < powers.size(); ++k) {
                Complex phase = phase_factors[k];
                for (uint64_t p = 0; p < powers[k]; ++p) {
                    character *= phase;
                }
            }
            
            size_t permuted_basis = applyPermutation(basis, perm);
            result[permuted_basis] += std::conj(character);
        }
        
        // Normalization factor: 1/√|G|
        double norm_factor = 1.0 / std::sqrt(symmetry_info.max_clique.size());
        for (auto& v : result) v *= norm_factor;
        
        return result;
    }
    
    void saveSymBasisVector(const std::string& dir, size_t index, const std::vector<Complex>& vec) const {
        std::string filepath = dir + "/sym_basis" + std::to_string(index) + ".dat";
        std::ofstream file(filepath);
        
        for (size_t i = 0; i < vec.size(); ++i) {
            if (std::abs(vec[i]) > 1e-12) {
                file << i << " " << vec[i].real() << " " << vec[i].imag() << "\n";
            }
        }
    }
    
    std::vector<Complex> readSymBasisVector(const std::string& dir, size_t index) const {
        std::string filepath = dir + "/sym_basis/sym_basis" + std::to_string(index) + ".dat";
        std::ifstream file(filepath);
        
        std::vector<Complex> vec(1ULL << n_bits_, Complex(0.0, 0.0));
        
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            size_t idx;
            double real, imag;
            if (iss >> idx >> real >> imag) {
                vec[idx] = Complex(real, imag);
            }
        }
        
        return vec;
    }
    
    void saveBlockSizes(const std::string& dir) const {
        std::ofstream file(dir + "/sym_basis/sym_block_sizes.txt");
        for (uint64_t size : symmetrized_block_ham_sizes) {
            file << size << "\n";
        }
        
        std::cout << "Block sizes: ";
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            std::cout << symmetrized_block_ham_sizes[i];
            if (i < symmetrized_block_ham_sizes.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    void loadBlockSizesIfNeeded(const std::string& dir) {
        if (!symmetrized_block_ham_sizes.empty()) return;
        
        std::string filepath = dir + "/sym_basis/sym_block_sizes.txt";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open block sizes file. Run generateSymmetrizedBasis first.");
        }
        
        uint64_t size;
        while (file >> size) {
            symmetrized_block_ham_sizes.push_back(size);
        }
    }
    
    void buildSingleBlock(const std::string& dir, const std::string& block_dir,
                         size_t block_idx, uint64_t block_start, uint64_t block_size) {
        
        std::cout << "Block " << block_idx << " (size " << block_size << ")..." << std::flush;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // OPTIMIZATION 1: Load all basis vectors once
        const size_t dim = 1ULL << n_bits_;
        std::vector<std::vector<Complex>> basis_vectors(block_size);
        for (uint64_t i = 0; i < block_size; ++i) {
            basis_vectors[i] = readSymBasisVector(dir, block_start + i);
        }
        
        // OPTIMIZATION 2: Parallel computation with Hermitian symmetry
        std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(block_size);
        
        #pragma omp parallel for schedule(dynamic, 1) if(block_size > 4)
        for (uint64_t col = 0; col < block_size; ++col) {
            const auto& basis_col = basis_vectors[col];
            
            // Apply Hamiltonian (matrix-free)
            std::vector<Complex> h_basis_col(dim, Complex(0.0, 0.0));
            for (size_t k = 0; k < basis_col.size(); ++k) {
                if (std::abs(basis_col[k]) < 1e-12) continue;
                
                for (const auto& transform : transforms_) {
                    auto [j, scalar] = transform(k);
                    if (j >= 0 && j < (1 << n_bits_)) {
                        h_basis_col[j] += scalar * basis_col[k];
                    }
                }
            }
            
            // Compute matrix elements (use Hermitian symmetry)
            for (uint64_t row = 0; row <= col; ++row) {  // Only lower triangle + diagonal
                const auto& basis_row = basis_vectors[row];
                
                Complex element(0.0, 0.0);
                for (size_t k = 0; k < dim; ++k) {
                    if (std::abs(basis_row[k]) > 1e-15 && std::abs(h_basis_col[k]) > 1e-15) {
                        element += std::conj(basis_row[k]) * h_basis_col[k];
                    }
                }
                
                if (std::abs(element) > 1e-12) {
                    thread_triplets[col].emplace_back(row, col, element);
                    if (row != col) {
                        thread_triplets[col].emplace_back(col, row, std::conj(element));
                    }
                }
            }
        }
        
        // Merge triplets
        std::vector<Eigen::Triplet<Complex>> triplets;
        size_t total_nnz = 0;
        for (const auto& t : thread_triplets) total_nnz += t.size();
        triplets.reserve(total_nnz);
        
        for (auto& t : thread_triplets) {
            triplets.insert(triplets.end(), 
                           std::make_move_iterator(t.begin()), 
                           std::make_move_iterator(t.end()));
        }
        
        // Create and save sparse matrix
        Eigen::SparseMatrix<Complex> block(block_size, block_size);
        block.setFromTriplets(triplets.begin(), triplets.end());
        block.makeCompressed();
        
        std::string filename = block_dir + "/block_" + std::to_string(block_idx) + ".dat";
        std::ofstream file(filename, std::ios::binary);
        
        uint64_t rows = block_size, cols = block_size;
        size_t nnz = triplets.size();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(&nnz), sizeof(size_t));
        
        for (const auto& t : triplets) {
            uint64_t row = t.row(), col = t.col();
            Complex val = t.value();
            file.write(reinterpret_cast<const char*>(&row), sizeof(int));
            file.write(reinterpret_cast<const char*>(&col), sizeof(int));
            file.write(reinterpret_cast<const char*>(&val), sizeof(Complex));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << " done (" << nnz << " nnz, "
                  << std::fixed << std::setprecision(2)
                  << (100.0 * nnz / (block_size * block_size)) << "% fill, "
                  << duration.count() << " ms)" << std::endl;
    }
};

// ============================================================================
// Fixed Sz Operator Class
// ============================================================================

/**
 * Operator class for fixed total Sz sector
 * Restricts Hilbert space to states with fixed number of up spins
 * Reduces dimension from 2^N to C(N, N_up)
 */
class FixedSzOperator : public Operator {
protected:
    int64_t n_up_;  // Number of up spins (fixed Sz = N/2 - n_up for spin-1/2)
    std::vector<uint64_t> basis_states_;  // Basis states in fixed Sz sector
    std::unordered_map<uint64_t, int> state_to_index_;  // Map state -> index
    uint64_t fixed_sz_dim_;  // Dimension of fixed Sz sector
    mutable Eigen::SparseMatrix<Complex> fixed_sz_matrix_;  // Sparse matrix in fixed Sz basis
    mutable bool fixed_sz_matrix_built_;
    
public:
    /**
     * Constructor
     * @param n_bits Number of sites
     * @param spin_l Spin length (1/2 for spin-1/2)
     * @param n_up Number of up spins (determines Sz sector)
     */
    FixedSzOperator(uint64_t n_bits, float spin_l, int64_t n_up) 
        : Operator(n_bits, spin_l), 
          n_up_(n_up),
          fixed_sz_matrix_built_(false) {
        
        if (n_up > n_bits) {
            throw std::invalid_argument("Invalid n_up: must be between 0 and n_bits");
        }
        
        // Generate fixed Sz basis
        basis_states_ = generateFixedSzBasis(n_bits, n_up);
        state_to_index_ = buildBasisIndexMap(basis_states_);
        fixed_sz_dim_ = basis_states_.size();
        
        std::cout << "Fixed Sz basis: n_bits=" << n_bits 
                  << ", n_up=" << n_up 
                  << ", dimension=" << fixed_sz_dim_ << std::endl;
    }
    
    // Get dimension of fixed Sz sector
    uint64_t getFixedSzDim() const { return fixed_sz_dim_; }
    
    // Get basis states
    const std::vector<uint64_t>& getBasisStates() const { return basis_states_; }
    
    /**
     * Matrix-free apply for fixed Sz basis
     * More efficient than building matrix
     * Parallelized with OpenMP
     */
    std::vector<Complex> apply(const std::vector<Complex>& vec) const {
        if (vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
        }
        
        std::vector<Complex> result(fixed_sz_dim_, Complex(0.0, 0.0));
        
        // Parallel with thread-local buffers
        #pragma omp parallel if(fixed_sz_dim_ > 512)
        {
            std::vector<Complex> local_result(fixed_sz_dim_, Complex(0.0, 0.0));
            
            #pragma omp for schedule(static) nowait
            for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                Complex coeff = vec[i];
                if (std::abs(coeff) < 1e-15) continue;
                
                uint64_t basis_i = basis_states_[i];
                
                // Prefetch next basis state
                if (i + 4 < fixed_sz_dim_) {
                    __builtin_prefetch(&basis_states_[i + 4], 0, 1);
                    __builtin_prefetch(&vec[i + 4], 0, 1);
                }
                
                // Apply all transforms
                for (const auto& transform : transforms_) {
                    auto [j_state, scalar] = transform(basis_i);
                    
                    // Check if resulting state is in the fixed Sz sector
                    if (j_state >= 0 && popcount(j_state) == n_up_ && std::abs(scalar) > 1e-15) {
                        auto it = state_to_index_.find(j_state);
                        if (it != state_to_index_.end()) {
                            uint64_t j = it->second;
                            local_result[j] += scalar * coeff;
                        }
                    }
                }
            }
            
            // Reduce thread-local results
            #pragma omp critical
            {
                for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                    result[i] += local_result[i];
                }
            }
        }
        
        return result;
    }
    
    /**
     * Matrix-free apply for raw arrays (optimized for solvers)
     * Parallelized with OpenMP
     */
    void apply(const Complex* in, Complex* out, size_t size) const {
        if (size != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input/output vector size mismatch with fixed Sz dimension");
        }
        
        // Zero output
        std::fill(out, out + fixed_sz_dim_, Complex(0.0, 0.0));
        
        // Parallel with thread-local buffers
        #pragma omp parallel if(fixed_sz_dim_ > 512)
        {
            std::vector<Complex> local_out(fixed_sz_dim_, Complex(0.0, 0.0));
            
            #pragma omp for schedule(static) nowait
            for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                Complex coeff = in[i];
                if (std::abs(coeff) < 1e-15) continue;
                
                uint64_t basis_i = basis_states_[i];
                
                // Prefetch ahead
                if (i + 4 < fixed_sz_dim_) {
                    __builtin_prefetch(&basis_states_[i + 4], 0, 1);
                    __builtin_prefetch(&in[i + 4], 0, 1);
                }
                
                // Apply all transforms
                for (const auto& transform : transforms_) {
                    auto [j_state, scalar] = transform(basis_i);
                    
                    if (j_state >= 0 && popcount(j_state) == n_up_ && std::abs(scalar) > 1e-15) {
                        auto it = state_to_index_.find(j_state);
                        if (it != state_to_index_.end()) {
                            uint64_t j = it->second;
                            local_out[j] += scalar * coeff;
                        }
                    }
                }
            }
            
            // Reduce
            #pragma omp critical
            {
                for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
                    out[i] += local_out[i];
                }
            }
        }
    }
    
    /**
     * Apply operator to vector in fixed Sz basis (uses sparse matrix)
     * For matrix-free operation, use apply() instead
     */
    std::vector<Complex> apply_sparse(const std::vector<Complex>& vec) const {
        if (vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
        }
        
        buildFixedSzMatrix();
        
        Eigen::VectorXcd eigenVec(fixed_sz_dim_);
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            eigenVec(i) = vec[i];
        }
        
        Eigen::VectorXcd result = fixed_sz_matrix_ * eigenVec;
        
        std::vector<Complex> resultVec(fixed_sz_dim_);
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            resultVec[i] = result(i);
        }
        return resultVec;
    }
    
    /**
     * Apply operator to raw arrays in fixed Sz basis
     */
    void apply_sparse(const Complex* in, Complex* out, size_t size) const {
        if (size != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input/output vector size mismatch with fixed Sz dimension");
        }
        
        buildFixedSzMatrix();
        
        Eigen::Map<const Eigen::VectorXcd> eigenIn(in, fixed_sz_dim_);
        Eigen::Map<Eigen::VectorXcd> eigenOut(out, fixed_sz_dim_);
        eigenOut = fixed_sz_matrix_ * eigenIn;
    }
    
    /**
     * Build sparse matrix in fixed Sz basis
     * Only computes matrix elements between states in the same Sz sector
     */
    void buildFixedSzMatrix() const {
        if (fixed_sz_matrix_built_) return;
        
        fixed_sz_matrix_.resize(fixed_sz_dim_, fixed_sz_dim_);
        std::vector<Eigen::Triplet<Complex>> triplets;
        
        // For each basis state
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            uint64_t basis_i = basis_states_[i];
            
            // Apply all transforms
            for (const auto& transform : transforms_) {
                auto [j_state, scalar] = transform(basis_i);
                
                // Check if resulting state is in the fixed Sz sector
                if (j_state >= 0 && popcount(j_state) == n_up_) {
                    auto it = state_to_index_.find(j_state);
                    if (it != state_to_index_.end()) {
                        uint64_t j = it->second;
                        triplets.emplace_back(j, i, scalar);
                    }
                }
            }
        }
        
        fixed_sz_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        fixed_sz_matrix_built_ = true;
        
        std::cout << "Built fixed Sz matrix: " << fixed_sz_dim_ << "x" << fixed_sz_dim_ 
                  << " with " << triplets.size() << " non-zero elements" << std::endl;
    }
    
    /**
     * Get sparse matrix in fixed Sz basis
     */
    Eigen::SparseMatrix<Complex> getFixedSzMatrix() const {
        buildFixedSzMatrix();
        return fixed_sz_matrix_;
    }
    
    /**
     * Convert vector from full basis to fixed Sz basis
     * Projects out components not in the fixed Sz sector
     */
    std::vector<Complex> projectToFixedSz(const std::vector<Complex>& full_vec) const {
        uint64_t full_dim = 1ULL << n_bits_;
        if (full_vec.size() != static_cast<size_t>(full_dim)) {
            throw std::invalid_argument("Input vector size mismatch with full dimension");
        }
        
        std::vector<Complex> fixed_sz_vec(fixed_sz_dim_, Complex(0.0, 0.0));
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            uint64_t state = basis_states_[i];
            fixed_sz_vec[i] = full_vec[state];
        }
        return fixed_sz_vec;
    }
    
    /**
     * Convert vector from fixed Sz basis to full basis
     * Embeds into full Hilbert space with zeros outside the sector
     */
    std::vector<Complex> embedToFull(const std::vector<Complex>& fixed_sz_vec) const {
        if (fixed_sz_vec.size() != static_cast<size_t>(fixed_sz_dim_)) {
            throw std::invalid_argument("Input vector size mismatch with fixed Sz dimension");
        }
        
        uint64_t full_dim = 1ULL << n_bits_;
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));
        for (uint64_t i = 0; i < fixed_sz_dim_; ++i) {
            uint64_t state = basis_states_[i];
            full_vec[state] = fixed_sz_vec[i];
        }
        return full_vec;
    }
    
    /**
     * Override addTransform to invalidate fixed Sz matrix cache
     */
    void addTransform(TransformFunction transform) {
        Operator::addTransform(transform);
        fixed_sz_matrix_built_ = false;
    }
    
    /**
     * Generate symmetrized basis for fixed Sz sector with spatial symmetries
     * Combines Sz conservation with spatial symmetry group
     */
    void generateSymmetrizedBasisFixedSz(const std::string& dir) {
        std::cout << "\n=== Generating Symmetrized Basis (Fixed Sz) ===" << std::endl;
        std::cout << "Fixed Sz sector: n_up=" << n_up_ 
                  << ", dimension=" << fixed_sz_dim_ << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        // Setup output directory
        std::string sym_basis_dir = dir + "/sym_basis_fixed_sz";
        safe_system_call("mkdir -p " + sym_basis_dir);
        
        // Generate basis for each sector
        size_t total_written = 0;
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (uint64_t qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            std::set<uint64_t> processed_orbits;
            size_t sector_basis_count = 0;
            
            // Only iterate over fixed Sz basis states
            for (uint64_t basis_idx = 0; basis_idx < fixed_sz_dim_; ++basis_idx) {
                uint64_t basis = basis_states_[basis_idx];
                
                size_t progress_interval = fixed_sz_dim_ / 20;
                if (progress_interval > 0 && basis_idx % progress_interval == 0 && fixed_sz_dim_ > 20) {
                    std::cout << "\r  Progress: " << (100 * basis_idx / fixed_sz_dim_) << "%" << std::flush;
                }
                
                // Check if this basis state's orbit was already processed
                uint64_t orbit_rep = getOrbitRepresentativeFixedSz(basis);
                if (processed_orbits.count(orbit_rep)) continue;
                processed_orbits.insert(orbit_rep);
                
                // Create symmetrized vector for this sector
                std::vector<Complex> sym_vec = createSymmetrizedVectorFixedSz(
                    basis, sector.quantum_numbers, sector.phase_factors);
                
                // Check if vector is valid (non-zero norm)
                double norm_sq = 0.0;
                for (const auto& v : sym_vec) norm_sq += std::norm(v);
                
                if (norm_sq > 1e-10) {
                    // Normalize
                    double norm = std::sqrt(norm_sq);
                    for (auto& v : sym_vec) v /= norm;
                    
                    // Save vector
                    saveSymBasisVectorFixedSz(sym_basis_dir, total_written, sym_vec);
                    sector_basis_count++;
                    total_written++;
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector_basis_count;
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector_basis_count << " basis vectors" << std::endl;
        }
        
        // Save block sizes
        saveBlockSizesFixedSz(dir);
        
        std::cout << "\nTotal symmetrized basis vectors (Fixed Sz): " << total_written << std::endl;
        std::cout << "=== Symmetrized Basis Generation Complete ===" << std::endl;
    }
    
    // ========================================================================
    // HDF5-based methods for Fixed Sz (recommended)
    // ========================================================================
    
    /**
     * Generate symmetrized basis vectors using HDF5 storage (Fixed Sz)
     * More efficient than individual text files for large systems
     */
    void generateSymmetrizedBasisFixedSzHDF5(const std::string& dir) {
        std::cout << "\n=== Generating Symmetrized Basis (Fixed Sz, HDF5) ===" << std::endl;
        std::cout << "Fixed Sz sector: n_up=" << n_up_ 
                  << ", dimension=" << fixed_sz_dim_ << std::endl;
        
        // Load symmetry information
        symmetry_info.loadFromDirectory(dir);
        
        // Create HDF5 file
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        try {
            // Create file (overwrite if exists)
            H5::H5File file(hdf5_file, H5F_ACC_TRUNC);
            
            // Create groups
            file.createGroup("/metadata");
            file.createGroup("/basis");
            file.createGroup("/blocks");
            
            // Store n_up as metadata
            H5::DataSpace scalar_space(H5S_SCALAR);
            H5::Attribute n_up_attr = file.createAttribute("n_up", H5::PredType::NATIVE_INT64, scalar_space);
            n_up_attr.write(H5::PredType::NATIVE_INT64, &n_up_);
            n_up_attr.close();
            
            H5::Attribute fixed_sz_dim_attr = file.createAttribute("fixed_sz_dim", H5::PredType::NATIVE_UINT64, scalar_space);
            fixed_sz_dim_attr.write(H5::PredType::NATIVE_UINT64, &fixed_sz_dim_);
            fixed_sz_dim_attr.close();
            
            file.close();
            std::cout << "Created HDF5 file: " << hdf5_file << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to create HDF5 file: " + std::string(e.getCDetailMsg()));
        }
        
        // Generate basis for each sector
        size_t total_written = 0;
        symmetrized_block_ham_sizes.assign(symmetry_info.sectors.size(), 0);
        
        for (size_t sector_idx = 0; sector_idx < symmetry_info.sectors.size(); ++sector_idx) {
            const auto& sector = symmetry_info.sectors[sector_idx];
            
            std::cout << "\nProcessing sector " << (sector_idx + 1) << "/"
                      << symmetry_info.sectors.size() << " (QN: ";
            for (uint64_t qn : sector.quantum_numbers) std::cout << qn << " ";
            std::cout << ")" << std::endl;
            
            std::set<uint64_t> processed_orbits;  // PER-SECTOR tracking
            size_t sector_basis_count = 0;
            
            // Only iterate over fixed Sz basis states (THE KEY DIFFERENCE FROM REGULAR VERSION)
            for (uint64_t basis_idx = 0; basis_idx < fixed_sz_dim_; ++basis_idx) {
                uint64_t basis = basis_states_[basis_idx];
                
                size_t progress_interval = fixed_sz_dim_ / 20;
                if (progress_interval > 0 && basis_idx % progress_interval == 0 && fixed_sz_dim_ > 20) {
                    std::cout << "\r  Progress: " << (100 * basis_idx / fixed_sz_dim_) << "%" << std::flush;
                }
                
                // Compute orbit representative using FULL group
                uint64_t orbit_rep = basis;
                for (const auto& perm : symmetry_info.max_clique) {
                    uint64_t transformed = applyPermutation(basis, perm);
                    if (transformed < orbit_rep) {
                        orbit_rep = transformed;
                    }
                }
                
                // Check processed orbits for THIS SECTOR only
                if (processed_orbits.count(orbit_rep)) continue;
                processed_orbits.insert(orbit_rep);
                
                // Create symmetrized vector using the SAME formula as regular version
                // but only for fixed-Sz basis states
                std::vector<Complex> sym_vec(fixed_sz_dim_, Complex(0.0, 0.0));
                
                // Apply symmetry projection: |ψ_q⟩ = (1/|G|) Σ_g χ_q(g)* g|basis⟩
                for (size_t g = 0; g < symmetry_info.max_clique.size(); ++g) {
                    const auto& perm = symmetry_info.max_clique[g];
                    const auto& powers = symmetry_info.power_representation[g];
                    
                    // Compute character: χ_q(g) = exp(2πi Σ_k q_k * n_k / order_k)
                    Complex character(1.0, 0.0);
                    for (size_t k = 0; k < powers.size(); ++k) {
                        Complex phase = sector.phase_factors[k];
                        for (uint64_t p = 0; p < powers[k]; ++p) {
                            character *= phase;
                        }
                    }
                    
                    uint64_t permuted_basis = applyPermutation(basis, perm);
                    
                    // Check if permuted state is in fixed-Sz sector
                    if (popcount(permuted_basis) == n_up_) {
                        // Find index in fixed-Sz basis
                        auto it = state_to_index_.find(permuted_basis);
                        if (it != state_to_index_.end()) {
                            sym_vec[it->second] += std::conj(character);
                        }
                    }
                }
                
                // Normalize
                double norm_sq = 0.0;
                for (const auto& v : sym_vec) norm_sq += std::norm(v);
                
                if (norm_sq > 1e-10) {
                    double norm = std::sqrt(norm_sq);
                    for (auto& v : sym_vec) v /= norm;
                    
                    // Save vector to HDF5
                    HDF5SymmetryIO::saveBasisVector(hdf5_file, total_written, sym_vec);
                    sector_basis_count++;
                    total_written++;
                }
            }
            
            symmetrized_block_ham_sizes[sector_idx] = sector_basis_count;
            std::cout << "\r  Sector " << (sector_idx + 1) << " complete: "
                      << sector_basis_count << " basis vectors" << std::endl;
        }
        
        // Save block sizes to HDF5
        HDF5SymmetryIO::saveSectorDimensions(hdf5_file, 
            std::vector<uint64_t>(symmetrized_block_ham_sizes.begin(), 
                                  symmetrized_block_ham_sizes.end()));
        
        std::cout << "\nTotal symmetrized basis vectors (Fixed Sz): " << total_written << std::endl;
        std::cout << "Fixed-Sz sector dimension: " << fixed_sz_dim_ << std::endl;
        std::cout << "=== Symmetrized Basis Generation Complete (HDF5) ===" << std::endl;
    }
    
    /**
     * Build and save block-diagonal Hamiltonian matrices using HDF5 (Fixed Sz)
     * All blocks are stored in a single HDF5 file for efficient access
     * OPTIMIZED: Caches basis vectors, parallelizes columns, minimizes I/O
     */
    void buildAndSaveSymmetrizedBlocksFixedSzHDF5(const std::string& dir) {
        std::cout << "\n=== Building Symmetrized Hamiltonian Blocks (Fixed Sz, HDF5) ===" << std::endl;
        
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        
        // Load block sizes from HDF5 if not already loaded
        if (symmetrized_block_ham_sizes.empty()) {
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
                
        uint64_t block_start = 0;
        
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            uint64_t block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                std::cout << "Block " << block_idx << ": empty (skipped)" << std::endl;
                continue;
            }
            
            std::cout << "Block " << block_idx << " (size " << block_size << ")..." << std::flush;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // OPTIMIZATION 1: Load ALL basis vectors for this block ONCE (batch I/O)
            std::cout << " [loading basis]" << std::flush;
            std::vector<std::vector<Complex>> basis_vectors(block_size);
            for (uint64_t i = 0; i < block_size; ++i) {
                basis_vectors[i] = HDF5SymmetryIO::loadBasisVector(
                    hdf5_file, block_start + i, fixed_sz_dim_);
            }
            
            // OPTIMIZATION 2: Parallel computation over columns
            std::cout << " [computing]" << std::flush;
            std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(block_size);
            
            #pragma omp parallel for schedule(dynamic, 1) if(block_size > 4)
            for (uint64_t col = 0; col < block_size; ++col) {
                // Apply Hamiltonian: H|ψ_j⟩
                std::vector<Complex> H_psi_j = apply(basis_vectors[col]);
                
                // Compute matrix elements with all rows (use Hermitian symmetry)
                for (uint64_t row = 0; row <= col; ++row) {  // Only compute lower triangle + diagonal
                    const auto& basis_row = basis_vectors[row];
                    
                    // H_ij = ⟨ψ_i|H|ψ_j⟩
                    Complex matrix_element(0.0, 0.0);
                    for (uint64_t k = 0; k < fixed_sz_dim_; ++k) {
                        if (std::abs(basis_row[k]) > 1e-15 && std::abs(H_psi_j[k]) > 1e-15) {
                            matrix_element += std::conj(basis_row[k]) * H_psi_j[k];
                        }
                    }
                    
                    if (std::abs(matrix_element) > 1e-12) {
                        thread_triplets[col].emplace_back(row, col, matrix_element);
                        // Add conjugate transpose element (if not diagonal)
                        if (row != col) {
                            thread_triplets[col].emplace_back(col, row, std::conj(matrix_element));
                        }
                    }
                }
            }
            
            // OPTIMIZATION 3: Merge triplets efficiently
            std::vector<Eigen::Triplet<Complex>> triplets;
            size_t total_nnz = 0;
            for (const auto& t : thread_triplets) total_nnz += t.size();
            triplets.reserve(total_nnz);
            
            for (auto& t : thread_triplets) {
                triplets.insert(triplets.end(), 
                               std::make_move_iterator(t.begin()), 
                               std::make_move_iterator(t.end()));
            }
            
            // Create and save sparse matrix
            Eigen::SparseMatrix<Complex> block(block_size, block_size);
            block.setFromTriplets(triplets.begin(), triplets.end());
            block.makeCompressed();
            
            HDF5SymmetryIO::saveBlockMatrix(hdf5_file, block_idx, block);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            double fill_percent = 100.0 * triplets.size() / (block_size * block_size);
            std::cout << " done (" << triplets.size() << " nnz, "
                      << std::fixed << std::setprecision(2) << fill_percent << "% fill, "
                      << duration.count() << " ms)" << std::endl;
            
            block_start += block_size;
        }
        
        std::cout << "=== Block Construction Complete (HDF5) ===" << std::endl;
    }
    
    /**
     * Load a specific symmetrized block matrix from HDF5 (Fixed Sz)
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockFixedSzHDF5(const std::string& dir, size_t block_idx) {
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        return HDF5SymmetryIO::loadBlockMatrix(hdf5_file, block_idx);
    }
    
    /**
     * Load all symmetrized blocks from HDF5 (Fixed Sz)
     */
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocksFixedSzHDF5(const std::string& dir) {
        std::string hdf5_file = dir + "/symmetry_data_fixed_sz.h5";
        
        // Load block sizes if not already loaded
        if (symmetrized_block_ham_sizes.empty()) {
            auto dims = HDF5SymmetryIO::loadSectorDimensions(hdf5_file);
            symmetrized_block_ham_sizes.assign(dims.begin(), dims.end());
        }
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            if (symmetrized_block_ham_sizes[i] > 0) {
                blocks.push_back(HDF5SymmetryIO::loadBlockMatrix(hdf5_file, i));
            } else {
                // Empty block
                blocks.emplace_back(0, 0);
            }
        }
        
        return blocks;
    }
    
    // ========================================================================
    // Legacy text-based methods for Fixed Sz (kept for backward compatibility)
    // ========================================================================
    
    /**
     * Build and save block-diagonal Hamiltonian matrices (Fixed Sz, text files)
     * Each block corresponds to one symmetry sector within the fixed Sz subspace
     */
    void buildAndSaveSymmetrizedBlocksFixedSz(const std::string& dir) {
        std::cout << "\n=== Building Symmetrized Hamiltonian Blocks (Fixed Sz) ===" << std::endl;
        
        loadBlockSizesFixedSzIfNeeded(dir);
        
        std::string block_dir = dir + "/sym_blocks_fixed_sz";
        safe_system_call("mkdir -p " + block_dir);
        
        uint64_t block_start = 0;
        for (size_t block_idx = 0; block_idx < symmetrized_block_ham_sizes.size(); ++block_idx) {
            uint64_t block_size = symmetrized_block_ham_sizes[block_idx];
            
            if (block_size == 0) {
                std::cout << "Block " << block_idx << ": empty (skipped)" << std::endl;
                continue;
            }
            
            buildSingleBlockFixedSz(dir, block_dir, block_idx, block_start, block_size);
            block_start += block_size;
        }
        
        std::cout << "=== Block Construction Complete ===" << std::endl;
    }
    
    /**
     * Load a specific symmetrized block matrix from disk (Fixed Sz)
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockFixedSz(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open block file: " + filepath);
        }
        
        // Read dimensions as int to match what was written
        int rows_int, cols_int;
        size_t nnz;
        file.read(reinterpret_cast<char*>(&rows_int), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols_int), sizeof(int));
        file.read(reinterpret_cast<char*>(&nnz), sizeof(size_t));
        
        uint64_t rows = rows_int;
        uint64_t cols = cols_int;
        
        std::vector<Eigen::Triplet<Complex>> triplets;
        triplets.reserve(nnz);
        
        for (size_t i = 0; i < nnz; ++i) {
            int row, col;
            Complex value;
            file.read(reinterpret_cast<char*>(&row), sizeof(int));
            file.read(reinterpret_cast<char*>(&col), sizeof(int));
            file.read(reinterpret_cast<char*>(&value), sizeof(Complex));
            triplets.emplace_back(row, col, value);
        }
        
        Eigen::SparseMatrix<Complex> matrix(rows, cols);
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        
        return matrix;
    }
    
    /**
     * Load a block by its index (Fixed Sz)
     */
    Eigen::SparseMatrix<Complex> loadSymmetrizedBlockFixedSzByIndex(const std::string& dir, size_t block_idx) {
        loadBlockSizesFixedSzIfNeeded(dir);
        
        if (block_idx >= symmetrized_block_ham_sizes.size()) {
            throw std::runtime_error("Block index out of range");
        }
        
        if (symmetrized_block_ham_sizes[block_idx] == 0) {
            return Eigen::SparseMatrix<Complex>(0, 0);
        }
        
        std::string filepath = dir + "/sym_blocks_fixed_sz/block_" + std::to_string(block_idx) + ".dat";
        return loadSymmetrizedBlockFixedSz(filepath);
    }
    
    /**
     * Load all symmetrized blocks (Fixed Sz)
     */
    std::vector<Eigen::SparseMatrix<Complex>> loadAllSymmetrizedBlocksFixedSz(const std::string& dir) {
        loadBlockSizesFixedSzIfNeeded(dir);
        
        std::vector<Eigen::SparseMatrix<Complex>> blocks;
        blocks.reserve(symmetrized_block_ham_sizes.size());
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            if (symmetrized_block_ham_sizes[i] > 0) {
                blocks.push_back(loadSymmetrizedBlockFixedSzByIndex(dir, i));
            } else {
                // Empty block
                blocks.emplace_back(0, 0);
            }
        }
        
        return blocks;
    }
    
protected:
    /**
     * Get orbit representative for a state in fixed Sz sector
     */
    uint64_t getOrbitRepresentativeFixedSz(uint64_t state) const {
        uint64_t min_state = state;
        
        for (const auto& generator : symmetry_info.generators) {
            uint64_t permuted = applyPermutation(state, generator);
            if (permuted < min_state) {
                min_state = permuted;
            }
        }
        
        return min_state;
    }
    
    /**
     * Create symmetrized vector in fixed Sz basis
     */
    std::vector<Complex> createSymmetrizedVectorFixedSz(
        uint64_t seed_state,
        const std::vector<int>& quantum_numbers,
        const std::vector<Complex>& phase_factors) const {
        
        std::vector<Complex> sym_vec(fixed_sz_dim_, Complex(0.0, 0.0));
        
        // Apply symmetry operations and accumulate
        std::set<uint64_t> orbit;
        std::queue<uint64_t> to_process;
        to_process.push(seed_state);
        orbit.insert(seed_state);
        
        while (!to_process.empty()) {
            uint64_t current = to_process.front();
            to_process.pop();
            
            for (size_t g = 0; g < symmetry_info.generators.size(); ++g) {
                uint64_t permuted = applyPermutation(current, symmetry_info.generators[g]);
                
                // Check if still in fixed Sz sector
                if (popcount(permuted) != n_up_) continue;
                
                if (orbit.find(permuted) == orbit.end()) {
                    orbit.insert(permuted);
                    to_process.push(permuted);
                }
            }
        }
        
        // Build symmetrized vector with phase factors
        for (uint64_t state : orbit) {
            auto it = state_to_index_.find(state);
            if (it != state_to_index_.end()) {
                uint64_t idx = it->second;
                
                // Calculate phase based on how we got to this state
                Complex phase = calculatePhaseFixedSz(seed_state, state, quantum_numbers, phase_factors);
                sym_vec[idx] = phase;
            }
        }
        
        return sym_vec;
    }
    
    /**
     * Calculate phase factor for symmetrized state
     */
    Complex calculatePhaseFixedSz(
        uint64_t seed_state,
        uint64_t target_state,
        const std::vector<int>& quantum_numbers,
        const std::vector<Complex>& phase_factors) const {
        
        // Simple implementation: accumulate phases along transformation path
        // For more sophisticated implementation, track actual group element
        Complex phase(1.0, 0.0);
        
        uint64_t current = seed_state;
        for (size_t g = 0; g < symmetry_info.generators.size(); ++g) {
            uint64_t permuted = applyPermutation(current, symmetry_info.generators[g]);
            if (permuted == target_state) {
                phase *= phase_factors[g];
                break;
            }
        }
        
        return phase;
    }
    
    /**
     * Save symmetrized basis vector for fixed Sz
     */
    void saveSymBasisVectorFixedSz(const std::string& dir, size_t index, 
                                    const std::vector<Complex>& vec) const {
        std::string filename = dir + "/basis_" + std::to_string(index) + ".dat";
        std::ofstream file(filename, std::ios::binary);
        
        uint64_t dim = fixed_sz_dim_;
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        file.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(Complex));
    }
    
    /**
     * Save block sizes for fixed Sz sectors
     */
    void saveBlockSizesFixedSz(const std::string& dir) const {
        std::string filename = dir + "/sym_basis_fixed_sz/block_sizes.txt";
        std::ofstream file(filename);
        
        file << "# Fixed Sz symmetrized block sizes\n";
        file << "# n_up = " << n_up_ << "\n";
        file << "# fixed_sz_dim = " << fixed_sz_dim_ << "\n";
        file << "# num_sectors = " << symmetrized_block_ham_sizes.size() << "\n\n";
        
        for (size_t i = 0; i < symmetrized_block_ham_sizes.size(); ++i) {
            file << i << " " << symmetrized_block_ham_sizes[i] << "\n";
        }
    }
    
private:
    /**
     * Load block sizes if needed (Fixed Sz)
     */
    void loadBlockSizesFixedSzIfNeeded(const std::string& dir) {
        if (!symmetrized_block_ham_sizes.empty()) return;
        
        std::string filepath = dir + "/sym_basis_fixed_sz/block_sizes.txt";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open block sizes file: " + filepath);
        }
        
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Parse "index size" format
            std::istringstream iss(line);
            uint64_t idx, size;
            if (iss >> idx >> size) {
                symmetrized_block_ham_sizes.push_back(size);
            }
        }
    }
    
    /**
     * Build a single block for fixed Sz sector
     */
    void buildSingleBlockFixedSz(const std::string& dir, const std::string& block_dir,
                                 size_t block_idx, uint64_t block_start, uint64_t block_size) {
        
        std::cout << "Block " << block_idx << " (size " << block_size << ")..." << std::flush;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // OPTIMIZATION 1: Load all basis vectors once
        std::vector<std::vector<Complex>> basis_vectors(block_size);
        for (uint64_t i = 0; i < block_size; ++i) {
            basis_vectors[i] = readSymBasisVectorFixedSz(
                dir + "/sym_basis_fixed_sz", block_start + i);
        }
        
        // OPTIMIZATION 2: Parallel computation with Hermitian symmetry
        std::vector<std::vector<Eigen::Triplet<Complex>>> thread_triplets(block_size);
        
        #pragma omp parallel for schedule(dynamic, 1) if(block_size > 4)
        for (uint64_t col = 0; col < block_size; ++col) {
            // Apply Hamiltonian: H|ψ_j⟩
            std::vector<Complex> H_psi_j = apply(basis_vectors[col]);
            
            // Compute matrix elements (use Hermitian symmetry)
            for (uint64_t row = 0; row <= col; ++row) {  // Only lower triangle + diagonal
                const auto& psi_i = basis_vectors[row];
                
                // H_ij = ⟨ψ_i|H|ψ_j⟩
                Complex matrix_element(0.0, 0.0);
                for (uint64_t k = 0; k < fixed_sz_dim_; ++k) {
                    if (std::abs(psi_i[k]) > 1e-15 && std::abs(H_psi_j[k]) > 1e-15) {
                        matrix_element += std::conj(psi_i[k]) * H_psi_j[k];
                    }
                }
                
                if (std::abs(matrix_element) > 1e-12) {
                    thread_triplets[col].emplace_back(row, col, matrix_element);
                    if (row != col) {
                        thread_triplets[col].emplace_back(col, row, std::conj(matrix_element));
                    }
                }
            }
        }
        
        // Merge triplets
        std::vector<Eigen::Triplet<Complex>> triplets;
        size_t total_nnz = 0;
        for (const auto& t : thread_triplets) total_nnz += t.size();
        triplets.reserve(total_nnz);
        
        for (auto& t : thread_triplets) {
            triplets.insert(triplets.end(), 
                           std::make_move_iterator(t.begin()), 
                           std::make_move_iterator(t.end()));
        }
        
        // Create and save sparse matrix
        Eigen::SparseMatrix<Complex> block(block_size, block_size);
        block.setFromTriplets(triplets.begin(), triplets.end());
        block.makeCompressed();
        
        std::string filename = block_dir + "/block_" + std::to_string(block_idx) + ".dat";
        std::ofstream file(filename, std::ios::binary);
        
        int rows = block_size, cols = block_size;
        size_t nnz = triplets.size();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(&nnz), sizeof(size_t));
        
        for (const auto& t : triplets) {
            int row = t.row(), col = t.col();
            Complex value = t.value();
            file.write(reinterpret_cast<const char*>(&row), sizeof(int));
            file.write(reinterpret_cast<const char*>(&col), sizeof(int));
            file.write(reinterpret_cast<const char*>(&value), sizeof(Complex));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << " done (" << nnz << " nnz, "
                  << std::fixed << std::setprecision(2)
                  << (100.0 * nnz / (block_size * block_size)) << "% fill, "
                  << duration.count() << " ms)" << std::endl;
    }
    
    /**
     * Read symmetrized basis vector for fixed Sz
     */
    std::vector<Complex> readSymBasisVectorFixedSz(const std::string& dir, size_t index) const {
        std::string filename = dir + "/basis_" + std::to_string(index) + ".dat";
        std::ifstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open basis vector file: " + filename);
        }
        
        int dim_int;
        file.read(reinterpret_cast<char*>(&dim_int), sizeof(int));
        uint64_t dim = dim_int;
        
        std::vector<Complex> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(Complex));
        
        return vec;
    }
};

// ============================================================================
// Derived Operator Classes
// ============================================================================

/**
 * Single site operator (S+, S-, Sz, Sx, Sy)
 */
class SingleSiteOperator : public Operator {
public:
    SingleSiteOperator(uint64_t num_site, float spin_l, uint64_t op, uint64_t site_j) 
        : Operator(num_site, spin_l) {
        
        if (op < 0 || op > 4) {
            throw std::invalid_argument("Invalid operator type");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site index");
        }
        
        if (op <= 2) {
            // S+, S-, Sz
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, Complex(spin_l * pow(-1, (basis >> site_j) & 1), 0.0)};
                } else {
                    if (((basis >> site_j) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site_j);
                        return {flipped, Complex(1.0, 0.0)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        } else {
            // Sx or Sy
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1 << site_j);
                if (op == 3) {
                    // Sx = (S+ + S-) / 2
                    return {flipped, Complex(0.5, 0.0)};
                } else {
                    // Sy = (S+ - S-) / (2i)
                    bool is_up = ((basis >> site_j) & 1) == 0;
                    return {flipped, Complex(0.0, is_up ? 0.5 : -0.5)};
                }
            });
        }
    }
};

/**
 * Two-site operator for interactions
 */
class DoubleSiteOperator : public Operator {
public:
    DoubleSiteOperator() : Operator(0, 0.0) {}
    
    DoubleSiteOperator(uint64_t num_site, float spin_l, uint64_t op_i, uint64_t site_i, uint64_t op_j, uint64_t site_j)
        : Operator(num_site, spin_l) {
        
        if (op_i < 0 || op_i > 2 || op_j < 0 || op_j > 2) {
            throw std::invalid_argument("Invalid operator types");
        }
        
        if (site_i < 0 || site_i >= num_site || site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site indices");
        }
        
        addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
            uint64_t bit_i = (basis >> site_i) & 1;
            uint64_t bit_j = (basis >> site_j) & 1;
            
            Complex factor(1.0, 0.0);
            
            if (op_i == 2 && op_j == 2) {
                return {basis, Complex(spin_l * spin_l * pow(-1, bit_i) * pow(-1, bit_j), 0.0)};
            }
            
            uint64_t new_basis = basis;
            bool valid = true;
            
            if (op_i != 2) {
                if (bit_i != op_i) {
                    new_basis ^= (1 << site_i);
                } else {
                    valid = false;
                }
            } else {
                factor *= Complex(spin_l * pow(-1, bit_i), 0.0);
            }
            
            if (valid && op_j != 2) {
                uint64_t new_bit_j = (new_basis >> site_j) & 1;
                if (new_bit_j != op_j) {
                    new_basis ^= (1 << site_j);
                } else {
                    valid = false;
                }
            } else if (valid) {
                uint64_t new_bit_j = (new_basis >> site_j) & 1;
                factor *= Complex(spin_l * pow(-1, new_bit_j), 0.0);
            }
            
            if (valid) return {new_basis, factor};
            return {basis, Complex(0.0, 0.0)};
        });
    }
};

/**
 * Base class for position-dependent operators
 */
class BasePositionOperator : public Operator {
protected:
    std::vector<std::vector<double>> readPositionsFromFile(const std::string& filename, uint64_t expected_sites) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open positions file: " + filename);
        }
        
        std::vector<std::vector<double>> positions(expected_sites);
        std::string line;
        
        while (std::getline(file, line) && line[0] == '#');
        
        bool process_current = !line.empty() && line[0] != '#';
        
        do {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            uint64_t site_id, matrix_idx, sublattice;
            double x, y, z;
            
            if (iss >> site_id >> matrix_idx >> sublattice >> x >> y >> z) {
                if (site_id >= 0 && site_id < expected_sites) {
                    positions[site_id] = {x, y, z};
                }
            }
        } while (std::getline(file, line));
        
        return positions;
    }
    
    std::vector<Complex> calculatePhaseFactors(const std::vector<double>& Q_vector,
                                                const std::vector<std::vector<double>>& positions,
                                                double normalization) {
        std::vector<Complex> phases(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            double dot_product = 0.0;
            for (size_t d = 0; d < 3; ++d) {
                dot_product += Q_vector[d] * positions[i][d];
            }
            phases[i] = normalization * std::exp(Complex(0.0, dot_product));
        }
        return phases;
    }
    
public:
    BasePositionOperator(uint64_t num_site, float spin_l) : Operator(num_site, spin_l) {}
};

/**
 * Sum operator: S^α = Σᵢ S^α_i e^(iQ·Rᵢ) / √N
 */
class SumOperator : public BasePositionOperator {
public:
    SumOperator(uint64_t num_site, float spin_l, uint64_t op, const std::vector<double>& Q_vector,
                const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                } else {
                    if (((basis >> site) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site);
                        return {flipped, phase};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

/**
 * Sum operator with Cartesian basis (Sx, Sy, Sz)
 */
class SumOperatorXYZ : public BasePositionOperator {
public:
    SumOperatorXYZ(uint64_t num_site, float spin_l, uint64_t op, const std::vector<double>& Q_vector,
                   const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx = (S+ + S-) / 2
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    return {flipped, phase * Complex(0.5, 0.0)};
                });
            } else if (op == 1) {
                // Sy = (S+ - S-) / (2i)
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    bool is_up = ((basis >> site) & 1) == 0;
                    return {flipped, phase * Complex(0.0, is_up ? 0.5 : -0.5)};
                });
            } else if (op == 2) {
                // Sz
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                });
            }
        }
    }
};

/**
 * Sublattice operator: sum over specific sublattice sites
 */
class SublatticeOperator : public BasePositionOperator {
public:
    SublatticeOperator(uint64_t sublattice_idx, uint64_t unit_cell_size, uint64_t num_site, float spin_l,
                      uint64_t op, const std::vector<double>& Q_vector,
                      const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = sublattice_idx; site < num_site; site += unit_cell_size) {
            Complex phase = phases[site];
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                } else {
                    if (((basis >> site) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site);
                        return {flipped, phase};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

/**
 * Transverse operator with sublattice-dependent weighting
 */
class TransverseOperator : public BasePositionOperator {
public:
    TransverseOperator(uint64_t num_site, float spin_l, uint64_t op,
                      const std::vector<double>& Q_vector,
                      const std::vector<double>& v,
                      const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors with sublattice weighting
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                } else {
                    if (((basis >> site) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site);
                        return {flipped, phase};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

/**
 * Transverse operator with Cartesian basis
 */
class TransverseOperatorXYZ : public BasePositionOperator {
public:
    TransverseOperatorXYZ(uint64_t num_site, float spin_l, uint64_t op,
                         const std::vector<double>& Q_vector,
                         const std::vector<double>& v,
                         const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors with sublattice weighting
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    return {flipped, phase * Complex(0.5, 0.0)};
                });
            } else if (op == 1) {
                // Sy
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    bool is_up = ((basis >> site) & 1) == 0;
                    return {flipped, phase * Complex(0.0, is_up ? 0.5 : -0.5)};
                });
            } else if (op == 2) {
                // Sz
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                });
            }
        }
    }
};

/**
 * Experimental operator: cos(θ)Sz + sin(θ)Sx
 */
class ExperimentalOperator : public BasePositionOperator {
public:
    ExperimentalOperator(uint64_t num_site, float spin_l, double theta,
                        const std::vector<double>& Q_vector,
                        const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Sz contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                return {basis, phase * Complex(cos_theta * spin_l * pow(-1, (basis >> site) & 1), 0.0)};
            });
            
            // Sx contribution = (S+ + S-) / 2
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1 << site);
                return {flipped, phase * Complex(sin_theta * 0.5, 0.0)};
            });
        }
    }
};

/**
 * Transverse experimental operator with sublattice weighting
 */
class TransverseExperimentalOperator : public BasePositionOperator {
public:
    TransverseExperimentalOperator(uint64_t num_site, float spin_l, double theta,
                                  const std::vector<double>& Q_vector,
                                  const std::vector<double>& v,
                                  const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Sz contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                return {basis, phase * Complex(cos_theta * spin_l * pow(-1, (basis >> site) & 1), 0.0)};
            });
            
            // Sx contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1 << site);
                return {flipped, phase * Complex(sin_theta * 0.5, 0.0)};
            });
        }
    }
};

// ============================================================================
// Fixed Sz Derived Operator Classes
// ============================================================================

/**
 * Single site operator for fixed Sz sector (S+, S-, Sz, Sx, Sy)
 */
class FixedSzSingleSiteOperator : public FixedSzOperator {
public:
    FixedSzSingleSiteOperator(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op, uint64_t site_j) 
        : FixedSzOperator(num_site, spin_l, n_up) {
        
        if (op < 0 || op > 4) {
            throw std::invalid_argument("Invalid operator type");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site index");
        }
        
        if (op <= 2) {
            // S+, S-, Sz
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, Complex(spin_l * pow(-1, (basis >> site_j) & 1), 0.0)};
                } else {
                    if (((basis >> site_j) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site_j);
                        return {flipped, Complex(1.0, 0.0)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        } else {
            // Sx or Sy
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1 << site_j);
                if (op == 3) {
                    // Sx = (S+ + S-) / 2
                    return {flipped, Complex(0.5, 0.0)};
                } else {
                    // Sy = (S+ - S-) / (2i)
                    bool is_up = ((basis >> site_j) & 1) == 0;
                    return {flipped, Complex(0.0, is_up ? 0.5 : -0.5)};
                }
            });
        }
    }
};

/**
 * Two-site operator for fixed Sz sector
 */
class FixedSzDoubleSiteOperator : public FixedSzOperator {
public:
    FixedSzDoubleSiteOperator(uint64_t num_site, float spin_l, int64_t n_up, 
                              uint64_t op_i, uint64_t site_i, uint64_t op_j, uint64_t site_j)
        : FixedSzOperator(num_site, spin_l, n_up) {
        
        if (op_i < 0 || op_i > 2 || op_j < 0 || op_j > 2) {
            throw std::invalid_argument("Invalid operator types");
        }
        
        if (site_i < 0 || site_i >= num_site || site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site indices");
        }
        
        addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
            uint64_t bit_i = (basis >> site_i) & 1;
            uint64_t bit_j = (basis >> site_j) & 1;
            
            Complex factor(1.0, 0.0);
            
            if (op_i == 2 && op_j == 2) {
                return {basis, Complex(spin_l * spin_l * pow(-1, bit_i) * pow(-1, bit_j), 0.0)};
            }
            
            uint64_t new_basis = basis;
            bool valid = true;
            
            if (op_i != 2) {
                if (bit_i != op_i) {
                    new_basis ^= (1 << site_i);
                } else {
                    valid = false;
                }
            } else {
                factor *= Complex(spin_l * pow(-1, bit_i), 0.0);
            }
            
            if (valid && op_j != 2) {
                uint64_t new_bit_j = (new_basis >> site_j) & 1;
                if (new_bit_j != op_j) {
                    new_basis ^= (1 << site_j);
                } else {
                    valid = false;
                }
            } else if (valid) {
                uint64_t new_bit_j = (new_basis >> site_j) & 1;
                factor *= Complex(spin_l * pow(-1, new_bit_j), 0.0);
            }
            
            if (valid) return {new_basis, factor};
            return {basis, Complex(0.0, 0.0)};
        });
    }
};

/**
 * Base class for position-dependent operators in fixed Sz sector
 */
class FixedSzBasePositionOperator : public FixedSzOperator {
protected:
    std::vector<std::vector<double>> readPositionsFromFile(const std::string& filename, uint64_t expected_sites) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open positions file: " + filename);
        }
        
        std::vector<std::vector<double>> positions(expected_sites);
        std::string line;
        
        while (std::getline(file, line) && line[0] == '#');
        
        bool process_current = !line.empty() && line[0] != '#';
        
        do {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            uint64_t site_id, matrix_idx, sublattice;
            double x, y, z;
            
            if (iss >> site_id >> matrix_idx >> sublattice >> x >> y >> z) {
                if (site_id >= 0 && site_id < expected_sites) {
                    positions[site_id] = {x, y, z};
                }
            }
        } while (std::getline(file, line));
        
        return positions;
    }
    
    std::vector<Complex> calculatePhaseFactors(const std::vector<double>& Q_vector,
                                                const std::vector<std::vector<double>>& positions,
                                                double normalization) {
        std::vector<Complex> phases(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            double dot_product = 0.0;
            for (size_t d = 0; d < 3; ++d) {
                dot_product += Q_vector[d] * positions[i][d];
            }
            phases[i] = normalization * std::exp(Complex(0.0, dot_product));
        }
        return phases;
    }
    
public:
    FixedSzBasePositionOperator(uint64_t num_site, float spin_l, int64_t n_up) 
        : FixedSzOperator(num_site, spin_l, n_up) {}
};

/**
 * Sum operator for fixed Sz sector: S^α = Σᵢ S^α_i e^(iQ·Rᵢ) / √N
 */
class FixedSzSumOperator : public FixedSzBasePositionOperator {
public:
    FixedSzSumOperator(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op, 
                       const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                } else {
                    if (((basis >> site) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site);
                        return {flipped, phase};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

/**
 * Sum operator with Cartesian basis for fixed Sz sector (Sx, Sy, Sz)
 */
class FixedSzSumOperatorXYZ : public FixedSzBasePositionOperator {
public:
    FixedSzSumOperatorXYZ(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op,
                          const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx = (S+ + S-) / 2
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    return {flipped, phase * Complex(0.5, 0.0)};
                });
            } else if (op == 1) {
                // Sy = (S+ - S-) / (2i)
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    bool is_up = ((basis >> site) & 1) == 0;
                    return {flipped, phase * Complex(0.0, is_up ? 0.5 : -0.5)};
                });
            } else if (op == 2) {
                // Sz
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                });
            }
        }
    }
};

/**
 * Sublattice operator for fixed Sz sector: sum over specific sublattice sites
 */
class FixedSzSublatticeOperator : public FixedSzBasePositionOperator {
public:
    FixedSzSublatticeOperator(uint64_t sublattice_idx, uint64_t unit_cell_size, uint64_t num_site, 
                              float spin_l, int64_t n_up, uint64_t op, 
                              const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = sublattice_idx; site < num_site; site += unit_cell_size) {
            Complex phase = phases[site];
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                } else {
                    if (((basis >> site) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site);
                        return {flipped, phase};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

/**
 * Transverse operator for fixed Sz sector with sublattice-dependent weighting
 */
class FixedSzTransverseOperator : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseOperator(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op,
                              const std::vector<double>& Q_vector, const std::vector<double>& v,
                              const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors with sublattice weighting
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                } else {
                    if (((basis >> site) & 1) != op) {
                        uint64_t flipped = basis ^ (1 << site);
                        return {flipped, phase};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        }
    }
};

/**
 * Transverse operator with Cartesian basis for fixed Sz sector
 */
class FixedSzTransverseOperatorXYZ : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseOperatorXYZ(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op,
                                 const std::vector<double>& Q_vector, const std::vector<double>& v,
                                 const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors with sublattice weighting
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    return {flipped, phase * Complex(0.5, 0.0)};
                });
            } else if (op == 1) {
                // Sy
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    uint64_t flipped = basis ^ (1 << site);
                    bool is_up = ((basis >> site) & 1) == 0;
                    return {flipped, phase * Complex(0.0, is_up ? 0.5 : -0.5)};
                });
            } else if (op == 2) {
                // Sz
                addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                    return {basis, phase * Complex(spin_l * pow(-1, (basis >> site) & 1), 0.0)};
                });
            }
        }
    }
};

/**
 * Experimental operator for fixed Sz sector: cos(θ)Sz + sin(θ)Sx
 */
class FixedSzExperimentalOperator : public FixedSzBasePositionOperator {
public:
    FixedSzExperimentalOperator(uint64_t num_site, float spin_l, int64_t n_up, double theta,
                                const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Sz contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                return {basis, phase * Complex(cos_theta * spin_l * pow(-1, (basis >> site) & 1), 0.0)};
            });
            
            // Sx contribution = (S+ + S-) / 2
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1 << site);
                return {flipped, phase * Complex(sin_theta * 0.5, 0.0)};
            });
        }
    }
};

/**
 * Transverse experimental operator for fixed Sz sector with sublattice weighting
 */
class FixedSzTransverseExperimentalOperator : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseExperimentalOperator(uint64_t num_site, float spin_l, int64_t n_up, double theta,
                                          const std::vector<double>& Q_vector, const std::vector<double>& v,
                                          const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Sz contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                return {basis, phase * Complex(cos_theta * spin_l * pow(-1, (basis >> site) & 1), 0.0)};
            });
            
            // Sx contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1 << site);
                return {flipped, phase * Complex(sin_theta * 0.5, 0.0)};
            });
        }
    }
};

#endif // CONSTRUCT_HAM_H

