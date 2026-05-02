#pragma once
// =============================================================================
// include/ed/core/symmetry_metadata.h
//
// Symmetry sector metadata and SymmetryGroupInfo:
//   - applyPermutation       free function
//   - SectorMetadata         struct
//   - SymmetryGroupInfo      class (loads generators, sectors, max-clique)
//
// Depends on: <vector>, <complex>, <map>, <set>, <queue>, nlohmann/json.
// Consumed by: operator.h (Operator has a SymmetryGroupInfo member).
// =============================================================================

#include <vector>
#include <complex>
#include <functional>
#include <string>
#include <map>
#include <set>
#include <queue>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>

using Complex = std::complex<double>;

// ============================================================================
// Symmetry Data Structures
// ============================================================================

/**
 * Structure to hold symmetry sector metadata
 */
struct SectorMetadata {
    uint64_t sector_id = 0;
    std::vector<int> quantum_numbers;
    std::vector<Complex> phase_factors;
    uint64_t dimension = 0;  // Computed during basis generation
};

/**
 * Structure to hold complete symmetry group information
 * Loads data from JSON files produced by automorphism_finder.py
 */
struct SymmetryGroupInfo {
    // Default-initialise the primitive scalar so freshly-constructed
    // Operators (which store SymmetryGroupInfo by value) report
    // ``num_generators == 0`` instead of whatever bit-pattern happened
    // to be in the heap slot. The CI lane on Ubuntu 22.04 / Python 3.11
    // tripped this with a ``4607182418800017408`` (= 1.0 as bits)
    // value reinterpreted from a previous allocation; on glibc / libstdc++
    // this is not guaranteed to be zero unless we explicitly say so.
    uint64_t num_generators = 0;
    std::vector<int> generator_orders;
    std::vector<std::vector<int>> generators;
    std::vector<std::vector<int>> max_clique;
    std::vector<std::vector<int>> power_representation;
    std::vector<SectorMetadata> sectors;

    // Phase F (May 2026): optional Sz quantum-number filter.
    //
    // Site permutations preserve the popcount of every basis state, so
    // restricting the symmetry-projected basis to "orbits whose
    // representative has popcount == n_up" yields a closed sub-block
    // that combines the U(1)-Sz quantum number with the lattice
    // symmetry sector. ``DistributedSymmetryOperator`` (and the CLI
    // ``--sz`` flag in ``ed_distributed_main``) honour this field;
    // the in-process streaming kernel currently ignores it (it has
    // its own FixedSzOperator path).
    //
    // ``-1`` (the default) means "no Sz constraint -- use the full
    // symmetry-projected basis." Values in ``[0, num_sites]`` select
    // a single Sz sector with magnetisation ``2*n_up - num_sites``.
    int n_up = -1;
    
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
        filterInvalidSectors();
        
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
        // P0.15: was a hand-rolled .find('[') / .substr() parser. Now uses
        // nlohmann/json. The on-disk schema is a flat array of integer arrays:
        //   [[0,1,...,N-1], [N-1,0,...,N-2], ...]
        const std::string filepath = auto_dir + "/max_clique.json";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open max_clique.json: " + filepath);
        }

        nlohmann::json j;
        try {
            file >> j;
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse " + filepath + ": " + e.what());
        }
        if (!j.is_array()) {
            throw std::runtime_error(filepath + ": expected a top-level JSON array");
        }
        max_clique.clear();
        max_clique.reserve(j.size());
        for (const auto& perm : j) {
            max_clique.push_back(perm.get<std::vector<int>>());
        }

        std::cout << "Loaded max clique with " << max_clique.size() << " automorphisms" << std::endl;
    }
    
    void loadMinimalGenerators(const std::string& auto_dir) {
        // P0.15: was a manual scan looking for "\"permutation\":" / "\"order\":"
        // substrings; that approach broke the moment the JSON producer changed
        // whitespace or key ordering. The on-disk schema is:
        //   {"generators":[{"permutation":[...], "order":N}, ...]}
        const std::string filepath = auto_dir + "/minimal_generators.json";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open minimal_generators.json: " + filepath);
        }

        nlohmann::json j;
        try {
            file >> j;
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse " + filepath + ": " + e.what());
        }

        generators.clear();
        generator_orders.clear();

        if (!j.contains("generators") || !j.at("generators").is_array()) {
            throw std::runtime_error(filepath + ": missing or non-array 'generators' field");
        }
        const auto& gens = j.at("generators");
        generators.reserve(gens.size());
        generator_orders.reserve(gens.size());
        for (const auto& g : gens) {
            generators.push_back(g.at("permutation").get<std::vector<int>>());
            generator_orders.push_back(g.at("order").get<int>());
        }

        num_generators = generators.size();
        std::cout << "Loaded " << num_generators << " minimal generators" << std::endl;
    }
    
    void loadSectorMetadata(const std::string& auto_dir) {
        // P0.15: was a 130-line manual scan with bracket-depth tracking. The
        // on-disk schema is:
        //   {"sectors":[
        //       {"sector_id": k,
        //        "quantum_numbers": [q_0, q_1, ...],
        //        "phase_factors":  [{"real": x, "imag": y}, ...]},
        //       ...
        //   ]}
        const std::string filepath = auto_dir + "/sector_metadata.json";
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open sector_metadata.json: " + filepath);
        }

        nlohmann::json j;
        try {
            file >> j;
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse " + filepath + ": " + e.what());
        }

        sectors.clear();

        if (!j.contains("sectors") || !j.at("sectors").is_array()) {
            throw std::runtime_error("Could not find 'sectors' array in " + filepath);
        }
        const auto& jsec = j.at("sectors");
        sectors.reserve(jsec.size());

        for (const auto& s : jsec) {
            SectorMetadata sector;
            sector.sector_id        = s.value("sector_id", 0);
            sector.dimension        = 0;  // Filled in during basis generation.
            sector.quantum_numbers  = s.value("quantum_numbers", std::vector<int>{});

            if (s.contains("phase_factors")) {
                const auto& pfs = s.at("phase_factors");
                sector.phase_factors.reserve(pfs.size());
                for (const auto& pf : pfs) {
                    sector.phase_factors.emplace_back(
                        pf.value("real", 0.0), pf.value("imag", 0.0));
                }
            }

            sectors.push_back(std::move(sector));
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
    
    /**
     * Filter out invalid symmetry sectors (phantom irreps).
     * 
     * When generators have non-trivial algebraic relations (i.e., the naive
     * product of orders exceeds |G|), some quantum number combinations produce
     * characters that are not group homomorphisms. These "phantom" sectors
     * must be removed.
     * 
     * The valid quantum numbers q are those satisfying: for every relation
     * r = (r_0,...,r_{k-1}) in the relation subgroup K (where 
     * g_0^{r_0} * ... * g_{k-1}^{r_{k-1}} = identity), we have
     * sum_k q_k * r_k / o_k is an integer.
     */
    void filterInvalidSectors() {
        // Quick check: if #sectors <= |G|, likely already correct
        if (sectors.size() <= max_clique.size()) return;
        
        uint64_t num_gen = generators.size();
        if (num_gen == 0) return;
        
        // Compute product of orders
        uint64_t product_orders = 1;
        for (int o : generator_orders) product_orders *= o;
        
        // Safety limit for enumeration
        if (product_orders > 100000) {
            std::cerr << "WARNING: Product of generator orders (" << product_orders 
                      << ") too large for relation detection. Skipping sector filter." << std::endl;
            return;
        }
        
        uint64_t n = generators[0].size();
        std::vector<int> identity(n);
        for (uint64_t i = 0; i < n; ++i) identity[i] = i;
        
        auto compose = [n](const std::vector<int>& a, const std::vector<int>& b) {
            std::vector<int> result(n);
            for (uint64_t i = 0; i < n; ++i) result[i] = a[b[i]];
            return result;
        };
        
        auto powerPerm = [&](const std::vector<int>& perm, int p) {
            std::vector<int> result = identity;
            for (int i = 0; i < p; ++i) result = compose(perm, result);
            return result;
        };
        
        // Find relations: tuples (r_0,...,r_{k-1}) where g_0^{r_0}*...*g_{k-1}^{r_{k-1}} = identity
        std::vector<std::vector<int>> relations;
        std::vector<int> r(num_gen, 0);
        
        while (true) {
            // Skip the all-zeros tuple
            bool all_zero = true;
            for (uint64_t i = 0; i < num_gen; ++i) {
                if (r[i] != 0) { all_zero = false; break; }
            }
            
            if (!all_zero) {
                std::vector<int> result = identity;
                for (uint64_t k = 0; k < num_gen; ++k) {
                    result = compose(result, powerPerm(generators[k], r[k]));
                }
                if (result == identity) {
                    relations.push_back(r);
                }
            }
            
            // Increment r (odometer)
            int carry = 1;
            for (int k = (int)num_gen - 1; k >= 0 && carry; --k) {
                r[k] += carry;
                if (r[k] >= generator_orders[k]) {
                    r[k] = 0;
                    carry = 1;
                } else {
                    carry = 0;
                }
            }
            if (carry) break;  // overflow = done
        }
        
        if (relations.empty()) return;
        
        std::cout << "Found " << relations.size() << " generator relation(s) "
                  << "(|K| = " << (relations.size() + 1) << ")" << std::endl;
        
        // Filter sectors: keep those where sum_k q_k*r_k/o_k is integer for all relations
        std::vector<SectorMetadata> valid_sectors;
        for (const auto& sector : sectors) {
            bool valid = true;
            for (const auto& rel : relations) {
                double s = 0.0;
                for (uint64_t k = 0; k < num_gen; ++k) {
                    s += (double)sector.quantum_numbers[k] * rel[k] / generator_orders[k];
                }
                if (std::abs(s - std::round(s)) > 1e-10) {
                    valid = false;
                    break;
                }
            }
            if (valid) valid_sectors.push_back(sector);
        }
        
        if (valid_sectors.size() < sectors.size()) {
            std::cout << "Filtered " << (sectors.size() - valid_sectors.size())
                      << " invalid sectors (phantom irreps from generator relations)" << std::endl;
            std::cout << "Valid sectors: " << valid_sectors.size() 
                      << " (group size: " << max_clique.size() << ")" << std::endl;
            
            // Re-number sector IDs
            for (uint64_t i = 0; i < valid_sectors.size(); ++i) {
                valid_sectors[i].sector_id = i;
            }
            sectors = valid_sectors;
        }
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
    
    // P0.15: parseJsonIntArrays() helper was removed -- nlohmann/json now does
    // the parsing in loadMaxClique() / loadMinimalGenerators() / loadSectorMetadata().
};

// ============================================================================
// Operator Class
// ============================================================================

/**
 * Operator class that represents quantum operators through bit flip operations
 * Supports symmetry-adapted basis and block diagonalization
 * 
 * OPTIMIZED: Structure-of-Arrays for transform data to enable vectorization
 * and eliminate std::function overhead (500-1000x speedup for large N)
 * 
 * ADDITIONAL OPTIMIZATIONS (v2):
 * - Branch-free separated storage: diagonal/off-diagonal transforms in separate arrays
 * - Radix sort for O(n) flush buffer sorting
 * - Cache blocking for improved memory access patterns
 * - Binary search for state lookup in FixedSzOperator
 */
