#pragma once
// =============================================================================
// include/ed/core/basis_utils.h
//
// Bit-basis utility functions:
//   - popcount / generateFixedSzBasis / buildBasisIndexMap
//   - LinIndexTable: O(1) state->index lookup for fixed-Sz bases (Lin 1990)
//
// This header has no dependencies beyond the C++ standard library.
// It is included by operator.h (which pulls it transitively into all
// Operator consumers via the construct_ham.h umbrella).
// =============================================================================

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <numeric>

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
    if (n_bits >= 64) {
        throw std::runtime_error("generateFixedSzBasis: n_bits = " + std::to_string(n_bits)
            + " >= 64 is not supported");
    }
    if (n_up < 0 || static_cast<uint64_t>(n_up) > n_bits) return basis;
    
    // Special case: n_up=0 has exactly one basis state (all spins down)
    // Gosper's hack divides by (state & -state) which is 0 when state=0
    if (n_up == 0) {
        basis.push_back(0);
        return basis;
    }
    
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
 *
 * NOTE: prefer LinIndexTable below for any non-trivial basis. unordered_map
 * uses ~25 GB for N=32 fixed-Sz; LinIndexTable uses ~768 KB for the same
 * problem with strictly faster lookup and no cache misses inside the SpMV
 * inner loop.
 *
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
 * Lin (1990) two-table O(1) state-to-index lookup for a fixed-Sz basis.
 *
 * Split the bit-packed state s = (u << n_lower) | l. Store two arrays:
 *   J_l[u] : index in the sorted basis of the first state with upper bits == u
 *   J_r[l] : rank of l among all length-n_lower bitstrings of the same popcount
 *
 * Then index(s) = J_l[u] + J_r[l] for any s in the basis.
 *
 * Performance characteristics
 * ---------------------------
 * Memory:  8 * 2^(N/2) + 4 * 2^(N/2) bytes total (10000-100000x smaller than
 *          the equivalent unordered_map<uint64_t,int> for large N).
 * Lookup:  one popcount verification + two L1/L2-resident array indexings + one add.
 *          For N <= 32 both tables fit in L2 (768 KB total for N=32) so the
 *          inner SpMV loop never cache-misses on lookup. This typically gives
 *          a 5-10x speedup over std::lower_bound binary search and a 2-5x
 *          speedup over unordered_map for fixed-Sz SpMV.
 *
 * Reference: H. Q. Lin, "Exact diagonalization of quantum-spin models",
 *            Phys. Rev. B 42, 6561 (1990).
 */
class LinIndexTable {
public:
    LinIndexTable() = default;

    /**
     * Build the two tables from a lexicographically sorted fixed-Sz basis.
     * Cost: O(2^(N/2)) for table allocation + O(|basis|) for the J_l fill.
     */
    void build(uint64_t n_bits, int64_t n_up,
               const std::vector<uint64_t>& sorted_basis_states) {
        n_bits_ = n_bits;
        n_up_ = n_up;
        if (n_bits == 0) {
            // Trivial: 0 sites means dim = 1 (only state is 0).
            J_l_.clear();
            J_r_.clear();
            return;
        }
        // Half-and-half split. For n_bits=1 we collapse to n_lower=0
        // (J_r has a single 0 entry, all work is in J_l).
        n_lower_ = n_bits / 2;
        n_upper_ = n_bits - n_lower_;
        lower_mask_ = (n_lower_ == 0) ? 0ULL : ((1ULL << n_lower_) - 1ULL);

        const uint64_t upper_size = 1ULL << n_upper_;
        const uint64_t lower_size = 1ULL << n_lower_;

        J_l_.assign(upper_size, kInvalid);

        J_r_.assign(lower_size, 0);
        // J_r[l] = rank of l among length-n_lower bitstrings with same popcount
        std::vector<uint32_t> per_pop(n_lower_ + 1, 0);
        for (uint64_t l = 0; l < lower_size; ++l) {
            int p = __builtin_popcountll(l);
            J_r_[l] = per_pop[p]++;
        }

        // sorted_basis_states is lex-sorted by the integer value of the state,
        // which is the same as sorting by (upper, lower). So the first index
        // at which each upper value appears is exactly J_l[u].
        uint64_t prev_upper = ~uint64_t(0);
        for (uint64_t i = 0; i < sorted_basis_states.size(); ++i) {
            uint64_t u = sorted_basis_states[i] >> n_lower_;
            if (u != prev_upper) {
                J_l_[u] = i;
                prev_upper = u;
            }
        }
    }

    /**
     * Look up a state. Returns -1 if the state is not in the basis (either
     * the popcount disagrees with n_up, or the upper-half slot is empty).
     */
    inline int64_t lookup(uint64_t state) const {
        if (J_l_.empty()) {
            // Degenerate n_bits=0 case: only state 0 maps to index 0.
            return (state == 0 && n_up_ == 0) ? 0 : -1;
        }
        if (static_cast<int64_t>(__builtin_popcountll(state)) != n_up_) return -1;
        const uint64_t u = state >> n_lower_;
        const uint64_t base = J_l_[u];
        if (base == kInvalid) return -1;
        return static_cast<int64_t>(base + J_r_[state & lower_mask_]);
    }

    bool empty() const { return J_l_.empty() && J_r_.empty(); }
    size_t memoryBytes() const {
        return J_l_.size() * sizeof(uint64_t) + J_r_.size() * sizeof(uint32_t);
    }

private:
    static constexpr uint64_t kInvalid = ~uint64_t(0);
    uint64_t n_bits_ = 0;
    int64_t n_up_ = 0;
    uint64_t n_lower_ = 0;
    uint64_t n_upper_ = 0;
    uint64_t lower_mask_ = 0;
    std::vector<uint64_t> J_l_;   // size 2^n_upper, sentinel kInvalid for empty slots
    std::vector<uint32_t> J_r_;   // size 2^n_lower
};

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

