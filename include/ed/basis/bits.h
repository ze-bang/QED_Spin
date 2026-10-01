#pragma once
// =============================================================================
// include/ed/basis/bits.h
//
// Bit-basis utility functions: popcount, applyPermutation.
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

