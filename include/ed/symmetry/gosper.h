#pragma once
// =============================================================================
// include/ed/symmetry/gosper.h
//
// Gosper's hack: the next bit-permutation with the same popcount, in ascending
// numeric order. This streaming generator avoids materializing a fixed-Sz
// basis vector -- it walks every length-`n_bits` state of popcount `k` in
// O(1) memory / O(1) time per step, in ascending numeric order (the same
// order as a sorted fixed-Sz basis scan).
//
// Combined with `ed::core::combinadic::unrank_to_state` (the colex rank of a
// fixed-popcount state equals its position in ascending numeric order), this
// lets an OpenMP region partition the combinadic rank range [0, C(n,k)) across
// threads: thread t unranks its start state, then iterates with
// `next_bit_permutation` for its share -- parallel, materialization-free.
// =============================================================================

#include <cstdint>

namespace ed::symmetry {

/// Next integer with the same popcount as `v`, strictly greater than `v`
/// (Gosper's hack). Precondition: `v != 0`.
[[nodiscard]] inline std::uint64_t next_bit_permutation(std::uint64_t v) noexcept {
    const std::uint64_t c = v & (~v + 1ULL);   // lowest set bit (== v & -v)
    const std::uint64_t r = v + c;
    return (((r ^ v) >> 2) / c) | r;
}

}  // namespace ed::symmetry
