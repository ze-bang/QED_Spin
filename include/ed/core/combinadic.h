#pragma once
// =============================================================================
// include/ed/core/combinadic.h
//
// Host combinadic rank/unrank for fixed-Sz state <-> dense-index mapping.
//
// This is the CPU twin of ``include/ed/gpu/combinadic.cuh``: it implements
// the SAME colex-order ranking so a host-built ``rep_index_of_rank`` table is
// bit-identical to the device one. The on-the-fly representative SpMV
// (``ed::matvec::basis::RepSymmetryBasisPolicy``) uses binary search on the
// sorted representative array as its PRIMARY reverse lookup; this header
// provides the OPTIONAL O(1) rank-table fast path (and the round-trip is
// useful for tests + the HDF5 basis cache).
//
// The Pascal/binomial table here is a small ``std::vector`` built on demand
// (max 65x65 ``uint64_t`` ~= 33 KiB) rather than a CUDA ``__constant__``
// symbol; ``rank_state`` / ``unrank_to_state`` take the table by const ref so
// callers build it once and reuse across many lookups.
// =============================================================================

#include <cstdint>
#include <vector>

namespace ed::core::combinadic {

// Dense (n+1) x (n+1) Pascal triangle of binomial coefficients, row-major.
// ``at(n, k) == C(n, k)``; out-of-range (k<0 || k>n) reads return 0.
class BinomialTable {
public:
    BinomialTable() = default;

    explicit BinomialTable(int max_n) { resize(max_n); }

    void resize(int max_n) {
        if (max_n < 0) max_n = 0;
        max_n_ = max_n;
        const std::size_t stride = static_cast<std::size_t>(max_n) + 1;
        c_.assign(stride * stride, 0ULL);
        for (int n = 0; n <= max_n; ++n) {
            c_[idx_(n, 0)] = 1ULL;
            for (int k = 1; k <= n; ++k) {
                c_[idx_(n, k)] = c_[idx_(n - 1, k - 1)] + c_[idx_(n - 1, k)];
            }
        }
    }

    [[nodiscard]] std::uint64_t at(int n, int k) const noexcept {
        if (k < 0 || n < 0 || k > n || n > max_n_) return 0ULL;
        return c_[idx_(n, k)];
    }

    [[nodiscard]] int max_n() const noexcept { return max_n_; }

private:
    [[nodiscard]] std::size_t idx_(int n, int k) const noexcept {
        return static_cast<std::size_t>(n) * (static_cast<std::size_t>(max_n_) + 1)
             + static_cast<std::size_t>(k);
    }
    int                        max_n_ = -1;
    std::vector<std::uint64_t> c_;
};

// Combinadic RANK in colex order (inverse of ``unrank_to_state``).
//
// For ``state`` with popcount exactly ``k``, returns the index in
// ``[0, C(n_bits, k))`` such that ``unrank_to_state(rank(state)) == state``.
// Scans bits low-to-high; the (seen)-th set bit at position ``bit``
// contributes ``C(bit, seen)`` to the rank. Matches the device
// ``ed::gpu::combinadic::rank_state`` bit-for-bit.
//
// Returns 0 (a valid rank) when ``popcount(state) != k``; callers that may
// pass out-of-sector states must guard with their own popcount check.
[[nodiscard]] inline std::int64_t
rank_state(std::uint64_t state, int n_bits, int k,
           const BinomialTable& binom) noexcept {
    std::int64_t rank = 0;
    int seen = 0;
    for (int bit = 0; bit < 64; ++bit) {
        if (bit >= n_bits) break;
        if (seen >= k) break;
        if ((state >> bit) & 1ULL) {
            ++seen;
            rank += static_cast<std::int64_t>(binom.at(bit, seen));
        }
    }
    return rank;
}

// Combinadic UNRANK: same colex convention as ``rank_state``.
[[nodiscard]] inline std::uint64_t
unrank_to_state(std::uint64_t rank, int n_bits, int k,
                const BinomialTable& binom) noexcept {
    std::uint64_t state = 0ULL;
    for (int i = k - 1; i >= 0; --i) {
        int p = i;
        while (p + 1 < n_bits && binom.at(p + 1, i + 1) <= rank) ++p;
        state |= (1ULL << p);
        rank -= binom.at(p, i + 1);
    }
    return state;
}

}  // namespace ed::core::combinadic
