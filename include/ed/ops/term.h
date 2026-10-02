// =============================================================================
// include/ed/ops/term.h -- one general spin-1/2 operator term, the unit of the operator
// algebra (algebra.h) and of the compiled programs swept in symmetry sectors (program.h).
//
// A MaskedTerm acts on a computational basis state s (bit set = spin DOWN, the
// engine convention of term_gate_math.h) as
//
//     T|s> = coeff * (-1)^{popcount(s & sign_mask)} |s ^ flip_mask>
//            if (s & cond_mask) == cond_val,   and 0 otherwise.
//
// Every product of S+, S-, S^z, S^x, S^y and the projectors |up><up|, |dn><dn| on any
// set of sites is a finite sum of such terms (see MaskedOperator, algebra.h). The kernels
// still read the six fixed Hamiltonian term bins (matvec/term_storage.h) until the walk is
// rebuilt on these terms.
//
// Canonical form (what MaskedOperator stores): cond_mask == flip_mask (a condition
// only where a ladder operator acts: bit 1 -> S+, bit 0 -> S-), and sign_mask is
// disjoint from flip_mask (Z = 2 S^z factors on unflipped sites). In that form every
// site carries exactly one of {I, Z, S+, S-}, which is a basis of the 2x2 matrices,
// so the expansion of an operator into terms is unique.
// =============================================================================
#pragma once

#include <complex>
#include <cstdint>

#if defined(__CUDACC__)
#define ED_OPS_HD __host__ __device__ __forceinline__
#else
#define ED_OPS_HD inline
#endif

namespace ed::ops {

/// The engine's basis convention: a SET bit is a DOWN spin (term_gate_math.h). The operator
/// algebra reads it only through up_bits / down_bits (P3.4 flips it here).
inline constexpr bool kSetBitIsDown = true;

/// The value bit pattern `b` takes on its sites when they are all up / all down.
ED_OPS_HD std::uint64_t up_bits(std::uint64_t b)   { return kSetBitIsDown ? 0 : b; }
ED_OPS_HD std::uint64_t down_bits(std::uint64_t b) { return kSetBitIsDown ? b : 0; }

struct MaskedTerm {
    std::uint64_t cond_mask{0};   ///< bits whose value the term requires
    std::uint64_t cond_val{0};    ///< required values on cond_mask
    std::uint64_t flip_mask{0};   ///< bits the term flips
    std::uint64_t sign_mask{0};   ///< bits contributing (-1)^bit (Z = 2 S^z)
    std::complex<double> coeff{0.0, 0.0};
};

ED_OPS_HD int masked_popcount(std::uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}

/// Gate of one term on state s: false if the term annihilates s; otherwise the
/// target state and the real sign (+1 / -1) that multiplies coeff.
ED_OPS_HD bool masked_apply(const MaskedTerm& t, std::uint64_t s,
                               std::uint64_t& target, double& sign) {
    if ((s & t.cond_mask) != t.cond_val) return false;
    target = s ^ t.flip_mask;
    sign = (masked_popcount(s & t.sign_mask) & 1) ? -1.0 : 1.0;
    return true;
}

/// Change of the number of set bits (down spins) the term produces, when it acts:
/// a flipped bit that was set (S+) removes one, a flipped bit that was clear (S-) adds one.
ED_OPS_HD int masked_delta_set_bits(const MaskedTerm& t) {
    const std::uint64_t flipped_set   = t.flip_mask & t.cond_val;             // S+ sites
    const std::uint64_t flipped_clear = t.flip_mask & ~t.cond_val & t.cond_mask;  // S- sites
    return masked_popcount(flipped_clear) - masked_popcount(flipped_set);
}

/// Image of a bit mask under a site permutation in the engine's apply_perm convention
/// (rep_symmetry_basis_policy.h): new bit i = old bit perm[i].
inline std::uint64_t permute_mask(std::uint64_t m, const int* perm, int n_sites) {
    std::uint64_t r = 0;
    for (int i = 0; i < n_sites; ++i) r |= ((m >> perm[i]) & 1ULL) << i;
    return r;
}

}  // namespace ed::ops
