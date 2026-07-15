// term_gate_math.h - the single source of truth for spin-operator term
// application math (Consolidation Family 4).
//
// The per-term "gate" arithmetic -- which bit to test, whether a ladder
// operator annihilates the state, which bit(s) to flip, and the real
// geometric factor from Sz signs -- was reimplemented identically in the CPU
// path (`apply_term_to_state`, term_kernels.h) and the GPU device path
// (`process_source_terms`, term_kernels_gpu.cuh). A physics fix (e.g. a sign
// or gating convention) had to be applied in both, and the regime no test hit
// was where they could silently diverge. These `__host__ __device__` helpers
// are that math, once, so both callers share it. Each helper is pure integer /
// double arithmetic (no complex, no scalar traits, no I/O): the caller still
// owns coefficient multiplication and output emission.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define ED_TERM_GATE_HD __host__ __device__
#else
#define ED_TERM_GATE_HD
#endif

namespace ed::matvec::gate {

// Sz op-type sentinel (matches TransformData::op_type: S+ = 0, S- = 1, Sz = 2).
static constexpr std::uint8_t kOpSz = 2;

// 1. One-body diagonal (Sz_k): state unchanged; returns the real factor
//    spin_l * sign(bit).
ED_TERM_GATE_HD inline double
diag_one_body_factor(std::uint64_t s, std::uint32_t site, double spin_l) {
    return spin_l * (((s >> site) & 1ULL) ? -1.0 : 1.0);
}

// 2. One-body off-diagonal (S+ / S-): gated bit flip. Returns false if the
//    ladder operator annihilates the state; otherwise sets s_prime.
ED_TERM_GATE_HD inline bool
offdiag_one_body(std::uint64_t s, std::uint32_t site, std::uint32_t op_type,
                 std::uint64_t& s_prime) {
    if (((s >> site) & 1ULL) == op_type) return false;
    s_prime = s ^ (1ULL << site);
    return true;
}

// 3. Two-body purely diagonal (Sz_i Sz_j): state unchanged; returns
//    spin_l^2 * sign_i * sign_j.
ED_TERM_GATE_HD inline double
diag_two_body_factor(std::uint64_t s, std::uint32_t s1, std::uint32_t s2,
                     double spin_sq) {
    const double sa = ((s >> s1) & 1ULL) ? -1.0 : 1.0;
    const double sb = ((s >> s2) & 1ULL) ? -1.0 : 1.0;
    return spin_sq * sa * sb;
}

// 4. Two-body mixed (Sz * S+/-): gated flip on flip_site, Sz sign on sz_site.
//    Returns false if the ladder gate annihilates; else sets s_prime and the
//    real factor spin_l * sign(sz_site).
ED_TERM_GATE_HD inline bool
mixed_two_body(std::uint64_t s, std::uint32_t flip_site, std::uint32_t flip_op,
               std::uint32_t sz_site, double spin_l,
               std::uint64_t& s_prime, double& factor) {
    if (((s >> flip_site) & 1ULL) == flip_op) return false;
    const double sz_sign = ((s >> sz_site) & 1ULL) ? -1.0 : 1.0;
    s_prime = s ^ (1ULL << flip_site);
    factor = spin_l * sz_sign;
    return true;
}

// 5. Two-body off-diagonal (S+- * S+-): two gated bit flips. Returns false if
//    either ladder gate annihilates; else sets s_prime.
ED_TERM_GATE_HD inline bool
offdiag_two_body(std::uint64_t s, std::uint32_t s1, std::uint32_t s2,
                 std::uint32_t op1, std::uint32_t op2, std::uint64_t& s_prime) {
    if (((s >> s1) & 1ULL) == op1 || ((s >> s2) & 1ULL) == op2) return false;
    s_prime = s ^ (1ULL << s1) ^ (1ULL << s2);
    return true;
}

// 6. Three-body (general): sequential gate walk over three (op_type, site)
//    pairs. Sz gates accumulate spin_l*sign into `factor`; ladder gates flip
//    the running state or annihilate it. Returns false if annihilated; else
//    sets cur (final state) and factor (product of the Sz geometric factors).
ED_TERM_GATE_HD inline bool
three_body_walk(std::uint64_t s,
                std::uint8_t op1, std::uint32_t site1,
                std::uint8_t op2, std::uint32_t site2,
                std::uint8_t op3, std::uint32_t site3,
                double spin_l,
                std::uint64_t& cur, double& factor) {
    cur = s;
    factor = 1.0;
    const std::uint8_t ops[3]   = {op1, op2, op3};
    const std::uint32_t sites[3] = {site1, site2, site3};
    for (int g = 0; g < 3; ++g) {
        if (ops[g] == kOpSz) {
            const double sg = ((cur >> sites[g]) & 1ULL) ? -1.0 : 1.0;
            factor *= spin_l * sg;
        } else {
            const std::uint64_t b = (cur >> sites[g]) & 1ULL;
            if (b != ops[g]) cur ^= (1ULL << sites[g]);
            else             return false;
        }
    }
    return true;
}

}  // namespace ed::matvec::gate
