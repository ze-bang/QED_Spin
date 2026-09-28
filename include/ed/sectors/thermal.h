#pragma once
// =============================================================================
// include/ed/sectors/thermal.h
//
// Finite-temperature thermodynamics over the symmetry sectors of H.
//
// Every block of every subspace (sectors.h) gives ln Z_b(beta), <E>_b and <E^2>_b,
// either exactly from its full spectrum or by sampling (FTLM, mTPQ) through the
// per-operator thermal kernels. The blocks combine in log space with their
// multiplicities w_b:
//
//   ln Z   = logsumexp_b (ln w_b + ln Z_b)
//   <X>    = sum_b p_b <X>_b,     p_b = w_b Z_b / Z
//   C      = beta^2 (<E^2> - <E>^2),   S = ln Z + beta <E>,   F = -ln Z / beta
//
// When H conserves Sz each block has a definite magnetisation Sz = (N - 2 n_up) / 2
// (a set bit is a down spin), so M = <Sz> and chi = beta (<Sz^2> - <Sz>^2) / N come
// from the same weights. A flip-mirrored block holds +Sz and -Sz in equal parts.
// =============================================================================

#include <ed/sectors/sectors.h>

#include <cstdint>
#include <vector>

namespace ed::sectors {

struct ThermalSpec {
    enum class Method { Exact, FTLM, mTPQ } method = Method::FTLM;
    std::vector<double> temperatures;
    std::size_t   samples      = 40;
    std::size_t   krylov       = 100;   ///< FTLM Lanczos depth; mTPQ step count (0 = automatic)
    std::size_t   exact_states = 0;     ///< FTLM: treat this many lowest states of each block exactly
    std::uint64_t seed         = 0;     ///< 0 = draw one
};

struct ThermalCurves {
    std::vector<double> T, lnZ, E, C, S, F;
    std::vector<double> M, chi;          ///< empty unless H conserves Sz
    double        e0        = 0.0;       ///< lowest energy seen (exact: the ground state)
    std::uint64_t total_dim = 0;
    std::size_t   blocks    = 0;
};

[[nodiscard]] ThermalCurves thermal(const ::Operator& H, int n_sites, const Spec& s,
                                    const ThermalSpec& t);

}  // namespace ed::sectors
