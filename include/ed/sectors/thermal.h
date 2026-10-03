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
// When H conserves Sz each block has a definite magnetisation Sz = n_up - N / 2
// (n_up up spins), so M = <Sz> and chi = beta (<Sz^2> - <Sz>^2) / N come
// from the same weights. A flip-mirrored block holds +Sz and -Sz in equal parts.
// =============================================================================

#include <ed/sectors/sectors.h>

#include <cstdint>
#include <vector>

namespace ed::sectors {

struct ThermalSpec {
    enum class Method { Exact, FTLM, mTPQ } method = Method::FTLM;
    std::vector<double> temperatures;
    std::size_t samples = 40;
    std::size_t krylov = 100;   ///< FTLM / OFTLM Lanczos depth (>= 2)
    std::size_t steps = 0;     ///< mTPQ steps per sample (0 = sized for the coldest T)
    std::size_t exact_states = 0;     ///< FTLM: treat this many lowest states of each block exactly
    std::uint64_t dense_max_dim = 512;  ///< FTLM / mTPQ diagonalise blocks up to this dimension; 0: always sample
    std::uint64_t seed = 0;     ///< 0 = draw one
    Device device = Device::Cpu;
    /// Static observables <O>(T) (method Exact or FTLM, with exact_states = 0). Each O is
    /// averaged over the symmetries every block uses, so it may break them. Under a spin
    /// restriction with an SU(2)-symmetric H an O that is not SU(2) invariant enters through its
    /// SU(2)-scalar part (a term on more than 5 sites then raises ed::Unsupported); in a uniform
    /// field O enters as it is. Exact runs diagonalise these blocks on the host.
    std::vector<const ::Operator*> observables;
};

struct ThermalCurves {
    std::vector<double> T, lnZ, E, C, S, F;
    std::vector<double> M, chi;          ///< empty unless H conserves Sz
    std::vector<std::vector<Complex>> O; ///< <O>(T) per ThermalSpec::observables
    /// The lowest energy resolved: exact, the ground state; FTLM, the lowest weighted Ritz value of
    /// any sample (an upper bound on E0, usually close); OFTLM, the lowest certified eigenvalue;
    /// mTPQ, the spectral-bounds Lanczos estimate (an upper bound).
    double e0 = 0.0;
    /// States in the ensemble: block dimension x multiplicity summed over the blocks; under a
    /// total-spin restriction the tower's states (2S + 1 per multiplet).
    std::uint64_t total_dim = 0;
    std::size_t blocks = 0;
    std::size_t device_blocks = 0;   ///< blocks solved on a GPU: sampled blocks, or (Exact) dense-batch blocks
    Placement placement;
    Diagnostics diagnostics;
};

[[nodiscard]] ThermalCurves thermal(const ::Operator& H, const Spec& s, const ThermalSpec& t);

}  // namespace ed::sectors
